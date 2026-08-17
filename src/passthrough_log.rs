//! Stream-time response logging for the same-format passthrough path.
//!
//! When the local entry and the upstream share a wire format (`handle_passthrough`
//! in `server.rs`), bytes are forwarded verbatim. This module wraps that byte
//! stream solely to append the streamed response text onto the shared
//! `[RESP #id]: ` log line — it never alters the forwarded bytes. Each frame is
//! parsed just enough to extract loggable text fragments per upstream format,
//! and the original `Bytes` chunks are re-yielded unchanged.

use crate::error::LocalEntry;
use crate::reqlog::ReqLog;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::collections::HashSet;
use std::sync::Arc;

/// The upstream wire format of the passthrough stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StreamFormat {
    Anthropic,
    Chat,
    Responses,
}

impl StreamFormat {
    /// In the passthrough path the upstream format always equals the local entry
    /// format (same wire protocol on both sides), so the entry selects the format.
    pub(crate) fn from_local_entry(entry: LocalEntry) -> Self {
        match entry {
            LocalEntry::AnthropicMessages => StreamFormat::Anthropic,
            LocalEntry::OaiChat => StreamFormat::Chat,
            LocalEntry::OaiResponses => StreamFormat::Responses,
        }
    }
}

/// Extract the loggable text fragment from one SSE `data:` JSON value, or `None`
/// if the frame carries no loggable content.
///
/// `resp_has_text` mirrors the conversion-stream semantics: a real text delta
/// flips it so a later tool marker is suppressed; reasoning/thinking deltas are
/// appended but do NOT flip it (matches `convert::chat_to_anthropic_sse`).
/// `tool_markers` de-duplicates tool names so repeated frames for the same tool
/// produce a single `[tool_use: {name}]` marker.
fn extract_log_fragment(
    format: StreamFormat,
    value: &Value,
    resp_has_text: &mut bool,
    tool_markers: &mut HashSet<String>,
) -> Option<String> {
    match format {
        StreamFormat::Anthropic => match value.get("type").and_then(Value::as_str) {
            Some("content_block_delta") => {
                match value.pointer("/delta/type").and_then(Value::as_str) {
                    Some("text_delta") => {
                        let text = value.pointer("/delta/text").and_then(Value::as_str)?;
                        *resp_has_text = true;
                        Some(text.to_string())
                    }
                    Some("thinking_delta") => value
                        .pointer("/delta/thinking")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    _ => None,
                }
            }
            Some("content_block_start") => {
                if value.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use")
                    && !*resp_has_text
                {
                    let name = value
                        .pointer("/content_block/name")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if !name.is_empty() && tool_markers.insert(name.to_string()) {
                        Some(format!("[tool_use: {name}]"))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        },

        StreamFormat::Chat => {
            let mut frags: Vec<String> = Vec::new();
            if let Some(delta) = value.pointer("/choices/0/delta") {
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    if !text.is_empty() {
                        *resp_has_text = true;
                        frags.push(text.to_string());
                    }
                }
                if let Some(reasoning) = delta.get("reasoning_content").and_then(Value::as_str) {
                    if !reasoning.is_empty() {
                        frags.push(reasoning.to_string());
                    }
                }
                if !*resp_has_text {
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                        for tc in tool_calls {
                            if let Some(name) = tc.pointer("/function/name").and_then(Value::as_str)
                            {
                                if !name.is_empty() && tool_markers.insert(name.to_string()) {
                                    frags.push(format!("[tool_use: {name}]"));
                                }
                            }
                        }
                    }
                }
            }
            if frags.is_empty() {
                None
            } else {
                Some(frags.concat())
            }
        }

        StreamFormat::Responses => match value.get("type").and_then(Value::as_str) {
            Some("response.output_text.delta") => {
                let text = value.get("delta").and_then(Value::as_str)?;
                *resp_has_text = true;
                Some(text.to_string())
            }
            Some("response.reasoning_summary_text.delta") => value
                .get("delta")
                .and_then(Value::as_str)
                .map(str::to_string),
            Some("response.output_item.added") => {
                if value.pointer("/item/type").and_then(Value::as_str) == Some("function_call")
                    && !*resp_has_text
                {
                    let name = value.pointer("/item/name").and_then(Value::as_str).unwrap_or("");
                    if !name.is_empty() && tool_markers.insert(name.to_string()) {
                        Some(format!("[tool_use: {name}]"))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        },
    }
}

/// Wrap an upstream byte stream: log response text from SSE frames onto the
/// `[RESP #id]` line while forwarding each `Bytes` chunk untouched.
///
/// The wrapped stream has the same `Item` type as the input (`Result<Bytes, E>`),
/// so it plugs straight into axum's `Body::from_stream` at the call site in
/// `handle_passthrough`. Best-effort logging: malformed/partial frames are
/// skipped for logging but always forwarded verbatim.
pub(crate) fn log_stream<S, E>(
    format: StreamFormat,
    stream: S,
    reqlog: Arc<ReqLog>,
) -> impl Stream<Item = Result<Bytes, E>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + 'static,
{
    use async_stream::stream;

    stream! {
        reqlog.resp_header("");
        let mut resp_has_text = false;
        let mut tool_markers: HashSet<String> = HashSet::new();
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    crate::convert::append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);
                    while let Some(block) = crate::convert::take_sse_block(&mut buffer) {
                        if block.trim().is_empty() {
                            continue;
                        }
                        let mut data = String::new();
                        for line in block.lines() {
                            if let Some(d) = crate::convert::strip_sse_field(line, "data") {
                                data = d.trim().to_string();
                            }
                        }
                        if data.is_empty() {
                            continue;
                        }
                        if let Ok(value) = serde_json::from_str::<Value>(&data) {
                            if let Some(frag) =
                                extract_log_fragment(format, &value, &mut resp_has_text, &mut tool_markers)
                            {
                                reqlog.append(&frag);
                            }
                        }
                    }
                    yield Ok(bytes);
                }
                Err(e) => {
                    reqlog.err_resp(&format!("Stream error: {e}"));
                    yield Err(e);
                    break;
                }
            }
        }
        reqlog.done();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::convert::Infallible;

    fn an(value: Value) -> Value {
        value
    }

    // -----------------------------------------------------------------------
    // extract_log_fragment
    // -----------------------------------------------------------------------

    #[test]
    fn extract_anthropic_text_delta() {
        let mut has_text = false;
        let mut marks = HashSet::new();
        let ev = an(json!({"type":"content_block_delta","index":0,
                           "delta":{"type":"text_delta","text":"hello"}}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Anthropic, &ev, &mut has_text, &mut marks),
            Some("hello".to_string())
        );
        assert!(has_text);
    }

    #[test]
    fn extract_anthropic_thinking_delta_appends_but_not_has_text() {
        let mut has_text = false;
        let mut marks = HashSet::new();
        let ev = an(json!({"type":"content_block_delta","index":0,
                           "delta":{"type":"thinking_delta","thinking":"reasoning"}}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Anthropic, &ev, &mut has_text, &mut marks),
            Some("reasoning".to_string())
        );
        // thinking does NOT flip resp_has_text (matches convert.rs semantics)
        assert!(!has_text);
    }

    #[test]
    fn extract_anthropic_tool_use_marker_only_before_text_and_deduped() {
        let mut has_text = false;
        let mut marks = HashSet::new();
        let ev = an(json!({"type":"content_block_start","index":0,
                           "content_block":{"type":"tool_use","id":"t_1","name":"get_weather"}}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Anthropic, &ev, &mut has_text, &mut marks),
            Some("[tool_use: get_weather]".to_string())
        );
        // Second frame for the same tool is deduped.
        assert_eq!(
            extract_log_fragment(StreamFormat::Anthropic, &ev, &mut has_text, &mut marks),
            None
        );
        // Once real text appears, the marker is suppressed entirely.
        has_text = true;
        let ev2 = an(json!({"type":"content_block_start","index":2,
                            "content_block":{"type":"tool_use","id":"t_2","name":"other"}}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Anthropic, &ev2, &mut has_text, &mut marks),
            None
        );
    }

    #[test]
    fn extract_chat_text_and_tool_marker() {
        let mut has_text = false;
        let mut marks = HashSet::new();
        let ev = an(json!({"id":"1","choices":[{"delta":{"content":"hi"}}]}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Chat, &ev, &mut has_text, &mut marks),
            Some("hi".to_string())
        );
        assert!(has_text);

        // Tool marker suppressed after text.
        let tool_ev = an(json!({"id":"1","choices":[{"delta":{"tool_calls":[
            {"function":{"name":"f","arguments":"{}"}}
        ]}}]}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Chat, &tool_ev, &mut has_text, &mut marks),
            None
        );

        // Tool marker appears when no text yet.
        let mut has_text2 = false;
        let mut marks2 = HashSet::new();
        assert_eq!(
            extract_log_fragment(StreamFormat::Chat, &tool_ev, &mut has_text2, &mut marks2),
            Some("[tool_use: f]".to_string())
        );
    }

    #[test]
    fn extract_chat_reasoning_content() {
        let mut has_text = false;
        let mut marks = HashSet::new();
        let ev = an(json!({"id":"1","choices":[{"delta":{"reasoning_content":"think"}}]}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Chat, &ev, &mut has_text, &mut marks),
            Some("think".to_string())
        );
        assert!(!has_text);
    }

    #[test]
    fn extract_responses_text_and_tool_marker() {
        let mut has_text = false;
        let mut marks = HashSet::new();
        let ev = an(json!({"type":"response.output_text.delta","delta":"world"}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Responses, &ev, &mut has_text, &mut marks),
            Some("world".to_string())
        );
        assert!(has_text);

        // Reasoning summary delta appends but not has_text.
        let reason_ev = an(json!({"type":"response.reasoning_summary_text.delta","delta":"notes"}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Responses, &reason_ev, &mut has_text, &mut marks),
            Some("notes".to_string())
        );

        // Function call marker only before text.
        let tool_ev = an(json!({"type":"response.output_item.added","item":{
            "type":"function_call","name":"tool_a"}}));
        assert_eq!(
            extract_log_fragment(StreamFormat::Responses, &tool_ev, &mut has_text, &mut marks),
            None
        );
        let mut has_text2 = false;
        let mut marks2 = HashSet::new();
        assert_eq!(
            extract_log_fragment(StreamFormat::Responses, &tool_ev, &mut has_text2, &mut marks2),
            Some("[tool_use: tool_a]".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // log_stream byte-preservation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn log_stream_forwards_bytes_unchanged() {
        let frames = [
            Bytes::from(
                "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n",
            ),
            Bytes::from(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
            ),
            Bytes::from(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"world\"}}\n\n",
            ),
            Bytes::from("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n"),
        ];
        let input_concat: Vec<u8> = frames.iter().flat_map(|b| b.iter().copied()).collect();

        let upstream = futures::stream::iter(frames.into_iter().map(Ok::<_, Infallible>));
        let out = log_stream(StreamFormat::Anthropic, upstream, ReqLog::new());
        let chunks: Vec<Bytes> = out.map(|r| r.unwrap()).collect().await;
        let output_concat: Vec<u8> = chunks.iter().flat_map(|b| b.iter().copied()).collect();

        assert_eq!(output_concat, input_concat);
    }

    #[tokio::test]
    async fn log_stream_forwards_chunk_with_split_multibyte_utf8_unchanged() {
        // "你好世界" — split the data mid-codepoint to exercise
        // `append_utf8_safe`'s incomplete-UTF8 remainder handling. Byte
        // preservation must hold regardless.
        let prefix =
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"";
        let full = format!("{prefix}你好世界\"}}\n\n").into_bytes();
        let split = prefix.len() + 1; // inside the first multibyte char

        let b = Bytes::from(full.clone());
        let (a, c) = b.split_at(split);
        let frames = [Bytes::from(a.to_vec()), Bytes::from(c.to_vec())];

        let upstream = futures::stream::iter(frames.into_iter().map(Ok::<_, Infallible>));
        let out = log_stream(StreamFormat::Anthropic, upstream, ReqLog::new());
        let chunks: Vec<Bytes> = out.map(|r| r.unwrap()).collect().await;
        let output_concat: Vec<u8> = chunks.iter().flat_map(|x| x.iter().copied()).collect();

        assert_eq!(output_concat, full);
    }

    #[tokio::test]
    async fn log_stream_forwards_raw_chat_frames_unchanged() {
        // OpenAI Chat SSE (no `event:` field) must pass through unaltered too.
        let frames = [
            Bytes::from("data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"),
            Bytes::from("data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\n"),
            Bytes::from("data: [DONE]\n\n"),
        ];
        let input_concat: Vec<u8> = frames.iter().flat_map(|b| b.iter().copied()).collect();

        let upstream = futures::stream::iter(frames.into_iter().map(Ok::<_, Infallible>));
        let out = log_stream(StreamFormat::Chat, upstream, ReqLog::new());
        let chunks: Vec<Bytes> = out.map(|r| r.unwrap()).collect().await;
        let output_concat: Vec<u8> = chunks.iter().flat_map(|b| b.iter().copied()).collect();

        assert_eq!(output_concat, input_concat);
    }
}