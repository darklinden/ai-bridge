//! OpenAI Responses API SSE → Anthropic SSE streaming conversion
//!
//! PORTED from `src-tauri/src/proxy/providers/streaming_responses.rs`.
//! This implements a state-machine-based SSE converter that translates
//! the Responses API's named-event SSE protocol into Anthropic SSE events.

use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing::warn;

use crate::reasoning_bridge::encode_openai_reasoning_item;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const INFINITE_WHITESPACE_THRESHOLD: usize = 500;

// ---------------------------------------------------------------------------
// SSE event parsing helpers
// ---------------------------------------------------------------------------

fn strip_sse_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    line.strip_prefix(&format!("{field}: "))
        .or_else(|| line.strip_prefix(&format!("{field}:")))
}

fn take_sse_block(buffer: &mut String) -> Option<String> {
    let mut best: Option<(usize, usize)> = None;

    for (delimiter, len) in [("\r\n\r\n", 4usize), ("\n\n", 2usize)] {
        if let Some(pos) = buffer.find(delimiter) {
            let current_len = len;
            match best {
                Some((_, best_len)) if pos < best_len => {
                    best = Some((pos, current_len));
                }
                None => {
                    best = Some((pos, current_len));
                }
                _ => {}
            }
        }
    }

    let (pos, len) = best?;
    let block = buffer[..pos].to_string();
    buffer.drain(..pos + len);
    Some(block)
}

fn append_utf8_safe(buffer: &mut String, remainder: &mut Vec<u8>, new_bytes: &[u8]) {
    let input: Vec<u8> = if remainder.is_empty() {
        new_bytes.to_vec()
    } else {
        let mut combined = std::mem::take(remainder);
        combined.extend_from_slice(new_bytes);
        combined
    };

    let mut pos = 0;
    loop {
        match std::str::from_utf8(&input[pos..]) {
            Ok(valid) => {
                buffer.push_str(valid);
                remainder.clear();
                break;
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    buffer.push_str(
                        std::str::from_utf8(&input[pos..pos + valid_up_to])
                            .unwrap_or(""),
                    );
                    pos += valid_up_to;
                }
                let error_len = e.error_len().unwrap_or(1);
                if pos + error_len > input.len() {
                    remainder.extend_from_slice(&input[pos..]);
                    break;
                } else {
                    buffer.push(char::REPLACEMENT_CHARACTER);
                    pos += error_len;
                }
            }
        }
    }

    if remainder.len() > 3 {
        buffer.push_str(&String::from_utf8_lossy(remainder));
        remainder.clear();
    }
}

/// Convert Responses API SSE stream to Anthropic SSE stream.
///
/// `model` is the model echoed back in `message_start` (the original model the
/// client requested), not the upstream's reported model.
pub fn responses_to_anthropic_sse<S, E>(
    stream: S,
    model: String,
    reqlog: Arc<crate::reqlog::ReqLog>,
) -> impl Stream<Item = Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + 'static,
{
    stream! {
        reqlog.resp_header("");
        let mut resp_has_text = false;
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();

        // State tracking
        let mut message_id = String::new();
        let current_model = model;
        let mut has_sent_message_start = false;
        let mut has_sent_message_delta = false;
        let has_sent_message_stop = false;
        let mut pending_message_delta: Option<(Option<String>, Option<Value>)> = None;
        // Non-tool block tracking
        let mut current_non_tool_content_index: Option<u32> = None;
        let mut next_content_index: u32 = 0;

        // Index tracking: Responses item ID + content index -> stable Anthropic index
        let mut index_by_key: HashMap<String, u32> = HashMap::new();
        let mut open_indices: HashSet<u32> = HashSet::new();

        // Tool tracking
        let mut tool_index_by_item_id: HashMap<String, u32> = HashMap::new();
        let mut tool_name_by_index: HashMap<u32, String> = HashMap::new();
        let mut tool_args_by_index: HashMap<u32, String> = HashMap::new();

        // Reasoning tracking
        let mut reasoning_index_by_item_id: HashMap<String, u32> = HashMap::new();
        let mut reasoning_item_by_index: HashMap<u32, Value> = HashMap::new();
        let mut reasoning_text_by_index: HashMap<u32, String> = HashMap::new();

        let mut stream_ended_with_error = false;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);

                    while let Some(block) = take_sse_block(&mut buffer) {
                        if block.trim().is_empty() {
                            continue;
                        }

                        let mut event_type = String::new();
                        let mut data = String::new();

                        for line in block.lines() {
                            if let Some(value) = strip_sse_field(line, "event") {
                                event_type = value.trim().to_string();
                            } else if let Some(value) = strip_sse_field(line, "data") {
                                data = value.to_string();
                            }
                        }

                        if data.is_empty() {
                            continue;
                        }

                        // ==========================================================
                        // response.created
                        // ==========================================================
                        if event_type == "response.created" || event_type == "response.in_progress" {
                            if has_sent_message_start {
                                continue;
                            }

                            if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                                if message_id.is_empty() {
                                    message_id = json_data
                                        .get("id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                }

                                let event = json!({
                                    "type": "message_start",
                                    "message": {
                                        "id": message_id.clone(),
                                        "type": "message",
                                        "role": "assistant",
                                        "model": current_model.clone(),
                                        "usage": {
                                            "input_tokens": 0,
                                            "output_tokens": 0
                                        }
                                    }
                                });
                                let sse_data = format!(
                                    "event: message_start\ndata: {}\n\n",
                                    serde_json::to_string(&event).unwrap_or_default()
                                );
                                has_sent_message_start = true;
                                yield Ok(Bytes::from(sse_data));
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.content_part.added
                        // ==========================================================
                        if event_type == "response.content_part.added" {
                            if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                                let part_type = json_data
                                    .get("type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let part_index = json_data
                                    .get("part_index")
                                    .and_then(|v| v.as_u64())
                                    .unwrap_or(0) as u32;
                                let item_id = json_data
                                    .get("item_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                let key = format!("{item_id}:{part_index}");

                                match part_type {
                                    "output_text" => {
                                        let index = next_content_index;
                                        next_content_index += 1;
                                        index_by_key.insert(key, index);

                                        // Close previous non-tool block if any
                                        if let Some(prev_index) = current_non_tool_content_index.take() {
                                            let event = json!({
                                                "type": "content_block_stop",
                                                "index": prev_index
                                            });
                                            let sse_data = format!(
                                                "event: content_block_stop\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default()
                                            );
                                            yield Ok(Bytes::from(sse_data));
                                        }

                                        let event = json!({
                                            "type": "content_block_start",
                                            "index": index,
                                            "content_block": {
                                                "type": "text",
                                                "text": ""
                                            }
                                        });
                                        let sse_data = format!(
                                            "event: content_block_start\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default()
                                        );
                                        current_non_tool_content_index = Some(index);
                                        open_indices.insert(index);
                                        yield Ok(Bytes::from(sse_data));
                                    }
                                    _ => {
                                        warn!("Unhandled content_part type: {part_type}");
                                    }
                                }
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.output_text.delta
                        // ==========================================================
                        if event_type == "response.output_text.delta" {
                            if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                                if let Some(delta) = json_data.get("delta").and_then(|v| v.as_str()) {
                                    let part_index = json_data
                                        .get("part_index")
                                        .and_then(|v| v.as_u64())
                                        .unwrap_or(0) as u32;
                                    let item_id = json_data
                                        .get("item_id")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");

                                    let key = format!("{item_id}:{part_index}");
                                    if let Some(&index) = index_by_key.get(&key) {
                                        let event = json!({
                                            "type": "content_block_delta",
                                            "index": index,
                                            "delta": {
                                                "type": "text_delta",
                                                "text": delta
                                            }
                                        });
                                        let sse_data = format!(
                                            "event: content_block_delta\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default()
                                        );
                                        resp_has_text = true;
                                        reqlog.append(delta);
                                        yield Ok(Bytes::from(sse_data));
                                    }
                                }
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.output_text.done
                        // ==========================================================
                        if event_type == "response.output_text.done" {
                            if let Some(index) = current_non_tool_content_index.take() {
                                let event = json!({
                                    "type": "content_block_stop",
                                    "index": index
                                });
                                let sse_data = format!(
                                    "event: content_block_stop\ndata: {}\n\n",
                                    serde_json::to_string(&event).unwrap_or_default()
                                );
                                open_indices.remove(&index);
                                yield Ok(Bytes::from(sse_data));
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.output_item.added
                        // ==========================================================
                        if event_type == "response.output_item.added" {
                            if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                                let item_type = json_data
                                    .get("type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let item_id = json_data
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("")
                                    .to_string();

                                match item_type {
                                    "function_call" => {
                                        let call_id = json_data
                                            .get("call_id")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();
                                        let name = json_data
                                            .get("name")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("")
                                            .to_string();

                                        let index = next_content_index;
                                        next_content_index += 1;
                                        tool_index_by_item_id.insert(item_id, index);
                                        tool_name_by_index.insert(index, name.clone());
                                        tool_args_by_index.insert(index, String::new());

                                        let event = json!({
                                            "type": "content_block_start",
                                            "index": index,
                                            "content_block": {
                                                "type": "tool_use",
                                                "id": call_id,
                                                "name": name
                                            }
                                        });
                                        let sse_data = format!(
                                            "event: content_block_start\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default()
                                        );
                                        open_indices.insert(index);
                                        if !resp_has_text {
                                            reqlog.append(&format!("[tool_use: {name}]"));
                                        }
                                        yield Ok(Bytes::from(sse_data));
                                    }
                                    "reasoning" => {
                                        let summary_text = json_data
                                            .get("summary")
                                            .and_then(|v| v.as_array())
                                            .into_iter()
                                            .flatten()
                                            .filter_map(|part| {
                                                part.get("text").and_then(|v| v.as_str())
                                            })
                                            .collect::<Vec<_>>()
                                            .join("");

                                        let index = next_content_index;
                                        next_content_index += 1;
                                        reasoning_index_by_item_id.insert(item_id, index);
                                        reasoning_item_by_index.insert(index, json_data.clone());
                                        reasoning_text_by_index.insert(index, summary_text.clone());

                                        let event = json!({
                                            "type": "content_block_start",
                                            "index": index,
                                            "content_block": {
                                                "type": "thinking",
                                                "thinking": ""
                                            }
                                        });
                                        let sse_data = format!(
                                            "event: content_block_start\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default()
                                        );
                                        open_indices.insert(index);
                                        yield Ok(Bytes::from(sse_data));

                                        // Emit the summary text as a thinking delta immediately
                                        if !summary_text.is_empty() {
                                            let delta_event = json!({
                                                "type": "content_block_delta",
                                                "index": index,
                                                "delta": {
                                                    "type": "thinking_delta",
                                                    "thinking": summary_text
                                                }
                                            });
                                            let sse_data = format!(
                                                "event: content_block_delta\ndata: {}\n\n",
                                                serde_json::to_string(&delta_event).unwrap_or_default()
                                            );
                                            reqlog.append(&summary_text);
                                            yield Ok(Bytes::from(sse_data));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.function_call_arguments.delta
                        // ==========================================================
                        if event_type == "response.function_call_arguments.delta" {
                            if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                                let item_id = json_data
                                    .get("item_id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                if let Some(delta) = json_data.get("delta").and_then(|v| v.as_str()) {
                                    if let Some(&index) = tool_index_by_item_id.get(item_id) {
                                        // Append to accumulated args
                                        let args = tool_args_by_index.entry(index).or_default();
                                        args.push_str(delta);

                                        // Infinite whitespace detection
                                        if args.len() > args.trim_end().len() + INFINITE_WHITESPACE_THRESHOLD {
                                            warn!(
                                                "Detected infinite whitespace bug (tool index: {index}), aborting"
                                            );
                                            continue;
                                        }

                                        let event = json!({
                                            "type": "content_block_delta",
                                            "index": index,
                                            "delta": {
                                                "type": "input_json_delta",
                                                "partial_json": delta
                                            }
                                        });
                                        let sse_data = format!(
                                            "event: content_block_delta\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default()
                                        );
                                        yield Ok(Bytes::from(sse_data));
                                    }
                                }
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.output_item.done
                        // ==========================================================
                        if event_type == "response.output_item.done" {
                            if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                                let item_type = json_data
                                    .get("type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                let item_id = json_data
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");

                                match item_type {
                                    "function_call" => {
                                        if let Some(&index) = tool_index_by_item_id.get(item_id) {
                                            let event = json!({
                                                "type": "content_block_stop",
                                                "index": index
                                            });
                                            let sse_data = format!(
                                                "event: content_block_stop\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default()
                                            );
                                            open_indices.remove(&index);
                                            tool_index_by_item_id.remove(item_id);
                                            yield Ok(Bytes::from(sse_data));
                                        }
                                    }
                                    "reasoning" => {
                                        if let Some(&index) = reasoning_index_by_item_id.get(item_id) {
                                            // Encode the full reasoning item as signature
                                            if let Some(reasoning_item) = reasoning_item_by_index.get(&index) {
                                                if let Some(signature) = encode_openai_reasoning_item(reasoning_item) {
                                                    // Close the thinking block and re-open as redacted_thinking
                                                    // with signature encoded in the delta
                                                    if let Some(current_text) = reasoning_text_by_index.get(&index) {
                                                        // If there was summary text, the thinking block is already done
                                                        // We need to emit a signature delta
                                                        if !current_text.is_empty() {
                                                            let sig_event = json!({
                                                                "type": "content_block_delta",
                                                                "index": index,
                                                                "delta": {
                                                                    "type": "signature_delta",
                                                                    "signature": signature
                                                                }
                                                            });
                                                            let sse_data = format!(
                                                                "event: content_block_delta\ndata: {}\n\n",
                                                                serde_json::to_string(&sig_event).unwrap_or_default()
                                                            );
                                                            yield Ok(Bytes::from(sse_data));
                                                        }
                                                    }
                                                }
                                            }

                                            let event = json!({
                                                "type": "content_block_stop",
                                                "index": index
                                            });
                                            let sse_data = format!(
                                                "event: content_block_stop\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default()
                                            );
                                            open_indices.remove(&index);
                                            reasoning_index_by_item_id.remove(item_id);
                                            yield Ok(Bytes::from(sse_data));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.completed / response.incomplete
                        // ==========================================================
                        if event_type == "response.completed" || event_type == "response.incomplete" {
                            if has_sent_message_delta {
                                continue;
                            }
                            has_sent_message_delta = true;

                            if let Ok(json_data) = serde_json::from_str::<Value>(&data) {
                                // Close any remaining open blocks
                                let mut open_sorted: Vec<u32> = open_indices.iter().copied().collect();
                                open_sorted.sort_unstable();
                                for index in open_sorted {
                                    let event = json!({
                                        "type": "content_block_stop",
                                        "index": index
                                    });
                                    let sse_data = format!(
                                        "event: content_block_stop\ndata: {}\n\n",
                                        serde_json::to_string(&event).unwrap_or_default()
                                    );
                                    open_indices.remove(&index);
                                    yield Ok(Bytes::from(sse_data));
                                }

                                // Build usage
                                let usage = build_anthropic_usage_from_responses_streaming(
                                    json_data.get("usage")
                                );

                                let stop_reason = if event_type == "response.incomplete" {
                                    Some("max_tokens".to_string())
                                } else {
                                    None
                                };

                                pending_message_delta = Some((stop_reason, Some(usage)));
                            }
                            continue;
                        }

                        // ==========================================================
                        // response.failed / error
                        // ==========================================================
                        if event_type == "response.failed" || event_type == "error" {
                            stream_ended_with_error = true;
                            let error_msg = serde_json::from_str::<Value>(&data)
                                .ok()
                                .and_then(|v| {
                                    v.pointer("/error/message")
                                        .or_else(|| v.get("message"))
                                        .and_then(|m| m.as_str())
                                        .map(|s| s.to_string())
                                })
                                .unwrap_or_else(|| "Upstream streaming error".to_string());

                            let error_event = json!({
                                "type": "error",
                                "error": {
                                    "type": "api_error",
                                    "message": error_msg
                                }
                            });
                            let sse_data = format!(
                                "event: error\ndata: {}\n\n",
                                serde_json::to_string(&error_event).unwrap_or_default()
                            );
                            yield Ok(Bytes::from(sse_data));
                            reqlog.err_resp(&error_msg);
                            break;
                        }
                    }
                }
                Err(e) => {
                    warn!("Stream error: {e}");
                    stream_ended_with_error = true;
                    let error_event = json!({
                        "type": "error",
                        "error": {
                            "type": "stream_error",
                            "message": format!("Stream error: {e}")
                        }
                    });
                    let sse_data = format!(
                        "event: error\ndata: {}\n\n",
                        serde_json::to_string(&error_event).unwrap_or_default()
                    );
                    yield Ok(Bytes::from(sse_data));
                    reqlog.err_resp(&format!("Stream error: {e}"));
                    break;
                }
            }
        }

        // --- Flush at end of stream ---
        if !stream_ended_with_error {
            // Close any remaining open blocks
            let mut open_sorted: Vec<u32> = open_indices.iter().copied().collect();
            open_sorted.sort_unstable();
            for index in open_sorted {
                let event = json!({
                    "type": "content_block_stop",
                    "index": index
                });
                let sse_data = format!(
                    "event: content_block_stop\ndata: {}\n\n",
                    serde_json::to_string(&event).unwrap_or_default()
                );
                open_indices.remove(&index);
                yield Ok(Bytes::from(sse_data));
            }

            // Flush pending message_delta
            if let Some((stop_reason, usage_json)) = pending_message_delta.take() {
                let usage = usage_json
                    .filter(|u| u.is_object())
                    .unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0}));

                let event = json!({
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": null
                    },
                    "usage": usage
                });
                let sse_data = format!(
                    "event: message_delta\ndata: {}\n\n",
                    serde_json::to_string(&event).unwrap_or_default()
                );
                yield Ok(Bytes::from(sse_data));
            }

            if !has_sent_message_stop {
                let sse_data =
                    "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string();
                yield Ok(Bytes::from(sse_data));
            }

            reqlog.done();
        }
    }
}

/// Build Anthropic-style usage JSON from Responses API usage (streaming variant).
fn build_anthropic_usage_from_responses_streaming(usage: Option<&Value>) -> Value {
    let u = match usage {
        Some(v) if !v.is_null() && v.is_object() => v,
        _ => return json!({"input_tokens": 0, "output_tokens": 0}),
    };

    if u.as_object().map(|obj| obj.is_empty()).unwrap_or(false) {
        return json!({"input_tokens": 0, "output_tokens": 0});
    }

    let input = u
        .get("input_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| u.get("prompt_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0);

    let output = u
        .get("output_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| u.get("completion_tokens").and_then(|v| v.as_u64()))
        .unwrap_or(0);

    let mut result = json!({
        "input_tokens": input,
        "output_tokens": output
    });

    // Cache tokens from OpenAI-style nested details
    if let Some(cached) = u
        .pointer("/input_tokens_details/cached_tokens")
        .and_then(|v| v.as_u64())
    {
        result["cache_read_input_tokens"] = json!(cached);
    }
    if result.get("cache_read_input_tokens").is_none() {
        if let Some(cached) = u
            .pointer("/prompt_tokens_details/cached_tokens")
            .and_then(|v| v.as_u64())
        {
            result["cache_read_input_tokens"] = json!(cached);
        }
    }

    // Cache write tokens
    if let Some(cw) = u
        .pointer("/input_tokens_details/cache_write_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            u.pointer("/prompt_tokens_details/cache_write_tokens")
                .and_then(|v| v.as_u64())
        })
    {
        result["cache_creation_input_tokens"] = json!(cw);
    }

    // Direct Anthropic-style cache fields override
    if let Some(v) = u.get("cache_read_input_tokens") {
        result["cache_read_input_tokens"] = v.clone();
    }
    if let Some(v) = u.get("cache_creation_input_tokens") {
        result["cache_creation_input_tokens"] = v.clone();
    }

    // Subtract cache from input_tokens for Anthropic-style fresh count
    let cached = result
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let cache_creation = result
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    if cached > 0 || cache_creation > 0 {
        result["input_tokens"] = json!(input.saturating_sub(cached).saturating_sub(cache_creation));
    }

    result
}
