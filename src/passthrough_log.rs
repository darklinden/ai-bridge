//! Stream-time response logging for the same-format passthrough path.
//!
//! When the local entry and the upstream share a wire format (`handle_passthrough`
//! in `server.rs`), bytes are forwarded verbatim. This module wraps that byte
//! stream solely to append the streamed response text onto the shared
//! `[RESP #id]: ` log line — it never alters the forwarded bytes. Each frame is
//! parsed just enough to extract loggable fragments per upstream format, and the
//! original `Bytes` chunks are re-yielded unchanged. A tool call is logged the
//! same way its text is: the head when the block is named, each argument delta as
//! it arrives.

use crate::error::LocalEntry;
use crate::reqlog::{ReqLog, ToolCall};
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::Value;
use std::collections::HashMap;
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

/// The tool calls currently open in the stream, keyed by the upstream's own
/// block identifier (Anthropic content index, Chat `tool_calls[].index`, or
/// Responses item id).
///
/// A block is recorded here the moment it is named: the head goes onto the log
/// line then and each argument delta follows, so a long call is visible while it
/// streams. The tracker also decides who may write (see `reqlog::ToolLine`), and
/// a call that never announces its name is rendered as one complete
/// `[tool_use: ?] {arguments}` marker at its close. Keying per block (rather than
/// de-duplicating by name) also keeps parallel calls of the same tool from
/// hiding each other.
#[derive(Default)]
struct ToolLogState {
    open: HashMap<String, ToolCall>,
    /// Which call may write to the line (see `reqlog::ToolLine`).
    line: crate::reqlog::ToolLine<String>,
    /// Argument deltas seen for a block that has not announced its name yet. A
    /// provider that sends `arguments` before `name` would otherwise lose them.
    pending: HashMap<String, String>,
    /// Insertion order, so parallel calls are written out in the order they opened.
    order: Vec<String>,
}

impl ToolLogState {
    /// Record a tool block that just opened, returning the fragments that put its
    /// head — and any arguments already known — onto the log line. Idempotent: a
    /// repeated opening frame keeps whatever has already been accumulated.
    ///
    /// A block that has not announced its name emits nothing here: it keeps
    /// buffering and its close-time marker renders the `?`.
    fn open(&mut self, key: &str, name: &str, args: &str) -> Vec<String> {
        let buffered = self.pending.remove(key).unwrap_or_default();
        if !self.open.contains_key(key) {
            self.open.insert(key.to_string(), ToolCall::open(name, ""));
            self.order.push(key.to_string());
        }
        let Some(call) = self.open.get_mut(key) else {
            return Vec::new();
        };
        if call.name.is_empty() {
            call.name = name.to_string();
        }
        call.push_args(&buffered);
        call.push_args(args);
        // Only one call may have text on the line at a time; a call that finds
        // it busy keeps buffering and is written whole by `close`.
        let owned_key = key.to_string();
        if !self.line.owns(&owned_key) && !self.line.claim(owned_key) {
            return Vec::new();
        }
        let mut frags = Vec::new();
        if let Some(frag) = call.begin_fragment() {
            frags.push(frag);
        }
        if let Some(frag) = call.tail_fragment() {
            frags.push(frag);
        }
        frags
    }

    /// Append one argument delta to an open block, returning what reaches the log
    /// line now. A delta for a block that has not announced its name yet is
    /// buffered until it does.
    fn push_args(&mut self, key: &str, delta: &str) -> Option<String> {
        match self.open.get_mut(key) {
            Some(tool) => tool.push_fragment(delta),
            None => {
                self.pending
                    .entry(key.to_string())
                    .or_default()
                    .push_str(delta);
                None
            }
        }
    }

    /// Take an authoritative whole-arguments value from a terminal frame. The
    /// Responses `function_call_arguments.done` event carries the complete
    /// argument string, so a stream that emits it without deltas still logs them.
    fn resolve_args(&mut self, key: &str, args: &str) -> Option<String> {
        let tool = self.open.get_mut(key)?;
        tool.resolve_args(args);
        tool.tail_fragment()
    }

    /// Finish a tool block, returning the fragments it still owes the log line —
    /// the whole marker for a block that never announced its name, plus any
    /// finished calls that were queued behind it.
    fn close(&mut self, key: &str) -> Vec<String> {
        // Drop any argument text buffered for a block that never announced its
        // name, so a later block reusing the key cannot inherit it.
        self.pending.remove(key);
        let Some(mut tool) = self.open.remove(key) else {
            return Vec::new();
        };
        self.order.retain(|k| k != key);
        let owned_key = key.to_string();
        if self.line.owns(&owned_key) {
            let mut frags: Vec<String> = tool.settle_fragment().into_iter().collect();
            frags.extend(self.line.release(&owned_key));
            return frags;
        }
        if self.line.is_busy() {
            // Another call owns the line: hold this finished marker until it is
            // released, so neither call's text lands under the other's head.
            if let Some(finished) = tool.settle_fragment() {
                self.line.hold(finished);
            }
            return Vec::new();
        }
        tool.settle_fragment().into_iter().collect()
    }

    /// Finish every block still open (a stream that ended mid-tool, or a Chat
    /// stream whose terminal `finish_reason` frame closed them all at once).
    fn close_all(&mut self) -> Vec<String> {
        let keys = std::mem::take(&mut self.order);
        keys.iter().flat_map(|key| self.close(key)).collect()
    }
}

/// The `index` an Anthropic frame refers to, as the tracker's key.
fn anthropic_key(value: &Value) -> String {
    value
        .get("index")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .to_string()
}

/// Extract the loggable fragments from one SSE `data:` JSON value. An empty
/// result means the frame carries nothing loggable.
///
/// `tools` carries the in-flight tool calls across frames: the head has to wait
/// for the name and the arguments for their deltas, so the tracker hands back
/// whatever is ready to be written *now* rather than one finished marker.
fn extract_log_fragment(
    format: StreamFormat,
    value: &Value,
    tools: &mut ToolLogState,
) -> Vec<String> {
    match format {
        StreamFormat::Anthropic => match value.get("type").and_then(Value::as_str) {
            Some("content_block_start") => {
                if value.pointer("/content_block/type").and_then(Value::as_str) == Some("tool_use")
                {
                    let name = value
                        .pointer("/content_block/name")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    // A block the upstream never named still reached the client,
                    // so it is tracked and logged as `[tool_use: ?]`.
                    return tools.open(&anthropic_key(value), name, "");
                }
                Vec::new()
            }
            Some("content_block_delta") => {
                match value.pointer("/delta/type").and_then(Value::as_str) {
                    Some("text_delta") => value
                        .pointer("/delta/text")
                        .and_then(Value::as_str)
                        .map(|text| vec![text.to_string()])
                        .unwrap_or_default(),
                    Some("thinking_delta") => value
                        .pointer("/delta/thinking")
                        .and_then(Value::as_str)
                        .map(|text| vec![text.to_string()])
                        .unwrap_or_default(),
                    Some("input_json_delta") => {
                        value
                            .pointer("/delta/partial_json")
                            .and_then(Value::as_str)
                            .and_then(|partial| tools.push_args(&anthropic_key(value), partial))
                            .into_iter()
                            .collect()
                    }
                    _ => Vec::new(),
                }
            }
            Some("content_block_stop") => tools.close(&anthropic_key(value)),
            _ => Vec::new(),
        },

        StreamFormat::Chat => {
            let mut frags: Vec<String> = Vec::new();
            if let Some(delta) = value.pointer("/choices/0/delta") {
                if let Some(text) = delta.get("content").and_then(Value::as_str) {
                    if !text.is_empty() {
                        frags.push(text.to_string());
                    }
                }
                // OpenRouter/Kimi spell it `reasoning`, DeepSeek spells it
                // `reasoning_content` — the same two spellings `convert::Delta`
                // accepts. Take the first non-empty spelling so the thinking
                // chain reaches the log whichever one the upstream sends,
                // without ever appending both copies when a gateway emits both.
                let reasoning = ["reasoning_content", "reasoning"]
                    .iter()
                    .filter_map(|k| delta.get(*k).and_then(Value::as_str))
                    .find(|s| !s.is_empty());
                if let Some(reasoning) = reasoning {
                    frags.push(reasoning.to_string());
                }
                if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
                    for tc in tool_calls {
                        // Chat tool deltas are correlated by their own index;
                        // single-tool upstreams omit it, which is index 0.
                        let key = tc
                            .get("index")
                            .and_then(Value::as_u64)
                            .unwrap_or(0)
                            .to_string();
                        // A call the upstream never named still reached the
                        // client, so it is tracked and logged as `[tool_use: ?]`.
                        // An empty name stays replaceable: a later frame that
                        // spells it fills it in (`ToolLogState::open`).
                        if let Some(name) = tc.pointer("/function/name").and_then(Value::as_str) {
                            frags.extend(tools.open(&key, name, ""));
                        }
                        if let Some(args) =
                            tc.pointer("/function/arguments").and_then(Value::as_str)
                        {
                            frags.extend(tools.push_args(&key, args));
                        }
                    }
                }
            }
            // Chat has no per-tool stop event: the terminal frame of the turn
            // (`finish_reason`) is where every call's arguments are whole.
            let finished = value
                .pointer("/choices/0/finish_reason")
                .is_some_and(|f| !f.is_null());
            if finished {
                frags.extend(tools.close_all());
            }
            frags
        }

        StreamFormat::Responses => {
            // Real OpenAI Responses nests the item under `item`; older flattened
            // streams put its fields at the top level.
            let item = value.get("item").unwrap_or(value);
            match value.get("type").and_then(Value::as_str) {
                Some("response.output_text.delta") => value
                    .get("delta")
                    .and_then(Value::as_str)
                    .map(|text| vec![text.to_string()])
                    .unwrap_or_default(),
                Some("response.reasoning_summary_text.delta") => value
                    .get("delta")
                    .and_then(Value::as_str)
                    .map(|text| vec![text.to_string()])
                    .unwrap_or_default(),
                Some("response.output_item.added") => {
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        let name = item.get("name").and_then(Value::as_str).unwrap_or("");
                        let key = item.get("id").and_then(Value::as_str).unwrap_or("");
                        // A nameless block is tracked too (the client gets the same
                        // one); only a keyless frame cannot be tracked at all.
                        if !key.is_empty() {
                            return tools.open(key, name, "");
                        }
                    }
                    Vec::new()
                }
                Some("response.function_call_arguments.delta") => {
                    let key = value.get("item_id").and_then(Value::as_str).unwrap_or("");
                    value
                        .get("delta")
                        .and_then(Value::as_str)
                        .and_then(|delta| tools.push_args(key, delta))
                        .into_iter()
                        .collect()
                }
                Some("response.function_call_arguments.done") => {
                    let key = value.get("item_id").and_then(Value::as_str).unwrap_or("");
                    value
                        .get("arguments")
                        .and_then(Value::as_str)
                        .and_then(|args| tools.resolve_args(key, args))
                        .into_iter()
                        .collect()
                }
                Some("response.output_item.done") => {
                    if item.get("type").and_then(Value::as_str) == Some("function_call") {
                        let key = item.get("id").and_then(Value::as_str).unwrap_or("");
                        return tools.close(key);
                    }
                    Vec::new()
                }
                Some("response.completed") | Some("response.incomplete") => tools.close_all(),
                _ => Vec::new(),
            }
        }
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
        let mut tools = ToolLogState::default();
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
                            for frag in extract_log_fragment(format, &value, &mut tools) {
                                reqlog.append(&frag);
                            }
                        }
                    }
                    yield Ok(bytes);
                }
                Err(e) => {
                    // `err_resp` closes the log line, so a tool call still open
                    // has to be written before it.
                    for frag in tools.close_all() {
                        reqlog.append(&frag);
                    }
                    reqlog.err_resp(&format!("Stream error: {e}"));
                    yield Err(e);
                    break;
                }
            }
        }
        // An upstream that ended without closing its tool blocks still gets them
        // onto the line.
        for frag in tools.close_all() {
            reqlog.append(&frag);
        }
        reqlog.done();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::convert::Infallible;

    /// One Anthropic tool-call block: open, arguments delta(s), close.
    fn anthropic_tool_frames(name: &str, args: &str) -> Vec<Value> {
        vec![
            json!({"type":"content_block_start","index":0,
                   "content_block":{"type":"tool_use","id":"t_1","name":name}}),
            json!({"type":"content_block_delta","index":0,
                   "delta":{"type":"input_json_delta","partial_json":args}}),
            json!({"type":"content_block_stop","index":0}),
        ]
    }

    fn run(format: StreamFormat, frames: &[Value]) -> Vec<String> {
        let mut tools = ToolLogState::default();
        frames
            .iter()
            .flat_map(|value| extract_log_fragment(format, value, &mut tools))
            .collect()
    }

    /// The fragments a frame sequence produces, concatenated — exactly the text
    /// that would land on the `[RESP]` line.
    fn run_joined(format: StreamFormat, frames: &[Value]) -> String {
        run(format, frames).concat()
    }

    // -----------------------------------------------------------------------
    // extract_log_fragment — text and reasoning
    // -----------------------------------------------------------------------

    #[test]
    fn extract_anthropic_text_delta() {
        let ev = json!({"type":"content_block_delta","index":0,
                        "delta":{"type":"text_delta","text":"hello"}});
        assert_eq!(
            run(StreamFormat::Anthropic, &[ev]),
            vec!["hello".to_string()]
        );
    }

    #[test]
    fn extract_anthropic_thinking_delta() {
        let ev = json!({"type":"content_block_delta","index":0,
                        "delta":{"type":"thinking_delta","thinking":"reasoning"}});
        assert_eq!(
            run(StreamFormat::Anthropic, &[ev]),
            vec!["reasoning".to_string()]
        );
    }

    #[test]
    fn extract_chat_text_and_reasoning() {
        let ev = json!({"id":"1","choices":[{"delta":{"content":"hi"}}]});
        assert_eq!(run(StreamFormat::Chat, &[ev]), vec!["hi".to_string()]);

        let ev = json!({"id":"1","choices":[{"delta":{"reasoning_content":"think"}}]});
        assert_eq!(run(StreamFormat::Chat, &[ev]), vec!["think".to_string()]);
    }

    #[test]
    fn extract_chat_reasoning_alias_field() {
        // OpenRouter/Kimi-style `delta.reasoning` must be logged too — otherwise
        // the thinking chain never reaches the [RESP] line on those upstreams.
        let ev = json!({"id":"1","choices":[{"delta":{"reasoning":"think"}}]});
        assert_eq!(run(StreamFormat::Chat, &[ev]), vec!["think".to_string()]);
    }

    #[test]
    fn extract_chat_reasoning_picks_one_spelling_without_duplication() {
        // Both spellings in one delta → exactly one copy, `reasoning_content`
        // preferred; a null/empty first spelling falls through to the other.
        let both = json!({"id":"1","choices":[{"delta":{
            "reasoning_content":"think","reasoning":"think"}}]});
        assert_eq!(run(StreamFormat::Chat, &[both]), vec!["think".to_string()]);

        let null_first = json!({"id":"1","choices":[{"delta":{
            "reasoning_content":null,"reasoning":"think"}}]});
        assert_eq!(
            run(StreamFormat::Chat, &[null_first]),
            vec!["think".to_string()]
        );

        let empty_first = json!({"id":"1","choices":[{"delta":{
            "reasoning_content":"","reasoning":"think"}}]});
        assert_eq!(
            run(StreamFormat::Chat, &[empty_first]),
            vec!["think".to_string()]
        );
    }

    #[test]
    fn extract_responses_text_and_reasoning() {
        let ev = json!({"type":"response.output_text.delta","delta":"world"});
        assert_eq!(
            run(StreamFormat::Responses, &[ev]),
            vec!["world".to_string()]
        );

        let ev = json!({"type":"response.reasoning_summary_text.delta","delta":"notes"});
        assert_eq!(
            run(StreamFormat::Responses, &[ev]),
            vec!["notes".to_string()]
        );
    }

    // -----------------------------------------------------------------------
    // extract_log_fragment — tool calls
    // -----------------------------------------------------------------------

    #[test]
    fn anthropic_tool_call_puts_its_head_out_before_its_arguments() {
        // The point of streaming a call: the head reaches the line while the
        // arguments are still arriving, so an operator sees which call is in
        // flight instead of a silent line.
        let frames = anthropic_tool_frames("get_weather", "{\"city\":\"SF\"}");
        let mut tools = ToolLogState::default();
        assert_eq!(
            extract_log_fragment(StreamFormat::Anthropic, &frames[0], &mut tools),
            vec!["[tool_use: get_weather]".to_string()]
        );
        assert_eq!(
            extract_log_fragment(StreamFormat::Anthropic, &frames[1], &mut tools),
            vec![" {\"city\":\"SF\"}".to_string()]
        );
        // The closing frame has nothing left to add.
        assert!(extract_log_fragment(StreamFormat::Anthropic, &frames[2], &mut tools).is_empty());
        assert_eq!(
            run_joined(StreamFormat::Anthropic, &frames),
            "[tool_use: get_weather] {\"city\":\"SF\"}"
        );
    }

    #[test]
    fn anthropic_tool_arguments_arriving_in_several_deltas_each_reach_the_line() {
        let frames = vec![
            json!({"type":"content_block_start","index":2,
                   "content_block":{"type":"tool_use","id":"t_1","name":"bash"}}),
            json!({"type":"content_block_delta","index":2,
                   "delta":{"type":"input_json_delta","partial_json":"{\"cmd\":"}}),
            json!({"type":"content_block_delta","index":2,
                   "delta":{"type":"input_json_delta","partial_json":"\"ls\"}"}}),
            json!({"type":"content_block_stop","index":2}),
        ];
        let frags = run(StreamFormat::Anthropic, &frames);
        // Head, then one fragment per delta — never the whole call at the end.
        assert_eq!(frags.len(), 3, "fragments: {frags:?}");
        assert_eq!(frags.concat(), "[tool_use: bash] {\"cmd\":\"ls\"}");
    }

    #[test]
    fn anthropic_tool_without_arguments_keeps_bare_marker() {
        let frames = vec![
            json!({"type":"content_block_start","index":0,
                   "content_block":{"type":"tool_use","id":"t_1","name":"ping"}}),
            json!({"type":"content_block_stop","index":0}),
        ];
        assert_eq!(run_joined(StreamFormat::Anthropic, &frames), "[tool_use: ping]");
    }

    #[test]
    fn anthropic_parallel_calls_of_the_same_tool_are_both_logged() {
        // De-duplicating by name (the old behaviour) hid the second call. Each
        // call now reaches the line where its own text arrives, so both keep
        // their own arguments.
        let frames = vec![
            json!({"type":"content_block_start","index":0,
                   "content_block":{"type":"tool_use","id":"t_1","name":"read"}}),
            json!({"type":"content_block_delta","index":0,
                   "delta":{"type":"input_json_delta","partial_json":"{\"p\":\"a\"}"}}),
            json!({"type":"content_block_start","index":1,
                   "content_block":{"type":"tool_use","id":"t_2","name":"read"}}),
            json!({"type":"content_block_delta","index":1,
                   "delta":{"type":"input_json_delta","partial_json":"{\"p\":\"b\"}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"content_block_stop","index":0}),
        ];
        assert_eq!(
            run_joined(StreamFormat::Anthropic, &frames),
            "[tool_use: read] {\"p\":\"a\"}[tool_use: read] {\"p\":\"b\"}"
        );
    }

    #[test]
    fn a_call_that_finishes_first_waits_for_the_line() {
        // Parallel calls: the second one opens while the first still streams and
        // closes before it. Its marker must wait until the first call's
        // arguments are complete, or the two would be spliced together.
        let frames = vec![
            json!({"type":"content_block_start","index":0,
                   "content_block":{"type":"tool_use","id":"t_0","name":"first"}}),
            json!({"type":"content_block_delta","index":0,
                   "delta":{"type":"input_json_delta","partial_json":"{\"a\":"}}),
            json!({"type":"content_block_start","index":1,
                   "content_block":{"type":"tool_use","id":"t_1","name":"second"}}),
            json!({"type":"content_block_delta","index":1,
                   "delta":{"type":"input_json_delta","partial_json":"{\"b\":2}"}}),
            json!({"type":"content_block_stop","index":1}),
            json!({"type":"content_block_delta","index":0,
                   "delta":{"type":"input_json_delta","partial_json":"1}"}}),
            json!({"type":"content_block_stop","index":0}),
        ];
        assert_eq!(
            run_joined(StreamFormat::Anthropic, &frames),
            "[tool_use: first] {\"a\":1}[tool_use: second] {\"b\":2}"
        );
    }

    #[test]
    fn chat_tool_call_streams_before_the_finish_reason_frame() {
        let frames = vec![
            json!({"id":"1","choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"f","arguments":""}}]}}]}),
            json!({"id":"1","choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"{\"a\":1}"}}]}}]}),
            json!({"id":"1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ];
        // Head at the opening frame, arguments at the delta, nothing at finish.
        let frags = run(StreamFormat::Chat, &frames);
        assert_eq!(
            frags,
            vec!["[tool_use: f]".to_string(), " {\"a\":1}".to_string()]
        );
        assert_eq!(frags.concat(), "[tool_use: f] {\"a\":1}");
    }

    #[test]
    fn chat_tool_call_without_index_is_keyed_as_zero() {
        let frames = vec![
            json!({"id":"1","choices":[{"delta":{"tool_calls":[
                {"function":{"name":"f","arguments":"{}"}}]}}]}),
            json!({"id":"1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ];
        assert_eq!(run_joined(StreamFormat::Chat, &frames), "[tool_use: f] {}");
    }

    #[test]
    fn chat_text_before_a_tool_does_not_suppress_the_marker() {
        // The old `resp_has_text` rule dropped any tool call that followed text.
        let frames = vec![
            json!({"id":"1","choices":[{"delta":{"content":"Let me check."}}]}),
            json!({"id":"1","choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"f","arguments":"{}"}}]}}]}),
            json!({"id":"1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ];
        assert_eq!(
            run_joined(StreamFormat::Chat, &frames),
            "Let me check.[tool_use: f] {}"
        );
    }

    #[test]
    fn responses_tool_call_streams_from_the_item_added_frame() {
        let frames = vec![
            json!({"type":"response.output_item.added","item":{
                "id":"fc_1","type":"function_call","call_id":"call_1","name":"get_weather"}}),
            json!({"type":"response.function_call_arguments.delta",
                   "item_id":"fc_1","delta":"{\"city\":"}),
            json!({"type":"response.function_call_arguments.delta",
                   "item_id":"fc_1","delta":"\"SF\"}"}),
            json!({"type":"response.output_item.done","item":{
                "id":"fc_1","type":"function_call","call_id":"call_1","name":"get_weather"}}),
        ];
        let frags = run(StreamFormat::Responses, &frames);
        assert_eq!(
            frags,
            vec![
                "[tool_use: get_weather]".to_string(),
                " {\"city\":".to_string(),
                "\"SF\"}".to_string(),
            ]
        );
        assert_eq!(
            frags.concat(),
            "[tool_use: get_weather] {\"city\":\"SF\"}"
        );
    }

    #[test]
    fn responses_arguments_done_frame_supplies_the_whole_arguments() {
        // A stream that sends no argument deltas still logs the call, because
        // `function_call_arguments.done` carries the complete string.
        let frames = vec![
            json!({"type":"response.output_item.added","item":{
                "id":"fc_1","type":"function_call","call_id":"call_1","name":"f"}}),
            json!({"type":"response.function_call_arguments.done",
                   "item_id":"fc_1","arguments":"{\"a\":1}"}),
            json!({"type":"response.output_item.done","item":{
                "id":"fc_1","type":"function_call","call_id":"call_1","name":"f"}}),
        ];
        assert_eq!(
            run_joined(StreamFormat::Responses, &frames),
            "[tool_use: f] {\"a\":1}"
        );
    }

    #[test]
    fn responses_flattened_item_shape_is_accepted() {
        // The tracker dispatches on the JSON `type`, which for a genuinely
        // flattened item frame is the *item* type — such a frame matches no
        // branch, and the tracker must not invent a call from it.
        let frames = vec![
            json!({"type":"response.output_item.added","id":"fc_1",
                   "type_placeholder":"","name":"f"}),
            json!({"type":"response.output_item.done","id":"fc_1","name":"f"}),
        ];
        assert!(run(StreamFormat::Responses, &frames).is_empty());
    }

    #[test]
    fn responses_done_frame_never_shrinks_accumulated_arguments() {
        // The terminal frame is supposed to agree with the deltas; a partial one
        // must not throw away what the deltas already produced.
        let frames = vec![
            json!({"type":"response.output_item.added","item":{
                "id":"fc_1","type":"function_call","call_id":"call_1","name":"f"}}),
            json!({"type":"response.function_call_arguments.delta",
                   "item_id":"fc_1","delta":"{\"a\":1}"}),
            json!({"type":"response.function_call_arguments.done",
                   "item_id":"fc_1","arguments":"{}"}),
            json!({"type":"response.output_item.done","item":{
                "id":"fc_1","type":"function_call","call_id":"call_1","name":"f"}}),
        ];
        assert_eq!(
            run_joined(StreamFormat::Responses, &frames),
            "[tool_use: f] {\"a\":1}"
        );
    }

    #[test]
    fn anthropic_tool_block_without_a_name_is_logged_as_a_question_mark() {
        // The client received the block, so the log records it even though the
        // upstream never spelled a name. A nameless block buffers and settles as
        // one marker: there is no head to stream.
        let frames = vec![
            json!({"type":"content_block_start","index":0,
                   "content_block":{"type":"tool_use","id":"t_1"}}),
            json!({"type":"content_block_delta","index":0,
                   "delta":{"type":"input_json_delta","partial_json":"{\"a\":1}"}}),
            json!({"type":"content_block_stop","index":0}),
        ];
        assert_eq!(
            run_joined(StreamFormat::Anthropic, &frames),
            "[tool_use: ?] {\"a\":1}"
        );
    }

    #[test]
    fn chat_tool_call_without_a_name_is_logged_as_a_question_mark() {
        let frames = vec![
            json!({"id":"1","choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"","arguments":"{\"a\":1}"}}]}}]}),
            json!({"id":"1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ];
        assert_eq!(
            run_joined(StreamFormat::Chat, &frames),
            "[tool_use: ?] {\"a\":1}"
        );
    }

    #[test]
    fn chat_arguments_arriving_before_the_name_are_kept() {
        let frames = vec![
            json!({"id":"1","choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"arguments":"{\"a\":1}"}}]}}]}),
            json!({"id":"1","choices":[{"delta":{"tool_calls":[
                {"index":0,"function":{"name":"f"}}]}}]}),
            json!({"id":"1","choices":[{"delta":{},"finish_reason":"tool_calls"}]}),
        ];
        // The arguments were buffered, so they leave with the head once the name
        // arrives — still before the turn's finish frame.
        assert_eq!(
            run_joined(StreamFormat::Chat, &frames),
            "[tool_use: f] {\"a\":1}"
        );
    }

    #[test]
    fn a_truncated_stream_leaves_nothing_unsaid() {
        // A named call has already put its head and arguments on the line, so the
        // flush at the end of the stream has nothing left to write.
        let mut tools = ToolLogState::default();
        extract_log_fragment(
            StreamFormat::Anthropic,
            &json!({"type":"content_block_start","index":0,
                    "content_block":{"type":"tool_use","id":"t_1","name":"f"}}),
            &mut tools,
        );
        extract_log_fragment(
            StreamFormat::Anthropic,
            &json!({"type":"content_block_delta","index":0,
                    "delta":{"type":"input_json_delta","partial_json":"{}"}}),
            &mut tools,
        );
        assert!(tools.close_all().is_empty());
        assert!(tools.close_all().is_empty());

        // A call that never learned its name could not stream, so the flush is
        // where it reaches the line — once.
        let mut tools = ToolLogState::default();
        extract_log_fragment(
            StreamFormat::Anthropic,
            &json!({"type":"content_block_start","index":0,
                    "content_block":{"type":"tool_use","id":"t_1"}}),
            &mut tools,
        );
        extract_log_fragment(
            StreamFormat::Anthropic,
            &json!({"type":"content_block_delta","index":0,
                    "delta":{"type":"input_json_delta","partial_json":"{}"}}),
            &mut tools,
        );
        assert_eq!(tools.close_all(), vec!["[tool_use: ?] {}".to_string()]);
        assert!(tools.close_all().is_empty());
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

    #[tokio::test]
    async fn log_stream_writes_open_tool_calls_before_the_error_line() {
        // `err_resp` closes the log line, so a call that already reached the
        // client must be written first — not silently dropped.
        let frames: Vec<Result<Bytes, std::io::Error>> = vec![
            Ok(Bytes::from(
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t_1\",\"name\":\"bash\"}}\n\n",
            )),
            Ok(Bytes::from(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"ls\\\"}\"}}\n\n",
            )),
            Err(std::io::Error::other("boom")),
        ];

        let (reqlog, captured) = ReqLog::captured();
        let upstream = futures::stream::iter(frames);
        let out = log_stream(StreamFormat::Anthropic, upstream, reqlog);
        let _: Vec<_> = out.collect().await;

        let logged = captured.lock().unwrap().clone();
        let marker = logged.find("[tool_use: bash] {\"command\":\"ls\"}");
        let error = logged.find("[ERR-RESP");
        assert!(marker.is_some(), "log was: {logged}");
        assert!(error.is_some(), "log was: {logged}");
        assert!(
            marker < error,
            "marker must precede the error line: {logged}"
        );
    }

    #[tokio::test]
    async fn log_stream_writes_a_tool_call_with_its_arguments() {
        let frames = [
            Bytes::from(
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t_1\",\"name\":\"bash\"}}\n\n",
            ),
            Bytes::from(
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"command\\\":\\\"ls\\\"}\"}}\n\n",
            ),
            Bytes::from("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"),
        ];
        let input_concat: Vec<u8> = frames.iter().flat_map(|b| b.iter().copied()).collect();

        let (reqlog, captured) = ReqLog::captured();
        let upstream = futures::stream::iter(frames.into_iter().map(Ok::<_, Infallible>));
        let out = log_stream(StreamFormat::Anthropic, upstream, reqlog);
        let chunks: Vec<Bytes> = out.map(|r| r.unwrap()).collect().await;
        let output_concat: Vec<u8> = chunks.iter().flat_map(|b| b.iter().copied()).collect();

        // Logging changed, forwarded bytes did not (INV-12).
        assert_eq!(output_concat, input_concat);
        let logged = captured.lock().unwrap().clone();
        assert!(
            logged.contains("[tool_use: bash] {\"command\":\"ls\"}"),
            "log was: {logged}"
        );
    }
}
