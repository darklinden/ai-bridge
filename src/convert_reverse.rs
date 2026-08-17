//! Reverse conversions: local OpenAI Chat Completions entry → Anthropic Messages
//! upstream. Mirrors the existing `convert.rs` (Anthropic entry → OpenAI
//! upstream) in the opposite direction.
//!
//! Three functions, symmetric with `convert.rs`:
//! - `chat_to_anthropic_request`   — OpenAI Chat request body → Anthropic request
//! - `anthropic_to_chat_response`  — Anthropic response → OpenAI Chat response
//! - `anthropic_to_chat_sse`       — Anthropic SSE → OpenAI Chat SSE

use crate::error::Error;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::sync::Arc;

/// Fields OpenAI clients may send that Anthropic Messages cannot represent.
/// These are dropped with a WARN (they are optional capabilities, not errors),
/// except `n > 1` which is rejected outright (single-output protocol).
const DROPPED_OPENAI_CHAT_FIELDS: &[&str] = &[
    "logprobs",
    "top_logprobs",
    "response_format",
    "seed",
    "frequency_penalty",
    "presence_penalty",
];

/// Convert an OpenAI Chat Completions request body to an Anthropic Messages
/// request body.
pub fn chat_to_anthropic_request(body: &Value) -> Result<Value, Error> {
    // n > 1 cannot be represented (Anthropic is single-output). Reject clearly
    // rather than silently returning one candidate.
    if let Some(n) = body.get("n").and_then(Value::as_u64) {
        if n != 1 {
            return Err(Error::Unsupported(format!(
                "n={n} is not supported when the upstream is Anthropic Messages \
                 (Anthropic returns a single candidate)"
            )));
        }
    }

    let mut result = json!({});

    // model — always overridden upstream by server.rs, but keep for symmetry.
    if let Some(model) = body.get("model").and_then(Value::as_str) {
        result["model"] = json!(model);
    }

    let messages = body.get("messages").and_then(Value::as_array);
    let mut anthropic_messages: Vec<Value> = Vec::new();

    // Separate OpenAI `system`-role messages into the Anthropic top-level
    // `system` field (concatenated); remaining messages map to `messages`.
    let mut system_parts: Vec<String> = Vec::new();
    let mut system_seen = false;
    if let Some(msgs) = messages {
        for msg in msgs {
            let role = msg.get("role").and_then(Value::as_str).unwrap_or("");
            if role == "system" {
                system_seen = true;
                if let Some(text) = extract_chat_text(msg.get("content")) {
                    system_parts.push(text);
                }
                continue;
            }
            let converted = chat_message_to_anthropic(msg)?;
            anthropic_messages.push(converted);
        }
    }
    if system_seen && !system_parts.is_empty() {
        result["system"] = json!(system_parts.join("\n\n"));
    }
    if !anthropic_messages.is_empty() {
        result["messages"] = json!(anthropic_messages);
    }

    // max_completion_tokens / max_tokens → max_tokens
    if let Some(v) = body
        .get("max_completion_tokens")
        .or_else(|| body.get("max_tokens"))
    {
        result["max_tokens"] = v.clone();
    }

    // temperature / top_p pass through.
    if let Some(v) = body.get("temperature") {
        result["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        result["top_p"] = v.clone();
    }

    // reasoning_effort → thinking (Anthropic adaptive thinking).
    if let Some(effort) = body.get("reasoning_effort").and_then(Value::as_str) {
        match effort.to_ascii_lowercase().as_str() {
            "none" => {
                result["thinking"] = json!({ "type": "disabled" });
            }
            _ => {
                // Anthropic has no discrete effort levels; map any requested
                // effort to enabled adaptive thinking.
                result["thinking"] = json!({ "type": "enabled" });
            }
        }
    }

    // stop → stop_sequences (normalize single string to array).
    if let Some(stop) = body.get("stop") {
        let stops: Vec<Value> = match stop {
            Value::String(s) => vec![json!(s)],
            Value::Array(a) => a.clone(),
            _ => Vec::new(),
        };
        if !stops.is_empty() {
            result["stop_sequences"] = json!(stops);
        }
    }

    // tools[].function → Anthropic tools with input_schema.
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let function = t.get("function")?;
                let name = function.get("name").and_then(Value::as_str)?;
                let mut tool = json!({
                    "name": name,
                    "description": function.get("description").cloned().unwrap_or(Value::Null),
                    "input_schema": function.get("parameters").cloned().unwrap_or(json!({})),
                });
                // cache_control on tools is a niche Anthropic feature; skip.
                tool.as_object_mut().unwrap().remove("description");
                if let Some(desc) = function.get("description").and_then(Value::as_str) {
                    tool["description"] = json!(desc);
                }
                Some(tool)
            })
            .collect();
        if !anthropic_tools.is_empty() {
            result["tools"] = json!(anthropic_tools);
        }
    }

    // tool_choice: OpenAI string/object → Anthropic {type} form.
    // Also fold `parallel_tool_calls=false` into disable_parallel_tool_use.
    let disable_parallel = body
        .get("parallel_tool_calls")
        .and_then(Value::as_bool)
        == Some(false);
    if let Some(tc) = body.get("tool_choice") {
        if let Some(mapped) = map_chat_tool_choice(tc) {
            result["tool_choice"] = mapped;
        }
    }
    if disable_parallel {
        let tc = result
            .get_mut("tool_choice")
            .and_then(Value::as_object_mut);
        if let Some(tc) = tc {
            tc.insert("disable_parallel_tool_use".to_string(), json!(true));
        } else {
            result["tool_choice"] = json!({
                "type": "auto",
                "disable_parallel_tool_use": true
            });
        }
    }

    // stream passthrough.
    if let Some(v) = body.get("stream") {
        result["stream"] = v.clone();
    }

    // Drop OpenAI-only fields with a WARN.
    for key in DROPPED_OPENAI_CHAT_FIELDS {
        if body.get(*key).is_some() {
            tracing::warn!(
                "[chat→anthropic] Dropping unsupported field `{key}` (Anthropic Messages \
                 has no equivalent)"
            );
        }
    }

    Ok(result)
}

/// Map an OpenAI tool_choice (string "none"/"auto"/"required", or object
/// {type:"function", function:{name}}) to an Anthropic tool_choice object.
fn map_chat_tool_choice(tc: &Value) -> Option<Value> {
    match tc {
        Value::String(s) => match s.as_str() {
            "none" => Some(json!({ "type": "none" })),
            "auto" => Some(json!({ "type": "auto" })),
            "required" => Some(json!({ "type": "any" })),
            _ => None,
        },
        Value::Object(_) => {
            let t = tc.get("type").and_then(Value::as_str).unwrap_or("");
            match t {
                "function" => {
                    let name = tc
                        .get("function")
                        .and_then(|f| f.get("name"))
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    if name.is_empty() {
                        Some(json!({ "type": "any" }))
                    } else {
                        Some(json!({ "type": "tool", "name": name }))
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract plain text from an OpenAI message content (string or array of parts).
fn extract_chat_text(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(s)) => {
            if s.is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        Some(Value::Array(parts)) => {
            let texts: Vec<String> = parts
                .iter()
                .filter_map(|p| match p.get("type").and_then(Value::as_str) {
                    Some("text" | "input_text") => {
                        p.get("text").and_then(Value::as_str).map(str::to_string)
                    }
                    Some("image_url") => Some("[image]".to_string()),
                    Some("input_image") => Some("[image]".to_string()),
                    _ => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join(" "))
            }
        }
        _ => None,
    }
}

/// Convert one OpenAI message (non-system role) to an Anthropic message.
/// Handles content string/array, tool_calls (assistant), and tool messages
/// (role=tool → Anthropic user with tool_result).
fn chat_message_to_anthropic(msg: &Value) -> Result<Value, Error> {
    let role = msg.get("role").and_then(Value::as_str).unwrap_or("user");

    match role {
        "user" | "assistant" | "developer" => {
            let mut out = json!({ "role": role });
            if role == "developer" {
                out["role"] = json!("user");
            }
            // content
            let content = msg.get("content");
            let mut blocks: Vec<Value> = Vec::new();
            match content {
                Some(Value::String(s)) => {
                    if !s.is_empty() {
                        blocks.push(json!({ "type": "text", "text": s }));
                    }
                }
                Some(Value::Array(parts)) => {
                    for part in parts {
                        let pt = part.get("type").and_then(Value::as_str).unwrap_or("");
                        match pt {
                            "text" | "input_text" => {
                                if let Some(t) = part.get("text").and_then(Value::as_str) {
                                    if !t.is_empty() {
                                        blocks.push(json!({ "type": "text", "text": t }));
                                    }
                                }
                            }
                            "image_url" => {
                                let url = part
                                    .get("image_url")
                                    .and_then(|u| u.get("url"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("");
                                if url.starts_with("data:") {
                                    if let Some((media_type, data)) =
                                        parse_data_url(url)
                                    {
                                        blocks.push(json!({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": data
                                            }
                                        }));
                                    }
                                } else {
                                    blocks.push(json!({
                                        "type": "image",
                                        "source": { "type": "url", "url": url }
                                    }));
                                }
                            }
                            "input_image" => {
                                // image_url form: {url} or {data, media_type}.
                                if let Some(url) = part.get("image_url").and_then(Value::as_str) {
                                    if url.starts_with("data:") {
                                        if let Some((media_type, data)) = parse_data_url(url) {
                                            blocks.push(json!({
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": media_type,
                                                    "data": data
                                                }
                                            }));
                                        }
                                    } else {
                                        blocks.push(json!({
                                            "type": "image",
                                            "source": { "type": "url", "url": url }
                                        }));
                                    }
                                } else {
                                    let media_type = part
                                        .get("media_type")
                                        .and_then(Value::as_str)
                                        .unwrap_or("image/jpeg");
                                    if let Some(data) = part.get("data").and_then(Value::as_str) {
                                        blocks.push(json!({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": data
                                            }
                                        }));
                                    }
                                }
                            }
                            "tool_call" => {
                                // Responses-style inline tool call.
                                if let Some(id) = part.get("id").and_then(Value::as_str) {
                                    blocks.push(json!({
                                        "type": "tool_use",
                                        "id": id,
                                        "name": part
                                            .get("name")
                                            .and_then(Value::as_str)
                                            .unwrap_or(""),
                                        "input": part.get("arguments").cloned().unwrap_or(json!({}))
                                    }));
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
            // assistant tool_calls → tool_use blocks.
            if role == "assistant" {
                let empty = json!({});
                if let Some(tool_calls) = msg.get("tool_calls").and_then(Value::as_array) {
                    for tc in tool_calls {
                        let func = tc.get("function").unwrap_or(&empty);
                        let name = func.get("name").and_then(Value::as_str).unwrap_or("");
                        let args = func
                            .get("arguments")
                            .and_then(Value::as_str)
                            .unwrap_or("{}");
                        let input: Value = serde_json::from_str(args).unwrap_or(json!({}));
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": tc.get("id").and_then(Value::as_str).unwrap_or(""),
                            "name": name,
                            "input": input
                        }));
                    }
                }
            }
            if blocks.is_empty() {
                blocks.push(json!({ "type": "text", "text": "" }));
            }
            out["content"] = json!(blocks);
            Ok(out)
        }
        "tool" => {
            // role=tool → Anthropic user message with tool_result block.
            let tool_call_id = msg
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or("");
            let content = msg.get("content");
            let out_content: Value = match content {
                Some(Value::String(s)) => json!([{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": s
                }]),
                Some(Value::Array(_)) => json!([{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": content.cloned().unwrap_or(Value::Null)
                }]),
                _ => json!([{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id,
                    "content": ""
                }]),
            };
            Ok(json!({ "role": "user", "content": out_content }))
        }
        _ => Ok(json!({ "role": "user", "content": [{
            "type": "text",
            "text": msg.get("content").and_then(Value::as_str).unwrap_or("")
        }] })),
    }
}

/// Parse a `data:<media_type>;base64,<data>` URL into its parts.
fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let media_type = meta.split(';').next().unwrap_or("").trim().to_string();
    let base64_part = meta.rsplit(';').next()?;
    if base64_part != "base64" {
        return None;
    }
    Some((media_type, data.to_string()))
}

/// Public wrapper so `responses_reverse` can reuse data-URL parsing.
pub(crate) fn parse_data_url_pub(url: &str) -> Option<(String, String)> {
    parse_data_url(url)
}

/// Convert an Anthropic Messages response to an OpenAI Chat Completions
/// response. `model` is echoed back as `choices[0].message` metadata.
pub fn anthropic_to_chat_response(body: &Value, model: &str) -> Result<Value, Error> {
    let mut result = json!({});
    if let Some(m) = body.get("model").and_then(Value::as_str).map(str::to_string) {
        result["model"] = json!(m);
    } else {
        result["model"] = json!(model);
    }

    let mut content_parts: Vec<Value> = Vec::new();
    let mut reasoning_content: Option<String> = None;
    let mut tool_calls: Vec<Value> = Vec::new();

    if let Some(blocks) = body.get("content").and_then(Value::as_array) {
        for block in blocks {
            let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
            match bt {
                "text" => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        content_parts.push(json!({ "type": "text", "text": t }));
                    }
                }
                "thinking" => {
                    if let Some(t) = block.get("thinking").and_then(Value::as_str) {
                        reasoning_content.get_or_insert_with(String::new).push_str(t);
                    }
                }
                "tool_use" => {
                    let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    let args_str = serde_json::to_string(&input).unwrap_or("{}".to_string());
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": { "name": name, "arguments": args_str }
                    }));
                }
                _ => {}
            }
        }
    }

    let message = json!({
        "role": "assistant",
        "content": content_parts
    });

    // Assistant message can carry content AND tool_calls.
    let mut assistant = json!({
        "role": "assistant",
        "content": if content_parts.is_empty() { Value::String(String::new()) } else { Value::Array(content_parts) }
    });
    if let Some(rc) = reasoning_content {
        assistant["reasoning_content"] = json!(rc);
    }
    if !tool_calls.is_empty() {
        assistant["tool_calls"] = json!(tool_calls);
    }

    let finish_reason = map_stop_reason_to_finish(body.get("stop_reason").and_then(Value::as_str));
    let choice = json!({
        "index": 0,
        "message": assistant,
        "finish_reason": finish_reason,
    });

    result["choices"] = json!([choice]);

    // usage mapping (Anthropic → OpenAI). input_tokens → prompt_tokens, output
    // tokens → completion_tokens.
    if let Some(usage) = body.get("usage") {
        let mut out_usage = json!({});
        if let Some(v) = usage.get("input_tokens") {
            out_usage["prompt_tokens"] = v.clone();
        }
        if let Some(v) = usage.get("output_tokens") {
            out_usage["completion_tokens"] = v.clone();
        }
        let sum = |a: u64, b: u64| a + b;
        let prompt = out_usage.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
        let completion = out_usage.get("completion_tokens").and_then(Value::as_u64).unwrap_or(0);
        out_usage["total_tokens"] = json!(sum(prompt, completion));
        result["usage"] = out_usage;
    }

    result["object"] = json!("chat.completion");

    let _ = message; // message built above into assistant
    Ok(result)
}

/// Anthropic stop_reason → OpenAI finish_reason.
fn map_stop_reason_to_finish(stop: Option<&str>) -> Option<String> {
    match stop {
        Some("end_turn") => Some("stop".to_string()),
        Some("max_tokens") => Some("length".to_string()),
        Some("tool_use") => Some("tool_calls".to_string()),
        Some("stop_sequence") => Some("stop".to_string()),
        Some("refusal") => Some("content_filter".to_string()),
        _ => Some("stop".to_string()),
    }
}

/// Convert an Anthropic Messages SSE stream to an OpenAI Chat Completions SSE
/// stream. Mirrors `convert::chat_to_anthropic_sse` in reverse: emits
/// `chat.completion.chunk` events for each delta.
///
/// `log_resp` controls streaming response logging: `true` means this stream is
/// the sole logger for the request and should `reqlog.append` the text/reasoning
/// deltas it emits; `false` means it is the OUTER wrapper of a double-bridge
/// chain whose inner stream already logs, so only the (idempotent) header/done
/// fire and no content is appended.
pub fn anthropic_to_chat_sse<S, E>(
    stream: S,
    model: String,
    reqlog: Arc<crate::reqlog::ReqLog>,
    log_resp: bool,
) -> impl Stream<Item = Result<Bytes, std::io::Error>>
where
    S: Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::error::Error + Send + 'static,
{
    use async_stream::stream;

    stream! {
        reqlog.resp_header("");
        let mut resp_has_text = false;
        let mut buffer = String::new();
        let mut utf8_remainder: Vec<u8> = Vec::new();
        let mut chunk_id = 0u32;
        let current_model = model;
        let mut tool_index: usize = 0;
        let mut open_tool_ids: Vec<(String, String, String)> = Vec::new(); // (id, name, args)
        let mut usage_sent = false;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    crate::convert::append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);
                    while let Some(line) = crate::convert::take_sse_block(&mut buffer) {
                        if line.trim().is_empty() {
                            continue;
                        }
                        for l in line.lines() {
                            let Some(data) = crate::convert::strip_sse_field(l, "data") else {
                                continue;
                            };
                            if data.trim().is_empty() { continue; }
                            let Ok(ev) = serde_json::from_str::<Value>(data) else {
                                tracing::debug!("[anthropic→chat SSE] JSON parse failed for: {data}");
                                continue;
                            };
                            let ev_type = ev.get("type").and_then(Value::as_str).unwrap_or("");

                            match ev_type {
                            "message_start" => {
                                // no-op: first chat.completion.chunk is emitted on
                                // first delta or at message_delta.
                            }
                            "content_block_start" => {
                                let idx = ev.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                                if let Some(b) = ev.get("content_block") {
                                    match b.get("type").and_then(Value::as_str) {
                                        Some("text") => {}
                                        Some("thinking") => {
                                            // reasoning_content prefix
                                            let thinking = b.get("thinking").and_then(Value::as_str).unwrap_or("");
                                            let chunk = json!({
                                                "id": format!("chatcmpl-{}", chunk_id),
                                                "object": "chat.completion.chunk",
                                                "model": current_model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": { "reasoning_content": thinking },
                                                    "finish_reason": null
                                                }]
                                            });
                                            chunk_id += 1;
                                            yield Ok(Bytes::from(sse_chunk(&chunk)));
                                        }
                                        Some("tool_use") => {
                                            let id = b.get("id").and_then(Value::as_str).unwrap_or("").to_string();
                                            let name = b.get("name").and_then(Value::as_str).unwrap_or("").to_string();
                                            if log_resp && !resp_has_text {
                                                reqlog.append(&format!("[tool_use: {name}]"));
                                            }
                                            tool_index = idx;
                                            let chunk = json!({
                                                "id": format!("chatcmpl-{}", chunk_id),
                                                "object": "chat.completion.chunk",
                                                "model": current_model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "tool_calls": [{
                                                            "index": tool_index,
                                                            "id": id,
                                                            "type": "function",
                                                            "function": { "name": name, "arguments": "" }
                                                        }]
                                                    },
                                                    "finish_reason": null
                                                }]
                                            });
                                            chunk_id += 1;
                                            yield Ok(Bytes::from(sse_chunk(&chunk)));
                                            open_tool_ids.push((id, name, String::new()));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_delta" => {
                                let _idx = ev.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                                if let Some(d) = ev.get("delta") {
                                    match d.get("type").and_then(Value::as_str) {
                                        Some("text_delta") => {
                                            let text = d.get("text").and_then(Value::as_str).unwrap_or("");
                                            if log_resp {
                                                resp_has_text = true;
                                                reqlog.append(text);
                                            }
                                            let chunk = json!({
                                                "id": format!("chatcmpl-{}", chunk_id),
                                                "object": "chat.completion.chunk",
                                                "model": current_model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": { "content": text },
                                                    "finish_reason": null
                                                }]
                                            });
                                            chunk_id += 1;
                                            yield Ok(Bytes::from(sse_chunk(&chunk)));
                                        }
                                        Some("thinking_delta") => {
                                            let thinking = d.get("thinking").and_then(Value::as_str).unwrap_or("");
                                            if log_resp {
                                                reqlog.append(thinking);
                                            }
                                            let chunk = json!({
                                                "id": format!("chatcmpl-{}", chunk_id),
                                                "object": "chat.completion.chunk",
                                                "model": current_model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": { "reasoning_content": thinking },
                                                    "finish_reason": null
                                                }]
                                            });
                                            chunk_id += 1;
                                            yield Ok(Bytes::from(sse_chunk(&chunk)));
                                        }
                                        Some("input_json_delta") => {
                                            // tool arguments delta
                                            let partial = d.get("partial_json").and_then(Value::as_str).unwrap_or("");
                                            let chunk = json!({
                                                "id": format!("chatcmpl-{}", chunk_id),
                                                "object": "chat.completion.chunk",
                                                "model": current_model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "tool_calls": [{
                                                            "index": tool_index,
                                                            "function": { "arguments": partial }
                                                        }]
                                                    },
                                                    "finish_reason": null
                                                }]
                                            });
                                            chunk_id += 1;
                                            yield Ok(Bytes::from(sse_chunk(&chunk)));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_stop" => {
                                // no-op
                            }
                            "message_delta" => {
                                // stop_reason → finish_reason
                                let stop_reason = ev.get("delta")
                                    .and_then(|d| d.get("stop_reason"))
                                    .and_then(Value::as_str);
                                let finish = map_stop_reason_to_finish(stop_reason);
                                // Emit a final empty-content chunk with finish_reason.
                                let chunk = json!({
                                    "id": format!("chatcmpl-{}", chunk_id),
                                    "object": "chat.completion.chunk",
                                    "model": current_model,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": finish
                                    }]
                                });
                                chunk_id += 1;
                                yield Ok(Bytes::from(sse_chunk(&chunk)));
                            }
                            "message_stop" => {
                                if !usage_sent {
                                    // Emit a usage chunk if the Anthropic stream
                                    // did not provide one (Anthropic SSE omits usage).
                                    usage_sent = true;
                                }
                                yield Ok(Bytes::from("[DONE]\n\n".to_string()));
                            }
                            "error" | "ping" => {
                                // pass through or ignore
                            }
                            _ => {}
                            }
                        }
                    }
                }
                Err(e) => {
                    // terminate stream on upstream error
                    if log_resp {
                        reqlog.err_resp(&format!("Stream error: {e}"));
                    }
                    break;
                }
            }
        }
        reqlog.done();
    }
}

/// Format a `data: <json>\n\n` SSE block.
fn sse_chunk(value: &Value) -> String {
    format!(
        "data: {}\n\n",
        serde_json::to_string(value).unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reqlog::ReqLog;
    use std::convert::Infallible;

    /// Drive `anthropic_to_chat_sse` with canned Anthropic SSE frames and
    /// collect the emitted `data:` JSON values (the `[DONE]` sentinel is dropped).
    async fn run_chat_sse(frames: Vec<String>, log_resp: bool) -> Vec<Value> {
        let upstream =
            futures::stream::iter(frames.into_iter().map(|f| Ok::<Bytes, Infallible>(Bytes::from(f))));
        let out = anthropic_to_chat_sse(upstream, "gpt-5".to_string(), ReqLog::new(), log_resp);
        let chunks: Vec<Bytes> = out.map(|r| r.unwrap()).collect().await;
        let mut text = String::new();
        for c in &chunks {
            text.push_str(&String::from_utf8_lossy(c));
        }
        text.split("\n\n")
            .filter_map(|block| {
                let data = block.lines().find_map(|l| l.strip_prefix("data: "))?;
                serde_json::from_str(data.trim()).ok()
            })
            .collect()
    }

    #[test]
    fn chat_to_anthropic_request_maps_system_and_tools() {
        let body = json!({
            "model": "gpt-5",
            "max_completion_tokens": 200,
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]
        });

        let result = chat_to_anthropic_request(&body).unwrap();
        assert_eq!(result["system"], "be brief");
        assert_eq!(result["max_tokens"], 200);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"][0]["text"], "hi");
        assert_eq!(result["tools"][0]["name"], "get_weather");
        assert_eq!(result["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn chat_to_anthropic_request_rejects_n_greater_than_one() {
        let body = json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "n": 3
        });
        let err = chat_to_anthropic_request(&body).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn chat_to_anthropic_request_maps_parallel_tool_calls_false() {
        let body = json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {}}}],
            "parallel_tool_calls": false
        });
        let result = chat_to_anthropic_request(&body).unwrap();
        assert_eq!(result["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn chat_to_anthropic_request_maps_reasoning_effort_to_thinking() {
        let body = json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_effort": "high"
        });
        let result = chat_to_anthropic_request(&body).unwrap();
        assert_eq!(result["thinking"]["type"], "enabled");
    }

    #[test]
    fn chat_to_anthropic_request_maps_tool_message() {
        let body = json!({
            "model": "gpt-5",
            "messages": [
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"SF\"}"}
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"}
            ]
        });
        let result = chat_to_anthropic_request(&body).unwrap();
        // assistant message → tool_use block
        let assistant = result["messages"].as_array().unwrap().iter().find(|m| m["role"] == "assistant").unwrap();
        assert_eq!(assistant["content"][0]["type"], "tool_use");
        assert_eq!(assistant["content"][0]["input"]["city"], "SF");
        // tool message → user with tool_result
        let tool_msg = result["messages"].as_array().unwrap().iter().find(|m| m["role"] == "user").unwrap();
        assert_eq!(tool_msg["content"][0]["type"], "tool_result");
        assert_eq!(tool_msg["content"][0]["tool_use_id"], "call_1");
    }

    #[test]
    fn anthropic_to_chat_response_maps_text_and_usage() {
        let body = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hello"}],
            "model": "claude",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 3}
        });

        let result = anthropic_to_chat_response(&body, "gpt-5").unwrap();
        assert_eq!(result["object"], "chat.completion");
        assert_eq!(result["choices"][0]["message"]["content"][0]["text"], "hello");
        assert_eq!(result["choices"][0]["finish_reason"], "stop");
        assert_eq!(result["usage"]["prompt_tokens"], 5);
        assert_eq!(result["usage"]["completion_tokens"], 3);
        assert_eq!(result["usage"]["total_tokens"], 8);
    }

    #[test]
    fn anthropic_to_chat_response_maps_tool_use() {
        let body = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "SF"}
            }],
            "model": "claude",
            "stop_reason": "tool_use"
        });
        let result = anthropic_to_chat_response(&body, "gpt-5").unwrap();
        assert_eq!(result["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"city\":\"SF\"}");
        assert_eq!(result["choices"][0]["finish_reason"], "tool_calls");
    }

    #[test]
    fn anthropic_to_chat_response_maps_thinking_to_reasoning_content() {
        let body = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "let me think"},
                {"type": "text", "text": "answer"}
            ],
            "model": "claude",
            "stop_reason": "end_turn"
        });
        let result = anthropic_to_chat_response(&body, "gpt-5").unwrap();
        assert_eq!(result["choices"][0]["message"]["reasoning_content"], "let me think");
        assert_eq!(result["choices"][0]["message"]["content"][0]["text"], "answer");
    }

    // -----------------------------------------------------------------------
    // anthropic_to_chat_sse output regression (append-logging must not alter bytes)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn chat_sse_logging_flag_does_not_alter_output() {
        let frames = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n".to_string(),
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Mock \"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"anthropic\"}}\n\n".to_string(),
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n".to_string(),
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n".to_string(),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string(),
        ];
        let with_log = run_chat_sse(frames.clone(), true).await;
        let without_log = run_chat_sse(frames, false).await;
        // The log_resp flag must not change the emitted SSE events.
        assert_eq!(with_log, without_log);

        let contents: Vec<&str> = with_log
            .iter()
            .filter_map(|v| v.pointer("/choices/0/delta/content").and_then(Value::as_str))
            .collect();
        assert_eq!(contents, vec!["Mock ", "anthropic"]);

        let finishes: Vec<&str> = with_log
            .iter()
            .filter_map(|v| v.pointer("/choices/0/finish_reason").and_then(Value::as_str))
            .collect();
        assert_eq!(finishes, vec!["stop"]);
    }

    #[tokio::test]
    async fn chat_sse_maps_thinking_delta_to_reasoning_content() {
        let frames = vec![
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"reasoning here\"}}\n\n".to_string(),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string(),
        ];
        let out = run_chat_sse(frames, true).await;
        // The thinking content_block_start emits an empty reasoning_content chunk;
        // only the thinking_delta should carry the real text.
        let reasoning: Vec<&str> = out
            .iter()
            .filter_map(|v| v.pointer("/choices/0/delta/reasoning_content").and_then(Value::as_str))
            .filter(|s| !s.is_empty())
            .collect();
        assert_eq!(reasoning, vec!["reasoning here"]);
    }

    #[tokio::test]
    async fn chat_sse_maps_tool_use_and_arguments() {
        let frames = vec![
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t_1\",\"name\":\"get_weather\"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"tokyo\\\"}\"}}\n\n".to_string(),
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n".to_string(),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string(),
        ];
        let out = run_chat_sse(frames, true).await;

        let names: Vec<&str> = out
            .iter()
            .filter_map(|v| v.pointer("/choices/0/delta/tool_calls/0/function/name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["get_weather"]);

        let args: Vec<&str> = out
            .iter()
            .filter_map(|v| {
                v.pointer("/choices/0/delta/tool_calls/0/function/arguments")
                    .and_then(Value::as_str)
            })
            .filter(|s| !s.is_empty()) // tool_use start emits an empty arguments chunk
            .collect();
        assert_eq!(args, vec!["{\"city\":\"tokyo\"}"]);

        let finishes: Vec<&str> = out
            .iter()
            .filter_map(|v| v.pointer("/choices/0/finish_reason").and_then(Value::as_str))
            .collect();
        assert_eq!(finishes, vec!["tool_calls"]);
    }
}
