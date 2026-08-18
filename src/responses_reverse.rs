//! Reverse conversions: local OpenAI Responses entry → Anthropic Messages
//! upstream. Mirrors `transform_responses.rs` + `streaming_responses.rs` in the
//! opposite direction.
//!
//! Three functions:
//! - `responses_to_anthropic_request` — Responses request body → Anthropic request
//! - `anthropic_to_responses_response` — Anthropic response → Responses response
//! - `anthropic_to_responses_sse`      — Anthropic SSE → Responses SSE

use crate::error::Error;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::sync::Arc;

/// Convert an OpenAI Responses API request body to an Anthropic Messages body.
pub fn responses_to_anthropic_request(body: &Value) -> Result<Value, Error> {
    let mut result = json!({});

    if let Some(model) = body.get("model").and_then(Value::as_str) {
        result["model"] = json!(model);
    }

    // instructions → system
    if let Some(instructions) = body.get("instructions") {
        let system = match instructions {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|p| p.get("text").and_then(Value::as_str).map(str::to_string))
                .collect::<Vec<_>>()
                .join("\n\n"),
            _ => String::new(),
        };
        if !system.is_empty() {
            result["system"] = json!(system);
        }
    }

    // input → messages. Responses `input` is an array of message items or
    // `function_call_output` items.
    if let Some(input) = body.get("input").and_then(Value::as_array) {
        let messages = responses_input_to_messages(input)?;
        if !messages.is_empty() {
            result["messages"] = json!(messages);
        }
    }

    // max_output_tokens → max_tokens
    if let Some(v) = body.get("max_output_tokens") {
        result["max_tokens"] = v.clone();
    }
    if let Some(v) = body.get("temperature") {
        result["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        result["top_p"] = v.clone();
    }

    // reasoning.effort → thinking
    if let Some(effort) = body
        .pointer("/reasoning/effort")
        .and_then(Value::as_str)
    {
        match effort.to_ascii_lowercase().as_str() {
            "none" => {
                result["thinking"] = json!({ "type": "disabled" });
            }
            _ => {
                result["thinking"] = json!({ "type": "enabled" });
            }
        }
    }

    // tools → Anthropic tools
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let anthropic_tools: Vec<Value> = tools
            .iter()
            .filter_map(|t| {
                let name = t.get("name").and_then(Value::as_str)?;
                Some(json!({
                    "name": name,
                    "description": t.get("description").and_then(Value::as_str).unwrap_or(""),
                    "input_schema": t.get("parameters").cloned().unwrap_or(json!({}))
                }))
            })
            .collect();
        if !anthropic_tools.is_empty() {
            result["tools"] = json!(anthropic_tools);
        }
    }

    // parallel_tool_calls=false → tool_choice.disable_parallel_tool_use
    let disable_parallel = body
        .get("parallel_tool_calls")
        .and_then(Value::as_bool)
        == Some(false);
    // tool_choice: Responses uses {type: "auto"|"required"|"none"|"function", name}
    if let Some(tc) = body.get("tool_choice") {
        if let Some(mapped) = map_responses_tool_choice(tc) {
            result["tool_choice"] = mapped;
        }
    }
    if disable_parallel {
        let tc = result.get_mut("tool_choice").and_then(Value::as_object_mut);
        if let Some(tc) = tc {
            tc.insert("disable_parallel_tool_use".to_string(), json!(true));
        } else {
            result["tool_choice"] = json!({
                "type": "auto",
                "disable_parallel_tool_use": true
            });
        }
    }

    // stream passthrough
    if let Some(v) = body.get("stream") {
        result["stream"] = v.clone();
    }

    Ok(result)
}

/// Convert a Responses `input` array to Anthropic messages. Handles:
/// - `{type:"message", role, content:[...]}` items
/// - `{type:"function_call_output", call_id, output}` → user tool_result
fn responses_input_to_messages(input: &[Value]) -> Result<Vec<Value>, Error> {
    let mut messages: Vec<Value> = Vec::new();

    for item in input {
        let item_type = item.get("type").and_then(Value::as_str).unwrap_or("");
        match item_type {
            "message" => {
                let role = item.get("role").and_then(Value::as_str).unwrap_or("user");
                let mut content_blocks: Vec<Value> = Vec::new();
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for block in content {
                        let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
                        match bt {
                            "input_text" | "output_text" => {
                                if let Some(t) = block.get("text").and_then(Value::as_str) {
                                    if !t.is_empty() {
                                        content_blocks.push(json!({"type":"text","text":t}));
                                    }
                                }
                            }
                            "input_image" => {
                                if let Some(url) = block.get("image_url").and_then(Value::as_str) {
                                    if url.starts_with("data:") {
                                        if let Some((mt, data)) = crate::convert_reverse::parse_data_url_pub(url) {
                                            content_blocks.push(json!({
                                                "type":"image",
                                                "source":{"type":"base64","media_type":mt,"data":data}
                                            }));
                                        }
                                    } else {
                                        content_blocks.push(json!({
                                            "type":"image",
                                            "source":{"type":"url","url":url}
                                        }));
                                    }
                                } else if let Some(data) = block.get("data").and_then(Value::as_str) {
                                    let mt = block.get("media_type").and_then(Value::as_str).unwrap_or("image/jpeg");
                                    content_blocks.push(json!({
                                        "type":"image",
                                        "source":{"type":"base64","media_type":mt,"data":data}
                                    }));
                                }
                            }
                            "function_call" => {
                                // inline function call → tool_use
                                content_blocks.push(json!({
                                    "type":"tool_use",
                                    "id": block.get("call_id").and_then(Value::as_str).unwrap_or(""),
                                    "name": block.get("name").and_then(Value::as_str).unwrap_or(""),
                                    "input": block.get("arguments").and_then(Value::as_str)
                                        .and_then(|a| serde_json::from_str(a).ok())
                                        .unwrap_or(json!({}))
                                }));
                            }
                            _ => {}
                        }
                    }
                }
                // role assistant + tool_calls at item level? Responses puts
                // function calls in separate items, so role/message here is text.
                if content_blocks.is_empty() {
                    content_blocks.push(json!({"type":"text","text":""}));
                }
                messages.push(json!({
                    "role": role,
                    "content": content_blocks
                }));
            }
            "function_call_output" => {
                let call_id = item.get("call_id").and_then(Value::as_str).unwrap_or("");
                let output = item.get("output").and_then(Value::as_str).unwrap_or("");
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": call_id,
                        "content": output
                    }]
                }));
            }
            "reasoning" => {
                // reasoning items in input are replay of prior reasoning; skip.
            }
            _ => {}
        }
    }

    Ok(messages)
}

/// Map a Responses tool_choice to Anthropic form.
fn map_responses_tool_choice(tc: &Value) -> Option<Value> {
    let t = tc.get("type").and_then(Value::as_str).unwrap_or("");
    match t {
        "none" => Some(json!({ "type": "none" })),
        "auto" => Some(json!({ "type": "auto" })),
        "required" => Some(json!({ "type": "any" })),
        "function" => {
            let name = tc.get("name").and_then(Value::as_str).unwrap_or("");
            if name.is_empty() {
                Some(json!({ "type": "any" }))
            } else {
                Some(json!({ "type": "tool", "name": name }))
            }
        }
        _ => None,
    }
}

/// Convert an Anthropic Messages response to an OpenAI Responses response.
/// `model` is echoed back.
pub fn anthropic_to_responses_response(body: &Value, model: &str) -> Result<Value, Error> {
    let mut result = json!({});

    let id = body.get("id").and_then(Value::as_str).unwrap_or("");
    result["id"] = json!(format!("resp_{id}"));
    result["object"] = json!("response");
    result["model"] = json!(body.get("model").and_then(Value::as_str).unwrap_or(model));
    result["status"] = json!(match body.get("stop_reason").and_then(Value::as_str) {
        Some("end_turn") => "completed",
        Some("stop_sequence") => "completed",
        Some("max_tokens") => "incomplete",
        Some("tool_use") => "in_progress",
        _ => "completed",
    });

    let mut output: Vec<Value> = Vec::new();
    let mut text_blocks: Vec<Value> = Vec::new();

    if let Some(blocks) = body.get("content").and_then(Value::as_array) {
        for block in blocks {
            let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
            match bt {
                "text" => {
                    if let Some(t) = block.get("text").and_then(Value::as_str) {
                        text_blocks.push(json!({ "type": "output_text", "text": t }));
                    }
                }
                "thinking" => {
                    // thinking → reasoning item (opaque summary).
                    let thinking = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                    let summary = block.get("signature").map(|_| json!({})).unwrap_or(json!({}));
                    output.push(json!({
                        "id": format!("rs_{}", output.len()),
                        "type": "reasoning",
                        "summary": if thinking.is_empty() { Vec::<Value>::new() } else { vec![json!({"type":"summary_text","text":thinking})] },
                        "content": vec![json!({"type":"summary_text","text":thinking})],
                    }));
                    let _ = summary;
                }
                "tool_use" => {
                    let id = block.get("id").and_then(Value::as_str).unwrap_or("");
                    let name = block.get("name").and_then(Value::as_str).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    let args = serde_json::to_string(&input).unwrap_or("{}".to_string());
                    output.push(json!({
                        "id": format!("fc_{}", output.len()),
                        "type": "function_call",
                        "call_id": id,
                        "name": name,
                        "arguments": args,
                        "status": "completed"
                    }));
                }
                _ => {}
            }
        }
    }

    // If there is any text, emit a single message item (per Responses API, a
    // response can have at most one message item).
    if !text_blocks.is_empty() {
        let mut msg = json!({
            "id": format!("msg_{}", output.len()),
            "type": "message",
            "role": "assistant",
            "content": text_blocks,
            "status": "completed"
        });
        let _ = &mut msg;
        output.insert(0, msg);
    }

    result["output"] = json!(output);
    result["usage"] = build_responses_usage(body.get("usage"));

    Ok(result)
}

/// Anthropic usage → Responses usage fields. `total_tokens` is REQUIRED by
/// OpenAI's responses wire format (codex hard-fails parsing ResponseCompleted
/// without it), so it is always present (input + output).
fn build_responses_usage(usage: Option<&Value>) -> Value {
    let mut out = json!({});
    let mut total: u64 = 0;
    if let Some(u) = usage {
        if let Some(v) = u.get("input_tokens").and_then(Value::as_u64) {
            out["input_tokens"] = json!(v);
            total += v;
        }
        if let Some(v) = u.get("output_tokens").and_then(Value::as_u64) {
            out["output_tokens"] = json!(v);
            total += v;
        }
        if let Some(v) = u.get("cache_creation_input_tokens").and_then(Value::as_u64) {
            out["input_tokens_details"]["cached_tokens"] = json!(v);
        }
    }
    out["total_tokens"] = json!(total);
    out
}

/// Convert an Anthropic Messages SSE stream to an OpenAI Responses SSE stream.
///
/// `log_resp` controls streaming response logging: `true` means this stream is
/// the sole logger for the request and should `reqlog.append` the text/tool
/// deltas it emits; `false` means it is the OUTER wrapper of a double-bridge
/// chain whose inner stream already logs, so only the (idempotent) header/done
/// fire and no content is appended.
pub fn anthropic_to_responses_sse<S, E>(
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
        let current_model = model;
        let mut response_id = String::new();
        let mut content_index: usize = 0;
        let mut tool_item_count: usize = 0;
        let mut done = false;
        // --- aggregation for the terminal snapshots codex renders from ---
        // Anthropic streams content as deltas only; codex assembles the final
        // reply from output_item.done / response.completed, so we must carry the
        // accumulated text/tool state forward into those events instead of
        // emitting them empty.
        let mut text_acc: std::collections::BTreeMap<usize, String> = std::collections::BTreeMap::new();
        let mut fc_meta: std::collections::BTreeMap<usize, (String, String)> = std::collections::BTreeMap::new(); // out_idx -> (call_id, name)
        let mut fc_args: std::collections::BTreeMap<usize, String> = std::collections::BTreeMap::new();           // out_idx -> accumulated JSON
        // anthropic block index -> responses output_index, plus whether it is a
        // tool block. codex finalizes each output item via a `*.output_item.done`
        // event; without one the function_call never becomes executable, so on
        // content_block_stop for a tool we must emit the terminal function_call
        // events (real OpenAI Responses sends these too).
        let mut block_out: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
        let mut block_is_tool: std::collections::BTreeMap<usize, bool> = std::collections::BTreeMap::new();
        let mut msg_status: &str = "completed";
        let mut response_usage: Option<Value> = None;

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

                        let Ok(ev) = serde_json::from_str::<Value>(data) else { continue };
                        let ev_type = ev.get("type").and_then(Value::as_str).unwrap_or("");

                        match ev_type {
                            "message_start" => {
                                response_id = ev.get("message")
                                    .and_then(|m| m.get("id"))
                                    .and_then(Value::as_str)
                                    .unwrap_or("resp_").to_string();
                                // response.created
                                let created = json!({
                                    "type": "response.created",
                                    "response": {
                                        "id": response_id,
                                        "object": "response",
                                        "created_at": 0,
                                        "status": "in_progress",
                                        "model": current_model,
                                        "output": []
                                    }
                                });
                                yield Ok(Bytes::from(sse_block("response.created", &created)));
                            }
                            "content_block_start" => {
                                let idx = ev.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                                if let Some(b) = ev.get("content_block") {
                                    match b.get("type").and_then(Value::as_str) {
                                        Some("text") => {
                                            content_index = idx;
                                            block_out.insert(idx, content_index);
                                            block_is_tool.insert(idx, false);
                                            // response.output_item.added + response.content_part.added
                                            let item = json!({
                                                "type": "response.output_item.added",
                                                "output_index": content_index,
                                                "item": {
                                                    "id": format!("msg_{}", content_index),
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "content": [],
                                                    "status": "in_progress"
                                                }
                                            });
                                            yield Ok(Bytes::from(sse_block("response.output_item.added", &item)));
                                        }
                                        Some("tool_use") => {
                                            tool_item_count += 1;
                                            block_out.insert(idx, tool_item_count);
                                            block_is_tool.insert(idx, true);
                                            let id = b.get("id").and_then(Value::as_str).unwrap_or("");
                                            let name = b.get("name").and_then(Value::as_str).unwrap_or("");
                                            fc_meta.insert(tool_item_count, (id.to_string(), name.to_string()));
                                            fc_args.insert(tool_item_count, String::new());
                                            if log_resp && !resp_has_text {
                                                reqlog.append(&format!("[tool_use: {name}]"));
                                            }
                                            let item = json!({
                                                "type": "response.output_item.added",
                                                "output_index": tool_item_count,
                                                "item": {
                                                    "id": format!("fc_{}", tool_item_count),
                                                    "type": "function_call",
                                                    "call_id": id,
                                                    "name": name,
                                                    "arguments": "",
                                                    "status": "in_progress"
                                                }
                                            });
                                            yield Ok(Bytes::from(sse_block("response.output_item.added", &item)));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_delta" => {
                                if let Some(d) = ev.get("delta") {
                                    match d.get("type").and_then(Value::as_str) {
                                        Some("text_delta") => {
                                            let text = d.get("text").and_then(Value::as_str).unwrap_or("");
                                            text_acc.entry(content_index).or_default().push_str(text);
                                            if log_resp {
                                                resp_has_text = true;
                                                reqlog.append(text);
                                            }
                                            let delta = json!({
                                                "type": "response.output_text.delta",
                                                "item_id": format!("msg_{}", content_index),
                                                "output_index": content_index,
                                                "content_index": 0,
                                                "delta": text
                                            });
                                            yield Ok(Bytes::from(sse_block("response.output_text.delta", &delta)));
                                        }
                                        Some("input_json_delta") => {
                                            let partial = d.get("partial_json").and_then(Value::as_str).unwrap_or("");
                                            fc_args.entry(tool_item_count).or_default().push_str(partial);
                                            let delta = json!({
                                                "type": "response.function_call_arguments.delta",
                                                "item_id": format!("fc_{}", tool_item_count),
                                                "output_index": tool_item_count,
                                                "delta": partial
                                            });
                                            yield Ok(Bytes::from(sse_block("response.function_call_arguments.delta", &delta)));
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            "content_block_stop" => {
                                let idx = ev.get("index").and_then(Value::as_u64).unwrap_or(0) as usize;
                                if block_is_tool.get(&idx).copied().unwrap_or(false) {
                                    // Emit the terminal function_call events so codex
                                    // finalizes the tool as executable. Its arguments
                                    // are complete here (all deltas accumulated above).
                                    let oidx = block_out.get(&idx).copied().unwrap_or(idx);
                                    let args = fc_args.get(&oidx).cloned().unwrap_or_default();
                                    let (call_id, name) = fc_meta
                                        .get(&oidx)
                                        .cloned()
                                        .unwrap_or_else(|| ("".to_string(), "".to_string()));
                                    let args_done = json!({
                                        "type": "response.function_call_arguments.done",
                                        "item_id": format!("fc_{}", oidx),
                                        "output_index": oidx,
                                        "arguments": args
                                    });
                                    yield Ok(Bytes::from(sse_block("response.function_call_arguments.done", &args_done)));
                                    let item_done = json!({
                                        "type": "response.output_item.done",
                                        "output_index": oidx,
                                        "item": {
                                            "id": format!("fc_{}", oidx),
                                            "type": "function_call",
                                            "call_id": call_id,
                                            "name": name,
                                            "arguments": args,
                                            "status": "completed"
                                        }
                                    });
                                    yield Ok(Bytes::from(sse_block("response.output_item.done", &item_done)));
                                } else {
                                    let part_text = text_acc.get(&idx).cloned().unwrap_or_default();
                                    let oidx = block_out.get(&idx).copied().unwrap_or(idx);
                                    let stop = json!({
                                        "type": "response.content_part.done",
                                        "item_id": format!("msg_{}", oidx),
                                        "output_index": oidx,
                                        "content_index": 0,
                                        "part": { "type": "output_text", "text": part_text }
                                    });
                                    yield Ok(Bytes::from(sse_block("response.content_part.done", &stop)));
                                }
                            }
                            "message_delta" => {
                                if let Some(u) = ev.get("usage") {
                                    response_usage = Some(u.clone());
                                }
                                let stop_reason = ev.get("delta")
                                    .and_then(|d| d.get("stop_reason"))
                                    .and_then(Value::as_str);
                                let status = match stop_reason {
                                    Some("end_turn") | Some("stop_sequence") => "completed",
                                    Some("max_tokens") => "incomplete",
                                    Some("tool_use") => "in_progress",
                                    _ => "completed",
                                };
                                msg_status = status;
                                let text: String = text_acc.values().cloned().collect();
                                let content = if text.is_empty() {
                                    Vec::new()
                                } else {
                                    vec![json!({ "type": "output_text", "text": text, "annotations": [] })]
                                };
                                let done_ev = json!({
                                    "type": "response.output_item.done",
                                    "output_index": content_index,
                                    "item": {
                                        "id": format!("msg_{}", content_index),
                                        "type": "message",
                                        "role": "assistant",
                                        "content": content,
                                        "status": status
                                    }
                                });
                                yield Ok(Bytes::from(sse_block("response.output_item.done", &done_ev)));
                            }
                            "message_stop" if !done => {
                                done = true;
                                // Assemble the final output: a single message item holding all
                                // streamed text, followed by any completed function calls,
                                // mirroring anthropic_to_responses_response.
                                let text: String = text_acc.values().cloned().collect();
                                let mut output: Vec<Value> = Vec::new();
                                if !text.is_empty() {
                                    output.push(json!({
                                        "id": format!("msg_{}", content_index),
                                        "type": "message",
                                        "role": "assistant",
                                        "content": vec![json!({ "type": "output_text", "text": text, "annotations": [] })],
                                        "status": msg_status
                                    }));
                                }
                                for (idx, (call_id, name)) in &fc_meta {
                                    let args = fc_args.get(idx).cloned().unwrap_or_default();
                                    output.push(json!({
                                        "id": format!("fc_{}", idx),
                                        "type": "function_call",
                                        "call_id": call_id,
                                        "name": name,
                                        "arguments": args,
                                        "status": "completed"
                                    }));
                                }
                                let completed = json!({
                                    "type": "response.completed",
                                    "response": {
                                        "id": response_id,
                                        "object": "response",
                                        "created_at": 0,
                                        "status": msg_status,
                                        "model": current_model,
                                        "output": output,
                                        "usage": build_responses_usage(response_usage.as_ref())
                                    }
                                });
                                yield Ok(Bytes::from(sse_block("response.completed", &completed)));
                                yield Ok(Bytes::from("data: [DONE]\n\n".to_string()));
                            }
                            _ => {}
                            }
                        }
                    }
                }
                Err(e) => {
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

/// Format a `event: <name>\ndata: <json>\n\n` SSE block.
fn sse_block(event: &str, value: &Value) -> String {
    format!(
        "event: {event}\ndata: {}\n\n",
        serde_json::to_string(value).unwrap_or_default()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reqlog::ReqLog;
    use std::convert::Infallible;

    /// Drive `anthropic_to_responses_sse` with canned Anthropic SSE frames and
    /// collect the emitted `(event, data)` pairs (the `data: [DONE]` terminator
    /// is dropped).
    async fn run_responses_sse(frames: Vec<String>, log_resp: bool) -> Vec<(String, Value)> {
        let upstream = futures::stream::iter(
            frames.into_iter().map(|f| Ok::<Bytes, Infallible>(Bytes::from(f))),
        );
        let out =
            anthropic_to_responses_sse(upstream, "gpt-5".to_string(), ReqLog::new(), log_resp);
        let chunks: Vec<Bytes> = out.map(|r| r.unwrap()).collect().await;
        let mut text = String::new();
        for c in &chunks {
            text.push_str(&String::from_utf8_lossy(c));
        }
        text.split("\n\n")
            .filter(|b| !b.trim().is_empty())
            .filter_map(|block| {
                let mut ev = String::new();
                let mut data = String::new();
                for line in block.lines() {
                    if let Some(v) = line.strip_prefix("event: ") {
                        ev = v.to_string();
                    } else if let Some(v) = line.strip_prefix("data: ") {
                        data = v.to_string();
                    }
                }
                if data.trim() == "[DONE]" {
                    return None;
                }
                Some((ev, serde_json::from_str(&data).unwrap_or(Value::Null)))
            })
            .collect()
    }

    #[test]
    fn responses_to_anthropic_request_maps_instructions_and_input() {
        let body = json!({
            "model": "gpt-5",
            "instructions": "be brief",
            "max_output_tokens": 300,
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}]
            }]
        });

        let result = responses_to_anthropic_request(&body).unwrap();
        assert_eq!(result["system"], "be brief");
        assert_eq!(result["max_tokens"], 300);
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"][0]["text"], "hello");
    }

    #[test]
    fn responses_to_anthropic_request_maps_function_call_output() {
        let body = json!({
            "model": "gpt-5",
            "input": [{
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "sunny"
            }]
        });
        let result = responses_to_anthropic_request(&body).unwrap();
        assert_eq!(result["messages"][0]["role"], "user");
        assert_eq!(result["messages"][0]["content"][0]["type"], "tool_result");
        assert_eq!(result["messages"][0]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(result["messages"][0]["content"][0]["content"], "sunny");
    }

    #[test]
    fn responses_to_anthropic_request_maps_reasoning_effort() {
        let body = json!({
            "model": "gpt-5",
            "input": [],
            "reasoning": {"effort": "high"}
        });
        let result = responses_to_anthropic_request(&body).unwrap();
        assert_eq!(result["thinking"]["type"], "enabled");
    }

    #[test]
    fn anthropic_to_responses_response_maps_text_and_tool() {
        let body = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "SF"}}
            ],
            "model": "claude",
            "stop_reason": "tool_use"
        });
        let result = anthropic_to_responses_response(&body, "gpt-5").unwrap();
        assert_eq!(result["object"], "response");
        assert_eq!(result["status"], "in_progress");
        // message item
        let msg_item = result["output"].as_array().unwrap().iter().find(|i| i["type"] == "message").unwrap();
        assert_eq!(msg_item["content"][0]["type"], "output_text");
        assert_eq!(msg_item["content"][0]["text"], "hello");
        // function_call item
        let fc = result["output"].as_array().unwrap().iter().find(|i| i["type"] == "function_call").unwrap();
        assert_eq!(fc["name"], "get_weather");
        assert_eq!(fc["arguments"], "{\"city\":\"SF\"}");
    }

    #[test]
    fn anthropic_to_responses_response_maps_end_turn_status() {
        let body = json!({
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "model": "claude",
            "stop_reason": "end_turn"
        });
        let result = anthropic_to_responses_response(&body, "gpt-5").unwrap();
        assert_eq!(result["status"], "completed");
    }

    // -----------------------------------------------------------------------
    // anthropic_to_responses_sse output regression (append-logging must not alter bytes)
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn responses_sse_logging_flag_does_not_alter_output() {
        let frames = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n".to_string(),
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Mock \"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"anthropic\"}}\n\n".to_string(),
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n".to_string(),
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n".to_string(),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string(),
        ];
        let with_log = run_responses_sse(frames.clone(), true).await;
        let without_log = run_responses_sse(frames, false).await;
        // The log_resp flag must not change the emitted SSE events.
        assert_eq!(with_log, without_log);

        let ev_names: Vec<&str> = with_log.iter().map(|(e, _)| e.as_str()).collect();
        assert_eq!(
            ev_names,
            vec![
                "response.created",
                "response.output_item.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );

        let deltas: Vec<&str> = with_log
            .iter()
            .filter(|(e, _)| e == "response.output_text.delta")
            .map(|(_, v)| v.get("delta").and_then(Value::as_str).unwrap_or(""))
            .collect();
        assert_eq!(deltas, vec!["Mock ", "anthropic"]);
    }

    #[tokio::test]
    async fn responses_sse_maps_tool_use_and_arguments() {
        let frames = vec![
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\"}}\n\n".to_string(),
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"t_1\",\"name\":\"get_weather\"}}\n\n".to_string(),
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\\\"tokyo\\\"}\"}}\n\n".to_string(),
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n".to_string(),
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n".to_string(),
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string(),
        ];
        let out = run_responses_sse(frames, true).await;

        let fc = out
            .iter()
            .find(|(e, v)| e == "response.output_item.added" && v["item"]["type"] == "function_call")
            .map(|(_, v)| v.clone())
            .expect("function_call item emitted");
        assert_eq!(fc["item"]["name"], "get_weather");

        let arg_deltas: Vec<&str> = out
            .iter()
            .filter(|(e, _)| e == "response.function_call_arguments.delta")
            .filter_map(|(_, v)| v.get("delta").and_then(Value::as_str))
            .collect();
        assert_eq!(arg_deltas, vec!["{\"city\":\"tokyo\"}"]);

        // codex finalizes a tool from its terminal events: it must receive a
        // `function_call_arguments.done` and a function_call `output_item.done`
        // (with the fully assembled arguments) or the tool is never executed.
        let args_done = out
            .iter()
            .find(|(e, _)| e == "response.function_call_arguments.done")
            .map(|(_, v)| v.clone())
            .expect("function_call_arguments.done emitted");
        assert_eq!(args_done["arguments"], "{\"city\":\"tokyo\"}");

        let fc_done = out
            .iter()
            .find(|(e, v)| e == "response.output_item.done" && v["item"]["type"] == "function_call")
            .map(|(_, v)| v.clone())
            .expect("function_call output_item.done emitted");
        assert_eq!(fc_done["item"]["call_id"], "t_1");
        assert_eq!(fc_done["item"]["name"], "get_weather");
        assert_eq!(fc_done["item"]["arguments"], "{\"city\":\"tokyo\"}");
        assert_eq!(fc_done["item"]["status"], "completed");
    }
}
