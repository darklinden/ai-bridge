//! Anthropic Messages ↔ OpenAI Chat Completions format conversion
//!
//! Pure transformation functions extracted from cc-switch's proxy. No Tauri or
//! application-specific dependencies beyond [`crate::error::Error`].

use crate::error::Error;
use crate::json_canonical::canonical_json_string;
use crate::tool_media::{
    flush_pending_chat_tool_media, plan_chat_tool_output_media, queue_chat_tool_output_media,
};
use async_stream::stream;
use bytes::Bytes;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ANTHROPIC_BILLING_HEADER_PREFIX: &str = "x-anthropic-billing-header:";

/// Infinite-whitespace bug threshold (Copilot tool call arguments).
const INFINITE_WHITESPACE_THRESHOLD: usize = 500;

// ---------------------------------------------------------------------------
// Model detection helpers
// ---------------------------------------------------------------------------

/// Detect OpenAI o-series reasoning models (o1, o3, o4-mini, etc.).
pub fn is_openai_o_series(model: &str) -> bool {
    model.len() > 1
        && model.starts_with('o')
        && model.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit())
}

/// Detect models that support `reasoning_effort`.
///
/// In addition to the OpenAI o-series / GPT-5+ / Grok families, DeepSeek
/// models expose a top-level `reasoning_effort` field whose legal values are
/// `low` / `high` / `max` (clamped via [`clamp_reasoning_effort_for_deepseek`]).
pub fn supports_reasoning_effort(model: &str) -> bool {
    let normalized = model.to_lowercase();
    is_openai_o_series(&normalized)
        || normalized
            .strip_prefix("gpt-")
            .and_then(|rest| rest.chars().next())
            .is_some_and(|c| c.is_ascii_digit() && c >= '5')
        || normalized == "grok-4.5"
        || normalized.starts_with("grok-4.5-")
        || normalized.starts_with("grok-build-")
        || normalized.contains("deepseek")
}

/// Resolve the appropriate OpenAI `reasoning_effort` from an Anthropic request body.
pub fn resolve_reasoning_effort(body: &Value) -> Option<&'static str> {
    // Priority 1: explicit output_config.effort
    if let Some(effort) = body
        .pointer("/output_config/effort")
        .and_then(|v| v.as_str())
    {
        return match effort {
            "low" => Some("low"),
            "medium" => Some("medium"),
            "high" => Some("high"),
            "max" => Some("xhigh"),
            _ => None,
        };
    }

    // Priority 2: thinking.type + budget_tokens fallback
    let thinking = body.get("thinking")?;
    match thinking.get("type").and_then(|t| t.as_str()) {
        Some("adaptive") => Some("xhigh"),
        Some("enabled") => {
            let budget = thinking.get("budget_tokens").and_then(|b| b.as_u64());
            match budget {
                Some(b) if b < 4_000 => Some("low"),
                Some(b) if b < 16_000 => Some("medium"),
                Some(_) => Some("high"),
                None => Some("high"),
            }
        }
        // Explicitly suppress effort when thinking is disabled: some upstreams
        // (DeepSeek) reject `thinking.type=disabled` combined with a
        // `reasoning_effort` parameter (400). See parent CHANGELOG:404.
        Some("disabled") => None,
        _ => None,
    }
}

/// Clamp a resolved OpenAI `reasoning_effort` value to the legal DeepSeek enum
/// (`low` / `high` / `max`). Mirrors the parent `effort_value_mode: "deepseek"`
/// mapping (transform_codex_chat.rs): `max`/`xhigh` → `max`, everything else
/// (`low`/`medium`/`high`) → `high`, except `low` stays `low`.
pub fn clamp_reasoning_effort_for_deepseek(effort: &str) -> &'static str {
    match effort {
        "max" | "xhigh" => "max",
        "low" => "low",
        _ => "high",
    }
}

// ---------------------------------------------------------------------------
// Anthropic billing-header stripping
// ---------------------------------------------------------------------------

/// Strip only a leading Claude Code attribution line from system text.
pub fn strip_leading_anthropic_billing_header(text: &str) -> &str {
    if !text.starts_with(ANTHROPIC_BILLING_HEADER_PREFIX) {
        return text;
    }

    let Some(line_end) = text
        .as_bytes()
        .iter()
        .position(|byte| *byte == b'\n' || *byte == b'\r')
    else {
        return "";
    };

    let bytes = text.as_bytes();
    let mut rest_start = line_end + 1;
    if bytes[line_end] == b'\r' && bytes.get(line_end + 1) == Some(&b'\n') {
        rest_start += 1;
    }

    let rest = &text[rest_start..];
    if let Some(stripped) = rest.strip_prefix("\r\n") {
        stripped
    } else if let Some(stripped) = rest.strip_prefix('\n') {
        stripped
    } else if let Some(stripped) = rest.strip_prefix('\r') {
        stripped
    } else {
        rest
    }
}

// ---------------------------------------------------------------------------
// Anthropic → OpenAI Chat request conversion
// ---------------------------------------------------------------------------

/// Anthropic request → OpenAI Chat Completions request.
#[allow(dead_code)]
pub fn anthropic_to_openai(body: Value) -> Result<Value, Error> {
    anthropic_to_openai_with_reasoning_content(body, false)
}

/// Anthropic request → OpenAI Chat Completions request.
///
/// `preserve_reasoning_content`: enable for DeepSeek-compatible Zen models that
/// understand `reasoning_content` on assistant tool-call messages.
pub fn anthropic_to_openai_with_reasoning_content(
    body: Value,
    preserve_reasoning_content: bool,
) -> Result<Value, Error> {
    let mut result = json!({});

    if let Some(model) = body.get("model").and_then(|m| m.as_str()) {
        result["model"] = json!(model);
    }

    let mut messages = Vec::new();

    // System prompt
    if let Some(system) = body.get("system") {
        if let Some(text) = system.as_str() {
            let text = strip_leading_anthropic_billing_header(text);
            if !text.is_empty() {
                messages.push(json!({"role": "system", "content": text}));
            }
        } else if let Some(arr) = system.as_array() {
            for msg in arr {
                if let Some(text) = msg.get("text").and_then(|t| t.as_str()) {
                    let text = strip_leading_anthropic_billing_header(text);
                    if text.is_empty() {
                        continue;
                    }
                    messages.push(json!({"role": "system", "content": text}));
                }
            }
        }
    }

    // Messages
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|r| r.as_str()).unwrap_or("user");
            let content = msg.get("content");
            let converted = convert_message_to_openai(role, content, preserve_reasoning_content)?;
            messages.extend(converted);
        }
    }

    normalize_openai_system_messages(&mut messages);
    result["messages"] = json!(messages);

    // Parameters
    let model = body.get("model").and_then(|m| m.as_str()).unwrap_or("");
    if let Some(v) = body.get("max_tokens") {
        if is_openai_o_series(model) {
            result["max_completion_tokens"] = v.clone();
        } else {
            result["max_tokens"] = v.clone();
        }
    }
    if let Some(v) = body.get("temperature") {
        result["temperature"] = v.clone();
    }
    if let Some(v) = body.get("top_p") {
        result["top_p"] = v.clone();
    }
    if let Some(v) = body.get("stop_sequences") {
        result["stop"] = v.clone();
    }
    if let Some(v) = body.get("stream") {
        result["stream"] = v.clone();
    }

    // Map Anthropic thinking → OpenAI reasoning_effort. `model` here is the
    // configured upstream model (server.rs overwrites body["model"] before
    // conversion), so clamp to DeepSeek's legal enum when applicable.
    if supports_reasoning_effort(model) {
        if let Some(effort) = resolve_reasoning_effort(&body) {
            let effort = if model.contains("deepseek") {
                clamp_reasoning_effort_for_deepseek(effort)
            } else {
                effort
            };
            result["reasoning_effort"] = json!(effort);
        }
    }

    // Tools (filter BatchTool and SendMessage — internal Claude tools)
    if let Some(tools) = body.get("tools").and_then(|t| t.as_array()) {
        let openai_tools: Vec<Value> = tools
            .iter()
            .filter(|t| {
                let name = t.get("name").and_then(|v| v.as_str());
                let type_ = t.get("type").and_then(|v| v.as_str());
                type_ != Some("BatchTool") && name != Some("SendMessage")
            })
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.get("name").and_then(|n| n.as_str()).unwrap_or(""),
                        "description": t.get("description"),
                        "parameters": clean_schema(t.get("input_schema").cloned().unwrap_or(json!({})))
                    }
                })
            })
            .collect();

        if !openai_tools.is_empty() {
            result["tools"] = json!(openai_tools);
        }
    }

    if let Some(v) = body.get("tool_choice") {
        result["tool_choice"] = map_tool_choice_to_chat(v);
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Streaming helpers
// ---------------------------------------------------------------------------

/// Inject `stream_options.include_usage` for streaming requests.
pub fn inject_openai_stream_include_usage(result: &mut Value) {
    let is_stream = result
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if !is_stream {
        return;
    }
    match result.get_mut("stream_options") {
        Some(Value::Object(opts)) => {
            opts.insert("include_usage".to_string(), json!(true));
        }
        _ => {
            result["stream_options"] = json!({ "include_usage": true });
        }
    }
}

// ---------------------------------------------------------------------------
// Tool choice mapping
// ---------------------------------------------------------------------------

fn map_tool_choice_to_chat(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::String(s) => match s.as_str() {
            "any" => json!("required"),
            _ => json!(s),
        },
        Value::Object(obj) => match obj.get("type").and_then(|t| t.as_str()) {
            Some("any") => json!("required"),
            Some("auto") => json!("auto"),
            Some("none") => json!("none"),
            Some("tool") => {
                let name = obj.get("name").and_then(|n| n.as_str()).unwrap_or("");
                json!({
                    "type": "function",
                    "function": { "name": name }
                })
            }
            _ => tool_choice.clone(),
        },
        _ => tool_choice.clone(),
    }
}

// ---------------------------------------------------------------------------
// System message normalization
// ---------------------------------------------------------------------------

fn normalize_openai_system_messages(messages: &mut Vec<Value>) {
    let system_count = messages
        .iter()
        .filter(|message| message.get("role").and_then(|value| value.as_str()) == Some("system"))
        .count();

    if system_count == 0 {
        return;
    }

    if system_count == 1 {
        if let Some(index) = messages.iter().position(|message| {
            message.get("role").and_then(|value| value.as_str()) == Some("system")
        }) {
            if index > 0 {
                let message = messages.remove(index);
                messages.insert(0, message);
            }
        }
        return;
    }

    let mut parts = Vec::new();
    messages.retain(|message| {
        if message.get("role").and_then(|value| value.as_str()) != Some("system") {
            return true;
        }

        match message.get("content") {
            Some(Value::String(text)) if !text.is_empty() => parts.push(text.clone()),
            Some(Value::Array(content_parts)) => {
                let text = content_parts
                    .iter()
                    .filter_map(|part| part.get("text").and_then(|value| value.as_str()))
                    .collect::<Vec<_>>()
                    .join("\n");
                if !text.is_empty() {
                    parts.push(text);
                }
            }
            _ => {}
        }

        false
    });

    if !parts.is_empty() {
        messages.insert(0, json!({"role": "system", "content": parts.join("\n")}));
    }
}

// ---------------------------------------------------------------------------
// Single message conversion
// ---------------------------------------------------------------------------

/// Convert a single Anthropic message to one or more OpenAI Chat messages.
///
/// This is adapted from cc-switch's `convert_message_to_openai`:
///   - Image blocks are converted inline to `image_url` parts.
///   - Tool results with media are detected and extracted into a synthetic
///     user message, keeping tool messages text-only as required by the
///     Chat Completions API.
fn convert_message_to_openai(
    role: &str,
    content: Option<&Value>,
    preserve_reasoning_content: bool,
) -> Result<Vec<Value>, Error> {
    let mut result = Vec::new();

    let content = match content {
        Some(c) => c,
        None => {
            result.push(json!({"role": role, "content": null}));
            return Ok(result);
        }
    };

    // String content
    if let Some(text) = content.as_str() {
        result.push(json!({"role": role, "content": text}));
        return Ok(result);
    }

    // Array content (multimodal / tool calls)
    if let Some(blocks) = content.as_array() {
        let mut content_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut pending_tool_media = Vec::new();
        let mut reasoning_parts = Vec::new();

        for block in blocks {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");

            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        content_parts.push(json!({"type": "text", "text": text}));
                    }
                }
                "image" => {
                    // Convert Anthropic image → OpenAI image_url
                    if let Some(source) = block.get("source") {
                        let media_type = source
                            .get("media_type")
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/png");
                        let data = source
                            .get("data")
                            .and_then(|d| d.as_str())
                            .unwrap_or("");
                        content_parts.push(json!({
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{media_type};base64,{data}")
                            }
                        }));
                    }
                }
                "tool_use" => {
                    let id = block.get("id").and_then(|i| i.as_str()).unwrap_or("");
                    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let input = block.get("input").cloned().unwrap_or(json!({}));
                    tool_calls.push(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(&input).unwrap_or_default()
                        }
                    }));
                }
                "tool_result" => {
                    let tool_use_id = block
                        .get("tool_use_id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("");
                    let content_val = block.get("content");
                    let media_plan =
                        content_val.cloned().and_then(plan_chat_tool_output_media);
                    let content_str = if let Some(plan) = media_plan {
                        queue_chat_tool_output_media(
                            &mut pending_tool_media,
                            tool_use_id,
                            plan.media_parts,
                        );
                        plan.tool_content
                    } else {
                        // Keep the no-media representation exactly equal to
                        // the legacy converter for prompt-cache stability.
                        match content_val {
                            Some(Value::String(s)) => s.clone(),
                            Some(v) => canonical_json_string(v),
                            None => String::new(),
                        }
                    };
                    result.push(json!({
                        "role": "tool",
                        "tool_call_id": tool_use_id,
                        "content": content_str
                    }));
                }
                "thinking" => {
                    // Prefer decoding our own bridge signature (carrying the
                    // full reasoning text) over the possibly-truncated
                    // `thinking` text; fall back to the text when the signature
                    // is absent or not ours (e.g. a real Anthropic signature).
                    let recovered = block
                        .get("signature")
                        .and_then(|s| s.as_str())
                        .and_then(crate::reasoning_bridge::decode_openai_reasoning_item)
                        .and_then(|item| {
                            let text = crate::reasoning_bridge::reasoning_summary_text(&item);
                            (!text.is_empty()).then_some(text)
                        });
                    match recovered {
                        Some(text) => reasoning_parts.push(text),
                        None => {
                            if let Some(thinking) = block.get("thinking").and_then(|t| t.as_str()) {
                                if !thinking.is_empty() {
                                    reasoning_parts.push(thinking.to_string());
                                }
                            }
                        }
                    }
                }
                "redacted_thinking" if preserve_reasoning_content => {
                    let recovered = block
                        .get("data")
                        .and_then(|s| s.as_str())
                        .and_then(crate::reasoning_bridge::decode_openai_reasoning_item)
                        .and_then(|item| {
                            let text = crate::reasoning_bridge::reasoning_summary_text(&item);
                            (!text.is_empty()).then_some(text)
                        });
                    reasoning_parts.push(
                        recovered.unwrap_or_else(|| "[redacted thinking]".to_string()),
                    );
                }
                _ => {}
            }
        }

        // Chat tool messages cannot carry image parts. Keep parallel tool
        // results adjacent, then present all extracted media in one user turn
        // before any ordinary message content from the same Anthropic turn.
        flush_pending_chat_tool_media(&mut result, &mut pending_tool_media);

        // Add message with content and/or tool calls
        if !content_parts.is_empty() || !tool_calls.is_empty() {
            let mut msg = json!({"role": role});

            // Content
            if content_parts.is_empty() {
                msg["content"] = Value::Null;
            } else if content_parts.len() == 1 {
                if let Some(text) = content_parts[0].get("text") {
                    msg["content"] = text.clone();
                } else {
                    msg["content"] = json!(content_parts);
                }
            } else {
                msg["content"] = json!(content_parts);
            }

            // Tool calls
            if !tool_calls.is_empty() {
                msg["tool_calls"] = json!(tool_calls);
            }

            // Reasoning content for DeepSeek-compatible models
            if preserve_reasoning_content && role == "assistant" && !tool_calls.is_empty() {
                let reasoning_content = if reasoning_parts.is_empty() {
                    "tool call".to_string()
                } else {
                    reasoning_parts.join("\n")
                };
                msg["reasoning_content"] = json!(reasoning_content);
            }

            result.push(msg);
        }

        return Ok(result);
    }

    // Passthrough for other content types
    result.push(json!({"role": role, "content": content}));
    Ok(result)
}

// ---------------------------------------------------------------------------
// Schema cleaning
// ---------------------------------------------------------------------------

/// Clean tool input schema for OpenAI compatibility.
pub fn clean_schema(schema: Value) -> Value {
    clean_schema_inner(schema, true)
}

fn clean_schema_inner(mut schema: Value, is_root: bool) -> Value {
    if let Some(obj) = schema.as_object_mut() {
        let missing_type = is_root && !obj.contains_key("type");
        if missing_type {
            obj.insert("type".to_string(), json!("object"));
        }
        if missing_type && !obj.contains_key("properties") {
            obj.insert("properties".to_string(), json!({}));
        }

        // Remove format: uri
        if obj.get("format").and_then(|v| v.as_str()) == Some("uri") {
            obj.remove("format");
        }

        // Recurse
        if let Some(properties) = obj.get_mut("properties").and_then(|v| v.as_object_mut()) {
            for (_, value) in properties.iter_mut() {
                *value = clean_schema_inner(value.clone(), false);
            }
        }

        if let Some(items) = obj.get_mut("items") {
            *items = clean_schema_inner(items.clone(), false);
        }
    }
    schema
}

// ---------------------------------------------------------------------------
// SendMessage summary injection
// ---------------------------------------------------------------------------

/// Ensure `SendMessage` tool calls include a `summary` field in `input`.
///
/// The upstream model often omits the required `summary` parameter when calling
/// `SendMessage`. This helper fills it from the `message` content (first ~50
/// chars) or a default placeholder so Claude Code's runtime validation passes.
pub fn ensure_send_message_summary(input: Value) -> Value {
    let mut map = match input {
        Value::Object(m) => m,
        _ => return input,
    };
    if map.contains_key("summary") {
        return Value::Object(map);
    }
    if let Some(msg) = map.get("message").and_then(|v| v.as_str()) {
        let summary = if msg.len() > 50 {
            format!("{}...", &msg[..47])
        } else {
            msg.to_string()
        };
        map.insert("summary".to_string(), Value::String(summary));
    } else {
        map.insert(
            "summary".to_string(),
            Value::String("Agent communication".to_string()),
        );
    }
    Value::Object(map)
}

// ---------------------------------------------------------------------------
// OpenAI Chat → Anthropic response conversion
// ---------------------------------------------------------------------------

/// OpenAI Chat Completions response → Anthropic Messages response.
///
/// `model` is the model echoed back to the Anthropic client (the original model
/// the client requested), not the upstream's reported model.
pub fn openai_to_anthropic(body: Value, model: &str) -> Result<Value, Error> {
    let choices = body
        .get("choices")
        .and_then(|c| c.as_array())
        .ok_or_else(|| Error::Transform("No choices in response".to_string()))?;

    let choice = choices
        .first()
        .ok_or_else(|| Error::Transform("Empty choices array".to_string()))?;

    let message = choice
        .get("message")
        .ok_or_else(|| Error::Transform("No message in choice".to_string()))?;

    let mut content = Vec::new();
    let mut has_tool_use = false;

    // reasoning_content → Anthropic thinking block
    if let Some(reasoning_content) = message.get("reasoning_content").and_then(|r| r.as_str()) {
        if !reasoning_content.is_empty() {
            content.push(json!({"type": "thinking", "thinking": reasoning_content}));
        }
    }

    // Text / refusal content
    if let Some(msg_content) = message.get("content") {
        if let Some(text) = msg_content.as_str() {
            if !text.is_empty() {
                content.push(json!({"type": "text", "text": text}));
            }
        } else if let Some(parts) = msg_content.as_array() {
            for part in parts {
                let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match part_type {
                    "text" | "output_text" => {
                        if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                            if !text.is_empty() {
                                content.push(json!({"type": "text", "text": text}));
                            }
                        }
                    }
                    "refusal" => {
                        if let Some(refusal) = part.get("refusal").and_then(|r| r.as_str()) {
                            if !refusal.is_empty() {
                                content.push(json!({"type": "text", "text": refusal}));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    // Some providers put refusal at message-level
    if let Some(refusal) = message.get("refusal").and_then(|r| r.as_str()) {
        if !refusal.is_empty() {
            content.push(json!({"type": "text", "text": refusal}));
        }
    }

    // Tool calls
    if let Some(tool_calls) = message.get("tool_calls").and_then(|t| t.as_array()) {
        if !tool_calls.is_empty() {
            has_tool_use = true;
        }
        for tc in tool_calls {
            let id = tc.get("id").and_then(|i| i.as_str()).unwrap_or("");
            let empty_obj = json!({});
            let func = tc.get("function").unwrap_or(&empty_obj);
            let name = func.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args_str = func
                .get("arguments")
                .and_then(|a| a.as_str())
                .unwrap_or("{}");
            let input: Value = serde_json::from_str(args_str).unwrap_or(json!({}));
            let input = if name == "SendMessage" {
                ensure_send_message_summary(input)
            } else {
                input
            };

            content.push(json!({
                "type": "tool_use",
                "id": id,
                "name": name,
                "input": input
            }));
        }
    }
    // Legacy function_call
    if !has_tool_use {
        if let Some(function_call) = message.get("function_call") {
            let id = function_call
                .get("id")
                .and_then(|i| i.as_str())
                .unwrap_or("");
            let name = function_call
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("");
            let has_arguments = function_call.get("arguments").is_some();

            let input = match function_call.get("arguments") {
                Some(Value::String(s)) => serde_json::from_str(s).unwrap_or(json!({})),
                Some(v @ Value::Object(_)) | Some(v @ Value::Array(_)) => v.clone(),
                _ => json!({}),
            };
            let input = if name == "SendMessage" {
                ensure_send_message_summary(input)
            } else {
                input
            };

            if !name.is_empty() || has_arguments {
                content.push(json!({
                    "type": "tool_use",
                    "id": id,
                    "name": name,
                    "input": input
                }));
                has_tool_use = true;
            }
        }
    }

    // Map finish_reason → stop_reason
    let stop_reason = choice
        .get("finish_reason")
        .and_then(|r| r.as_str())
        .map(|r| match r {
            "stop" => "end_turn",
            "length" => "max_tokens",
            "tool_calls" | "function_call" => "tool_use",
            "content_filter" => "end_turn",
            other => {
                tracing::warn!("Unknown finish_reason in non-streaming: {other}");
                "end_turn"
            }
        })
        .or(if has_tool_use { Some("tool_use") } else { None });

    // Usage mapping
    let usage = body.get("usage").cloned().unwrap_or(json!({}));
    let cached = usage
        .get("cache_read_input_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            usage
                .pointer("/prompt_tokens_details/cached_tokens")
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0);
    let cache_creation = usage
        .get("cache_creation_input_tokens")
        .and_then(|v| v.as_u64())
        .or_else(|| {
            usage
                .pointer("/prompt_tokens_details/cache_write_tokens")
                .or_else(|| usage.pointer("/input_tokens_details/cache_write_tokens"))
                .and_then(|v| v.as_u64())
        })
        .unwrap_or(0);
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0)
        .saturating_sub(cached)
        .saturating_sub(cache_creation) as u32;
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as u32;

    let mut usage_json = json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens
    });

    if cached > 0 {
        usage_json["cache_read_input_tokens"] = json!(cached);
    }
    if cache_creation > 0 {
        usage_json["cache_creation_input_tokens"] = json!(cache_creation);
    }

    let result = json!({
        "id": body.get("id").and_then(|i| i.as_str()).unwrap_or(""),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": usage_json
    });

    Ok(result)
}

// ---------------------------------------------------------------------------
// Streaming: OpenAI Chat SSE → Anthropic SSE
// ---------------------------------------------------------------------------

/// OpenAI streaming chunk (parsed from SSE `data:` lines).
#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    #[serde(default)]
    id: String,
    #[serde(default)]
    choices: Vec<StreamChoice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    delta: Delta,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct Delta {
    #[serde(default)]
    content: Option<String>,
    // OpenRouter/Kimi use `reasoning`, DeepSeek uses `reasoning_content`
    #[serde(default, alias = "reasoning_content")]
    reasoning: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<DeltaToolCall>>,
    #[serde(default)]
    #[allow(dead_code)]
    role: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DeltaToolCall {
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type", default)]
    call_type: Option<String>,
    #[serde(default)]
    function: Option<DeltaFunction>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DeltaFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

#[derive(Debug, Deserialize, Clone)]
struct Usage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<PromptTokensDetails>,
    /// Some compatible servers return Anthropic-style cache fields directly
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, Clone)]
struct PromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
    #[serde(default)]
    cache_write_tokens: u32,
}

#[derive(Debug, Clone)]
struct ToolBlockState {
    anthropic_index: u32,
    id: String,
    name: String,
    started: bool,
    pending_args: String,
    consecutive_whitespace: usize,
    aborted: bool,
}

/// Convert OpenAI Chat SSE stream to Anthropic SSE stream.
///
/// Wraps the upstream streaming response and translates each SSE chunk from
/// OpenAI Chat Completions format to Anthropic Messages events. `model` is the
/// model echoed back in `message_start` (the original model the client
/// requested), not the upstream's reported model.
pub fn chat_to_anthropic_sse<S, E>(
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
        let mut next_content_index: u32 = 0;
        let mut message_id = String::new();
        let current_model = model;
        let mut has_sent_message_start = false;
        let mut has_emitted_message_delta = false;
        let mut pending_message_delta: Option<(Option<String>, Option<Value>)> = None;
        let mut latest_usage: Option<Value> = None;
        let mut current_non_tool_block_type: Option<&'static str> = None;
        let mut current_non_tool_block_index: Option<u32> = None;
        // Accumulated reasoning text of the current thinking block, used to
        // synthesize a `signature_delta` (the Anthropic protocol requires one
        // before the thinking block's `content_block_stop`).
        let mut current_reasoning_text = String::new();
        // Monotonic id source for synthetic reasoning items in signatures.
        let mut reasoning_sequence: u64 = 0;
        let mut tool_blocks_by_index: std::collections::HashMap<usize, ToolBlockState> = std::collections::HashMap::new();
        let mut open_tool_block_indices: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut has_sent_message_stop = false;
        let mut stream_ended_with_error = false;

        tokio::pin!(stream);

        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    append_utf8_safe(&mut buffer, &mut utf8_remainder, &bytes);

                    while let Some(line) = take_sse_block(&mut buffer) {
                        if line.trim().is_empty() {
                            continue;
                        }

                        for l in line.lines() {
                            if let Some(data) = strip_sse_field(l, "data") {
                                if data.trim() == "[DONE]" {
                                    // Flush pending message_delta
                                    if let Some((stop_reason, usage_json)) =
                                        pending_message_delta.take()
                                    {
                                        let event = build_message_delta_event(stop_reason, usage_json);
                                        let sse_data = format!(
                                            "event: message_delta\ndata: {}\n\n",
                                            serde_json::to_string(&event).unwrap_or_default()
                                        );
                                        yield Ok(Bytes::from(sse_data));
                                    }

                                    let sse_data =
                                        "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n".to_string();
                                    has_sent_message_stop = true;
                                    yield Ok(Bytes::from(sse_data));
                                    continue;
                                }

                                if let Ok(chunk) = serde_json::from_str::<OpenAIStreamChunk>(data) {
                                    if message_id.is_empty() && !chunk.id.is_empty() {
                                        message_id = chunk.id.clone();
                                    }

                                    let chunk_usage_json = chunk.usage.as_ref().map(build_anthropic_usage_json);
                                    if let Some(usage_json) = &chunk_usage_json {
                                        latest_usage = Some(usage_json.clone());
                                        if let Some((_, pending_usage)) = pending_message_delta.as_mut() {
                                            *pending_usage = Some(usage_json.clone());
                                        }
                                    }

                                    if let Some(choice) = chunk.choices.first() {
                                        // message_start
                                        if !has_sent_message_start {
                                            let mut start_usage = json!({
                                                "input_tokens": 0,
                                                "output_tokens": 0
                                            });
                                            if let Some(u) = &chunk.usage {
                                                let cached = extract_cache_read_tokens(u).unwrap_or(0);
                                                let cache_creation = extract_cache_write_tokens(u).unwrap_or(0);
                                                let input = u.prompt_tokens
                                                    .saturating_sub(cached)
                                                    .saturating_sub(cache_creation);
                                                start_usage["input_tokens"] = json!(input);
                                                if cached > 0 {
                                                    start_usage["cache_read_input_tokens"] = json!(cached);
                                                }
                                                if cache_creation > 0 {
                                                    start_usage["cache_creation_input_tokens"] = json!(cache_creation);
                                                }
                                            }

                                            let event = json!({
                                                "type": "message_start",
                                                "message": {
                                                    "id": message_id.clone(),
                                                    "type": "message",
                                                    "role": "assistant",
                                                    "model": current_model.clone(),
                                                    "usage": start_usage
                                                }
                                            });
                                            let sse_data = format!(
                                                "event: message_start\ndata: {}\n\n",
                                                serde_json::to_string(&event).unwrap_or_default()
                                            );
                                            has_sent_message_start = true;
                                            yield Ok(Bytes::from(sse_data));
                                        }

                                        // reasoning (thinking) content. Skip empty
                                        // fragments so we never open a zero-text
                                        // thinking block (which could be dropped by
                                        // the client for lack of a signature).
                                        if let Some(reasoning) = &choice.delta.reasoning {
                                            if !reasoning.is_empty() {
                                                if current_non_tool_block_type != Some("thinking") {
                                                    // Close previous non-tool block
                                                    if let Some(index) = current_non_tool_block_index.take() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": index
                                                        });
                                                        let sse_data = format!(
                                                            "event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default()
                                                        );
                                                        yield Ok(Bytes::from(sse_data));
                                                    }
                                                    let index = next_content_index;
                                                    next_content_index += 1;
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
                                                    current_non_tool_block_type = Some("thinking");
                                                    current_non_tool_block_index = Some(index);
                                                    current_reasoning_text.clear();
                                                    yield Ok(Bytes::from(sse_data));
                                                }

                                                current_reasoning_text.push_str(reasoning);
                                                if let Some(index) = current_non_tool_block_index {
                                                    let event = json!({
                                                        "type": "content_block_delta",
                                                        "index": index,
                                                        "delta": {
                                                            "type": "thinking_delta",
                                                            "thinking": reasoning
                                                        }
                                                    });
                                                    let sse_data = format!(
                                                        "event: content_block_delta\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default()
                                                    );
                                                    reqlog.append(reasoning);
                                                    yield Ok(Bytes::from(sse_data));
                                                }
                                            }
                                        }

                                        // text content
                                        if let Some(content) = &choice.delta.content {
                                            if !content.is_empty() {
                                                if current_non_tool_block_type != Some("text") {
                                                    // Emit the thinking block's
                                                    // signature before closing it.
                                                    if current_non_tool_block_type == Some("thinking") {
                                                        if let Some(index) = current_non_tool_block_index {
                                                            reasoning_sequence += 1;
                                                            let rid = format!("rs_{:04x}", reasoning_sequence);
                                                            if let Some(sse) = build_chat_signature_delta_sse(
                                                                index,
                                                                &rid,
                                                                &current_reasoning_text,
                                                            ) {
                                                                yield Ok(Bytes::from(sse));
                                                            }
                                                            current_reasoning_text.clear();
                                                        }
                                                    }
                                                    if let Some(index) = current_non_tool_block_index.take() {
                                                        let event = json!({
                                                            "type": "content_block_stop",
                                                            "index": index
                                                        });
                                                        let sse_data = format!(
                                                            "event: content_block_stop\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default()
                                                        );
                                                        yield Ok(Bytes::from(sse_data));
                                                    }

                                                    let index = next_content_index;
                                                    next_content_index += 1;
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
                                                    current_non_tool_block_type = Some("text");
                                                    current_non_tool_block_index = Some(index);
                                                    current_reasoning_text.clear();
                                                    yield Ok(Bytes::from(sse_data));
                                                }

                                                if let Some(index) = current_non_tool_block_index {
                                                    let event = json!({
                                                        "type": "content_block_delta",
                                                        "index": index,
                                                        "delta": {
                                                            "type": "text_delta",
                                                            "text": content
                                                        }
                                                    });
                                                    let sse_data = format!(
                                                        "event: content_block_delta\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default()
                                                    );
                                                    resp_has_text = true;
                                                    reqlog.append(content);
                                                    yield Ok(Bytes::from(sse_data));
                                                }
                                            }
                                        }

                                        // tool calls
                                        if let Some(tool_calls) = &choice.delta.tool_calls {
                                            if !tool_calls.is_empty() {
                                                // Emit the thinking block's
                                                // signature before closing it.
                                                if current_non_tool_block_type == Some("thinking") {
                                                    if let Some(index) = current_non_tool_block_index {
                                                        reasoning_sequence += 1;
                                                        let rid = format!("rs_{:04x}", reasoning_sequence);
                                                        if let Some(sse) = build_chat_signature_delta_sse(
                                                            index,
                                                            &rid,
                                                            &current_reasoning_text,
                                                        ) {
                                                            yield Ok(Bytes::from(sse));
                                                        }
                                                        current_reasoning_text.clear();
                                                    }
                                                }
                                                // Close current non-tool block
                                                if let Some(index) = current_non_tool_block_index.take() {
                                                    let event = json!({
                                                        "type": "content_block_stop",
                                                        "index": index
                                                    });
                                                    let sse_data = format!(
                                                        "event: content_block_stop\ndata: {}\n\n",
                                                        serde_json::to_string(&event).unwrap_or_default()
                                                    );
                                                    yield Ok(Bytes::from(sse_data));
                                                }

                                                for tool_call in tool_calls {
                                                    let (anthropic_index, id, name, should_start, pending_after_start, immediate_delta) = {
                                                        let state = tool_blocks_by_index
                                                            .entry(tool_call.index)
                                                            .or_insert_with(|| {
                                                                let index = next_content_index;
                                                                next_content_index += 1;
                                                                ToolBlockState {
                                                                    anthropic_index: index,
                                                                    id: String::new(),
                                                                    name: String::new(),
                                                                    started: false,
                                                                    pending_args: String::new(),
                                                                    consecutive_whitespace: 0,
                                                                    aborted: false,
                                                                }
                                                            });

                                                        if state.aborted {
                                                            continue;
                                                        }

                                                        if let Some(id) = &tool_call.id {
                                                            state.id = id.clone();
                                                        }
                                                        if let Some(function) = &tool_call.function {
                                                            if let Some(name) = &function.name {
                                                                state.name = name.clone();
                                                            }
                                                        }

                                                        let should_start = !state.started
                                                            && !state.id.is_empty()
                                                            && !state.name.is_empty();
                                                        if should_start {
                                                            state.started = true;
                                                        }
                                                        let pending_after_start = if should_start && !state.pending_args.is_empty() {
                                                            Some(std::mem::take(&mut state.pending_args))
                                                        } else {
                                                            None
                                                        };
                                                        let args_delta = tool_call.function.as_ref().and_then(|f| f.arguments.clone());
                                                        let immediate_delta = if let Some(args) = args_delta {
                                                            // Infinite whitespace detection
                                                            for ch in args.chars() {
                                                                if ch.is_whitespace() {
                                                                    state.consecutive_whitespace += 1;
                                                                } else {
                                                                    state.consecutive_whitespace = 0;
                                                                }
                                                            }
                                                            if state.consecutive_whitespace >= INFINITE_WHITESPACE_THRESHOLD {
                                                                tracing::warn!("Detected infinite whitespace bug (tool: {}), aborting", state.name);
                                                                state.aborted = true;
                                                                None
                                                            } else if state.started {
                                                                Some(args)
                                                            } else {
                                                                state.pending_args.push_str(&args);
                                                                None
                                                            }
                                                        } else {
                                                            None
                                                        };
                                                        (state.anthropic_index, state.id.clone(), state.name.clone(), should_start, pending_after_start, immediate_delta)
                                                    };

                                                    if should_start {
                                                        let event = json!({
                                                            "type": "content_block_start",
                                                            "index": anthropic_index,
                                                            "content_block": {
                                                                "type": "tool_use",
                                                                "id": id,
                                                                "name": name
                                                            }
                                                        });
                                                        let sse_data = format!(
                                                            "event: content_block_start\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default()
                                                        );
                                                        open_tool_block_indices.insert(anthropic_index);
                                                        if !resp_has_text {
                                                            reqlog.append(&format!("[tool_use: {name}]"));
                                                        }
                                                        yield Ok(Bytes::from(sse_data));
                                                    }

                                                    if let Some(args) = pending_after_start {
                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": anthropic_index,
                                                            "delta": {
                                                                "type": "input_json_delta",
                                                                "partial_json": args
                                                            }
                                                        });
                                                        let sse_data = format!(
                                                            "event: content_block_delta\ndata: {}\n\n",
                                                            serde_json::to_string(&event).unwrap_or_default()
                                                        );
                                                        yield Ok(Bytes::from(sse_data));
                                                    }

                                                    if let Some(args) = immediate_delta {
                                                        let event = json!({
                                                            "type": "content_block_delta",
                                                            "index": anthropic_index,
                                                            "delta": {
                                                                "type": "input_json_delta",
                                                                "partial_json": args
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
                                        }

                                        // finish_reason
                                        if let Some(finish_reason) = &choice.finish_reason {
                                            let stop_reason = map_stop_reason(Some(finish_reason));
                                            let usage_json = chunk_usage_json.clone().or_else(|| latest_usage.clone());

                                            if has_emitted_message_delta {
                                                if let (Some((_, ref mut usage)), Some(uj)) = (&mut pending_message_delta, usage_json) {
                                                    *usage = Some(uj);
                                                }
                                                continue;
                                            }
                                            has_emitted_message_delta = true;

                                            // Emit the thinking block's signature
                                            // before closing it.
                                            if current_non_tool_block_type == Some("thinking") {
                                                if let Some(index) = current_non_tool_block_index {
                                                    reasoning_sequence += 1;
                                                    let rid = format!("rs_{:04x}", reasoning_sequence);
                                                    if let Some(sse) = build_chat_signature_delta_sse(
                                                        index,
                                                        &rid,
                                                        &current_reasoning_text,
                                                    ) {
                                                        yield Ok(Bytes::from(sse));
                                                    }
                                                    current_reasoning_text.clear();
                                                }
                                            }

                                            // Close current non-tool block
                                            if let Some(index) = current_non_tool_block_index.take() {
                                                let event = json!({
                                                    "type": "content_block_stop",
                                                    "index": index
                                                });
                                                let sse_data = format!(
                                                    "event: content_block_stop\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default()
                                                );
                                                yield Ok(Bytes::from(sse_data));
                                            }

                                            // Late tool starts
                                            let mut late_tool_starts: Vec<(u32, String, String, String)> = Vec::new();
                                            for (tool_idx, state) in tool_blocks_by_index.iter_mut() {
                                                if state.started { continue; }
                                                let has_payload = !state.pending_args.is_empty()
                                                    || !state.id.is_empty()
                                                    || !state.name.is_empty();
                                                if !has_payload { continue; }
                                                let fallback_id = if state.id.is_empty() {
                                                    format!("tool_call_{tool_idx}")
                                                } else {
                                                    state.id.clone()
                                                };
                                                let fallback_name = if state.name.is_empty() {
                                                    "unknown_tool".to_string()
                                                } else {
                                                    state.name.clone()
                                                };
                                                state.started = true;
                                                let pending = std::mem::take(&mut state.pending_args);
                                                late_tool_starts.push((state.anthropic_index, fallback_id, fallback_name, pending));
                                            }
                                            late_tool_starts.sort_unstable_by_key(|(index, _, _, _)| *index);
                                            for (index, id, name, _pending) in late_tool_starts {
                                                let event = json!({
                                                    "type": "content_block_start",
                                                    "index": index,
                                                    "content_block": {
                                                        "type": "tool_use",
                                                        "id": id,
                                                        "name": name
                                                    }
                                                });
                                                let sse_data = format!(
                                                    "event: content_block_start\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default()
                                                );
                                                open_tool_block_indices.insert(index);
                                                if !resp_has_text {
                                                    reqlog.append(&format!("[tool_use: {name}]"));
                                                }
                                                yield Ok(Bytes::from(sse_data));
                                            }

                                            // Close tool blocks
                                            let mut tool_indices: Vec<u32> = open_tool_block_indices.iter().copied().collect();
                                            tool_indices.sort_unstable();
                                            for index in tool_indices {
                                                let event = json!({
                                                    "type": "content_block_stop",
                                                    "index": index
                                                });
                                                let sse_data = format!(
                                                    "event: content_block_stop\ndata: {}\n\n",
                                                    serde_json::to_string(&event).unwrap_or_default()
                                                );
                                                open_tool_block_indices.remove(&index);
                                                yield Ok(Bytes::from(sse_data));
                                            }

                                            // Cache message_delta for [DONE]
                                            pending_message_delta = Some((stop_reason, usage_json));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Stream error: {e}");
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

        // Stream ended naturally — flush pending message_delta and message_stop
        if !stream_ended_with_error {
            // Close any non-tool block left open by an upstream that ended
            // without a finish_reason / [DONE]. Emit the thinking signature
            // first if the leftover block is a thinking block.
            if current_non_tool_block_type == Some("thinking") {
                if let Some(index) = current_non_tool_block_index {
                    reasoning_sequence += 1;
                    let rid = format!("rs_{:04x}", reasoning_sequence);
                    if let Some(sse) = build_chat_signature_delta_sse(
                        index,
                        &rid,
                        &current_reasoning_text,
                    ) {
                        yield Ok(Bytes::from(sse));
                    }
                    current_reasoning_text.clear();
                }
            }
            if let Some(index) = current_non_tool_block_index.take() {
                let event = json!({ "type": "content_block_stop", "index": index });
                let sse_data = format!(
                    "event: content_block_stop\ndata: {}\n\n",
                    serde_json::to_string(&event).unwrap_or_default()
                );
                yield Ok(Bytes::from(sse_data));
            }

            if let Some((stop_reason, usage_json)) = pending_message_delta.take() {
                let event = build_message_delta_event(stop_reason, usage_json);
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

/// Build an SSE `signature_delta` for the chat path's synthetic reasoning item.
///
/// The Anthropic streaming protocol requires a thinking block to end with a
/// single `signature_delta` before its `content_block_stop`. Chat upstreams
/// (e.g. DeepSeek) have no native signature, so we synthesize one carrying the
/// accumulated reasoning text; it round-trips back through
/// [`crate::reasoning_bridge::decode_openai_reasoning_item`] on replay.
/// Returns `None` when there is no reasoning text to sign (mirrors
/// `streaming_responses.rs` which skips the signature for empty summaries).
fn build_chat_signature_delta_sse(
    index: u32,
    reasoning_id: &str,
    reasoning_text: &str,
) -> Option<String> {
    if reasoning_text.is_empty() {
        return None;
    }
    let item = json!({
        "type": "reasoning",
        "id": reasoning_id,
        "summary": [{ "type": "summary_text", "text": reasoning_text }]
    });
    let signature = crate::reasoning_bridge::encode_openai_reasoning_item(&item)?;
    let event = json!({
        "type": "content_block_delta",
        "index": index,
        "delta": { "type": "signature_delta", "signature": signature }
    });
    Some(format!(
        "event: content_block_delta\ndata: {}\n\n",
        serde_json::to_string(&event).unwrap_or_default()
    ))
}

fn build_anthropic_usage_json(usage: &Usage) -> Value {
    let cached = extract_cache_read_tokens(usage).unwrap_or(0);
    let cache_creation = extract_cache_write_tokens(usage).unwrap_or(0);
    let input_tokens = usage
        .prompt_tokens
        .saturating_sub(cached)
        .saturating_sub(cache_creation);
    let mut usage_json = json!({
        "input_tokens": input_tokens,
        "output_tokens": usage.completion_tokens
    });
    if cached > 0 {
        usage_json["cache_read_input_tokens"] = json!(cached);
    }
    if cache_creation > 0 {
        usage_json["cache_creation_input_tokens"] = json!(cache_creation);
    }
    usage_json
}

fn default_anthropic_usage_json() -> Value {
    json!({
        "input_tokens": 0,
        "output_tokens": 0
    })
}

fn build_message_delta_event(stop_reason: Option<String>, usage_json: Option<Value>) -> Value {
    let usage = usage_json
        .filter(|usage| usage.is_object())
        .unwrap_or_else(default_anthropic_usage_json);

    json!({
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": null
        },
        "usage": usage
    })
}

fn extract_cache_read_tokens(usage: &Usage) -> Option<u32> {
    if let Some(v) = usage.cache_read_input_tokens {
        return Some(v);
    }
    usage
        .prompt_tokens_details
        .as_ref()
        .map(|d| d.cached_tokens)
        .filter(|&v| v > 0)
}

fn extract_cache_write_tokens(usage: &Usage) -> Option<u32> {
    if let Some(value) = usage.cache_creation_input_tokens {
        return Some(value);
    }
    usage
        .prompt_tokens_details
        .as_ref()
        .map(|details| details.cache_write_tokens)
        .filter(|value| *value > 0)
}

fn map_stop_reason(finish_reason: Option<&str>) -> Option<String> {
    finish_reason.map(|r| {
        match r {
            "tool_calls" | "function_call" => "tool_use",
            "stop" => "end_turn",
            "length" => "max_tokens",
            "content_filter" => "end_turn",
            other => {
                tracing::warn!("Unknown finish_reason in streaming: {other}");
                "end_turn"
            }
        }
        .to_string()
    })
}

// ---------------------------------------------------------------------------
// SSE parsing helpers
// ---------------------------------------------------------------------------

pub(crate) fn strip_sse_field<'a>(line: &'a str, field: &str) -> Option<&'a str> {
    line.strip_prefix(&format!("{field}: "))
        .or_else(|| line.strip_prefix(&format!("{field}:")))
}

pub(crate) fn take_sse_block(buffer: &mut String) -> Option<String> {
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

pub(crate) fn append_utf8_safe(buffer: &mut String, remainder: &mut Vec<u8>, new_bytes: &[u8]) {
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
                    // Incomplete sequence at end, save as remainder
                    remainder.extend_from_slice(&input[pos..]);
                    break;
                } else {
                    // Invalid byte, skip it with lossy replacement
                    buffer.push(char::REPLACEMENT_CHARACTER);
                    pos += error_len;
                }
            }
        }
    }

    // Defensive guard: remainder should never exceed 3 bytes for valid UTF-8
    if remainder.len() > 3 {
        buffer.push_str(&String::from_utf8_lossy(remainder));
        remainder.clear();
    }
}

// Stream is implemented via `chat_to_anthropic_sse()` above using async_stream::stream!.
// No manual Stream impl needed.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reqlog::ReqLog;
    use bytes::Bytes;
    use futures::StreamExt;
    use serde_json::Value;
    use std::convert::Infallible;

    /// Feed upstream SSE frames into `chat_to_anthropic_sse`, collect output,
    /// and parse into `(event, data)` pairs. Each frame must be a complete SSE
    /// block (`data: {...}\n\n`) since `take_sse_block` frames on `\n\n`.
    async fn run(frames: Vec<&'static str>) -> Vec<(String, Value)> {
        let upstream = futures::stream::iter(
            frames.into_iter().map(|f| Ok::<Bytes, Infallible>(Bytes::from(f))),
        );
        let out = chat_to_anthropic_sse(upstream, "claude-sonnet-5".to_string(), ReqLog::new());
        let chunks: Vec<Bytes> = out.map(|r| r.unwrap()).collect().await;
        let mut text = String::new();
        for c in &chunks {
            text.push_str(&String::from_utf8_lossy(c));
        }
        text.split("\n\n")
            .filter(|b| !b.trim().is_empty())
            .map(|block| {
                let mut ev = String::new();
                let mut data = String::new();
                for line in block.lines() {
                    if let Some(v) = line.strip_prefix("event: ") {
                        ev = v.to_string();
                    } else if let Some(v) = line.strip_prefix("data: ") {
                        data = v.to_string();
                    }
                }
                (ev, serde_json::from_str(&data).unwrap_or(Value::Null))
            })
            .collect()
    }

    fn event_names(events: &[(String, Value)]) -> Vec<&str> {
        events.iter().map(|(e, _)| e.as_str()).collect()
    }

    #[tokio::test]
    async fn reasoning_then_text_emits_signature_delta() {
        let events = run(vec![
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"Let me think\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ])
        .await;

        let names = event_names(&events);
        let expect: Vec<&str> = vec![
            "message_start",
            "content_block_start", // thinking block
            "content_block_delta", // thinking_delta
            "content_block_delta", // signature_delta
            "content_block_stop",  // thinking block closed
            "content_block_start", // text block
            "content_block_delta", // text_delta
            "content_block_stop",  // text block closed on finish_reason
            "message_delta",
            "message_stop",
        ];
        assert_eq!(names, expect);

        // The 4th frame is the signature_delta with the correct prefix.
        assert_eq!(events[3].0, "content_block_delta");
        assert_eq!(events[3].1["delta"]["type"], "signature_delta");
        let sig = events[3].1["delta"]["signature"].as_str().unwrap();
        assert!(sig.starts_with(crate::reasoning_bridge::OPENAI_REASONING_ITEM_PREFIX));
    }

    #[tokio::test]
    async fn plain_text_stream_has_no_signature() {
        let events = run(vec![
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ])
        .await;
        let sigs: Vec<_> = events
            .iter()
            .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "signature_delta")
            .collect();
        assert!(sigs.is_empty());
        assert!(event_names(&events).contains(&"content_block_start"));
    }

    #[tokio::test]
    async fn stream_ends_mid_reasoning_flushes_thinking_block() {
        let events = run(vec![
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"Partial\"}}]}\n\n",
        ])
        .await; // upstream ends without finish_reason / [DONE]

        let names = event_names(&events);
        assert!(names.contains(&"content_block_start"));
        // signature_delta must exist (as content_block_delta), the thinking
        // block's stop must precede message_stop.
        assert!(names.contains(&"content_block_delta"));
        let stop_pos = names.iter().position(|&n| n == "content_block_stop").unwrap();
        let msg_stop_pos = names.iter().position(|&n| n == "message_stop").unwrap();
        assert!(stop_pos < msg_stop_pos);
    }

    #[tokio::test]
    async fn reasoning_then_tool_call_emits_signature_before_tool_block() {
        let events = run(vec![
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"Need tool\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"t_1\",\"type\":\"function\",\"function\":{\"name\":\"f\",\"arguments\":\"{}\"}}]}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        ])
        .await;

        let sig_pos = events
            .iter()
            .position(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "signature_delta")
            .unwrap();
        let stop_pos = events
            .iter()
            .position(|(e, d)| e == "content_block_stop" && d["index"] == 0)
            .unwrap();
        let tool_start_pos = events
            .iter()
            .position(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "tool_use")
            .unwrap();
        assert!(sig_pos < stop_pos && stop_pos < tool_start_pos);
        // Signature emitted exactly once.
        assert_eq!(
            events
                .iter()
                .filter(|(e, d)| e == "content_block_delta" && d["delta"]["type"] == "signature_delta")
                .count(),
            1
        );
    }

    #[tokio::test]
    async fn multiple_reasoning_deltas_accumulate_and_decode() {
        let events = run(vec![
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"Step A \"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"reasoning_content\":\"Step B\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{\"content\":\"Done\"}}]}\n\n",
            "data: {\"id\":\"1\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ])
        .await;

        let sig = events
            .iter()
            .find_map(|(e, d)| {
                (e == "content_block_delta" && d["delta"]["type"] == "signature_delta")
                    .then(|| d["delta"]["signature"].as_str().unwrap().to_string())
            })
            .unwrap();
        let item = crate::reasoning_bridge::decode_openai_reasoning_item(&sig).unwrap();
        assert_eq!(
            crate::reasoning_bridge::reasoning_summary_text(&item),
            "Step A Step B"
        );
    }

    #[test]
    fn replay_decodes_bridge_signature() {
        let item = json!({
            "type": "reasoning",
            "id": "rs_0001",
            "summary": [{"type": "summary_text", "text": "Full reasoning"}]
        });
        let signature = crate::reasoning_bridge::encode_openai_reasoning_item(&item).unwrap();
        let body = json!({
            "model": "deepseek-v4-flash",
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "truncated", "signature": signature},
                    {"type": "text", "text": "answer"},
                    {"type": "tool_use", "id": "t_1", "name": "f", "input": {}}
                ]
            }]
        });
        let out = anthropic_to_openai_with_reasoning_content(body, true).unwrap();
        let msg = out["messages"]
            .as_array()
            .unwrap()
            .iter()
            .find(|m| m["role"] == "assistant")
            .unwrap();
        assert_eq!(msg["reasoning_content"], "Full reasoning");
    }

    #[test]
    fn supports_reasoning_effort_deepseek() {
        assert!(supports_reasoning_effort("deepseek-v4-flash"));
        assert!(supports_reasoning_effort("deepseek-v4-pro"));
    }

    #[test]
    fn resolve_reasoning_effort_disabled_is_none() {
        let body = json!({
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": "hi"}]
        });
        assert_eq!(resolve_reasoning_effort(&body), None);
    }

    #[test]
    fn deepseek_effort_clamp() {
        // adaptive thinking on deepseek → clamped to max (legal DeepSeek enum).
        let body = json!({
            "model": "deepseek-v4-flash",
            "thinking": {"type": "adaptive"},
            "messages": [{"role": "user", "content": "hi"}]
        });
        let out = anthropic_to_openai_with_reasoning_content(body, true).unwrap();
        assert_eq!(out["reasoning_effort"], "max");
    }
}
