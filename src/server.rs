use crate::convert;
use crate::convert_reverse;
use crate::error::{Error, LocalEntry};
use crate::forward::{apply_reasoning_policy, AppState, UpstreamTarget, UpstreamType};
use crate::reqlog::ReqLog;
use crate::responses_reverse;
use crate::transform_responses;
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, Response, StatusCode},
    response::IntoResponse,
    routing::post,
    Json, Router,
};
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::CorsLayer;

/// Collapse newlines/whitespace so prose stays on a single log line (replaces
/// line breaks with spaces, does NOT truncate content).
///
/// Applied to prose fragments only. A tool payload goes through
/// [`crate::reqlog::tool_use_marker`] / [`crate::reqlog::tool_result_marker`]
/// verbatim, line breaks included (ADR-0012).
fn single_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// How one extracted fragment is rendered onto the log line.
#[derive(Clone, Copy, PartialEq, Eq)]
enum TextMode {
    /// Prose: folded onto the single log line.
    Fold,
    /// A tool payload: written as it came, line breaks included (ADR-0012).
    Verbatim,
}

/// Render one prose fragment for the log line.
fn render_text(s: &str, mode: TextMode) -> String {
    match mode {
        TextMode::Fold => single_line(s),
        TextMode::Verbatim => s.to_string(),
    }
}

/// Join extracted fragments, dropping the empty ones so a blank fragment cannot
/// leave a doubled or dangling separator. The mode only picks the separator —
/// folding happens per fragment in [`render_text`].
fn join_fragments(parts: &[String], mode: TextMode) -> String {
    let sep = match mode {
        TextMode::Fold => " ",
        TextMode::Verbatim => "\n",
    };
    parts
        .iter()
        .filter(|p| !p.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(sep)
}

/// Brief summary of an assistant `tool_use` block: name plus the serialized
/// arguments the model sent (no truncation).
fn tool_use_summary(block: &Value) -> String {
    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("?");
    let args = block
        .get("input")
        .map(|input| serde_json::to_string(input).unwrap_or_default())
        .unwrap_or_default();
    crate::reqlog::tool_use_marker(name, &args)
}

/// Brief summary of a `tool_result` block: id plus the actual tool output
/// content (string, or flattened text/tool blocks), no truncation and no
/// folding — the tool's output is written as it came, newlines included.
fn tool_result_summary(block: &Value) -> String {
    let id = block.get("tool_use_id").and_then(|i| i.as_str()).unwrap_or("?");
    // A missing/null payload degrades to the bare marker, the same rule the
    // Chat/Responses path applies in `tool_result_text`.
    let content = match block.get("content") {
        None | Some(Value::Null) => String::new(),
        Some(c) => extract_text_content(Some(c), TextMode::Verbatim),
    };
    crate::reqlog::tool_result_marker(id, &content)
}

/// Extract plain text from an Anthropic content field (String, Array of blocks, or null).
/// Joins multiple text/thinking blocks into the complete text (no truncation).
/// `mode` decides how prose is rendered; tool payloads are always verbatim.
fn extract_text_content(content: Option<&Value>, mode: TextMode) -> String {
    match content {
        Some(Value::String(s)) => render_text(s, mode),
        Some(Value::Array(blocks)) => {
            let texts: Vec<String> = blocks
                .iter()
                .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => b
                        .get("text")
                        .and_then(|t| t.as_str())
                        .map(|t| render_text(t, mode)),
                    Some("thinking") => b
                        .get("thinking")
                        .and_then(|t| t.as_str())
                        .map(|t| render_text(&format!("{} (thinking)", t), mode)),
                    Some("tool_use") => Some(tool_use_summary(b)),
                    Some("tool_result") => Some(tool_result_summary(b)),
                    Some("image") => Some("[image]".to_string()),
                    Some("document") => Some("[document]".to_string()),
                    Some(other) => Some(format!("[{}]", other)),
                    None => Some("[unknown]".to_string()),
                })
                .collect();
            if texts.is_empty() {
                "[empty]".to_string()
            } else {
                join_fragments(&texts, mode)
            }
        }
        Some(Value::Null) | None => "null".to_string(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Extract the user prompt text from an OpenAI Chat message content. `mode`
/// decides how the text parts are rendered (see [`extract_text_content`]).
fn extract_openai_chat_text(content: Option<&Value>, mode: TextMode) -> String {
    match content {
        Some(Value::String(s)) => render_text(s, mode),
        Some(Value::Array(parts)) => {
            let texts: Vec<String> = parts
                .iter()
                .filter_map(|p| {
                    let t = p.get("type").and_then(|v| v.as_str()).unwrap_or("");
                    match t {
                        "text" | "input_text" | "output_text" => p
                            .get("text")
                            .and_then(|v| v.as_str())
                            .map(|s| render_text(s, mode)),
                        "image_url" | "input_image" => Some("[image]".to_string()),
                        _ => None,
                    }
                })
                .collect();
            join_fragments(&texts, mode)
        }
        _ => "(no content)".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Non-streaming response text extraction (appended onto the `[RESP]` line)
// ---------------------------------------------------------------------------
//
// Streaming handlers log response text live from SSE deltas; the non-streaming
// handlers only print a fixed summary (`ok` / `stop_reason=...`). These
// extractors pull the loggable plain text out of each wire format's *final*
// response body so a non-streaming reply (e.g. a safety-judge `{"allow":false}`
// verdict) is visible on the `[RESP #id]` line too. They mirror the streaming
// logger's conventions: text verbatim, hidden reasoning tagged `(thinking)`,
// tool calls rendered as `[tool_use: {name}] {arguments}` (arguments verbatim,
// ADR-0012). Empty responses return `""` so callers keep the bare summary line.

/// Anthropic Messages response → plain text (`content[]` blocks).
fn anthropic_resp_text(body: &Value) -> String {
    let Some(blocks) = body.get("content").and_then(Value::as_array) else {
        return String::new();
    };
    let parts: Vec<String> = blocks
        .iter()
        .filter_map(|b| match b.get("type").and_then(Value::as_str) {
            Some("text") => b
                .get("text")
                .and_then(Value::as_str)
                .map(|t| render_text(t, TextMode::Fold)),
            Some("thinking") => b
                .get("thinking")
                .and_then(Value::as_str)
                .map(|t| render_text(&format!("{t} (thinking)"), TextMode::Fold)),
            Some("tool_use") => Some(tool_use_summary(b)),
            _ => None,
        })
        .collect();
    if parts.is_empty() {
        String::new()
    } else {
        // Prose was folded per fragment; a tool call's arguments keep their own
        // line breaks, so the join must not fold the result again.
        join_fragments(&parts, TextMode::Fold)
    }
}

/// OpenAI Chat response → plain text (`choices[0].message`).
fn chat_resp_text(body: &Value) -> String {
    let Some(message) = body.pointer("/choices/0/message") else {
        return String::new();
    };
    let mut parts: Vec<String> = Vec::new();

    if let Some(rc) = message.get("reasoning_content").and_then(Value::as_str) {
        if !rc.is_empty() {
            parts.push(render_text(&format!("{rc} (thinking)"), TextMode::Fold));
        }
    }
    match message.get("content") {
        Some(Value::String(s)) => {
            if !s.is_empty() {
                parts.push(render_text(s, TextMode::Fold));
            }
        }
        Some(Value::Array(items)) => {
            for p in items {
                let t = p.get("type").and_then(Value::as_str).unwrap_or("");
                match t {
                    "text" | "output_text" | "input_text" => {
                        if let Some(s) = p.get("text").and_then(Value::as_str) {
                            if !s.is_empty() {
                                parts.push(render_text(s, TextMode::Fold));
                            }
                        }
                    }
                    "refusal" => {
                        if let Some(s) = p.get("refusal").and_then(Value::as_str) {
                            if !s.is_empty() {
                                parts.push(render_text(s, TextMode::Fold));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    // Some providers put a refusal at message level (not in content parts).
    if let Some(refusal) = message.get("refusal").and_then(Value::as_str) {
        if !refusal.is_empty() {
            parts.push(render_text(refusal, TextMode::Fold));
        }
    }
    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
        for c in calls {
            // A call the upstream never named still reached the client, so it
            // is recorded as `[tool_use: ?]` rather than hidden.
            let n = c
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("?");
            let args = c
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("");
            parts.push(crate::reqlog::tool_use_marker(n, args));
        }
    }

    if parts.is_empty() {
        String::new()
    } else {
        join_fragments(&parts, TextMode::Fold)
    }
}

/// Compose the non-streaming `[RESP]` line summary: the fixed marker (`ok` /
/// `stop_reason=...`) plus the extracted response text when there is any.
fn resp_summary(base: &str, text: &str) -> String {
    if text.is_empty() {
        base.to_string()
    } else {
        format!("{base} {text}")
    }
}

/// OpenAI Responses response → plain text (`output[]` items).
fn responses_resp_text(body: &Value) -> String {
    let Some(output) = body.get("output").and_then(Value::as_array) else {
        return String::new();
    };
    let mut parts: Vec<String> = Vec::new();
    for item in output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for block in content {
                        match block.get("type").and_then(Value::as_str) {
                            Some("output_text") => {
                                if let Some(s) = block.get("text").and_then(Value::as_str) {
                                    if !s.is_empty() {
                                        parts.push(render_text(s, TextMode::Fold));
                                    }
                                }
                            }
                            Some("refusal") => {
                                if let Some(s) = block.get("refusal").and_then(Value::as_str) {
                                    if !s.is_empty() {
                                        parts.push(render_text(s, TextMode::Fold));
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            Some("function_call") => {
                // A call the upstream never named still reached the client, so
                // it is recorded as `[tool_use: ?]` rather than hidden.
                let n = item.get("name").and_then(Value::as_str).unwrap_or("?");
                let args = item.get("arguments").and_then(Value::as_str).unwrap_or("");
                parts.push(crate::reqlog::tool_use_marker(n, args));
            }
            Some("reasoning") => {
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    for s in summary {
                        if let Some(t) = s.get("text").and_then(Value::as_str) {
                            if !t.is_empty() {
                                parts.push(render_text(&format!("{t} (thinking)"), TextMode::Fold));
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        String::new()
    } else {
        join_fragments(&parts, TextMode::Fold)
    }
}

pub(crate) fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/messages", post(handle_messages))
        .route("/v1/chat/completions", post(handle_chat_completions_local))
        .route("/v1/responses", post(handle_responses_local))
        .layer(CorsLayer::permissive())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(200 * 1024 * 1024)) // 200 MB
        .with_state(state)
}

/// The local API style of the current handler, used to shape error responses.
const ANTHROPIC_ENTRY: LocalEntry = LocalEntry::AnthropicMessages;
const CHAT_ENTRY: LocalEntry = LocalEntry::OaiChat;
const RESPONSES_ENTRY: LocalEntry = LocalEntry::OaiResponses;

/// Authenticate the inbound request against the profile's `auth_key` if configured.
/// Accepts `x-api-key` or `Authorization: Bearer`; exact match required.
fn authenticate(headers: &HeaderMap, expected: Option<&str>) -> Result<(), Error> {
    if let Some(expected) = expected {
        let actual = headers
            .get("x-api-key")
            .and_then(|v| v.to_str().ok())
            .or_else(|| {
                headers
                    .get("authorization")
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.strip_prefix("Bearer "))
            })
            .unwrap_or("");
        if actual != expected {
            return Err(Error::Unauthorized(
                "Invalid or missing authentication token".into(),
            ));
        }
    }
    Ok(())
}

/// Shared reqlog setup: emit the system prompt (deduped) and the `[REQ #id]`
/// line. Returns the ReqLog handle.
fn setup_reqlog(body: &Value, entry: LocalEntry, model: &str) -> Arc<ReqLog> {
    crate::reqlog::report_system_prompt(extract_system_for_entry(body, entry).as_deref());
    let reqlog = crate::reqlog::ReqLog::new();
    let req_text = request_text_for_entry(body, entry);
    let req_summary = format!("{model} {req_text}");
    reqlog.req(&req_summary);
    reqlog
}

/// Extract the system prompt for the given local entry.
fn extract_system_for_entry(body: &Value, entry: LocalEntry) -> Option<String> {
    match entry {
        LocalEntry::AnthropicMessages => {
            let msgs = body.get("messages").and_then(|m| m.as_array());
            extract_system(body, msgs.map(|v| v.as_slice()))
        }
        LocalEntry::OaiChat => {
            let mut parts: Vec<String> = Vec::new();
            if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
                for m in msgs
                    .iter()
                    .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
                {
                    let t = extract_openai_chat_text(m.get("content"), TextMode::Fold);
                    if !t.trim().is_empty() {
                        parts.push(t);
                    }
                }
            }
            (!parts.is_empty()).then(|| parts.join(" | "))
        }
        LocalEntry::OaiResponses => {
            let mut parts: Vec<String> = Vec::new();
            if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
                if !instructions.trim().is_empty() {
                    parts.push(instructions.to_string());
                }
            }
            if let Some(input) = body.get("input").and_then(|i| i.as_array()) {
                for m in input
                    .iter()
                    .filter(|m| m.get("type").and_then(|t| t.as_str()) == Some("message"))
                    .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
                {
                    let t = extract_openai_chat_text(m.get("content"), TextMode::Fold);
                    if !t.trim().is_empty() {
                        parts.push(t);
                    }
                }
            }
            (!parts.is_empty()).then(|| parts.join(" | "))
        }
    }
}

/// Extract the user prompt text for the given local entry, plus any tool results
/// that follow it (see [`trailing_tool_results`]).
fn request_text_for_entry(body: &Value, entry: LocalEntry) -> String {
    match entry {
        LocalEntry::AnthropicMessages => request_text(body),
        LocalEntry::OaiChat => {
            let msgs = body.get("messages").and_then(|m| m.as_array());
            let text = match msgs {
                Some(msgs) => {
                    let last_user = msgs
                        .iter()
                        .rposition(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"));
                    let user = match last_user {
                        Some(i) => join_user_and_results(
                            &extract_openai_chat_text(msgs[i].get("content"), TextMode::Fold),
                            &trailing_tool_results(&msgs[i + 1..], LocalEntry::OaiChat),
                        ),
                        None => {
                            // No user turn at all: fold every non-tool message
                            // into the line and render the tool results with the
                            // usual marker, so they are not spelled two ways.
                            let all: Vec<String> = msgs
                                .iter()
                                .filter(|m| m.get("role").and_then(|r| r.as_str()) != Some("tool"))
                                .map(|m| extract_openai_chat_text(m.get("content"), TextMode::Fold))
                                .collect();
                            join_user_and_results(
                                &all.join(" | "),
                                &trailing_tool_results(msgs, LocalEntry::OaiChat),
                            )
                        }
                    };
                    // The prompt side is already folded; a tool result keeps the
                    // line breaks it arrived with, so the line is not folded again.
                    user
                }
                None => "(no messages)".to_string(),
            };
            if text.trim().is_empty() {
                "(no messages)".to_string()
            } else {
                text
            }
        }
        LocalEntry::OaiResponses => {
            let input = body.get("input").and_then(|i| i.as_array());
            let text = match input {
                Some(input) => {
                    let last_user = input.iter().rposition(|m| {
                        m.get("type").and_then(|t| t.as_str()) == Some("message")
                            && m.get("role").and_then(|r| r.as_str()) == Some("user")
                    });
                    let user = match last_user {
                        Some(i) => join_user_and_results(
                            &extract_openai_chat_text(input[i].get("content"), TextMode::Fold),
                            &trailing_tool_results(&input[i + 1..], LocalEntry::OaiResponses),
                        ),
                        None => {
                            // No user turn at all: the only thing worth showing
                            // is whatever tool results the request carries.
                            join_user_and_results(
                                "",
                                &trailing_tool_results(input, LocalEntry::OaiResponses),
                            )
                        }
                    };
                    // The prompt side is already folded; a tool result keeps the
                    // line breaks it arrived with, so the line is not folded again.
                    user
                }
                None => "(no messages)".to_string(),
            };
            if text.trim().is_empty() {
                "(no messages)".to_string()
            } else {
                text
            }
        }
    }
}

/// Join the user turn with the tool results that follow it. `" | "` is the
/// separator already used when several messages are folded onto one line, and it
/// marks the results as separate messages rather than part of the prompt. An
/// empty side is dropped rather than leaving a dangling separator.
fn join_user_and_results(user: &str, results: &str) -> String {
    match (user.trim().is_empty(), results.is_empty()) {
        (_, true) => user.to_string(),
        (true, false) => results.to_string(),
        (false, false) => format!("{user} | {results}"),
    }
}

/// Tool results carried by the messages *after* the last user message, rendered
/// with the same `[tool_result: {id}] {content}` marker the Anthropic entry
/// already produces for the blocks it finds inline in that message.
///
/// Chat carries them in `role:"tool"` messages and Responses in
/// `function_call_output` items — both are separate messages, so without this
/// they never reach the `[REQ]` line at all. Anthropic needs nothing here: its
/// results are content blocks inside the user message itself.
fn trailing_tool_results(msgs: &[Value], entry: LocalEntry) -> String {
    let results: Vec<String> = msgs
        .iter()
        .filter_map(|m| match entry {
            LocalEntry::OaiChat if m.get("role").and_then(Value::as_str) == Some("tool") => {
                let id = m.get("tool_call_id").and_then(Value::as_str).unwrap_or("?");
                Some(crate::reqlog::tool_result_marker(
                    id,
                    &tool_result_text(m.get("content")),
                ))
            }
            LocalEntry::OaiResponses
                if m.get("type").and_then(Value::as_str) == Some("function_call_output") =>
            {
                let id = m.get("call_id").and_then(Value::as_str).unwrap_or("?");
                Some(crate::reqlog::tool_result_marker(
                    id,
                    &tool_result_text(m.get("output")),
                ))
            }
            _ => None,
        })
        .collect();
    results.join(" ")
}

/// Text of a tool-result payload, written verbatim (line breaks included). A
/// missing/null payload renders as empty so the marker degrades to
/// `[tool_result: {id}]`; a payload that is neither text nor a content array is
/// serialised as JSON rather than described as "no content".
fn tool_result_text(payload: Option<&Value>) -> String {
    match payload {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(v @ Value::Array(_)) => extract_openai_chat_text(Some(v), TextMode::Verbatim),
        Some(v) => serde_json::to_string(v).unwrap_or_default(),
    }
}

/// Media handling before forwarding: when the upstream model is confirmed
/// text-only, either describe images via VISION (if configured) or strip them.
/// Returns the number of images handled.
async fn preprocess_media(
    body: &mut Value,
    entry: LocalEntry,
    state: &AppState,
) -> Result<usize, Error> {
    // Third-party vision supplement disabled → images pass through to the
    // upstream untouched so the upstream's own vision handles them.
    if !state.config.vision_supplement_enabled {
        return Ok(0);
    }
    if !crate::media_sanitizer::is_confirmed_text_only_model(&state.config.model) {
        return Ok(0);
    }
    match &state.config.vision {
        Some(vision) => {
            crate::vision::describe_images_in_body(body, entry, state, vision).await
        }
        None => Ok(crate::media_sanitizer::replace_images_for_text_only_model(
            body, &state.config.model,
        )),
    }
}

/// Whether the local-entry request asks for reasoning at all — any effort /
/// thinking signal other than an explicit disable. Only then does the outbound
/// reasoning policy stamp a `Set` effort value (`apply_reasoning_policy`); an
/// explicitly disabled request never gains an effort field.
fn reasoning_requested_for_entry(body: &Value, entry: LocalEntry) -> bool {
    let not_disabled = |effort: &str| {
        !matches!(
            effort.to_ascii_lowercase().as_str(),
            "none" | "disable" | "disabled"
        )
    };
    match entry {
        LocalEntry::AnthropicMessages => convert::thinking_requested(body),
        LocalEntry::OaiChat => body
            .get("reasoning_effort")
            .and_then(Value::as_str)
            .is_some_and(not_disabled),
        LocalEntry::OaiResponses => body
            .pointer("/reasoning/effort")
            .and_then(Value::as_str)
            .is_some_and(not_disabled),
    }
}

/// Handler for the local Anthropic Messages entry (`POST /v1/messages`).
async fn handle_messages(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Response<Body>, Error> {
    let entry = ANTHROPIC_ENTRY;
    let result: Result<Response<Body>, Error> = async {
    authenticate(&headers, state.config.auth_key.as_deref())?;

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    let media_handled = preprocess_media(&mut body, ANTHROPIC_ENTRY, &state).await?;
    let reqlog = setup_reqlog(&body, ANTHROPIC_ENTRY, &model);
    if media_handled > 0 {
        reqlog.append_media_note(&format!(
            " [media: {media_handled} image(s) → {}]",
            crate::media_sanitizer::UNSUPPORTED_IMAGE_MARKER
        ));
    }

    // Always override the outgoing model with the configured upstream model.
    body["model"] = json!(state.config.model.clone());

    let upstream = UpstreamTarget::from(&state.config);

    match state.config.upstream_type {
        UpstreamType::OaiChat => {
            handle_anthropic_entry_to_chat(state, body, is_stream, &model, &reqlog, upstream).await
        }
        UpstreamType::OaiResponses => {
            handle_anthropic_entry_to_responses(state, body, is_stream, &model, &reqlog, upstream)
                .await
        }
        UpstreamType::AnthropicMessages => {
            handle_passthrough(state, body, is_stream, &model, &reqlog, upstream, ANTHROPIC_ENTRY)
                .await
        }
    }
    }
    .await;
    match result {
        Ok(r) => Ok(r),
        Err(e) => Ok(e.into_entry_response(entry)),
    }
}

/// Handler for the local OpenAI Chat Completions entry (`POST /v1/chat/completions`).
async fn handle_chat_completions_local(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Response<Body>, Error> {
    let entry = CHAT_ENTRY;
    let result: Result<Response<Body>, Error> = async {
    authenticate(&headers, state.config.auth_key.as_deref())?;

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    let media_handled = preprocess_media(&mut body, CHAT_ENTRY, &state).await?;
    let reqlog = setup_reqlog(&body, CHAT_ENTRY, &model);
    if media_handled > 0 {
        reqlog.append_media_note(&format!(
            " [media: {media_handled} image(s) → {}]",
            crate::media_sanitizer::UNSUPPORTED_IMAGE_MARKER
        ));
    }

    body["model"] = json!(state.config.model.clone());

    let upstream = UpstreamTarget::from(&state.config);

    match state.config.upstream_type {
        UpstreamType::OaiChat => {
            handle_passthrough(state, body, is_stream, &model, &reqlog, upstream, CHAT_ENTRY).await
        }
        UpstreamType::OaiResponses => {
            // chat request → responses request, responses response → chat response.
            handle_chat_entry_to_responses(state, body, is_stream, &model, &reqlog, upstream).await
        }
        UpstreamType::AnthropicMessages => {
            handle_chat_entry_to_anthropic(state, body, is_stream, &model, &reqlog, upstream).await
        }
    }
    }
    .await;
    match result {
        Ok(r) => Ok(r),
        Err(e) => Ok(e.into_entry_response(entry)),
    }
}

/// Handler for the local OpenAI Responses entry (`POST /v1/responses`).
async fn handle_responses_local(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(mut body): Json<Value>,
) -> Result<Response<Body>, Error> {
    let entry = RESPONSES_ENTRY;
    let result: Result<Response<Body>, Error> = async {
    authenticate(&headers, state.config.auth_key.as_deref())?;

    let is_stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = body
        .get("model")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();

    let media_handled = preprocess_media(&mut body, RESPONSES_ENTRY, &state).await?;
    let reqlog = setup_reqlog(&body, RESPONSES_ENTRY, &model);
    if media_handled > 0 {
        reqlog.append_media_note(&format!(
            " [media: {media_handled} image(s) → {}]",
            crate::media_sanitizer::UNSUPPORTED_IMAGE_MARKER
        ));
    }

    body["model"] = json!(state.config.model.clone());

    let upstream = UpstreamTarget::from(&state.config);

    match state.config.upstream_type {
        UpstreamType::OaiResponses => {
            handle_passthrough(state, body, is_stream, &model, &reqlog, upstream, RESPONSES_ENTRY)
                .await
        }
        UpstreamType::OaiChat => {
            // responses request → chat request, chat response → responses response.
            handle_responses_entry_to_chat(state, body, is_stream, &model, &reqlog, upstream).await
        }
        UpstreamType::AnthropicMessages => {
            handle_responses_entry_to_anthropic(state, body, is_stream, &model, &reqlog, upstream)
                .await
        }
    }
    }
    .await;
    match result {
        Ok(r) => Ok(r),
        Err(e) => Ok(e.into_entry_response(entry)),
    }
}

// ---------------------------------------------------------------------------
// Entry × upstream pipelines
// ---------------------------------------------------------------------------

/// Forward when the local entry and upstream use the same wire format. Applies
/// only model override (already done) and stream pass-through; no conversion.
async fn handle_passthrough(
    state: Arc<AppState>,
    mut body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
    entry: LocalEntry,
) -> Result<Response<Body>, Error> {
    let wants_reasoning = reasoning_requested_for_entry(&body, entry);
    apply_reasoning_policy(
        &mut body,
        upstream.upstream_type,
        &state.config.reasoning_policy,
        wants_reasoning,
    );
    if is_stream {
        let upstream_resp = match crate::forward::forward_to_zen_streaming(
            body,
            &state.client,
            &upstream,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Ok(handle_forward_error(reqlog, e, entry)),
        };
        let stream = upstream_resp.bytes_stream();
        // Log the streamed response text onto the `[RESP]` line while forwarding
        // the bytes verbatim (the local entry and upstream share a wire format).
        let stream = crate::passthrough_log::log_stream(
            crate::passthrough_log::StreamFormat::from_local_entry(entry),
            stream,
            reqlog.clone(),
        );
        Ok(sse_response(Body::from_stream(stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(handle_forward_error(reqlog, e, entry)),
        };
        let text = match entry {
            LocalEntry::AnthropicMessages => anthropic_resp_text(&upstream_resp),
            LocalEntry::OaiChat => chat_resp_text(&upstream_resp),
            LocalEntry::OaiResponses => responses_resp_text(&upstream_resp),
        };
        reqlog.resp(&resp_summary(
            &format!("stop_reason=passthrough ({model})"),
            &text,
        ));
        Ok(Json(upstream_resp).into_response())
    }
}

/// Anthropic entry → OpenAI Chat upstream (existing behavior).
async fn handle_anthropic_entry_to_chat(
    state: Arc<AppState>,
    body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
) -> Result<Response<Body>, Error> {
    let wants_reasoning = reasoning_requested_for_entry(&body, ANTHROPIC_ENTRY);
    let mut chat_body = convert::anthropic_to_openai_with_reasoning_content(body, true)?;
    apply_reasoning_policy(
        &mut chat_body,
        UpstreamType::OaiChat,
        &state.config.reasoning_policy,
        wants_reasoning,
    );
    convert::inject_openai_stream_include_usage(&mut chat_body);

    if is_stream {
        let upstream_resp = match crate::forward::forward_to_zen_streaming(
            chat_body,
            &state.client,
            &upstream,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Ok(handle_forward_error(reqlog, e, ANTHROPIC_ENTRY)),
        };
        let stream = upstream_resp.bytes_stream();
        let anthropic_stream = convert::chat_to_anthropic_sse(stream, model.to_string(), reqlog.clone());
        Ok(sse_response(Body::from_stream(anthropic_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            chat_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(handle_forward_error(reqlog, e, ANTHROPIC_ENTRY)),
        };
        let anthropic_response = match convert::openai_to_anthropic(upstream_resp, model) {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
        };
        let stop_reason = anthropic_response
            .get("stop_reason")
            .and_then(|r| r.as_str())
            .unwrap_or("?");
        reqlog.resp(&resp_summary(
            &format!("stop_reason={stop_reason}"),
            &anthropic_resp_text(&anthropic_response),
        ));
        Ok(Json(anthropic_response).into_response())
    }
}

/// Anthropic entry → OpenAI Responses upstream (existing behavior).
async fn handle_anthropic_entry_to_responses(
    state: Arc<AppState>,
    body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
) -> Result<Response<Body>, Error> {
    let wants_reasoning = reasoning_requested_for_entry(&body, ANTHROPIC_ENTRY);
    let mut responses_body = transform_responses::anthropic_to_responses(body)?;
    apply_reasoning_policy(
        &mut responses_body,
        UpstreamType::OaiResponses,
        &state.config.reasoning_policy,
        wants_reasoning,
    );

    if is_stream {
        responses_body["stream"] = json!(true);
        let upstream_resp = match crate::forward::forward_to_responses_streaming(
            responses_body,
            &state.client,
            &upstream,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Ok(handle_forward_error(reqlog, e, ANTHROPIC_ENTRY)),
        };
        let stream = upstream_resp.bytes_stream();
        let anthropic_stream = crate::streaming_responses::responses_to_anthropic_sse(
            stream,
            model.to_string(),
            reqlog.clone(),
        );
        Ok(sse_response(Body::from_stream(anthropic_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_responses(
            responses_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(handle_forward_error(reqlog, e, ANTHROPIC_ENTRY)),
        };
        let anthropic_response = match transform_responses::responses_to_anthropic(
            upstream_resp, model,
        ) {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
        };
        let stop_reason = anthropic_response
            .get("stop_reason")
            .and_then(|r| r.as_str())
            .unwrap_or("?");
        reqlog.resp(&resp_summary(
            &format!("stop_reason={stop_reason}"),
            &anthropic_resp_text(&anthropic_response),
        ));
        Ok(Json(anthropic_response).into_response())
    }
}

/// OpenAI Chat entry → Anthropic upstream (reverse conversion).
async fn handle_chat_entry_to_anthropic(
    state: Arc<AppState>,
    body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
) -> Result<Response<Body>, Error> {
    let wants_reasoning = reasoning_requested_for_entry(&body, CHAT_ENTRY);
    let mut anthropic_body = match convert_reverse::chat_to_anthropic_request(&body) {
        Ok(v) => v,
        Err(e) => {
            reqlog.err_req(&e.to_string());
            return Err(e);
        }
    };
    apply_reasoning_policy(
        &mut anthropic_body,
        UpstreamType::AnthropicMessages,
        &state.config.reasoning_policy,
        wants_reasoning,
    );

    if is_stream {
        let upstream_resp = match crate::forward::forward_to_zen_streaming(
            anthropic_body,
            &state.client,
            &upstream,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Ok(handle_forward_error(reqlog, e, CHAT_ENTRY)),
        };
        let stream = upstream_resp.bytes_stream();
        let chat_stream = convert_reverse::anthropic_to_chat_sse(
            stream,
            model.to_string(),
            reqlog.clone(),
            true, // standalone: this stream is the sole response-text logger
        );
        Ok(sse_response(Body::from_stream(chat_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            anthropic_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(handle_forward_error(reqlog, e, CHAT_ENTRY)),
        };
        let chat_response = match convert_reverse::anthropic_to_chat_response(&upstream_resp, model) {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
        };
        reqlog.resp(&resp_summary("ok", &chat_resp_text(&chat_response)));
        Ok(Json(chat_response).into_response())
    }
}

/// OpenAI Chat entry → OpenAI Responses upstream.
async fn handle_chat_entry_to_responses(
    state: Arc<AppState>,
    body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
) -> Result<Response<Body>, Error> {
    // Convert chat request → responses request by reusing the anthropic
    // intermediate: chat → anthropic → responses.
    let anthropic_mid = match convert_reverse::chat_to_anthropic_request(&body) {
        Ok(v) => v,
        Err(e) => {
            reqlog.err_req(&e.to_string());
            return Err(e);
        }
    };
    let wants_reasoning = reasoning_requested_for_entry(&body, CHAT_ENTRY);
    let mut responses_body = transform_responses::anthropic_to_responses(anthropic_mid)?;
    apply_reasoning_policy(
        &mut responses_body,
        UpstreamType::OaiResponses,
        &state.config.reasoning_policy,
        wants_reasoning,
    );

    if is_stream {
        responses_body["stream"] = json!(true);
        let upstream_resp = match crate::forward::forward_to_responses_streaming(
            responses_body,
            &state.client,
            &upstream,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Ok(handle_forward_error(reqlog, e, CHAT_ENTRY)),
        };
        let stream = upstream_resp.bytes_stream();
        // responses SSE → anthropic SSE → chat SSE (double bridge).
        let anthropic_stream =
            crate::streaming_responses::responses_to_anthropic_sse(stream, model.to_string(), reqlog.clone());
        let chat_stream = convert_reverse::anthropic_to_chat_sse(
            anthropic_stream,
            model.to_string(),
            reqlog.clone(),
            false, // outer wrapper: inner responses_to_anthropic_sse already logs
        );
        Ok(sse_response(Body::from_stream(chat_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_responses(
            responses_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(handle_forward_error(reqlog, e, CHAT_ENTRY)),
        };
        let anthropic_mid =
            match transform_responses::responses_to_anthropic(upstream_resp, model) {
                Ok(v) => v,
                Err(e) => {
                    reqlog.err_req(&e.to_string());
                    return Err(e);
                }
            };
        let chat_response = match convert_reverse::anthropic_to_chat_response(&anthropic_mid, model) {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
        };
        reqlog.resp(&resp_summary("ok", &chat_resp_text(&chat_response)));
        Ok(Json(chat_response).into_response())
    }
}

/// OpenAI Responses entry → Anthropic upstream (reverse conversion).
async fn handle_responses_entry_to_anthropic(
    state: Arc<AppState>,
    body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
) -> Result<Response<Body>, Error> {
    let wants_reasoning = reasoning_requested_for_entry(&body, RESPONSES_ENTRY);
    let mut anthropic_body = match responses_reverse::responses_to_anthropic_request(&body) {
        Ok(v) => v,
        Err(e) => {
            reqlog.err_req(&e.to_string());
            return Err(e);
        }
    };
    apply_reasoning_policy(
        &mut anthropic_body,
        UpstreamType::AnthropicMessages,
        &state.config.reasoning_policy,
        wants_reasoning,
    );

    if is_stream {
        let upstream_resp = match crate::forward::forward_to_zen_streaming(
            anthropic_body,
            &state.client,
            &upstream,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Ok(handle_forward_error(reqlog, e, RESPONSES_ENTRY)),
        };
        let stream = upstream_resp.bytes_stream();
        let responses_stream = responses_reverse::anthropic_to_responses_sse(
            stream,
            model.to_string(),
            reqlog.clone(),
            true, // standalone: this stream is the sole response-text logger
        );
        Ok(sse_response(Body::from_stream(responses_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            anthropic_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(handle_forward_error(reqlog, e, RESPONSES_ENTRY)),
        };
        let responses_response = match responses_reverse::anthropic_to_responses_response(
            &upstream_resp, model,
        ) {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
        };
        reqlog.resp(&resp_summary("ok", &responses_resp_text(&responses_response)));
        Ok(Json(responses_response).into_response())
    }
}

/// OpenAI Responses entry → OpenAI Chat upstream.
async fn handle_responses_entry_to_chat(
    state: Arc<AppState>,
    body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
) -> Result<Response<Body>, Error> {
    // responses request → anthropic → chat request (double bridge).
    let anthropic_mid = match responses_reverse::responses_to_anthropic_request(&body) {
        Ok(v) => v,
        Err(e) => {
            reqlog.err_req(&e.to_string());
            return Err(e);
        }
    };
    let wants_reasoning = reasoning_requested_for_entry(&body, RESPONSES_ENTRY);
    let mut chat_body = convert::anthropic_to_openai_with_reasoning_content(anthropic_mid, true)?;
    apply_reasoning_policy(
        &mut chat_body,
        UpstreamType::OaiChat,
        &state.config.reasoning_policy,
        wants_reasoning,
    );
    convert::inject_openai_stream_include_usage(&mut chat_body);

    if is_stream {
        let upstream_resp = match crate::forward::forward_to_zen_streaming(
            chat_body,
            &state.client,
            &upstream,
        )
        .await
        {
            Ok(r) => r,
            Err(e) => return Ok(handle_forward_error(reqlog, e, RESPONSES_ENTRY)),
        };
        let stream = upstream_resp.bytes_stream();
        // chat SSE → anthropic SSE → responses SSE.
        let anthropic_stream =
            convert::chat_to_anthropic_sse(stream, model.to_string(), reqlog.clone());
        let responses_stream = responses_reverse::anthropic_to_responses_sse(
            anthropic_stream,
            model.to_string(),
            reqlog.clone(),
            false, // outer wrapper: inner chat_to_anthropic_sse already logs
        );
        Ok(sse_response(Body::from_stream(responses_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            chat_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => return Ok(handle_forward_error(reqlog, e, RESPONSES_ENTRY)),
        };
        let anthropic_mid = match convert::openai_to_anthropic(upstream_resp, model) {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
        };
        let responses_response =
            match responses_reverse::anthropic_to_responses_response(&anthropic_mid, model) {
                Ok(v) => v,
                Err(e) => {
                    reqlog.err_req(&e.to_string());
                    return Err(e);
                }
            };
        reqlog.resp(&resp_summary("ok", &responses_resp_text(&responses_response)));
        Ok(Json(responses_response).into_response())
    }
}

/// Build a `text/event-stream` response from a byte stream.
fn sse_response(stream: Body) -> Response<Body> {
    let mut headers = axum::http::HeaderMap::new();
    headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
    headers.insert("Cache-Control", HeaderValue::from_static("no-cache"));
    (headers, stream).into_response()
}

/// Wrap an upstream error response into the local entry's error format.
fn upstream_error_response(status: StatusCode, body: Value, entry: LocalEntry) -> Response<Body> {
    let message = body
        .get("error")
        .and_then(|e| e.get("message"))
        .and_then(|m| m.as_str())
        .unwrap_or("Upstream request failed");

    match entry {
        LocalEntry::AnthropicMessages => {
            let anthropic_error = json!({
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": message
                }
            });
            (status, Json(anthropic_error)).into_response()
        }
        LocalEntry::OaiChat | LocalEntry::OaiResponses => {
            let openai_error = json!({
                "error": {
                    "message": message,
                    "type": "upstream_error"
                }
            });
            (status, Json(openai_error)).into_response()
        }
    }
}

/// Log an upstream/forward error on the request line and shape the client
/// response. [`Error::Upstream`] (a non-2xx upstream reply) is logged as
/// `upstream status {code}: {reason}` and relayed with the real HTTP status;
/// any other error keeps the existing `e.to_string()` log and maps through
/// `into_entry_response`.
fn handle_forward_error(reqlog: &Arc<ReqLog>, e: Error, entry: LocalEntry) -> Response<Body> {
    match e {
        Error::Upstream { status, reason, body } => {
            reqlog.err_req(&format!("upstream status {}: {reason}", status.as_u16()));
            upstream_error_response(status, body, entry)
        }
        other => {
            reqlog.err_req(&other.to_string());
            other.into_entry_response(entry)
        }
    }
}

/// Extract the Anthropic system prompt from either the top-level `system` field
/// or `role == "system"` messages (joined if both present).
fn extract_system(body: &Value, msgs: Option<&[Value]>) -> Option<String> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(top) = body
        .get("system")
        .map(|s| extract_text_content(Some(s), TextMode::Fold))
    {
        if !top.trim().is_empty() {
            parts.push(top);
        }
    }
    if let Some(msgs) = msgs {
        for m in msgs
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        {
            let t = extract_text_content(m.get("content"), TextMode::Fold);
            if !t.trim().is_empty() {
                parts.push(t);
            }
        }
    }
    (!parts.is_empty()).then(|| parts.join(" | "))
}

/// Build the `[REQ]` text (after model): the complete last user message's text.
/// Falls back to joining all messages' text when there is no user message. The
/// system prompt is handled separately via `report_system_prompt`.
///
/// Prose is folded per fragment; a `tool_result` block in the message keeps the
/// line breaks of the tool's output, so the assembled line is not folded again.
fn request_text(body: &Value) -> String {
    let msgs = body.get("messages").and_then(|m| m.as_array());
    match msgs {
        Some(msgs) => {
            let user = msgs
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
                .map(|m| extract_text_content(m.get("content"), TextMode::Fold))
                .unwrap_or_else(|| {
                    let all: Vec<String> = msgs
                        .iter()
                        .map(|m| extract_text_content(m.get("content"), TextMode::Fold))
                        .collect();
                    all.join(" | ")
                });
            if user.trim().is_empty() {
                "(no messages)".to_string()
            } else {
                user
            }
        }
        None => "(no messages)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use serde_json::{json, Value};

    fn header(map: &mut HeaderMap, name: &'static str, value: &'static str) {
        map.insert(name, HeaderValue::from_static(value));
    }

    // -----------------------------------------------------------------------
    // Non-streaming response text extractors
    // -----------------------------------------------------------------------

    #[test]
    fn anthropic_resp_text_joins_text_thinking_and_tool() {
        let body = json!({
            "content": [
                {"type": "text", "text": "Verdict"},
                {"type": "thinking", "thinking": "because risky"},
                {"type": "tool_use", "name": "bash", "input": {"command": "rm -rf /"}},
            ]
        });
        assert_eq!(
            anthropic_resp_text(&body),
            "Verdict because risky (thinking) [tool_use: bash] {\"command\":\"rm -rf /\"}"
        );
    }

    #[test]
    fn anthropic_resp_text_empty_or_no_content_is_empty() {
        assert_eq!(anthropic_resp_text(&json!({"content": []})), "");
        assert_eq!(anthropic_resp_text(&json!({"stop_reason": "end_turn"})), "");
        assert_eq!(anthropic_resp_text(&json!({"content": [{"type": "image", "source": {}}]})), "");
    }

    #[test]
    fn anthropic_resp_text_collapses_newlines_to_one_line() {
        let body = json!({"content": [{"type": "text", "text": "line1\nline2"}]});
        assert_eq!(anthropic_resp_text(&body), "line1 line2");
    }

    #[test]
    fn chat_resp_text_keeps_tool_arguments_on_their_own_lines() {
        // Prose is folded, but the argument string the model produced is written
        // as it came — a multi-line payload must not be collapsed.
        let body = json!({
            "choices": [{
                "message": {
                    "content": "line1\nline2",
                    "tool_calls": [{
                        "function": {"name": "bash", "arguments": "{\n\"cmd\": \"ls\"\n}"}
                    }]
                }
            }]
        });
        assert_eq!(
            chat_resp_text(&body),
            "line1 line2 [tool_use: bash] {\n\"cmd\": \"ls\"\n}"
        );
    }

    #[test]
    fn responses_resp_text_keeps_tool_arguments_on_their_own_lines() {
        let body = json!({
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "a\nb"}]},
                {"type": "function_call", "name": "bash", "arguments": "{\n\"x\": 1\n}"}
            ]
        });
        assert_eq!(
            responses_resp_text(&body),
            "a b [tool_use: bash] {\n\"x\": 1\n}"
        );
    }

    #[test]
    fn chat_resp_text_string_content_and_reasoning() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": "hello",
                    "reasoning_content": "let me think"
                }
            }]
        });
        assert_eq!(
            chat_resp_text(&body),
            "let me think (thinking) hello"
        );
    }

    #[test]
    fn chat_resp_text_array_parts_and_refusal() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": [
                        {"type": "text", "text": "a"},
                        {"type": "refusal", "refusal": "cannot do that"}
                    ],
                    "refusal": "cannot do that"
                }
            }]
        });
        // message-level refusal is also folded in (dedup not required, but
        // it must appear).
        let text = chat_resp_text(&body);
        assert!(text.contains("a"));
        assert!(text.contains("cannot do that"));
    }

    #[test]
    fn chat_resp_text_renders_an_unnamed_tool_call_as_a_question_mark() {
        // The client received a tool call, so the log records it even when the
        // upstream never spelled the name.
        let body = json!({
            "choices": [{"message": {"content": null, "tool_calls": [
                {"function": {"arguments": "{}"}},
                {"function": {"name": "", "arguments": "{}"}}
            ]}}]
        });
        assert_eq!(chat_resp_text(&body), "[tool_use: ?] {} [tool_use: ?] {}");
    }

    #[test]
    fn anthropic_resp_text_drops_a_blank_fragment_instead_of_a_stray_separator() {
        // An empty prose block must not leave a doubled or leading space now
        // that the line is joined per fragment instead of folded as a whole.
        let body = json!({
            "content": [
                {"type": "text", "text": ""},
                {"type": "tool_use", "name": "bash", "input": {}},
                {"type": "text", "text": "tail"},
            ]
        });
        assert_eq!(anthropic_resp_text(&body), "[tool_use: bash] {} tail");
    }

    #[test]
    fn chat_resp_text_tool_calls_marker() {
        let body = json!({
            "choices": [{
                "message": {
                    "content": null,
                    "tool_calls": [
                        {"function": {"name": "bash", "arguments": "{\"command\":\"ls\"}"}},
                        {"function": {"name": "read_file", "arguments": "{}"}},
                    ]
                }
            }]
        });
        assert_eq!(
            chat_resp_text(&body),
            "[tool_use: bash] {\"command\":\"ls\"} [tool_use: read_file] {}"
        );
    }

    #[test]
    fn chat_resp_text_tool_call_without_arguments_keeps_bare_marker() {
        let body = json!({
            "choices": [{"message": {"content": null,
                "tool_calls": [{"function": {"name": "bash"}}]}}]
        });
        assert_eq!(chat_resp_text(&body), "[tool_use: bash]");
    }

    #[test]
    fn chat_resp_text_empty_is_empty() {
        assert_eq!(
            chat_resp_text(&json!({"choices": [{"message": {"content": null}}]})),
            ""
        );
        assert_eq!(chat_resp_text(&json!({})), "");
    }

    #[test]
    fn responses_resp_text_message_and_function_call() {
        let body = json!({
            "output": [
                {"type": "message", "role": "assistant", "content": [
                    {"type": "output_text", "text": "answer"}
                ]},
                {"type": "function_call", "name": "get_weather", "arguments": "{}"},
                {"type": "reasoning", "summary": [
                    {"type": "summary_text", "text": "reasoning here"}
                ]}
            ]
        });
        assert_eq!(
            responses_resp_text(&body),
            "answer [tool_use: get_weather] {} reasoning here (thinking)"
        );
    }

    #[test]
    fn responses_resp_text_renders_an_unnamed_function_call_as_a_question_mark() {
        let body = json!({
            "output": [{"type": "function_call", "arguments": "{}"}]
        });
        assert_eq!(responses_resp_text(&body), "[tool_use: ?] {}");
    }

    #[test]
    fn responses_resp_text_empty_is_empty() {
        assert_eq!(responses_resp_text(&json!({"output": []})), "");
        assert_eq!(responses_resp_text(&json!({})), "");
    }

    #[test]
    fn resp_summary_appends_text_only_when_present() {
        assert_eq!(resp_summary("ok", ""), "ok");
        assert_eq!(resp_summary("ok", "hello"), "ok hello");
        assert_eq!(resp_summary("stop_reason=end_turn", "hi"), "stop_reason=end_turn hi");
    }

    // -----------------------------------------------------------------------
    // authenticate
    // -----------------------------------------------------------------------

    #[test]
    fn authenticate_none_expected_ok() {
        let headers = HeaderMap::new();
        assert!(authenticate(&headers, None).is_ok());
    }

    #[test]
    fn authenticate_x_api_key_match_ok() {
        let mut headers = HeaderMap::new();
        header(&mut headers, "x-api-key", "secret");
        assert!(authenticate(&headers, Some("secret")).is_ok());
    }

    #[test]
    fn authenticate_x_api_key_wrong_err() {
        let mut headers = HeaderMap::new();
        header(&mut headers, "x-api-key", "nope");
        assert!(matches!(
            authenticate(&headers, Some("secret")),
            Err(Error::Unauthorized(_))
        ));
    }

    #[test]
    fn authenticate_bearer_match_ok() {
        let mut headers = HeaderMap::new();
        header(&mut headers, "authorization", "Bearer secret");
        assert!(authenticate(&headers, Some("secret")).is_ok());
    }

    #[test]
    fn authenticate_bearer_wrong_err() {
        let mut headers = HeaderMap::new();
        header(&mut headers, "authorization", "Bearer nope");
        assert!(matches!(
            authenticate(&headers, Some("secret")),
            Err(Error::Unauthorized(_))
        ));
    }

    #[test]
    fn authenticate_missing_headers_err() {
        let headers = HeaderMap::new();
        assert!(matches!(
            authenticate(&headers, Some("secret")),
            Err(Error::Unauthorized(_))
        ));
    }

    #[test]
    fn authenticate_x_api_key_takes_precedence_over_bearer() {
        // `x-api-key` wins via `or_else`, so a wrong x-api-key still fails even
        // when the Bearer token is correct.
        let mut headers = HeaderMap::new();
        header(&mut headers, "x-api-key", "wrong");
        header(&mut headers, "authorization", "Bearer secret");
        assert!(matches!(
            authenticate(&headers, Some("secret")),
            Err(Error::Unauthorized(_))
        ));
    }

    // -----------------------------------------------------------------------
    // Text utilities
    // -----------------------------------------------------------------------

    #[test]
    fn single_line_collapses_whitespace() {
        assert_eq!(single_line("a  b\nc\td"), "a b c d");
        assert_eq!(single_line(""), "");
        assert_eq!(single_line("  leading and trailing  "), "leading and trailing");
        assert_eq!(single_line("already"), "already");
    }

    #[test]
    fn tool_use_summary_variants() {
        assert_eq!(tool_use_summary(&json!({"name": "f", "input": {}})), "[tool_use: f] {}");
        assert_eq!(tool_use_summary(&json!({"name": "f"})), "[tool_use: f]");
        assert_eq!(
            tool_use_summary(&json!({"name": "f", "input": {"a": 1}})),
            "[tool_use: f] {\"a\":1}"
        );
        assert_eq!(tool_use_summary(&json!({"input": {}})), "[tool_use: ?] {}");
        // Empty-string input serializes to `""` (non-empty), so it is appended.
        assert_eq!(tool_use_summary(&json!({"name": "f", "input": ""})), "[tool_use: f] \"\"");
    }

    #[test]
    fn tool_result_summary_variants() {
        assert_eq!(
            tool_result_summary(&json!({"tool_use_id": "t_1", "content": "out"})),
            "[tool_result: t_1] out"
        );
        assert_eq!(
            tool_result_summary(&json!({"tool_use_id": "t_1", "content": ""})),
            "[tool_result: t_1]"
        );
        // Array content goes through extract_text_content; empty array -> `[empty]`.
        assert_eq!(
            tool_result_summary(&json!({"tool_use_id": "t_1", "content": []})),
            "[tool_result: t_1] [empty]"
        );
        assert_eq!(
            tool_result_summary(&json!({"content": "x"})),
            "[tool_result: ?] x"
        );
        // Non-string/array content degrades to the bare marker, not a placeholder.
        assert_eq!(
            tool_result_summary(&json!({"tool_use_id": "t_1", "content": null})),
            "[tool_result: t_1]"
        );
        // A tool's output is written as it came, line breaks included.
        assert_eq!(
            tool_result_summary(&json!({"tool_use_id": "t_1", "content": "a\nb"})),
            "[tool_result: t_1] a\nb"
        );
        assert_eq!(
            tool_result_summary(&json!({
                "tool_use_id": "t_1",
                "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
            })),
            "[tool_result: t_1] a\nb"
        );
    }

    #[test]
    fn extract_text_content_variants() {
        assert_eq!(
            extract_text_content(Some(&json!("hello")), TextMode::Fold),
            "hello"
        );

        let blocks = json!([
            {"type": "text", "text": "a"},
            {"type": "thinking", "thinking": "b"},
            {"type": "image"},
            {"type": "document"},
            {"type": "tool_use", "name": "f", "input": {"x": 1}},
            {"type": "tool_result", "tool_use_id": "t", "content": "out"}
        ]);
        assert_eq!(
            extract_text_content(Some(&blocks), TextMode::Fold),
            "a b (thinking) [image] [document] [tool_use: f] {\"x\":1} [tool_result: t] out"
        );

        assert_eq!(
            extract_text_content(Some(&json!([])), TextMode::Fold),
            "[empty]"
        );
        assert_eq!(extract_text_content(None, TextMode::Fold), "null");
        assert_eq!(
            extract_text_content(Some(&Value::Null), TextMode::Fold),
            "null"
        );
        assert_eq!(extract_text_content(Some(&json!(42)), TextMode::Fold), "42");
    }

    #[test]
    fn extract_openai_chat_text_variants() {
        assert_eq!(
            extract_openai_chat_text(Some(&json!("hi")), TextMode::Fold),
            "hi"
        );

        let parts = json!([
            {"type": "text", "text": "a"},
            {"type": "input_text", "text": "b"},
            {"type": "output_text", "text": "c"},
            {"type": "image_url"},
            {"type": "input_image"},
            {"type": "other", "text": "ignored"}
        ]);
        assert_eq!(
            extract_openai_chat_text(Some(&parts), TextMode::Fold),
            "a b c [image] [image]"
        );

        assert_eq!(
            extract_openai_chat_text(None, TextMode::Fold),
            "(no content)"
        );
        assert_eq!(
            extract_openai_chat_text(Some(&Value::Null), TextMode::Fold),
            "(no content)"
        );
        assert_eq!(
            extract_openai_chat_text(Some(&json!([])), TextMode::Fold),
            ""
        );
    }

    // -----------------------------------------------------------------------
    // System / request extraction
    // -----------------------------------------------------------------------

    #[test]
    fn extract_system_combines_top_level_and_messages() {
        let body = json!({
            "system": "top",
            "messages": [{"role": "system", "content": "msg"}]
        });
        let msgs = body.get("messages").and_then(|m| m.as_array());
        assert_eq!(
            extract_system(&body, msgs.map(|v| v.as_slice())),
            Some("top | msg".to_string())
        );

        let only_msg = json!({
            "messages": [{"role": "system", "content": "msg"}, {"role": "user", "content": "hi"}]
        });
        let msgs = only_msg.get("messages").and_then(|m| m.as_array());
        assert_eq!(
            extract_system(&only_msg, msgs.map(|v| v.as_slice())),
            Some("msg".to_string())
        );

        let none = json!({"messages": [{"role": "user", "content": "hi"}]});
        let msgs = none.get("messages").and_then(|m| m.as_array());
        assert_eq!(extract_system(&none, msgs.map(|v| v.as_slice())), None);
    }

    #[test]
    fn request_text_last_user_or_fallback() {
        // Last user message wins, whitespace collapsed.
        let body = json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "last  user\nline"}
            ]
        });
        assert_eq!(request_text(&body), "last user line");

        // No user message: join all messages.
        let no_user = json!({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "assistant", "content": "reply"}
            ]
        });
        assert_eq!(request_text(&no_user), "sys | reply");

        // User content empty -> "(no messages)".
        let empty_user = json!({"messages": [{"role": "user", "content": "  "}]});
        assert_eq!(request_text(&empty_user), "(no messages)");

        // No messages key -> "(no messages)".
        assert_eq!(request_text(&json!({"system": "x"})), "(no messages)");
    }

    #[test]
    fn extract_system_for_entry_variants() {
        let anthropic = json!({
            "system": "top",
            "messages": [{"role": "system", "content": "msg"}]
        });
        assert_eq!(
            extract_system_for_entry(&anthropic, LocalEntry::AnthropicMessages),
            Some("top | msg".to_string())
        );

        let chat = json!({
            "messages": [
                {"role": "system", "content": "s1"},
                {"role": "user", "content": "hi"}
            ]
        });
        assert_eq!(
            extract_system_for_entry(&chat, LocalEntry::OaiChat),
            Some("s1".to_string())
        );

        let responses = json!({
            "instructions": "inst",
            "input": [
                {"type": "message", "role": "system", "content": "s2"},
                {"type": "message", "role": "user", "content": "hi"}
            ]
        });
        assert_eq!(
            extract_system_for_entry(&responses, LocalEntry::OaiResponses),
            Some("inst | s2".to_string())
        );
    }

    #[test]
    fn request_text_for_entry_variants() {
        let anthropic = json!({
            "messages": [{"role": "user", "content": "hello  world"}]
        });
        assert_eq!(
            request_text_for_entry(&anthropic, LocalEntry::AnthropicMessages),
            "hello world"
        );

        let chat = json!({
            "messages": [
                {"role": "assistant", "content": "reply"},
                {"role": "user", "content": "hi there"}
            ]
        });
        assert_eq!(request_text_for_entry(&chat, LocalEntry::OaiChat), "hi there");

        let responses = json!({
            "input": [
                {"type": "message", "role": "assistant", "content": "reply"},
                {"type": "message", "role": "user", "content": "hi responses"}
            ]
        });
        assert_eq!(
            request_text_for_entry(&responses, LocalEntry::OaiResponses),
            "hi responses"
        );

        // OaiResponses with no user message -> "(no messages)".
        let no_user = json!({
            "input": [{"type": "message", "role": "assistant", "content": "reply"}]
        });
        assert_eq!(
            request_text_for_entry(&no_user, LocalEntry::OaiResponses),
            "(no messages)"
        );
    }

    #[test]
    fn request_text_keeps_anthropic_tool_result_newlines() {
        // The `[REQ]` line folds the prompt, but the tool's own output is the
        // tool's content, so its line breaks survive (ADR-0012).
        let body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "read\nthat"},
                    {"type": "tool_result", "tool_use_id": "t_1", "content": "line one\nline two"}
                ]
            }]
        });
        assert_eq!(
            request_text(&body),
            "read that [tool_result: t_1] line one\nline two"
        );
    }

    #[test]
    fn request_text_for_entry_renders_object_tool_result_as_json() {
        // A payload that is neither text nor a content array is serialised, not
        // described as "no content".
        let body = json!({
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "call_1", "content": {"ok": true}}
            ]
        });
        assert_eq!(
            request_text_for_entry(&body, LocalEntry::OaiChat),
            "go | [tool_result: call_1] {\"ok\":true}"
        );
    }

    #[test]
    fn request_text_for_entry_appends_chat_tool_results() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "read that file"},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "call_1", "function": {"name": "read_file", "arguments": "{}"}}
                ]},
                {"role": "tool", "tool_call_id": "call_1", "content": "line one\nline two"}
            ]
        });
        assert_eq!(
            request_text_for_entry(&body, LocalEntry::OaiChat),
            // The tool's own output keeps the line breaks it arrived with; only
            // the prompt side is folded onto the log line.
            "read that file | [tool_result: call_1] line one\nline two"
        );
    }

    #[test]
    fn request_text_for_entry_appends_responses_tool_results() {
        let body = json!({
            "input": [
                {"type": "message", "role": "user", "content": "read that file"},
                {"type": "function_call", "call_id": "call_1", "name": "read_file",
                 "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "contents"}
            ]
        });
        assert_eq!(
            request_text_for_entry(&body, LocalEntry::OaiResponses),
            "read that file | [tool_result: call_1] contents"
        );
    }

    #[test]
    fn request_text_for_entry_ignores_tool_results_before_last_user_message() {
        // Only the turn being answered is logged: an earlier turn's result must
        // not reappear on a later request's line.
        let body = json!({
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "tool", "tool_call_id": "call_0", "content": "old output"},
                {"role": "user", "content": "second"}
            ]
        });
        assert_eq!(request_text_for_entry(&body, LocalEntry::OaiChat), "second");
    }

    #[test]
    fn request_text_for_entry_tool_result_without_payload_keeps_bare_marker() {
        let body = json!({
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "tool", "tool_call_id": "call_1", "content": null}
            ]
        });
        assert_eq!(
            request_text_for_entry(&body, LocalEntry::OaiChat),
            "go | [tool_result: call_1]"
        );
    }

    #[test]
    fn request_text_for_entry_with_only_tool_results_drops_the_separator() {
        let body = json!({
            "messages": [
                {"role": "user", "content": ""},
                {"role": "tool", "tool_call_id": "call_1", "content": "out"}
            ]
        });
        assert_eq!(
            request_text_for_entry(&body, LocalEntry::OaiChat),
            "[tool_result: call_1] out"
        );
    }

    #[test]
    fn request_text_for_entry_without_a_user_turn_still_shows_tool_results() {
        let chat = json!({
            "messages": [
                {"role": "assistant", "content": "reply"},
                {"role": "tool", "tool_call_id": "call_1", "content": "out"}
            ]
        });
        assert_eq!(
            request_text_for_entry(&chat, LocalEntry::OaiChat),
            "reply | [tool_result: call_1] out"
        );

        let responses = json!({
            "input": [
                {"type": "function_call_output", "call_id": "call_1", "output": "out"}
            ]
        });
        assert_eq!(
            request_text_for_entry(&responses, LocalEntry::OaiResponses),
            "[tool_result: call_1] out"
        );

        // Nothing at all to show still falls back to the placeholder.
        assert_eq!(
            request_text_for_entry(&json!({"input": []}), LocalEntry::OaiResponses),
            "(no messages)"
        );
    }

    // -----------------------------------------------------------------------
    // Reasoning-request detection (feeds apply_reasoning_policy)
    // -----------------------------------------------------------------------

    #[test]
    fn reasoning_requested_for_entry_variants() {
        // Anthropic entry: thinking_requested semantics.
        let enabled = json!({"thinking": {"type": "enabled", "budget_tokens": 32000}});
        assert!(reasoning_requested_for_entry(&enabled, LocalEntry::AnthropicMessages));
        let adaptive = json!({"thinking": {"type": "adaptive"}});
        assert!(reasoning_requested_for_entry(&adaptive, LocalEntry::AnthropicMessages));
        let disabled = json!({"thinking": {"type": "disabled"}});
        assert!(!reasoning_requested_for_entry(&disabled, LocalEntry::AnthropicMessages));
        let absent = json!({"messages": []});
        assert!(!reasoning_requested_for_entry(&absent, LocalEntry::AnthropicMessages));
        let output_cfg = json!({"output_config": {"effort": "low"}});
        assert!(reasoning_requested_for_entry(&output_cfg, LocalEntry::AnthropicMessages));

        // Chat entry: reasoning_effort presence, "none" counts as disabled.
        let chat = json!({"reasoning_effort": "high"});
        assert!(reasoning_requested_for_entry(&chat, LocalEntry::OaiChat));
        let chat_none = json!({"reasoning_effort": "none"});
        assert!(!reasoning_requested_for_entry(&chat_none, LocalEntry::OaiChat));
        let chat_empty = json!({});
        assert!(!reasoning_requested_for_entry(&chat_empty, LocalEntry::OaiChat));

        // Responses entry: reasoning.effort.
        let responses = json!({"reasoning": {"effort": "medium", "summary": "auto"}});
        assert!(reasoning_requested_for_entry(&responses, LocalEntry::OaiResponses));
        let responses_none = json!({"reasoning": {"effort": "none"}});
        assert!(!reasoning_requested_for_entry(&responses_none, LocalEntry::OaiResponses));
        let responses_no_effort = json!({"reasoning": {"summary": "auto"}});
        assert!(!reasoning_requested_for_entry(&responses_no_effort, LocalEntry::OaiResponses));
    }

    // -----------------------------------------------------------------------
    // Response shaping
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn upstream_error_response_anthropic_shape() {
        let resp = upstream_error_response(
            StatusCode::BAD_GATEWAY,
            json!({"error": {"message": "boom"}}),
            LocalEntry::AnthropicMessages,
        );
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            body,
            json!({"type": "error", "error": {"type": "api_error", "message": "boom"}})
        );

        // Missing `error.message` falls back to a default message.
        let resp = upstream_error_response(
            StatusCode::BAD_GATEWAY,
            json!({}),
            LocalEntry::AnthropicMessages,
        );
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["message"], "Upstream request failed");
    }

    #[tokio::test]
    async fn upstream_error_response_openai_shape() {
        for entry in [LocalEntry::OaiChat, LocalEntry::OaiResponses] {
            let resp = upstream_error_response(
                StatusCode::BAD_GATEWAY,
                json!({"error": {"message": "boom"}}),
                entry,
            );
            assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
            let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
            let body: Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(
                body,
                json!({"error": {"message": "boom", "type": "upstream_error"}})
            );
        }
    }

    // -----------------------------------------------------------------------
    // handle_forward_error
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn handle_forward_error_relays_upstream_status_and_reason() {
        let reqlog = ReqLog::new();

        // Anthropic entry, upstream 503 → same status + anthropic error shape.
        let resp = handle_forward_error(
            &reqlog,
            Error::Upstream {
                status: StatusCode::SERVICE_UNAVAILABLE,
                reason: "Service Unavailable".into(),
                body: json!({"error": {"message": "Service Unavailable"}}),
            },
            LocalEntry::AnthropicMessages,
        );
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["message"], "Service Unavailable");

        // OpenAI entry, upstream 429 → status relayed instead of blanket 502.
        let resp = handle_forward_error(
            &reqlog,
            Error::Upstream {
                status: StatusCode::TOO_MANY_REQUESTS,
                reason: "Too Many Requests".into(),
                body: json!({"error": {"message": "Too Many Requests"}}),
            },
            LocalEntry::OaiChat,
        );
        assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(body["error"]["message"], "Too Many Requests");
    }

    #[test]
    fn handle_forward_error_non_upstream_keeps_old_mapping() {
        let reqlog = ReqLog::new();
        // Non-upstream forward errors still collapse to 502 via into_entry_response.
        let resp = handle_forward_error(&reqlog, Error::Forward("boom".into()), LocalEntry::OaiChat);
        assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);
    }

    #[test]
    fn upstream_error_preserves_status_in_status_and_message() {
        let err = Error::Upstream {
            status: StatusCode::TOO_MANY_REQUESTS,
            reason: "rate limit".into(),
            body: json!({}),
        };
        let (status, message) = err.status_and_message();
        assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(message, "rate limit");
    }

    #[tokio::test]
    async fn sse_response_sets_streaming_headers() {
        let resp = sse_response(Body::from(""));
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers().get("Content-Type").unwrap(),
            "text/event-stream"
        );
        assert_eq!(resp.headers().get("Cache-Control").unwrap(), "no-cache");
    }
}
