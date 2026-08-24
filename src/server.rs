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

/// Collapse newlines/whitespace so the complete text stays on a single log line
/// (replaces line breaks with spaces, does NOT truncate content).
fn single_line(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Brief summary of an assistant `tool_use` block: name plus the serialized
/// arguments the model sent (no truncation).
fn tool_use_summary(block: &Value) -> String {
    let name = block.get("name").and_then(|n| n.as_str()).unwrap_or("?");
    if let Some(input) = block.get("input") {
        let s = serde_json::to_string(input).unwrap_or_default();
        if !s.is_empty() {
            return format!("[tool_use: {name}] {s}");
        }
    }
    format!("[tool_use: {name}]")
}

/// Brief summary of a `tool_result` block: id plus the actual tool output
/// content (string, or flattened text/tool blocks), no truncation.
fn tool_result_summary(block: &Value) -> String {
    let id = block.get("tool_use_id").and_then(|i| i.as_str()).unwrap_or("?");
    let content = match block.get("content") {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(_)) => extract_text_content(block.get("content")),
        _ => "/".to_string(),
    };
    if content.is_empty() {
        format!("[tool_result: {id}]")
    } else {
        format!("[tool_result: {id}] {content}")
    }
}

/// Extract plain text from an Anthropic content field (String, Array of blocks, or null).
/// Joins multiple text/thinking blocks into the complete text (no truncation).
fn extract_text_content(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(blocks)) => {
            let texts: Vec<String> = blocks
                .iter()
                .filter_map(|b| match b.get("type").and_then(|t| t.as_str()) {
                    Some("text") => b.get("text").and_then(|t| t.as_str()).map(|t| t.to_string()),
                    Some("thinking") => b.get("thinking").and_then(|t| t.as_str()).map(|t| {
                        format!("{} (thinking)", t)
                    }),
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
                texts.join(" ")
            }
        }
        Some(Value::Null) | None => "null".to_string(),
        Some(other) => serde_json::to_string(other).unwrap_or_default(),
    }
}

/// Extract the user prompt text from an OpenAI Chat message content.
fn extract_openai_chat_text(content: Option<&Value>) -> String {
    match content {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|p| {
                let t = p.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match t {
                    "text" | "input_text" | "output_text" => {
                        p.get("text").and_then(|v| v.as_str()).map(str::to_string)
                    }
                    "image_url" | "input_image" => Some("[image]".to_string()),
                    _ => None,
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => "(no content)".to_string(),
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
                    let t = extract_openai_chat_text(m.get("content"));
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
                    let t = extract_openai_chat_text(m.get("content"));
                    if !t.trim().is_empty() {
                        parts.push(t);
                    }
                }
            }
            (!parts.is_empty()).then(|| parts.join(" | "))
        }
    }
}

/// Extract the user prompt text for the given local entry.
fn request_text_for_entry(body: &Value, entry: LocalEntry) -> String {
    match entry {
        LocalEntry::AnthropicMessages => request_text(body),
        LocalEntry::OaiChat => {
            let msgs = body.get("messages").and_then(|m| m.as_array());
            let text = match msgs {
                Some(msgs) => {
                    let user = msgs
                        .iter()
                        .rev()
                        .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
                        .map(|m| extract_openai_chat_text(m.get("content")))
                        .unwrap_or_else(|| {
                            let all: Vec<String> = msgs
                                .iter()
                                .map(|m| extract_openai_chat_text(m.get("content")))
                                .collect();
                            all.join(" | ")
                        });
                    single_line(&user)
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
                    let user = input
                        .iter()
                        .rev()
                        .find(|m| {
                            m.get("type").and_then(|t| t.as_str()) == Some("message")
                                && m.get("role").and_then(|r| r.as_str()) == Some("user")
                        })
                        .map(|m| extract_openai_chat_text(m.get("content")))
                        .unwrap_or_else(|| "(no messages)".to_string());
                    single_line(&user)
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
        reqlog.resp(&format!("stop_reason=passthrough ({model})"));
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
        reqlog.resp(&format!("stop_reason={}", stop_reason));
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
        reqlog.resp(&format!("stop_reason={}", stop_reason));
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
        reqlog.resp("ok");
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
        reqlog.resp("ok");
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
        reqlog.resp("ok");
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
        reqlog.resp("ok");
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
    if let Some(top) = body.get("system").map(|s| extract_text_content(Some(s))) {
        if !top.trim().is_empty() {
            parts.push(top);
        }
    }
    if let Some(msgs) = msgs {
        for m in msgs
            .iter()
            .filter(|m| m.get("role").and_then(|r| r.as_str()) == Some("system"))
        {
            let t = extract_text_content(m.get("content"));
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
fn request_text(body: &Value) -> String {
    let msgs = body.get("messages").and_then(|m| m.as_array());
    match msgs {
        Some(msgs) => {
            let user = msgs
                .iter()
                .rev()
                .find(|m| m.get("role").and_then(|r| r.as_str()) == Some("user"))
                .map(|m| extract_text_content(m.get("content")))
                .unwrap_or_else(|| {
                    let all: Vec<String> = msgs
                        .iter()
                        .map(|m| extract_text_content(m.get("content")))
                        .collect();
                    all.join(" | ")
                });
            let user = single_line(&user);
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
        // Non-string/array content -> "/".
        assert_eq!(
            tool_result_summary(&json!({"tool_use_id": "t_1", "content": null})),
            "[tool_result: t_1] /"
        );
    }

    #[test]
    fn extract_text_content_variants() {
        assert_eq!(extract_text_content(Some(&json!("hello"))), "hello");

        let blocks = json!([
            {"type": "text", "text": "a"},
            {"type": "thinking", "thinking": "b"},
            {"type": "image"},
            {"type": "document"},
            {"type": "tool_use", "name": "f", "input": {"x": 1}},
            {"type": "tool_result", "tool_use_id": "t", "content": "out"}
        ]);
        assert_eq!(
            extract_text_content(Some(&blocks)),
            "a b (thinking) [image] [document] [tool_use: f] {\"x\":1} [tool_result: t] out"
        );

        assert_eq!(extract_text_content(Some(&json!([]))), "[empty]");
        assert_eq!(extract_text_content(None), "null");
        assert_eq!(extract_text_content(Some(&Value::Null)), "null");
        assert_eq!(extract_text_content(Some(&json!(42))), "42");
    }

    #[test]
    fn extract_openai_chat_text_variants() {
        assert_eq!(extract_openai_chat_text(Some(&json!("hi"))), "hi");

        let parts = json!([
            {"type": "text", "text": "a"},
            {"type": "input_text", "text": "b"},
            {"type": "output_text", "text": "c"},
            {"type": "image_url"},
            {"type": "input_image"},
            {"type": "other", "text": "ignored"}
        ]);
        assert_eq!(extract_openai_chat_text(Some(&parts)), "a b c [image] [image]");

        assert_eq!(extract_openai_chat_text(None), "(no content)");
        assert_eq!(extract_openai_chat_text(Some(&Value::Null)), "(no content)");
        assert_eq!(extract_openai_chat_text(Some(&json!([]))), "");
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
