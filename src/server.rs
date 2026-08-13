use crate::convert;
use crate::convert_reverse;
use crate::error::{Error, LocalEntry};
use crate::forward::{AppState, UpstreamTarget, UpstreamType};
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

/// Authenticate the inbound request against `UPSTREAM_AUTH_KEY` if configured.
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
    body: Value,
    is_stream: bool,
    model: &str,
    reqlog: &Arc<ReqLog>,
    upstream: UpstreamTarget,
    entry: LocalEntry,
) -> Result<Response<Body>, Error> {
    if is_stream {
        let upstream_resp = crate::forward::forward_to_zen_streaming(
            body,
            &state.client,
            &upstream,
        )
        .await?;
        let status = upstream_resp.status();
        if !status.is_success() {
            let body = upstream_resp
                .json::<Value>()
                .await
                .unwrap_or(json!({"error": {"message": "Unknown error"}}));
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            reqlog.err_req(&format!("upstream status {}: {message}", status.as_u16()));
            return Ok(upstream_error_response(status, body, entry));
        }
        let stream = upstream_resp.bytes_stream();
        Ok(sse_response(Body::from_stream(stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
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
    let mut chat_body = convert::anthropic_to_openai_with_reasoning_content(body, true)?;
    convert::inject_openai_stream_include_usage(&mut chat_body);

    if is_stream {
        let upstream_resp = crate::forward::forward_to_zen_streaming(
            chat_body,
            &state.client,
            &upstream,
        )
        .await?;
        let status = upstream_resp.status();
        if !status.is_success() {
            let body = upstream_resp
                .json::<Value>()
                .await
                .unwrap_or(json!({"error": {"message": "Unknown error"}}));
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            reqlog.err_req(&format!("upstream status {}: {message}", status.as_u16()));
            return Ok(upstream_error_response(status, body, ANTHROPIC_ENTRY));
        }
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
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
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
    let mut responses_body = transform_responses::anthropic_to_responses(body)?;

    if is_stream {
        responses_body["stream"] = json!(true);
        let upstream_resp = crate::forward::forward_to_responses_streaming(
            responses_body,
            &state.client,
            &upstream,
        )
        .await?;
        let status = upstream_resp.status();
        if !status.is_success() {
            let body = upstream_resp
                .json::<Value>()
                .await
                .unwrap_or(json!({"error": {"message": "Unknown error"}}));
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            reqlog.err_req(&format!("upstream status {}: {message}", status.as_u16()));
            return Ok(upstream_error_response(status, body, ANTHROPIC_ENTRY));
        }
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
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
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
    let anthropic_body = match convert_reverse::chat_to_anthropic_request(&body) {
        Ok(v) => v,
        Err(e) => {
            reqlog.err_req(&e.to_string());
            return Err(e);
        }
    };

    if is_stream {
        let upstream_resp = crate::forward::forward_to_zen_streaming(
            anthropic_body,
            &state.client,
            &upstream,
        )
        .await?;
        let status = upstream_resp.status();
        if !status.is_success() {
            let body = upstream_resp
                .json::<Value>()
                .await
                .unwrap_or(json!({"error": {"message": "Unknown error"}}));
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            reqlog.err_req(&format!("upstream status {}: {message}", status.as_u16()));
            return Ok(upstream_error_response(status, body, CHAT_ENTRY));
        }
        let stream = upstream_resp.bytes_stream();
        let chat_stream = convert_reverse::anthropic_to_chat_sse(
            stream,
            model.to_string(),
            reqlog.clone(),
        );
        Ok(sse_response(Body::from_stream(chat_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            anthropic_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
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
    let mut responses_body = transform_responses::anthropic_to_responses(anthropic_mid)?;

    if is_stream {
        responses_body["stream"] = json!(true);
        let upstream_resp = crate::forward::forward_to_responses_streaming(
            responses_body,
            &state.client,
            &upstream,
        )
        .await?;
        let status = upstream_resp.status();
        if !status.is_success() {
            let body = upstream_resp
                .json::<Value>()
                .await
                .unwrap_or(json!({"error": {"message": "Unknown error"}}));
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            reqlog.err_req(&format!("upstream status {}: {message}", status.as_u16()));
            return Ok(upstream_error_response(status, body, CHAT_ENTRY));
        }
        let stream = upstream_resp.bytes_stream();
        // responses SSE → anthropic SSE → chat SSE (double bridge).
        let anthropic_stream =
            crate::streaming_responses::responses_to_anthropic_sse(stream, model.to_string(), reqlog.clone());
        let chat_stream = convert_reverse::anthropic_to_chat_sse(
            anthropic_stream,
            model.to_string(),
            reqlog.clone(),
        );
        Ok(sse_response(Body::from_stream(chat_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_responses(
            responses_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
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
    let anthropic_body = match responses_reverse::responses_to_anthropic_request(&body) {
        Ok(v) => v,
        Err(e) => {
            reqlog.err_req(&e.to_string());
            return Err(e);
        }
    };

    if is_stream {
        let upstream_resp = crate::forward::forward_to_zen_streaming(
            anthropic_body,
            &state.client,
            &upstream,
        )
        .await?;
        let status = upstream_resp.status();
        if !status.is_success() {
            let body = upstream_resp
                .json::<Value>()
                .await
                .unwrap_or(json!({"error": {"message": "Unknown error"}}));
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            reqlog.err_req(&format!("upstream status {}: {message}", status.as_u16()));
            return Ok(upstream_error_response(status, body, RESPONSES_ENTRY));
        }
        let stream = upstream_resp.bytes_stream();
        let responses_stream = responses_reverse::anthropic_to_responses_sse(
            stream,
            model.to_string(),
            reqlog.clone(),
        );
        Ok(sse_response(Body::from_stream(responses_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            anthropic_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
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
    let mut chat_body = convert::anthropic_to_openai_with_reasoning_content(anthropic_mid, true)?;
    convert::inject_openai_stream_include_usage(&mut chat_body);

    if is_stream {
        let upstream_resp = crate::forward::forward_to_zen_streaming(
            chat_body,
            &state.client,
            &upstream,
        )
        .await?;
        let status = upstream_resp.status();
        if !status.is_success() {
            let body = upstream_resp
                .json::<Value>()
                .await
                .unwrap_or(json!({"error": {"message": "Unknown error"}}));
            let message = body
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .unwrap_or("Unknown error");
            reqlog.err_req(&format!("upstream status {}: {message}", status.as_u16()));
            return Ok(upstream_error_response(status, body, RESPONSES_ENTRY));
        }
        let stream = upstream_resp.bytes_stream();
        // chat SSE → anthropic SSE → responses SSE.
        let anthropic_stream =
            convert::chat_to_anthropic_sse(stream, model.to_string(), reqlog.clone());
        let responses_stream = responses_reverse::anthropic_to_responses_sse(
            anthropic_stream,
            model.to_string(),
            reqlog.clone(),
        );
        Ok(sse_response(Body::from_stream(responses_stream)))
    } else {
        let upstream_resp = match crate::forward::forward_to_zen(
            chat_body, &state.client, &upstream,
        )
        .await
        {
            Ok(v) => v,
            Err(e) => {
                reqlog.err_req(&e.to_string());
                return Err(e);
            }
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
