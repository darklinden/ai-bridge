use crate::error::Error;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::{json, Value};

/// The upstream API format, declared explicitly via the `upstream_type`
/// config key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UpstreamType {
    AnthropicMessages,
    OaiChat,
    OaiResponses,
}

impl UpstreamType {
    pub(crate) fn parse(name: &str) -> Result<Self, Error> {
        match name.trim().to_ascii_lowercase().as_str() {
            "anthropic-messages" => Ok(UpstreamType::AnthropicMessages),
            "oai-chat" => Ok(UpstreamType::OaiChat),
            "oai-responses" => Ok(UpstreamType::OaiResponses),
            other => Err(Error::Config(format!(
                "Invalid upstream_type \"{other}\": expected one of \
                 anthropic-messages | oai-chat | oai-responses"
            ))),
        }
    }

    /// The canonical lowercase name, for startup banners and logs.
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            UpstreamType::AnthropicMessages => "anthropic-messages",
            UpstreamType::OaiChat => "oai-chat",
            UpstreamType::OaiResponses => "oai-responses",
        }
    }
}

/// Config for a separately configured vision model used to describe images
/// before forwarding to a text-only upstream (the optional `[vision]` table).
#[derive(Debug, Clone)]
pub(crate) struct VisionConfig {
    pub(crate) url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) upstream_type: UpstreamType,
    /// Headers from `[vision.headers]` overridden on every vision request.
    pub(crate) override_headers: HeaderMap,
    /// Prompt template mode: "auto" (default) | "general" | "ui" | "compact".
    /// Normalized at parse time; unknown values fall back to "auto".
    pub(crate) prompt_mode: String,
    /// Custom description prompt from the `prompt` key; overrides `prompt_mode`.
    pub(crate) custom_prompt: Option<String>,
    /// Output token cap from the `max_tokens` key; None = omit. 0 → None.
    pub(crate) max_tokens: Option<u32>,
}

/// Configuration for forwarding requests to the configured upstream API.
/// Parsed once at startup from a TOML profile file by [`crate::config`].
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) upstream_type: UpstreamType,
    pub(crate) url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) listen_addr: String,
    pub(crate) listen_port: u16,
    pub(crate) auth_key: Option<String>,
    /// Headers from the `[headers]` table overridden on every upstream request.
    pub(crate) override_headers: HeaderMap,
    /// Fields from the `[extra_body]` table merged into every outgoing request
    /// body for the main upstream (never the vision endpoint). Applied at the
    /// single outbound seam, [`build_upstream_request`], after the reasoning
    /// policy and with the configured value winning on conflicts; keys the
    /// proxy owns are rejected at config time by `config::validate_extra_body`.
    pub(crate) extra_body: Option<Value>,
    /// Optional separately configured vision model for image description.
    pub(crate) vision: Option<VisionConfig>,
    /// Whether the third-party vision supplement is enabled (the
    /// `vision_supplement` key, default off). When off, images pass through to
    /// the upstream untouched so the upstream's own vision handles them; when
    /// on, text-only upstreams get the `[vision]` describe/strip path.
    pub(crate) vision_supplement_enabled: bool,
    /// Outbound reasoning policy (the `[reasoning]` table), applied to every
    /// forwarded request via [`apply_reasoning_policy`].
    pub(crate) reasoning_policy: UpstreamReasoningPolicy,
}

/// Infer the wire format of a `[vision]` endpoint from its URL:
/// contains `/responses` → OaiResponses; contains `/messages` but not
/// `/chat/completions` → AnthropicMessages; otherwise OaiChat. Vision
/// endpoints are OpenAI Chat or Anthropic Messages; anthropic URLs end in
/// /v1/messages which contains neither chat nor responses. This is the one
/// deliberate heuristic in the codebase — the main upstream type stays
/// explicit (ADR-0002).
pub(crate) fn vision_upstream_type_from_url(url: &str) -> UpstreamType {
    if url.contains("/responses") {
        UpstreamType::OaiResponses
    } else if url.contains("/messages") && !url.contains("/chat/completions") {
        UpstreamType::AnthropicMessages
    } else {
        UpstreamType::OaiChat
    }
}

/// Build a validated header map from `(name, value)` pairs sourced from a TOML
/// table (`[headers]` / `[vision.headers]`). Names and values are trimmed.
/// Entries with empty names or empty/invalid header values are skipped with a
/// WARN labeled by `label` (mirrors cc-switch's
/// `apply_local_proxy_header_overrides`). Later pairs win over earlier ones.
pub(crate) fn header_overrides_from_pairs(
    label: &str,
    entries: impl IntoIterator<Item = (String, String)>,
) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for (raw_name, raw_value) in entries {
        let name = raw_name.trim();
        let value = raw_value.trim();
        if name.is_empty() {
            tracing::warn!("[{label}] Ignoring entry with empty header name");
            continue;
        }
        if value.is_empty() {
            tracing::warn!("[{label}] Ignoring entry with empty value for {name}");
            continue;
        }
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            tracing::warn!("[{label}] Ignoring invalid header name: {raw_name}");
            continue;
        };
        let Ok(value) = HeaderValue::from_str(value) else {
            tracing::warn!("[{label}] Ignoring invalid header value for {name}");
            continue;
        };
        headers.insert(name, value);
    }
    headers
}

/// Parse a boolean config value: `1|true|yes|on` → true, `0|false|no|off` →
/// false; anything else (including a missing value) → `default`.
fn parse_bool(raw: Option<&str>, default: bool) -> bool {
    match raw.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("1" | "true" | "yes" | "on") => true,
        Some("0" | "false" | "no" | "off") => false,
        _ => default,
    }
}

// ---------------------------------------------------------------------------
// Outbound reasoning policy (`[reasoning]` table: thinking / effort)
// ---------------------------------------------------------------------------

/// What to do with the outgoing reasoning-effort field (`reasoning_effort` on
/// Chat Completions, `reasoning.effort` on Responses), resolved once at startup
/// from the `[reasoning]` config table. The configured value is the single
/// source of truth — no per-model mapping is applied on top of it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ReasoningEffortOverride {
    /// Stamp this exact value onto outgoing requests (lowercased).
    Set(String),
    /// Remove the field from outgoing requests entirely (for upstreams that
    /// reject the parameter altogether).
    Drop,
}

impl ReasoningEffortOverride {
    pub(crate) fn describe(&self) -> String {
        match self {
            ReasoningEffortOverride::Set(v) => v.clone(),
            ReasoningEffortOverride::Drop => "drop".into(),
        }
    }
}

/// Outbound reasoning policy applied to every forwarded request:
/// - `thinking_enabled == false` (master switch off): all thinking affordances
///   are stripped from outgoing requests and `effort` is ignored.
/// - otherwise the effort override decides the outgoing effort value, but only
///   for requests whose client actually asked for reasoning (an explicitly
///   disabled request never gains an effort field — some upstreams reject that
///   combination).
#[derive(Debug, Clone)]
pub(crate) struct UpstreamReasoningPolicy {
    /// Master switch from `[reasoning] thinking` (default on).
    pub(crate) thinking_enabled: bool,
    /// Effort value policy from `[reasoning] effort` (default `max`).
    pub(crate) effort: ReasoningEffortOverride,
}

impl UpstreamReasoningPolicy {
    /// Resolve the policy from raw string values so callers with typed config
    /// (e.g. TOML bools) stringify into the same accepted spellings, keeping a
    /// single source of truth. Used by tests without process-env mutation.
    pub(crate) fn parse(thinking: Option<&str>, effort: Option<&str>) -> Self {
        let thinking_enabled = parse_bool(thinking, true);
        let effort = match effort.map(str::trim).filter(|v| !v.is_empty()) {
            None => ReasoningEffortOverride::Set("max".into()),
            Some(raw) => match raw.to_ascii_lowercase().as_str() {
                "off" | "drop" | "none" | "disable" | "disabled" => ReasoningEffortOverride::Drop,
                _ => ReasoningEffortOverride::Set(raw.to_ascii_lowercase()),
            },
        };
        Self {
            thinking_enabled,
            effort,
        }
    }
}

/// Apply the outbound reasoning policy to a request body about to be forwarded.
///
/// `client_wants_reasoning`: whether the original local-entry request asked for
/// reasoning at all (any effort/thinking signal other than an explicit disable).
/// Only then is a `Set` effort stamped; explicit disables stay effort-free.
///
/// Wire formats:
/// - OaiChat: top-level `reasoning_effort`
/// - OaiResponses: `reasoning.effort` (other `reasoning` keys such as `summary`
///   are preserved; the object is pruned when it becomes empty)
/// - AnthropicMessages: no effort concept — only the master switch acts, by
///   removing the top-level `thinking` field.
pub(crate) fn apply_reasoning_policy(
    body: &mut Value,
    upstream_type: UpstreamType,
    policy: &UpstreamReasoningPolicy,
    client_wants_reasoning: bool,
) {
    match upstream_type {
        UpstreamType::AnthropicMessages => {
            if !policy.thinking_enabled && remove_json_key(body, "thinking") {
                tracing::debug!("reasoning policy: removed thinking ([reasoning] thinking off)");
            }
        }
        UpstreamType::OaiChat => {
            if !policy.thinking_enabled || policy.effort == ReasoningEffortOverride::Drop {
                if remove_json_key(body, "reasoning_effort") {
                    tracing::debug!("reasoning policy: removed reasoning_effort");
                }
            } else if client_wants_reasoning {
                let ReasoningEffortOverride::Set(value) = &policy.effort else {
                    return;
                };
                body["reasoning_effort"] = json!(value);
                tracing::debug!("reasoning policy: reasoning_effort → {value}");
            }
        }
        UpstreamType::OaiResponses => {
            if !policy.thinking_enabled || policy.effort == ReasoningEffortOverride::Drop {
                let removed = body
                    .get_mut("reasoning")
                    .and_then(Value::as_object_mut)
                    .is_some_and(|obj| obj.remove("effort").is_some());
                // Prune the now-empty `reasoning` object.
                if body.get("reasoning").and_then(Value::as_object) == Some(&serde_json::Map::new())
                {
                    remove_json_key(body, "reasoning");
                }
                if removed {
                    tracing::debug!("reasoning policy: removed reasoning.effort");
                }
            } else if client_wants_reasoning {
                let ReasoningEffortOverride::Set(value) = &policy.effort else {
                    return;
                };
                if body
                    .get("reasoning")
                    .map(|r| !r.is_object())
                    .unwrap_or(true)
                {
                    body["reasoning"] = json!({});
                }
                body["reasoning"]["effort"] = json!(value);
                tracing::debug!("reasoning policy: reasoning.effort → {value}");
            }
        }
    }
}

/// Remove `key` from an object body; returns whether anything was removed.
fn remove_json_key(body: &mut Value, key: &str) -> bool {
    body.as_object_mut()
        .is_some_and(|obj| obj.remove(key).is_some())
}

/// Merge the profile's `[extra_body]` fields into a request body about to be
/// forwarded, with the configured value winning every conflict.
///
/// Objects merge key by key (recursively, so a nested vendor object the client
/// or a conversion already produced is completed rather than replaced); any
/// other value — scalar, array, or a type mismatch — replaces the existing one
/// wholesale. New keys are appended after the existing ones in a fixed key
/// order, so the same profile always serialises the same way.
///
/// Keys the proxy owns (`model`, `stream`, the conversation container, the
/// reasoning fields, `stream_options.include_usage`) are rejected at startup
/// by `config::validate_extra_body`, so nothing here needs to special-case
/// them.
pub(crate) fn apply_extra_body(body: &mut Value, extra: &Value) {
    merge_json(body, extra);
}

fn merge_json(target: &mut Value, extra: &Value) {
    match (target, extra) {
        (Value::Object(target_map), Value::Object(extra_map)) => {
            for (key, value) in extra_map {
                match target_map.get_mut(key) {
                    Some(existing) => merge_json(existing, value),
                    None => {
                        target_map.insert(key.clone(), value.clone());
                    }
                }
            }
        }
        (target, extra) => *target = extra.clone(),
    }
}

/// Shared application state holding configuration and an HTTP client with connection pooling.
pub(crate) struct AppState {
    pub(crate) config: Config,
    pub(crate) client: reqwest::Client,
}

impl AppState {
    /// Create state from an already-parsed [`Config`].
    ///
    /// Builds a single `reqwest::Client` with connection pooling and a 5-minute
    /// upstream timeout, shared by all requests.
    pub(crate) fn new(config: Config) -> Result<Self, Error> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .pool_max_idle_per_host(32)
            .build()
            .map_err(|e| Error::Config(format!("Failed to build HTTP client: {e}")))?;
        Ok(Self { config, client })
    }
}

/// The auth/URL target for an outbound request: either the main upstream or the
/// optional vision model. Keeps `build_upstream_request` shared between both.
#[derive(Debug, Clone)]
pub(crate) struct UpstreamTarget {
    pub(crate) url: String,
    pub(crate) api_key: String,
    pub(crate) upstream_type: UpstreamType,
    pub(crate) override_headers: HeaderMap,
    /// `[extra_body]` fields for this target; `None` for the vision endpoint.
    pub(crate) extra_body: Option<Value>,
}

impl From<&Config> for UpstreamTarget {
    fn from(c: &Config) -> Self {
        UpstreamTarget {
            url: c.url.clone(),
            api_key: c.api_key.clone(),
            upstream_type: c.upstream_type,
            override_headers: c.override_headers.clone(),
            extra_body: c.extra_body.clone(),
        }
    }
}

impl From<&VisionConfig> for UpstreamTarget {
    fn from(v: &VisionConfig) -> Self {
        UpstreamTarget {
            url: v.url.clone(),
            api_key: v.api_key.clone(),
            upstream_type: v.upstream_type,
            override_headers: v.override_headers.clone(),
            // `[extra_body]` belongs to the main upstream only: a vision call is
            // a generated describe request, and vendor fields could break it.
            extra_body: None,
        }
    }
}

/// Build a POST request to an upstream with the default auth/content-type
/// headers plus any override headers (overrides win).
///
/// The profile's `[extra_body]` fields are merged into `request_body` here —
/// the one place every outbound request passes through, so every entry ×
/// upstream combination (pass-through included) inherits them. The merge
/// happens after conversion and after the reasoning policy, and the configured
/// value wins (reserved keys are rejected at config time).
///
/// Anthropic Messages upstreams authenticate with `x-api-key` +
/// `anthropic-version`; OpenAI-style upstreams use `Authorization: Bearer`.
/// Uses an explicit `HeaderMap` + `Body` instead of `.json()` so that a
/// `Content-Type` (or `Authorization`) override truly replaces the default
/// rather than producing a duplicate header (reqwest's `.json()` and
/// `.headers()` append).
fn build_upstream_request(
    client: &reqwest::Client,
    target: &UpstreamTarget,
    mut request_body: Value,
) -> Result<reqwest::RequestBuilder, Error> {
    let mut headers = HeaderMap::new();
    match target.upstream_type {
        UpstreamType::AnthropicMessages => {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(&target.api_key)
                    .map_err(|e| Error::Config(format!("Invalid API key header: {e}")))?,
            );
            headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        }
        UpstreamType::OaiChat | UpstreamType::OaiResponses => {
            headers.insert(
                http::header::AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {}", target.api_key))
                    .map_err(|e| Error::Config(format!("Invalid API key header: {e}")))?,
            );
        }
    }
    headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    for (name, value) in &target.override_headers {
        headers.insert(name.clone(), value.clone());
    }

    if let Some(extra) = &target.extra_body {
        apply_extra_body(&mut request_body, extra);
    }

    let body_bytes = serde_json::to_vec(&request_body)
        .map_err(|e| Error::Config(format!("Failed to serialize request body: {e}")))?;

    Ok(client
        .post(&target.url)
        .headers(headers)
        .body(reqwest::Body::from(body_bytes)))
}

/// Normalize a non-2xx upstream error body into `(Value body, single-line reason)`.
///
/// JSON bodies contribute `error.message` (or a top-level `message`); anything
/// unparseable falls back to the raw text. The returned reason is collapsed to
/// one line and truncated so gateway/HTML error pages can't flood the log, and
/// the body is guaranteed to carry `error.message == reason` so the client
/// response and the log line always agree.
fn parse_error_payload(text: &str) -> (Value, String) {
    let parsed: Option<Value> = serde_json::from_str(text).ok();

    let reason = parsed
        .as_ref()
        .and_then(|v| v.pointer("/error/message").and_then(|m| m.as_str()))
        .or_else(|| {
            parsed
                .as_ref()
                .and_then(|v| v.get("message").and_then(|m| m.as_str()))
        })
        .map(|s| s.split_whitespace().collect::<Vec<_>>().join(" "))
        .unwrap_or_else(|| text.split_whitespace().collect::<Vec<_>>().join(" "));
    let mut reason = reason.chars().take(500).collect::<String>();
    if reason.is_empty() {
        reason = "Unknown error".to_string();
    }

    let mut body = parsed.unwrap_or_else(|| json!({}));
    if body.get("error").is_none() {
        body["error"] = json!({});
    }
    body["error"]["message"] = json!(reason);

    (body, reason)
}

/// Read the full body of a non-2xx upstream response into a normalized
/// `(Value, reason)` pair via [`parse_error_payload`].
async fn read_upstream_error(resp: reqwest::Response) -> (Value, String) {
    let text = resp.text().await.unwrap_or_default();
    parse_error_payload(&text)
}

/// Check an upstream response status: 2xx passes through untouched for the
/// caller to parse/stream; otherwise the body is consumed and relayed as
/// [`Error::Upstream`] with the real status code and an extracted reason.
async fn ensure_upstream_success(response: reqwest::Response) -> Result<reqwest::Response, Error> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let (body, reason) = read_upstream_error(response).await;
    Err(Error::Upstream {
        status,
        reason,
        body,
    })
}

/// Forward a non-streaming request to the upstream API.
pub(crate) async fn forward_to_zen(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<Value, Error> {
    let response = build_upstream_request(client, target, request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    // Non-2xx → Error::Upstream (real status + reason); 2xx → parse JSON below.
    let response = ensure_upstream_success(response).await?;
    let status = response.status();

    let resp_body: Value = response
        .json::<Value>()
        .await
        .map_err(|e| Error::Forward(format!("Failed to parse response: {e}")))?;

    tracing::debug!(
        "[RECV OpenAI Chat] status={}, body_preview={}",
        status.as_u16(),
        &serde_json::to_string(&resp_body)
            .unwrap_or_default()
            .chars()
            .take(300)
            .collect::<String>()
    );

    Ok(resp_body)
}

/// Forward a streaming request to the upstream API and return the checked
/// response. Non-2xx statuses surface as [`Error::Upstream`] so the handler
/// only deals with clean 2xx streams (send/network failures stay `Err` too).
pub(crate) async fn forward_to_zen_streaming(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<reqwest::Response, Error> {
    let response = build_upstream_request(client, target, request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    ensure_upstream_success(response).await
}

/// Forward a non-streaming request to a Responses API endpoint.
pub(crate) async fn forward_to_responses(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<Value, Error> {
    let response = build_upstream_request(client, target, request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    // Responses API may return errors in 2xx with status: "failed" — handled
    // downstream by the converter; here only the wire status is checked.
    let response = ensure_upstream_success(response).await?;
    let status = response.status();

    let resp_body: Value = response
        .json::<Value>()
        .await
        .map_err(|e| Error::Forward(format!("Failed to parse response: {e}")))?;

    tracing::debug!(
        "[RECV OpenAI Responses] status={}, body_preview={}",
        status.as_u16(),
        &serde_json::to_string(&resp_body)
            .unwrap_or_default()
            .chars()
            .take(300)
            .collect::<String>()
    );

    Ok(resp_body)
}

/// Forward a streaming request to a Responses API endpoint and return the
/// checked response. Non-2xx statuses surface as [`Error::Upstream`].
pub(crate) async fn forward_to_responses_streaming(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<reqwest::Response, Error> {
    let response = build_upstream_request(client, target, request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    ensure_upstream_success(response).await
}

#[cfg(test)]
mod tests {
    use super::{
        apply_reasoning_policy, header_overrides_from_pairs, parse_bool, parse_error_payload,
        ReasoningEffortOverride, UpstreamReasoningPolicy, UpstreamType,
    };
    use serde_json::{json, Value};

    #[test]
    fn parse_bool_maps_truthy_values() {
        for value in ["1", "true", "TRUE", " yes ", "YES", "on", "On", " 1 "] {
            assert!(parse_bool(Some(value), false), "value: {value:?}");
        }
    }

    #[test]
    fn parse_bool_maps_falsy_values() {
        for value in ["0", "false", "FALSE", " no ", "off", "Off"] {
            assert!(!parse_bool(Some(value), true), "value: {value:?}");
        }
    }

    #[test]
    fn parse_bool_defaults_on_missing_or_invalid() {
        // Missing and unrecognized values fall back to the given default.
        assert!(!parse_bool(None, false));
        assert!(parse_bool(None, true));
        for value in ["", "   ", "maybe", "2", "tru"] {
            assert!(!parse_bool(Some(value), false), "value: {value:?}");
        }
    }

    // -----------------------------------------------------------------------
    // Outbound reasoning policy ([reasoning] thinking / [reasoning] effort)
    // -----------------------------------------------------------------------

    fn policy(thinking: Option<&str>, effort: Option<&str>) -> UpstreamReasoningPolicy {
        UpstreamReasoningPolicy::parse(thinking, effort)
    }

    #[test]
    fn reasoning_policy_defaults_to_thinking_on_and_max_effort() {
        let p = policy(None, None);
        assert!(p.thinking_enabled);
        assert_eq!(p.effort, ReasoningEffortOverride::Set("max".into()));

        // Empty strings behave like missing values.
        let p = policy(Some(""), Some("   "));
        assert!(p.thinking_enabled);
        assert_eq!(p.effort, ReasoningEffortOverride::Set("max".into()));
    }

    #[test]
    fn reasoning_policy_parses_explicit_values() {
        // Master switch off.
        assert!(!policy(Some("0"), None).thinking_enabled);
        assert!(!policy(Some("off"), None).thinking_enabled);
        assert!(policy(Some("1"), None).thinking_enabled);
        assert!(policy(Some("yes"), None).thinking_enabled);

        // Effort values are lowercased verbatim — no whitelist, upstreams vary.
        assert_eq!(
            policy(None, Some("XHigh")).effort,
            ReasoningEffortOverride::Set("xhigh".into())
        );
        assert_eq!(
            policy(None, Some("  Medium ")).effort,
            ReasoningEffortOverride::Set("medium".into())
        );

        // Reserved words drop the field entirely.
        for raw in ["off", "drop", "none", "disable", "disabled", "DROP"] {
            assert_eq!(
                policy(None, Some(raw)).effort,
                ReasoningEffortOverride::Drop,
                "raw: {raw}"
            );
        }
    }

    #[test]
    fn apply_policy_chat_stamps_overwrites_and_drops() {
        let on_max = policy(None, None);

        // Client asked for reasoning → effort stamped with the configured value.
        let mut body = json!({"model": "m", "messages": []});
        apply_reasoning_policy(&mut body, UpstreamType::OaiChat, &on_max, true);
        assert_eq!(body["reasoning_effort"], "max");

        // Explicit value wins over whatever the client sent.
        let mut body = json!({"reasoning_effort": "high"});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiChat,
            &policy(None, Some("xhigh")),
            true,
        );
        assert_eq!(body["reasoning_effort"], "xhigh");

        // Drop removes an existing field…
        let mut body = json!({"reasoning_effort": "high"});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiChat,
            &policy(None, Some("off")),
            true,
        );
        assert!(body.get("reasoning_effort").is_none());

        // …and master-off removes it even with a Set configured.
        let mut body = json!({"reasoning_effort": "high"});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiChat,
            &policy(Some("0"), None),
            true,
        );
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn apply_policy_chat_never_injects_without_client_request() {
        // No client reasoning signal → no field injected, even at default max.
        // (A thinking-disabled request reaches this stamp with wants=false too,
        // preserving the DeepSeek "disabled + reasoning_effort" 400 avoidance.)
        let mut body = json!({"model": "m", "messages": []});
        apply_reasoning_policy(&mut body, UpstreamType::OaiChat, &policy(None, None), false);
        assert!(body.get("reasoning_effort").is_none());

        // Master off wins over everything: even a requesting client gets stripped.
        let mut body = json!({"reasoning_effort": "high"});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiChat,
            &policy(Some("0"), None),
            true,
        );
        assert!(body.get("reasoning_effort").is_none());
    }

    #[test]
    fn apply_policy_responses_preserves_other_reasoning_keys() {
        // Set: effort stamped inside the existing object; summary preserved.
        let mut body = json!({"reasoning": {"effort": "high", "summary": "auto"}});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiResponses,
            &policy(None, Some("low")),
            true,
        );
        assert_eq!(body["reasoning"]["effort"], "low");
        assert_eq!(body["reasoning"]["summary"], "auto");

        // Set: missing reasoning object is created.
        let mut body = json!({});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiResponses,
            &policy(None, None),
            true,
        );
        assert_eq!(body["reasoning"]["effort"], "max");

        // Drop: effort removed, sibling keys survive.
        let mut body = json!({"reasoning": {"effort": "high", "summary": "auto"}});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiResponses,
            &policy(None, Some("off")),
            true,
        );
        assert!(body["reasoning"].get("effort").is_none());
        assert_eq!(body["reasoning"]["summary"], "auto");

        // Drop: empty reasoning object is pruned altogether.
        let mut body = json!({"reasoning": {"effort": "high"}});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::OaiResponses,
            &policy(None, Some("off")),
            true,
        );
        assert!(body.get("reasoning").is_none());
    }

    #[test]
    fn apply_policy_anthropic_only_acts_when_master_off() {
        // Master on: untouched (no effort concept on this wire format).
        let mut body = json!({"thinking": {"type": "enabled"}, "messages": []});
        apply_reasoning_policy(
            &mut body,
            UpstreamType::AnthropicMessages,
            &policy(None, None),
            true,
        );
        assert!(body.get("thinking").is_some());

        // Master off: thinking stripped.
        apply_reasoning_policy(
            &mut body,
            UpstreamType::AnthropicMessages,
            &policy(Some("false"), None),
            true,
        );
        assert!(body.get("thinking").is_none());
    }

    #[test]
    fn upstream_type_accepts_all_three_values() {
        assert_eq!(
            UpstreamType::parse("anthropic-messages").unwrap(),
            UpstreamType::AnthropicMessages
        );
        assert_eq!(
            UpstreamType::parse("oai-chat").unwrap(),
            UpstreamType::OaiChat
        );
        assert_eq!(
            UpstreamType::parse("oai-responses").unwrap(),
            UpstreamType::OaiResponses
        );
    }

    #[test]
    fn upstream_type_is_case_insensitive() {
        assert_eq!(
            UpstreamType::parse("OAI-CHAT").unwrap(),
            UpstreamType::OaiChat
        );
        assert_eq!(
            UpstreamType::parse("  Anthropic-Messages  ").unwrap(),
            UpstreamType::AnthropicMessages
        );
    }

    #[test]
    fn upstream_type_rejects_unknown_values() {
        assert!(UpstreamType::parse("bogus").is_err());
        assert!(UpstreamType::parse("").is_err());
        assert!(UpstreamType::parse("responses").is_err());
    }

    #[test]
    fn vision_upstream_type_inferred_from_url_markers() {
        use super::vision_upstream_type_from_url;
        assert_eq!(
            vision_upstream_type_from_url("https://v.example/v1/responses"),
            UpstreamType::OaiResponses
        );
        assert_eq!(
            vision_upstream_type_from_url("https://v.example/v1/messages"),
            UpstreamType::AnthropicMessages
        );
        // /chat/completions contains neither marker → chat; it also contains
        // no "/messages", but guard against a URL carrying both markers.
        assert_eq!(
            vision_upstream_type_from_url("https://v.example/v1/chat/completions"),
            UpstreamType::OaiChat
        );
        assert_eq!(
            vision_upstream_type_from_url("https://messages.example/v1/chat/completions"),
            UpstreamType::OaiChat
        );
    }

    fn overrides(entries: &[(&str, &str)]) -> http::HeaderMap {
        header_overrides_from_pairs(
            "[headers]",
            entries.iter().map(|(n, v)| (n.to_string(), v.to_string())),
        )
    }

    #[test]
    fn builds_simple_pairs() {
        let headers = overrides(&[("A", "a"), ("B", "b")]);
        assert_eq!(headers.len(), 2);
        assert_eq!(headers.get("a").unwrap(), "a");
        assert_eq!(headers.get("B").unwrap(), "b");
    }

    #[test]
    fn trims_whitespace_around_names_and_values() {
        let headers = overrides(&[(" X-Trace ", "123"), ("Y", " 2 ")]);
        assert_eq!(headers.len(), 2);
        assert_eq!(headers.get("x-trace").unwrap(), "123");
        assert_eq!(headers.get("y").unwrap(), "2");
    }

    #[test]
    fn empty_input_yields_empty_map() {
        assert!(overrides(&[]).is_empty());
    }

    #[test]
    fn skips_invalid_entries_but_keeps_valid_siblings() {
        let headers = overrides(&[("bad name", "x"), ("X-Trace", "1")]);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("x-trace").unwrap(), "1");
    }

    #[test]
    fn skips_empty_value() {
        let headers = overrides(&[("A", "")]);
        assert!(headers.is_empty());
    }

    #[test]
    fn duplicate_names_last_wins() {
        let headers = overrides(&[("X", "1"), ("X", "2")]);
        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("x").unwrap(), "2");
    }

    // -----------------------------------------------------------------------
    // parse_error_payload
    // -----------------------------------------------------------------------

    #[test]
    fn error_payload_uses_error_message_field() {
        let text = r#"{"error": {"message": "Service Unavailable", "type": "overloaded_error"}}"#;
        let (body, reason) = parse_error_payload(text);
        assert_eq!(reason, "Service Unavailable");
        assert_eq!(body["error"]["message"], "Service Unavailable");
        // Other error fields are preserved.
        assert_eq!(body["error"]["type"], "overloaded_error");
    }

    #[test]
    fn error_payload_falls_back_to_top_level_message() {
        let text = r#"{"message": "Too Many Requests"}"#;
        let (body, reason) = parse_error_payload(text);
        assert_eq!(reason, "Too Many Requests");
        // The normalized body gains an error.message so shaping always works.
        assert_eq!(body["error"]["message"], "Too Many Requests");
    }

    #[test]
    fn error_payload_non_json_uses_raw_text() {
        let text = "Service Unavailable";
        let (body, reason) = parse_error_payload(text);
        assert_eq!(reason, "Service Unavailable");
        assert_eq!(body["error"]["message"], "Service Unavailable");
    }

    #[test]
    fn error_payload_collapses_multiline_plain_text() {
        let text = "<html>\n  <h1>502 Bad Gateway</h1>\n</html>";
        let (_, reason) = parse_error_payload(text);
        assert_eq!(reason, "<html> <h1>502 Bad Gateway</h1> </html>");
    }

    #[test]
    fn error_payload_empty_body_defaults_to_unknown() {
        let (body, reason) = parse_error_payload("");
        assert_eq!(reason, "Unknown error");
        assert_eq!(body["error"]["message"], "Unknown error");
    }

    #[test]
    fn error_payload_truncates_overlong_reason() {
        let long = "x".repeat(2000);
        let (_, reason) = parse_error_payload(&long);
        assert_eq!(reason.chars().count(), 500);
    }

    // -----------------------------------------------------------------------
    // [extra_body] merging
    // -----------------------------------------------------------------------

    fn merged(base: Value, extra: Value) -> Value {
        let mut body = base;
        super::apply_extra_body(&mut body, &extra);
        body
    }

    #[test]
    fn extra_body_adds_new_keys_and_overrides_wholesale_values() {
        // Scalars and arrays are replaced, not merged element by element.
        let body = merged(
            json!({"model": "m", "messages": [], "stop": ["a"], "temperature": 0.2}),
            json!({
                "providerOptions": {"gateway": {"only": ["deepseek"]}},
                "stop": ["b", "c"],
                "temperature": 0.0
            }),
        );
        assert_eq!(
            body["providerOptions"]["gateway"]["only"],
            json!(["deepseek"])
        );
        assert_eq!(body["stop"], json!(["b", "c"]));
        assert_eq!(body["temperature"], 0.0);
        // Untouched proxy fields survive.
        assert_eq!(body["model"], "m");
        assert_eq!(body["messages"], json!([]));
    }

    #[test]
    fn extra_body_deep_merges_objects_with_config_winning() {
        let body = merged(
            json!({
                "providerOptions": {
                    "gateway": {"only": ["a"], "order": ["x"]},
                    "client": "keep"
                }
            }),
            json!({
                "providerOptions": {
                    "gateway": {"only": ["deepseek"], "has": ["vision"]}
                }
            }),
        );
        // Conflicting leaf: config wins.
        assert_eq!(
            body["providerOptions"]["gateway"]["only"],
            json!(["deepseek"])
        );
        // Sibling leaves and objects the config did not mention survive.
        assert_eq!(body["providerOptions"]["gateway"]["order"], json!(["x"]));
        assert_eq!(body["providerOptions"]["gateway"]["has"], json!(["vision"]));
        assert_eq!(body["providerOptions"]["client"], "keep");
    }

    #[test]
    fn extra_body_replaces_on_type_mismatch() {
        // Object vs array is not a merge — config wins as a whole.
        let body = merged(
            json!({"providerOptions": {"a": 1}}),
            json!({"providerOptions": ["x"]}),
        );
        assert_eq!(body["providerOptions"], json!(["x"]));

        let body = merged(
            json!({"providerOptions": ["x"]}),
            json!({"providerOptions": {"a": 1}}),
        );
        assert_eq!(body["providerOptions"], json!({"a": 1}));
    }

    #[test]
    fn extra_body_empty_object_is_a_noop() {
        let body = merged(json!({"model": "m", "messages": []}), json!({}));
        assert_eq!(body, json!({"model": "m", "messages": []}));
    }

    /// Merge order only: existing keys keep their position and a new key is
    /// appended after them (`serde_json` preserve_order). The profile's own key
    /// order (TOML table iteration) is fixed too — `config` covers that end.
    #[test]
    fn extra_body_appends_new_keys_after_existing_ones() {
        let body = merged(
            json!({"model": "m", "messages": []}),
            json!({"zeta": 1, "alpha": 2}),
        );
        assert_eq!(
            serde_json::to_string(&body).unwrap(),
            r#"{"model":"m","messages":[],"zeta":1,"alpha":2}"#
        );
    }

    /// `[extra_body]` belongs to the main upstream; vision calls are generated
    /// describe requests and must not inherit vendor fields.
    #[test]
    fn vision_target_carries_no_extra_body() {
        let vision = super::VisionConfig {
            url: "https://v.example/v1/chat/completions".into(),
            api_key: "k".into(),
            model: "vl".into(),
            upstream_type: UpstreamType::OaiChat,
            override_headers: http::HeaderMap::new(),
            prompt_mode: "auto".into(),
            custom_prompt: None,
            max_tokens: None,
        };
        let target = super::UpstreamTarget::from(&vision);
        assert!(target.extra_body.is_none());
    }

    // -----------------------------------------------------------------------
    // The outbound seam: build_upstream_request actually applies the merge
    // -----------------------------------------------------------------------

    fn main_target(
        upstream_type: UpstreamType,
        extra_body: Option<Value>,
    ) -> super::UpstreamTarget {
        super::UpstreamTarget {
            url: "https://up.example/v1/chat/completions".into(),
            api_key: "k".into(),
            upstream_type,
            override_headers: http::HeaderMap::new(),
            extra_body,
        }
    }

    fn main_config(extra_body: Option<Value>) -> super::Config {
        super::Config {
            upstream_type: UpstreamType::OaiChat,
            url: "https://up.example/v1/chat/completions".into(),
            api_key: "k".into(),
            model: "m".into(),
            listen_addr: "127.0.0.1".into(),
            listen_port: 18650,
            auth_key: None,
            override_headers: http::HeaderMap::new(),
            extra_body,
            vision: None,
            vision_supplement_enabled: false,
            reasoning_policy: UpstreamReasoningPolicy::parse(None, None),
        }
    }

    /// Read back the in-memory request body as JSON (every request this proxy
    /// builds is a full `Vec<u8>` body).
    fn sent_body(request: reqwest::Request) -> Value {
        let bytes = request
            .body()
            .and_then(reqwest::Body::as_bytes)
            .expect("in-memory request body");
        serde_json::from_slice(bytes).expect("request body is JSON")
    }

    #[test]
    fn build_upstream_request_merges_extra_body_into_the_sent_body() {
        let client = reqwest::Client::new();
        let target = main_target(
            UpstreamType::OaiChat,
            Some(json!({
                "providerOptions": {"gateway": {"only": ["deepseek"]}},
                "temperature": 0.0
            })),
        );
        let request = super::build_upstream_request(
            &client,
            &target,
            json!({"model": "m", "messages": [], "temperature": 0.7}),
        )
        .expect("builder")
        .build()
        .expect("complete request");

        assert_eq!(request.method().as_str(), "POST");
        assert_eq!(
            request.url().as_str(),
            "https://up.example/v1/chat/completions"
        );

        // The bytes that would go on the wire carry the merged fields: config
        // wins on the conflict, untouched keys survive.
        let body = sent_body(request);
        assert_eq!(
            body["providerOptions"]["gateway"]["only"],
            json!(["deepseek"])
        );
        assert_eq!(body["temperature"], 0.0);
        assert_eq!(body["model"], "m");
        assert_eq!(body["messages"], json!([]));
    }

    #[test]
    fn build_upstream_request_without_extra_body_sends_the_body_unchanged() {
        let client = reqwest::Client::new();
        let target = main_target(UpstreamType::OaiChat, None);
        let request =
            super::build_upstream_request(&client, &target, json!({"model": "m", "messages": []}))
                .expect("builder")
                .build()
                .expect("complete request");

        assert_eq!(sent_body(request), json!({"model": "m", "messages": []}));
    }

    #[test]
    fn main_target_carries_the_configured_extra_body() {
        let extra = json!({"providerOptions": {"gateway": {"only": ["deepseek"]}}});
        let target = super::UpstreamTarget::from(&main_config(Some(extra.clone())));
        assert_eq!(target.extra_body, Some(extra));
        assert!(super::UpstreamTarget::from(&main_config(None))
            .extra_body
            .is_none());
    }
}
