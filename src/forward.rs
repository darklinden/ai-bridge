use crate::error::Error;
use http::{HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;

/// The upstream API format, declared explicitly via `UPSTREAM_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UpstreamType {
    AnthropicMessages,
    OaiChat,
    OaiResponses,
}

impl UpstreamType {
    pub(crate) fn from_env_var(name: &str) -> Result<Self, Error> {
        match name.trim().to_ascii_lowercase().as_str() {
            "anthropic-messages" => Ok(UpstreamType::AnthropicMessages),
            "oai-chat" => Ok(UpstreamType::OaiChat),
            "oai-responses" => Ok(UpstreamType::OaiResponses),
            other => Err(Error::Config(format!(
                "Invalid UPSTREAM_TYPE \"{other}\": expected one of \
                 anthropic-messages | oai-chat | oai-responses"
            ))),
        }
    }
}

/// Config for a separately configured vision model used to describe images
/// before forwarding to a text-only upstream (`VISION_*`).
#[derive(Debug, Clone)]
pub(crate) struct VisionConfig {
    pub(crate) url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) upstream_type: UpstreamType,
    /// Headers from `VISION_HEADERS` overridden on every vision request.
    pub(crate) override_headers: HeaderMap,
}

/// Configuration for forwarding requests to the configured upstream API.
#[derive(Debug, Clone)]
pub(crate) struct Config {
    pub(crate) upstream_type: UpstreamType,
    pub(crate) url: String,
    pub(crate) api_key: String,
    pub(crate) model: String,
    pub(crate) listen_addr: String,
    pub(crate) listen_port: u16,
    pub(crate) auth_key: Option<String>,
    /// Headers from `UPSTREAM_HEADERS` overridden on every upstream request.
    pub(crate) override_headers: HeaderMap,
    /// Optional separately configured vision model for image description.
    pub(crate) vision: Option<VisionConfig>,
}

impl Config {
    /// Read configuration from environment variables.
    ///
    /// Required: `UPSTREAM_TYPE`, `UPSTREAM_URL`, `UPSTREAM_API_KEY`
    /// Optional: `UPSTREAM_MODEL` (default: `deepseek-v4-flash`),
    ///           `LISTEN_ADDR` (default: `0.0.0.0`),
    ///           `LISTEN_PORT` (default: `18650`),
    ///           `UPSTREAM_AUTH_KEY`, `UPSTREAM_HEADERS`,
    ///           `VISION_URL` / `VISION_API_KEY` / `VISION_MODEL` / `VISION_HEADERS`
    pub fn from_env() -> Result<Self, Error> {
        reject_legacy_env()?;

        let upstream_type = UpstreamType::from_env_var(
            &std::env::var("UPSTREAM_TYPE").map_err(|_| {
                Error::Config("Missing UPSTREAM_TYPE environment variable".into())
            })?,
        )?;
        let url = std::env::var("UPSTREAM_URL")
            .map_err(|_| Error::Config("Missing UPSTREAM_URL environment variable".into()))?;
        let api_key = std::env::var("UPSTREAM_API_KEY")
            .map_err(|_| Error::Config("Missing UPSTREAM_API_KEY environment variable".into()))?;
        let model = std::env::var("UPSTREAM_MODEL").unwrap_or_else(|_| "deepseek-v4-flash".into());
        let listen_addr = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0".into());
        let listen_port = std::env::var("LISTEN_PORT")
            .unwrap_or_else(|_| "18650".into())
            .parse::<u16>()?;
        let auth_key = std::env::var("UPSTREAM_AUTH_KEY").ok();
        let override_headers =
            parse_header_overrides(&std::env::var("UPSTREAM_HEADERS").unwrap_or_default());

        for (name, value) in &override_headers {
            tracing::debug!(
                "Header override: {}={}",
                name,
                value.to_str().unwrap_or("<non-ascii>")
            );
        }

        let vision = parse_vision_config()?;

        Ok(Self {
            upstream_type,
            url,
            api_key,
            model,
            listen_addr,
            listen_port,
            auth_key,
            override_headers,
            vision,
        })
    }
}

/// Reject any legacy `OPENAI_COMP_*` variable with a clear migration error.
/// The naming migration is intentionally non-compatible (ADR-0003).
fn reject_legacy_env() -> Result<(), Error> {
    for legacy in [
        "OPENAI_COMP_URL",
        "OPENAI_COMP_API_KEY",
        "OPENAI_COMP_MODEL",
        "OPENAI_COMP_AUTH_KEY",
        "OPENAI_COMP_HEADERS",
    ] {
        if std::env::var_os(legacy).is_some() {
            return Err(Error::Config(format!(
                "Legacy environment variable {legacy} is no longer supported. \
                 Rename to the corresponding UPSTREAM_* variable (see AGENTS.md / ADR-0003)."
            )));
        }
    }
    Ok(())
}

/// Parse the optional `VISION_*` config. Returns `None` when any required piece
/// is absent — vision is strictly additive and never falls back to the upstream.
fn parse_vision_config() -> Result<Option<VisionConfig>, Error> {
    let url = std::env::var("VISION_URL").ok();
    let api_key = std::env::var("VISION_API_KEY").ok();
    let model = std::env::var("VISION_MODEL").ok();

    let Some(url) = url else {
        return Ok(None);
    };
    let Some(api_key) = api_key else {
        return Ok(None);
    };
    let Some(model) = model else {
        return Ok(None);
    };

    let upstream_type = if url.contains("/responses") {
        UpstreamType::OaiResponses
    } else {
        // Vision endpoints are OpenAI Chat or Anthropic Messages; anthropic URLs
        // end in /v1/messages which contains neither chat nor responses.
        if url.contains("/messages") && !url.contains("/chat/completions") {
            UpstreamType::AnthropicMessages
        } else {
            UpstreamType::OaiChat
        }
    };

    let override_headers =
        parse_header_overrides(&std::env::var("VISION_HEADERS").unwrap_or_default());

    Ok(Some(VisionConfig {
        url,
        api_key,
        model,
        upstream_type,
        override_headers,
    }))
}

/// Parse `OPENAI_COMP_HEADERS` (`A:a|B:b`) into a validated header map.
///
/// Pairs are `|`-separated; each value is split on the first `:` so values may
/// contain `:` (but not `|`). Names and values are trimmed. Entries with empty
/// names or empty/invalid header values are skipped with a WARN (mirrors
/// cc-switch's `apply_local_proxy_header_overrides`). Duplicate names: last wins.
fn parse_header_overrides(raw: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    for pair in raw.split('|') {
        let Some((raw_name, raw_value)) = pair.split_once(':') else {
            continue;
        };
        let name = raw_name.trim();
        let value = raw_value.trim();
        if name.is_empty() {
            tracing::warn!("[OPENAI_COMP_HEADERS] Ignoring entry with empty header name");
            continue;
        }
        if value.is_empty() {
            tracing::warn!("[OPENAI_COMP_HEADERS] Ignoring entry with empty value for {name}");
            continue;
        }
        let Ok(name) = HeaderName::from_bytes(name.as_bytes()) else {
            tracing::warn!("[OPENAI_COMP_HEADERS] Ignoring invalid header name: {raw_name}");
            continue;
        };
        let Ok(value) = HeaderValue::from_str(value) else {
            tracing::warn!("[OPENAI_COMP_HEADERS] Ignoring invalid header value for {name}");
            continue;
        };
        headers.insert(name, value);
    }
    headers
}

/// Shared application state holding configuration and an HTTP client with connection pooling.
pub(crate) struct AppState {
    pub(crate) config: Config,
    pub(crate) client: reqwest::Client,
}

impl AppState {
    /// Create state from environment variables.
    ///
    /// Builds a single `reqwest::Client` with connection pooling and a 5-minute
    /// upstream timeout, shared by all requests.
    pub(crate) fn from_env() -> Result<Self, Error> {
        let config = Config::from_env()?;
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
}

impl From<&Config> for UpstreamTarget {
    fn from(c: &Config) -> Self {
        UpstreamTarget {
            url: c.url.clone(),
            api_key: c.api_key.clone(),
            upstream_type: c.upstream_type,
            override_headers: c.override_headers.clone(),
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
        }
    }
}

/// Build a POST request to an upstream with the default auth/content-type
/// headers plus any override headers (overrides win).
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
    request_body: &Value,
) -> Result<reqwest::RequestBuilder, Error> {
    let mut headers = HeaderMap::new();
    match target.upstream_type {
        UpstreamType::AnthropicMessages => {
            headers.insert(
                "x-api-key",
                HeaderValue::from_str(&target.api_key)
                    .map_err(|e| Error::Config(format!("Invalid API key header: {e}")))?,
            );
            headers.insert(
                "anthropic-version",
                HeaderValue::from_static("2023-06-01"),
            );
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

    let body_bytes = serde_json::to_vec(request_body)
        .map_err(|e| Error::Config(format!("Failed to serialize request body: {e}")))?;

    Ok(client
        .post(&target.url)
        .headers(headers)
        .body(reqwest::Body::from(body_bytes)))
}

/// Forward a non-streaming request to the upstream API.
pub(crate) async fn forward_to_zen(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<Value, Error> {
    let response = build_upstream_request(client, target, &request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Error::Forward(format!(
            "Upstream error (status {}): {}",
            status.as_u16(),
            body
        )));
    }

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

/// Forward a streaming request to the upstream API and return the raw response.
pub(crate) async fn forward_to_zen_streaming(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<reqwest::Response, Error> {
    let response = build_upstream_request(client, target, &request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    Ok(response)
}

/// Forward a non-streaming request to a Responses API endpoint.
pub(crate) async fn forward_to_responses(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<Value, Error> {
    let response = build_upstream_request(client, target, &request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    // Responses API may return errors in 2xx with status: "failed"
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(Error::Forward(format!(
            "Upstream error (status {}): {}",
            status.as_u16(),
            body
        )));
    }

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

/// Forward a streaming request to a Responses API endpoint and return the raw response.
pub(crate) async fn forward_to_responses_streaming(
    request_body: Value,
    client: &reqwest::Client,
    target: &UpstreamTarget,
) -> Result<reqwest::Response, Error> {
    let response = build_upstream_request(client, target, &request_body)?
        .send()
        .await
        .map_err(|e| Error::Forward(format!("Request failed: {e}")))?;

    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::{parse_header_overrides, UpstreamType};

    #[test]
    fn upstream_type_accepts_all_three_values() {
        assert_eq!(
            UpstreamType::from_env_var("anthropic-messages").unwrap(),
            UpstreamType::AnthropicMessages
        );
        assert_eq!(
            UpstreamType::from_env_var("oai-chat").unwrap(),
            UpstreamType::OaiChat
        );
        assert_eq!(
            UpstreamType::from_env_var("oai-responses").unwrap(),
            UpstreamType::OaiResponses
        );
    }

    #[test]
    fn upstream_type_is_case_insensitive() {
        assert_eq!(
            UpstreamType::from_env_var("OAI-CHAT").unwrap(),
            UpstreamType::OaiChat
        );
        assert_eq!(
            UpstreamType::from_env_var("  Anthropic-Messages  ").unwrap(),
            UpstreamType::AnthropicMessages
        );
    }

    #[test]
    fn upstream_type_rejects_unknown_values() {
        assert!(UpstreamType::from_env_var("bogus").is_err());
        assert!(UpstreamType::from_env_var("").is_err());
        assert!(UpstreamType::from_env_var("responses").is_err());
    }

    #[test]
    fn parses_simple_pairs() {
        let headers = parse_header_overrides("A:a|B:b");
        assert_eq!(headers.len(), 2);
        assert_eq!(headers.get("a").unwrap(), "a");
        assert_eq!(headers.get("B").unwrap(), "b");
    }

    #[test]
    fn value_may_contain_colons() {
        let headers = parse_header_overrides("Cookie:x=y:z");
        assert_eq!(headers.get("cookie").unwrap(), "x=y:z");
    }

    #[test]
    fn trims_whitespace_around_names_and_values() {
        let headers = parse_header_overrides(" X-Trace : 123 | Y : 2 ");
        assert_eq!(headers.len(), 2);
        assert_eq!(headers.get("x-trace").unwrap(), "123");
        assert_eq!(headers.get("y").unwrap(), "2");
    }

    #[test]
    fn empty_input_yields_empty_map() {
        let headers = parse_header_overrides("");
        assert!(headers.is_empty());
    }

    #[test]
    fn skips_invalid_entries_but_keeps_valid_siblings() {
        let headers = parse_header_overrides("bad name:x|X-Trace:1");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("x-trace").unwrap(), "1");
    }

    #[test]
    fn skips_empty_value() {
        let headers = parse_header_overrides("A:");
        assert!(headers.is_empty());
    }

    #[test]
    fn duplicate_names_last_wins() {
        let headers = parse_header_overrides("X:1|X:2");
        assert_eq!(headers.len(), 1);
        assert_eq!(headers.get("x").unwrap(), "2");
    }
}
