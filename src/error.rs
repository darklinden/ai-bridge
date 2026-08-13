use axum::{
    http::StatusCode,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use thiserror::Error;

/// The local API style a request came in through. Error responses are shaped to
/// match this format so client SDKs can parse them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LocalEntry {
    AnthropicMessages,
    OaiChat,
    OaiResponses,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("格式转换错误: {0}")]
    Transform(String),

    #[error("转发失败: {0}")]
    Forward(String),

    #[error("配置错误: {0}")]
    Config(String),

    #[error("服务器错误: {0}")]
    Server(String),

    #[error("未授权: {0}")]
    Unauthorized(String),

    /// A request feature the upstream format cannot represent (e.g. `n > 1`
    /// against an Anthropic upstream). Maps to HTTP 400.
    #[error("不支持的请求: {0}")]
    Unsupported(String),
}

impl From<std::num::ParseIntError> for Error {
    fn from(e: std::num::ParseIntError) -> Self {
        Error::Config(format!("Invalid port: {e}"))
    }
}

impl Error {
    /// HTTP status and message for this error, independent of local entry format.
    pub(crate) fn status_and_message(&self) -> (StatusCode, String) {
        match self {
            Error::Transform(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            Error::Forward(msg) => (StatusCode::BAD_GATEWAY, msg.clone()),
            Error::Config(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            Error::Server(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg.clone()),
            Error::Unauthorized(msg) => (StatusCode::UNAUTHORIZED, msg.clone()),
            Error::Unsupported(msg) => (StatusCode::BAD_REQUEST, msg.clone()),
        }
    }

    /// Render this error in the JSON shape of the given local entry format.
    pub(crate) fn into_entry_response(self, entry: LocalEntry) -> Response {
        let (status, message) = self.status_and_message();
        match entry {
            LocalEntry::AnthropicMessages => {
                (status, Json(json!({ "error": { "type": "error", "message": message } })))
                    .into_response()
            }
            LocalEntry::OaiChat | LocalEntry::OaiResponses => {
                let error_type = if status == StatusCode::BAD_REQUEST {
                    "invalid_request_error"
                } else if status == StatusCode::UNAUTHORIZED {
                    "authentication_error"
                } else {
                    "api_error"
                };
                (
                    status,
                    Json(json!({ "error": { "message": message, "type": error_type } })),
                )
                    .into_response()
            }
        }
    }
}

impl IntoResponse for Error {
    fn into_response(self) -> Response {
        // Default to the Anthropic error shape; entry-aware callers use
        // `into_entry_response` instead.
        self.into_entry_response(LocalEntry::AnthropicMessages)
    }
}
