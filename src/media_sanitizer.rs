//! Proactive image stripping for confirmed text-only upstream models.
//!
//! DeepSeek-family and other text-only models reject any OpenAI `image_url` (or
//! Responses `input_image`) content part with a deserialization error like
//!
//! ```text
//! Failed to deserialize the JSON body into the target type:
//! messages[2]: unknown variant `image_url`
//! ```
//!
//! Because the offending image block lives on in the conversation history, the
//! failure then repeats on *every* later request of the same session.
//!
//! This module replaces image blocks in the Anthropic request body with a text
//! marker *before* conversion, so no image content ever reaches the upstream.
//! The text-only model registry is ported from
//! `src-tauri/src/model_capabilities.rs::is_confirmed_text_only_model`; the
//! sanitizer mirrors `src-tauri/src/proxy/media_sanitizer.rs`.

use crate::tool_media::{strip_and_clamp_media_from_tool_value, ToolMediaScope};
use serde_json::{json, Value};

pub(crate) const UNSUPPORTED_IMAGE_MARKER: &str = "[Unsupported Image]";

/// Replace every image block in `body["messages"]` with `[Unsupported Image]`
/// when `model` is a confirmed text-only model. Returns the number of blocks
/// replaced (0 = model supports images, unknown, or no images present).
///
/// The decision is driven by the *upstream* model that will actually receive
/// the request (the configured `OPENAI_COMP_MODEL`), not the model the client
/// asked for.
pub(crate) fn replace_images_for_text_only_model(body: &mut Value, model: &str) -> usize {
    if !is_confirmed_text_only_model(model) {
        return 0;
    }

    let replacement = json!({ "type": "text", "text": UNSUPPORTED_IMAGE_MARKER });
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return 0;
    };

    messages
        .iter_mut()
        .map(|message| replace_images_in_message(message, &replacement))
        .sum()
}

fn replace_images_in_message(message: &mut Value, replacement: &Value) -> usize {
    let Some(content) = message.get_mut("content") else {
        return 0;
    };
    let Some(blocks) = content.as_array_mut() else {
        return 0;
    };

    let mut replaced = 0;
    for block in blocks.iter_mut() {
        match block.get("type").and_then(Value::as_str) {
            // Direct image block (Anthropic `image`, or an already Chat/Responses
            // shaped `image_url` / `input_image` the client sent verbatim).
            Some("image" | "image_url" | "input_image") => {
                replace_block_with_text_marker(block, replacement);
                replaced += 1;
            }
            // Tool results carry images as nested content (or as a stringified
            // JSON payload). The shared traversal replaces those while leaving
            // ordinary text output byte-for-byte untouched.
            Some("tool_result") => {
                if let Some(nested) = block.get_mut("content") {
                    let mut discarded_media = Vec::new();
                    replaced += strip_and_clamp_media_from_tool_value(
                        nested,
                        &mut discarded_media,
                        ToolMediaScope::ImagesOnly,
                        replacement,
                        UNSUPPORTED_IMAGE_MARKER,
                    );
                }
            }
            _ => {}
        }
    }
    replaced
}

/// Replace an image block with the text marker, carrying over any
/// `cache_control` breakpoint so prompt caching is not disrupted.
fn replace_block_with_text_marker(block: &mut Value, replacement: &Value) {
    let cache_control = block.get("cache_control").cloned();
    *block = replacement.clone();
    if let (Some(cache_control), Some(object)) = (cache_control, block.as_object_mut()) {
        object.insert("cache_control".to_string(), cache_control);
    }
}

/// Models that CC Switch is willing to treat as text-only.
///
/// This registry is deliberately exact and fail-open: an unlisted model (or an
/// unconfirmed `-vision`/`-vl` variant of a listed family) keeps its images.
/// Ported from `src-tauri/src/model_capabilities.rs`.
pub(crate) fn is_confirmed_text_only_model(model: &str) -> bool {
    let normalized = normalize_model_id(model);
    let tail = normalized.rsplit('/').next().unwrap_or(normalized.as_str());

    const CONFIRMED_TAILS: &[&str] = &[
        "ark-code-latest",
        "deepseek-chat",
        "deepseek-reasoner",
        "deepseek-v4-flash",
        "deepseek-v4-flash-free",
        "deepseek-v4-pro",
        "glm-5.1",
        // Exact rather than prefix matching: GLM visual models use a `v`
        // suffix (for example glm-5.2v), which must remain image-capable.
        "glm-5.2",
        "kat-coder",
        "kat-coder-pro",
        "kat-coder-pro v1",
        "kat-coder-pro v2",
        "kat-coder-pro-v1",
        "kat-coder-pro-v2",
        "ling-2.5-1t",
        "longcat-2.0",
        "longcat-flash-chat",
        "minimax-m2.7",
        "minimax-m2.7-highspeed",
        "mimo-v2.5-pro",
        "qwen3-coder-480b",
        "qwen3-coder-480b-a35b-instruct",
        "qwen3-coder-flash",
        "qwen3-coder-next",
        "qwen3-coder-plus",
        "step-3.5-flash",
        "step-3.5-flash-2603",
        "us.deepseek.r1-v1",
    ];

    CONFIRMED_TAILS.contains(&tail)
}

fn normalize_model_id(value: &str) -> String {
    let mut normalized = value
        .trim()
        .trim_start_matches("models/")
        .trim()
        .to_ascii_lowercase();
    // `claude_desktop_config::ONE_M_CONTEXT_MARKER` == "[1m]"
    if let Some(stripped) = normalized.strip_suffix("[1m]") {
        normalized = stripped.trim().to_string();
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_block() -> Value {
        json!({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc"}
        })
    }

    #[test]
    fn deepseek_models_are_classified_text_only() {
        for model in [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek/deepseek-v4-pro",
            "deepseek-chat",
            "deepseek-reasoner",
        ] {
            assert!(is_confirmed_text_only_model(model), "{model}");
        }
    }

    #[test]
    fn registry_normalizes_namespaces_and_context_markers() {
        assert!(is_confirmed_text_only_model("deepseek/deepseek-v4-pro"));
        assert!(is_confirmed_text_only_model("GLM-5.2[1M]"));
        assert!(is_confirmed_text_only_model("Qwen/Qwen3-Coder-480B-A35B-Instruct"));
        assert!(is_confirmed_text_only_model("MiniMax-M2.7-Highspeed"));
        assert!(is_confirmed_text_only_model("LongCat-2.0"));
        assert!(!is_confirmed_text_only_model("glm-5.2v"));
    }

    #[test]
    fn vision_and_unknown_models_fail_open() {
        for model in ["gpt-5.4", "claude-sonnet-5", "qwen3-coder-vl", "glm-5.2v"] {
            assert!(!is_confirmed_text_only_model(model), "{model}");
        }
    }

    #[test]
    fn deepseek_replaces_direct_image_block_and_preserves_cache_control() {
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    image_block().clone(),
                    {"type": "image", "cache_control": {"type": "ephemeral"}, "source": {"type": "base64", "media_type": "image/jpeg", "data": "def"}}
                ]
            }]
        });

        let count = replace_images_for_text_only_model(&mut body, "deepseek-v4-flash");

        assert_eq!(count, 2);
        let content = &body["messages"][0]["content"];
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], UNSUPPORTED_IMAGE_MARKER);
        assert_eq!(content[2]["type"], "text");
        assert_eq!(content[2]["cache_control"]["type"], "ephemeral");
        assert!(!body.to_string().contains("image_url"));
    }

    #[test]
    fn deepseek_replaces_tool_result_image_blocks() {
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "here is the screenshot"},
                        image_block().clone()
                    ]
                }]
            }]
        });

        let count = replace_images_for_text_only_model(&mut body, "deepseek/deepseek-v4-pro");

        assert_eq!(count, 1);
        let tool_content = &body["messages"][0]["content"][0]["content"];
        assert_eq!(tool_content[0]["type"], "text");
        assert_eq!(tool_content[1]["type"], "text");
        assert_eq!(tool_content[1]["text"], UNSUPPORTED_IMAGE_MARKER);
        assert!(!body.to_string().contains("image_url"));
    }

    #[test]
    fn deepseek_replaces_stringified_json_tool_result_image() {
        let content = json!({
            "content": [{
                "type": "image",
                "mimeType": "image/png",
                "data": "STRINGIFIED_IMAGE_SENTINEL"
            }]
        })
        .to_string();
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": content
                }]
            }]
        });

        let count = replace_images_for_text_only_model(&mut body, "deepseek-v4-flash");

        assert_eq!(count, 1);
        let rewritten = body["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap();
        assert!(rewritten.contains(UNSUPPORTED_IMAGE_MARKER));
        assert!(!rewritten.contains("STRINGIFIED_IMAGE_SENTINEL"));
    }

    #[test]
    fn vision_model_and_unknown_model_keep_images() {
        for model in ["gpt-5.4", "claude-sonnet-5", "deepseek-vision-unknown"] {
            let mut body = json!({
                "messages": [{"role": "user", "content": [image_block().clone()]}]
            });
            let original = body.clone();

            let count = replace_images_for_text_only_model(&mut body, model);

            assert_eq!(count, 0, "{model}");
            assert_eq!(body, original);
        }
    }

    #[test]
    fn plain_text_tool_result_is_byte_stable() {
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "just some text output"
                }]
            }]
        });
        let original = body.clone();

        let count = replace_images_for_text_only_model(&mut body, "deepseek-v4-flash");

        assert_eq!(count, 0);
        assert_eq!(body, original);
    }

    #[test]
    fn whole_string_image_data_url_is_replaced() {
        let data_url = format!(
            "data:image/png;base64,{}",
            "iVBORw0KGgoAAAANSUhEUgAAAAE".repeat(400)
        );
        let mut body = json!({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": data_url.clone()
                }]
            }]
        });

        let count = replace_images_for_text_only_model(&mut body, "deepseek-v4-pro");

        assert_eq!(count, 1);
        assert!(!body.to_string().contains(&data_url));
    }

    /// The end-to-end failure path from the bug report: a conversation that
    /// already carries a `Read`-tool image block (so the failure would otherwise
    /// repeat on every later request). After sanitizing + converting to OpenAI
    /// Chat, no `image_url` part may reach the upstream — and a plain-text model
    /// that has never seen an image must pass through unchanged.
    #[test]
    fn sanitize_then_convert_emits_no_image_url_for_text_only_model() {
        let read_image_body = json!({
            "model": "claude-sonnet-5",
            "max_tokens": 4096,
            "stream": false,
            "messages": [
                {"role": "user", "content": "read the file"},
                {
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [
                            {"type": "text", "text": "screenshot:"},
                            image_block().clone()
                        ]
                    }]
                }
            ]
        });

        // Text-only upstream: images stripped before conversion.
        let mut sanitized = read_image_body.clone();
        let replaced = replace_images_for_text_only_model(&mut sanitized, "deepseek-v4-flash");
        assert_eq!(replaced, 1);
        let chat = crate::convert::anthropic_to_openai_with_reasoning_content(sanitized, true)
            .unwrap();
        let serialized = chat.to_string();
        assert!(
            !serialized.contains("image_url"),
            "image_url must not reach a text-only upstream: {serialized}"
        );

        // Vision-capable upstream: images must be preserved end to end.
        let unsanitized = read_image_body.clone();
        let chat = crate::convert::anthropic_to_openai_with_reasoning_content(unsanitized, true)
            .unwrap();
        assert!(
            chat.to_string().contains("image_url"),
            "vision-capable upstream should still receive image_url"
        );
    }
}
