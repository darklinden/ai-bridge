//! Image description for text-only upstreams.
//!
//! When a local request carries images and the upstream model is confirmed
//! text-only (e.g. DeepSeek), a separately configured vision model (`VISION_*`)
//! describes the images and the description text replaces the image blocks
//! before forwarding. This mirrors Plugin-Deepseek-Vision's approach, but with
//! our own config surface instead of a plugin host.
//!
//! Behavior:
//! - All images in one request are analyzed together in a single non-streaming
//!   vision call (multi-image joint analysis).
//! - Descriptions are cached in-process keyed by image fingerprint with a TTL.
//! - A vision failure degrades to the `[Unsupported Image]` placeholder and does
//!   not block the request (the main upstream is unaffected).

use crate::error::{Error, LocalEntry};
use crate::forward::{UpstreamTarget, VisionConfig};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// TTL for the description cache.
const CACHE_TTL: Duration = Duration::from_secs(3600);

/// A cached description entry.
struct CacheEntry {
    expires_at: Instant,
    description: String,
}

static CACHE: Mutex<Option<HashMap<u64, CacheEntry>>> = Mutex::new(None);

/// Fingerprint an image block (data URL or base64 payload) for caching.
fn image_fingerprint(block: &Value) -> Option<u64> {
    let data = block
        .pointer("/source/data")
        .and_then(Value::as_str)
        .map(str::as_bytes)
        .or_else(|| {
            block
                .get("image_url")
                .and_then(|u| u.get("url"))
                .and_then(Value::as_str)
                .map(str::as_bytes)
        })?;
    // FNV-1a 64-bit — cheap and good enough for cache keying.
    Some(
        data.iter()
            .fold(0xcbf29ce484222325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3)),
    )
}

/// Read a cached description, if present and not expired.
fn cache_get(fingerprint: u64) -> Option<String> {
    let mut guard = CACHE.lock().unwrap();
    let cache = guard.as_mut()?;
    match cache.get(&fingerprint) {
        Some(entry) if entry.expires_at > Instant::now() => {
            Some(entry.description.clone())
        }
        Some(_) => {
            cache.remove(&fingerprint);
            None
        }
        None => None,
    }
}

/// Store a description in the cache.
fn cache_put(fingerprint: u64, description: String) {
    let mut guard = CACHE.lock().unwrap();
    let cache = guard.get_or_insert_with(HashMap::new);
    cache.insert(
        fingerprint,
        CacheEntry {
            expires_at: Instant::now() + CACHE_TTL,
            description,
        },
    );
}

/// Describe all images in a request body for a text-only upstream, replacing
/// the image blocks with description text. Returns the number of images handled.
///
/// `entry` determines the structure to scan (Anthropic `messages`, Chat
/// `messages`, or Responses `input`). `state` provides the shared HTTP client;
/// `vision` is the separately configured vision model.
pub async fn describe_images_in_body(
    body: &mut Value,
    entry: LocalEntry,
    state: &crate::forward::AppState,
    vision: &VisionConfig,
) -> Result<usize, Error> {
    // Collect image blocks from the entry structure.
    let mut images: Vec<Value> = Vec::new();
    let mut fingerprints: Vec<Option<u64>> = Vec::new();

    collect_images(body, entry, &mut images, &mut fingerprints);

    if images.is_empty() {
        return Ok(0);
    }

    // Split into cached vs uncached images. Cached ones are pre-filled; the
    // uncached subset goes through one joint vision call.
    let mut uncached: Vec<(usize, u64, &Value)> = Vec::new();
    let mut descriptions: Vec<Option<String>> = vec![None; images.len()];
    for (i, fp) in fingerprints.iter().enumerate() {
        match fp {
            Some(fp) => {
                if let Some(cached) = cache_get(*fp) {
                    descriptions[i] = Some(cached);
                } else {
                    uncached.push((i, *fp, &images[i]));
                }
            }
            None => uncached.push((i, 0, &images[i])),
        }
    }

    // Joint analysis: one vision call describing all uncached images together.
    if !uncached.is_empty() {
        let vision_request = build_vision_request(&images, &uncached, vision);
        let description = call_vision(vision_request, state, vision).await;

        match description {
            Ok(text) if !text.trim().is_empty() => {
                for (i, fp, _) in &uncached {
                    descriptions[*i] = Some(text.clone());
                    if *fp != 0 {
                        cache_put(*fp, text.clone());
                    }
                }
            }
            _ => {
                // Vision failure → degrade to placeholder, do not block.
                for (i, _, _) in &uncached {
                    descriptions[*i] = None;
                }
            }
        }
    }

    // Replace images in the body.
    let mut handled = 0;
    replace_images_in_body(body, entry, &images, &descriptions, &mut handled);
    Ok(handled)
}

/// Scan the body for image blocks according to the entry structure.
fn collect_images(
    body: &Value,
    entry: LocalEntry,
    images: &mut Vec<Value>,
    fingerprints: &mut Vec<Option<u64>>,
) {
    let containers: Vec<&Vec<Value>> = match entry {
        LocalEntry::AnthropicMessages | LocalEntry::OaiChat => body
            .get("messages")
            .and_then(Value::as_array)
            .map(|a| vec![a])
            .unwrap_or_default(),
        LocalEntry::OaiResponses => body
            .get("input")
            .and_then(Value::as_array)
            .map(|a| vec![a])
            .unwrap_or_default(),
    };

    for container in containers {
        for msg in container {
            let content = match entry {
                LocalEntry::OaiResponses => msg.get("content").and_then(Value::as_array),
                _ => msg.get("content").and_then(Value::as_array),
            };
            if let Some(blocks) = content {
                for block in blocks {
                    let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
                    match bt {
                        "image" | "image_url" | "input_image" => {
                            images.push(block.clone());
                            fingerprints.push(image_fingerprint(block));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

/// Build a joint vision request describing all images at once.
fn build_vision_request(
    _all_images: &[Value],
    uncached: &[(usize, u64, &Value)],
    vision: &VisionConfig,
) -> Value {
    // Describe the uncached subset in one prompt, in order.
    let mut content: Vec<Value> = Vec::with_capacity(uncached.len() + 1);
    content.push(json!({
        "type": "text",
        "text": "Describe each of these images concisely. For each image, describe \
                 its content, any visible text, and how the images relate to each \
                 other. Prefix each description with [Image N]."
    }));
    for (_, _, block) in uncached {
        if let Some(part) = image_to_content_part(block) {
            content.push(part);
        }
    }

    match vision.upstream_type {
        crate::forward::UpstreamType::AnthropicMessages => json!({
            "model": vision.model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": content
            }]
        }),
        _ => json!({
            "model": vision.model,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "stream": false
        }),
    }
}

/// Convert an image block to a Chat/Anthropic content part for the vision call.
fn image_to_content_part(block: &Value) -> Option<Value> {
    let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
    match bt {
        "image" => {
            let source = block.get("source").cloned().unwrap_or(json!({}));
            let source_type = source.get("type").and_then(Value::as_str).unwrap_or("");
            if source_type == "base64" {
                let media_type = source
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/jpeg");
                let data = source.get("data").and_then(Value::as_str).unwrap_or("");
                Some(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{media_type};base64,{data}")
                    }
                }))
            } else if source_type == "url" {
                Some(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": source.get("url").and_then(Value::as_str).unwrap_or("")
                    }
                }))
            } else {
                None
            }
        }
        "image_url" => Some(block.clone()),
        "input_image" => {
            // Responses input_image → image_url for the vision call.
            if let Some(url) = block.get("image_url").and_then(Value::as_str) {
                Some(json!({ "type": "image_url", "image_url": { "url": url } }))
            } else if let Some(data) = block.get("data").and_then(Value::as_str) {
                let media_type = block
                    .get("media_type")
                    .and_then(Value::as_str)
                    .unwrap_or("image/jpeg");
                Some(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{media_type};base64,{data}")
                    }
                }))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Call the vision model (non-streaming) and return the description text.
async fn call_vision(
    request: Value,
    state: &crate::forward::AppState,
    vision: &VisionConfig,
) -> Result<String, Error> {
    let target = UpstreamTarget::from(vision);
    let response = match crate::forward::forward_to_zen(request, &state.client, &target).await {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // Extract text from the vision response.
    match vision.upstream_type {
        crate::forward::UpstreamType::AnthropicMessages => {
            let content = response.get("content").and_then(Value::as_array);
            if let Some(blocks) = content {
                let text: Vec<String> = blocks
                    .iter()
                    .filter_map(|b| b.get("text").and_then(Value::as_str).map(str::to_string))
                    .collect();
                if !text.is_empty() {
                    return Ok(text.join("\n"));
                }
            }
            Err(Error::Transform("Vision model returned no text".into()))
        }
        _ => {
            let text = response
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .map(str::to_string)
                .or_else(|| {
                    response
                        .pointer("/choices/0/message/content/0/text")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                });
            text.ok_or_else(|| Error::Transform("Vision model returned no text".into()))
        }
    }
}

/// Replace image blocks in the body with the description text (or the
/// placeholder on failure), per entry structure.
fn replace_images_in_body(
    body: &mut Value,
    entry: LocalEntry,
    images: &[Value],
    descriptions: &[Option<String>],
    handled: &mut usize,
) {
    let container_keys: Vec<&str> = match entry {
        LocalEntry::AnthropicMessages | LocalEntry::OaiChat => vec!["messages"],
        LocalEntry::OaiResponses => vec!["input"],
    };

    for key in container_keys {
        let Some(messages) = body.get_mut(key).and_then(Value::as_array_mut) else {
            continue;
        };
        let mut img_idx = 0;
        for msg in messages.iter_mut() {
            let Some(blocks) = msg.get_mut("content").and_then(Value::as_array_mut) else {
                continue;
            };
            for block in blocks.iter_mut() {
                let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
                if bt != "image" && bt != "image_url" && bt != "input_image" {
                    continue;
                }
                if img_idx >= images.len() {
                    continue;
                }
                let replacement_text = descriptions
                    .get(img_idx)
                    .and_then(|d| d.clone())
                    .unwrap_or_else(|| crate::media_sanitizer::UNSUPPORTED_IMAGE_MARKER.to_string());
                *block = json!({ "type": "text", "text": replacement_text });
                img_idx += 1;
                *handled += 1;
            }
        }
    }
}
