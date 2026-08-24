//! Image description for text-only upstreams.
//!
//! When a local request carries images and the upstream model is confirmed
//! text-only (e.g. DeepSeek), a separately configured vision model (the
//! `[vision]` table) describes the images and the description text replaces
//! the image blocks before forwarding. This mirrors Plugin-Deepseek-Vision's
//! approach, but with our own config surface instead of a plugin host.
//!
//! Behavior:
//! - All images in one request are analyzed together in a single non-streaming
//!   vision call (multi-image joint analysis).
//! - Prompt selection: `prompt` (custom) wins, else `prompt_mode`
//!   selects a built-in template (`auto` | `general` | `ui` | `compact`,
//!   default `auto`); `max_tokens` caps the vision output.
//! - Each image is labeled by its true 1-based document index (`[Image N]`)
//!   with an injected text part before its image block, so partial-cache calls
//!   keep the correct mapping.
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
    // Collect image blocks from the entry structure. Each slot is normalized to
    // a Chat `image_url` part so fingerprints are stable across string vs object
    // representations of the same image (cache interop), and so the vision call
    // always receives a known shape.
    let (images, fingerprints) = collect_image_slots(body, entry);

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
        let vision_request = build_vision_request(&uncached, vision);
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

    // Replace images in the body, walking the same traversal as collection so
    // each slot maps to its description by document-order index.
    Ok(replace_images_with_descriptions(body, entry, &descriptions))
}

/// Collect every image slot in the body as a normalized Chat `image_url` part
/// plus its cache fingerprint, in document order.
fn collect_image_slots(body: &Value, entry: LocalEntry) -> (Vec<Value>, Vec<Option<u64>>) {
    let mut body_mut = body.clone();
    let mut images: Vec<Value> = Vec::new();
    let mut fingerprints: Vec<Option<u64>> = Vec::new();
    {
        let mut collect = |slot: &mut Value| {
            let normalized = slot
                .as_str()
                .and_then(crate::tool_media::whole_string_image_data_url)
                .or_else(|| image_slot_part(slot));
            let Some(part) = normalized else {
                return;
            };
            images.push(part.clone());
            fingerprints.push(image_fingerprint(&part));
        };
        visit_images_in_body(&mut body_mut, entry, &mut collect);
    }
    (images, fingerprints)
}

/// Replace every image slot in the body with its description text (or the
/// `[Unsupported Image]` placeholder when a description is absent), walking the
/// same traversal as collection. Returns the number of slots replaced.
fn replace_images_with_descriptions(
    body: &mut Value,
    entry: LocalEntry,
    descriptions: &[Option<String>],
) -> usize {
    let mut handled = 0;
    {
        let mut img_idx = 0;
        let mut replace = |slot: &mut Value| {
            let replacement_text = descriptions
                .get(img_idx)
                .and_then(|d| d.clone())
                .unwrap_or_else(|| crate::media_sanitizer::UNSUPPORTED_IMAGE_MARKER.to_string());
            // Write back in the container's own shape: string slots stay strings
            // (so `convert` sees a plain-text tool result), objects become text
            // content blocks.
            if slot.is_string() {
                *slot = Value::String(replacement_text);
            } else {
                *slot = json!({ "type": "text", "text": replacement_text });
            }
            img_idx += 1;
            handled += 1;
        };
        visit_images_in_body(body, entry, &mut replace);
    }
    handled
}

/// Traverse every image slot in the body in document order — top-level content
/// blocks plus images nested inside `tool_result` content (including
/// stringified-JSON and whole-string data URLs) — calling `visit` on each slot.
/// Returns the number of slots visited.
///
/// Collect and replace use the *same* traversal so image indices stay aligned.
/// The recursion mirrors `tool_media::strip_media_from_tool_value_at_depth`,
/// which is what `convert` uses to extract media from tool results: a slot we
/// skip here would otherwise be re-extracted as an `image_url` on conversion.
fn visit_images_in_body(
    body: &mut Value,
    entry: LocalEntry,
    visit: &mut dyn FnMut(&mut Value),
) -> usize {
    let key = match entry {
        LocalEntry::AnthropicMessages | LocalEntry::OaiChat => "messages",
        LocalEntry::OaiResponses => "input",
    };
    let Some(messages) = body.get_mut(key).and_then(Value::as_array_mut) else {
        return 0;
    };

    let mut visited = 0;
    for msg in messages.iter_mut() {
        let Some(blocks) = msg.get_mut("content").and_then(Value::as_array_mut) else {
            continue;
        };
        for block in blocks.iter_mut() {
            let bt = block.get("type").and_then(Value::as_str).unwrap_or("");
            if bt == "tool_result" {
                // Enter tool-result content at depth 0, mirroring where `convert`
                // roots its own extraction.
                if let Some(content) = block.get_mut("content") {
                    visited += visit_images_in_value(content, visit, 0);
                }
            } else {
                visited += visit_images_in_value(block, visit, 0);
            }
        }
    }
    visited
}

/// Recurse into one tool-output value, visiting image slots. Mirrors
/// `tool_media::strip_media_from_tool_value_at_depth` so the slot set here is a
/// superset of what `convert` will extract (nothing leaks through).
fn visit_images_in_value(
    value: &mut Value,
    visit: &mut dyn FnMut(&mut Value),
    depth: usize,
) -> usize {
    if depth > crate::tool_media::MAX_MEDIA_TRAVERSAL_DEPTH {
        return 0;
    }

    let mut visited = 0;
    match value {
        Value::String(text) => {
            // Whole-string data URL first (with the shared 8 KB threshold);
            // only then try parsing as JSON. Order matters: the same decision
            // order as tool_media keeps the 8 KB boundary aligned.
            if crate::tool_media::whole_string_image_data_url(text).is_some() {
                visit(value);
                return 1;
            }

            let trimmed = text.trim();
            if trimmed.is_empty() {
                return 0;
            }
            let Ok(mut parsed) = serde_json::from_str::<Value>(trimmed) else {
                return 0;
            };
            let replaced = visit_images_in_value(&mut parsed, visit, depth + 1);
            // Only re-serialize when something changed, so no-image strings stay
            // byte-for-byte stable.
            if replaced > 0 {
                *text = crate::json_canonical::canonical_json_string(&parsed);
            }
            replaced
        }
        Value::Array(items) => {
            for item in items.iter_mut() {
                visited += visit_images_in_value(item, visit, depth + 1);
            }
            visited
        }
        Value::Object(_) => {
            if image_slot_part(value).is_some() {
                visit(value);
                return 1;
            }
            // Only descend the `content` field of an object, mirroring
            // tool_media's object arm.
            if let Some(content) = value.get_mut("content") {
                visited += visit_images_in_value(content, visit, depth + 1);
            }
            visited
        }
        _ => 0,
    }
}

/// Recognize an image slot in a tool-output value and normalize it to a Chat
/// `image_url` content part. Combines the exact recognizer `convert` uses
/// (`chat_media_part_from_tool_part`, which also catches the type-less loose
/// `{"image_url": "data:..."}` object) with the existing typed-image fallback
/// (e.g. `input_image` carrying `data`/`media_type` with no `image_url` field).
fn image_slot_part(value: &Value) -> Option<Value> {
    crate::tool_media::chat_media_part_from_tool_part(
        value,
        crate::tool_media::ToolMediaScope::ImagesOnly,
    )
    .or_else(|| image_to_content_part(value))
}

// Description prompt templates, adapted from the `vision-skill` read-image
// script. All are multi-image aware: the vision call labels each image with a
// `[Image N]` text part (see build_vision_request), and the model is told to
// open every per-image section with the same `[Image N]` heading so the text
// stays mappable to document positions. The "keep the original language"
// clause refers to transcribed on-screen text, not the prompt itself.
//
// `auto` is the default: it lets the vision model self-route each image (UI
// screenshot → UI structure, otherwise → General structure) in a single call.

const PROMPT_AUTO: &str = r#"You are a rigorous, exhaustive image description engine. Each image below will be replaced into a text-only LLM that cannot see the original — your output is that model's entire knowledge of the image, so it must be complete and faithful.

Every image below is preceded by a label such as [Image N]. For EACH image, output a Markdown section that begins with the SAME [Image N] heading.

First judge the image type, then use the matching structure:
- If it is a UI screenshot (web page, app, desktop software, terminal, etc.) → use the UI Reconstruction structure;
- Otherwise (photo, document, table, chart, illustration, diagram, etc.) → use the General Description structure.

UI Reconstruction:
1. Layout structure: overall composition, grid/alignment, light/dark theme;
2. Component inventory: hierarchical list of navigation/cards/buttons/inputs/labels/lists/modals, noting relative position;
3. Text: transcribe all visible text verbatim (keep the original language) and state which component it belongs to;
4. Visual state: color scheme, selection/disabled/warning states, icon semantics;
5. Interaction hints: clickable regions, form fields, button behavior (when determinable);
6. Uncertainty: honestly describe any blurry or ambiguous regions.

General Description:
1. Overview: image type, subject, background, aspect ratio, dominant colors, light/dark;
2. Text content: if text is visible, transcribe it verbatim (keep the original language) and note its position;
3. Layout and space: describe element organization, grouping, alignment, nesting, top to bottom and left to right;
4. Visual details: icons, shapes, numbers, color meanings, status indicators, highlights;
5. Summary: one or two sentences capturing the core information.

Common requirements: output in Markdown; wrap text/keywords in backticks; if something cannot be made out, honestly write "cannot make out" — never fabricate or guess; write more rather than less. When there are multiple images, after the per-image sections, briefly note how the images relate to one another."#;

const PROMPT_UI: &str = r#"These are UI interface screenshots. Generate a UI reconstruction guide for each so a text-only AI can understand the interface without seeing the original image.

Every image below is preceded by a label such as [Image N]. For EACH image, output a Markdown section that begins with the SAME [Image N] heading:

1. Layout structure: overall composition, grid and alignment, light/dark theme;
2. Component inventory: hierarchical list of navigation/cards/buttons/inputs/labels/lists/modals, noting relative position;
3. Text: transcribe all visible text verbatim (keep the original language), wrapped in backticks, and state the component it belongs to;
4. Visual state: color scheme, light/dark theme, selected/disabled/warning states, icon semantics;
5. Interaction hints: clickable regions, form fields, button behavior (when determinable);
6. Uncertainty: honestly describe any blurry or ambiguous regions.

Goal: faithful reconstruction — do not infer, do not embellish. When there are multiple screenshots, also briefly note how they relate to one another (e.g. same flow, before/after, parent/child)."#;

const PROMPT_GENERAL: &str = r#"You are a rigorous, exhaustive image description engine. Each image below will be replaced into a text-only LLM that cannot see the original — your description must be complete and faithful.

Every image below is preceded by a label such as [Image N]. For EACH image, output a Markdown section that begins with the SAME [Image N] heading, using this structure:

## [Image N]
### Overview
Image type (photo/document/chart/illustration/diagram/other), subject, background, aspect ratio, dominant colors, light/dark theme.

### Text content
If visible text exists, transcribe it verbatim (keep the original language), wrapped in backticks, and note its position (e.g. "top-center title: `xxx`").

### Layout and spatial relationships
Describe element organization from top to bottom and left to right; note grouping, alignment, nesting.

### Visual details
Icons, shapes, numbers, color meanings, status indicators, highlights, etc.

### Summary
One or two sentences capturing the core information.

Requirements: highlight key information with backticks; honestly write "cannot make out" for anything blurry or unreadable — never fabricate; write more rather than less. When there are multiple images, also briefly note how they relate to one another."#;

const PROMPT_COMPACT: &str = r#"Describe each image in detail for another AI that cannot see the original.

Every image below is preceded by a label such as [Image N]. For EACH image, output a Markdown section that begins with the SAME [Image N] heading:
1) Transcribe all visible text verbatim (keep the original language); 2) describe the layout and the position of each element; 3) describe icons, colors, numbers and other details;
4) if something cannot be made out, write "cannot make out" — do not fabricate.
Use Markdown bullet points; the more information the better. When there are multiple images, also briefly note how they relate to one another."#;

/// Choose the description prompt: custom (`prompt`) wins, else the
/// mode template (`prompt_mode`), else the auto-routing default.
fn select_prompt(vision: &VisionConfig) -> &str {
    if let Some(custom) = vision.custom_prompt.as_deref() {
        return custom;
    }
    match vision.prompt_mode.as_str() {
        "general" => PROMPT_GENERAL,
        "ui" => PROMPT_UI,
        "compact" => PROMPT_COMPACT,
        _ => PROMPT_AUTO,
    }
}

/// Build a joint vision request describing all images at once.
fn build_vision_request(
    uncached: &[(usize, u64, &Value)],
    vision: &VisionConfig,
) -> Value {
    let prompt = select_prompt(vision);
    // Prompt text + a `[Image N]` label text part before each image, so the
    // model numbers every section by its TRUE 1-based document position. This
    // stays correct when part of a request was served from cache and the
    // uncached indices are non-contiguous.
    let mut content: Vec<Value> = Vec::with_capacity(uncached.len() * 2 + 1);
    content.push(json!({ "type": "text", "text": prompt }));
    for (i, _, block) in uncached {
        if let Some(part) = image_to_content_part(block) {
            content.push(json!({ "type": "text", "text": format!("[Image {}]", i + 1) }));
            content.push(part);
        }
    }

    match vision.upstream_type {
        crate::forward::UpstreamType::AnthropicMessages => json!({
            "model": vision.model,
            "max_tokens": vision.max_tokens.unwrap_or(1024),
            "messages": [{
                "role": "user",
                "content": content
            }]
        }),
        _ => {
            // Chat shape for both OaiChat and OaiResponses (existing behavior,
            // intentionally unchanged — including the field name: this arm
            // speaks Chat, so the token cap is `max_tokens`, not
            // `max_output_tokens`). Emitted only when configured.
            let mut request = json!({
                "model": vision.model,
                "messages": [{
                    "role": "user",
                    "content": content
                }],
                "stream": false
            });
            if let Some(limit) = vision.max_tokens {
                request["max_tokens"] = json!(limit);
            }
            request
        }
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


#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::UpstreamType;
    use crate::tool_media::plan_chat_tool_output_media;

    fn image_block() -> Value {
        json!({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": "abc"}
        })
    }

    fn image_url_object(url: &str) -> Value {
        json!({ "type": "image_url", "image_url": { "url": url } })
    }

    fn descs(n: usize) -> Vec<Option<String>> {
        (0..n).map(|i| Some(format!("description {i}"))).collect()
    }

    fn big_data_url() -> String {
        format!("data:image/png;base64,{}", "A".repeat(9000))
    }

    fn anthropic_body(content: Value) -> Value {
        json!({ "model": "claude-sonnet-5", "messages": [{"role": "user", "content": content}] })
    }

    fn vision_config(
        ut: UpstreamType,
        mode: &str,
        custom: Option<&str>,
        max_tokens: Option<u32>,
    ) -> VisionConfig {
        VisionConfig {
            url: "http://vision".into(),
            api_key: "k".into(),
            model: "m".into(),
            upstream_type: ut,
            override_headers: http::HeaderMap::new(),
            prompt_mode: mode.into(),
            custom_prompt: custom.map(str::to_string),
            max_tokens,
        }
    }

    fn image_url_part(url: &str) -> Value {
        json!({ "type": "image_url", "image_url": { "url": url } })
    }

    /// Build a request and return (request, first-message content array).
    fn request_for(
        uncached: &[(usize, u64, &Value)],
        vision: &VisionConfig,
    ) -> (Value, Vec<Value>) {
        let req = build_vision_request(uncached, vision);
        let content = req["messages"][0]["content"].as_array().unwrap().clone();
        (req, content)
    }

    #[test]
    fn collects_and_replaces_tool_result_array_image() {
        let mut body = anthropic_body(json!([
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": [{"type": "text", "text": "screenshot:"}, image_block()]}
        ]));

        let (images, fps) = collect_image_slots(&body, LocalEntry::AnthropicMessages);
        assert_eq!(images.len(), 1);
        assert_eq!(fps.len(), 1);
        assert!(fps[0].is_some());
        assert_eq!(images[0]["type"], "image_url");

        let handled = replace_images_with_descriptions(&mut body, LocalEntry::AnthropicMessages, &descs(1));
        assert_eq!(handled, 1);

        let content = &body["messages"][0]["content"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "text");
        assert_eq!(content[1]["text"], "description 0");
        // convert must no longer see any media to extract.
        assert!(plan_chat_tool_output_media(content.clone()).is_none());
    }

    #[test]
    fn replaces_stringified_json_tool_result_image() {
        let inner = json!({"content": [image_block()]}).to_string();
        let mut body = anthropic_body(json!([
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": inner}
        ]));

        let handled = replace_images_with_descriptions(&mut body, LocalEntry::AnthropicMessages, &descs(1));
        assert_eq!(handled, 1);

        let rewritten = body["messages"][0]["content"][0]["content"]
            .as_str()
            .unwrap()
            .to_string();
        assert!(rewritten.contains("description 0"));
        assert!(!rewritten.contains("\"image\""));
        let parsed: Value = serde_json::from_str(&rewritten).unwrap();
        assert!(plan_chat_tool_output_media(parsed).is_none());
    }

    #[test]
    fn replaces_whole_string_data_url_tool_result() {
        let url = big_data_url();
        let mut body = anthropic_body(json!([
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": url}
        ]));

        let (images, _) = collect_image_slots(&body, LocalEntry::AnthropicMessages);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0]["image_url"]["url"], big_data_url());

        let handled = replace_images_with_descriptions(&mut body, LocalEntry::AnthropicMessages, &descs(1));
        assert_eq!(handled, 1);
        let content = body["messages"][0]["content"][0]["content"].as_str().unwrap();
        assert_eq!(content, "description 0");
        assert!(!content.contains("base64"));
    }

    #[test]
    fn missing_description_falls_back_to_placeholder() {
        let mut body = anthropic_body(json!([
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": [image_block()]}
        ]));

        let handled = replace_images_with_descriptions(
            &mut body,
            LocalEntry::AnthropicMessages,
            &[None],
        );
        assert_eq!(handled, 1);
        let content = &body["messages"][0]["content"][0]["content"];
        assert_eq!(content[0]["type"], "text");
        assert_eq!(
            content[0]["text"],
            crate::media_sanitizer::UNSUPPORTED_IMAGE_MARKER
        );
    }

    #[test]
    fn indices_stay_aligned_across_messages_and_nesting() {
        // Message 0: top-level image + tool_result image. Message 1: top-level
        // image. Document order = [top0, tool0, top1] → 3 slots.
        let mut body = json!({
            "messages": [
                {"role": "user", "content": [image_block()]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1",
                     "content": [image_block()]}
                ]},
                {"role": "user", "content": [image_block()]}
            ]
        });

        let (images, _) = collect_image_slots(&body, LocalEntry::AnthropicMessages);
        assert_eq!(images.len(), 3);

        let handled = replace_images_with_descriptions(&mut body, LocalEntry::AnthropicMessages, &descs(3));
        assert_eq!(handled, 3);

        // Top-level slot 0 → description 0.
        assert_eq!(body["messages"][0]["content"][0]["text"], "description 0");
        // Tool_result slot 1 → description 1.
        assert_eq!(body["messages"][1]["content"][0]["content"][0]["text"], "description 1");
        // Top-level slot 2 → description 2.
        assert_eq!(body["messages"][2]["content"][0]["text"], "description 2");
    }

    #[test]
    fn no_images_is_byte_stable() {
        let body = anthropic_body(json!([
            {"type": "text", "text": "hello"}
        ]));
        let original = body.to_string();

        let (images, fps) = collect_image_slots(&body, LocalEntry::AnthropicMessages);
        assert!(images.is_empty());
        assert!(fps.is_empty());

        let mut mutated = body.clone();
        let handled = replace_images_with_descriptions(&mut mutated, LocalEntry::AnthropicMessages, &[]);
        assert_eq!(handled, 0);
        assert_eq!(mutated.to_string(), original);
    }

    #[test]
    fn small_string_data_url_is_left_as_text() {
        // Below the shared 8 KB threshold: neither vision nor convert treat it
        // as an image.
        let small = format!("data:image/png;base64,{}", "A".repeat(4000));
        let mut body = anthropic_body(json!([
            {"type": "tool_result", "tool_use_id": "toolu_1", "content": small}
        ]));
        let original = body.to_string();

        let (images, _) = collect_image_slots(&body, LocalEntry::AnthropicMessages);
        assert!(images.is_empty());

        let handled = replace_images_with_descriptions(&mut body, LocalEntry::AnthropicMessages, &[]);
        assert_eq!(handled, 0);
        assert_eq!(body.to_string(), original);
    }

    #[test]
    fn loose_typeless_image_url_object_is_collected() {
        // convert's recognizer catches a type-less `{"image_url": "data:..."}`
        // object; vision must too, or it leaks through as image_url.
        let body = anthropic_body(json!([
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": [json!({"image_url": big_data_url()})]}
        ]));

        let (images, _) = collect_image_slots(&body, LocalEntry::AnthropicMessages);
        assert_eq!(images.len(), 1);
        assert_eq!(images[0]["image_url"]["url"], big_data_url());
    }

    #[test]
    fn string_and_object_same_image_share_fingerprint() {
        let url = big_data_url();
        // The code normalizes string slots to a Chat image_url part before
        // fingerprinting, so the string form and the object form of the same
        // image must share a fingerprint (cache interop).
        let string_part =
            crate::tool_media::whole_string_image_data_url(&url).unwrap();
        let object_part = image_url_object(&url);

        let fp_str = image_fingerprint(&string_part);
        let fp_obj = image_fingerprint(&object_part);
        assert_eq!(fp_str, fp_obj);
        assert!(fp_str.is_some());
    }

    #[test]
    fn end_to_end_no_image_url_reaches_upstream_shape() {
        // After collect + replace, converting to OpenAI Chat must not emit any
        // image_url part nor a synthetic user message for tool media.
        let mut body = anthropic_body(json!([
            {"type": "tool_result", "tool_use_id": "toolu_1",
             "content": [{"type": "text", "text": "screenshot:"}, image_block()]}
        ]));

        let handled = replace_images_with_descriptions(&mut body, LocalEntry::AnthropicMessages, &descs(1));
        assert_eq!(handled, 1);

        let chat = crate::convert::anthropic_to_openai_with_reasoning_content(body, true).unwrap();
        let serialized = chat.to_string();
        assert!(
            !serialized.contains("image_url"),
            "image_url must not reach the upstream: {serialized}"
        );
        // No synthetic user message carrying tool media was emitted.
        let messages = chat["messages"].as_array().unwrap();
        for m in messages {
            assert_ne!(m["role"], "user", "synthetic user message leaked: {serialized}");
        }
    }

    #[test]
    fn prompt_custom_overrides_everything() {
        let img = image_url_part("data:image/png;base64,AAA");
        let cfg = vision_config(UpstreamType::OaiChat, "ui", Some("my custom prompt"), None);
        let uncached = [(0, 0, &img)];
        let (_, content) = request_for(&uncached, &cfg);
        assert_eq!(content[0]["text"], "my custom prompt");
    }

    #[test]
    fn prompt_mode_selects_template() {
        let cases = [
            ("general", PROMPT_GENERAL),
            ("ui", PROMPT_UI),
            ("compact", PROMPT_COMPACT),
            ("auto", PROMPT_AUTO),
        ];
        for (mode, expected) in cases {
            let img = image_url_part("data:image/png;base64,AAA");
            let cfg = vision_config(UpstreamType::OaiChat, mode, None, None);
            let uncached = [(0, 0, &img)];
            let (_, content) = request_for(&uncached, &cfg);
            assert_eq!(content[0]["text"], expected, "mode {mode}");
        }
    }

    #[test]
    fn prompt_unknown_mode_falls_back_to_auto() {
        let img = image_url_part("data:image/png;base64,AAA");
        let cfg = vision_config(UpstreamType::OaiChat, "bogus", None, None);
        let uncached = [(0, 0, &img)];
        let (_, content) = request_for(&uncached, &cfg);
        assert_eq!(content[0]["text"], PROMPT_AUTO);
    }

    #[test]
    fn labels_use_true_document_indices() {
        let img1 = image_url_part("data:image/png;base64,AAA");
        let img2 = image_url_part("data:image/png;base64,BBB");
        // 0-based document positions 4 and 8 → labels [Image 5] and [Image 9].
        let uncached = [(4, 1, &img1), (8, 2, &img2)];
        let cfg = vision_config(UpstreamType::OaiChat, "auto", None, None);
        let (_, content) = request_for(&uncached, &cfg);
        assert_eq!(content[1]["text"], "[Image 5]");
        assert_eq!(content[2], img1);
        assert_eq!(content[3]["text"], "[Image 9]");
        assert_eq!(content[4], img2);
    }

    #[test]
    fn labels_interleave_even_for_adjacent_images() {
        let img1 = image_url_part("data:image/png;base64,AAA");
        let img2 = image_url_part("data:image/png;base64,BBB");
        let uncached = [(0, 1, &img1), (1, 2, &img2)];
        let cfg = vision_config(UpstreamType::OaiChat, "auto", None, None);
        let (_, content) = request_for(&uncached, &cfg);
        assert_eq!(content[1]["text"], "[Image 1]");
        assert_eq!(content[2], img1);
        assert_eq!(content[3]["text"], "[Image 2]");
        assert_eq!(content[4], img2);
    }

    #[test]
    fn anthropic_max_tokens_config_or_default() {
        let img = image_url_part("data:image/png;base64,AAA");
        let uncached = [(0, 0, &img)];

        let cfg = vision_config(UpstreamType::AnthropicMessages, "auto", None, Some(2048));
        let (req, _) = request_for(&uncached, &cfg);
        assert_eq!(req["max_tokens"], 2048);
        assert!(req.get("stream").is_none(), "anthropic arm emits no stream");
        assert!(req["messages"][0]["content"].is_array());

        let cfg = vision_config(UpstreamType::AnthropicMessages, "auto", None, None);
        let (req, _) = request_for(&uncached, &cfg);
        assert_eq!(req["max_tokens"], 1024);
    }

    #[test]
    fn chat_max_tokens_only_when_configured() {
        let img = image_url_part("data:image/png;base64,AAA");
        let uncached = [(0, 0, &img)];

        // OaiChat and OaiResponses share the same `_` arm, so both behave
        // identically: Chat field `max_tokens`, emitted only when configured.
        for ut in [UpstreamType::OaiChat, UpstreamType::OaiResponses] {
            let cfg = vision_config(ut, "auto", None, Some(512));
            let (req, _) = request_for(&uncached, &cfg);
            assert_eq!(req["max_tokens"], 512);
            assert_eq!(req["stream"], false);

            let cfg = vision_config(ut, "auto", None, None);
            let (req, _) = request_for(&uncached, &cfg);
            assert!(req.get("max_tokens").is_none());
            assert_eq!(req["stream"], false);
        }
    }
}
