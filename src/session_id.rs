//! Conversation identity for upstream session headers.
//!
//! Some upstreams require every request to carry a stable per-conversation id
//! and reject calls without one — OpenCode Go answers `400 MissingSessionID`
//! before it even checks auth. The proxy is stateless and has no conversation
//! table, so the id is derived from the one thing a single request still
//! carries about the conversation's identity: its opening — the system prompt
//! and the first user message. Those stay identical while later turns append
//! messages, so the id is stable across a conversation without any storage.
//!
//! The identity is canonicalised (object key order cannot change the result)
//! and hashed with FNV-1a: a routing id needs stability and cheapness, not
//! cryptographic strength. Two conversations that open identically collapse
//! into one id; that costs some upstream routing spread, never correctness.

use crate::error::LocalEntry;
use crate::json_canonical::canonical_json_string;
use serde_json::{json, Value};

/// Prefix for derived ids. No upstream inspected so far validates the shape,
/// but a prefixed value is recognisable in a capture and in upstream logs.
const SESSION_PREFIX: &str = "ses_";

/// Derive the conversation id for an inbound request, before conversion (and
/// therefore independent of the upstream format the profile selected).
pub(crate) fn derive(entry: LocalEntry, body: &Value) -> String {
    let identity = identity_for(entry, body);
    let digest = fnv1a64(canonical_json_string(&identity).as_bytes());
    format!("{SESSION_PREFIX}{digest:016x}")
}

/// The conversation's opening, in the shape the local entry uses for it.
///
/// Anthropic Messages carries the system prompt beside `messages`; Chat has it
/// as the leading `role: "system"` message; Responses keeps it in
/// `instructions` with the conversation in `input`. When neither part is
/// present the whole body is hashed instead, so the fallback is still a
/// deterministic value rather than one shared constant.
fn identity_for(entry: LocalEntry, body: &Value) -> Value {
    let (system, first_user) = match entry {
        LocalEntry::AnthropicMessages => (
            body.get("system").cloned(),
            first_message_with_role(body.get("messages"), "user"),
        ),
        LocalEntry::OaiChat => (
            first_message_with_role(body.get("messages"), "system"),
            first_message_with_role(body.get("messages"), "user"),
        ),
        LocalEntry::OaiResponses => (
            body.get("instructions").cloned(),
            first_responses_input(body.get("input")),
        ),
    };

    match (system, first_user) {
        (None, None) => body.clone(),
        (system, first_user) => json!({ "system": system, "first_user": first_user }),
    }
}

fn first_message_with_role(messages: Option<&Value>, role: &str) -> Option<Value> {
    messages?
        .as_array()?
        .iter()
        .find(|message| message.get("role").and_then(Value::as_str) == Some(role))
        .cloned()
}

/// Responses' `input` is either a plain string (a single user turn) or an
/// array of items carrying their own `role`.
fn first_responses_input(input: Option<&Value>) -> Option<Value> {
    match input? {
        Value::String(text) => Some(Value::String(text.clone())),
        value => first_message_with_role(Some(value), "user"),
    }
}

/// FNV-1a, 64-bit: tiny, dependency-free and deterministic across builds
/// (unlike `DefaultHasher`, whose output is not a stable contract).
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::derive;
    use crate::error::LocalEntry;
    use serde_json::{json, Value};

    const ANTHROPIC: LocalEntry = LocalEntry::AnthropicMessages;
    const CHAT: LocalEntry = LocalEntry::OaiChat;
    const RESPONSES: LocalEntry = LocalEntry::OaiResponses;

    fn anthropic_body() -> Value {
        json!({
            "model": "m",
            "system": "you are a coding agent",
            "messages": [
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "looking"},
                {"role": "user", "content": "and the tests"}
            ]
        })
    }

    #[test]
    fn derived_id_is_prefixed_and_fixed_width() {
        let id = derive(ANTHROPIC, &anthropic_body());
        assert!(id.starts_with("ses_"), "{id}");
        assert_eq!(id.len(), "ses_".len() + 16, "{id}");
    }

    /// The whole mechanism exists for this: the conversation grows, the id
    /// does not change.
    #[test]
    fn id_is_stable_across_growing_conversations() {
        let mut body = anthropic_body();
        let first = derive(ANTHROPIC, &body);
        body["messages"].as_array_mut().unwrap().push(json!({
            "role": "user", "content": "thanks"
        }));
        assert_eq!(derive(ANTHROPIC, &body), first);
    }

    #[test]
    fn different_opening_means_different_id() {
        let mut other = anthropic_body();
        other["messages"][0]["content"] = json!("write a new feature");
        assert_ne!(
            derive(ANTHROPIC, &anthropic_body()),
            derive(ANTHROPIC, &other)
        );
    }

    #[test]
    fn differing_system_prompt_means_different_id() {
        let mut other = anthropic_body();
        other["system"] = json!("you are a reviewer");
        assert_ne!(
            derive(ANTHROPIC, &anthropic_body()),
            derive(ANTHROPIC, &other)
        );
    }

    /// Canonical serialisation: a client may rebuild its JSON with any key
    /// order and must still land in the same conversation.
    #[test]
    fn key_order_does_not_change_the_id() {
        let left = anthropic_body();
        let right = json!({
            "messages": [
                {"content": "fix the bug", "role": "user"},
                {"content": "looking", "role": "assistant"},
                {"content": "and the tests", "role": "user"}
            ],
            "model": "m",
            "system": "you are a coding agent"
        });
        assert_eq!(derive(ANTHROPIC, &left), derive(ANTHROPIC, &right));
    }

    #[test]
    fn chat_system_message_is_part_of_the_identity() {
        let base = json!({
            "messages": [
                {"role": "system", "content": "s1"},
                {"role": "user", "content": "hello"}
            ]
        });
        let other_system = json!({
            "messages": [
                {"role": "system", "content": "s2"},
                {"role": "user", "content": "hello"}
            ]
        });
        assert_ne!(derive(CHAT, &base), derive(CHAT, &other_system));

        let with_more_turns = json!({
            "messages": [
                {"role": "system", "content": "s1"},
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "tool", "content": "result"}
            ]
        });
        assert_eq!(derive(CHAT, &base), derive(CHAT, &with_more_turns));
    }

    /// Responses clients lead with `developer` items and nested content blocks;
    /// the id must track the first user turn and ignore the roles before it.
    #[test]
    fn responses_instructions_and_input_items_are_understood() {
        let base = json!({
            "instructions": "be terse",
            "input": [
                {"role": "developer", "content": [{"type": "input_text", "text": "env"}]},
                {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
            ]
        });
        let mut growing = base.clone();
        growing["input"].as_array_mut().unwrap().push(
            json!({"role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}),
        );
        assert_eq!(derive(RESPONSES, &base), derive(RESPONSES, &growing));

        let mut other_instructions = base.clone();
        other_instructions["instructions"] = json!("be verbose");
        assert_ne!(
            derive(RESPONSES, &base),
            derive(RESPONSES, &other_instructions)
        );

        // A plain-string `input` is a single user turn and still yields an id.
        let string_input = json!({"instructions": "be terse", "input": "hi"});
        assert!(derive(RESPONSES, &string_input).starts_with("ses_"));
    }

    /// No system prompt and no user turn (a bare probe, say): the body itself
    /// is the identity, so two identical requests still share an id.
    #[test]
    fn empty_conversation_falls_back_to_the_body() {
        let bare = json!({"model": "m", "max_tokens": 1});
        assert_eq!(derive(ANTHROPIC, &bare), derive(ANTHROPIC, &bare.clone()));
        assert_ne!(
            derive(ANTHROPIC, &bare),
            derive(ANTHROPIC, &json!({"model": "m", "max_tokens": 2}))
        );
    }
}
