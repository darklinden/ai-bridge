# ai-bridge — Implementation

How each mechanism actually works, and what must stay true while changing it. Organised by mechanism, not by module: a mechanism that spans three files is described in one place, so moving code around does not invalidate this document. Requirements are in `requirements.md`; module boundaries are in `architecture.md`.

## 1. The life of a request

A `/v1/messages` request served by an `oai-chat` upstream:

1. **Route** — `server.rs` picks `handle_messages` and dispatches on the profile's `upstream_type`.
2. **Guard** — the inbound `auth_key` is checked; a mismatch returns 401 in the Anthropic shape.
3. **Override** — the body's `model` is replaced with the profile's `model`.
4. **Preprocess media** — if the request carries images and the profile has vision enabled, image blocks are replaced by descriptions (see §4).
5. **Convert the request** — the Anthropic body becomes an OpenAI Chat body.
6. **Stamp effort** — `forward::apply_reasoning_policy` writes the configured effort value; only *presence* was decided in step 5.
7. **Forward** — `forward_to_*` performs the outbound HTTP call.
8. **Convert the response** — non-streaming bodies are translated in place; SSE streams are translated frame by frame as they arrive, with streamed text appended onto the open log line.
9. **On failure** — the upstream's status and message are relayed inside the local entry's error envelope.

The same-format case short-circuits steps 5 and 8: `handle_passthrough` performs no conversion, so the response is re-emitted byte for byte (FR-4).

**Log destinations.** The log stream is written to stdout by default; `--log <path>` tees the same lines into an appended file (ADR-0012). Both destinations are fed from the same writer, so the ordering in the file matches the terminal. The file is opened once at startup — an unopenable path fails the launch — and the proxy never rotates it. Because the request log bypasses `tracing` (it appends partial lines, which the line-oriented subscriber cannot do), the tee is the log's own writer, not a second subscriber.

## 2. Reasoning bridge

OpenAI `reasoning` items have no Anthropic counterpart, so the mechanism is *transport, not translation*: the whole item is base64-encoded into the Anthropic `signature` payload under `OPENAI_REASONING_ITEM_PREFIX` and decoded when the conversation comes back. A bridged turn therefore loses nothing, and an Anthropic client that never used reasoning never sees the prefix.

The outbound **effort** value is deliberately not part of this mechanism. It comes only from `[reasoning] effort` (default `max`) and is stamped by `apply_reasoning_policy` on every pipeline, pass-through included. Conversions decide only whether reasoning was requested at all; a thinking-disabled request must never gain an effort field. There is no model-name logic anywhere in this path (ADR-0011).

**Failure mode to watch**: treating effort as a conversion concern. It is not — it is a policy applied after conversion, which is why it cannot drift per model.

## 3. Media inside tool results

Tool results carry structured media blocks in Anthropic and Responses, but only plain text in Chat. The mechanism lifts the media out of the tool result and re-emits it as a synthetic user message, which is how a Chat-shaped upstream can still be shown the image:

- **Chat direction** — `plan_chat_tool_output_media` decides what can be lifted, `queue_chat_tool_output_media` accumulates it, `flush_pending_chat_tool_media` emits it as a user message.
- **Responses direction** — `strip_and_clamp_media_from_tool_value` removes media that the tool-result shape cannot hold.

**Failure mode to watch**: emitting the media *before* the tool result, which reorders the conversation the upstream sees.

## 4. Image description

Three pieces, in order:

1. **Decide whether the upstream can see images** — `media_sanitizer::is_confirmed_text_only_model` consults a model-name list. It is a heuristic, which is exactly why the feature below is opt-in: a vision-capable model missing from the list would otherwise have its images silently replaced (ADR-0004).
2. **Describe them** — with `vision_supplement` on and a complete `[vision]` table, `vision.rs` makes one non-streaming call for all images in the request jointly. Prompt selection: a blank `prompt` falls back to the `prompt_mode` template (`auto` / `general` / `ui` / `compact`); `max_tokens` caps the output. Each image is labelled by its true document index.
3. **Cache** — results are cached in-process keyed by image fingerprint with a TTL, so a repeated image costs nothing.

**Failure mode to watch**: a vision error must degrade to `[Unsupported Image]` and let the request through. Turning a description failure into a request failure would make an optional feature load-bearing.

## 5. Byte-stable payloads for prompt caching

Where a tool-result payload is semantically identical across turns, it must also be byte-identical, or the upstream's prompt cache misses. `json_canonical::canonical_json_string` produces that canonical form and is the only serialiser to use on cache-sensitive payloads. Key order is part of the payload's identity — do not reorder keys "for tidiness" (FR-13).

**Failure mode to watch**: reaching for `serde_json::to_string` on a map-derived value, whose iteration order is not the canonical one. The payload still parses; only the cache silently suffers.

## 6. Error mapping

Every error carries enough to be rendered in the caller's own format:

- **Status** — `Transform` and `Forward` → 502, `Config` and `Server` → 500, `Unauthorized` → 401, `Unsupported` → 400. A failure relayed from the upstream keeps *the upstream's* status instead.
- **Envelope** — `Error::into_entry_response` renders the local entry's shape: Anthropic gets `{"type":"error","error":{"type":…,"message":…}}` with `type` from Anthropic's vocabulary (`invalid_request_error` / `authentication_error` / `api_error`), OpenAI gets `{"error":{"message":…,"type":…}}`. One shaping function per entry owns that shape (ADR-0010).
- **Message** — for a relayed upstream failure, the upstream's own reason text, not a proxy paraphrase.

**Failure mode to watch**: a second copy of the Anthropic envelope. Shape it in one place only — a duplicate site is what lets the envelope drift out of the entry's own vocabulary.

## 7. Invariants

Each invariant states what must hold, how it gets broken, and what currently guards it.

| | Invariant | How it breaks | Guard |
|---|---|---|---|
| INV-1 | Wire formats are declared, never inferred (main upstream and `[vision]`) | adding a URL heuristic "just for this one case" | required `upstream_type`, no URL branch (ADR-0002, ADR-0004) |
| INV-2 | Configuration is TOML profiles; the only env var read is `RUST_LOG` | moving a setting back into the environment, where nothing validates it | `deny_unknown_fields`; env reads are greppable (ADR-0005) |
| INV-3 | Outbound effort comes only from `[reasoning] effort`; no per-model logic | mapping effort from the model name | single stamping point, `apply_reasoning_policy` (ADR-0011) |
| INV-4 | The client's `model` is always overridden | honouring the client's model for "compatibility" | override in every entry (ADR-0013) |
| INV-5 | The error envelope follows the local entry, from one shaping function | adding a second envelope site | `into_entry_response`, per-entry tests (ADR-0010) |
| INV-6 | Lossy conversion is graded by meaning: map / drop+WARN / 400 | blanket-dropping, or 400-ing on inert fields | the three tiers in ADR-0009 |
| INV-7 | Upstream status and reason are relayed, not flattened to 502 | mapping every upstream failure to 502 | `Error::Upstream` carries status and reason |
| INV-8 | Cache-sensitive payloads are byte-stable | `serde_json::to_string` on a map value | `canonical_json_string` (FR-13) |
| INV-9 | Serving outranks bookkeeping: persistence failures WARN and continue | making a settings write fatal | best-effort save in `run_server` (ADR-0006) |
| INV-10 | A missing profile is a loud error, never a silent fallback | defaulting to another profile "to be helpful" | `validate_profile_name` + `load_profile` (ADR-0007) |
| INV-11 | Inbound `auth_key` and upstream `api_key` stay separate credentials | conflating them into one key | two distinct config keys (ADR-0008) |
| INV-12 | The log observes, it never alters what is forwarded | logging by rewriting a stream instead of teeing it | `passthrough_log` re-yields original chunks (FR-20) |
| INV-13 | Conversion changes ship with tests | landing a conversion with manual testing only | inline `#[cfg(test)]` convention (FR-9, FR-10) |

## 8. Known deviations and open items

Deviations of the code from the requirements above, and questions still open. IDs are stable so they can be referenced from commits. This list is the successor to the earlier review findings; items that turned out to be *requirements* changes rather than code defects are not here.

| ID | Deviation | Requirement | Notes |
|---|---|---|---|
| DEV-1 | The Anthropic envelope is malformed for every `/v1/messages` failure: no top-level `"type"`, `"error"` as the inner type | FR-17 | a real defect; the correct shape already exists at a second site that must be merged into one |
| DEV-2 | On the Responses → Anthropic path, `store` / `include` / `metadata` / `truncation` / `text.format` are dropped silently | FR-10 | inert fields must be dropped *with a WARN* |
| DEV-3 | `[vision]` has no `upstream_type` key and its format is guessed from the URL | FR-5, FR-15 | ADR-0002 and ADR-0004 require an explicit key instead |
| DEV-4 | The outbound token-cap field name is chosen by a model-name list (`is_openai_o_series`) | FR-7 | replace with `[chat] max_tokens_field` |
| DEV-5 | `forward_to_zen` / `forward_to_zen_streaming` name the upstream "zen" | vocabulary | the glossary's term is *Upstream* |
| DEV-6 | `HOME` is read for path expansion but is not registered as an exception | FR-6 | either register it in the docs or stop reading it |
| DEV-7 | The SSE field/block parsing helpers exist twice, near-verbatim | — | two copies drift independently |
| DEV-8 | The three conversion handlers repeat the same preamble | — | routing plumbing is duplicated per entry |
| DEV-9 | Several conversion functions are very long | — | the SSE translator is the largest |
| DEV-10 | `match upstream_type` recurs across the handlers and the forward module | — | one table would serve all sites |
| DEV-11 | `convert.rs` and `server.rs` change for many unrelated reasons | — | the largest files also carry the most responsibilities |
| DEV-12 | The request log records a text *summary*, not the content it forwarded — the `[REQ #id]` line is built from an extracted `{model} {text}` string | FR-19 | non-text payloads, base64 included, never reach the log at all; ADR-0012 requires the content verbatim |
| DEV-13 | There is no `--log` option; the CLI parser accepts only `--list`, `--help` and a profile name, and the log can only be redirected from the shell | FR-19 | `run.sh` exists precisely to work around this, by teeing the whole process to a dated file |

**Open**

| ID | Question | Why it is open |
|---|---|---|
| OPEN-1 | Should the `[REQ SYSTEM]` de-duplication be global or per-profile? | it currently suppresses a repeated system prompt across *all* requests, so two clients with different prompts can hide each other's line |
