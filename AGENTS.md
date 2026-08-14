# AGENTS.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`ai-bridge` is a Rust HTTP proxy that lets local Anthropic Messages, OpenAI Chat Completions, and OpenAI Responses clients talk to one configured upstream API. Requests are translated between the local entry format and the upstream format, forwarded, and the response is translated back; every request/response pair is logged to stdout as a single-line stream. Upstream format is declared explicitly via `UPSTREAM_TYPE`.

## Commands

```bash
cargo build          # build
cargo run            # run the server (needs UPSTREAM_TYPE + UPSTREAM_URL + UPSTREAM_API_KEY)
cargo test           # run all unit tests
cargo test <module>            # run one module's tests, e.g. cargo test reasoning_bridge
cargo test <module>::<test>    # run a single test, e.g. cargo test forward::parses_simple_pairs
```

Tests are inline `#[cfg(test)] mod tests` — there is no `tests/` integration directory. Module with the most coverage: `transform_responses` (18), `convert_reverse` (8), `responses_reverse` (5), `tool_media` (14), `media_sanitizer` (10), `reasoning_bridge` (9). `streaming_responses` and `server` have no unit tests and are verified manually / via end-to-end.

## Architecture

The service is a **single binary** (`ai-bridge`, no lib.rs) with **three local entries**, each forwarding to one configured upstream (format declared by `UPSTREAM_TYPE`). Conversions happen on request and on response, in both directions.

```
local entry → upstream
  POST /v1/messages          (Anthropic Messages)
  POST /v1/chat/completions  (OpenAI Chat)
  POST /v1/responses         (OpenAI Responses)
       │
       └─ upstream = oai-chat | oai-responses | anthropic-messages
            request conversion → forward → response conversion (mirrored back)
```

Three layers, in dependency order:

- **Entry/route layer** (`server.rs`, `main.rs`) — three handlers (`handle_messages`, `handle_chat_completions_local`, `handle_responses_local`), each a small `match` over `UpstreamType` dispatching to an entry × upstream pipeline. Common plumbing: inbound auth via `UPSTREAM_AUTH_KEY`, unconditional model override to `UPSTREAM_MODEL`, media preprocessing, and error responses shaped to the *local entry* format (`Error::into_entry_response`). `forward.rs` owns config parsing (`Config::from_env`) and outbound HTTP (`forward_to_*`). `error.rs` maps errors to status (Transform/Forward→502, Config/Server→500, Unauthorized→401, Unsupported→400).
- **Conversion layer** — six existing functions cover the "Anthropic entry → OpenAI upstream" direction: `convert.rs` (Chat request/response/SSE), `transform_responses.rs` + `streaming_responses.rs` (Responses). Six **reverse** functions cover "OpenAI entry → Anthropic upstream": `convert_reverse.rs` (Chat request/response/SSE) and `responses_reverse.rs` (Responses request/response/SSE). Cross-format paths (e.g. Chat entry → Responses upstream) double-bridge through the Anthropic intermediate.
- **Cross-cutting mechanisms** — shared concerns centralized because both channels need them:
  - `reasoning_bridge.rs` — OpenAI `reasoning` items have no Anthropic field, so the full item is base64-encoded into the Anthropic `signature`/redacted-thinking payload and decoded on replay (`OPENAI_REASONING_ITEM_PREFIX`).
  - `tool_media.rs` — tool results carry structured media blocks in Anthropic/Responses but plain text in Chat; media is extracted and re-emitted as a synthetic user message (`plan_chat_tool_output_media` / `queue_chat_tool_output_media` / `flush_pending_chat_tool_media` for Chat, `strip_and_clamp_media_from_tool_value` for Responses).
  - `media_sanitizer.rs` — strips images from the request before forwarding when the upstream model is confirmed text-only (`is_confirmed_text_only_model`).
  - `vision.rs` — when a text-only upstream receives images and a `VISION_*` config exists, describes the images via the vision model (multi-image joint call, non-streaming, in-process fingerprint TTL cache). Prompt selection: `VISION_PROMPT` custom > `VISION_PROMPT_MODE` template (`auto`/`general`/`ui`/`compact`); `VISION_MAX_TOKENS` caps output; each image is labeled by true document index. Vision failure degrades to `[Unsupported Image]` and does not block the request.
  - `json_canonical.rs` — canonical JSON serialization (`canonical_json_string`) so tool-result payloads byte-match for prompt-cache stability.
  - `reqlog.rs` — always-on stdout logger, one `[REQ #id]` / `[RESP #id]` line per request (each preceded by two blank lines), independent of `RUST_LOG`.

## Key invariants to preserve

- **Upstream format is explicit** — declared by `UPSTREAM_TYPE` (required, one of `anthropic-messages | oai-chat | oai-responses`), never inferred from the URL. Keep it that way; Anthropic URLs contain no `/responses` marker.
- **Config naming is `UPSTREAM_*`** — `UPSTREAM_URL` / `UPSTREAM_API_KEY` / `UPSTREAM_MODEL` / `UPSTREAM_AUTH_KEY` / `UPSTREAM_HEADERS`, plus optional `VISION_*`. The legacy `OPENAI_COMP_*` names are gone and rejected with a config error; do not reintroduce them.
- **Model is always overridden** to `UPSTREAM_MODEL` — client-supplied `model` is ignored on all local entries.
- **Anthropic upstream auth** uses `x-api-key` + `anthropic-version: 2023-06-01`; OpenAI upstreams use `Authorization: Bearer`.
- **Error format follows the local entry**, not the upstream.
- **OpenAI-only fields that Anthropic cannot represent** are dropped with a WARN; `n > 1` is rejected (400) because Anthropic is single-output; `parallel_tool_calls=false` maps to `tool_choice.disable_parallel_tool_use`.
- **Prompt-cache stability matters** — tool-result payloads that are byte-identical reuse upstream prompt caches. Use `json_canonical::canonical_json_string` for cache-sensitive payloads; do not reorder keys casually.
- **Add tests alongside conversion changes** — `streaming_responses` has none and is the riskiest untested surface; conversion modules expect inline `#[cfg(test)]` tests.

## Terminology

See `CONTEXT.md` for the glossary (Upstream, Local entry, Request log, Media description). Architectural decisions live in `docs/adr/0001-0004`.
