# ai-bridge

A lightweight Rust HTTP proxy that lets local **Anthropic Messages**, **OpenAI Chat Completions**, and **OpenAI Responses** clients all talk to a single configured upstream API. Requests are translated from the local entry format to the upstream format, forwarded, and the response is translated back — so existing SDKs and CLIs keep working unchanged while you point them at a different backend.

Every request/response pair is also written to stdout as a single-line stream, making `ai-bridge` useful both as a gateway and as a conversation logger.

## Features

- **Three local endpoints, one upstream** — serve `/v1/messages`, `/v1/chat/completions`, and `/v1/responses` simultaneously; all forward to a single upstream chosen via `UPSTREAM_TYPE`.
- **Bidirectional format conversion** — requests and responses (non-streaming **and** SSE streaming) are converted in both directions, including cross-format paths (e.g. Chat entry → Responses upstream) via an Anthropic intermediate.
- **Explicit upstream declaration** — the upstream format is declared by `UPSTREAM_TYPE`, never guessed from the URL.
- **Always-on request logging** — a compact `[REQ #id]` / `[RESP #id]` line per request, independent of `RUST_LOG`.
- **Prompt-cache-friendly** — tool-result payloads are serialized canonically so byte-identical payloads reuse upstream prompt caches.
- **Reasoning bridging** — OpenAI `reasoning` items are preserved across Anthropic's `signature`/redacted-thinking field.
- **Tool-result media** — structured media blocks in tool results are re-emitted correctly for each entry format.
- **Vision media description** — when the upstream is a text-only model, images are either described by a configured vision model (ADR-0004) or replaced with an `[Unsupported Image]` placeholder.
- **Inbound auth & CORS** — optional token auth for local clients, permissive CORS, and a 200 MB request-body limit.

## How it works

```
local entry  →  upstream (format declared by UPSTREAM_TYPE)

  POST /v1/messages          (Anthropic Messages)
  POST /v1/chat/completions  (OpenAI Chat)
  POST /v1/responses         (OpenAI Responses)
        │
        └─ upstream = anthropic-messages | oai-chat | oai-responses
             request conversion → forward → response conversion (mirrored back)
```

When the local entry and upstream share a wire format, the request is passed through (only the model is overridden). Otherwise the request body is converted to the upstream format before forwarding, and the upstream response is converted back into the local entry's format — including live SSE stream translation.

## Requirements

- Rust (stable toolchain, edition 2021). Install with [rustup](https://rustup.rs/).

## Build & run

```bash
cargo build --release
```

Configure the environment (see below), then:

```bash
cargo run
```

You should see:

```
ai-bridge listening on 0.0.0.0:18650
  → POST /v1/messages           (Anthropic Messages)
  → POST /v1/chat/completions   (OpenAI Chat Completions)
  → POST /v1/responses          (OpenAI Responses)
  → UPSTREAM_TYPE = oai-chat
  → UPSTREAM_URL = https://api.example.com/v1/chat/completions
  ...
```

### Quick start

```bash
cp .env.example .env   # then edit to taste
cargo run
```

`.env` is loaded automatically at startup (existing exported variables take precedence; a missing `.env` is silently ignored).

## Configuration

All configuration is via environment variables (or `.env`). The upstream format is **required** and explicit — there is no URL-based auto-detection (ADR-0002).

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `UPSTREAM_TYPE` | ✅ | — | Upstream format: `anthropic-messages`, `oai-chat`, or `oai-responses`. |
| `UPSTREAM_URL` | ✅ | — | Upstream API endpoint matching `UPSTREAM_TYPE`. |
| `UPSTREAM_API_KEY` | ✅ | — | Upstream key. OpenAI-style upstreams authenticate with `Authorization: Bearer`; Anthropic with `x-api-key` + `anthropic-version`. |
| `UPSTREAM_MODEL` | | `deepseek-v4-flash` | Model forced on every request — a client-supplied `model` is always overridden. |
| `UPSTREAM_AUTH_KEY` | | *(unset)* | Optional inbound auth token for local clients. Accepted via `x-api-key` or `Authorization: Bearer`. Leave unset (not empty) to disable. |
| `UPSTREAM_HEADERS` | | *(none)* | Extra upstream headers, format `A:a\|B:b` (pipe-separated, split on first `:`). Overrides default headers; last duplicate wins. |
| `LISTEN_ADDR` | | `0.0.0.0` | Bind address. |
| `LISTEN_PORT` | | `18650` | Bind port. |
| `VISION_URL` / `VISION_API_KEY` / `VISION_MODEL` | | *(unset)* | Vision model config (all three required to enable). Endpoint format is detected from the URL (`/responses`, `/messages`, else chat). |
| `VISION_HEADERS` | | *(none)* | Extra headers for vision requests, same format as `UPSTREAM_HEADERS`. |
| `VISION_PROMPT_MODE` | | `auto` | Vision description prompt template: `auto` (self-routes UI vs general), `general`, `ui`, or `compact`. Unknown values fall back to `auto`. Ignored when `VISION_PROMPT` is set. |
| `VISION_PROMPT` | | *(none)* | Custom vision description prompt; overrides `VISION_PROMPT_MODE`. Each image is labeled `[Image N]`; mention the labels in a custom prompt. |
| `VISION_MAX_TOKENS` | | *(omitted)* | Output token cap for the vision call. Empty/0/invalid = omitted. |
| `RUST_LOG` | | `info` | `tracing` filter, e.g. `debug`, `warn`, or `ai_bridge=trace`. |

### Legacy variables are rejected

The old `OPENAI_COMP_URL` / `OPENAI_COMP_API_KEY` / `OPENAI_COMP_MODEL` / `OPENAI_COMP_AUTH_KEY` / `OPENAI_COMP_HEADERS` names were removed (ADR-0003). If any is present the process refuses to start with a migration error — rename them to the `UPSTREAM_*` equivalents.

## Endpoints

| Route | Local entry style |
| --- | --- |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |

Each accepts the corresponding SDK/CLI request shape (streaming supported), and returns the same shape you would have gotten from the upstream — errors included.

### Example

```bash
curl -X POST http://127.0.0.1:18650/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-token" \     # only if UPSTREAM_AUTH_KEY is set
  -d '{"model": "anything", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Conversion matrix

| Local entry \ Upstream | `oai-chat` | `oai-responses` | `anthropic-messages` |
| --- | --- | --- | --- |
| Anthropic Messages | direct conversion | direct conversion | pass-through |
| Chat Completions | pass-through | double-bridge via Anthropic | direct (reverse) conversion |
| Responses | double-bridge via Anthropic | pass-through | direct (reverse) conversion |

## Behavior notes

- **Model is always overridden** to `UPSTREAM_MODEL` on every local entry; a client-supplied `model` never reaches the upstream.
- **Error format follows the local entry**, not the upstream: Anthropic entries return Anthropic-style errors, OpenAI entries return OpenAI-style errors. Mapping: `Transform`/`Forward` → 502, `Config`/`Server` → 500, `Unauthorized` → 401, `Unsupported` → 400.
- **OpenAI-only fields** Anthropic cannot represent are dropped with a WARN; `n > 1` is rejected (400) because Anthropic is single-output; `parallel_tool_calls=false` maps to `tool_choice.disable_parallel_tool_use`.
- **Reasoning items** (OpenAI) have no Anthropic field, so each item is base64-encoded into the Anthropic `signature`/redacted-thinking payload and decoded on replay.
- **Tool-result media** is extracted from structured blocks and re-emitted as a synthetic user message where the target format only supports plain text.
- **Prompt-cache stability** — cache-sensitive payloads are serialized with canonical JSON so identical tool results stay byte-identical and reuse upstream caches.

## Request logging

Each request/response pair is printed to stdout as single-line records, independent of `RUST_LOG`:

```text
[REQ SYSTEM]: You are a helpful assistant...          # printed once per distinct system prompt
[REQ #0001]: claude-sonnet-5 Hello, please do something ...
[RESP #0001]: Sure! Here is the response text, streamed live onto one line...
```

- Streaming text appends onto the same open `[RESP #id]: ` line, terminated on completion.
- Errors break to a tagged `[ERR-REQ #id]` / `[ERR-RESP #id]` line.
- Each pair is preceded by two blank lines for readability.
- This is a terminal aid, not a durable log — nothing is written to disk.

## Media & vision

When the upstream model is confirmed text-only, image blocks are preprocessed before forwarding:

- If a vision model is configured (`VISION_URL` + `VISION_API_KEY` + `VISION_MODEL`), all images in the request are described in a single non-streaming vision call and the image blocks are replaced with the resulting text. Descriptions come from a built-in prompt template selected by `VISION_PROMPT_MODE` (`auto` self-routes UI screenshots to a UI-reconstruction structure and other images to a general structure), or from a custom `VISION_PROMPT`; `VISION_MAX_TOKENS` caps the description length. Each image is labeled by its document position (`[Image N]`), which stays correct when only a subset of images is uncached. Results are cached in-process keyed by image fingerprint with a TTL; a vision failure degrades to `[Unsupported Image]` and never blocks the request.
- Otherwise images are stripped and replaced with an `[Unsupported Image]` placeholder.

The request log notes handled images, e.g. `[media: 2 image(s) → [Unsupported Image]]`.

## Development

Tests are inline `#[cfg(test)]` modules (no `tests/` integration directory):

```bash
cargo test                    # all unit tests
cargo test <module>           # one module, e.g. cargo test reasoning_bridge
cargo test <module>::<test>   # a single test, e.g. cargo test forward::parses_simple_pairs
```

Add tests alongside conversion changes — especially `streaming_responses`, which has no unit tests yet and is the riskiest untested surface.

## Docs

- [`AGENTS.md`](AGENTS.md) — architecture and invariants for AI agents working on the repo.
- [`CONTEXT.md`](CONTEXT.md) — glossary of domain terms.
- [`docs/adr/`](docs/adr/) — architectural decision records:
  - [0001: Three local entries over one upstream](docs/adr/0001-three-local-entries-over-one-upstream.md)
  - [0002: Upstream type is explicit, not guessed from URL](docs/adr/0002-upstream-type-is-explicit.md)
  - [0003: Upstream config renamed to `UPSTREAM_*`](docs/adr/0003-upstream-naming.md)
  - [0004: Media description via a vision model](docs/adr/0004-media-description-via-vision-model.md)

## Acknowledgments

`ai-bridge` builds on ideas from two neighboring projects in this workspace, and we're grateful to both communities:

- **[CLI Proxy API] <https://github.com/router-for-me/CLIProxyAPI>** — a proxy server that exposes OpenAI/Gemini/Claude/Codex/Grok-compatible API endpoints for CLIs, letting any OpenAI- or Claude-compatible client reach multiple providers and accounts. Its approach of surfacing multiple upstream providers behind a single compatible local surface was a key inspiration for `ai-bridge`'s "many local entries, one upstream" design.

- **[CC Switch] <https://github.com/farion1231/cc-switch>** — the all-in-one manager for Claude Code, Claude Desktop, Codex, Gemini CLI, and other agentic tools, including a built-in local proxy. `ai-bridge`'s `UPSTREAM_HEADERS` parsing (pipe-separated `A:a|B:b` overrides) directly mirrors cc-switch's `apply_local_proxy_header_overrides` implementation, and its header-override semantics follow the same conventions.

Thank you to the maintainers and contributors of both projects for their work and for making the source available.
