# ai-bridge

A lightweight Rust HTTP proxy that lets local **Anthropic Messages**, **OpenAI Chat Completions**, and **OpenAI Responses** clients all talk to a single configured upstream API. Requests are translated from the local entry format to the upstream format, forwarded, and the response is translated back — so existing SDKs and CLIs keep working unchanged while you point them at a different backend.

Every request/response pair is also written to stdout as a single-line stream, making `ai-bridge` useful both as a gateway and as a conversation logger.

## Features

- **Three local endpoints, one upstream** — serve `/v1/messages`, `/v1/chat/completions`, and `/v1/responses` simultaneously; all forward to a single upstream chosen via the `upstream_type` key of the selected profile.
- **Bidirectional format conversion** — requests and responses (non-streaming **and** SSE streaming) are converted in both directions, including cross-format paths (e.g. Chat entry → Responses upstream) via an Anthropic intermediate.
- **TOML profiles** — configuration lives in `~/.ai-bridge/<name>.toml`; keep several upstream configs side by side (`ai-bridge deepseek`, `ai-bridge anthropic`) and list them with `ai-bridge --list`. No environment variables except `RUST_LOG` (ADR-0005).
- **Explicit upstream declaration** — the upstream format is declared by the `upstream_type` key, never guessed from the URL.
- **Always-on request logging** — a compact `[REQ #id]` / `[RESP #id]` line per request, independent of `RUST_LOG`.
- **Prompt-cache-friendly** — tool-result payloads are serialized canonically so byte-identical payloads reuse upstream prompt caches.
- **Reasoning bridging** — OpenAI `reasoning` items are preserved across Anthropic's `signature`/redacted-thinking field.
- **Tool-result media** — structured media blocks in tool results are re-emitted correctly for each entry format.
- **Vision media description** — when the upstream is a text-only model, images are either described by a configured vision model (ADR-0004) or replaced with an `[Unsupported Image]` placeholder.
- **Inbound auth & CORS** — optional token auth for local clients, permissive CORS, and a 200 MB request-body limit.

## How it works

```
local entry  →  upstream (format declared by the upstream_type key)

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

Create a profile config (see below), then:

```bash
./target/release/ai-bridge            # loads ~/.ai-bridge/default.toml
./target/release/ai-bridge deepseek   # loads ~/.ai-bridge/deepseek.toml
```

You should see:

```
ai-bridge listening on 127.0.0.1:18650
  → profile   = default (/Users/you/.ai-bridge/default.toml)
  → upstream  = oai-chat https://api.example.com/v1/chat/completions
  → model     = deepseek-v4-flash
  ...
```

### Quick start

Just launch once — a missing config file reports an error **and** drops a fully-commented starter template at `~/.ai-bridge/default.toml`:

```bash
./target/release/ai-bridge        # errors + writes ~/.ai-bridge/default.toml template
$EDITOR ~/.ai-bridge/default.toml # fill in the three required keys
./target/release/ai-bridge        # up and running
```

Existing files are never overwritten. Copy `default.toml` under another name to keep several upstream configs side by side; `ai-bridge --list` shows what's available.

## CLI usage

| Command | Behavior |
| --- | --- |
| `ai-bridge` | Serve using `~/.ai-bridge/default.toml`. |
| `ai-bridge <profile>` | Serve using `~/.ai-bridge/<profile>.toml`. Profile names accept letters, digits, `_`, `-`. |
| `ai-bridge -l`, `--list` | List available profiles (`*` marks `default`). Missing/empty directory prints a hint, not an error. |
| `ai-bridge -h`, `--help` | Show usage. |

Startup config problems exit with status 1 and a `配置错误:` message naming the exact file and cause (missing file, TOML syntax, unknown key, missing required key); CLI misuse exits with status 2 plus usage.

## Configuration

One file = one upstream configuration, stored as `~/.ai-bridge/<profile>.toml` (ADR-0005). Unknown keys are rejected at startup to catch typos.

```toml
# ---- required ----
upstream_type = "oai-chat"     # anthropic-messages | oai-chat | oai-responses (trimmed+lowercased)
url     = "https://api.example.com/v1/chat/completions"
api_key = "sk-your-upstream-key"   # Bearer for OpenAI-style upstreams; x-api-key for Anthropic

# ---- optional ----
model             = "deepseek-v4-flash"  # forced onto every request; default deepseek-v4-flash
listen_addr       = "127.0.0.1"          # default 127.0.0.1 (loopback only; use 0.0.0.0 to expose)
listen_port       = 18650                # default 18650
auth_key          = "local-token"        # inbound auth for local clients; absent = auth disabled
vision_supplement = false                # default false: images pass through untouched

[headers]                      # extra upstream headers, overriding defaults
X-Tenant = "default"

[reasoning]                    # outbound reasoning policy
thinking = true                # master switch, default true
effort   = "max"               # default max; off/drop/none/disable/disabled removes the field;
                               # any other value passes through lowercased verbatim (e.g. xhigh)

[vision]                       # optional vision model for image description (all three keys required)
url      = "https://vl.example.com/v1/chat/completions"  # format inferred: /responses, /messages, else chat
api_key  = "sk-your-vision-key"
model    = "vl-model"
prompt_mode = "auto"           # auto | general | ui | compact; unknown falls back to auto
prompt   = ""                  # non-blank overrides prompt_mode
max_tokens = 1024              # 0 omits the field entirely
[vision.headers]               # extra headers for vision requests
```

Key semantics:

- `model` — every request body has its `model` replaced by this value; a client-supplied model never reaches the upstream.
- `[reasoning] effort` — the single source of truth for the outgoing effort value; only stamped when the client actually asked for reasoning. No per-model mapping exists.
- `[vision]` — enabled only when `url`, `api_key`, and `model` are all present and non-blank; otherwise vision logs which keys are missing and disables itself.
- The only environment variable still read is `RUST_LOG` (`tracing` filter, default `info`) — all previous `UPSTREAM_*` / `VISION_*` / `LISTEN_*` variables were removed (ADR-0005), so configuration can no longer leak through shells or `.env` files.

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
  -H "Authorization: Bearer local-token" \     # only if auth_key is set in the profile
  -d '{"model": "anything", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Conversion matrix

| Local entry \ Upstream | `oai-chat` | `oai-responses` | `anthropic-messages` |
| --- | --- | --- | --- |
| Anthropic Messages | direct conversion | direct conversion | pass-through |
| Chat Completions | pass-through | double-bridge via Anthropic | direct (reverse) conversion |
| Responses | double-bridge via Anthropic | pass-through | direct (reverse) conversion |

## Behavior notes

- **Model is always overridden** to the profile's `model` on every local entry; a client-supplied `model` never reaches the upstream.
- **Error format follows the local entry**, not the upstream: Anthropic entries return Anthropic-style errors, OpenAI entries return OpenAI-style errors. Mapping: `Transform`/`Forward` → 502, `Config`/`Server` → 500, `Unauthorized` → 401, `Unsupported` → 400.
- **OpenAI-only fields** Anthropic cannot represent are dropped with a WARN; `n > 1` is rejected (400) because Anthropic is single-output; `parallel_tool_calls=false` maps to `tool_choice.disable_parallel_tool_use`.
- **Reasoning items** (OpenAI) have no Anthropic field, so each item is base64-encoded into the Anthropic `signature`/redacted-thinking payload and decoded on replay.
- **Outbound reasoning policy** — the effort value sent upstream is decided solely by `[reasoning] effort` (default `max`), gated by the `[reasoning] thinking` master switch; no per-model effort mapping exists.
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
- Errors break to a tagged `[ERR-REQ #id]` / `[ERR-RESP #id]` line, also preceded by two blank lines like the `[REQ]`/`[RESP]` markers:

  ```text
  [ERR-REQ #0001]: upstream status 503: Service Unavailable
  ```

  `[ERR-REQ #id]` lines report the upstream HTTP status (`429`, `503`, ...) plus a human-readable reason extracted from the upstream body (JSON `error.message` or raw text), and the same real status is relayed to the client — no blanket 502.
- Each pair is preceded by two blank lines for readability.
- This is a terminal aid, not a durable log — nothing is written to disk.

## Media & vision

By default (with `vision_supplement` absent or `false`) images are passed through to the upstream untouched, so the upstream's own vision handles them. When `vision_supplement = true` and the upstream model is confirmed text-only, image blocks are preprocessed before forwarding:

- If a vision model is configured (the `[vision]` table with `url` + `api_key` + `model`), all images in the request are described in a single non-streaming vision call and the image blocks are replaced with the resulting text. Descriptions come from a built-in prompt template selected by `prompt_mode` (`auto` self-routes UI screenshots to a UI-reconstruction structure and other images to a general structure), or from a custom `prompt`; `max_tokens` caps the description length. Each image is labeled by its document position (`[Image N]`), which stays correct when only a subset of images is uncached. Results are cached in-process keyed by image fingerprint with a TTL; a vision failure degrades to `[Unsupported Image]` and never blocks the request.
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
  - [0005: Configuration moves from environment variables to TOML profiles](docs/adr/0005-toml-profile-config.md)

## Acknowledgments

`ai-bridge` builds on ideas from three neighboring projects in this workspace, and we're grateful to all three communities:

- **[CLI Proxy API] <https://github.com/router-for-me/CLIProxyAPI>** — a proxy server that exposes OpenAI/Gemini/Claude/Codex/Grok-compatible API endpoints for CLIs, letting any OpenAI- or Claude-compatible client reach multiple providers and accounts. Its approach of surfacing multiple upstream providers behind a single compatible local surface was a key inspiration for `ai-bridge`'s "many local entries, one upstream" design.

- **[CC Switch] <https://github.com/farion1231/cc-switch>** — the all-in-one manager for Claude Code, Claude Desktop, Codex, Gemini CLI, and other agentic tools, including a built-in local proxy. `ai-bridge`'s `[headers]` table overrides mirror cc-switch's `apply_local_proxy_header_overrides` implementation: invalid entries are skipped with a WARN instead of failing startup, and later entries win over earlier ones.

- **[Plugin-Deepseek-Vision] <https://github.com/Zesuy/Plugin-Deepseek-Vision>** — a CLIProxyAPI v7 plugin (Go, MIT) that gives DeepSeek's text-only models the ability to understand images: it intercepts incoming requests after routing, has the host's vision models analyze the images (grouping multiple images per prompt into one joint VLM call), and rewrites the image blocks into a joint text analysis before DeepSeek ever sees the prompt. That is the same "describe images with a vision model when the target model is text-only" strategy behind `ai-bridge`'s vision preprocessing (ADR-0004), and its fail-closed philosophy — on any vision failure the plugin refuses to forward raw images rather than silently degrade — mirrored `ai-bridge`'s decision to drop to `[Unsupported Image]` and never let a vision failure block a request.

Thank you to the maintainers and contributors of all three projects for their work and for making the source available.
