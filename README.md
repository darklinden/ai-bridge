# ai-bridge

A lightweight Rust HTTP proxy that lets local **Anthropic Messages**, **OpenAI Chat Completions**, and **OpenAI Responses** clients all talk to a single configured upstream API. Requests are translated from the local entry format to the upstream format, forwarded, and the response translated back — so existing SDKs and CLIs keep working unchanged while you point them at a different backend.

Every request/response pair is also written out as a single-line log, which makes `ai-bridge` useful as a conversation viewer alongside its job as a gateway.

What you get:

- **Three local endpoints, one upstream** — `/v1/messages`, `/v1/chat/completions` and `/v1/responses`, all forwarding to one upstream chosen by the profile.
- **Conversion in both directions**, streaming included, with cross-format paths (e.g. Chat entry → Responses upstream) bridged through Anthropic.
- **TOML profiles** — keep several upstreams side by side and switch by name; nothing is configured through the environment.
- **Optional image description** — a text-only upstream can be shown images described by a separately configured vision model.
- **Inbound token auth** for local clients, permissive CORS, and a 200 MB request-body limit.

This file describes the program as it ships today. Where it disagrees with [`docs/requirements.md`](docs/requirements.md), the requirement is the target and the difference is tracked as a known deviation in [`docs/implementation.md`](docs/implementation.md) § 8.

## Requirements

- Rust (stable toolchain, edition 2021). Install with [rustup](https://rustup.rs/).

## Build & run

```bash
cargo build --release
```

Create a profile config (see below), then:

```bash
./target/release/ai-bridge            # loads the recorded profile (default.toml on first run)
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
| `ai-bridge` | Serve using the recorded profile (`~/.ai-bridge/default.toml` on first run). |
| `ai-bridge <profile>` | Serve using `~/.ai-bridge/<profile>.toml`. Profile names accept letters, digits, `_`, `-`. |
| `ai-bridge -l`, `--list` | List available profiles (`*` marks the current selection). Missing/empty directory prints a hint, not an error. |
| `ai-bridge -h`, `--help` | Show usage. |

Startup config problems exit with status 1 and a `configuration error:` message naming the exact file and cause (missing file, TOML syntax, unknown key, missing required key); CLI misuse exits with status 2 plus usage.

## Configuration

One file = one upstream configuration, stored as `~/.ai-bridge/<profile>.toml`. Unknown keys are rejected at startup to catch typos.

Every successful launch records the served profile in `~/.ai-bridge/.settings.toml` — a dot-prefixed file auto-managed by `ai-bridge` that holds `current_profile = "<name>"` (so `ai-bridge a` then `ai-bridge b` switches the recorded selection). A bare `ai-bridge` launch re-serves the recorded profile; a missing, corrupt, or keyless record falls back to `default`, while a recorded name that is invalid or whose file was deleted errors exactly like typing `ai-bridge <name>`. `--list` marks the current selection with `*` and ignores all dot-prefixed files, so the settings file never shows up as a profile; a missing or corrupt settings file only means `--list` shows no marker. Saving the selection is best-effort: a read-only config dir still serves, with a WARN.

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
- `[vision]` — enabled only when `url`, `api_key`, and `model` are all present and non-blank; otherwise vision logs which keys are missing and disables itself. The endpoint's format is inferred from its path today; making it an explicit declaration is a pending change (see the deviations list linked above).
- Configuration comes only from the profile file. The only variables read are `RUST_LOG` (a `tracing` filter, default `info`) and `HOME` (to locate `~/.ai-bridge`).

Configuration decisions and their rationale are recorded in [`docs/adr/`](docs/adr/).

## Endpoints

| Route | Local entry style |
| --- | --- |
| `POST /v1/messages` | Anthropic Messages |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/responses` | OpenAI Responses |

Each accepts the corresponding SDK/CLI request shape (streaming supported), and returns the same shape you would have gotten from the upstream — errors included.

### Example

```bash
# The Authorization header is only needed when auth_key is set in the profile.
curl -X POST http://127.0.0.1:18650/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer local-token" \
  -d '{"model": "anything", "messages": [{"role": "user", "content": "Hello"}]}'
```

Which entry converts to which upstream — and which combinations are pass-through — is laid out in [`docs/architecture.md`](docs/architecture.md) § 3.

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
- Today the log goes to stdout only — redirect it if you want it on disk (`./run.sh` does exactly that, teeing the whole process into a dated file). A built-in `--log` option is a pending change; the target contract for the log's content and destinations is [ADR-0012](docs/adr/0012-request-log-records-content-verbatim.md).

## Media & vision

By default (with `vision_supplement` absent or `false`) images are passed through to the upstream untouched, so the upstream's own vision handles them. When `vision_supplement = true` and the upstream model is confirmed text-only, image blocks are preprocessed before forwarding:

- If a vision model is configured (the `[vision]` table with `url` + `api_key` + `model`), all images in the request are described in a single non-streaming vision call and the image blocks are replaced with the resulting text. Descriptions come from a built-in prompt template selected by `prompt_mode` (`auto` self-routes UI screenshots to a UI-reconstruction structure and other images to a general structure), or from a custom `prompt`; `max_tokens` caps the description length. Each image is labeled by its document position (`[Image N]`), which stays correct when only a subset of images is uncached. Results are cached in-process keyed by image fingerprint with a TTL; a vision failure degrades to `[Unsupported Image]` and never blocks the request.
- Otherwise images are stripped and replaced with an `[Unsupported Image]` placeholder.

The request log notes handled images, e.g. `[media: 2 image(s) → [Unsupported Image]]`.

Why this is opt-in rather than automatic: the "upstream is text-only" check is a model-name heuristic, so a newly released vision-capable model the list has not caught up with would have its images silently swapped for descriptions. See [ADR-0004](docs/adr/0004-media-description-via-vision-model.md).

## Development

Build and test commands, the testing conventions, and where each kind of change should be documented all live in [`AGENTS.md`](AGENTS.md). In short:

```bash
cargo test                            # all unit tests
cargo run                             # serve the recorded profile
cargo run -- --list                   # list available profiles
```

The scripts in [`manual-test/`](manual-test/) drive the three entries end-to-end against a running instance.

## Docs

| Document | Contents |
| --- | --- |
| [`docs/requirements.md`](docs/requirements.md) | What the proxy must do — functional requirements with how each is verified. |
| [`docs/architecture.md`](docs/architecture.md) | Module boundaries, the entry × upstream matrix, dependency direction. |
| [`docs/implementation.md`](docs/implementation.md) | How each mechanism works, the invariants to preserve, and the known deviations. |
| [`CONTEXT.md`](CONTEXT.md) | Glossary of this project's terms. |
| [`AGENTS.md`](AGENTS.md) | Map for AI agents working in the repo: commands and conventions. |
| [`docs/adr/`](docs/adr/) | Architectural decision records, numbered from 0001. |

## Acknowledgments

`ai-bridge` builds on ideas from three neighboring projects in this workspace, and we're grateful to all three communities:

- **[CLI Proxy API] <https://github.com/router-for-me/CLIProxyAPI>** — a proxy server that exposes OpenAI/Gemini/Claude/Codex/Grok-compatible API endpoints for CLIs, letting any OpenAI- or Claude-compatible client reach multiple providers and accounts. Its approach of surfacing multiple upstream providers behind a single compatible local surface was a key inspiration for `ai-bridge`'s "many local entries, one upstream" design.

- **[CC Switch] <https://github.com/farion1231/cc-switch>** — the all-in-one manager for Claude Code, Claude Desktop, Codex, Gemini CLI, and other agentic tools, including a built-in local proxy. `ai-bridge`'s `[headers]` table overrides mirror cc-switch's `apply_local_proxy_header_overrides` implementation: invalid entries are skipped with a WARN instead of failing startup, and later entries win over earlier ones.

- **[Plugin-Deepseek-Vision] <https://github.com/Zesuy/Plugin-Deepseek-Vision>** — a CLIProxyAPI v7 plugin (Go, MIT) that gives DeepSeek's text-only models the ability to understand images: it intercepts incoming requests after routing, has the host's vision models analyze the images (grouping multiple images per prompt into one joint VLM call), and rewrites the image blocks into a joint text analysis before DeepSeek ever sees the prompt. That is the same "describe images with a vision model when the target model is text-only" strategy behind `ai-bridge`'s vision preprocessing (ADR-0004), and its fail-closed philosophy — on any vision failure the plugin refuses to forward raw images rather than silently degrade — mirrored `ai-bridge`'s decision to drop to `[Unsupported Image]` and never let a vision failure block a request.

Thank you to the maintainers and contributors of all three projects for their work and for making the source available.
