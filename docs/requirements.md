# ai-bridge — Functional Requirements

What the proxy must do, as *to-be* requirements. Each one names how it is verified today. No counts (of tests, pipelines, or modules) appear here on purpose — they go stale.

**Scope.** A local, single-user proxy: one user, one configured upstream at a time, loopback by default. It is deliberately *not* a multi-upstream gateway (ADR-0008).

## 1. Entries and routing

**FR-1 — Three local entries, one upstream.** The proxy exposes `POST /v1/messages` (Anthropic Messages), `POST /v1/chat/completions` (OpenAI Chat) and `POST /v1/responses` (OpenAI Responses). All three are served against the single upstream the running profile names.
_Verified by_: `manual-test/test-messages.sh`, `test-chat.sh`, `test-responses.sh`.

**FR-2 — Inbound auth is separate from upstream auth.** When a profile sets `auth_key`, the local entries require it; the upstream's own credential (`api_key`) is never the inbound one. A rejected request is answered 401 in the caller's entry format.
_Verified by_: inline tests in `server`.

**FR-3 — The local model is always overridden.** A client-supplied `model` is ignored on all three entries and replaced with the profile's `model` (ADR-0013).
_Verified by_: inline tests in `server`.

**FR-4 — Same-format requests skip format conversion.** When the entry's format and the upstream's format are the same, no conversion runs in either direction: the response is re-emitted byte for byte, and the request body is sent without translation. The shared per-request plumbing still applies — inbound auth, the `model` override (FR-3) and the reasoning policy (FR-8) — so "verbatim" here means *unconverted*, not *unmodified*.
_Verified by_: `manual-test` scripts run against a same-format upstream; the passthrough wrapper re-yields the original chunks.

## 2. Upstream declaration and configuration

**FR-5 — Every upstream format is declared, never inferred.** The main upstream declares `upstream_type` in its profile; the `[vision]` endpoint declares `[vision].upstream_type`. Valid values are `anthropic-messages`, `oai-chat`, `oai-responses`. No URL is inspected to decide a format (ADR-0002, ADR-0004).
_Verified by_: inline tests in `config`; a URL that contains no format marker still resolves correctly.

**FR-6 — Configuration is TOML profiles.** A profile lives at `~/.ai-bridge/<name>.toml`, selects one upstream by name, and holds `upstream_type` / `url` / `api_key` plus optional `[headers]` / `[reasoning]` / `[vision]` / `[chat]` tables. Unknown keys fail startup rather than silently doing nothing. No configuration *value* comes from the environment; `RUST_LOG` is the one exception, because it is the `tracing` convention (ADR-0005). The config directory is not configured either: it is the platform home directory (`std::env::home_dir`) joined with `.ai-bridge`, and on Windows that lookup never consults `HOME` (ADR-0005).
_Verified by_: inline tests in `config` for unknown-key rejection and for the config-directory resolution (`config_dir` rejects a missing or blank home and keeps a non-blank one verbatim, `home_config_dir` joins the platform home); the Windows leg of the `test` CI job runs those tests on Windows. That `HOME` is never consulted there is `std::env::home_dir`'s own guarantee (fixed in Rust 1.85), which `rust-version` pins.

**FR-7 — The profile's token-cap field name is explicit.** `[chat] max_tokens_field` says which spelling the upstream wants (`max_tokens`, the default, or `max_completion_tokens`). The proxy never picks a field name by reading the model name (ADR-0011).
_Verified by_: inline tests in `convert`.

**FR-8 — Outbound reasoning effort is a single configured value.** `reasoning_effort` / `reasoning.effort` comes only from `[reasoning] effort` (default `max`), stamped on every outbound request, including pass-through. Conversions decide only *whether* reasoning was requested; a thinking-disabled request must not gain an effort field (ADR-0011).
_Verified by_: inline tests in `convert` and `forward`.

## 3. Conversion

**FR-9 — Conversion happens in the proxy, in both directions.** Requests are translated from the entry format to the upstream format, responses back. Cross-format paths (e.g. Chat entry → Responses upstream) bridge through the Anthropic intermediate representation.
_Verified by_: inline tests across `convert`, `transform_responses`, `streaming_responses`, `convert_reverse`, `responses_reverse`.

**FR-10 — Lossy conversions are graded by meaning.** A field the upstream cannot express is (a) mapped when a meaning-preserving target exists, (b) dropped with a WARN when it is inert for that upstream, or (c) rejected 400 when dropping it would silently change what the caller asked for (`n > 1`, since Anthropic is single-output). The same field can be inert on one path and semantic on another; meaning, not format membership, decides (ADR-0009).
_Verified by_: inline tests in `convert_reverse` and `responses_reverse`.

**FR-11 — Reasoning items survive a round trip.** An OpenAI `reasoning` item, which has no Anthropic field, is carried through the Anthropic `signature` payload and replayed on the way back, so a bridged conversation does not lose it (ADR-0011 keeps the outbound *effort* policy separate from this).
_Verified by_: inline tests in `reasoning_bridge`.

**FR-12 — Tool-result media survives both directions.** Structured media inside a tool result is re-emitted in a form the target format can carry, rather than being flattened to text.
_Verified by_: inline tests in `tool_media`.

**FR-13 — Cache-sensitive payloads are byte-stable.** Tool-result payloads that are semantically identical must serialise identically, so the upstream's prompt cache can hit. Key order is not to be changed casually.
_Verified by_: inline tests in `json_canonical`.

## 4. Media and vision

**FR-14 — Describing images is opt-in.** When a local request carries images and the upstream is text-only, the proxy can replace the image blocks with a description produced by the `[vision]` model. The feature is off unless the profile turns it on; "the upstream is text-only" is a model-name heuristic, so the default must not act on it (ADR-0004).
_Verified by_: inline tests in `vision`; the toggle's default is asserted in `config`.

**FR-15 — Vision configuration is all-or-nothing, and an incomplete `[vision]` never breaks serving.** Without `url`, `api_key` or `model` the feature is disabled with a WARN and the request proceeds. A missing or unknown `[vision].upstream_type` is a malformed config and fails startup instead (ADR-0004).
_Verified by_: inline tests in `config`.

**FR-16 — Vision failure degrades, never blocks.** A failed description falls back to an `[Unsupported Image]` placeholder and the request still completes. Descriptions are cached in-process by image fingerprint with a TTL.
_Verified by_: inline tests in `vision`.

## 5. Errors

**FR-17 — The error envelope follows the local entry.** An Anthropic entry receives `{"type":"error","error":{"type":…,"message":…}}` with `type` from Anthropic's vocabulary; an OpenAI entry receives the OpenAI shape. Each entry's envelope is produced by exactly one shaping function (ADR-0010).
_Verified by_: inline tests in `server`.

**FR-18 — Upstream status and reason are relayed.** A failure from the upstream keeps the upstream's status code and message rather than being flattened to 502; only the envelope is rewritten (ADR-0010).
_Verified by_: inline tests in `server`.

## 6. Observability

**FR-19 — The request log is a complete mirror, always on, with two destinations.** Every request and response is written as a line-oriented stream — request line, response line with streamed text appended onto it, a system-prompt line, and tagged error lines — independent of `RUST_LOG`. Content is recorded as it is, including base64 payloads. Tool calls are content: the model's call is written on the response line as `[tool_use: {name}] {arguments}` and the tool's output on the request line as `[tool_result: {id}] {content}`, both verbatim and untruncated — a tool payload is written as it came, line breaks included, so one record may span several lines, while ordinary prose stays folded onto the single line — and a tool call is recorded whether or not the model wrote text before it. The stream goes to stdout by default; `--log <path>` additionally *tees* the same lines into that file, appended, never rotated by the proxy. An unopenable path fails startup; a write failure after startup warns once and serving continues (ADR-0012).
_Verified by_: `manual-test` runs — the lines are read directly from the run's stdout; `--log` runs are checked by hand, plus an unwritable path for the startup failure.

**FR-20 — Logging never alters what is forwarded.** The log is written alongside the forwarded bytes, never in place of them.
_Verified by_: the passthrough wrapper re-yields the original chunks.

## 7. Lifecycle

**FR-21 — The current selection is persisted and resumable.** A successful launch records the served profile; `--list` marks it and ignores dot-prefixed files; a bare launch serves the recorded profile. Persistence is best-effort — a write failure warns and must not block serving (ADR-0006, ADR-0007).
_Verified by_: inline tests in `config`.

**FR-22 — A missing profile is a loud error, never a silent fallback.** An unresolvable profile name fails with the same error as typing an invalid name; it never falls back to another upstream (ADR-0007).
_Verified by_: inline tests in `config`.
