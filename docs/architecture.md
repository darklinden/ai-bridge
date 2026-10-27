# ai-bridge — Architecture

How the system is shaped: what the parts are, where the seams are, and which way the dependencies point. Mechanisms and invariants live in `implementation.md`; requirements in `requirements.md`.

## 1. System shape

A **single binary** (`ai-bridge`, no `lib.rs`). Three local entries sit in front of exactly one configured upstream; requests are translated on the way out and responses on the way back. The upstream's wire format is declared by the profile, never inferred from a URL (ADR-0002).

```
local entry ──────────────► upstream
  POST /v1/messages          upstream_type =
  POST /v1/chat/completions    anthropic-messages
  POST /v1/responses           oai-chat
                               oai-responses
       │
       └─ same format as upstream? ──no conversion───► upstream
          (handle_passthrough)
```

## 2. Entry and routing layer

`server.rs` owns routing and per-entry plumbing; `main.rs` owns startup and profile selection.

- **Three conversion handlers** — `handle_messages`, `handle_chat_completions_local`, `handle_responses_local`. Each dispatches on `UpstreamType` to the entry × upstream pipeline for its entry.
- **One pass-through handler** — `handle_passthrough` serves the case where the entry's format already equals the upstream's, so no conversion runs in either direction and the response is re-emitted byte for byte (FR-4). It is a distinct path, not a degenerate conversion: `passthrough_log` exists only to observe this stream. The shared request plumbing still applies, so the forwarded body carries the overridden `model` and the stamped effort — *unconverted*, not untouched.
- **Per-request plumbing, shared by all entries** — inbound auth against the profile's `auth_key`, unconditional override of `model` to the profile's value, media preprocessing, and error responses shaped to the *local entry* (`Error::into_entry_response`, ADR-0010).
- **Configuration** — `config.rs` discovers profiles and parses TOML into runtime types, including `.settings.toml` bookkeeping; `forward.rs` owns those runtime config types, the outbound HTTP calls, and the reasoning policy; `error.rs` owns error taxonomy and status mapping.

## 3. Conversion layer

Both directions live in the proxy, one module per entry format and direction:

| Direction | Request body, non-streaming response, SSE stream |
|---|---|
| Anthropic entry → OpenAI upstream | `convert.rs`, `transform_responses.rs`, `streaming_responses.rs` |
| OpenAI entry → Anthropic upstream | `convert_reverse.rs`, `responses_reverse.rs` |

Cross-format paths — a Chat entry served by a Responses upstream, say — do **not** get their own module: they bridge through the Anthropic intermediate representation, which keeps the matrix from growing as `entries × upstreams`.

## 4. Cross-cutting mechanisms

Shared concerns, each centralised so both directions use one implementation:

| Mechanism | Module | Why it is centralised |
|---|---|---|
| Reasoning items across formats | `reasoning_bridge.rs` | OpenAI `reasoning` items have no Anthropic field; the whole item rides in the Anthropic `signature` payload and is decoded on replay. |
| Media inside tool results | `tool_media.rs` | Tool results carry structured media in Anthropic/Responses but plain text in Chat; media is lifted out and re-emitted in a shape the target accepts. |
| Images for text-only upstreams | `media_sanitizer.rs` | Decides whether the upstream can see images at all. |
| Image description | `vision.rs` | Owns the vision call, prompt selection, and the fingerprint cache. |
| Byte-stable serialisation | `json_canonical.rs` | Cache-sensitive payloads must serialise identically. |
| Request/response log | `reqlog.rs` | One always-on stdout stream, shared by every path. |
| Pass-through stream observation | `passthrough_log.rs` | Extracts loggable text from a stream that must otherwise stay byte-identical. |

## 5. Dependency direction

```
server.rs / main.rs        entry, routing, startup
      │
      ▼
convert*.rs / *responses*.rs    format translation
      │
      ▼
reasoning_bridge · tool_media · media_sanitizer · vision
json_canonical · reqlog · passthrough_log          cross-cutting
      │
      ▼
forward.rs                 outbound HTTP, reasoning policy
error.rs · config.rs       taxonomy, configuration
```

Nothing below depends on anything above; the cross-cutting modules are used *by* the conversion layer rather than calling into it. `forward.rs` is the only place that performs outbound HTTP.

## 6. Runtime shape

- Binds `127.0.0.1` by default; exposing the proxy requires an explicit `listen_addr = "0.0.0.0"` (ADR-0005, ADR-0008).
- Serves exactly one profile per process; profile selection and persistence are described in ADR-0006 and ADR-0007.
- Logs to stdout by default, or simultaneously to stdout and a file with `--log <path>`; no telemetry endpoint. The CLI surface is hand-rolled, so a new option is a parser change, not a library one.
- Tests are inline `#[cfg(test)] mod tests`; there is no `tests/` directory. `error.rs` carries the least unit coverage; routing and streaming are end-to-end surfaces, so changes there are checked with the `manual-test/` scripts rather than by unit tests alone.
