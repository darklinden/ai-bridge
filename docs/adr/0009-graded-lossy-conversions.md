# Lossy conversions are graded in three tiers, not one blanket policy

When a local entry carries a field the upstream cannot express, the proxy decides by whether dropping it changes the request's meaning:

- **Map** — a target exists that preserves the meaning, so translate it (`parallel_tool_calls = false` → `tool_choice.disable_parallel_use`).
- **Drop with a WARN** — the field is inert for this upstream and omitting it changes nothing (`store`, `include`, `metadata`, `truncation`, `text.format` on the Responses → Anthropic path).
- **Reject with 400** — omitting it would silently change what the caller asked for (`n > 1`, because Anthropic is single-output).

A single blanket rule was rejected from both ends: always-warn silently degrades `n > 1` to one output, which the caller never asked for; always-reject fails everyday requests over fields the upstream has no opinion about at all. The tie-breaker is therefore *meaning*, not format membership — the same field can be inert on one path and semantic on another.
