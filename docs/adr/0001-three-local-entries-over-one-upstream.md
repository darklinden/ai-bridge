---
status: superseded
superseded-by: ADR-0005, ADR-0008
---

# Local proxy exposes three API styles over one upstream

*Superseded: the entry model described here is restated in ADR-0008, and its conversion-layer inventory now lives in `docs/architecture.md`. Kept as history.*

The service forwards three local entry points — Anthropic Messages (`/v1/messages`), OpenAI Chat Completions (`/v1/chat/completions`), and OpenAI Responses (`/v1/responses`) — to a single configured upstream. The upstream format is chosen once via `UPSTREAM_TYPE` and applies to all three entries.

All conversions between entry format and upstream format happen inside the proxy. Six reverse conversion functions (request body, non-streaming response, SSE stream × chat/responses) had to be added on top of the six that already existed for the Anthropic-entry → OpenAI-upstream direction.
