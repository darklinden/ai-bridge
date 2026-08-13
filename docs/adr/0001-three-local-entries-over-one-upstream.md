# Local proxy exposes three API styles over one upstream

The service forwards three local entry points — Anthropic Messages (`/v1/messages`), OpenAI Chat Completions (`/v1/chat/completions`), and OpenAI Responses (`/v1/responses`) — to a single configured upstream. The upstream format is chosen once via `UPSTREAM_TYPE` and applies to all three entries.

All conversions between entry format and upstream format happen inside the proxy. Six reverse conversion functions (request body, non-streaming response, SSE stream × chat/responses) had to be added on top of the six that already existed for the Anthropic-entry → OpenAI-upstream direction.
