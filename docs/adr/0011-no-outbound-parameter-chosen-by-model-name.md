# Outbound parameters are never chosen by model name

No outbound request field is selected or valued by inspecting the model name. Anything that varies between upstream models is an explicit profile key instead:

- `reasoning_effort` / `reasoning.effort` comes only from `[reasoning] effort` (default `max`). Conversions decide only *whether* reasoning is requested at all; a thinking-disabled request never gains an effort field.
- The token-cap field name comes from `[chat] max_tokens_field` (default `max_tokens`), because newer OpenAI models reject `max_tokens` in favour of `max_completion_tokens`.

A model-name list was used for the second case (`is_openai_o_series`); this decision replaces it. The model name is itself a value the user writes in the profile, so the user — not a hardcoded list the proxy must keep current — is the authority on what their upstream accepts; a stale list fails silently, which is the same objection that made the upstream type explicit (ADR-0002). The inbound direction stays permissive: both `max_tokens` and `max_completion_tokens` are accepted and normalised to Anthropic's `max_tokens`, because accepting an extra spelling costs nothing.
