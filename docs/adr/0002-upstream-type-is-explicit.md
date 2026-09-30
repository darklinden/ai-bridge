# Upstream type is explicit, not guessed from URL

Every upstream the proxy talks to declares its wire format explicitly: the main upstream through its profile's `upstream_type` key, the `[vision]` endpoint through `[vision].upstream_type` (ADR-0004). Both must name one of exactly `anthropic-messages` / `oai-chat` / `oai-responses`, and a missing or invalid value must fail configuration at startup. No URL is to be inspected to decide a wire format.

The alternative was the old `detect_api_format` heuristic, which inferred the format from whether the URL contained `/responses`. That breaks as soon as Anthropic is a valid upstream: `https://api.anthropic.com/v1/messages` contains neither `chat` nor `responses` and would be misclassified as chat. Explicit declaration removes the ambiguity, at the cost of one required key. The `[vision]` endpoint falls under the same rule for the same reason (ADR-0004).
