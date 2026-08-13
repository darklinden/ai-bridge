# Upstream type is explicit, not guessed from URL

`UPSTREAM_TYPE` is a required environment variable with exactly three allowed values: `anthropic-messages`, `oai-chat`, `oai-responses`. A missing or invalid value fails configuration at startup.

The previous `detect_api_format` heuristic inferred the upstream format from whether the URL contained `/responses`. That breaks once Anthropic is a valid upstream: `https://api.anthropic.com/v1/messages` contains neither `chat` nor `responses` and would be misclassified. Explicit declaration removes the ambiguity, at the cost of a breaking change for existing deployments.
