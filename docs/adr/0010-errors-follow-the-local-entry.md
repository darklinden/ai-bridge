# Error responses are shaped by the local entry, in that entry's own vocabulary

A failed request is answered in the format of the entry the caller used, never the upstream's. An Anthropic entry gets `{"type":"error","error":{"type":"<type>","message":"<msg>"}}`, with `type` drawn from Anthropic's own vocabulary — `invalid_request_error` for a 400, `authentication_error` for a 401, `api_error` otherwise; an OpenAI entry gets `{"error":{"message":"<msg>","type":"<type>"}}` with the OpenAI equivalents. Each entry's shape is owned by exactly one shaping function: a second site for the same envelope is what lets it drift, so a single owner is a rule rather than a tidiness preference.

The upstream's real status code and reason are relayed to the caller rather than flattened to 502; only the envelope is rewritten
into the local entry's shape. A caller therefore sees the upstream's status and message inside the error format it already knows how to parse.
