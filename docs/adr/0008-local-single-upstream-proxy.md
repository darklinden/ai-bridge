# The bridge stays a local single-user proxy, not a multi-upstream gateway

`ai-bridge` serves exactly one configured upstream at a time — one profile, one `upstream_type`, shared by all three local entries; it binds `127.0.0.1` unless `listen_addr` opts out. Auth is split by design: `auth_key` guards the inbound local clients, `api_key` is the upstream's own credential.

Routing a request to different upstreams by entry, path, or model name was considered and rejected. The proxy's value is translating one client format to one upstream format; a routing layer would make "which upstream serves this request?" depend on runtime state, which is the same ambiguity `upstream_type` was made explicit to remove (ADR-0002). Exposing the bridge past the loopback interface stays an explicit opt-in rather than a default.
