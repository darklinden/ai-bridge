# Upstream config renamed from OPENAI_COMP_* to UPSTREAM_*

All upstream configuration moved to `UPSTREAM_URL` / `UPSTREAM_API_KEY` / `UPSTREAM_MODEL` / `UPSTREAM_AUTH_KEY`, replacing the `OPENAI_COMP_*` names. The old names are deprecated and removed.

Since `UPSTREAM_TYPE` was already a required breaking change, renaming the rest in the same release keeps the migration to one shot instead of two. The old names would also mislead now that the upstream can be Anthropic.
