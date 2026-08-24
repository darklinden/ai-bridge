# Configuration moves from environment variables to TOML profiles

All upstream configuration now lives in `~/.ai-bridge/<name>.toml` — one file per upstream configuration ("profile"). The binary selects a profile by CLI argument (`ai-bridge deepseek` → `deepseek.toml`), defaulting to `default.toml`; `ai-bridge --list` shows what is available. Required keys are `upstream_type` / `url` / `api_key`, with optional `[headers]`, `[reasoning]`, and `[vision]` tables. Unknown keys fail startup, so a typo cannot silently disable a setting.

Every previous environment variable (`UPSTREAM_*`, `VISION_*`, `LISTEN_*`, `THIRDPARTY_VISION_SUPPLEMENT`) was removed, along with `.env` loading (dotenvy). Only `RUST_LOG` is still read, because it is the `tracing` convention.

Rationale:

- **Multiple simultaneous configs** — switching upstreams used to mean editing one shared env set; profiles keep several complete configurations side by side and switch by name, which matches how the proxy is actually used (one upstream per CLI/client setup).
- **No leakage** — exported variables and `.env` files leak into shells, process listings, and logs; a per-user config directory does not.
- **Self-documenting and discoverable** — the file carries comments next to every key, and `--list` answers "what can I run?" without reading shell rc files.
- **Typo safety** — `deny_unknown_fields` rejects misspelled keys at startup instead of ignoring them.
- **Low-cost first run** — when a requested profile file does not exist, startup reports an error and drops a fully-commented starter template at `default.toml` (never overwriting an existing file), so onboarding is: run once, fill three keys, run again.

Two deliberate behavior changes ride along: the default bind address tightened from `0.0.0.0` to `127.0.0.1` (exposing the proxy requires an explicit `listen_addr = "0.0.0.0"`), and blank strings in `[vision]` now count as absent instead of half-configuring the feature.

Consequences: this is a breaking change for existing deployments — migrate by creating `~/.ai-bridge/default.toml` with the same values the old environment carried. Header overrides move from the pipe-encoded `A:a|B:b` string to a native TOML table.
