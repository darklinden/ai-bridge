# Configuration lives in TOML profiles

All upstream configuration lives in `~/.ai-bridge/<name>.toml` — one file per upstream configuration ("profile"). The binary selects a profile by CLI argument (`ai-bridge deepseek` → `deepseek.toml`), defaulting to `default.toml`; `ai-bridge --list` shows what is available. Required keys are `upstream_type` / `url` / `api_key`, with optional `[headers]`, `[reasoning]`, and `[vision]` tables. Unknown keys fail startup, so a typo cannot silently disable a setting.

Only `RUST_LOG` is read, because it is the `tracing` convention.

Rationale:

- **Multiple simultaneous configs** — profiles keep several complete configurations side by side and switch by name, which matches how the proxy is actually used (one upstream per CLI/client setup).
- **No leakage** — exported variables and `.env` files leak into shells, process listings, and logs; a per-user config directory does not.
- **Self-documenting and discoverable** — the file carries comments next to every key, and `--list` answers "what can I run?" without reading shell rc files.
- **Typo safety** — `deny_unknown_fields` rejects misspelled keys at startup instead of ignoring them.
- **Low-cost first run** — when a requested profile file does not exist, startup reports an error and drops a fully-commented starter template at `default.toml` (never overwriting an existing file), so onboarding is: run once, fill three keys, run again.

The default bind address is `127.0.0.1`, so exposing the proxy requires an explicit `listen_addr = "0.0.0.0"`. Blank strings in `[vision]` count as absent rather than half-configuring the feature.
