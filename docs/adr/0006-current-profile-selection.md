# Current profile selection is persisted in `.settings.toml` and marked by `--list`

Every successful launch records the profile it served into `~/.ai-bridge/.settings.toml` — a dot-prefixed, ai-bridge-managed file holding a single `current_profile = "<name>"` key. `ai-bridge --list` marks that selection with `*` instead of always marking `default`, and ignores every dot-prefixed file so the settings file never appears as a profile. Running `ai-bridge a`, then `ai-bridge b`, then `ai-bridge --list` shows `b` marked; running bare `ai-bridge` re-serves the recorded profile (ADR-0007), serving and recording `default` only before any selection exists.

The file is written with `config::save_current_profile` after a profile loads successfully, read best-effort with `config::current_profile` (missing file, invalid TOML, or absent key all degrade to no marker), and hidden from profile discovery by `list_profiles`' dot-prefix filter. No reserved profile name is needed: a user-authored `settings.toml` (no dot) stays a perfectly legal profile, and `.settings.toml` cannot be selected anyway because `.` fails `validate_profile_name`'s `[A-Za-z0-9_-]` charset rule.

Rationale:

- **"What am I running?" is a recurring question** — scripts, CLIs, and humans all point at the bridge; the selection file makes the current upstream answerable without reading shell history or process listings.
- **Switching is now explicit** — `ai-bridge b` both serves and records, so the recorded state never drifts from what was actually served.
- **Dot-prefix over reserved-name** — hiding by convention (`.settings.toml`) costs nothing and keeps every `*.toml` name usable as a profile; a reserved name would silently break an existing setup for anyone who already had a `settings.toml` profile, and validation asymmetry (one name refused for non-syntax reasons) is a footgun.
- **`--list` must never fail** — the list is a discovery aid; a corrupt or missing settings file degrades to "no marker" rather than an error, matching the existing "`--list` is never an error" rule for empty directories.
- **Serving outranks bookkeeping** — a read-only config dir (container mounts, `$HOME` locked down) must still serve; the save is best-effort and warns, the same philosophy as the `default.toml` starter template (`write_default_template`).

Consequences: `--list` output can carry at most one `*`, and it no longer necessarily marks `default`; `settings` remains a legal profile name; `.settings.toml` is auto-rewritten on every launch (hand edits are overwritten, though a hand-added `current_profile` is respected as the marker until the next launch); environments with read-only `~/.ai-bridge` run with a WARN and an unmarked list.