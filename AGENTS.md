# AGENTS.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

`ai-bridge` is a Rust HTTP proxy that lets local Anthropic Messages, OpenAI Chat Completions and OpenAI Responses clients talk to one configured upstream API — translating between formats in both directions and logging every request/response pair to stdout.

This file is a **map, not a manual**. It holds the commands you need to build and test, and points at the documents that own the actual content.

## Read before you change something

| If you are changing… | Read |
|---|---|
| what the product must do | `docs/requirements.md` |
| module boundaries, seams, dependency direction | `docs/architecture.md` |
| how a mechanism works, and what must stay true | `docs/implementation.md` |
| what a term means | `CONTEXT.md` |
| why something is the way it is | `docs/adr/` |

## Commands

```bash
cargo build                     # build
cargo run                       # serve the recorded profile (default.toml on first run; needs upstream_type + url + api_key)
cargo run -- deepseek           # serve ~/.ai-bridge/deepseek.toml instead
cargo run -- --list             # list available profiles
cargo test                      # run all unit tests
cargo test <module>             # run one module's tests, e.g. cargo test reasoning_bridge
cargo test <module>::<test>     # run a single test, e.g. cargo test forward::builds_simple_pairs
```

## Conventions

- **Tests are inline** — `#[cfg(test)] mod tests` inside the module under test. There is no `tests/` integration directory.
- **Conversion changes ship with tests.** The conversion modules are the ones whose behaviour is hardest to eyeball; inline tests are the expected form (see INV-13 in `docs/implementation.md`).
- **Verify end-to-end when touching routing or streaming.** `server.rs`, `main.rs` and `streaming_responses.rs` carry the least unit coverage — use the scripts in `manual-test/`.
- **Do not write counts into documentation.** Counts of tests, modules or pipelines go stale silently, and a stale number in a doc is worse than no number. Describe the shape instead.
- **Known deviations and open questions live in `docs/implementation.md` § 8**, with stable IDs — reference the ID from a commit rather than re-describing the problem.
