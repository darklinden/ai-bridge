mod config;
mod convert;
mod convert_reverse;
mod error;
mod forward;
mod json_canonical;
mod media_sanitizer;
mod passthrough_log;
mod reasoning_bridge;
mod reqlog;
mod responses_reverse;
mod server;
mod streaming_responses;
mod tool_media;
mod transform_responses;
mod vision;

use crate::config::{LoadedProfile, DEFAULT_PROFILE};
use crate::error::Error;
use crate::forward::AppState;
use std::ffi::OsString;
use std::path::Path;
use std::sync::Arc;

const USAGE: &str = "\
ai-bridge — local Anthropic/OpenAI-compatible bridge to one configured upstream

Usage:
  ai-bridge              Load the recorded profile (default.toml on first run)
  ai-bridge <profile>    Load ~/.ai-bridge/<profile>.toml
  ai-bridge -l, --list   List available profiles (* marks the current selection)
  ai-bridge -h, --help   Show this help";

enum Cli {
    /// Serve the named profile; `None` = bare launch, resolved to the
    /// persisted current profile (or `default`) at serve time.
    Run { profile: Option<String> },
    List,
    Help,
}

/// Parse argv into a [`Cli`]. Hand-rolled because the surface is exactly three
/// behaviors; anything ambiguous or unknown is a usage error.
fn parse_cli(args: &[OsString]) -> Result<Cli, String> {
    let mut positional: Vec<String> = Vec::new();
    let mut list = false;
    let mut help = false;
    for arg in &args[1..] {
        match arg.to_str() {
            Some("-l" | "--list") => list = true,
            Some("-h" | "--help") => help = true,
            Some(flag) if flag.starts_with('-') => {
                return Err(format!("Unknown option: {flag}"));
            }
            Some(name) => positional.push(name.to_string()),
            None => {
                return Err(format!(
                    "Invalid non-UTF-8 argument: {}",
                    arg.to_string_lossy()
                ));
            }
        }
    }

    if help {
        return Ok(Cli::Help);
    }
    if list {
        if !positional.is_empty() {
            return Err(format!(
                "--list takes no arguments (got: {})",
                positional.join(" ")
            ));
        }
        return Ok(Cli::List);
    }

    match positional.len() {
        0 => Ok(Cli::Run { profile: None }),
        1 => Ok(Cli::Run {
            profile: Some(positional.remove(0)),
        }),
        n => Err(format!(
            "Expected at most one profile name (got {n}: {})",
            positional.join(" ")
        )),
    }
}

fn main() {
    let args: Vec<OsString> = std::env::args_os().collect();
    match parse_cli(&args) {
        Err(msg) => {
            eprintln!("{msg}\n\n{USAGE}");
            std::process::exit(2);
        }
        Ok(Cli::Help) => println!("{USAGE}"),
        Ok(cli) => {
            if let Err(e) = run(cli) {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }
}

#[tokio::main]
async fn run(cli: Cli) -> Result<(), Error> {
    // Init tracing before loading config so parse-time warnings render.
    // Only RUST_LOG is read from the environment (tracing convention).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    match cli {
        Cli::List => {
            print_profiles(&config::home_config_dir()?);
            Ok(())
        }
        // Help is answered synchronously in main(); nothing to do here.
        Cli::Help => Ok(()),
        Cli::Run { profile: Some(profile) } => run_server(&profile).await,
        Cli::Run { profile: None } => {
            // Bare launch: serve the recorded current profile (ADR-0007),
            // `default` when nothing has been recorded yet. An invalid or
            // deleted recorded profile errors exactly like an explicit
            // `ai-bridge <name>` — `run_server` validates and loads it.
            let base_dir = config::home_config_dir()?;
            let profile = config::resolve_bare_profile(&base_dir);
            run_server(&profile).await
        }
    }
}

/// Print the sorted profile list (`--list`). A missing or empty directory is
/// a friendly empty state, not an error; rendering lives in
/// [`profile_list_lines`] so the output is testable.
fn print_profiles(base_dir: &Path) {
    for line in profile_list_lines(base_dir) {
        println!("{line}");
    }
}

/// Lines printed by `--list`: one line per profile with `*` marking the
/// persisted current selection (only when that profile actually exists), plus
/// a hint line for the empty state.
fn profile_list_lines(base_dir: &Path) -> Vec<String> {
    let profiles = config::list_profiles(base_dir);
    if profiles.is_empty() {
        return vec![format!(
            "No profiles found in {}. Create {} to get started.",
            base_dir.display(),
            config::profile_path(base_dir, DEFAULT_PROFILE).display()
        )];
    }
    let current = config::current_profile(base_dir);
    let mut lines = vec![format!("Available profiles in {}:", base_dir.display())];
    for name in &profiles {
        let marker = if current.as_deref() == Some(name.as_str()) {
            "*"
        } else {
            " "
        };
        lines.push(format!("  {marker} {name}"));
    }
    lines
}

async fn run_server(profile: &str) -> Result<(), Error> {
    config::validate_profile_name(profile)?;
    let base_dir = config::home_config_dir()?;
    let loaded = config::load_profile(&base_dir, profile)?;
    // Record the current selection in ~/.ai-bridge/.settings.toml. Best
    // effort: warn, never fail, so a read-only config dir still serves
    // (same philosophy as write_default_template).
    if let Err(e) = config::save_current_profile(&base_dir, &loaded.name) {
        tracing::warn!(
            "Could not record \"{}\" as the current profile: {e}",
            loaded.name
        );
    }
    let LoadedProfile {
        name,
        path,
        config,
    } = loaded;
    let state = Arc::new(AppState::new(config)?);

    let addr = format!("{}:{}", state.config.listen_addr, state.config.listen_port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| Error::Server(format!("Failed to bind to {addr}: {e}")))?;

    println!("ai-bridge listening on {addr}");
    println!("  → profile   = {name} ({})", path.display());
    println!(
        "  → upstream  = {} {}",
        state.config.upstream_type.as_str(),
        state.config.url
    );
    println!("  → model     = {}", state.config.model);
    println!(
        "  → headers   = {} override(s)",
        state.config.override_headers.len()
    );
    match &state.config.vision {
        Some(vision) => println!(
            "  → vision    = {} (model: {})",
            vision.url, vision.model
        ),
        None => println!("  → vision    = (not configured)"),
    }
    println!(
        "  → vision supplement = {}",
        if state.config.vision_supplement_enabled {
            "ON (text-only upstreams get image descriptions)"
        } else {
            "OFF (images pass through to upstream vision)"
        }
    );
    println!(
        "  → reasoning = thinking={}, effort={}",
        if state.config.reasoning_policy.thinking_enabled {
            "on"
        } else {
            "off"
        },
        state.config.reasoning_policy.effort.describe()
    );

    let app = server::build_router(state);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| Error::Server(format!("Server error: {e}")))?;

    Ok(())
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!("shutdown signal received, waiting for in-flight requests to complete...");
}

#[cfg(test)]
mod tests {
    use super::{config, parse_cli, profile_list_lines, Cli};
    use std::ffi::OsString;
    use tempfile::tempdir;

    fn cli(args: &[&str]) -> Result<Cli, String> {
        let os: Vec<OsString> = std::iter::once(OsString::from("ai-bridge"))
            .chain(args.iter().map(OsString::from))
            .collect();
        parse_cli(&os)
    }

    const MINIMAL_TOML: &str =
        "upstream_type = \"oai-chat\"\nurl = \"https://x.example/v1/chat/completions\"\napi_key = \"k\"\n";

    fn write_profile(dir: &std::path::Path, name: &str) {
        std::fs::write(dir.join(format!("{name}.toml")), MINIMAL_TOML).expect("write profile");
    }

    #[test]
    fn profile_list_lines_marks_current() {
        let dir = tempdir().unwrap();
        write_profile(dir.path(), "a");
        write_profile(dir.path(), "b");
        config::save_current_profile(dir.path(), "a").unwrap();

        let lines = profile_list_lines(dir.path());
        assert!(lines.iter().any(|l| l == "  * a"), "a is current: {lines:?}");
        assert!(lines.iter().any(|l| l == "    b"), "b unmarked: {lines:?}");
    }

    #[test]
    fn profile_list_lines_no_star_when_current_missing() {
        let dir = tempdir().unwrap();
        write_profile(dir.path(), "a");
        // Point the settings file at a profile that no longer exists.
        config::save_current_profile(dir.path(), "ghost").unwrap();

        let lines = profile_list_lines(dir.path());
        assert!(lines.iter().any(|l| l == "    a"));
        assert!(!lines.iter().any(|l| l.starts_with("  *")), "no star: {lines:?}");
    }

    #[test]
    fn profile_list_lines_empty_dir_hint() {
        let dir = tempdir().unwrap();
        let lines = profile_list_lines(dir.path());
        assert_eq!(lines.len(), 1);
        assert!(lines[0].starts_with("No profiles found in"));
    }

    #[test]
    fn no_args_is_bare_run() {
        assert!(matches!(cli(&[]).unwrap(), Cli::Run { profile: None }));
    }

    #[test]
    fn positional_selects_profile() {
        assert!(
            matches!(cli(&["deepseek"]).unwrap(), Cli::Run { profile: Some(profile) } if profile == "deepseek")
        );
    }

    #[test]
    fn explicit_default_is_passed_through() {
        // `ai-bridge default` is explicit, not a bare launch: it must load
        // default even when another profile is recorded.
        assert!(
            matches!(cli(&["default"]).unwrap(), Cli::Run { profile: Some(profile) } if profile == "default")
        );
    }

    #[test]
    fn list_flags_are_recognized() {
        assert!(matches!(cli(&["-l"]).unwrap(), Cli::List));
        assert!(matches!(cli(&["--list"]).unwrap(), Cli::List));
    }

    #[test]
    fn help_flags_are_recognized_and_win_over_other_args() {
        assert!(matches!(cli(&["-h"]).unwrap(), Cli::Help));
        assert!(matches!(cli(&["--help", "-l"]).unwrap(), Cli::Help));
        assert!(matches!(cli(&["a", "-h"]).unwrap(), Cli::Help));
    }

    #[test]
    fn unknown_option_is_rejected() {
        assert!(cli(&["-x"]).is_err());
        assert!(cli(&["--bogus"]).is_err());
        // A dash-prefixed token is an option, never a profile name.
        assert!(cli(&["-weird-profile"]).is_err());
    }

    #[test]
    fn list_rejects_extra_arguments() {
        assert!(cli(&["-l", "extra"]).is_err());
    }

    #[test]
    fn multiple_positionals_are_rejected() {
        assert!(cli(&["a", "b"]).is_err());
    }

    #[test]
    fn non_utf8_argument_is_rejected() {
        use std::os::unix::ffi::OsStrExt;
        let bad = OsString::from(std::ffi::OsStr::from_bytes(b"\xff\xfe"));
        assert!(parse_cli(&[OsString::from("ai-bridge"), bad]).is_err());
    }
}
