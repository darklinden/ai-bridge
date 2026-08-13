mod convert;
mod convert_reverse;
mod error;
mod forward;
mod json_canonical;
mod media_sanitizer;
mod reasoning_bridge;
mod reqlog;
mod responses_reverse;
mod server;
mod streaming_responses;
mod tool_media;
mod transform_responses;
mod vision;

use crate::error::Error;
use crate::forward::AppState;
use std::path::PathBuf;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Error> {
    // Load `.env` before tracing init and config parsing so `Config::from_env`
    // sees any values it provides. Existing env vars take precedence (dotenv
    // does not override), and a missing `.env` is not an error.
    load_dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    // Read configuration from environment and build shared HTTP client
    let state = AppState::from_env()?;
    let state = Arc::new(state);

    let addr = format!("{}:{}", state.config.listen_addr, state.config.listen_port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| Error::Server(format!("Failed to bind to {addr}: {e}")))?;

    println!("ai-bridge listening on {addr}");
    println!("  → POST /v1/messages           (Anthropic Messages)");
    println!("  → POST /v1/chat/completions   (OpenAI Chat Completions)");
    println!("  → POST /v1/responses          (OpenAI Responses)");
    println!("  → UPSTREAM_TYPE = {:?}", state.config.upstream_type);
    println!("  → UPSTREAM_URL = {}", state.config.url);
    println!("  → UPSTREAM_MODEL = {}", state.config.model);
    println!(
        "  → UPSTREAM_HEADERS = {} override(s)",
        state.config.override_headers.len()
    );
    if let Some(vision) = &state.config.vision {
        println!(
            "  → VISION_URL = {} (model: {})",
            vision.url, vision.model
        );
    } else {
        println!("  → VISION = (not configured)");
    }

    let app = server::build_router(state);

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| Error::Server(format!("Server error: {e}")))?;

    Ok(())
}

/// Load `.env` into the process environment, preferring the Cargo manifest
/// directory (dev: `cargo run`) and falling back to the executable's directory
/// (deploy). Uses `from_path` (no override) so an already-exported env var wins.
/// Missing `.env` is silently ignored.
fn load_dotenv() {
    let mut candidates = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".env")];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join(".env"));
        }
    }
    for path in candidates {
        if path.is_file() {
            match dotenvy::from_path(&path) {
                Ok(_) => println!("Loaded .env from {}", path.display()),
                Err(e) => eprintln!("Warning: failed to load .env from {}: {e}", path.display()),
            }
            return;
        }
    }
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
