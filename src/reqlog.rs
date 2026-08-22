//! Always-on compact request/response logger.
//!
//! Prints a single-line streaming log per request, independent of `RUST_LOG`:
//!
//! ```text
//! [REQ #0001]: claude-sonnet-5 hello, please do something ...
//! [RESP #0001]: Hello! This is the response text, streamed live onto one line...
//! ```
//!
//! Streaming text is appended onto the same open `[RESP #id]: ` line (no repeated
//! header per chunk) and terminated with a newline on completion. Errors break to
//! a tagged `[ERR-REQ #id]` / `[ERR-RESP #id]` line.
//!
//! Partial-line appends can't go through the line-oriented `tracing_subscriber::fmt`,
//! so this writes directly to stdout under a global mutex. Each `ReqLog` is shared
//! via `Arc` between the request handler and its stream coroutine, and flag bit
//! flips use `AtomicBool` (not `Cell`) so `Arc<ReqLog>` stays `Send + Sync` and the
//! generator remains sendable across threads. Concurrent streams are disambiguated
//! by their short hex id.

use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Global stdout lock shared by every `ReqLog` so concurrent appends are atomic.
static GLOBAL_LOCK: Mutex<()> = Mutex::new(());

/// The last system prompt emitted via `[REQ SYSTEM]`, shared across requests so
/// an unchanged prompt is only printed once (not on every request).
static LAST_SYSTEM_PROMPT: Mutex<Option<String>> = Mutex::new(None);

/// Print a `[REQ SYSTEM]: {prompt}` line only when the system prompt differs from
/// the previously emitted one. Called once per request before `[REQ #id]`.
///
/// The prompt is compared against the last one we printed; on a match it's
/// suppressed, so an unchanged system prompt appears just once per session/cache.
pub fn report_system_prompt(system: Option<&str>) {
    let Some(sys) = system.map(str::trim).filter(|s| !s.is_empty()) else {
        return;
    };

    let mut last = LAST_SYSTEM_PROMPT.lock().unwrap();
    if last.as_deref() == Some(sys) {
        return;
    }
    last.replace(sys.to_string());

    // Hold the global lock while writing so the line isn't torn with concurrent
    // streamed `[RESP]` appends.
    let _g = GLOBAL_LOCK.lock().unwrap();
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(format!("[REQ SYSTEM]: {sys}\n").as_bytes());
    let _ = lock.flush();
}

/// Monotonic id counter. `{:04x}` yields `0001`, `0002`, ... (wraps at `ffff`).
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

pub struct ReqLog {
    id: String,
    header_printed: AtomicBool,
    done: AtomicBool,
}

impl ReqLog {
    /// Allocate a fresh short hex id and return an `Arc` handle for one request.
    pub fn new() -> Arc<ReqLog> {
        let n = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let id = format!("{:04x}", n & 0xffff);
        Arc::new(ReqLog {
            id,
            header_printed: AtomicBool::new(false),
            done: AtomicBool::new(false),
        })
    }

    /// Atomic output primitive: hold the global lock, write, flush (so streamed
    /// text appears live).
    fn write(&self, s: &str) {
        let _g = GLOBAL_LOCK.lock().unwrap();
        let stdout = std::io::stdout();
        let mut lock = stdout.lock();
        let _ = lock.write_all(s.as_bytes());
        let _ = lock.flush();
    }

    /// Request summary line (complete, newline-terminated). Preceded by two
    /// blank lines so each request/response pair is visually separated in the
    /// terminal.
    pub fn req(&self, summary: &str) {
        self.write(&format!("\n\n[REQ #{}]: {summary}\n", self.id));
    }

    /// Append a note to the just-printed `[REQ #id]` line (no leading newline),
    /// e.g. a media-handling summary. Call immediately after `req`.
    pub fn append_media_note(&self, note: &str) {
        self.write(note);
    }

    /// Begin a response line: prints `[RESP #id]: {prefix}` with NO trailing newline.
    /// For streaming, call with `""` so text appends directly after `": "`.
    /// Preceded by two blank lines to visually separate the response from the
    /// previous request/response pair.
    pub fn resp_header(&self, prefix: &str) {
        if self.header_printed.swap(true, Ordering::SeqCst) {
            return;
        }
        self.write(&format!("\n\n[RESP #{}]: {prefix}", self.id));
    }

    /// Append text to the open response line (no newline). No-op if never started.
    pub fn append(&self, text: &str) {
        if !self.header_printed.load(Ordering::SeqCst) || self.done.load(Ordering::SeqCst) {
            return;
        }
        self.write(text);
    }

    /// Non-streaming single-shot response line (complete, newline-terminated).
    pub fn resp(&self, summary: &str) {
        self.resp_header("");
        self.write(summary);
        self.done();
    }

    /// Terminate an open response line with a newline.
    pub fn done(&self) {
        if self.done.swap(true, Ordering::SeqCst) {
            return;
        }
        self.write("\n");
    }

    /// Request-phase error (no response started) — full tagged line. Preceded
    /// by two blank lines so it separates from the previous pair just like the
    /// `[REQ #id]` / `[RESP #id]` markers.
    pub fn err_req(&self, msg: &str) {
        self.write(&format!("\n\n[ERR-REQ #{}]: {msg}\n", self.id));
    }

    /// Response-phase error: break the open line (first newline), then print a
    /// tagged line — also preceded by two blank lines for consistent grouping.
    pub fn err_resp(&self, msg: &str) {
        self.write(&format!("\n\n[ERR-RESP #{}]: {msg}\n", self.id));
        self.done();
    }
}
