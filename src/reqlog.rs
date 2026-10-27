//! Always-on compact request/response logger.
//!
//! Prints a line-oriented streaming log per request, independent of `RUST_LOG`:
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
//! Tool calls and their results are part of those same lines, rendered by
//! [`tool_use_marker`] / [`tool_result_marker`]: `[tool_use: name] {arguments}` and
//! `[tool_result: id] {content}`, arguments verbatim and untruncated (ADR-0012).
//! A tool payload is the one part written as it came, line breaks included, so a
//! record can span more than one line; ordinary prose is folded onto one line by
//! the caller that owns the line.
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

/// Render a tool call for the log: `[tool_use: {name}] {args}`.
///
/// `args` is the upstream's own argument text — an Anthropic `input` serialised
/// as JSON, or a Chat/Responses `arguments` string — written verbatim and
/// without truncation (ADR-0012). Empty `args` degrades to the bare marker, and
/// an absent or empty `name` to `?`: a call the upstream never named still
/// reached the client, so it is recorded rather than hidden.
///
/// This is the single place the log spells a tool call, so the request line and
/// every response path (streaming or not) render one identically.
pub(crate) fn tool_use_marker(name: &str, args: &str) -> String {
    let name = if name.is_empty() { "?" } else { name };
    if args.is_empty() {
        format!("[tool_use: {name}]")
    } else {
        format!("[tool_use: {name}] {args}")
    }
}

/// Render a tool result for the log: `[tool_result: {id}] {content}`.
///
/// The counterpart of [`tool_use_marker`], with the same verbatim, untruncated
/// content rule (ADR-0012) and the same `?` for an absent or empty id.
pub(crate) fn tool_result_marker(id: &str, content: &str) -> String {
    let id = if id.is_empty() { "?" } else { id };
    if content.is_empty() {
        format!("[tool_result: {id}]")
    } else {
        format!("[tool_result: {id}] {content}")
    }
}

/// An in-flight tool call: the name the upstream announced, plus the argument
/// text accumulated from its deltas.
///
/// A stream cannot render the marker when the block opens — the arguments only
/// arrive afterwards — so it keeps one of these per open block and hands it to
/// [`tool_use_marker`] when the block completes.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolCall {
    pub(crate) name: String,
    pub(crate) args: String,
}

impl ToolCall {
    /// A call whose block just opened, carrying any arguments that arrived
    /// before the name did.
    pub(crate) fn open(name: &str, args: &str) -> Self {
        ToolCall {
            name: name.to_string(),
            args: args.to_string(),
        }
    }

    /// Append one argument delta.
    pub(crate) fn push_args(&mut self, delta: &str) {
        self.args.push_str(delta);
    }

    /// Take the arguments from a terminal frame, which carries the whole string
    /// rather than a delta. Whichever is longer wins: a stream that only sends
    /// the terminal frame has nothing accumulated, and a truncated delta run must
    /// not shrink what is already there.
    pub(crate) fn resolve_args(&mut self, args: &str) {
        if args.len() > self.args.len() {
            self.args.clear();
            self.args.push_str(args);
        }
    }

    /// The complete marker for this call. An upstream that never named the tool
    /// renders as `[tool_use: ?]` rather than a bare `[tool_use: ]` (see
    /// [`tool_use_marker`]), so an unnamed call is recorded wherever it was
    /// forwarded instead of vanishing.
    pub(crate) fn marker(&self) -> String {
        tool_use_marker(&self.name, &self.args)
    }
}

/// Write one marker per call, in the order given.
///
/// This is the single place a flushed tool call reaches the log, so every path
/// writes the vocabulary in the same shape.
pub(crate) fn append_markers<I: IntoIterator<Item = ToolCall>>(reqlog: &ReqLog, calls: I) {
    for call in calls {
        reqlog.append(&call.marker());
    }
}

/// Drain an index-keyed map of calls still waiting for their block to close,
/// writing them in key order.
///
/// Called both at the natural end of a stream and on the error path — the log
/// line ends either way, and a call that already reached the client must not
/// vanish from the record.
pub(crate) fn flush_open_tools<K, S>(
    open: &mut std::collections::HashMap<K, ToolCall, S>,
    reqlog: &ReqLog,
) where
    K: Ord + Copy + Eq + std::hash::Hash,
    S: std::hash::BuildHasher,
{
    let mut leftover: Vec<K> = open.keys().copied().collect();
    leftover.sort_unstable();
    let calls: Vec<ToolCall> = leftover
        .into_iter()
        .filter_map(|key| open.remove(&key))
        .collect();
    append_markers(reqlog, calls);
}

pub struct ReqLog {
    id: String,
    header_printed: AtomicBool,
    done: AtomicBool,
    /// The buffer a captured handle diverts its output into, installed by
    /// `captured` so a test can assert the assembled `[RESP]` line. `None` in
    /// production, and absent from the build entirely, so nothing sits between
    /// the write and stdout.
    #[cfg(test)]
    capture: Mutex<Option<Arc<Mutex<String>>>>,
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
            #[cfg(test)]
            capture: Mutex::new(None),
        })
    }

    /// A handle whose output goes into the returned buffer instead of stdout.
    #[cfg(test)]
    pub(crate) fn captured() -> (Arc<ReqLog>, Arc<Mutex<String>>) {
        let log = ReqLog::new();
        let buf = Arc::new(Mutex::new(String::new()));
        *log.capture.lock().unwrap() = Some(buf.clone());
        (log, buf)
    }

    /// Atomic output primitive: hold the global lock, write, flush (so streamed
    /// text appears live).
    fn write(&self, s: &str) {
        #[cfg(test)]
        if let Some(buf) = self.capture.lock().unwrap().as_ref() {
            // A captured handle diverts its output into the buffer instead of
            // the process's stdout, so a test asserts the assembled line without
            // the run's own output being interleaved into the test log.
            buf.lock().unwrap().push_str(s);
            return;
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_use_marker_appends_args_verbatim() {
        assert_eq!(
            tool_use_marker("bash", "{\"command\":\"ls\"}"),
            "[tool_use: bash] {\"command\":\"ls\"}"
        );
        // Empty args (a tool call whose arguments never arrived) -> bare marker.
        assert_eq!(tool_use_marker("bash", ""), "[tool_use: bash]");
        // `{}` is non-empty text, so it is kept — the reader can tell "no
        // arguments" from "arguments unknown".
        assert_eq!(tool_use_marker("bash", "{}"), "[tool_use: bash] {}");
        // Multi-byte and newlines are not escaped or collapsed here; the
        // single-line rule is applied by the caller that owns the line.
        assert_eq!(
            tool_use_marker("读", "{\n\"路径\": \"a\"\n}"),
            "[tool_use: 读] {\n\"路径\": \"a\"\n}"
        );
    }

    #[test]
    fn tool_call_marker_falls_back_to_a_question_mark_for_an_unnamed_call() {
        let mut call = ToolCall::default();
        call.push_args("{}");
        // A nameless call still reached the client, so it is recorded — with the
        // same `?` the request-line extractors use — rather than skipped.
        assert_eq!(call.marker(), "[tool_use: ?] {}");

        call.name = "bash".to_string();
        assert_eq!(call.marker(), "[tool_use: bash] {}");
    }

    #[test]
    fn flush_open_tools_writes_every_call_in_key_order() {
        let (reqlog, buf) = ReqLog::captured();
        reqlog.resp_header("");
        let mut open: std::collections::HashMap<u32, ToolCall> = std::collections::HashMap::new();
        // Inserted out of order: the flush is what gives the log a stable order.
        open.insert(2, ToolCall::open("c", "{}"));
        open.insert(0, ToolCall::open("a", "{}"));
        open.insert(1, ToolCall::open("b", "{}"));
        flush_open_tools(&mut open, &reqlog);
        reqlog.done();

        assert!(open.is_empty(), "the flush must drain the map");
        let text = buf.lock().unwrap().clone();
        assert!(
            text.contains("[tool_use: a] {}[tool_use: b] {}[tool_use: c] {}\n"),
            "log was: {text}"
        );
    }

    #[test]
    fn tool_call_resolve_args_keeps_the_more_complete_value() {
        let mut call = ToolCall::open("f", "");
        call.resolve_args("{\"a\":1}");
        assert_eq!(call.args, "{\"a\":1}");
        // A shorter terminal payload must not shrink what the deltas produced.
        call.resolve_args("{}");
        assert_eq!(call.args, "{\"a\":1}");
        // A longer one wins (a truncated delta run, completed by the terminal frame).
        call.resolve_args("{\"a\":1,\"b\":2}");
        assert_eq!(call.args, "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn markers_render_an_absent_name_or_id_as_a_question_mark() {
        // A call or a result the upstream never identified still reached the
        // client, so it is recorded rather than hidden.
        assert_eq!(tool_use_marker("", "{}"), "[tool_use: ?] {}");
        assert_eq!(tool_use_marker("", ""), "[tool_use: ?]");
        assert_eq!(tool_result_marker("", "out"), "[tool_result: ?] out");
    }

    #[test]
    fn tool_result_marker_variants() {
        assert_eq!(
            tool_result_marker("t_1", "file contents"),
            "[tool_result: t_1] file contents"
        );
        assert_eq!(tool_result_marker("t_1", ""), "[tool_result: t_1]");
    }

    #[test]
    fn captured_mirrors_every_write() {
        let (reqlog, buf) = ReqLog::captured();
        reqlog.req("model hello");
        reqlog.resp_header("");
        reqlog.append("[tool_use: bash] {}");
        reqlog.done();

        let text = buf.lock().unwrap().clone();
        assert!(text.starts_with("\n\n[REQ #"));
        assert!(text.contains("]: model hello\n"));
        assert!(text.contains("[RESP #"));
        assert!(text.contains("]: [tool_use: bash] {}\n"));
    }

    #[test]
    fn capture_does_not_leak_between_handles() {
        let (first, buf) = ReqLog::captured();
        let second = ReqLog::new();
        first.req("one");
        second.req("two");

        let text = buf.lock().unwrap().clone();
        assert!(text.contains("one"));
        assert!(!text.contains("two"));
    }
}
