//! Always-on compact request/response logger.
//!
//! Prints a line-oriented streaming log per request, independent of `RUST_LOG`:
//!
//! ```text
//! [REQ #0001]: claude-sonnet-5 hello, please do something ... [body=412B stream=on]
//! [RESP #0001] after=1834ms: Hello! This is the response text, streamed live...
//! ```
//!
//! Streaming text is appended onto the same open `[RESP #id] after={ms}: ` line (no
//! repeated header per chunk) and terminated with a newline on completion. The
//! `after=` figure is the wait from the `[REQ]` line to the first response byte:
//! upload/queue/prefill all land in it, which is what makes a slow turn
//! attributable. Errors break to
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
    let head = tool_use_head(name);
    if args.is_empty() {
        head
    } else {
        format!("{head} {args}")
    }
}

/// The head of a call's marker, `[tool_use: {name}]`, with `?` for an absent or
/// empty name. Streamed calls put this on the line first and append their
/// arguments behind it, so the head and the full marker are spelled together.
pub(crate) fn tool_use_head(name: &str) -> String {
    let name = if name.is_empty() { "?" } else { name };
    format!("[tool_use: {name}]")
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
/// A stream does not have to wait for the block to close before writing the
/// call. As soon as the name is known, [`ToolCall::begin`] puts
/// `[tool_use: {name}]` on the log line; while the call owns the line (see
/// [`ToolLine`]) each argument delta reaches it through
/// [`ToolCall::stream_args`], so a long tool call shows which call is in flight
/// instead of leaving the line silent. [`ToolCall::settle`] writes whatever has
/// not reached the line when the block closes. A call that never learns its name
/// keeps buffering and is settled in one piece, exactly as before.
///
/// However the text arrives, a finished call reads `[tool_use: {name}] {args}`
/// — the shape [`tool_use_marker`] renders — so the record is unchanged.
#[derive(Debug, Clone, Default)]
pub(crate) struct ToolCall {
    pub(crate) name: String,
    pub(crate) args: String,
    /// Bytes of `args` already written after the head; `None` until the head is
    /// out. The offset is always `0` or the `args.len()` of an earlier value, so
    /// it stays a char boundary; `resolve_args` re-checks the printed prefix
    /// before it replaces the text.
    printed: Option<usize>,
}

impl ToolCall {
    /// A call whose block just opened, carrying any arguments that arrived
    /// before the name did.
    pub(crate) fn open(name: &str, args: &str) -> Self {
        ToolCall {
            name: name.to_string(),
            args: args.to_string(),
            printed: None,
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
    ///
    /// Once part of the call is on the log line it cannot be retracted, so a
    /// terminal value is only accepted when it extends what was already written;
    /// a contradictory one leaves the streamed text in place.
    pub(crate) fn resolve_args(&mut self, args: &str) {
        if args.len() <= self.args.len() {
            return;
        }
        if let Some(printed) = self.printed {
            let prefix = &self.args[..printed];
            if !args.starts_with(prefix) {
                return;
            }
        }
        self.args.clear();
        self.args.push_str(args);
    }

    /// The `[tool_use: {name}]` head, once, as soon as the name is known.
    /// `None` afterwards, and for a nameless block — its `?` only ever appears
    /// in the close-time marker.
    pub(crate) fn begin_fragment(&mut self) -> Option<String> {
        if self.printed.is_some() || self.name.is_empty() {
            return None;
        }
        self.printed = Some(0);
        Some(tool_use_head(&self.name))
    }

    /// The not-yet-written tail of the arguments: `" {args}"` for the first
    /// bytes after the head, the raw remainder after that. `None` when nothing
    /// is new or the head is not out yet.
    pub(crate) fn tail_fragment(&mut self) -> Option<String> {
        let printed = self.printed?;
        debug_assert!(printed <= self.args.len(), "printed offset past the arguments");
        let rest = &self.args[printed..];
        if rest.is_empty() {
            return None;
        }
        let fragment = if printed == 0 {
            format!(" {rest}")
        } else {
            rest.to_string()
        };
        self.printed = Some(self.args.len());
        Some(fragment)
    }

    /// Accumulate one delta and return the fragment that carries it to the log.
    pub(crate) fn push_fragment(&mut self, delta: &str) -> Option<String> {
        self.push_args(delta);
        self.tail_fragment()
    }

    /// Finish the call, returning whatever is still missing from the log line —
    /// the whole marker for a call that never started streaming, the argument
    /// tail otherwise.
    pub(crate) fn settle_fragment(&mut self) -> Option<String> {
        if self.printed.is_none() {
            let marker = self.marker();
            self.printed = Some(self.args.len());
            return Some(marker);
        }
        self.tail_fragment()
    }

    // --- the same three steps, writing to a `ReqLog` directly --------------

    /// Put the head on the log line, once.
    pub(crate) fn begin(&mut self, reqlog: &ReqLog) {
        if let Some(fragment) = self.begin_fragment() {
            reqlog.append(&fragment);
        }
    }

    /// One argument delta arrives: accumulate it and, when this call owns the
    /// line, append the new text. A call that never claimed the line only
    /// accumulates — see [`ToolLine`] for why only one call may write.
    pub(crate) fn stream_args(&mut self, reqlog: &ReqLog, delta: &str) {
        self.push_args(delta);
        if let Some(fragment) = self.tail_fragment() {
            reqlog.append(&fragment);
        }
    }

    /// Whether this call already has text on the log line.
    pub(crate) fn is_streaming(&self) -> bool {
        self.printed.is_some()
    }

    /// Write whatever has not reached the log line yet.
    pub(crate) fn settle(&mut self, reqlog: &ReqLog) {
        if let Some(fragment) = self.settle_fragment() {
            reqlog.append(&fragment);
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

/// Arbitration for one response line's tool text.
///
/// The line is append-only: once a call's arguments are on it, a second call
/// cannot start writing without each one's text landing under the other's head.
/// So exactly one call may own the line. A call that finds it busy keeps
/// buffering; if it finishes before the owner, its finished marker waits in the
/// backlog and is written the moment the line is released — so the line still
/// ends up as a run of complete `[tool_use: {name}] {arguments}` markers, each
/// call's arguments under its own head.
///
/// Sequential calls (the normal shape) never touch the backlog: each one claims
/// the line when its block opens and releases it when the block closes.
#[derive(Debug, Default)]
pub(crate) struct ToolLine<K> {
    active: Option<K>,
    backlog: Vec<String>,
}

impl<K: PartialEq + Clone> ToolLine<K> {
    /// Whether this call owns the line (it is the one writing).
    pub(crate) fn owns(&self, key: &K) -> bool {
        self.active.as_ref() == Some(key)
    }

    /// Whether any call owns the line (the caller checks [`ToolLine::owns`]
    /// first when it needs "someone else").
    pub(crate) fn is_busy(&self) -> bool {
        self.active.is_some()
    }

    /// Claim the line for `key` when it is free. `false` means the call must
    /// stay buffered and settle in one piece later.
    pub(crate) fn claim(&mut self, key: K) -> bool {
        if self.active.is_some() {
            return false;
        }
        self.active = Some(key);
        true
    }

    /// A finished call that could not write: hold its text until the line frees.
    pub(crate) fn hold(&mut self, text: String) {
        self.backlog.push(text);
    }

    /// Give the line back, returning the finished calls that were waiting for it.
    /// A key that does not own the line gets nothing: releasing on its behalf
    /// would put held text under someone else's call.
    pub(crate) fn release(&mut self, key: &K) -> Vec<String> {
        if self.active.as_ref() != Some(key) {
            return Vec::new();
        }
        self.active = None;
        std::mem::take(&mut self.backlog)
    }

    /// Start `tool`'s entry on the line if the line is free, so a long call is
    /// visible while it streams. Returns whether the call now owns the line.
    ///
    /// A nameless call never claims: it has no head to write, so holding the line
    /// would only keep every other call from streaming.
    pub(crate) fn begin_tool(&mut self, key: &K, tool: &mut ToolCall, reqlog: &ReqLog) -> bool {
        if self.owns(key) {
            tool.begin(reqlog);
            return true;
        }
        if tool.name.is_empty() || !self.claim(key.clone()) {
            return false;
        }
        tool.begin(reqlog);
        true
    }

    /// Finish `tool`: write it now when the line is free or this call owns it,
    /// otherwise hold its finished marker until the owner releases the line.
    pub(crate) fn settle_tool(&mut self, key: &K, tool: &mut ToolCall, reqlog: &ReqLog) {
        if self.owns(key) {
            tool.settle(reqlog);
            for waiting in self.release(key) {
                reqlog.append(&waiting);
            }
            return;
        }
        if !self.is_busy() {
            tool.settle(reqlog);
            return;
        }
        if let Some(finished) = tool.settle_fragment() {
            self.backlog.push(finished);
        }
    }
}

/// Drain an index-keyed map of calls still waiting for their block to close,
/// settling them in key order (each writes only what is not on the line yet;
/// the line's owner, when one is among them, settles first and lets the rest
/// follow).
///
/// Called both at the natural end of a stream and on the error path — the log
/// line ends either way, and a call that already reached the client must not
/// vanish from the record.
pub(crate) fn flush_open_tools<K, S>(
    open: &mut std::collections::HashMap<K, ToolCall, S>,
    line: &mut ToolLine<K>,
    reqlog: &ReqLog,
) where
    K: Ord + Copy + Eq + Clone + std::hash::Hash,
    S: std::hash::BuildHasher,
{
    let mut leftover: Vec<K> = open.keys().copied().collect();
    leftover.sort_unstable();
    for key in leftover {
        if let Some(mut call) = open.remove(&key) {
            line.settle_tool(&key, &mut call, reqlog);
        }
    }
}

pub struct ReqLog {
    id: String,
    /// When this request's log was opened: the zero point of `[RESP] after=Nms`.
    started: std::time::Instant,
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
            started: std::time::Instant::now(),
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

    /// Begin a response line: prints `[RESP #id] after={ms}: {prefix}` with NO
    /// trailing newline, where `{ms}` is the wait since this log was opened (the
    /// `[REQ]` line) — the number that separates upstream queue/prefill from
    /// actual generation. For streaming, call with `""` so text appends directly
    /// after `": "`.
    /// Preceded by two blank lines to visually separate the response from the
    /// previous request/response pair.
    pub fn resp_header(&self, prefix: &str) {
        if self.header_printed.swap(true, Ordering::SeqCst) {
            return;
        }
        let after_ms = self.started.elapsed().as_millis();
        self.write(&format!("\n\n[RESP #{}] after={}ms: {prefix}", self.id, after_ms));
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
    fn resp_header_records_the_wait_before_the_first_byte() {
        let (log, buf) = ReqLog::captured();
        log.req("m hello [body=12B stream=on]");
        std::thread::sleep(std::time::Duration::from_millis(2));
        log.resp_header("");
        log.done();
        let text = buf.lock().unwrap().clone();
        assert!(text.contains("[REQ #"), "{text}");
        assert!(text.contains("[RESP #"), "{text}");
        // The `after=` figure is the whole point: it separates queue/prefill
        // from generation, so it must be present on every response line.
        assert!(text.contains("after="), "{text}");
        assert!(text.contains("ms: "), "{text}");
    }

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
    fn a_streamed_call_reproduces_the_marker_byte_for_byte() {
        let (log, buf) = ReqLog::captured();
        log.resp_header("");
        let mut call = ToolCall::open("bash", "");
        call.begin(&log);
        call.stream_args(&log, "{\"cmd\":");
        call.stream_args(&log, "\"ls\"}");
        call.settle(&log);

        let text = buf.lock().unwrap().clone();
        assert!(
            text.ends_with("[tool_use: bash] {\"cmd\":\"ls\"}"),
            "log was: {text}"
        );
        // Settling a finished call must not repeat any of it.
        let len = text.len();
        call.settle(&log);
        assert_eq!(buf.lock().unwrap().len(), len);
    }

    #[test]
    fn a_nameless_call_streams_nothing_until_it_settles() {
        let (log, buf) = ReqLog::captured();
        log.resp_header("");
        let mut call = ToolCall::open("", "");
        call.stream_args(&log, "{\"a\":1}");
        // No name, no head — the text waits rather than opening a `[tool_use: ]`.
        let text = buf.lock().unwrap().clone();
        assert!(!text.contains("[tool_use"), "log was: {text}");

        // A name that arrives late still lets everything already collected out.
        call.name = "f".to_string();
        call.stream_args(&log, "{\"b\":2}");
        call.settle(&log);
        let text = buf.lock().unwrap().clone();
        assert!(
            text.ends_with("[tool_use: f] {\"a\":1}{\"b\":2}"),
            "log was: {text}"
        );
    }

    #[test]
    fn resolve_args_cannot_rewrite_what_is_already_on_the_line() {
        let (log, buf) = ReqLog::captured();
        log.resp_header("");
        let mut call = ToolCall::open("f", "");
        call.begin(&log);
        call.stream_args(&log, "{\"a\":1");

        // Shorter than the deltas: ignored, the deltas are the fuller record.
        call.resolve_args("{}");
        // A terminal value that extends the printed prefix fills in the rest —
        // the truncated-delta case.
        call.resolve_args("{\"a\":1}");
        call.settle(&log);
        let text = buf.lock().unwrap().clone();
        assert!(text.ends_with("[tool_use: f] {\"a\":1}"), "log was: {text}");

        // A different document cannot retract what is already written, so the
        // streamed text stays as it was.
        call.resolve_args("{\"a\":1,\"b\":2}");
        call.settle(&log);
        let text = buf.lock().unwrap().clone();
        assert!(text.ends_with("[tool_use: f] {\"a\":1}"), "log was: {text}");
        assert!(!text.contains("\"b\""), "log was: {text}");
    }

    #[test]
    fn one_call_at_a_time_owns_the_line() {
        let (log, buf) = ReqLog::captured();
        log.resp_header("");
        let mut line: ToolLine<u32> = ToolLine::default();
        let mut first = ToolCall::open("first", "");
        let mut second = ToolCall::open("second", "");

        assert!(line.begin_tool(&0, &mut first, &log));
        // The line is taken: the second call must not write its head yet.
        assert!(!line.begin_tool(&1, &mut second, &log));
        first.stream_args(&log, "{\"a\":");
        second.stream_args(&log, "{\"b\":2}");

        // The second one finishes first and waits for the line.
        line.settle_tool(&1, &mut second, &log);
        let text = buf.lock().unwrap().clone();
        assert!(!text.contains("second"), "log was: {text}");

        // Closing the owner writes its tail, then releases the line for the
        // waiting call, which lands complete and after it.
        first.stream_args(&log, "1}");
        line.settle_tool(&0, &mut first, &log);
        let text = buf.lock().unwrap().clone();
        assert!(
            text.ends_with("[tool_use: first] {\"a\":1}[tool_use: second] {\"b\":2}"),
            "log was: {text}"
        );
    }

    #[test]
    fn a_nameless_call_does_not_hold_the_line() {
        let (log, buf) = ReqLog::captured();
        log.resp_header("");
        let mut line: ToolLine<u32> = ToolLine::default();
        let mut nameless = ToolCall::open("", "");
        let mut named = ToolCall::open("f", "");

        // No head to write, so the nameless call must not take the line.
        assert!(!line.begin_tool(&0, &mut nameless, &log));
        assert!(line.begin_tool(&1, &mut named, &log));
        named.stream_args(&log, "{}");

        // The nameless call closes first: it waits, then lands behind the call
        // that owns the line.
        nameless.push_args("{\"a\":1}");
        line.settle_tool(&0, &mut nameless, &log);
        line.settle_tool(&1, &mut named, &log);

        let text = buf.lock().unwrap().clone();
        assert!(
            text.ends_with("[tool_use: f] {}[tool_use: ?] {\"a\":1}"),
            "log was: {text}"
        );
    }

    #[test]
    fn three_calls_keep_their_own_arguments() {
        let (log, buf) = ReqLog::captured();
        log.resp_header("");
        let mut line: ToolLine<u32> = ToolLine::default();
        let mut a = ToolCall::open("a", "");
        let mut b = ToolCall::open("b", "");
        let mut c = ToolCall::open("c", "");

        assert!(line.begin_tool(&0, &mut a, &log));
        assert!(!line.begin_tool(&1, &mut b, &log));
        assert!(!line.begin_tool(&2, &mut c, &log));
        a.stream_args(&log, "{\"a\":");
        b.stream_args(&log, "{\"b\":2}");
        c.stream_args(&log, "{\"c\":3}");

        // Both later calls finish first; they queue in that order.
        line.settle_tool(&1, &mut b, &log);
        line.settle_tool(&2, &mut c, &log);

        a.stream_args(&log, "1}");
        line.settle_tool(&0, &mut a, &log);

        let text = buf.lock().unwrap().clone();
        assert!(
            text.ends_with(
                "[tool_use: a] {\"a\":1}[tool_use: b] {\"b\":2}[tool_use: c] {\"c\":3}"
            ),
            "log was: {text}"
        );
    }

    #[test]
    fn a_sequential_call_takes_the_line_after_the_previous_one() {
        let (log, buf) = ReqLog::captured();
        log.resp_header("");
        let mut line: ToolLine<u32> = ToolLine::default();

        let mut first = ToolCall::open("first", "");
        assert!(line.begin_tool(&0, &mut first, &log));
        first.stream_args(&log, "{}");
        line.settle_tool(&0, &mut first, &log);

        let mut second = ToolCall::open("second", "");
        assert!(line.begin_tool(&1, &mut second, &log));
        second.stream_args(&log, "{}");
        line.settle_tool(&1, &mut second, &log);

        let text = buf.lock().unwrap().clone();
        assert!(text.ends_with("[tool_use: first] {}[tool_use: second] {}"), "log was: {text}");
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
        let mut line: ToolLine<u32> = ToolLine::default();
        flush_open_tools(&mut open, &mut line, &reqlog);
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
        // The header carries the `after=` wait, so the appended text follows
        // `ms: ` rather than the bare `]: ` of the old format.
        assert!(text.contains("ms: [tool_use: bash] {}\n"), "{text}");
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
