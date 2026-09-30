# The request log mirrors content verbatim, to stdout or to a file

Every request and response is written as it is — including base64 media payloads — so the log is a complete mirror of what the proxy saw and forwarded. Nothing is summarised or redacted, and the log stays independent of `RUST_LOG`.

**Two destinations, one stream.** With no option the log goes to stdout. With `--log <path>` it is *teed*: the same lines still reach stdout and are appended to the file as well. Tee is chosen over file-only because it strictly contains the other behaviour — an operator who wants only the file discards stdout (`>/dev/null`, or a background run), while a built-in file-only mode would make "watch it live *and* keep it" impossible without falling back to the shell redirection this replaces.

The file is opened for append at startup and never rotated by the proxy; rotation stays the operator's business. A path that cannot be opened **fails startup**: the operator asked for a file, and quietly writing elsewhere would hide the loss. A write failure *after* startup warns once and lets serving continue — the same "serving outranks bookkeeping" rule as ADR-0006.

Summarising content would put a judgement call between the operator and the data, and the log's main job is answering "what exactly was sent, and what came back?" after a bad conversion — a question a summary cannot answer. Fidelity beats legibility: a terminal filling with a base64 blob is an accepted cost, and how much of it to read is the reader's call.
