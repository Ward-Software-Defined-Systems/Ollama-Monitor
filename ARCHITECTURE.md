# Architecture

## Goal

A single Rust binary that sits in front of a local Ollama as an HTTP reverse proxy, records per-request inference stats from the responses passing through, and shows them, with hypothetical frontier-API costs and hardware telemetry, in a terminal UI. It is an active proxy, byte-transparent for responses, with exactly one request rewrite (see [Request body rewrite](#request-body-rewrite-openai-compatible-only)). No daemon and no frontier API calls: the only network peer is the Ollama at `--ollama-url`.

## Process model

One OS process and one multi-threaded `tokio` runtime, plus a plain thread for keyboard input, the SQLite writer on tokio's blocking pool, and a GPU telemetry child (`sudo powermetrics` on macOS, `nvidia-smi` on Linux). Tasks talk over bounded `tokio::sync::mpsc` channels (capacities in parentheses); shutdown fans out through one `tokio::sync::watch::channel<bool>`.

```
clients ──HTTP──▶ proxy (axum) ──reqwest──▶ Ollama
                    │
                    │ tracked paths: a per-request driver task
                    │ tees each chunk into a parser::Accumulator;
                    │ the handler reports a 4xx / 5xx answer itself
                    ▼
       records_tx (256), failures_tx (256)
                    ▼
         ┌─────────────────────┐    persist     ┌──────────────────┐
         │ TUI run loop        │───────────────▶│ db writer        │──▶ usage.db
         │ (or headless loop)  │ DbHandle (256) │ (spawn_blocking) │
         └─────────────────────┘                └──────────────────┘
            ▲      ▲      ▲      ▲
            │      │      │      └─ lifetime_tx (4) ◀─ lifetime poller (2 s, second connection)
            │      │      └──────── key_tx (64) ◀──── input thread (crossterm)
            │      └─────────────── hw_tx (8) ◀────── hardware::sample ◀── telemetry child (sudo -n powermetrics / nvidia-smi)
            └────────────────────── models_tx (8) ◀── api::poll_models (2 s)

shutdown: watch<bool> reaches every task; q / Ctrl-C / SIGINT / SIGTERM set it
```

`main` spawns the proxy, the signal task (SIGINT, SIGTERM, SIGHUP) and, in TUI mode, the model poller and the hardware sampler. `tui::run` spawns the lifetime poller and the input thread, `db::open_and_spawn_writer` the writer, and the proxy one driver task per tracked response. Headless mode (`--no-tui`) runs only the proxy, the writer and the signal task: no poller, no sampler (so no telemetry child and, on macOS, no sudo), no lifetime poller.

## Modules

| file | role |
|---|---|
| `src/main.rs` | CLI, startup order, task wiring, signal handling, TUI vs headless, the 3 s shutdown drain |
| `src/proxy.rs` | `bind` (fail-fast listener) and `serve` (an axum fallback handler that forwards every path via reqwest); the OpenAI `include_usage` rewrite; the per-request driver task that tees tracked responses into the parser and emits `InferenceRecord`s; `report_failure`, which turns a 4xx / 5xx on a tracked path into a `FailedRequest` |
| `src/parser.rs` | `is_inference_path` (the tracked paths); `classify` picks an envelope from path + content type; `Accumulator` drains NDJSON / SSE frames as they arrive and `finalize()`s into `ParsedStats`; `ParsedStats::is_load_or_unload` |
| `src/api.rs` | `poll_models`: `GET /api/version`, then `/api/tags` + `/api/ps` concurrently; `merge_models` turns them into LMS-shaped `ModelInfo` rows (state loaded / cloud / not-loaded); `resolve_model_id` maps a record's model name to its row for the `▸` marker |
| `src/aggregate.rs` | The session's records in memory; 1m / 5m / 15m / session windows; mean / p50 / p95; per-model breakdown |
| `src/pricing.rs` | `FRONTIER_MODELS` (the cost columns); the baked-in `pricing.toml` merged with user overrides; `hypothetical_cost` |
| `src/db.rs` | `InferenceRecord` and `FailedRequest`; the SQLite schema (v2), the writer task behind `DbHandle`, sessions, `lifetime_totals` |
| `src/config.rs` | Per-platform directories for db / config / log (Application Support on macOS, XDG on Linux); the optional user TOML |
| `src/hardware/mod.rs` | `sysinfo` CPU / memory and Ollama process detection; `prime` and the shared reader task for the telemetry child, with the platform backend chosen at compile time |
| `src/hardware/powermetrics.rs` | macOS backend: `sudo -v`, the `sudo -n powermetrics` command and its GPU residency / ANE power parser |
| `src/hardware/nvidia_smi.rs` | Linux backend: the `nvidia-smi` loop command and its CSV parser (GPU utilization, VRAM used, aggregated across GPUs) |
| `src/tui/mod.rs` | `AppState`, the event loop, `render()`, the lifetime poller, the input thread, panic-safe terminal restore |
| `src/tui/layout.rs` | Top-to-bottom panel layout, byte-identical to LMS-Monitor's |
| `src/tui/widgets.rs` | Per-panel render functions, ported from LMS-Monitor; `FeedEntry`, a completed or failed request in the feed |

## Event source: the design pivot

The LM Studio version of this app observes inference passively via `lms log stream -s model --stats --json`, a JSON-Lines event stream LM Studio publishes for any third party to consume.

**Ollama has no equivalent.** Per-inference stats (`eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`, `total_duration`, `done_reason`) ride back to the calling client in the HTTP response body; nothing is broadcast to a third-party observer. The server log only emits GIN HTTP access lines (method, path, status, total duration).

So instead of subscribing to a log stream, this app **becomes** the path the response travels through:

- An axum reverse proxy listens on `--proxy-listen` (default `127.0.0.1:11435`) and forwards every request to `--ollama-url` (default `http://127.0.0.1:11434`).
- For tracked inference paths (`/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`), each upstream chunk is forwarded to the client unchanged and also fed to a per-request `parser::Accumulator`.
- When the response finishes, the accumulator is finalised; if it found usable stats, an `InferenceRecord` goes to `records_tx`.
- A 4xx or 5xx response on those paths isn't parsed; a `FailedRequest` goes to `failures_tx` instead.

Clients must point at the proxy port (or the user moves Ollama to another port and gives `:11434` to the proxy). That is the cost of capturing real per-inference stats. Direct hits on Ollama's port are not captured, by design.

## Proxy request path

`proxy::bind` parses `--proxy-listen` as a `SocketAddr` (an IP and port, no host names) and binds a `std::net::TcpListener`. `main` calls it before the telemetry prime step (the sudo prompt on macOS) and before the runtime starts, so a taken port or a malformed address exits with the error instead of leaving a dashboard with nothing behind it. `serve` hands the socket to tokio and runs axum with a single fallback handler for every method and path.

For each request the handler:

1. Picks how to send the request body. On the OpenAI rewrite paths (next section) it reads the whole body into memory, capped at 64 MiB (a bigger one gets 413, an unreadable one 400), and applies the rewrite. Every other body streams to Ollama as it arrives (`reqwest::Body::wrap_stream`), whatever its size, so an `ollama create` blob push of several gigabytes goes straight through.
2. Forwards the method, path + query and headers through one shared `reqwest::Client`, dropping the hop-by-hop headers and `Host`. `Content-Length` is kept for a streamed body, which arrives unchanged, and dropped for a buffered one, whose length reqwest sets after any rewrite. The client has no overall timeout (a long generation must not be cut off), a 90 s idle-pool timeout and `TCP_NODELAY`. It's built without decompression features, so bytes arrive exactly as Ollama sent them.
3. Returns Ollama's status and headers, minus the hop-by-hop headers and `Content-Length`, so response bodies always go out chunked.
4. Tees the body (see [Streaming response tee](#streaming-response-tee)) only when the status is 2xx and `parser::classify` recognises the path; everything else streams straight through.
5. On a tracked path, reports a 4xx or 5xx response as a failed request (see [Failed requests](#failed-requests)).

The proxy's own errors are 502 (Ollama unreachable), 413 (a buffered body over 64 MiB), 400 (a request body it can't read), 500 (a response it can't build) and 405 (a method reqwest can't represent). On a tracked path they're failed requests too.

Because reqwest sets `Host` from `--ollama-url`, Ollama never sees the client's `Host`. Ollama's DNS-rebinding protection (while it listens on loopback it accepts only local host names) therefore doesn't cover requests that arrive through the proxy. The client's `Origin` header is forwarded untouched, so Ollama's CORS check still applies.

## Request body rewrite (OpenAI-compatible only)

The proxy is byte-transparent for *responses* but does one targeted *request* rewrite. For paths containing `/v1/chat/completions` or `/v1/completions` whose JSON body doesn't set `stream: false` (a missing or non-boolean `stream` counts as streaming), the body is re-serialised with `stream_options.include_usage = true` injected, creating the `stream_options` object if it's absent. Without this, Ollama's OpenAI-compat SSE response omits the `usage` block entirely, and almost no client sets the option by default (Open WebUI, Continue, the raw OpenAI SDK, Cursor and others all skip it).

The injection is silent to clients: the extra final SSE chunk has `choices: []`, which standard OpenAI renderers iterate and ignore. Existing values inside `stream_options` are preserved (only `include_usage` is forced); a non-object `stream_options` is overwritten. Re-serialising goes through `serde_json::Value` without the `preserve_order` feature, so the forwarded body has its keys in alphabetical order and no insignificant whitespace: the same JSON, not the same bytes. Empty bodies (CORS preflights, GETs), non-JSON bodies, non-objects, non-streaming requests and other paths pass through untouched; a body that fails to parse is logged by path and length only, never content. See `proxy::ensure_openai_include_usage` and its tests.

This is the one place the proxy is *not* transparent. The trade-off (a one-key addition versus losing nearly every OpenAI-client request) is worth it; it's documented here and in the README so it isn't a surprise.

## Streaming response tee

Each tracked upstream response is a `futures_util::Stream<Item = Result<Bytes, _>>`, handled by `proxy::build_response_body`:

1. A per-request **driver task** owns the upstream stream and a `parser::Accumulator`. For each chunk it calls `acc.push()` and then forwards the chunk to the client through a bounded `mpsc` channel (64 chunks), whose receiver is the response body the client reads. The bound means a slow client still backpressures upstream reads.
2. The task stops reading when upstream ends, when upstream errors, or when the client goes away (`client_tx.closed()`, or a failed send).
3. **Client gone early.** If the stream had already logically finished (a `finish_reason` arrived) and only the trailing `usage` frame is missing (`Accumulator::awaiting_usage_trailer`), the task keeps draining upstream for up to `USAGE_TRAILER_GRACE` (3 s) so the record keeps real token counts. Agent clients hang up the moment they see `finish_reason`, and Ollama's `:cloud` relay sends usage a beat later, so for that traffic this is the normal path, not an edge case. A client that aborts mid-generation gets no drain: dropping the upstream stream is what propagates the cancellation to Ollama and the cloud relay.
4. The task then calls `finalize()`. Usable stats become an `InferenceRecord` (stamped `completed_at = now`) sent to `records_tx`. Ollama model load / unload calls (`ParsedStats::is_load_or_unload`) are the exception: they're logged at debug and dropped. No usable stats means an info log line, `no parsable stats in response`, and no record.

Failure isolation: parser trouble never changes what the client receives; the client gets every byte Ollama produced, in order. One known gap: an upstream error mid-body ends the client's response as if it were complete (the tee path closes the channel; the untracked path yields an empty chunk) instead of aborting the connection, so a client can't tell a truncated body from a whole one.

## Failed requests

`proxy::handle` wraps `forward`, which does everything in the request path above. When the path is tracked (`parser::is_inference_path`: the four inference paths) and the client's response is a 4xx or 5xx, `report_failure` builds a `FailedRequest`. Its fields are the time, path, status, model when known, and source. The source is `ollama` for a status Ollama sent, including one it relays from ollama.com for a `:cloud` model, and `proxy` for one of the proxy's own errors. A 3xx isn't a failure: gin answers a trailing slash (`/api/chat/`) with a redirect.

- **Status code only.** The error body is a response body: it streams to the client untouched, and nothing reads, logs or stores it.
- **Model from the request.** Only the OpenAI-compatible paths buffer their request bodies, so only they have a model. `ensure_openai_include_usage` returns the `model` field from the parse it already does, if it's a non-empty string of at most 256 bytes without control characters; headless mode prints it as is. Native bodies stream through unread, so a native failed request has no model and the feed shows `?`. Neither does a 413 or 400, whose body was never read. A failed row names the model the way the request did (`deepseek-v4-pro:cloud`); a completed row has the response's name (`deepseek-v4-pro`). The `▸` resolves both.
- **Synchronous reporting.** `report_failure` writes one info line (`inference request failed`, with path, status, model and source), then calls `try_send` on `failures_tx`. Nothing is awaited between learning the status and returning the response, so reporting can't delay the client. A full or closed channel drops the report with a warning.
- **Client hang-ups.** A client that hangs up before its response is ready isn't reported: hyper drops the handler with the connection. The exception is a client that aborts while a native body is still uploading. reqwest's send fails, so the request is reported as `proxy (HTTP 502)`.
- **Errors inside a 2xx stream.** An error that arrives after a 2xx response has started isn't a failed request, because the status was already sent. Examples are a native `{"error": …}` line or an error frame in an OpenAI stream. The stream ends like any cut-off stream (see the table below).

## Capture outcomes

### Envelopes

`parser::classify(path, content_type)` picks one of four envelopes; `finalize` can emit a fifth.

| envelope | path contains | content type | terminal record |
|---|---|---|---|
| `OllamaStream` | `/api/chat`, `/api/generate` | `application/x-ndjson`, `application/jsonl`, none, or anything else that isn't JSON | the NDJSON line with `done: true` |
| `OllamaSingle` | `/api/chat`, `/api/generate` | `application/json` | the single JSON object (it needs `done: true` or `eval_count`) |
| `OpenAiSse` | `/v1/chat/completions`, `/v1/completions` | `text/event-stream` | `usage` in a `data:` frame (forced on by the request rewrite); `finish_reason` may arrive in an earlier frame |
| `OpenAiSingle` | `/v1/chat/completions`, `/v1/completions` | anything but `text/event-stream` | the single JSON object with `usage` |
| `OpenAiSseNoUsage` *(from `finalize` only)* | | | an SSE stream that ended without a `usage` frame; stored as `"openai-sse-approx"` |

`OpenAiSseNoUsage` covers Ollama's `:cloud` relay dropping the usage frame ([ollama/ollama#15169](https://github.com/ollama/ollama/issues/15169)) and streams cut off before it arrived. Its `gen_tokens` is the count of SSE chunks with non-empty `choices` (a coarse estimate), and `prompt_tokens` is 0 because the prompt size never travels in the stream.

### Field mapping

| field | Ollama native | OpenAI-compatible |
|---|---|---|
| `model_id` | `model` | `model` (the first frame that has one) |
| `prompt_tokens` | `prompt_eval_count` | `usage.prompt_tokens` (0 when approximate) |
| `gen_tokens` | `eval_count` | `usage.completion_tokens` (the chunk count when approximate) |
| `tokens_per_sec` | `eval_count / (eval_duration / 1e9)` | `completion_tokens / gen_window` (below) |
| `ttft_sec` | `(prompt_eval_duration + load_duration) / 1e9` | proxy wall clock to the first body byte for SSE; 0 for non-streaming |
| `total_time_sec` | `total_duration / 1e9` | proxy wall clock from request receipt (at least 0.1 ms) |
| `stop_reason` | `done_reason` (`"stop"` if absent) | the last `choices[0].finish_reason` seen (`"unknown"` when approximate and none arrived; `""`, shown as `-`, when a non-streaming response has none) |
| `envelope` | `"ollama-stream"` / `"ollama-single"` | `"openai-sse"` / `"openai-single"` / `"openai-sse-approx"` |

For OpenAI SSE, `gen_window = wall_total - wall_ttft`, floored at 50 ms. For OpenAI non-streaming the proxy can't tell prompt time from generation time (all bytes arrive together), so `gen_window = wall_total` and tokens per second is end to end, a lower bound on the decode rate. Native TTFT comes from Ollama's own durations, so unlike the OpenAI wall-clock figures it leaves out time spent queued inside Ollama.

### What gets recorded

| situation | outcome |
|---|---|
| any envelope completes normally | recorded |
| client hangs up after `finish_reason` and `usage` follows within 3 s | recorded with real counts |
| OpenAI SSE ends without `usage` (cloud relay, upstream reset, client abort mid-generation, or `usage` later than 3 s after a hang-up) | recorded as `openai-sse-approx` |
| OpenAI SSE with no data frames at all | dropped |
| native stream ends before its `done: true` line (client abort, truncation) | dropped |
| native response to a model load / unload (`done_reason` `load` / `unload`) | dropped (debug log) |
| 4xx / 5xx response on a tracked path, from Ollama or the proxy | not parsed; saved to `failed_requests` and shown as a red feed row (see [Failed requests](#failed-requests)) |
| 3xx response (gin's trailing-slash redirect) | passed through, not parsed |
| embeddings (`/api/embed`), `/api/tags`, pulls, blobs and every other path | passed through, not parsed |

## Record lifecycle

Each captured request, and each failed one, is persisted exactly once, by whichever loop owns `records_rx` and `failures_rx`: the TUI run loop or, with `--no-tui`, the headless loop. Both send every record to `DbHandle::persist` and every failed request to `DbHandle::persist_failure`, whatever the UI is doing. The UI's view is separate and lossy by design:

- The feed keeps the newest 30 entries (`FEED_CAPACITY`), completed and failed alike, in arrival order.
- Failed requests go to the feed only: never the aggregator, the cost panel or the lifetime totals, which count `inference_records`. One with a known model moves the `▸`; one without leaves it where it was.
- `r` clears the feed and the aggregator; the database and the lifetime totals keep everything.
- While paused (`p`), records and failed requests are persisted but not ingested: they never reach the feed, the aggregator, the cost panel or the `▸` marker, even after resuming.

So `inference_records` and `failed_requests` are the complete history, and the panels show the session since launch or the last reset, minus anything that arrived while paused.

## Aggregator

`Aggregator` keeps the session's records in a `Vec` and builds a snapshot on demand: the 1m / 5m / 15m windows filter by `completed_at`, plus the whole session. Percentiles sort and index (`round((n - 1) × q)`). The tokens-per-second mean and p95 skip zero and non-finite values (requests that generated nothing); the TTFT mean includes every record, so non-streaming OpenAI requests (TTFT 0) pull it down.

The TUI rebuilds the snapshot on every redraw, at least four times a second, scanning every session record each time. That's cheap at hundreds or thousands of records but grows with the session. `p50_tps` and the per-model breakdown are computed for parity with LMS-Monitor but not displayed.

Cross-session totals come from `db::lifetime_totals` instead: the TUI's lifetime poller opens a second SQLite connection and queries it every 2 s. That keeps the aggregator focused on the live session and avoids loading history at startup.

## Persistence

SQLite via bundled `rusqlite`, in the default rollback-journal mode. On every open the schema is applied idempotently: `V1_SQL` (frozen), then `V2_SQL`, both `CREATE … IF NOT EXISTS`. `V2_SQL` ends with an `INSERT OR IGNORE` that gives `schema_version` one row per version applied, 1 and 2.

| table | columns |
|---|---|
| `sessions` | `id` INTEGER PK, `started_at` TEXT, `ended_at` TEXT (NULL until a clean shutdown) |
| `inference_records` | `id` INTEGER PK, `session_id` → `sessions.id`, `completed_at` TEXT, `model_id` TEXT, `prompt_tokens` INTEGER, `gen_tokens` INTEGER, `tokens_per_sec` REAL, `ttft_sec` REAL, `total_time_sec` REAL, `stop_reason` TEXT, `envelope` TEXT |
| `failed_requests` (v2) | `id` INTEGER PK, `session_id` → `sessions.id`, `failed_at` TEXT, `model_id` TEXT (NULL when unknown), `path` TEXT, `status` INTEGER, `source` TEXT (`ollama` or `proxy`) |
| `schema_version` | `version` INTEGER PK |

Timestamps are RFC 3339 strings in UTC (`2026-10-03T23:53:26.322907+00:00`). `envelope` is one of the five strings in the mapping table above; filter on `openai-sse-approx` to separate estimated rows. Indexes cover `inference_records(session_id)`, `inference_records(model_id)` and `failed_requests(session_id)`.

v2 only added `failed_requests`, so there is no migration step: opening a v1 file creates the table and records version 2. A monitor built before v2 keeps working on a v2 file, even while a newer one runs on it. Its schema SQL is all no-ops there, its version check (`SELECT version … LIMIT 1`) finds 1, and it never touches the new table. Each statement commits on its own under the 5 s busy timeout, so an upgrade next to a running writer just waits its turn. LMS-Monitor's v2 needed an `IMMEDIATE` transaction only because it alters a table after reading it.

All writes go through one writer: `db::open_and_spawn_writer` runs a loop on tokio's blocking pool (`spawn_blocking` + `blocking_recv`) that owns the connection, and `DbHandle` is a cloneable `mpsc::Sender` (256) for its commands. Each record and each failed request is its own autocommit `INSERT`. The lifetime poller's second connection is an ordinary read-write one that only ever reads. The writer keeps rusqlite's default 5 s busy timeout and the reader sets 500 ms, which covers the brief locks rollback journaling takes.

A session row is inserted at startup and its `ended_at` filled in on a clean shutdown (see [Startup and shutdown](#startup-and-shutdown)). Closing the terminal window (SIGHUP) and a TUI error both still end the session cleanly; only a crash or a `kill -9` leaves it NULL.

## Pricing

[`pricing.toml`](./pricing.toml) is compiled in with `include_str!`. `FRONTIER_MODELS` decides which models get a cost column, in order:

```rust
pub const FRONTIER_MODELS: &[&str] = &[
    "claude-fable-5-1",
    "claude-fable-5",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "gemini-3-1-pro",
];
```

`pricing::load` flattens every provider's models into one map keyed by model, then applies the user's `[pricing.providers.<any>.models.<key>]` overrides on top. Provider names are ignored, and each entry needs both `input_per_mtok_usd` and `output_per_mtok_usd`. If the override table doesn't deserialize, all of it is ignored with a warning; a frontier key with no rate logs a warning, and its column shows `(no rate)`. Keys are TOML-friendly (hyphens, no dots: `gemini-3-1-pro`, not `gemini-3.1-pro`), and a mistyped override key is silently unused.

Pricing is flat-rate. Gemini's over-200K-token tier isn't modelled; override the rate to use it.

Changing the list means editing all of these together: `pricing.toml` (and its "updated" date), `FRONTIER_MODELS`, the `defaults_match_pricing_toml` test, the `cost_panel_is_transposed` assertion in `src/tui/mod.rs`, the README feature list and this section, and LMS-Monitor's copy, whose table is kept identical.

## Hardware sampling

Two sources, merged into one `HardwareSnapshot` every 2 s. The sampler runs only in TUI mode.

- **CPU, memory and Ollama's processes** via `sysinfo`, no privilege needed. A process counts as Ollama's if its executable path or `argv[0]` contains one of:
  - `/Applications/Ollama.app/` (macOS desktop install; this also catches the app's own menu-bar process)
  - `/opt/homebrew/opt/ollama/`, `/opt/homebrew/Cellar/ollama/` (Apple Silicon Homebrew)
  - `/usr/local/opt/ollama/`, `/usr/local/Cellar/ollama/` (Intel Homebrew)
  - `/usr/local/lib/ollama/`, `/usr/lib/ollama/`, `/snap/ollama/` (Linux install script, distro packages, snap; older releases kept their `ollama_llama_server` runners under the first)
  - `/.ollama/` (model store / runner workdir)

  or if its executable name is exactly `ollama`. That catches the `ollama serve` parent and the `ollama runner` subprocesses, where a loaded model's memory lives. The name match is deliberately exact: a prefix match would count `ollama-monitor` itself. The hints are directories for the same reason: a bare `/usr/local/bin/ollama` would match `/usr/local/bin/ollama-monitor`.

  The refresh kind asks for CPU, memory, `exe` and `cmd` (both `OnlyIfNotSet`) and is `without_tasks()`. The last part matters on Linux, where sysinfo otherwise lists every thread as a process of its own; each Ollama thread is named `ollama`, so the count, RSS and CPU would all be inflated. Also on Linux, `/proc/<pid>/exe` is unreadable for another user's processes, so `exe()` is empty for the systemd `ollama` service; the name (`/proc/<pid>/stat`) and `argv[0]` (`/proc/<pid>/cmdline`) are world-readable and do the matching.

- **GPU telemetry** from a child process whose stdout the shared reader task in `hardware/mod.rs` parses line by line. Both backends compile on every platform and expose the same four items (`PROGRAM`, `prime`, `command`, `State`); a `cfg(target_os)` alias picks the active one, and the inactive one is `allow(dead_code)` so both parsers' tests run everywhere.

  - **macOS** (`powermetrics.rs`): `sudo -n powermetrics --samplers cpu_power,gpu_power,ane_power -i 2000`, text-parsed for the `GPU HW active residency:` and `ANE Power:` lines. `hardware::prime()` runs `sudo -v` before the TUI enters raw mode, so the password prompt gets a normal terminal; `-n` then keeps the real launch non-interactive. The `cpu_power` sampler is included alongside `ane_power` because on M1/M4 Macs the unified power summary that holds the `ANE Power:` line only appears when `cpu_power` is requested.

    The sudo child gets a null stdin and its own process group (`process_group(0)`). That's hardening: since sudo 1.9.14 `use_pty` is on by default, and a sudo in the terminal's foreground process group may read terminal input to relay to its command, competing with crossterm for keystrokes. Outside the foreground group and without stdin, it never reads from or reconfigures the TUI's terminal. This is safe only with `-n` (a background sudo that prompted would stop on SIGTTIN), and it must not be `setsid`, because sudo's cached credentials are tied to the terminal session.

  - **Linux** (`nvidia_smi.rs`): `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 2000`, one `index, percent, MiB` line per NVIDIA GPU per interval, no privilege, so `prime()` does nothing. The state keeps the latest reading per GPU index; the row shows the busiest GPU's utilization and the summed memory, in the panel's decimal units (16 376 MiB reads as 17.2 GB). nvidia-smi never touches the terminal, so it stays in the TUI's process group and a closing terminal's SIGHUP reaches it too. Intel and AMD GPUs have no unprivileged source here and show `n/a`. Ollama's `/api/ps` reports `size_vram` per loaded model, which would be a vendor-neutral figure for the models' VRAM; not used yet.

  If the child can't be spawned or exits later, the log says `spawn … failed` or `powermetrics exited` / `nvidia-smi exited`, the GPU figures fall back to `n/a`, and nothing restarts it; everything else keeps working. On shutdown the sampler sends SIGTERM to the child via `libc::kill`; on macOS sudo relays it to powermetrics, so nothing is left running as root.

`sysinfo` is pinned to the same 0.38 line as LMS-Monitor so both TUIs report identical memory figures. On macOS "free" is XNU's available-non-compressed memory (active + inactive + free pages); 0.32 subtracted compressor pages instead and floored at 0 under heavy compression. On Linux it is the kernel's `MemAvailable`.

## TUI

`ratatui` 0.30 + `crossterm` 0.29. A port of [LMS-Monitor](https://github.com/Ward-Software-Defined-Systems/LMS-Monitor)'s TUI: same layout, columns, colours and keys. The fixed panels take 29 rows and the feed needs at least 7, so 120×36 is the smallest terminal that shows everything. Top to bottom:

| panel | rows | content |
|---|---|---|
| header | 3 | `ollama-monitor · server: ● reachable/unreachable/unknown (url) · err · [PAUSED] · lifetime: reqs / sessions / prompt tok / gen tok · local clock`; Ollama-only `ollama vX · proxy ADDR` right-aligned in the top border |
| loaded models | 6 (3 model rows) | every `/api/tags` model merged with `/api/ps`, sorted loaded → cloud → not-loaded: id · type · compat · quant · ctx · state, `▸` on the most recent inference target. Rows past the third are cut off |
| hardware | 3 | one line: system CPU / MEM (free) │ Ollama CPU / RSS / process count │ GPU / ANE (macOS) or GPU / VRAM (Linux) |
| live feed | `Min(7)` (4 entries at 36 rows) | up to 30 entries, newest first: local completion time · model · prompt · gen · TTFT ms · tok/s · stop (25 columns, LMS-Monitor's width); `~` marks approximate (`openai-sse-approx`) rows. A failed request is a whole red row: time, model or `?`, four `-`, then `failed (HTTP 429)` or `proxy (HTTP 502)` |
| rolling metrics | 9 | `1m / 5m / 15m / session` columns; rows: requests, prompt tok, gen tok, mean tok/s, p95 tok/s, mean TTFT |
| hypothetical cost | 7 | frontier models as columns; input / output / total USD rows for the session |
| footer | 1 | `q quit · r reset session · p pause` |

Feed time is completion time (LMS-Monitor shows start time) because Ollama-Monitor's rolling windows, DB rows and feed order are all keyed on `completed_at`. A failed row's time is when its status was known.

The `▸` marker needs name normalisation: OpenAI-compatible responses report `deepseek-v4-pro` for the `deepseek-v4-pro:cloud` tag, and `qwen3` for `qwen3:latest`. `api::canonical_model_name` lowercases and strips `:latest`, `:cloud` and a `-cloud` tag suffix; an exact id match always wins over a canonical one. The marker is resolved on every frame, so a record that lands before the first `/api/tags` poll still gets it once the list arrives. While Ollama is unreachable, the last model list and version stay on screen.

Keys: `q` or `Ctrl-C` quits (in raw mode Ctrl-C arrives as a key press, not SIGINT), `r` resets the session panels and `p` pauses (see [Record lifecycle](#record-lifecycle)). Only key presses act; the releases that terminals with enhanced keyboard reporting send are ignored.

The screen redraws after every event (records, polls, keys, resize) plus a 250 ms tick for the clock. Channel reads go through a biased `tokio::select!` (shutdown, keys and records first). Input comes from a dedicated thread (`crossterm::event::poll` 250 ms + `read` → mpsc).

Terminal restore is bracketed by:

1. a panic hook, installed first, that restores the terminal and then chains the original hook;
2. `tui::init_terminal`, which enters raw mode and the alternate screen the way `ratatui::try_init` does, but without installing ratatui's own panic hook;
3. a restore on every normal exit path.

Restoring always goes through `ratatui::try_restore()`, with errors only logged at debug. `ratatui::restore()` and the hook `ratatui::init()` installs report failure with `eprintln!`, and once a SIGHUP has taken the terminal away that write fails and panics, which inside a panic hook aborts the process. For the same reason a failed draw after shutdown has been signalled counts as a clean exit, and headless mode writes its summary lines with `writeln!` and ignores errors.

`q`, `Ctrl-C`, SIGINT, SIGTERM, SIGHUP and panics all restore the terminal (or try to, when it's already gone).

## Logging

`tracing` always writes to the log file in the app data directory (README › Files) (`tracing_appender::rolling::never`: appended, never rotated). In headless mode it also writes to stderr. In TUI mode nothing goes to stdout or stderr, which would corrupt the screen.

The filter comes from `OLLAMA_MONITOR_LOG` (default `info`). `tracing-subscriber`'s default `tracing-log` feature bridges the `log` crate (reqwest uses it), and hyper-util and h2 emit tracing events of their own, so a bare `debug` or `trace` includes library noise; scope it with `ollama_monitor=debug`. Lines don't carry their target, so library lines aren't labelled as such.

At the default level the log holds startup and shutdown lines, warnings, a line for each tracked request that yields no stats, and one for each failed request (`inference request failed`). Request and response bodies never reach it: the proxy logs metadata (method, path, status, model, token counts), never content.

## Startup and shutdown

Startup, in order:

1. Parse the CLI; resolve paths (creating the data directory: `~/Library/Application Support/ollama-monitor/` on macOS, `~/.local/share/ollama-monitor/` on Linux).
2. Start logging, then load the config and pricing, so their warnings are logged.
3. `proxy::bind`: a taken port or a bad address exits here, before any prompt.
4. In TUI mode, `hardware::prime` (the sudo password prompt on macOS; nothing on Linux).
5. Build the runtime; open the database and insert the session row.
6. Spawn the proxy and the signal task, plus, in TUI mode, the poller and the hardware sampler (which starts the telemetry child).
7. Run the TUI or the headless loop.

Shutdown starts with `q`, `Ctrl-C` (a key press in the TUI), SIGINT, SIGTERM or SIGHUP (the terminal window closing), all of which set the `watch` flag:

1. The TUI loop returns and restores the terminal (or the headless loop returns). If the UI exits with an error instead, it's logged and the remaining steps still run.
2. Workers get 3 s to finish. axum's graceful shutdown stops accepting and waits for open connections, so a generation still streaming holds it up; after 3 s it's abandoned.
3. `end_session` writes `ended_at`.
4. `runtime.shutdown_background()` drops whatever is left, and the log is flushed.

A record that completes after the UI loop exits is dropped (nothing reads `records_rx` any more), and streams still open after the 3 s grace are cut. On macOS the sudo child sits in its own process group, so a closing terminal doesn't signal it directly; the sampler's SIGTERM in step 2 stops it. On Linux nvidia-smi shares the terminal's process group and gets the SIGHUP as well.

## Testing and CI

All tests run without Ollama, sudo, a GPU or a terminal:

- **Parser**: fixture-driven (`fixtures/`, captured from real Ollama responses), with bodies fed in 64-byte chunks to exercise frame reassembly.
- **API**: `/api/tags` and `/api/ps` parsing, `merge_models` ordering and states, name canonicalisation.
- **Proxy**: the request rewrite and its 64 MiB cap; the driver task's behaviour (trailer drain after `finish_reason`, no drain mid-generation, load calls not recorded) on tokio's paused clock; `bind` errors; and end-to-end tests over real sockets (client → `serve` → mock upstream).
  - One checks a response arrives unchanged with exactly one record emitted.
  - One streams an upload just over the buffering cap through intact.
  - One checks that 4xx / 5xx answers pass through unchanged and that each one on a tracked path is reported once, with the model for the OpenAI path and none for the native one. An untracked path isn't reported.
  - One checks that an unreachable upstream is reported as `proxy (HTTP 502)`.
- **TUI**: renders into ratatui's `TestBackend`. `hardware_row_survives_at_minimum_height` guards the 120×36 minimum, and `screen_snapshot` prints 120×36 and 160×44 screens for eyeballing. The feed tests read cell colours from the buffer: a failed row is red end to end and a completed row's `length` stop isn't. Pause, the 30-entry cap and newest-first order apply to failed rows too.
- **DB, pricing, aggregator, config directories, hardware formatting and both telemetry parsers**: unit tests. The two telemetry backends compile on every platform, so the powermetrics and nvidia-smi parsers are both tested in Linux CI and on a Mac. The DB helper names temp files with a counter, not the clock, so parallel tests can't collide. The DB tests replay a v1 binary's open on the same file around the v2 upgrade.

Two tests are `#[ignore]`d because they touch the real machine: `live_ollama_snapshot` (read-only GETs to `127.0.0.1:11434`) and `live_sysinfo_snapshot`.

GitLab CI (`.gitlab-ci.yml`, shared with LMS-Monitor) runs `cargo fmt --all --check`, `cargo clippy --all-targets --locked -- -D warnings` and `cargo test --locked` on a `rust:1.97` Linux image. Linux is a runtime target as well as the CI platform. The only code CI can't exercise is the macOS side of `config::default_dirs` and of the `telemetry` alias, so a change there wants a build on a Mac.

## Notable invariants

- Each captured request and each failed one is persisted exactly once, by the loop that owns `records_rx` and `failures_rx`, and the DB writer is the only writer.
- Reporting a failed request never waits: the handler calls `try_send` before it returns the response. Only the status code is kept, never the error body.
- Responses are forwarded byte for byte. The deliberate exceptions are the OpenAI `include_usage` request rewrite (with its 64 MiB cap on those bodies) and the hop-by-hop / `Host` / `Content-Length` header handling. Parser failures never change what the client sees.
- The proxy is the only capture path. Direct hits on Ollama's port are invisible by design.
- Nothing writes to stdout or stderr while the TUI is up.
- A client abort mid-generation must drop the upstream stream (that's how cancellation reaches Ollama); only a finished stream waiting on its usage frame gets drained.
- TUI parity with LMS-Monitor: `tui/layout.rs` is byte-identical, and `tui/widgets.rs` differs only in data-model mapping plus the Ollama extras (version / proxy border title, `cloud` state, `~` markers) and the hardware row's Linux `vram` slot, a `cfg!` branch that leaves the macOS rendering unchanged. The red feed rows are the same idea for different events.
  - Here they're failed requests (`FeedEntry::Failed`, `failed_row`); in LMS-Monitor they're context-overflow refusals (`FeedEntry::Rejected`, `rejected_row`).
  - LMS's `stop_cell` isn't ported, because no Ollama stop reason means the context ran out (`length` also covers a max-tokens cap).

  The `pricing.toml` values match too. Change both apps together.
- Pricing keys are TOML-friendly (hyphenated, no dots): `gemini-3-1-pro`, not `gemini-3.1-pro`.
- The telemetry child is the only subprocess: on macOS `sudo -n powermetrics` (plus the startup `sudo -v`), with no stdin and its own process group; on Linux `nvidia-smi`, unprivileged, with no stdin.

## Reference files

- [`README.md`](./README.md): usage, install, troubleshooting
- [`pricing.toml`](./pricing.toml): baked-in frontier prices
- [`fixtures/`](./fixtures): captured Ollama and OpenAI-compatible responses plus `/api/tags` and `/api/ps` samples, used by the parser, API and TUI tests
