# Architecture

## Goal

Single Rust binary that observes a local Ollama instance via a transparent HTTP reverse proxy and surfaces inference metrics + hypothetical frontier-API costs in a terminal UI. No daemon, no remote, no real frontier API calls outside `localhost`.

## Process model

One OS process, one `tokio` multi-thread runtime. Tasks communicate through bounded `tokio::sync::mpsc` channels; shutdown propagates through one `tokio::sync::watch::channel<bool>`.

```
                     ┌──────────────────────────────────┐
                     │            main()                │
                     │  parse CLI · open DB · spawn     │
                     │  workers · run TUI or headless · │
                     │  SIGINT / SIGTERM cleanup        │
                     └────────────────┬─────────────────┘
                                      │
   ┌──────────┬──────────────┬────────┼──────────────┬─────────────┬────────────┐
   ▼          ▼              ▼        ▼              ▼             ▼            ▼
┌──────┐  ┌──────────┐  ┌─────────┐ ┌──────────────┐ ┌─────────┐ ┌────────────┐
│api   │  │PROXY     │  │parser   │ │hardware      │ │db_writer│ │ TUI render │
│poll  │  │HTTP svr  │  │acc per  │ │sampler       │ │task     │ │ task       │
│2 s   │  │+forwarder│  │request  │ │ + powermetr. │ │         │ │            │
│/api/ps│ │          │  │         │ │              │ │         │ │            │
└──┬───┘  └────┬─────┘  └────┬────┘ └──────┬───────┘ └────┬────┘ └─────┬──────┘
   │           │             │             │              │            │
   │       chunks            │records      │ hw snaps     │            │
   │           ├──tee inside─┤             │              │            │
   │           │             ▼             │              │            │
   │ models    │       records_tx          │              │            │
   │ snaps     │                           │              │            │
   ├───────────┴─────────────┴─────────────┴──────────────┴────────────►│
   │                                                                   │   render
   │  (lifetime poller: read-only DB conn every 2s,                    │   loop
   │   pushes totals to TUI header)                                    │   250 ms
   └──────────────────────────────────────────────────────────────────┘
```

## Modules

| file | role |
|---|---|
| `src/main.rs` | CLI parse, runtime bootstrap, channel + task wiring, signal handling, TUI vs headless dispatch |
| `src/api.rs` | `poll_models` task: `GET /api/version`, then `/api/tags` + `/api/ps` concurrently; `merge_models` turns them into LMS-shaped `ModelInfo` rows (state loaded / cloud / not-loaded); `resolve_model_id` maps a record's model name to its row for the `▸` marker; `ModelsSnapshot::{Loaded, Unreachable}` |
| `src/proxy.rs` | axum reverse proxy. Transparent forwarder for all paths. For inference paths it tees the response stream into a per-request `parser::Accumulator`. |
| `src/parser.rs` | Per-request stats accumulator. Detects envelope from path + content-type, drains NDJSON / SSE frames as they arrive, emits `ParsedStats` on `finalize()`. |
| `src/aggregate.rs` | Rolling 1m / 5m / 15m / session-lifetime windows; per-model breakdown; mean / p50 / p95 |
| `src/pricing.rs` | TOML-loaded pricing table; baked-in defaults from `pricing.toml` |
| `src/db.rs` | SQLite schema + sessions + dedicated tokio writer task; `lifetime_totals` query |
| `src/config.rs` | Optional user TOML loader; macOS app-support path resolution for db / config / log |
| `src/hardware.rs` | `sysinfo` CPU/MEM sampler + `sudo powermetrics` GPU/ANE parser; Ollama-tree process detection |
| `src/tui/mod.rs` | Event loop; `AppState`; `render()`; panic-safe terminal restore |
| `src/tui/layout.rs` | Top-to-bottom panel layout — byte-identical to LMS-Monitor's |
| `src/tui/widgets.rs` | Per-panel render fns (header, models, hardware, feed, rolling, costs, footer), ported from LMS-Monitor |

## Event source — the design pivot

The LM Studio version of this app observes inference passively via `lms log stream -s model --stats --json`, a JSON-Lines event stream LM Studio publishes for any third party to consume.

**Ollama has no equivalent.** Per-inference stats (`eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`, `total_duration`, `done_reason`) ride back to the calling client in the HTTP response body — nothing is broadcast to a third-party observer. The server log only emits GIN HTTP access lines (method, path, status, total duration).

So instead of subscribing to a log stream, this app **becomes** the path the response travels through:

- An axum reverse proxy listens on `--proxy-listen` (default `127.0.0.1:11435`) and forwards every request to `--ollama-url` (default `http://127.0.0.1:11434`).
- For inference paths (`/api/chat`, `/api/generate`, `/v1/chat/completions`), each upstream chunk is tee'd: byte-for-byte forwarded to the client AND cloned into a per-request `parser::Accumulator`.
- When the response finishes, the accumulator is finalised; if it found a stats record, it emits an `InferenceRecord` to `records_tx`.

Clients must point at the proxy port (or the user reconfigures Ollama to listen elsewhere and the proxy takes `:11434`). This is the cost of capturing real per-inference stats. Direct hits on `:11434` are not captured — by design.

## Request body rewrite (OpenAI-compat only)

The proxy is byte-transparent for *responses* but does one targeted *request* rewrite. For paths matching `/v1/chat/completions` or `/v1/completions` with `stream: true`, the request body is re-serialised with `stream_options.include_usage = true` injected (creating the `stream_options` object if absent). Without this, Ollama's OpenAI-compat SSE response omits the `usage` block entirely — and almost no client sets the option by default (Open WebUI, Continue, the raw OpenAI SDK, Cursor, etc. all skip it). That left ~0% of real-world OpenAI traffic captureable.

The injection is silent to clients: the extra final SSE chunk has `choices: []`, which standard OpenAI renderers iterate-and-ignore. Existing values inside `stream_options` are preserved (only `include_usage` is forced); a non-object `stream_options` is overwritten. Non-streaming requests, non-OpenAI paths, and non-JSON bodies pass through untouched. See `proxy::ensure_openai_include_usage` and its tests in `src/proxy.rs`.

This is the one place the proxy is *not* transparent. The trade-off — a one-key addition versus losing nearly every OpenAI-client request — is worth it; it's documented here and in the README so it isn't a surprise.

## Streaming response tee

The non-trivial bit. Each upstream response is a `futures_util::Stream<Item = Result<Bytes, _>>`. We:

1. Spawn a per-request task that owns a `parser::Accumulator` and an `mpsc::UnboundedReceiver<Bytes>`.
2. Map the upstream stream so each successful chunk is `clone()`'d to the accumulator's mpsc sender and forwarded to the client unchanged.
3. When the upstream stream ends (or the client disconnects and the stream is dropped), the mpsc sender drops, the accumulator task drains remaining chunks, calls `finalize()`, and on success sends an `InferenceRecord` to the records channel.

Failure isolation: parser errors never affect proxy correctness. Truncated upstream → finalize returns `None` → the request goes unrecorded. The client always sees the bytes Ollama produced.

## Envelope variants

`parser::classify(path, content_type)` selects one of four:

| envelope | path match | content-type | terminal record |
|---|---|---|---|
| `OllamaStream` | `/api/chat`, `/api/generate` | `application/x-ndjson` (or none) | NDJSON line with `done: true` |
| `OllamaSingle` | `/api/chat`, `/api/generate` | `application/json` | single JSON object |
| `OpenAiSse` | `/v1/chat/completions` | `text/event-stream` | usage in final `data:` frame; the proxy auto-injects `stream_options.include_usage: true` into the request body so the upstream always returns it (see "Request body rewrite" below) |
| `OpenAiSingle` | `/v1/chat/completions` | `application/json` | single JSON object with `usage` |
| `OpenAiSseNoUsage` *(emitted by `finalize` only, never by `classify`)* | — | — | fallback when a `/v1/chat/completions` SSE stream ended without a `usage` frame; serialised as `"openai-sse-approx"` in the DB. `gen_tokens` is the count of SSE chunks with non-empty `choices` — a coarse proxy used when upstream (typically Ollama's `:cloud` proxy, ollama/ollama#15169) drops the terminal usage frame. `prompt_tokens` is `0` because the prompt size never travels in the stream. |

Mapping → `InferenceRecord`:

| field | Ollama-native | OpenAI |
|---|---|---|
| `model_id` | `model` | `model` |
| `prompt_tokens` | `prompt_eval_count` | `usage.prompt_tokens` |
| `gen_tokens` | `eval_count` | `usage.completion_tokens` |
| `tokens_per_sec` | `eval_count / (eval_duration / 1e9)` | `completion_tokens / wall_window` (see below) |
| `ttft_sec` | `prompt_eval_duration / 1e9 + load_duration / 1e9` | proxy wall-clock for SSE; 0 for non-stream |
| `total_time_sec` | `total_duration / 1e9` | proxy wall-clock from `request_start` |
| `stop_reason` | `done_reason` | `choices[0].finish_reason` |
| `envelope` | `"ollama-stream"` / `"ollama-single"` | `"openai-sse"` / `"openai-single"` |

For OpenAI SSE `wall_window = wall_total - wall_ttft`. For OpenAI non-streaming, the proxy can't distinguish prompt from gen time (all bytes arrive together), so it reports `tok/s = completion_tokens / wall_total` — a lower bound on pure decode rate. Honest about what we can observe.

## Aggregator

`Vec<InferenceRecord>` in memory for the current session. Snapshots compute per-window metrics on-demand (1m / 5m / 15m filters by `completed_at`, plus session-lifetime). Percentiles via sort + index. Per-model breakdown groups by `model_id`.

Cross-session totals come from a separate `db::lifetime_totals` query. The TUI's lifetime-totals poller opens its own read-only SQLite connection and queries every 2 s — keeps the aggregator focused on the live session and eliminates a startup bootstrap from disk.

## Persistence

SQLite via bundled `rusqlite`. Tables: `sessions`, `inference_records`, `schema_version`. Schema is applied idempotently on every open.

All writes go through one dedicated tokio task (`db::open_and_spawn_writer`); `DbHandle` is a clone-able `mpsc::Sender` wrapping its command channel. The lifetime poller opens its own read-only connection — SQLite handles concurrent readers natively.

Sessions are stamped on app start (`started_at`); `ended_at` is filled on graceful shutdown — `q`, `Ctrl-C` (SIGINT), or SIGTERM all route through the same shutdown broadcast.

## Pricing

[`pricing.toml`](./pricing.toml) is `include_str!`-baked at compile time. Three frontier model keys:

```rust
pub const FRONTIER_MODELS: &[&str] = &[
    "claude-fable-5",
    "claude-opus-4-8",
    "gemini-3-1-pro",
];
```

Pricing is flat-rate; long-context tier (Gemini > 200K) is deferred. User config at `~/Library/Application Support/ollama-monitor/config.toml` overrides per-model.

## Hardware sampling

Two paths:

- **CPU + memory + Ollama process tree** via `sysinfo` — no privilege required. Detection is **path-based**, matching substrings:
  - `/Applications/Ollama.app/` (desktop install)
  - `/opt/homebrew/opt/ollama/`, `/opt/homebrew/Cellar/ollama/` (Apple Silicon brew)
  - `/usr/local/opt/ollama/`, `/usr/local/Cellar/ollama/` (Intel-Mac brew)
  - `/.ollama/` (model store / runner workdir)
  - Plus name-based fallback for `ollama` and `ollama runner` binaries.

  This catches the `ollama serve` parent and the `ollama runner --ollama-engine ...` subprocess (where the loaded model RSS lives — typically several GB).

- **GPU active residency + ANE power** via `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 2000`. sudo is invoked **before** TUI raw-mode entry (`hardware::prime_sudo()`) so the password prompt works against a normal cooked terminal. The output is text-parsed for `GPU HW active residency:` and `ANE Power:` lines.

If `sudo` or `powermetrics` fails, the failure logs a warning and GPU/ANE fall back to `n/a` in the panel. The TUI still launches.

On shutdown, the powermetrics child receives `SIGTERM` via `libc::kill` so it exits cleanly rather than orphaning as root.

The `cpu_power` sampler is included alongside `ane_power` because on M1/M4 Macs the unified power summary that includes the `ANE Power:` line only emits when `cpu_power` is requested.

`sysinfo` is pinned to the same 0.38 line as LMS-Monitor so both TUIs report identical memory figures. "free" is XNU's available-non-compressed memory (active + inactive + free); 0.32 subtracted compressor pages instead and floored at 0 under heavy compression.

## TUI

`ratatui` 0.30 + `crossterm` 0.29. A port of LMS-Monitor's TUI: same layout (heights fit 120×36 minimum), columns, colours and keys. Top-to-bottom:

| panel | rows | widget |
|---|---|---|
| header | 3 | `ollama-monitor · server: ● reachable/unreachable/unknown (url) · err · [PAUSED] · lifetime: reqs / sessions / prompt tok / gen tok · local clock`; Ollama-only `ollama vX · proxy ADDR` right-aligned in the top border |
| loaded models | 6 | every `/api/tags` model merged with `/api/ps`: id · type · compat · quant · ctx · state (`loaded` / `cloud` / `not-loaded`), `▸` on the most recent inference target |
| hardware | 3 | one line: system CPU/MEM (free) │ Ollama CPU/RSS/proc count │ GPU/ANE |
| live feed | `Min(7)` | last 30 records, newest at top: local completion time · model · prompt · gen · TTFT ms · tok/s · stop; `~` marks approximate (`openai-sse-approx`) rows |
| rolling metrics | 9 | `1m / 5m / 15m / session` columns; rows = requests, prompt tok, gen tok, mean tok/s, p95 tok/s, mean TTFT |
| hypothetical cost | 7 | frontier models as columns; input / output / total USD rows for the session |
| footer | 1 | `q quit · r reset session · p pause` |

Feed time is completion time (LMS-Monitor shows start time) because Ollama-Monitor's rolling windows, DB rows and feed order are all keyed on `completed_at`.

The `▸` marker needs name normalisation: OpenAI-compatible responses report `deepseek-v4-pro` for the `deepseek-v4-pro:cloud` tag, and `qwen3` for `qwen3:latest`. `api::canonical_model_name` strips `:latest`, `:cloud` and `-cloud`; an exact id match always wins over a canonical one.

Redraw happens after every event (records, polls, keys, resize) plus a 250 ms tick for the clock. Channel reads via `tokio::select!` (biased: shutdown, keys, records first). Input arrives from a dedicated blocking thread (`crossterm::event::poll` 250 ms + `read` → mpsc); only key presses act.

Terminal restore is bracketed by:

1. `ratatui::init()` enters raw mode + alternate screen
2. **Panic hook installed before** `ratatui::init()` calls `ratatui::restore()` first, then chains the original hook
3. `ratatui::restore()` exits raw mode + alternate screen on any clean path

`q`, `Ctrl-C`, SIGINT, SIGTERM, and panics all route to a clean restore.

## Notable invariants

- Every record in `inference_records` ⇔ one TUI feed entry ⇔ one aggregator ingest. The TUI is the sole tee point — no double-counting.
- TUI parity with LMS-Monitor: `tui/layout.rs` is byte-identical; `tui/widgets.rs` differs only in data-model mapping plus the Ollama extras (version/proxy border title, `cloud` state, `~` markers). Change both apps together.
- The proxy is the only path through which records are captured. Direct hits to upstream `:11434` are intentionally invisible.
- Proxy parser failures never affect proxy correctness — clients always see exactly what Ollama returned. (Exception: the OpenAI-compat request-body rewrite described above; clients never see *less* than they would have, and the upstream response is forwarded byte-for-byte.)
- Pricing keys are TOML-friendly (hyphenated, no dots): `gemini-3-1-pro`, not `gemini-3.1-pro`. Mismatch = silent miss.
- `tracing` writes to a file *only* when in TUI mode — never stdout/stderr. Headless mode (`--no-tui`) also writes to stderr.
- Subprocess: `powermetrics` is the only one (no log-stream subprocess like the LM Studio version). It uses sudo with `-n` (non-interactive); priming via `prime_sudo()` makes that succeed.
- DB writer is the only writer; the lifetime poller has its own read-only connection.

## Reference files

- [`README.md`](./README.md) — usage, install, troubleshooting
- [`pricing.toml`](./pricing.toml) — baked-in frontier defaults
- [`fixtures/`](./fixtures) — captured Ollama + OpenAI-compat response samples used by parser tests
