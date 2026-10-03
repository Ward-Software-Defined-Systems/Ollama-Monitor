# ollama-monitor

A standalone Rust TUI that observes [Ollama](https://ollama.com) inference activity on `localhost` via a transparent reverse proxy and surfaces:

- **Live request feed** — last 30 completed inferences (timestamp, model, prompt/gen tokens, TTFT, tok/s, stop reason)
- **Rolling throughput** — 1m / 5m / 15m / session-lifetime windows
- **Hypothetical frontier cost** — Claude Fable 5, Opus 4.8, Gemini 3.1 Pro priced against local token counts
- **Hardware** — system + Ollama process tree CPU%, memory, GPU active residency, ANE power
- **Cross-session SQLite persistence** — lifetime totals available across restarts

No real frontier API calls — costs come from a baked-in pricing table. No daemon. Single binary.

## How it observes inference

Ollama returns per-request stats (`eval_count`, `eval_duration`, `prompt_eval_count`, `prompt_eval_duration`, `total_duration`, `done_reason`) only in the response body to the calling client — nothing is broadcast to a third-party observer like LM Studio's `lms log stream` does. To capture those stats, this app runs an HTTP **reverse proxy** in front of Ollama:

```
client → :11435 (ollama-monitor)  →  :11434 (ollama)
                ↓ tees response stream
        parses final stats chunk → SQLite + TUI
```

Point your Ollama clients at the monitor port (default `11435`), and the proxy transparently forwards everything to Ollama on `11434`. Token counts and timings come straight from the upstream response — the proxy does not synthesize them.

Four response envelopes are recognised automatically:

| envelope | path | trigger |
|---|---|---|
| Ollama-native streaming | `/api/chat`, `/api/generate` | `stream: true` (default) |
| Ollama-native non-streaming | `/api/chat`, `/api/generate` | `stream: false` |
| OpenAI-compatible SSE | `/v1/chat/completions` | `stream: true` — proxy auto-injects `stream_options.include_usage` so token counts always come back |
| OpenAI-compatible non-streaming | `/v1/chat/completions` | `stream: false` |

Requests that bypass the proxy (direct hits on `:11434`) are not captured — by design.

## Status

v0.1.0. Built on top of an existing LM Studio monitor architecture; replaces the `lms log stream` event source with the reverse proxy.

## Requirements

- macOS (Apple Silicon recommended for GPU/ANE telemetry)
- Ollama installed and running (`ollama serve`, default port 11434)
- Rust 1.94+ (2024 edition)
- `sudo` access for the GPU/ANE row (`powermetrics` requires elevation; you'll be prompted once per launch)

## Build

```sh
cargo build --release
# Produces target/release/ollama-monitor (~7.9 MB)
```

## Run

```sh
target/release/ollama-monitor
```

You'll be prompted for your sudo password — used solely to spawn `powermetrics --samplers cpu_power,gpu_power,ane_power -i 2000`. The TUI then opens.

If you'd rather skip GPU/ANE and avoid the prompt, use headless mode:

```sh
target/release/ollama-monitor --no-tui
```

Headless prints one summary line per completed inference to stderr and persists records to SQLite.

### Pointing clients at the proxy

By default the proxy listens on `127.0.0.1:11435` and forwards to `http://127.0.0.1:11434`. Configure clients (Open WebUI, custom code, etc.) to use the proxy port:

```sh
# direct (untracked)
curl http://localhost:11434/api/chat -d '{"model":"qwen3:14b", ...}'

# through the monitor (tracked)
curl http://localhost:11435/api/chat -d '{"model":"qwen3:14b", ...}'
```

For network-accessible monitoring (other machines hitting this Mac Mini), bind the proxy on `0.0.0.0`:

```sh
ollama-monitor --proxy-listen 0.0.0.0:11435
```

If you want the monitor to take over `:11434` entirely so existing clients work unchanged, restart Ollama on a different port (`OLLAMA_HOST=127.0.0.1:11400 ollama serve`) and run the monitor with `--proxy-listen 0.0.0.0:11434 --ollama-url http://127.0.0.1:11400`.

### Flags

| flag | default | purpose |
|---|---|---|
| `--proxy-listen <ADDR>` | `127.0.0.1:11435` | Address the reverse proxy listens on |
| `--ollama-url <URL>` | `http://127.0.0.1:11434` (env `OLLAMA_URL`) | Upstream Ollama base URL |
| `--config <PATH>` | macOS app-support dir | Optional user config TOML |
| `--db <PATH>` | macOS app-support dir | SQLite usage DB |
| `--no-tui` | off | Headless: print one summary line per inference; no UI |

### Keys (TUI)

| key | action |
|---|---|
| `q` / `Ctrl-C` | quit (terminal restored, session closed in DB) |
| `r` | reset session counters (records remain in DB) |
| `p` | pause UI updates (records still persist) |

## File locations (macOS)

- DB: `~/Library/Application Support/ollama-monitor/usage.db`
- App log: `~/Library/Application Support/ollama-monitor/ollama-monitor.log`
- User config (optional): `~/Library/Application Support/ollama-monitor/config.toml`

Set `OLLAMA_MONITOR_LOG=debug` (or `trace`) for verbose tracing in the log file. `trace` includes raw `powermetrics` lines, useful for diagnosing GPU/ANE parsing.

## Override pricing

Defaults are baked in from [`pricing.toml`](./pricing.toml). To override, drop a TOML file at `~/Library/Application Support/ollama-monitor/config.toml`:

```toml
[pricing.providers.anthropic.models.claude-opus-4-8]
input_per_mtok_usd  = 4.00
output_per_mtok_usd = 20.00
```

## Caveats per envelope

- **Ollama-native** (both stream and non-stream) is the highest-fidelity source: token counts and durations come from llama.cpp directly. `tok/s = eval_count / eval_duration`.
- **OpenAI-compatible streaming**: the proxy rewrites the request body on the way through, ensuring `stream_options.include_usage: true` is set so the upstream emits a final usage chunk with token counts. The injection is invisible to clients (the extra chunk has `choices: []` and most renderers ignore it). Without this, almost no OpenAI clients (Open WebUI, Continue, raw OpenAI SDK, Cursor, …) would ever produce captureable stats.
- **OpenAI-compatible non-streaming** has all bytes arrive at once, so the proxy can't separate prompt-eval time from generation time. We report `tok/s` as `completion_tokens / wall_total` — an end-to-end throughput, lower bound on pure decode rate.
- **`:cloud` models (approximate envelope)**: requests to cloud-hosted models (e.g. `deepseek-v4-pro:cloud`) are proxied by the local Ollama daemon to ollama.com. The upstream cloud-proxy intermittently drops the terminal `usage` SSE frame (tracked in [ollama/ollama#15169](https://github.com/ollama/ollama/issues/15169)) and can also reset the connection mid-stream ([ollama/ollama#15910](https://github.com/ollama/ollama/issues/15910)). When that happens, the monitor still records the request but tags it with envelope `openai-sse-approx`: `prompt_tokens` is `0` (we never see it), `gen_tokens` is the count of streamed SSE chunks (a coarse proxy for completion tokens), and the TUI live-feed prefixes the affected columns with `~` to flag the estimate. Full captures (`openai-sse`) are still preferred whenever the usage frame does arrive.

If a request still goes unrecorded, the monitor logs `no parsable stats in response (request not recorded)` at info level — check the log to see which path was unparseable.

## Troubleshooting

| symptom | check |
|---|---|
| header shows "● unreachable" | `curl http://localhost:11434/api/version` — is `ollama serve` actually running? |
| GPU or ANE shows `n/a` | `OLLAMA_MONITOR_LOG=trace` then grep `powermetrics` in the log — the parser tolerates label variants but isn't psychic |
| no records appear despite traffic | clients still pointed at `:11434` instead of `:11435` — verify with `curl http://localhost:11435/api/version` (should return Ollama's version) |
| sudo prompt fails / app exits | `sudo -v` once before launch, or use `--no-tui` |

## Development

```sh
cargo test                # unit + fixture tests for parser, aggregator, pricing, db, hardware
cargo run -- --help
```

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for module layout, data flow, and design decisions.
