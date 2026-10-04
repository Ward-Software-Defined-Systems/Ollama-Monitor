# Ollama-Monitor

A terminal dashboard for [Ollama](https://ollama.com) on macOS. It sits in front of your local Ollama as a transparent proxy and shows every request's tokens, time to first token and throughput, what the same tokens would have cost on frontier APIs, and live Apple Silicon hardware telemetry.

![Ollama-Monitor running in a terminal](assets/Ollama-Monitor.png)

## Features

- **Live request feed**: the last 30 completed requests with time, model, prompt and generated tokens, time to first token, tokens per second and stop reason. A `~` marks counts that had to be estimated (see [Capture accuracy](#capture-accuracy)).
- **Models**: every installed model, local and `:cloud`, with its type, format, quantization, context length and state (loaded, cloud or not-loaded); `▸` marks the model the last request went to.
- **Rolling metrics**: requests, tokens, mean and p95 tokens per second, and mean time to first token over the last 1, 5 and 15 minutes and the whole session.
- **Hypothetical cost**: your session's token counts priced at list rates for Claude Fable 5.1, Fable 5, Opus 5.5, Opus 5, Opus 4.8 and Gemini 3.1 Pro.
- **Hardware**: system CPU and memory, CPU and memory of the Ollama process tree, GPU active residency and Neural Engine power.
- **History**: lifetime request and token totals across sessions, kept in a local SQLite database.

## How it works

Ollama returns per-request stats (token counts, durations, stop reason) only in the response to the client that made the request; there's nothing a separate program can subscribe to. So Ollama-Monitor runs a reverse proxy in front of Ollama:

```
client ──▶ :11435 ollama-monitor ──▶ :11434 ollama
                  │ copies each response as it streams past
                  ▼
            parses the final stats ──▶ SQLite + dashboard
```

Point your clients at the proxy's port instead of Ollama's. Requests and responses pass through unchanged, with one exception: for streaming OpenAI-compatible requests the proxy adds `stream_options.include_usage: true`, so that Ollama reports token counts. Clients ignore the extra final chunk this produces. Requests sent straight to Ollama's own port aren't seen.

It also polls Ollama's `/api/version`, `/api/tags` and `/api/ps` every 2 seconds for the model list. Hardware figures come from [`sysinfo`](https://crates.io/crates/sysinfo) and from macOS's `powermetrics`, which needs sudo.

**Privacy:** the database stores only per-request metadata (model, token counts, timings), never prompt or response text. The monitor talks only to the Ollama server you point it at; the cost figures come from a pricing table compiled into the binary.

## Requirements

- macOS. Apple Silicon is recommended: the GPU and Neural Engine figures come from `powermetrics`.
- Ollama running (the desktop app or `ollama serve`), by default on port 11434.
- Rust 1.94 or newer, to build.
- sudo, for the GPU and Neural Engine figures only.

## Install

Clone the repository, then from its directory:

```sh
cargo build --release          # binary at target/release/ollama-monitor
cargo install --path .         # or: put ollama-monitor on your PATH via ~/.cargo/bin
```

## Usage

```sh
ollama-monitor
```

Then point your clients at `http://127.0.0.1:11435` instead of Ollama's `:11434`:

```sh
curl http://localhost:11435/api/chat -d '{"model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}]}'
```

At launch it asks for your sudo password, used only to start `powermetrics` for the GPU and Neural Engine figures. Run it as your normal user, not under `sudo`, so the proxy never runs as root. If sudo fails, those two figures show `n/a` and everything else still works.

`--no-tui` runs headless instead: one summary line per request on stderr, still recorded to the database, and no sudo prompt.

To let other machines or VMs use the proxy, bind it to all interfaces with `--proxy-listen 0.0.0.0:11435`. The proxy has no authentication of its own, so this exposes your Ollama to that network, just as `OLLAMA_HOST=0.0.0.0` would.

To capture clients you can't reconfigure, move Ollama to another port and give its usual port to the proxy:

```sh
OLLAMA_HOST=127.0.0.1:11400 ollama serve
ollama-monitor --proxy-listen 127.0.0.1:11434 --ollama-url http://127.0.0.1:11400
```

### Flags

| flag | default | purpose |
|---|---|---|
| `--proxy-listen <ADDR>` | `127.0.0.1:11435` | where the proxy listens |
| `--ollama-url <URL>` | `http://127.0.0.1:11434` (env `OLLAMA_URL`) | upstream Ollama |
| `--config <PATH>` | `~/Library/Application Support/ollama-monitor/config.toml` | optional config file |
| `--db <PATH>` | `~/Library/Application Support/ollama-monitor/usage.db` | SQLite database |
| `--no-tui` | off | headless mode |

### Keys

| key | action |
|---|---|
| `q` or `Ctrl-C` | quit |
| `r` | reset the session counters (the database keeps everything) |
| `p` | pause the display (requests are still recorded) |

## Capture accuracy

| request type | what's measured |
|---|---|
| Ollama native (`/api/chat`, `/api/generate`), streaming or not | exact: token counts and durations come from Ollama itself |
| OpenAI-compatible streaming (`/v1/chat/completions`, `/v1/completions`) | exact token counts; timings measured at the proxy |
| OpenAI-compatible, non-streaming | exact token counts; tokens per second is end to end (it includes prompt processing), so it understates decode speed |
| `:cloud` models when Ollama's cloud relay drops the usage chunk ([ollama/ollama#15169](https://github.com/ollama/ollama/issues/15169)) | estimated: generated tokens are counted from streamed chunks and prompt tokens show as 0; marked with `~` |

A request that ends without any usable stats isn't recorded; the log notes it as `no parsable stats in response`.

## Configuration

Prices live in [`pricing.toml`](pricing.toml) and are compiled in. To change one, add an override to `~/Library/Application Support/ollama-monitor/config.toml`:

```toml
[pricing.providers.anthropic.models.claude-opus-4-8]
input_per_mtok_usd  = 5.00
output_per_mtok_usd = 25.00
```

The cost panel is a rough comparison, not a quote. It applies list prices to your local model's token counts (a frontier model would tokenize the same text differently), and it ignores prompt caching, batch discounts and long-context pricing tiers.

## Files

| file | location |
|---|---|
| database | `~/Library/Application Support/ollama-monitor/usage.db` |
| log | `~/Library/Application Support/ollama-monitor/ollama-monitor.log` |
| config (optional) | `~/Library/Application Support/ollama-monitor/config.toml` |

Set `OLLAMA_MONITOR_LOG=debug` or `OLLAMA_MONITOR_LOG=trace` for more detail in the log; `trace` includes the raw `powermetrics` output.

## Troubleshooting

| symptom | check |
|---|---|
| header shows `server: ● unreachable` | Is Ollama running (`curl http://localhost:11434/api/version`)? The `err:` text after the status names the call that failed. |
| no requests appear | Clients are probably still talking to `:11434`. `curl http://localhost:11435/api/version` should answer through the proxy. |
| "Permission denied" on the log or database at startup | An earlier run under `sudo` left root-owned files: `sudo chown -R "$USER":staff ~/Library/Application\ Support/ollama-monitor`. |
| GPU or ANE shows `n/a` | Run with `OLLAMA_MONITOR_LOG=trace` and search the log for `powermetrics`. |
| sudo prompt fails | Run `sudo -v` first, or use `--no-tui`. |

## Development

```sh
cargo test
cargo fmt --check && cargo clippy --all-targets -- -D warnings
cargo test screen_snapshot -- --nocapture         # print the dashboard at 120x36 and 160x44 from fixtures
cargo test live_ollama -- --ignored --nocapture   # render the models panel from your local Ollama (read-only)
```

CI ([`.gitlab-ci.yml`](.gitlab-ci.yml)) runs the same checks on Linux. [ARCHITECTURE.md](ARCHITECTURE.md) covers the module layout, data flow and design decisions.

Ollama-Monitor shares its dashboard with [LMS-Monitor](https://github.com/Ward-Software-Defined-Systems/LMS-Monitor), a sibling project for LM Studio.

## License

MIT, see [LICENSE](LICENSE).

Ollama-Monitor is an independent project, not affiliated with or endorsed by Ollama, Anthropic or Google.
