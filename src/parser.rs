//! Stats extraction from Ollama / OpenAI-compatible response bodies.
//!
//! The proxy hands us each chunk of an upstream response as it arrives. We accumulate
//! lines / SSE frames until we see the terminal record with token-count fields, then
//! emit a `ParsedStats`. Failure modes (truncation, missing fields, wrong content) all
//! return `None` so the proxy never fails a request because of us.

use std::time::Duration;

use serde::Deserialize;
use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Envelope {
    OllamaStream,
    OllamaSingle,
    OpenAiSse,
    OpenAiSingle,
    /// Streamed OpenAI-compat response that finished without a `usage` chunk. Emitted
    /// (only by `finalize`, never by `classify`) when upstream — typically Ollama's
    /// `:cloud` proxy, see ollama/ollama#15169 — drops the terminal usage frame.
    /// Token counts are an approximation derived from SSE chunk count; the record is
    /// stored with envelope `"openai-sse-approx"` to keep it distinguishable.
    OpenAiSseNoUsage,
}

impl Envelope {
    pub fn as_str(self) -> &'static str {
        match self {
            Envelope::OllamaStream => "ollama-stream",
            Envelope::OllamaSingle => "ollama-single",
            Envelope::OpenAiSse => "openai-sse",
            Envelope::OpenAiSingle => "openai-single",
            Envelope::OpenAiSseNoUsage => "openai-sse-approx",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ParsedStats {
    pub model_id: String,
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    pub tokens_per_sec: f64,
    pub ttft_sec: f64,
    pub total_time_sec: f64,
    pub stop_reason: String,
    pub envelope: Envelope,
}

/// Pick the envelope from the request URL path and the response Content-Type header.
pub fn classify(path: &str, content_type: Option<&str>) -> Option<Envelope> {
    let ct = content_type.unwrap_or("").to_ascii_lowercase();
    let is_sse = ct.contains("text/event-stream");
    let is_ndjson = ct.contains("application/x-ndjson") || ct.contains("application/jsonl");
    let is_json = ct.contains("application/json");

    let openai_path = path.contains("/v1/chat/completions") || path.contains("/v1/completions");
    let ollama_inference = path.contains("/api/chat")
        || path.contains("/api/generate")
        || path.contains("/api/embed");

    if openai_path {
        return Some(if is_sse {
            Envelope::OpenAiSse
        } else {
            Envelope::OpenAiSingle
        });
    }
    if ollama_inference {
        return Some(if is_ndjson || ct.is_empty() {
            // Ollama default Content-Type for stream=true is application/x-ndjson;
            // for stream=false it's application/json. Some clients set neither.
            // When in doubt, treat as stream — the parser tolerates either layout
            // by falling through to single-object parsing if no newline appears.
            Envelope::OllamaStream
        } else if is_json {
            Envelope::OllamaSingle
        } else {
            Envelope::OllamaStream
        });
    }
    None
}

/// Streaming accumulator. Feed it byte chunks; the proxy tee invokes `finalize`
/// once the upstream body is fully consumed. We also track wall-clock TTFT.
pub struct Accumulator {
    envelope: Envelope,
    buf: Vec<u8>,
    /// Last seen ollama-native NDJSON object that had `done: true`.
    ollama_done: Option<OllamaTerminal>,
    /// OpenAI: usage and finish_reason can appear in different SSE chunks; track separately.
    openai_model: Option<String>,
    openai_usage: Option<OpenAiUsage>,
    openai_finish_reason: Option<String>,
    /// Count of SSE chunks with a non-empty `choices` array. Used as a coarse proxy
    /// for completion tokens when `usage` is absent (cloud-proxy fallback).
    openai_delta_count: u64,
    /// Wall-clock TTFT measured at proxy: time from request received to first body byte.
    pub first_byte: Option<std::time::Instant>,
    /// When the proxy first saw the request — used as the wall-clock baseline.
    pub started: std::time::Instant,
}

impl Accumulator {
    pub fn new(envelope: Envelope, request_start: std::time::Instant) -> Self {
        Self {
            envelope,
            buf: Vec::with_capacity(4096),
            ollama_done: None,
            openai_model: None,
            openai_usage: None,
            openai_finish_reason: None,
            openai_delta_count: 0,
            first_byte: None,
            started: request_start,
        }
    }

    pub fn push(&mut self, chunk: &[u8]) {
        if self.first_byte.is_none() && !chunk.is_empty() {
            self.first_byte = Some(std::time::Instant::now());
        }
        self.buf.extend_from_slice(chunk);
        // For streaming envelopes, drain complete lines / frames eagerly so we don't
        // hold on to the entire response body when we only need the last record.
        match self.envelope {
            Envelope::OllamaStream => self.drain_ndjson(),
            Envelope::OpenAiSse | Envelope::OpenAiSseNoUsage => self.drain_sse(),
            Envelope::OllamaSingle | Envelope::OpenAiSingle => {}
        }
    }

    /// True while the stream has logically finished (a `finish_reason` arrived) but
    /// the trailing `usage` frame has not. Ollama's `:cloud` relay sends the usage
    /// frame a beat after finish_reason, and agent clients often hang up the moment
    /// they see finish_reason — this is the window where the proxy should keep
    /// draining upstream so the record keeps real token counts.
    pub fn awaiting_usage_trailer(&self) -> bool {
        matches!(self.envelope, Envelope::OpenAiSse | Envelope::OpenAiSseNoUsage)
            && self.openai_finish_reason.is_some()
            && self.openai_usage.is_none()
    }

    pub fn finalize(self) -> Option<ParsedStats> {
        let wall_total = self.started.elapsed();
        let wall_ttft = self
            .first_byte
            .map(|t| t.duration_since(self.started))
            .unwrap_or(Duration::ZERO);

        match self.envelope {
            Envelope::OllamaStream => self.ollama_done.map(|t| t.into_parsed(self.envelope)),
            Envelope::OllamaSingle => parse_ollama_single(&self.buf)
                .map(|t| t.into_parsed(Envelope::OllamaSingle)),
            Envelope::OpenAiSse | Envelope::OpenAiSseNoUsage => {
                if let Some(usage) = self.openai_usage {
                    let terminal = OpenAiTerminal {
                        model: self.openai_model.unwrap_or_default(),
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        finish_reason: self.openai_finish_reason.unwrap_or_default(),
                    };
                    Some(terminal.into_parsed(Envelope::OpenAiSse, wall_ttft, wall_total))
                } else if self.openai_model.is_some() || self.openai_delta_count > 0 {
                    // No `usage` chunk arrived — typical for `:cloud` models routed
                    // through ollama.com (see ollama/ollama#15169) and for streams
                    // that get reset before the terminal frame. Record what we know:
                    // chunk count as a rough completion-token estimate.
                    let terminal = OpenAiTerminal {
                        model: self.openai_model.unwrap_or_default(),
                        prompt_tokens: 0,
                        completion_tokens: self.openai_delta_count,
                        finish_reason: self
                            .openai_finish_reason
                            .unwrap_or_else(|| "unknown".to_string()),
                    };
                    Some(terminal.into_parsed(
                        Envelope::OpenAiSseNoUsage,
                        wall_ttft,
                        wall_total,
                    ))
                } else {
                    None
                }
            }
            Envelope::OpenAiSingle => parse_openai_single(&self.buf)
                .map(|t| t.into_parsed(Envelope::OpenAiSingle, wall_ttft, wall_total)),
        }
    }

    fn drain_ndjson(&mut self) {
        while let Some(idx) = self.buf.iter().position(|b| *b == b'\n') {
            let (line, rest) = self.buf.split_at(idx + 1);
            let line = line[..line.len() - 1].to_vec();
            let remainder = rest.to_vec();
            self.buf = remainder;
            if line.is_empty() {
                continue;
            }
            if let Ok(parsed) = serde_json::from_slice::<OllamaChunk>(&line) {
                if parsed.done {
                    self.ollama_done = Some(parsed.into_terminal());
                }
            } else {
                debug!(
                    line_bytes = line.len(),
                    "ollama ndjson line failed to parse; ignoring"
                );
            }
        }
    }

    fn drain_sse(&mut self) {
        // SSE frames are separated by blank line ("\n\n").
        while let Some(idx) = find_double_newline(&self.buf) {
            let frame: Vec<u8> = self.buf.drain(..idx + 2).collect();
            for line in frame.split(|b| *b == b'\n') {
                let line = trim_ascii(line);
                if let Some(rest) = line.strip_prefix(b"data:") {
                    let payload = trim_ascii(rest);
                    if payload == b"[DONE]" {
                        continue;
                    }
                    if let Ok(parsed) = serde_json::from_slice::<OpenAiSseChunk>(payload) {
                        if let Some(model) = parsed.model {
                            self.openai_model.get_or_insert(model);
                        }
                        if let Some(choices) = parsed.choices.as_ref() {
                            if !choices.is_empty() {
                                // Each content-bearing chunk counts as ~1 generation
                                // unit. Used as a fallback gen-token estimate when
                                // `usage` never arrives (cloud proxy).
                                self.openai_delta_count += 1;
                            }
                            if let Some(reason) =
                                choices.first().and_then(|c| c.finish_reason.clone())
                            {
                                self.openai_finish_reason = Some(reason);
                            }
                        }
                        if let Some(usage) = parsed.usage {
                            self.openai_usage = Some(usage);
                        }
                    }
                }
            }
        }
    }
}

fn find_double_newline(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

fn trim_ascii(b: &[u8]) -> &[u8] {
    let start = b.iter().position(|c| !c.is_ascii_whitespace()).unwrap_or(b.len());
    let end = b
        .iter()
        .rposition(|c| !c.is_ascii_whitespace())
        .map(|i| i + 1)
        .unwrap_or(0);
    if start >= end { &[] } else { &b[start..end] }
}

// ---------------------------------------------------------------------------
// Ollama-native (chat / generate)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OllamaChunk {
    #[serde(default)]
    model: String,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    eval_count: Option<u64>,
    #[serde(default)]
    eval_duration: Option<u64>,
    #[serde(default)]
    prompt_eval_count: Option<u64>,
    #[serde(default)]
    prompt_eval_duration: Option<u64>,
    #[serde(default)]
    total_duration: Option<u64>,
    #[serde(default)]
    load_duration: Option<u64>,
}

#[derive(Debug, Clone)]
struct OllamaTerminal {
    model: String,
    prompt_tokens: u64,
    gen_tokens: u64,
    eval_duration_ns: u64,
    prompt_eval_duration_ns: u64,
    load_duration_ns: u64,
    total_duration_ns: u64,
    done_reason: String,
}

impl OllamaChunk {
    fn into_terminal(self) -> OllamaTerminal {
        OllamaTerminal {
            model: self.model,
            prompt_tokens: self.prompt_eval_count.unwrap_or(0),
            gen_tokens: self.eval_count.unwrap_or(0),
            eval_duration_ns: self.eval_duration.unwrap_or(0),
            prompt_eval_duration_ns: self.prompt_eval_duration.unwrap_or(0),
            load_duration_ns: self.load_duration.unwrap_or(0),
            total_duration_ns: self.total_duration.unwrap_or(0),
            done_reason: self.done_reason.unwrap_or_else(|| "stop".into()),
        }
    }
}

impl OllamaTerminal {
    fn into_parsed(self, envelope: Envelope) -> ParsedStats {
        let eval_secs = ns_to_secs(self.eval_duration_ns);
        let tokens_per_sec = if eval_secs > 0.0 {
            self.gen_tokens as f64 / eval_secs
        } else {
            0.0
        };
        // First-token approximation: prompt-eval + load (load contributes only on cold start).
        let ttft_sec =
            ns_to_secs(self.prompt_eval_duration_ns) + ns_to_secs(self.load_duration_ns);
        ParsedStats {
            model_id: self.model,
            prompt_tokens: self.prompt_tokens,
            gen_tokens: self.gen_tokens,
            tokens_per_sec,
            ttft_sec,
            total_time_sec: ns_to_secs(self.total_duration_ns),
            stop_reason: self.done_reason,
            envelope,
        }
    }
}

fn parse_ollama_single(body: &[u8]) -> Option<OllamaTerminal> {
    serde_json::from_slice::<OllamaChunk>(body)
        .ok()
        .filter(|c| c.done || c.eval_count.is_some())
        .map(OllamaChunk::into_terminal)
}

fn ns_to_secs(ns: u64) -> f64 {
    (ns as f64) / 1_000_000_000.0
}

// ---------------------------------------------------------------------------
// OpenAI-compatible
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct OpenAiSseChunk {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Option<Vec<OpenAiChoice>>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Debug, Clone)]
struct OpenAiTerminal {
    model: String,
    prompt_tokens: u64,
    completion_tokens: u64,
    finish_reason: String,
}

impl OpenAiTerminal {
    fn into_parsed(self, envelope: Envelope, wall_ttft: Duration, wall_total: Duration) -> ParsedStats {
        let total_secs = wall_total.as_secs_f64().max(0.0001);
        let (ttft_sec, gen_window) = match envelope {
            // SSE: wall_ttft is meaningful (time to first chunk); the gap to total is
            // the generation window.
            Envelope::OpenAiSse | Envelope::OpenAiSseNoUsage => {
                let t = wall_ttft.as_secs_f64();
                let gen_window = (total_secs - t).max(0.05);
                (t, gen_window)
            }
            // Non-streaming: every byte arrives together; we can't separate ttft from
            // total. Report ttft=0 and use total wall-clock as the throughput
            // denominator. The resulting tok/s is end-to-end (prompt+gen+net), a lower
            // bound on the pure decode rate. Honest given what we can observe.
            _ => (0.0, total_secs),
        };
        let tokens_per_sec = if self.completion_tokens > 0 {
            self.completion_tokens as f64 / gen_window
        } else {
            0.0
        };
        ParsedStats {
            model_id: self.model,
            prompt_tokens: self.prompt_tokens,
            gen_tokens: self.completion_tokens,
            tokens_per_sec,
            ttft_sec,
            total_time_sec: total_secs,
            stop_reason: self.finish_reason,
            envelope,
        }
    }
}

#[derive(Debug, Deserialize)]
struct OpenAiNonStreamResp {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Option<Vec<OpenAiChoice>>,
    usage: OpenAiUsage,
}

fn parse_openai_single(body: &[u8]) -> Option<OpenAiTerminal> {
    serde_json::from_slice::<OpenAiNonStreamResp>(body)
        .ok()
        .map(|r| OpenAiTerminal {
            model: r.model.unwrap_or_default(),
            prompt_tokens: r.usage.prompt_tokens,
            completion_tokens: r.usage.completion_tokens,
            finish_reason: r
                .choices
                .as_ref()
                .and_then(|c| c.first())
                .and_then(|c| c.finish_reason.clone())
                .unwrap_or_default(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_stream(env: Envelope, body: &[u8]) -> Option<ParsedStats> {
        let mut acc = Accumulator::new(env, std::time::Instant::now());
        // Slow-feed in 64-byte chunks to exercise the streaming buffering.
        for chunk in body.chunks(64) {
            acc.push(chunk);
        }
        acc.finalize()
    }

    #[test]
    fn classify_routes_correctly() {
        assert_eq!(
            classify("/api/chat", Some("application/x-ndjson")),
            Some(Envelope::OllamaStream)
        );
        assert_eq!(
            classify("/api/generate", Some("application/json")),
            Some(Envelope::OllamaSingle)
        );
        assert_eq!(
            classify("/v1/chat/completions", Some("text/event-stream")),
            Some(Envelope::OpenAiSse)
        );
        assert_eq!(
            classify("/v1/chat/completions", Some("application/json")),
            Some(Envelope::OpenAiSingle)
        );
        assert_eq!(classify("/api/tags", Some("application/json")), None);
    }

    #[test]
    fn ollama_stream_extracts_final_chunk() {
        let body = concat!(
            r#"{"model":"qwen3:14b","done":false}"#, "\n",
            r#"{"model":"qwen3:14b","done":false,"response":"hi"}"#, "\n",
            r#"{"model":"qwen3:14b","done":true,"done_reason":"stop","total_duration":2000000000,"load_duration":100000000,"prompt_eval_count":12,"prompt_eval_duration":300000000,"eval_count":80,"eval_duration":1500000000}"#, "\n"
        );
        let stats = parse_stream(Envelope::OllamaStream, body.as_bytes()).expect("stats");
        assert_eq!(stats.model_id, "qwen3:14b");
        assert_eq!(stats.prompt_tokens, 12);
        assert_eq!(stats.gen_tokens, 80);
        assert!((stats.tokens_per_sec - (80.0 / 1.5)).abs() < 1e-6);
        assert!((stats.total_time_sec - 2.0).abs() < 1e-6);
        assert!((stats.ttft_sec - 0.4).abs() < 1e-6);
        assert_eq!(stats.stop_reason, "stop");
        assert_eq!(stats.envelope, Envelope::OllamaStream);
    }

    #[test]
    fn ollama_single_extracts_object() {
        let body = r#"{"model":"qwen3:14b","done":true,"done_reason":"length","total_duration":1000000000,"prompt_eval_count":5,"prompt_eval_duration":200000000,"eval_count":40,"eval_duration":600000000}"#;
        let stats = parse_stream(Envelope::OllamaSingle, body.as_bytes()).expect("stats");
        assert_eq!(stats.gen_tokens, 40);
        assert_eq!(stats.stop_reason, "length");
    }

    #[test]
    fn openai_sse_extracts_usage_when_present() {
        let body = "data: {\"id\":\"x\",\"model\":\"qwen3:14b\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                    data: {\"id\":\"x\",\"model\":\"qwen3:14b\",\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}\n\n\
                    data: {\"id\":\"x\",\"model\":\"qwen3:14b\",\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":50,\"total_tokens\":60}}\n\n\
                    data: [DONE]\n\n";
        let stats = parse_stream(Envelope::OpenAiSse, body.as_bytes()).expect("stats");
        assert_eq!(stats.prompt_tokens, 10);
        assert_eq!(stats.gen_tokens, 50);
        assert_eq!(stats.envelope, Envelope::OpenAiSse);
        // wall-clock TTFT in tests is essentially 0; we just make sure it's a finite number.
        assert!(stats.total_time_sec.is_finite());
    }

    #[test]
    fn openai_sse_falls_back_to_chunk_count_without_usage() {
        // Cloud-shaped response: chunks stream, but no terminal `usage` frame ever
        // arrives. We should still record the request, using chunk count as the
        // completion-token estimate and tagging the envelope as approximate.
        let body = "data: {\"id\":\"x\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n\
                    data: {\"id\":\"x\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"delta\":{\"content\":\" there\"}}]}\n\n\
                    data: {\"id\":\"x\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"finish_reason\":\"stop\",\"delta\":{}}]}\n\n\
                    data: [DONE]\n\n";
        let stats = parse_stream(Envelope::OpenAiSse, body.as_bytes()).expect("stats");
        assert_eq!(stats.envelope, Envelope::OpenAiSseNoUsage);
        assert_eq!(stats.envelope.as_str(), "openai-sse-approx");
        assert_eq!(stats.model_id, "deepseek-v4-pro");
        assert_eq!(stats.prompt_tokens, 0);
        // 3 chunks had non-empty `choices` (two content + one finish_reason frame).
        assert_eq!(stats.gen_tokens, 3);
        assert_eq!(stats.stop_reason, "stop");
        assert!(stats.tokens_per_sec.is_finite());
    }

    #[test]
    fn openai_sse_returns_none_when_truly_empty() {
        // No chunks at all (e.g. immediate upstream close) → nothing to record.
        let body = "data: [DONE]\n\n";
        assert!(parse_stream(Envelope::OpenAiSse, body.as_bytes()).is_none());
    }

    #[test]
    fn openai_sse_fallback_uses_unknown_when_no_finish_reason() {
        // Cloud response that gets reset mid-stream: chunks arrive, no finish_reason,
        // no usage. Stop reason should default to "unknown".
        let body = "data: {\"id\":\"x\",\"model\":\"deepseek-v4-pro\",\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n";
        let stats = parse_stream(Envelope::OpenAiSse, body.as_bytes()).expect("stats");
        assert_eq!(stats.envelope, Envelope::OpenAiSseNoUsage);
        assert_eq!(stats.stop_reason, "unknown");
        assert_eq!(stats.gen_tokens, 1);
    }

    #[test]
    fn openai_single_extracts_usage() {
        let body = r#"{"id":"x","model":"qwen3:14b","choices":[{"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":42,"total_tokens":49}}"#;
        let stats = parse_stream(Envelope::OpenAiSingle, body.as_bytes()).expect("stats");
        assert_eq!(stats.prompt_tokens, 7);
        assert_eq!(stats.gen_tokens, 42);
        assert_eq!(stats.stop_reason, "stop");
    }

    #[test]
    fn truncated_ollama_stream_is_dropped() {
        let body = r#"{"model":"qwen3:14b","done":false}"#;
        // No newline, no done:true — accumulator never sees a complete terminal line.
        assert!(parse_stream(Envelope::OllamaStream, body.as_bytes()).is_none());
    }

    #[test]
    fn fixture_ollama_chat_stream() {
        let body = include_bytes!("../fixtures/ollama-chat-stream.jsonl");
        let stats = parse_stream(Envelope::OllamaStream, body).expect("stats");
        assert_eq!(stats.model_id, "qwen3:14b");
        assert_eq!(stats.envelope, Envelope::OllamaStream);
        assert!(stats.gen_tokens > 0);
        assert!(stats.prompt_tokens > 0);
        assert!(stats.tokens_per_sec > 0.0);
    }

    #[test]
    fn fixture_ollama_chat_nonstream() {
        let body = include_bytes!("../fixtures/ollama-chat-nonstream.json");
        let stats = parse_stream(Envelope::OllamaSingle, body).expect("stats");
        assert_eq!(stats.model_id, "qwen3:14b");
        assert_eq!(stats.envelope, Envelope::OllamaSingle);
        assert!(stats.gen_tokens > 0);
    }

    #[test]
    fn fixture_openai_sse_keeps_finish_reason_across_chunks() {
        // Real fixture: finish_reason in chunk N-1, usage in chunk N.
        let body = include_bytes!("../fixtures/openai-compat-sse.txt");
        let stats = parse_stream(Envelope::OpenAiSse, body).expect("stats");
        assert_eq!(stats.model_id, "qwen3:14b");
        assert_eq!(stats.envelope, Envelope::OpenAiSse);
        assert_eq!(stats.prompt_tokens, 11);
        assert_eq!(stats.gen_tokens, 5);
        // The fix: finish_reason "length" lives in a separate chunk from usage.
        assert_eq!(stats.stop_reason, "length");
    }

    #[test]
    fn fixture_openai_sse_cloud_falls_back_without_usage() {
        // Real-shape cloud response: same fixture as above with the terminal `usage`
        // frame stripped — models ollama/ollama#15169.
        let body = include_bytes!("../fixtures/openai-compat-sse-cloud-nousage.txt");
        let stats = parse_stream(Envelope::OpenAiSse, body).expect("stats");
        assert_eq!(stats.envelope, Envelope::OpenAiSseNoUsage);
        assert_eq!(stats.envelope.as_str(), "openai-sse-approx");
        assert_eq!(stats.prompt_tokens, 0);
        assert!(stats.gen_tokens > 0, "chunk count should be non-zero");
        assert_eq!(stats.stop_reason, "length");
    }

    #[test]
    fn fixture_openai_nonstream() {
        let body = include_bytes!("../fixtures/openai-compat-nonstream.json");
        let stats = parse_stream(Envelope::OpenAiSingle, body).expect("stats");
        assert_eq!(stats.model_id, "qwen3:14b");
        assert_eq!(stats.prompt_tokens, 11);
        assert_eq!(stats.gen_tokens, 5);
        assert_eq!(stats.stop_reason, "length");
    }
}
