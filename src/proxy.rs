//! HTTP reverse proxy in front of Ollama.
//!
//! - Listens on `--proxy-listen`, forwards every request to `--ollama-url`.
//! - Streams the request body to upstream and the response body to the client without
//!   buffering: clients see no added latency.
//! - For inference paths (`/api/chat`, `/api/generate`, `/v1/chat/completions`),
//!   tees each response chunk into `parser::Accumulator`. When the body finishes,
//!   the accumulator is finalized and an `InferenceRecord` is sent to `records_tx`.
//! - Parser failures are isolated; the proxy itself only fails when upstream is
//!   genuinely unreachable.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::str::FromStr;

use anyhow::{Context, Result};
use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderName, HeaderValue, Method, StatusCode};
use axum::response::Response;
use axum::routing::any;
use bytes::Bytes;
use chrono::Utc;
use futures_util::StreamExt;
use tokio::net::TcpListener;
use tokio::sync::{mpsc, watch};
use tracing::{debug, info, warn};

use crate::db::InferenceRecord;
use crate::parser::{self, Accumulator, ParsedStats};

#[derive(Clone)]
struct ProxyState {
    upstream_base: String,
    client: reqwest::Client,
    session_id: i64,
    records_tx: mpsc::Sender<InferenceRecord>,
}

pub async fn serve(
    listen: String,
    upstream_base: String,
    session_id: i64,
    records_tx: mpsc::Sender<InferenceRecord>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    let addr: SocketAddr = listen
        .parse()
        .with_context(|| format!("invalid --proxy-listen value: {}", listen))?;

    let client = reqwest::Client::builder()
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_nodelay(true)
        .build()
        .context("build proxy http client")?;

    let upstream_clean = upstream_base.trim_end_matches('/').to_string();
    let state = ProxyState {
        upstream_base: upstream_clean.clone(),
        client,
        session_id,
        records_tx,
    };

    let app = Router::new().fallback(any(handle)).with_state(state);

    let listener = TcpListener::bind(addr)
        .await
        .with_context(|| format!("bind {}", addr))?;
    info!(
        listen = %addr,
        upstream = %upstream_clean,
        "reverse proxy ready — point clients at http://{} to capture inference stats",
        addr
    );

    let shutdown_signal = async move {
        let _ = shutdown_rx.changed().await;
        info!("proxy received shutdown signal");
    };

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal)
        .await
        .context("axum serve")?;

    Ok(())
}

async fn handle(State(state): State<ProxyState>, req: Request) -> Result<Response, ProxyError> {
    let request_start = std::time::Instant::now();
    let (parts, body) = req.into_parts();
    let method = parts.method.clone();
    let uri = parts.uri.clone();
    let headers = parts.headers;
    let path_and_query = uri
        .path_and_query()
        .map(|pq| pq.as_str().to_string())
        .unwrap_or_else(|| uri.path().to_string());
    let upstream_url = format!("{}{}", state.upstream_base, path_and_query);

    debug!(method = %method, path = %uri.path(), "proxy request received");

    // Read the full body up-front via axum::body::to_bytes (the supported
    // axum 0.8 + hyper 1.x path). 64 MiB cap covers chat/generate easily and
    // bounds the worst case for /api/pull etc.
    let body_bytes = match axum::body::to_bytes(body, 64 * 1024 * 1024).await {
        Ok(b) => b,
        Err(err) => {
            warn!(error = %err, "failed to read incoming request body");
            return Err(ProxyError::ResponseBuild);
        }
    };

    // For OpenAI-compatible streaming endpoints, ensure the upstream returns
    // a usage block so the parser can record the request. Without
    // stream_options.include_usage: true, Ollama omits usage entirely from
    // SSE — and that's the default in most clients (Open WebUI, Continue,
    // raw OpenAI SDK, ...). Injecting it is silent to clients (the extra
    // chunk has choices: []) and turns capture rate from ~0% to ~100%.
    let body_bytes = ensure_openai_include_usage(uri.path(), body_bytes);

    let mut upstream_req = state
        .client
        .request(reqwest_method(&method)?, &upstream_url)
        .body(body_bytes);

    for (name, value) in headers.iter() {
        if is_hop_by_hop(name) {
            continue;
        }
        if matches!(name.as_str(), "host" | "content-length") {
            continue;
        }
        upstream_req = upstream_req.header(name.as_str(), value);
    }

    let upstream_resp = match upstream_req.send().await {
        Ok(r) => r,
        Err(err) => {
            warn!(error = %err, %upstream_url, "upstream request failed");
            return Err(ProxyError::UpstreamUnreachable);
        }
    };

    debug!(method = %method, path = %uri.path(), status = %upstream_resp.status(), "proxy upstream response received");

    let status = upstream_resp.status();
    let upstream_headers = upstream_resp.headers().clone();
    let content_type = upstream_headers
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let envelope = if status.is_success() {
        parser::classify(uri.path(), content_type.as_deref())
    } else {
        None
    };

    let upstream_stream = upstream_resp.bytes_stream();
    let response_body = build_response_body(
        upstream_stream,
        envelope.map(|env| TeeContext {
            envelope: env,
            session_id: state.session_id,
            records_tx: state.records_tx.clone(),
            path: uri.path().to_string(),
            request_start,
        }),
    );

    let mut builder = Response::builder()
        .status(StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY));
    for (name, value) in upstream_headers.iter() {
        if is_hop_by_hop_str(name.as_str()) {
            continue;
        }
        if matches!(name.as_str(), "content-length") {
            continue;
        }
        if let (Ok(n), Ok(v)) = (
            HeaderName::from_str(name.as_str()),
            HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(n, v);
        }
    }

    let resp = builder.body(response_body).map_err(|err| {
        warn!(error = %err, "failed to build response");
        ProxyError::ResponseBuild
    })?;

    Ok(resp)
}

struct TeeContext {
    envelope: parser::Envelope,
    session_id: i64,
    records_tx: mpsc::Sender<InferenceRecord>,
    path: String,
    request_start: std::time::Instant,
}

/// How long to keep draining upstream for the trailing `usage` frame after the
/// client hangs up on an already-finished stream. The `:cloud` relay delivers it
/// within ~100ms of finish_reason; 3s is a generous bound that still can't pin
/// sockets for long.
const USAGE_TRAILER_GRACE: std::time::Duration = std::time::Duration::from_secs(3);

fn build_response_body<S, E>(stream: S, tee: Option<TeeContext>) -> Body
where
    S: futures_util::Stream<Item = Result<Bytes, E>> + Send + 'static,
    E: std::fmt::Display + Send + 'static,
{
    if let Some(tee) = tee {
        // The driver task owns the upstream stream: it tees every chunk into the
        // accumulator and forwards it to the client through a bounded channel
        // (bounded = the client's consumption still backpressures upstream reads).
        // Owning the stream here — instead of the old client-pulled `map` — means a
        // client hangup no longer tears down the upstream read mid-frame: when the
        // stream has already logically finished, we can keep draining briefly to
        // catch the trailing usage frame that agent clients routinely skip out on.
        let (client_tx, client_rx) = mpsc::channel::<Bytes>(64);
        let envelope = tee.envelope;
        let records_tx = tee.records_tx;
        let session_id = tee.session_id;
        let path = tee.path;
        let request_start = tee.request_start;

        tokio::spawn(async move {
            let mut acc = Accumulator::new(envelope, request_start);
            let mut stream = std::pin::pin!(stream);

            enum StreamEnd {
                UpstreamDone,
                ClientGone,
            }

            let end = loop {
                tokio::select! {
                    _ = client_tx.closed() => break StreamEnd::ClientGone,
                    chunk = stream.next() => match chunk {
                        Some(Ok(bytes)) => {
                            acc.push(&bytes);
                            if client_tx.send(bytes).await.is_err() {
                                break StreamEnd::ClientGone;
                            }
                        }
                        Some(Err(err)) => {
                            warn!(error = %err, "upstream stream error mid-response");
                            break StreamEnd::UpstreamDone;
                        }
                        None => break StreamEnd::UpstreamDone,
                    },
                }
            };

            // The client hung up early. If only the usage trailer is outstanding,
            // drain upstream a little longer so the record keeps real token counts.
            // A mid-generation abort skips this on purpose: dropping the upstream
            // stream is what propagates cancellation to Ollama and the cloud relay.
            if matches!(end, StreamEnd::ClientGone) && acc.awaiting_usage_trailer() {
                let drained = tokio::time::timeout(USAGE_TRAILER_GRACE, async {
                    while acc.awaiting_usage_trailer() {
                        match stream.next().await {
                            Some(Ok(bytes)) => acc.push(&bytes),
                            Some(Err(_)) | None => break,
                        }
                    }
                })
                .await;
                match drained {
                    Ok(()) if !acc.awaiting_usage_trailer() => info!(
                        path = %path,
                        "client disconnected before usage frame; drained upstream to keep real token counts"
                    ),
                    _ => debug!(
                        path = %path,
                        "client disconnected before usage frame; trailer never arrived"
                    ),
                }
            }

            match acc.finalize() {
                Some(stats) => {
                    let record = stats_to_record(stats, session_id);
                    debug!(
                        path = %path,
                        model = %record.model_id,
                        prompt = record.prompt_tokens,
                        gen = record.gen_tokens,
                        tps = record.tokens_per_sec,
                        "captured inference stats"
                    );
                    if let Err(err) = records_tx.send(record).await {
                        warn!(error = %err, "records channel closed; dropping record");
                    }
                }
                None => info!(path = %path, "no parsable stats in response (request not recorded)"),
            }
        });

        Body::from_stream(
            tokio_stream::wrappers::ReceiverStream::new(client_rx).map(Ok::<Bytes, Infallible>),
        )
    } else {
        let mapped = stream.map(|chunk_result| match chunk_result {
            Ok(bytes) => Ok::<Bytes, Infallible>(bytes),
            Err(err) => {
                warn!(error = %err, "upstream stream error mid-response (untracked path)");
                Ok(Bytes::new())
            }
        });
        Body::from_stream(mapped)
    }
}

/// For /v1/chat/completions and /v1/completions, set stream_options.include_usage = true
/// in the JSON body so Ollama emits a final SSE chunk containing token counts.
/// Returns the original bytes if the path doesn't match, the body isn't valid JSON,
/// or it isn't a JSON object.
fn ensure_openai_include_usage(path: &str, body: Bytes) -> Bytes {
    if !(path.contains("/v1/chat/completions") || path.contains("/v1/completions")) {
        return body;
    }
    let mut value: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => {
            let prefix = String::from_utf8_lossy(&body[..body.len().min(120)]).into_owned();
            warn!(%path, body_len = body.len(), %err, %prefix, "v1 request body not parseable as JSON; skipping include_usage injection");
            return body;
        }
    };
    let Some(obj) = value.as_object_mut() else {
        warn!(%path, "v1 request body is JSON but not an object; skipping include_usage injection");
        return body;
    };
    let stream = obj.get("stream").and_then(|v| v.as_bool()).unwrap_or(true);
    if !stream {
        return body; // non-streaming returns usage in the single response anyway
    }
    let had_stream_options = obj.contains_key("stream_options");
    let opts = obj
        .entry("stream_options")
        .or_insert_with(|| serde_json::json!({}));
    if let Some(opts_obj) = opts.as_object_mut() {
        opts_obj.insert("include_usage".to_string(), serde_json::Value::Bool(true));
    } else {
        // stream_options was set to a non-object; replace with an object.
        *opts = serde_json::json!({"include_usage": true});
    }
    debug!(%path, had_stream_options, "include_usage injected into streaming v1 request");
    match serde_json::to_vec(&value) {
        Ok(v) => Bytes::from(v),
        Err(err) => {
            warn!(%path, %err, "re-serializing v1 request body failed; forwarding original");
            body
        }
    }
}

fn stats_to_record(stats: ParsedStats, session_id: i64) -> InferenceRecord {
    InferenceRecord {
        session_id,
        model_id: stats.model_id,
        prompt_tokens: stats.prompt_tokens,
        gen_tokens: stats.gen_tokens,
        tokens_per_sec: stats.tokens_per_sec,
        ttft_sec: stats.ttft_sec,
        total_time_sec: stats.total_time_sec,
        stop_reason: stats.stop_reason,
        completed_at: Utc::now(),
        envelope: stats.envelope.as_str().to_string(),
    }
}

// ---------- helpers ----------

#[derive(Debug, thiserror::Error)]
enum ProxyError {
    #[error("upstream unreachable")]
    UpstreamUnreachable,
    #[error("could not build response")]
    ResponseBuild,
    #[error("unsupported method")]
    UnsupportedMethod,
}

impl axum::response::IntoResponse for ProxyError {
    fn into_response(self) -> Response {
        let status = match self {
            ProxyError::UpstreamUnreachable => StatusCode::BAD_GATEWAY,
            ProxyError::ResponseBuild => StatusCode::INTERNAL_SERVER_ERROR,
            ProxyError::UnsupportedMethod => StatusCode::METHOD_NOT_ALLOWED,
        };
        Response::builder()
            .status(status)
            .body(Body::from(self.to_string()))
            .unwrap_or_else(|_| Response::new(Body::empty()))
    }
}

fn reqwest_method(m: &Method) -> Result<reqwest::Method, ProxyError> {
    reqwest::Method::from_bytes(m.as_str().as_bytes()).map_err(|_| ProxyError::UnsupportedMethod)
}

fn is_hop_by_hop(name: &HeaderName) -> bool {
    is_hop_by_hop_str(name.as_str())
}

fn is_hop_by_hop_str(name: &str) -> bool {
    matches!(
        name.to_ascii_lowercase().as_str(),
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "te"
            | "trailers"
            | "transfer-encoding"
            | "upgrade"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn val(b: &Bytes) -> serde_json::Value {
        serde_json::from_slice(b).unwrap()
    }

    #[test]
    fn injects_include_usage_on_streaming_chat_completions() {
        let body = Bytes::from(
            r#"{"model":"qwen3:14b","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
        );
        let out = ensure_openai_include_usage("/v1/chat/completions", body);
        let v = val(&out);
        assert_eq!(
            v["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    #[test]
    fn preserves_existing_stream_options() {
        let body = Bytes::from(
            r#"{"model":"x","messages":[],"stream":true,"stream_options":{"include_usage":false,"foo":"bar"}}"#,
        );
        let out = ensure_openai_include_usage("/v1/chat/completions", body);
        let v = val(&out);
        assert_eq!(
            v["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
        assert_eq!(v["stream_options"]["foo"], serde_json::json!("bar"));
    }

    #[test]
    fn skips_non_streaming() {
        let body_str = r#"{"model":"x","messages":[],"stream":false}"#;
        let body = Bytes::from(body_str);
        let out = ensure_openai_include_usage("/v1/chat/completions", body);
        // Non-streaming: untouched (usage is in the single response anyway).
        assert_eq!(std::str::from_utf8(&out).unwrap(), body_str);
    }

    #[test]
    fn skips_non_openai_path() {
        let body_str = r#"{"model":"x","messages":[],"stream":true}"#;
        let body = Bytes::from(body_str);
        let out = ensure_openai_include_usage("/api/chat", body);
        assert_eq!(std::str::from_utf8(&out).unwrap(), body_str);
    }

    #[test]
    fn skips_invalid_json() {
        let body_str = "not json {{";
        let body = Bytes::from(body_str);
        let out = ensure_openai_include_usage("/v1/chat/completions", body);
        assert_eq!(std::str::from_utf8(&out).unwrap(), body_str);
    }

    #[test]
    fn handles_missing_stream_field_as_streaming() {
        // OpenAI default is non-streaming, but Ollama clients often omit stream and rely on
        // server default. Treat missing as streaming so we still capture.
        let body = Bytes::from(r#"{"model":"x","messages":[]}"#);
        let out = ensure_openai_include_usage("/v1/chat/completions", body);
        let v = val(&out);
        assert_eq!(
            v["stream_options"]["include_usage"],
            serde_json::json!(true)
        );
    }

    fn sse_frame(json: &str) -> Bytes {
        Bytes::from(format!("data: {json}\n\n"))
    }

    fn tee(records_tx: mpsc::Sender<InferenceRecord>) -> TeeContext {
        TeeContext {
            envelope: parser::Envelope::OpenAiSse,
            session_id: 1,
            records_tx,
            path: "/v1/chat/completions".to_string(),
            request_start: std::time::Instant::now(),
        }
    }

    #[tokio::test(start_paused = true)]
    async fn client_abort_after_finish_reason_still_captures_usage() {
        let (records_tx, mut records_rx) = mpsc::channel(4);
        // Upstream: delta, finish_reason, then a cloud-relay-style delay before usage.
        let upstream = futures_util::stream::unfold(0u8, |step| async move {
            match step {
                0 => Some((
                    Ok::<Bytes, Infallible>(sse_frame(
                        r#"{"model":"m","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
                    )),
                    1,
                )),
                1 => Some((
                    Ok(sse_frame(
                        r#"{"choices":[{"delta":{},"finish_reason":"stop"}]}"#,
                    )),
                    2,
                )),
                2 => {
                    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
                    Some((
                        Ok(sse_frame(
                            r#"{"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":7}}"#,
                        )),
                        3,
                    ))
                }
                _ => None,
            }
        });

        let body = build_response_body(upstream, Some(tee(records_tx)));
        let mut data = body.into_data_stream();
        // Client reads through the finish_reason frame, then hangs up (agent style).
        data.next().await.unwrap().unwrap();
        data.next().await.unwrap().unwrap();
        drop(data);

        let record = records_rx.recv().await.expect("record should be emitted");
        assert_eq!(record.envelope, "openai-sse");
        assert_eq!(record.prompt_tokens, 42);
        assert_eq!(record.gen_tokens, 7);
        assert_eq!(record.stop_reason, "stop");
    }

    #[tokio::test(start_paused = true)]
    async fn client_abort_mid_generation_skips_trailer_drain() {
        let (records_tx, mut records_rx) = mpsc::channel(4);
        // Upstream: one delta, then hangs forever (generation still in progress).
        let upstream = futures_util::stream::unfold(0u8, |step| async move {
            match step {
                0 => Some((
                    Ok::<Bytes, Infallible>(sse_frame(
                        r#"{"model":"m","choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#,
                    )),
                    1,
                )),
                _ => {
                    std::future::pending::<()>().await;
                    None
                }
            }
        });

        let body = build_response_body(upstream, Some(tee(records_tx)));
        let mut data = body.into_data_stream();
        data.next().await.unwrap().unwrap();
        drop(data);

        // No finish_reason yet -> the driver must not linger draining; the partial
        // stream finalizes as an approx record (and the upstream stream is dropped,
        // which is what propagates cancellation).
        let record = records_rx.recv().await.expect("record should be emitted");
        assert_eq!(record.envelope, "openai-sse-approx");
        assert_eq!(record.prompt_tokens, 0);
        assert_eq!(record.gen_tokens, 1);
    }
}
