use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use clap::Parser;
use tokio::signal::unix::{SignalKind, signal};
use tokio::sync::{mpsc, watch};
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

mod aggregate;
mod api;
mod config;
mod db;
mod hardware;
mod parser;
mod pricing;
mod proxy;
mod tui;

use crate::api::ModelsSnapshot;
use crate::db::{DbHandle, FailedRequest, InferenceRecord};
use crate::hardware::HardwareSnapshot;

#[derive(Parser, Debug, Clone)]
#[command(name = "ollama-monitor", version, about)]
pub struct Cli {
    /// Address the reverse proxy listens on. Clients should target this; it forwards to --ollama-url.
    #[arg(long, default_value = "127.0.0.1:11435")]
    pub proxy_listen: String,

    /// Upstream Ollama base URL. The proxy forwards to it; the poller queries it directly.
    #[arg(long, default_value = "http://127.0.0.1:11434", env = "OLLAMA_URL")]
    pub ollama_url: String,

    /// Optional user config TOML. Default: config.toml under ~/Library/Application Support/ollama-monitor (macOS) or $XDG_CONFIG_HOME/ollama-monitor (Linux; ~/.config).
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// SQLite usage DB. Default: usage.db under ~/Library/Application Support/ollama-monitor (macOS) or $XDG_DATA_HOME/ollama-monitor (Linux; ~/.local/share).
    #[arg(long)]
    pub db: Option<PathBuf>,

    /// Headless mode: print one summary line per completed or failed inference request to stderr; no TUI.
    #[arg(long)]
    pub no_tui: bool,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let paths = config::resolve_paths(&cli)?;

    // Logging first, so warnings from loading the config and pricing reach the log.
    let log_writer = init_tracing(&paths.log_file, cli.no_tui)?;
    info!(
        proxy = %cli.proxy_listen,
        upstream = %cli.ollama_url,
        db = %paths.db_file.display(),
        log = %paths.log_file.display(),
        "ollama-monitor starting"
    );
    let user_config = config::load_user_config(&paths.config_file)?;
    let pricing = pricing::load(user_config.as_ref());

    // Claim the proxy port before the telemetry prime step (the sudo prompt on macOS) and
    // the TUI: a taken port or a bad address should stop us here, not leave a dashboard
    // with nothing behind it.
    let listener = proxy::bind(&cli.proxy_listen)
        .inspect_err(|err| tracing::error!("proxy could not start: {err:#}"))?;

    if !cli.no_tui {
        // Prime the telemetry backend BEFORE entering raw mode: on macOS that's the sudo
        // prompt, which needs a cooked terminal. A no-op on Linux.
        hardware::prime();
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    let result = runtime.block_on(async_main(cli, paths, pricing, listener));

    runtime.shutdown_background();
    drop(log_writer);
    result
}

async fn async_main(
    cli: Cli,
    paths: config::Paths,
    pricing: Arc<pricing::PricingTable>,
    listener: std::net::TcpListener,
) -> Result<()> {
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    let db_handle = db::open_and_spawn_writer(&paths.db_file, shutdown_rx.clone()).await?;
    let session_id = db_handle.start_session().await?;
    info!(session_id, "db opened, session started");

    let (records_tx, records_rx) = mpsc::channel::<InferenceRecord>(256);
    let (failures_tx, failures_rx) = mpsc::channel::<FailedRequest>(256);

    // proxy
    let serve = proxy::serve(
        listener,
        cli.ollama_url.clone(),
        session_id,
        records_tx.clone(),
        failures_tx,
        shutdown_rx.clone(),
    );
    let mut workers = vec![tokio::spawn(async move {
        if let Err(err) = serve.await {
            tracing::error!("proxy stopped: {err:#}");
        }
    })];

    // signal handling -> shutdown
    let shutdown_for_signals = shutdown_tx.clone();
    tokio::spawn(async move {
        let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
        let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
        // Closing the terminal window sends SIGHUP; shut down cleanly instead of dying.
        let mut sighup = signal(SignalKind::hangup()).expect("install SIGHUP handler");
        info!("signal handlers installed (SIGTERM, SIGINT, SIGHUP)");
        tokio::select! {
            _ = sigterm.recv() => info!("SIGTERM received"),
            _ = sigint.recv() => info!("SIGINT received"),
            _ = sighup.recv() => info!("SIGHUP received (terminal closed)"),
        }
        if let Err(err) = shutdown_for_signals.send(true) {
            tracing::warn!(error = %err, "failed to broadcast shutdown");
        }
    });

    // No `?` on the UI: even when it fails, workers still get shut down and the session
    // row still gets its ended_at.
    let ui_result = if cli.no_tui {
        // Headless has no models or hardware panels, so the poller and the sampler (with
        // its telemetry child) never start.
        run_headless(
            records_rx,
            failures_rx,
            db_handle.clone(),
            shutdown_rx.clone(),
        )
        .await
    } else {
        let (models_tx, models_rx) = mpsc::channel::<ModelsSnapshot>(8);
        let (hw_tx, hw_rx) = mpsc::channel::<HardwareSnapshot>(8);
        workers.push(tokio::spawn(api::poll_models(
            cli.ollama_url.clone(),
            models_tx,
            shutdown_rx.clone(),
        )));
        workers.push(tokio::spawn(hardware::sample(hw_tx, shutdown_rx.clone())));
        tui::run(
            session_id,
            records_rx,
            failures_rx,
            models_rx,
            hw_rx,
            db_handle.clone(),
            pricing.clone(),
            cli.ollama_url.clone(),
            cli.proxy_listen.clone(),
            shutdown_tx.clone(),
            shutdown_rx.clone(),
        )
        .await
    };
    if let Err(err) = &ui_result {
        tracing::error!("ui failed: {err:#}");
    }

    info!("ui exited; broadcasting shutdown to workers");
    let _ = shutdown_tx.send(true);

    // Give workers up to 3s to drain. Whoever's left is dropped along with the runtime
    // (shutdown_background in main), so we never hang.
    if tokio::time::timeout(
        std::time::Duration::from_secs(3),
        futures_util::future::join_all(workers),
    )
    .await
    .is_err()
    {
        tracing::warn!("workers did not exit within 3s; forcing shutdown");
    }
    db_handle.end_session(session_id).await?;
    info!("shutdown complete");
    ui_result
}

async fn run_headless(
    mut records_rx: mpsc::Receiver<InferenceRecord>,
    mut failures_rx: mpsc::Receiver<FailedRequest>,
    db: DbHandle,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    loop {
        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() { break; }
            }
            maybe_record = records_rx.recv() => {
                let Some(record) = maybe_record else { break };
                // ~ prefixes flag approximate captures from the cloud fallback path.
                let approx = record.envelope == "openai-sse-approx";
                let p = if approx { "~" } else { "" };
                // writeln!, not eprintln!: once a SIGHUP has taken the terminal away,
                // eprintln! would panic on the failed write.
                let _ = writeln!(
                    std::io::stderr(),
                    "[{}] {} | prompt={} gen={}{} tok/s={}{:.1} ttft={:.2}s total={:.2}s ({}{})",
                    record.completed_at.with_timezone(&chrono::Local).format("%H:%M:%S"),
                    record.model_id,
                    record.prompt_tokens,
                    p,
                    record.gen_tokens,
                    p,
                    record.tokens_per_sec,
                    record.ttft_sec,
                    record.total_time_sec,
                    record.stop_reason,
                    if approx { ", approx" } else { "" },
                );
                let _ = db.persist(record).await;
            }
            // `Some(..)`, unlike the records arm: if the failures channel closes, only this
            // arm stops.
            Some(failure) = failures_rx.recv() => {
                let _ = writeln!(
                    std::io::stderr(),
                    "[{}] {} | {} {}",
                    failure.failed_at.with_timezone(&chrono::Local).format("%H:%M:%S"),
                    failure.model_id.as_deref().unwrap_or("?"),
                    failure.summary(),
                    failure.path,
                );
                let _ = db.persist_failure(failure).await;
            }
        }
    }
    Ok(())
}

fn init_tracing(
    log_path: &Path,
    allow_stderr: bool,
) -> Result<tracing_appender::non_blocking::WorkerGuard> {
    let parent = log_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("log path has no parent: {}", log_path.display()))?;
    std::fs::create_dir_all(parent)?;

    let file_appender = tracing_appender::rolling::never(
        parent,
        log_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("log path has no filename"))?,
    );
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);

    let env =
        EnvFilter::try_from_env("OLLAMA_MONITOR_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

    let file_layer = fmt::layer()
        .with_writer(file_writer)
        .with_ansi(false)
        .with_target(false);

    let registry = tracing_subscriber::registry().with(env).with(file_layer);
    if allow_stderr {
        let stderr_layer = fmt::layer()
            .with_writer(std::io::stderr)
            .with_ansi(true)
            .with_target(false);
        registry.with(stderr_layer).init();
    } else {
        registry.init();
    }
    Ok(guard)
}
