//! Top-level TUI entrypoint. Sets up terminal, event loop, and per-frame render.
//! Panel layout and widgets mirror LMS-Monitor's TUI.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::{DefaultTerminal, Frame};
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, interval};
use tracing::{debug, info, warn};

use crate::aggregate::Aggregator;
use crate::api::{self, ModelInfo, ModelsSnapshot};
use crate::db::{DbHandle, InferenceRecord, LifetimeTotals};
use crate::hardware::HardwareSnapshot;
use crate::pricing::PricingTable;

mod layout;
mod widgets;

use widgets::ServerStatus;

const FEED_CAPACITY: usize = 30;
const TICK: Duration = Duration::from_millis(250);

pub struct AppState {
    #[allow(dead_code)] // surfaced in tracing breadcrumbs; future header chip
    pub session_id: i64,
    pub base_url: String,
    pub proxy_listen: String,
    pub version: Option<String>,
    pub aggregator: Aggregator,
    pub pricing: Arc<PricingTable>,
    /// Newest first.
    pub feed: VecDeque<InferenceRecord>,
    pub models: Vec<ModelInfo>,
    pub server_status: ServerStatus,
    pub server_error: Option<String>,
    pub last_inference_model_id: Option<String>,
    pub paused: bool,
    pub lifetime: LifetimeTotals,
    pub hardware: HardwareSnapshot,
    pub now: DateTime<Utc>,
}

impl AppState {
    fn new(
        session_id: i64,
        base_url: String,
        proxy_listen: String,
        pricing: Arc<PricingTable>,
    ) -> Self {
        Self {
            session_id,
            base_url,
            proxy_listen,
            version: None,
            aggregator: Aggregator::new(),
            pricing,
            feed: VecDeque::with_capacity(FEED_CAPACITY),
            models: Vec::new(),
            server_status: ServerStatus::Unknown,
            server_error: None,
            last_inference_model_id: None,
            paused: false,
            lifetime: LifetimeTotals::default(),
            hardware: HardwareSnapshot::default(),
            now: Utc::now(),
        }
    }

    fn ingest_models(&mut self, snap: ModelsSnapshot) {
        match snap {
            ModelsSnapshot::Loaded { version, models } => {
                self.server_status = ServerStatus::Reachable;
                self.server_error = None;
                if version.is_some() {
                    self.version = version;
                }
                self.models = models;
            }
            ModelsSnapshot::Unreachable { reason } => {
                // Keep the last model list and version, as LMS-Monitor does: a blip
                // shouldn't blank the panel.
                self.server_status = ServerStatus::Unreachable;
                self.server_error = Some(reason);
            }
        }
    }

    /// UI-side ingest only; the run loop persists every record, paused or not.
    fn ingest_record(&mut self, record: InferenceRecord) {
        if self.paused {
            return;
        }
        self.last_inference_model_id = Some(record.model_id.clone());
        self.aggregator.ingest(record.clone());
        if self.feed.len() == FEED_CAPACITY {
            self.feed.pop_back();
        }
        self.feed.push_front(record);
    }

    fn reset_session(&mut self) {
        self.feed.clear();
        self.aggregator.reset();
    }

    fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }
}

fn render(f: &mut Frame, state: &AppState) {
    let l = layout::compute(f.area());
    widgets::render_header(
        f,
        l.header,
        widgets::HeaderInfo {
            server_status: state.server_status,
            server_error: state.server_error.as_deref(),
            base_url: &state.base_url,
            version: state.version.as_deref(),
            proxy_listen: &state.proxy_listen,
            paused: state.paused,
            lifetime: &state.lifetime,
            now: state.now,
        },
    );
    // Resolved per frame so a record that lands before the first /api/tags poll still
    // gets its ▸ once the model list arrives.
    let recent = api::resolve_model_id(&state.models, state.last_inference_model_id.as_deref());
    widgets::render_models(f, l.models, &state.models, recent);
    widgets::render_hardware(f, l.hardware, &state.hardware);
    widgets::render_feed(f, l.feed, &state.feed);
    let snap = state.aggregator.snapshot(state.now);
    widgets::render_rolling(f, l.rolling, &snap);
    widgets::render_costs(f, l.costs, &snap, &state.pricing);
    widgets::render_footer(f, l.footer);
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    session_id: i64,
    mut records_rx: mpsc::Receiver<InferenceRecord>,
    mut models_rx: mpsc::Receiver<ModelsSnapshot>,
    mut hw_rx: mpsc::Receiver<HardwareSnapshot>,
    db: DbHandle,
    pricing: Arc<PricingTable>,
    base_url: String,
    proxy_listen: String,
    shutdown_tx: watch::Sender<bool>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    install_panic_hook();
    let mut terminal = ratatui::init();

    let mut state = AppState::new(session_id, base_url, proxy_listen, pricing);

    // Lifetime-totals poller: own read-only connection, every 2s.
    let (lifetime_tx, mut lifetime_rx) = mpsc::channel::<LifetimeTotals>(4);
    spawn_lifetime_poller(db.clone(), lifetime_tx, shutdown_rx.clone());

    // Keyboard input on its own blocking thread → mpsc.
    let (key_tx, mut key_rx) = mpsc::channel::<Event>(64);
    spawn_input_thread(key_tx, shutdown_rx.clone());

    let mut tick = interval(TICK);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let result = run_loop(
        &mut terminal,
        &mut state,
        &db,
        &shutdown_tx,
        &mut shutdown_rx,
        &mut records_rx,
        &mut models_rx,
        &mut hw_rx,
        &mut lifetime_rx,
        &mut key_rx,
        &mut tick,
    )
    .await;

    ratatui::restore();
    info!("tui exited");
    result
}

#[allow(clippy::too_many_arguments)]
async fn run_loop(
    terminal: &mut DefaultTerminal,
    state: &mut AppState,
    db: &DbHandle,
    shutdown_tx: &watch::Sender<bool>,
    shutdown_rx: &mut watch::Receiver<bool>,
    records_rx: &mut mpsc::Receiver<InferenceRecord>,
    models_rx: &mut mpsc::Receiver<ModelsSnapshot>,
    hw_rx: &mut mpsc::Receiver<HardwareSnapshot>,
    lifetime_rx: &mut mpsc::Receiver<LifetimeTotals>,
    key_rx: &mut mpsc::Receiver<Event>,
    tick: &mut tokio::time::Interval,
) -> Result<()> {
    loop {
        // Redraw after every event (as LMS-Monitor does), not just on the tick, so a busy
        // record stream can't starve the screen; the tick keeps the clock moving.
        state.now = Utc::now();
        terminal
            .draw(|f| render(f, state))
            .context("draw")?;

        tokio::select! {
            biased;
            _ = shutdown_rx.changed() => {
                if *shutdown_rx.borrow() { return Ok(()); }
            }
            Some(event) = key_rx.recv() => {
                if handle_input(event, state, shutdown_tx) { return Ok(()); }
            }
            Some(record) = records_rx.recv() => {
                let to_persist = record.clone();
                state.ingest_record(record);
                let _ = db.persist(to_persist).await;
            }
            Some(snap) = models_rx.recv() => state.ingest_models(snap),
            Some(hw) = hw_rx.recv() => state.hardware = hw,
            Some(totals) = lifetime_rx.recv() => state.lifetime = totals,
            _ = tick.tick() => {}
        }
    }
}

fn handle_input(event: Event, state: &mut AppState, shutdown_tx: &watch::Sender<bool>) -> bool {
    let Event::Key(key) = event else {
        return false;
    };
    // Presses only: terminals with enhanced keyboard reporting also send releases,
    // which would toggle pause twice.
    if key.kind != KeyEventKind::Press {
        return false;
    }
    match key.code {
        KeyCode::Char('q') | KeyCode::Char('Q') => {
            let _ = shutdown_tx.send(true);
            return true;
        }
        KeyCode::Char('c') | KeyCode::Char('C') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            let _ = shutdown_tx.send(true);
            return true;
        }
        KeyCode::Char('r') | KeyCode::Char('R') => {
            state.reset_session();
            debug!("session counters reset (records remain in DB)");
        }
        KeyCode::Char('p') | KeyCode::Char('P') => {
            state.toggle_pause();
            debug!(paused = state.paused, "pause toggled");
        }
        _ => {}
    }
    false
}

fn install_panic_hook() {
    let original = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        ratatui::restore();
        original(info);
    }));
}

fn spawn_input_thread(tx: mpsc::Sender<Event>, shutdown_rx: watch::Receiver<bool>) {
    std::thread::spawn(move || {
        loop {
            if *shutdown_rx.borrow() {
                return;
            }
            match crossterm::event::poll(Duration::from_millis(250)) {
                Ok(true) => match crossterm::event::read() {
                    Ok(ev) => {
                        if tx.blocking_send(ev).is_err() {
                            return;
                        }
                    }
                    Err(err) => {
                        warn!(error = %err, "crossterm read failed");
                        return;
                    }
                },
                Ok(false) => {}
                Err(err) => {
                    warn!(error = %err, "crossterm poll failed");
                    return;
                }
            }
        }
    });
}

fn spawn_lifetime_poller(
    db: DbHandle,
    tx: mpsc::Sender<LifetimeTotals>,
    mut shutdown_rx: watch::Receiver<bool>,
) {
    tokio::spawn(async move {
        let reader = match db.open_reader() {
            Ok(c) => c,
            Err(err) => {
                warn!(error = %err, "could not open db reader for lifetime totals");
                return;
            }
        };
        let mut next = Instant::now();
        loop {
            let totals_res = tokio::task::block_in_place(|| crate::db::lifetime_totals(&reader));
            match totals_res {
                Ok(t) => {
                    if tx.send(t).await.is_err() {
                        return;
                    }
                }
                Err(err) => warn!(error = %err, "lifetime totals query failed"),
            }
            next += Duration::from_secs(2);
            tokio::select! {
                biased;
                _ = shutdown_rx.changed() => {
                    if *shutdown_rx.borrow() { return; }
                }
                _ = tokio::time::sleep_until(next) => {}
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Local, Offset, TimeZone};
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    const TAGS_CLOUD_FIXTURE: &str = include_str!("../../fixtures/api-tags-cloud.json");

    fn cloud_models() -> Vec<ModelInfo> {
        #[derive(serde::Deserialize)]
        struct Tags {
            models: Vec<api::TagModel>,
        }
        let tags: Tags = serde_json::from_str(TAGS_CLOUD_FIXTURE).unwrap();
        api::merge_models(tags.models, vec![])
    }

    fn sample_state() -> AppState {
        let mut state = AppState::new(
            1,
            "http://127.0.0.1:11434".into(),
            "0.0.0.0:11435".into(),
            crate::pricing::load(None),
        );
        state.hardware = HardwareSnapshot {
            system_cpu_percent: 23.4,
            system_mem_used_bytes: 38_200_000_000,
            system_mem_total_bytes: 128_000_000_000,
            system_mem_available_bytes: 89_800_000_000,
            ollama_process_count: 2,
            ollama_cpu_percent: 12.3,
            ollama_rss_bytes: 103_000_000,
            gpu_active_residency_pct: Some(38.5),
            ane_power_mw: Some(234.0),
        };
        state
    }

    fn reachable_state() -> AppState {
        let mut state = sample_state();
        state.ingest_models(ModelsSnapshot::Loaded {
            version: Some("0.35.0".into()),
            models: cloud_models(),
        });
        state
    }

    fn record(model: &str, envelope: &str) -> InferenceRecord {
        InferenceRecord {
            session_id: 1,
            model_id: model.into(),
            prompt_tokens: 64_000,
            gen_tokens: 1_200,
            tokens_per_sec: 213.3,
            ttft_sec: 0.5,
            total_time_sec: 6.1,
            stop_reason: "stop".into(),
            completed_at: Utc::now(),
            envelope: envelope.into(),
        }
    }

    fn render_to_lines(width: u16, height: u16, state: &AppState) -> Vec<String> {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| render(f, state)).unwrap();
        let buf = terminal.backend().buffer();
        (0..buf.area.height)
            .map(|y| {
                (0..buf.area.width)
                    .map(|x| buf[(x, y)].symbol())
                    .collect::<String>()
            })
            .collect()
    }

    fn line_with<'a>(lines: &'a [String], needle: &str) -> &'a str {
        lines
            .iter()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("no line contains {needle:?}:\n{}", lines.join("\n")))
    }

    /// The layout's fixed panels sum to 29 rows and the feed needs 7, so 36 rows is the
    /// shortest terminal that fits everything. Same guard as LMS-Monitor's test.
    #[test]
    fn hardware_row_survives_at_minimum_height() {
        let state = reachable_state();
        let lines = render_to_lines(120, 36, &state);
        for title in [
            "status",
            "loaded models",
            "hardware",
            "live request feed",
            "rolling metrics",
            "hypothetical session cost",
        ] {
            assert!(lines.iter().any(|l| l.contains(title)), "panel {title:?} missing");
        }
        let hw_row = line_with(&lines, "ollama cpu");
        assert!(hw_row.contains("ane  234 mW"), "hardware row clipped: {hw_row}");
        assert!(lines.last().unwrap().contains("q quit"), "footer missing");
    }

    #[test]
    fn header_shows_version_and_proxy_in_border() {
        let lines = render_to_lines(120, 36, &reachable_state());
        assert!(lines[0].contains("status"), "{}", lines[0]);
        assert!(
            lines[0].contains("ollama v0.35.0 · proxy 0.0.0.0:11435"),
            "{}",
            lines[0]
        );
        assert!(lines[1].contains("server: ● reachable (http://127.0.0.1:11434)"), "{}", lines[1]);
    }

    #[test]
    fn cloud_model_row_has_recent_marker() {
        let mut state = reachable_state();
        let lines = render_to_lines(120, 36, &state);
        let row = line_with(&lines, "deepseek-v4-pro:cloud");
        for cell in ["llm", "cloud", "FP8", "1048576"] {
            assert!(row.contains(cell), "missing {cell:?}: {row}");
        }
        assert!(!lines.iter().any(|l| l.contains('▸')), "no marker before any inference");

        // OpenAI-compat responses name the model without the `:cloud` tag.
        state.ingest_record(record("deepseek-v4-pro", "openai-sse"));
        let lines = render_to_lines(120, 36, &state);
        line_with(&lines, "▸ deepseek-v4-pro:cloud");
    }

    #[test]
    fn approx_record_renders_tilde() {
        let mut state = reachable_state();
        state.ingest_record(record("qwen3:14b", "openai-sse"));
        let mut approx = record("deepseek-v4-pro", "openai-sse-approx");
        approx.prompt_tokens = 0;
        approx.gen_tokens = 37;
        approx.tokens_per_sec = 12.5;
        state.ingest_record(approx);
        let lines = render_to_lines(120, 36, &state);
        let row = line_with(&lines, "~37");
        assert!(row.contains("~12.5"), "{row}");
        let exact = line_with(&lines, "qwen3:14b");
        assert!(!exact.contains('~'), "exact rows carry no marker: {exact}");
    }

    #[test]
    fn rolling_panel_shows_mean_ttft() {
        let mut state = reachable_state();
        state.ingest_record(record("deepseek-v4-pro", "openai-sse"));
        let lines = render_to_lines(120, 36, &state);
        let row = line_with(&lines, "mean TTFT");
        assert!(row.contains("500ms"), "{row}");
        line_with(&lines, "p95 tok/s");
    }

    #[test]
    fn cost_panel_is_transposed() {
        let mut state = reachable_state();
        let mut rec = record("deepseek-v4-pro", "openai-sse");
        rec.prompt_tokens = 1_000_000;
        rec.gen_tokens = 1_000_000;
        state.ingest_record(rec);
        let lines = render_to_lines(120, 36, &state);
        let header = line_with(&lines, "claude-fable-5");
        assert!(header.contains("claude-opus-4-8") && header.contains("gemini-3-1-pro"), "{header}");
        let total = line_with(&lines, "total USD");
        for usd in ["$60.0000", "$30.0000", "$11.2500"] {
            assert!(total.contains(usd), "missing {usd}: {total}");
        }
        line_with(&lines, "(prompt=1000000 tok / gen=1000000 tok)");
    }

    #[test]
    fn paused_visible_at_120_cols() {
        let mut state = reachable_state();
        state.lifetime = LifetimeTotals {
            session_count: 3,
            total_requests: 916,
            total_prompt_tokens: 58_926_675,
            total_gen_tokens: 1_129_014,
        };
        state.toggle_pause();
        let lines = render_to_lines(120, 36, &state);
        line_with(&lines, "[PAUSED]");

        // Paused: records still persist (run loop) but the UI doesn't ingest them.
        state.ingest_record(record("deepseek-v4-pro", "openai-sse"));
        assert!(state.feed.is_empty());
        assert!(state.last_inference_model_id.is_none());
    }

    #[test]
    fn header_unknown_then_unreachable_keeps_last_models() {
        let mut state = sample_state();
        let lines = render_to_lines(160, 36, &state);
        line_with(&lines, "● unknown");

        state.ingest_models(ModelsSnapshot::Loaded {
            version: Some("0.35.0".into()),
            models: cloud_models(),
        });
        state.ingest_models(ModelsSnapshot::Unreachable {
            reason: "/api/version: Connection refused (os error 61)".into(),
        });
        let lines = render_to_lines(160, 36, &state);
        let header = line_with(&lines, "● unreachable");
        assert!(header.contains("err: /api/version: Connection refused"), "{header}");
        // The last-known list and version survive a blip.
        line_with(&lines, "deepseek-v4-pro:cloud");
        assert!(lines[0].contains("v0.35.0"), "{}", lines[0]);
    }

    #[test]
    fn feed_and_clock_render_local_time() {
        let mut state = reachable_state();
        state.now = Utc.with_ymd_and_hms(2026, 1, 2, 3, 4, 5).unwrap();
        let mut rec = record("deepseek-v4-pro", "openai-sse");
        rec.completed_at = Utc.with_ymd_and_hms(2026, 1, 2, 3, 0, 9).unwrap();
        state.ingest_record(rec);
        // Wide enough that the header clock isn't clipped.
        let lines = render_to_lines(200, 36, &state);

        let local_clock = state.now.with_timezone(&Local).format("%Y-%m-%d %H:%M:%S").to_string();
        let local_feed = feed_local(&state.feed[0]);
        line_with(&lines, &local_clock);
        line_with(&lines, &local_feed);

        let offset = state.now.with_timezone(&Local).offset().fix().local_minus_utc();
        if offset != 0 {
            let screen = lines.join("\n");
            assert!(!screen.contains("2026-01-02 03:04:05"), "clock rendered in UTC");
            assert!(!screen.contains("03:00:09"), "feed rendered in UTC");
        }
    }

    fn feed_local(rec: &InferenceRecord) -> String {
        widgets::feed_time(rec, &Local)
    }

    /// Eyeball check: `cargo test screen_snapshot -- --nocapture`.
    #[test]
    fn screen_snapshot() {
        let mut state = reachable_state();
        state.lifetime = LifetimeTotals {
            session_count: 3,
            total_requests: 916,
            total_prompt_tokens: 58_926_675,
            total_gen_tokens: 1_129_014,
        };
        state.ingest_record(record("deepseek-v4-pro", "openai-sse"));
        let mut approx = record("deepseek-v4-pro", "openai-sse-approx");
        approx.prompt_tokens = 0;
        approx.gen_tokens = 37;
        state.ingest_record(approx);
        for (w, h) in [(120, 36), (160, 44)] {
            println!("--- {w}x{h} ---\n{}", render_to_lines(w, h, &state).join("\n"));
        }
    }

    /// Against the real local Ollama (read-only GETs; no proxy, no DB):
    /// `cargo test live_ollama -- --ignored --nocapture`.
    #[tokio::test]
    #[ignore]
    async fn live_ollama_snapshot() {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap();
        let snap = api::fetch_snapshot(&client, "http://127.0.0.1:11434").await;
        let ModelsSnapshot::Loaded { models, version } = &snap else {
            panic!("ollama not reachable: {snap:?}");
        };
        println!("version={version:?}");
        for m in models {
            println!("{m:?}");
            if m.id.ends_with(":cloud") || m.id.ends_with("-cloud") {
                assert_eq!(m.state, "cloud", "{m:?}");
            }
        }
        let mut state = sample_state();
        state.ingest_models(snap);
        println!("{}", render_to_lines(160, 44, &state).join("\n"));
    }
}
