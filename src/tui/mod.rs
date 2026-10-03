//! Top-level TUI entrypoint. Sets up terminal, event loop, and per-tick render.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use crossterm::event::{Event, KeyCode, KeyModifiers};
use ratatui::DefaultTerminal;
use tokio::sync::{mpsc, watch};
use tokio::time::{Instant, interval};
use tracing::{debug, info, warn};

use crate::aggregate::Aggregator;
use crate::api::ModelsSnapshot;
use crate::db::{DbHandle, InferenceRecord, LifetimeTotals};
use crate::hardware::HardwareSnapshot;
use crate::pricing::PricingTable;

mod layout;
mod widgets;

const FEED_CAPACITY: usize = 30;
const TICK: Duration = Duration::from_millis(250);

pub struct AppState {
    #[allow(dead_code)] // surfaced in tracing breadcrumbs; future header chip
    pub session_id: i64,
    pub aggregator: Aggregator,
    pub feed: std::collections::VecDeque<InferenceRecord>,
    pub models: Option<ModelsSnapshot>,
    pub hardware: HardwareSnapshot,
    pub lifetime: LifetimeTotals,
    pub session_request_count: u64,
    pub paused: bool,
    pub proxy_listen: String,
    pub now: DateTime<Utc>,
    pub pricing: Arc<PricingTable>,
}

impl AppState {
    fn new(session_id: i64, proxy_listen: String, pricing: Arc<PricingTable>) -> Self {
        Self {
            session_id,
            aggregator: Aggregator::new(),
            feed: std::collections::VecDeque::with_capacity(FEED_CAPACITY),
            models: None,
            hardware: HardwareSnapshot::default(),
            lifetime: LifetimeTotals::default(),
            session_request_count: 0,
            paused: false,
            proxy_listen,
            now: Utc::now(),
            pricing,
        }
    }

    fn ingest_record(&mut self, record: InferenceRecord) {
        self.session_request_count += 1;
        self.aggregator.ingest(record.clone());
        if self.feed.len() == FEED_CAPACITY {
            self.feed.pop_back();
        }
        self.feed.push_front(record);
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run(
    session_id: i64,
    mut records_rx: mpsc::Receiver<InferenceRecord>,
    mut models_rx: mpsc::Receiver<ModelsSnapshot>,
    mut hw_rx: mpsc::Receiver<HardwareSnapshot>,
    db: DbHandle,
    pricing: Arc<PricingTable>,
    proxy_listen: String,
    shutdown_tx: watch::Sender<bool>,
    mut shutdown_rx: watch::Receiver<bool>,
) -> Result<()> {
    install_panic_hook();
    let mut terminal = ratatui::init();

    let mut state = AppState::new(session_id, proxy_listen, pricing);

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
                if !state.paused {
                    state.ingest_record(record);
                }
                let _ = db.persist(to_persist).await;
            }
            Some(snap) = models_rx.recv() => state.models = Some(snap),
            Some(hw) = hw_rx.recv() => state.hardware = hw,
            Some(totals) = lifetime_rx.recv() => state.lifetime = totals,
            _ = tick.tick() => {
                state.now = Utc::now();
                draw(terminal, state)?;
            }
        }
    }
}

fn handle_input(event: Event, state: &mut AppState, shutdown_tx: &watch::Sender<bool>) -> bool {
    if let Event::Key(key) = event {
        match key.code {
            KeyCode::Char('q') | KeyCode::Char('Q') => {
                let _ = shutdown_tx.send(true);
                return true;
            }
            KeyCode::Char('c') | KeyCode::Char('C')
                if key.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                let _ = shutdown_tx.send(true);
                return true;
            }
            KeyCode::Char('r') | KeyCode::Char('R') => {
                state.aggregator.reset();
                state.feed.clear();
                state.session_request_count = 0;
                debug!("session counters reset (records remain in DB)");
            }
            KeyCode::Char('p') | KeyCode::Char('P') => {
                state.paused = !state.paused;
                debug!(paused = state.paused, "pause toggled");
            }
            _ => {}
        }
    }
    false
}

fn draw(terminal: &mut DefaultTerminal, state: &AppState) -> Result<()> {
    terminal
        .draw(|frame| layout::render(frame, state))
        .context("draw")?;
    Ok(())
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
