//! Per-panel render fns. Ported from LMS-Monitor's `tui/widgets.rs` so the two TUIs stay
//! visually identical; Ollama-only additions are the header's version/proxy title, the
//! `cloud` model state, and `~` on approximate feed rows.

use std::collections::VecDeque;

use chrono::{DateTime, TimeZone, Utc};
use ratatui::Frame;
use ratatui::layout::{Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};

use super::FEED_CAPACITY;
use crate::aggregate::{AggregateSnapshot, WindowStats};
use crate::api::ModelInfo;
use crate::db::{InferenceRecord, LifetimeTotals};
use crate::hardware::{HardwareSnapshot, format_bytes, format_bytes_ratio};
use crate::pricing::{FRONTIER_MODELS, HypotheticalCost, PricingTable, hypothetical_cost};

/// Envelope of OpenAI-compat streams that ended without a `usage` frame (the `:cloud`
/// relay drops it, ollama/ollama#15169); their gen / tok/s numbers are estimates.
const APPROX_ENVELOPE: &str = "openai-sse-approx";

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ServerStatus {
    Unknown,
    Reachable,
    Unreachable,
}

pub struct HeaderInfo<'a> {
    pub server_status: ServerStatus,
    pub server_error: Option<&'a str>,
    pub base_url: &'a str,
    pub version: Option<&'a str>,
    pub proxy_listen: &'a str,
    pub paused: bool,
    pub lifetime: &'a LifetimeTotals,
    pub now: DateTime<Utc>,
}

pub fn render_header(f: &mut Frame, area: Rect, info: HeaderInfo<'_>) {
    let now = info
        .now
        .with_timezone(&chrono::Local)
        .format("%Y-%m-%d %H:%M:%S")
        .to_string();
    let (status_label, status_color) = match info.server_status {
        ServerStatus::Reachable => ("reachable", Color::Green),
        ServerStatus::Unreachable => ("unreachable", Color::Red),
        ServerStatus::Unknown => ("unknown", Color::Yellow),
    };
    let mut spans = vec![
        Span::styled(
            "ollama-monitor",
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw("   "),
        Span::raw("server: "),
        Span::styled(
            format!("● {status_label}"),
            Style::default().fg(status_color),
        ),
        Span::raw(format!(" ({})", info.base_url)),
    ];
    if let Some(err) = info.server_error {
        spans.push(Span::raw("  err: "));
        spans.push(Span::styled(
            truncate(err, 60),
            Style::default().fg(Color::Red),
        ));
    }
    if info.paused {
        spans.push(Span::raw("   "));
        spans.push(Span::styled(
            "[PAUSED]",
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ));
    }
    spans.push(Span::raw("   lifetime: "));
    spans.push(Span::styled(
        format!(
            "{} reqs / {} sessions / {} prompt tok / {} gen tok",
            info.lifetime.total_requests,
            info.lifetime.session_count,
            info.lifetime.total_prompt_tokens,
            info.lifetime.total_gen_tokens,
        ),
        Style::default().fg(Color::Cyan),
    ));
    spans.push(Span::raw("   "));
    spans.push(Span::styled(now, Style::default().fg(Color::DarkGray)));

    // Ollama-only: upstream version + the proxy address clients must target. They ride in
    // the top border so they cost no columns on the (already long) content row.
    let mut endpoint = String::from(" ollama");
    if let Some(v) = info.version {
        endpoint.push_str(&format!(" v{v}"));
    }
    endpoint.push_str(&format!(" · proxy {} ", info.proxy_listen));

    let para = Paragraph::new(Line::from(spans)).block(
        Block::default()
            .borders(Borders::ALL)
            .title("status")
            .title(Line::from(endpoint).right_aligned()),
    );
    f.render_widget(para, area);
}

pub fn render_models(
    f: &mut Frame,
    area: Rect,
    models: &[ModelInfo],
    recent_model_id: Option<&str>,
) {
    let header = Row::new(vec![
        Cell::from("id"),
        Cell::from("type"),
        Cell::from("compat"),
        Cell::from("quant"),
        Cell::from("ctx"),
        Cell::from("state"),
    ])
    .style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );

    let rows: Vec<Row> = models
        .iter()
        .map(|m| {
            let is_recent = recent_model_id.is_some_and(|r| r == m.id);
            let state_color = match m.state.as_str() {
                "loaded" => Color::Green,
                "not-loaded" => Color::DarkGray,
                // Runs on ollama.com; cyan is taken by the ▸ marker.
                "cloud" => Color::Magenta,
                _ => Color::Yellow,
            };
            let id_span = if is_recent {
                Span::styled(
                    format!("▸ {}", m.id),
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::raw(format!("  {}", m.id))
            };
            Row::new(vec![
                Cell::from(Line::from(id_span)),
                Cell::from(m.kind.as_deref().unwrap_or("-").to_string()),
                Cell::from(m.compatibility_type.as_deref().unwrap_or("-").to_string()),
                Cell::from(m.quantization.as_deref().unwrap_or("-").to_string()),
                Cell::from(
                    m.max_context_length
                        .map_or("-".to_string(), |c| c.to_string()),
                ),
                Cell::from(Span::styled(
                    m.state.clone(),
                    Style::default().fg(state_color),
                )),
            ])
        })
        .collect();

    let widths = [
        Constraint::Min(30),
        Constraint::Length(12),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(10),
        Constraint::Length(12),
    ];
    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .borders(Borders::ALL)
            .title("loaded models"),
    );
    f.render_widget(table, area);
}

/// `feed` is newest-first already.
pub fn render_feed(f: &mut Frame, area: Rect, feed: &VecDeque<InferenceRecord>) {
    let header = Row::new(vec![
        Cell::from("time"),
        Cell::from("model"),
        Cell::from("prompt"),
        Cell::from("gen"),
        Cell::from("TTFT"),
        Cell::from("tok/s"),
        Cell::from("stop"),
    ])
    .style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );

    let rows: Vec<Row> = feed
        .iter()
        .map(|r| {
            // Cloud-proxy fallback (ollama/ollama#15169): no `usage` block arrived, so gen
            // and tok/s are chunk-count estimates. A leading `~` flags them at a glance.
            let approx = if r.envelope == APPROX_ENVELOPE {
                "~"
            } else {
                ""
            };
            let stop = if r.stop_reason.is_empty() {
                "-"
            } else {
                r.stop_reason.as_str()
            };
            Row::new(vec![
                Cell::from(feed_time(r, &chrono::Local)),
                Cell::from(truncate(&r.model_id, 28).to_string()),
                Cell::from(r.prompt_tokens.to_string()),
                Cell::from(format!("{approx}{}", r.gen_tokens)),
                Cell::from(format!("{:.0}ms", r.ttft_sec * 1000.0)),
                Cell::from(format!("{approx}{:.1}", r.tokens_per_sec)),
                Cell::from(stop.to_string()),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(10),
        Constraint::Min(20),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(24),
    ];
    let title = format!("live request feed ({}/{FEED_CAPACITY})", feed.len());
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(table, area);
}

/// Feed timestamp in `tz` (`chrono::Local` on screen). LMS-Monitor shows start time; this
/// shows completion time, which Ollama-Monitor's rolling windows, DB and feed order use.
pub fn feed_time<Tz: TimeZone>(r: &InferenceRecord, tz: &Tz) -> String
where
    Tz::Offset: std::fmt::Display,
{
    r.completed_at
        .with_timezone(tz)
        .format("%H:%M:%S")
        .to_string()
}

pub fn render_rolling(f: &mut Frame, area: Rect, snap: &AggregateSnapshot) {
    let header = Row::new(vec![
        Cell::from(""),
        Cell::from("1m"),
        Cell::from("5m"),
        Cell::from("15m"),
        Cell::from("session"),
    ])
    .style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );

    fn metric_row(
        label: &str,
        f: impl Fn(&WindowStats) -> String,
        snap: &AggregateSnapshot,
    ) -> Row<'static> {
        Row::new(vec![
            Cell::from(Span::styled(
                label.to_string(),
                Style::default().fg(Color::DarkGray),
            )),
            Cell::from(f(&snap.one_minute)),
            Cell::from(f(&snap.five_minute)),
            Cell::from(f(&snap.fifteen_minute)),
            Cell::from(f(&snap.session)),
        ])
    }

    let rows = vec![
        metric_row("requests", |m| m.request_count.to_string(), snap),
        metric_row("prompt tok", |m| m.prompt_tokens.to_string(), snap),
        metric_row("gen tok", |m| m.gen_tokens.to_string(), snap),
        metric_row("mean tok/s", |m| format!("{:.1}", m.mean_tps), snap),
        metric_row("p95 tok/s", |m| format!("{:.1}", m.p95_tps), snap),
        metric_row(
            "mean TTFT",
            |m| format!("{:.0}ms", m.mean_ttft_sec * 1000.0),
            snap,
        ),
    ];

    let widths = [
        Constraint::Length(14),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(12),
    ];
    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .borders(Borders::ALL)
            .title("rolling metrics"),
    );
    f.render_widget(table, area);
}

pub fn render_costs(f: &mut Frame, area: Rect, snap: &AggregateSnapshot, pricing: &PricingTable) {
    let mut header_cells: Vec<Cell> = vec![Cell::from("")];
    for key in FRONTIER_MODELS {
        header_cells.push(Cell::from(*key));
    }
    let header = Row::new(header_cells).style(
        Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::BOLD),
    );

    // Aggregate session totals
    let total_prompt = snap.session.prompt_tokens;
    let total_gen = snap.session.gen_tokens;

    let costs: Vec<_> = FRONTIER_MODELS
        .iter()
        .map(|k| {
            pricing
                .get(k)
                .map(|p| hypothetical_cost(total_prompt, total_gen, &p))
        })
        .collect();

    fn cost_cell(c: &Option<HypotheticalCost>, sel: fn(&HypotheticalCost) -> f64) -> Cell<'static> {
        match c {
            Some(c) => Cell::from(format!("${:.4}", sel(c))),
            None => Cell::from("(no rate)"),
        }
    }

    fn row(
        label: &str,
        costs: &[Option<HypotheticalCost>],
        sel: fn(&HypotheticalCost) -> f64,
    ) -> Row<'static> {
        let mut cells = vec![Cell::from(Span::styled(
            label.to_string(),
            Style::default().fg(Color::DarkGray),
        ))];
        for c in costs {
            cells.push(cost_cell(c, sel));
        }
        Row::new(cells)
    }

    let rows = vec![
        row("input USD", &costs, |c| c.input_usd),
        row("output USD", &costs, |c| c.output_usd),
        row("total USD", &costs, |c| c.total_usd),
    ];

    let mut widths = vec![Constraint::Length(14)];
    for _ in FRONTIER_MODELS {
        widths.push(Constraint::Min(14));
    }
    let title =
        format!("hypothetical session cost  (prompt={total_prompt} tok / gen={total_gen} tok)");
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(table, area);
}

/// One-line hardware summary: system cpu/mem │ Ollama processes │ GPU telemetry, where
/// the tail is `gpu … ane …` on macOS and `gpu … vram …` elsewhere (the Linux backend
/// has no ANE figure). Groups run from most to least important left→right, so a narrow
/// terminal clips the GPU tail before anything else.
fn hardware_line(hw: &HardwareSnapshot) -> Line<'static> {
    let dim = Style::default().fg(Color::DarkGray);
    let sep = || Span::styled(" │ ", dim);

    let mut spans = vec![
        Span::raw("cpu "),
        Span::styled(
            format!("{:>5.1}%", hw.system_cpu_percent),
            cpu_style(hw.system_cpu_percent),
        ),
        Span::raw("  mem "),
        Span::styled(
            format_bytes_ratio(hw.system_mem_used_bytes, hw.system_mem_total_bytes),
            Style::default().fg(Color::Cyan),
        ),
        Span::raw(" ("),
        Span::styled(
            format!("{} free", format_bytes(hw.system_mem_available_bytes)),
            Style::default().fg(Color::Green),
        ),
        Span::raw(")"),
        sep(),
        Span::styled("ollama ", dim.add_modifier(Modifier::BOLD)),
    ];

    if hw.ollama_process_count == 0 {
        spans.push(Span::styled("no process detected", dim));
    } else {
        spans.extend([
            Span::raw("cpu "),
            Span::styled(
                format!("{:>5.1}%", hw.ollama_cpu_percent),
                cpu_style(hw.ollama_cpu_percent),
            ),
            Span::raw("  rss "),
            Span::styled(
                format_bytes(hw.ollama_rss_bytes),
                Style::default().fg(Color::Cyan),
            ),
            Span::raw(format!(
                "  {} proc{}",
                hw.ollama_process_count,
                if hw.ollama_process_count == 1 {
                    ""
                } else {
                    "s"
                }
            )),
        ]);
    }

    spans.push(sep());
    spans.push(Span::raw("gpu "));
    spans.push(match hw.gpu_util_pct {
        Some(pct) => Span::styled(format!("{pct:>5.1}%"), cpu_style(pct)),
        None => Span::styled("  n/a", dim),
    });
    // `cfg!` rather than `#[cfg]` so both arms type-check on every platform.
    if cfg!(target_os = "macos") {
        spans.push(Span::raw("  ane "));
        spans.push(match hw.ane_power_mw {
            Some(mw) => Span::styled(
                format!("{mw:>4.0} mW"),
                Style::default().fg(if mw > 100.0 {
                    Color::Yellow
                } else {
                    Color::Green
                }),
            ),
            None => Span::styled(" n/a", dim),
        });
    } else {
        spans.push(Span::raw("  vram "));
        spans.push(match hw.gpu_mem_used_bytes {
            Some(bytes) => Span::styled(format_bytes(bytes), Style::default().fg(Color::Cyan)),
            None => Span::styled(" n/a", dim),
        });
    }

    Line::from(spans)
}

pub fn render_hardware(f: &mut Frame, area: Rect, hw: &HardwareSnapshot) {
    let para = Paragraph::new(hardware_line(hw))
        .block(Block::default().borders(Borders::ALL).title("hardware"));
    f.render_widget(para, area);
}

fn cpu_style(pct: f32) -> Style {
    let color = if pct >= 80.0 {
        Color::Red
    } else if pct >= 50.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    Style::default().fg(color)
}

pub fn render_footer(f: &mut Frame, area: Rect) {
    let line = Line::from(vec![
        Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" quit  ·  "),
        Span::styled("r", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" reset session  ·  "),
        Span::styled("p", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" pause"),
    ]);
    let para = Paragraph::new(line).style(Style::default().fg(Color::DarkGray));
    f.render_widget(para, area);
}

fn truncate(s: &str, max: usize) -> &str {
    if s.len() <= max {
        s
    } else {
        // Safe-byte truncation
        let mut end = max;
        while !s.is_char_boundary(end) && end > 0 {
            end -= 1;
        }
        &s[..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::FixedOffset;

    #[test]
    fn truncate_handles_unicode() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello", 3), "hel");
        // multi-byte: truncating inside a char must back up to a valid boundary
        let s = "héllo";
        let t = truncate(s, 2);
        assert!(s.starts_with(t));
    }

    /// Worst-case-ish values: three-digit percentages, three-digit GB, double-digit
    /// proc count, four-digit ANE mW / two-digit GB of VRAM. The whole row must stay one
    /// line and fit the 118 inner columns of a 120-column terminal (typical values land
    /// near 110). Arithmetic: 92 columns through the second separator, 10 for
    /// `gpu 100.0%`, then 13 for `  ane 1234 mW` (115) or 14 for `  vram 15.9 GB` (116).
    /// A ≥100 GB VRAM sum or a 32-core `ollama cpu 3200.0%` adds one column each, still
    /// within 118.
    #[test]
    fn hardware_line_is_one_compact_row() {
        let hw = HardwareSnapshot {
            system_cpu_percent: 100.0,
            system_mem_used_bytes: 123_400_000_000,
            system_mem_total_bytes: 137_400_000_000,
            system_mem_available_bytes: 101_200_000_000,
            ollama_process_count: 12,
            ollama_cpu_percent: 850.3,
            ollama_rss_bytes: 98_700_000_000,
            gpu_util_pct: Some(100.0),
            gpu_mem_used_bytes: Some(15_900_000_000),
            ane_power_mw: Some(1234.0),
        };
        let line = hardware_line(&hw);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(!text.contains('\n'), "must be a single row: {text:?}");
        assert!(
            line.width() <= 116,
            "hardware row too wide ({} cols): {text}",
            line.width()
        );
        for needle in [
            "cpu 100.0%",
            "mem 123.4/137.4 GB",
            "101.2 GB free",
            "ollama cpu 850.3%",
            "rss 98.7 GB",
            "12 procs",
            "gpu 100.0%",
            if cfg!(target_os = "macos") {
                "ane 1234 mW"
            } else {
                "vram 15.9 GB"
            },
        ] {
            assert!(text.contains(needle), "missing {needle:?} in {text:?}");
        }
    }

    #[test]
    fn hardware_line_collapses_missing_ollama_and_telemetry() {
        let hw = HardwareSnapshot::default();
        let line = hardware_line(&hw);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("ollama no process detected"), "{text:?}");
        assert!(
            !text.contains("rss"),
            "rss should be omitted without a process: {text:?}"
        );
        assert!(text.contains("gpu   n/a"), "{text:?}");
        let telemetry_na = if cfg!(target_os = "macos") {
            "ane  n/a"
        } else {
            "vram  n/a"
        };
        assert!(text.contains(telemetry_na), "{text:?}");
    }

    #[test]
    fn feed_time_uses_given_zone() {
        let rec = InferenceRecord {
            session_id: 1,
            model_id: "m".into(),
            prompt_tokens: 0,
            gen_tokens: 0,
            tokens_per_sec: 0.0,
            ttft_sec: 0.0,
            total_time_sec: 0.0,
            stop_reason: String::new(),
            completed_at: Utc.with_ymd_and_hms(2026, 10, 3, 19, 4, 5).unwrap(),
            envelope: "ollama-stream".into(),
        };
        let pdt = FixedOffset::west_opt(7 * 3600).unwrap();
        assert_eq!(feed_time(&rec, &pdt), "12:04:05");
        assert_eq!(feed_time(&rec, &Utc), "19:04:05");
    }
}
