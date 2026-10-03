use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Row, Table, Wrap};

use super::AppState;
use crate::api::{LoadedModel, ModelsSnapshot};
use crate::pricing::FRONTIER_MODELS;

pub fn render_header(frame: &mut Frame, area: Rect, state: &AppState) {
    let (status_span, version_str) = match &state.models {
        Some(ModelsSnapshot::Loaded { version, .. }) => (
            Span::styled(
                "● ready",
                Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            ),
            version.clone().unwrap_or_else(|| "?".into()),
        ),
        Some(ModelsSnapshot::Unreachable { .. }) | None => (
            Span::styled(
                "● unreachable",
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            ),
            "?".into(),
        ),
    };

    let proxy_chip = Span::styled(
        format!(" proxy {} ", state.proxy_listen),
        Style::default().bg(Color::DarkGray).fg(Color::White),
    );
    let session_chip = Span::styled(
        format!(" session reqs {} ", state.session_request_count),
        Style::default().bg(Color::DarkGray).fg(Color::White),
    );
    let lifetime = Span::raw(format!(
        " lifetime: {} req · {} prompt tok · {} gen tok ",
        state.lifetime.total_requests,
        state.lifetime.total_prompt_tokens,
        state.lifetime.total_gen_tokens
    ));
    let pause = if state.paused {
        Span::styled(" PAUSED ", Style::default().bg(Color::Yellow).fg(Color::Black))
    } else {
        Span::raw("")
    };
    let clock = Span::raw(state.now.format("%H:%M:%S").to_string());

    let title = Line::from(vec![
        Span::raw("ollama "),
        status_span,
        Span::raw(format!(" (v{}) ", version_str)),
        proxy_chip,
        Span::raw(" "),
        session_chip,
        Span::raw(" "),
        lifetime,
        pause,
    ]);
    let block = Block::default().borders(Borders::ALL).title(title);
    let p = Paragraph::new(Line::from(clock))
        .block(block)
        .alignment(Alignment::Right);
    frame.render_widget(p, area);
}

pub fn render_loaded_models(frame: &mut Frame, area: Rect, state: &AppState) {
    let header = Row::new(vec!["id", "family", "params", "quant", "ctx", "vram", "expires"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = match &state.models {
        Some(ModelsSnapshot::Loaded { models, .. }) => models
            .iter()
            .map(|m| {
                let family = m
                    .details
                    .as_ref()
                    .and_then(|d| d.family.clone())
                    .unwrap_or_else(|| "?".into());
                let params = m
                    .details
                    .as_ref()
                    .and_then(|d| d.parameter_size.clone())
                    .unwrap_or_else(|| "?".into());
                let quant = m
                    .details
                    .as_ref()
                    .and_then(|d| d.quantization_level.clone())
                    .unwrap_or_else(|| "?".into());
                let ctx = m
                    .context_length
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "-".into());
                let vram = format_bytes(m.size_vram);
                let expires = m.expires_at.clone().unwrap_or_else(|| "-".into());
                let expires = humanize_expires(&expires);
                Row::new(vec![m.name.clone(), family, params, quant, ctx, vram, expires])
            })
            .collect(),
        _ => vec![Row::new(vec!["(no loaded models or upstream unreachable)"])],
    };

    let widths = [
        Constraint::Length(28),
        Constraint::Length(10),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(20),
    ];
    let table = Table::new(rows, widths).header(header).block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" loaded models ({}) ", count_models(&state.models))),
    );
    frame.render_widget(table, area);
}

fn count_models(snap: &Option<ModelsSnapshot>) -> usize {
    match snap {
        Some(ModelsSnapshot::Loaded { models, .. }) => models.len(),
        _ => 0,
    }
}

#[allow(dead_code)]
fn first_loaded(snap: &Option<ModelsSnapshot>) -> Option<&LoadedModel> {
    if let Some(ModelsSnapshot::Loaded { models, .. }) = snap {
        models.first()
    } else {
        None
    }
}

pub fn render_hardware(frame: &mut Frame, area: Rect, state: &AppState) {
    let h = &state.hardware;
    let mem_pct = if h.system_mem_total_bytes > 0 {
        100.0 * (h.system_mem_used_bytes as f32) / (h.system_mem_total_bytes as f32)
    } else {
        0.0
    };
    let line1 = Line::from(format!(
        "system: cpu {:>5.1}% · mem {:>5.1}% ({} / {})",
        h.system_cpu_percent,
        mem_pct,
        format_bytes(h.system_mem_used_bytes),
        format_bytes(h.system_mem_total_bytes)
    ));
    let line2 = Line::from(format!(
        "ollama: procs {:>2} · cpu {:>6.1}% · rss {} · gpu {} · ane {}",
        h.ollama_process_count,
        h.ollama_cpu_percent,
        format_bytes(h.ollama_rss_bytes),
        h.gpu_active_residency_pct
            .map(|v| format!("{:>5.1}%", v))
            .unwrap_or_else(|| "  n/a".into()),
        h.ane_power_mw
            .map(|v| format!("{:>5.0} mW", v))
            .unwrap_or_else(|| " n/a".into()),
    ));
    let p = Paragraph::new(vec![line1, line2])
        .block(Block::default().borders(Borders::ALL).title(" hardware "));
    frame.render_widget(p, area);
}

pub fn render_feed(frame: &mut Frame, area: Rect, state: &AppState) {
    let header = Row::new(vec!["time", "model", "prompt", "gen", "tok/s", "ttft", "stop"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let rows: Vec<Row> = state
        .feed
        .iter()
        .map(|r| {
            // Cloud-proxy fallback (ollama/ollama#15169): no `usage` block arrived,
            // so the gen/tok-s numbers are chunk-count estimates, not real token
            // counts. Mark the row with leading `~` so it's obvious at a glance.
            let approx = r.envelope == "openai-sse-approx";
            let prefix = if approx { "~" } else { "" };
            Row::new(vec![
                r.completed_at.format("%H:%M:%S").to_string(),
                truncate(&r.model_id, 28),
                r.prompt_tokens.to_string(),
                format!("{}{}", prefix, r.gen_tokens),
                format!("{}{:.1}", prefix, r.tokens_per_sec),
                format!("{:.2}s", r.ttft_sec),
                r.stop_reason.clone(),
            ])
        })
        .collect();

    let widths = [
        Constraint::Length(8),
        Constraint::Length(28),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(8),
        Constraint::Length(7),
        Constraint::Length(10),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" live feed (newest first) "));
    frame.render_widget(table, area);
}

pub fn render_rolling(frame: &mut Frame, area: Rect, state: &AppState) {
    let snap = state.aggregator.snapshot(state.now);
    let header = Row::new(vec!["", "1m", "5m", "15m", "session"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let mk_row = |label: &str, f: fn(&crate::aggregate::WindowStats) -> String| {
        Row::new(vec![
            label.to_string(),
            f(&snap.one_minute),
            f(&snap.five_minute),
            f(&snap.fifteen_minute),
            f(&snap.session),
        ])
    };

    let rows = vec![
        mk_row("requests", |w| w.request_count.to_string()),
        mk_row("prompt tok", |w| w.prompt_tokens.to_string()),
        mk_row("gen tok", |w| w.gen_tokens.to_string()),
        mk_row("mean tok/s", |w| format!("{:.1}", w.mean_tps)),
        mk_row("p50 tok/s", |w| format!("{:.1}", w.p50_tps)),
        mk_row("p95 tok/s", |w| format!("{:.1}", w.p95_tps)),
        mk_row("mean ttft", |w| format!("{:.2}s", w.mean_ttft_sec)),
    ];

    let widths = [
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(12),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" rolling metrics "));
    frame.render_widget(table, area);
}

pub fn render_costs(frame: &mut Frame, area: Rect, state: &AppState) {
    let header = Row::new(vec!["frontier model", "input USD", "output USD", "total USD"])
        .style(Style::default().add_modifier(Modifier::BOLD));

    let snap = state.aggregator.snapshot(state.now);
    let prompt_total = snap.session.prompt_tokens;
    let gen_total = snap.session.gen_tokens;

    let rows: Vec<Row> = FRONTIER_MODELS
        .iter()
        .map(|m| {
            let pricing = state.pricing.get(m);
            match pricing {
                Some(p) => {
                    let input = (prompt_total as f64) / 1_000_000.0 * p.input_per_mtok_usd;
                    let output = (gen_total as f64) / 1_000_000.0 * p.output_per_mtok_usd;
                    let total = input + output;
                    Row::new(vec![
                        m.to_string(),
                        format!("{:>9.4}", input),
                        format!("{:>9.4}", output),
                        format!("{:>9.4}", total),
                    ])
                }
                None => Row::new(vec![
                    m.to_string(),
                    "n/a".into(),
                    "n/a".into(),
                    "n/a".into(),
                ]),
            }
        })
        .collect();

    let widths = [
        Constraint::Length(22),
        Constraint::Length(12),
        Constraint::Length(12),
        Constraint::Length(12),
    ];
    let table = Table::new(rows, widths)
        .header(header)
        .block(Block::default().borders(Borders::ALL).title(" hypothetical cost (session) "));
    frame.render_widget(table, area);
}

pub fn render_footer(frame: &mut Frame, area: Rect, _state: &AppState) {
    let line = Line::from(vec![
        Span::styled("q", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" quit · "),
        Span::styled("r", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" reset session counters · "),
        Span::styled("p", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw(" pause"),
    ]);
    let p = Paragraph::new(line).wrap(Wrap { trim: true });
    frame.render_widget(p, area);
}

// ---------- helpers ----------

fn format_bytes(b: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let f = b as f64;
    if f >= GIB {
        format!("{:.1} GB", f / GIB)
    } else if f >= MIB {
        format!("{:.1} MB", f / MIB)
    } else if f >= KIB {
        format!("{:.1} KB", f / KIB)
    } else {
        format!("{} B", b)
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
        out.push('…');
        out
    }
}

fn humanize_expires(s: &str) -> String {
    // Best-effort: render the time portion if it parses, else show as-is.
    match chrono::DateTime::parse_from_rfc3339(s) {
        Ok(dt) => dt.with_timezone(&chrono::Local).format("%H:%M:%S").to_string(),
        Err(_) => s.to_string(),
    }
}

