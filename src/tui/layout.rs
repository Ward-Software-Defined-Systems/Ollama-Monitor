use ratatui::layout::{Constraint, Direction, Layout, Rect};

pub struct AppLayout {
    pub header: Rect,
    pub models: Rect,
    pub hardware: Rect,
    pub feed: Rect,
    pub rolling: Rect,
    pub costs: Rect,
    pub footer: Rect,
}

pub fn compute(area: Rect) -> AppLayout {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // header
            Constraint::Length(6), // loaded models
            Constraint::Length(3), // hardware (1 content row + border)
            Constraint::Min(7),    // live feed (absorbs slack)
            Constraint::Length(9), // rolling metrics
            Constraint::Length(7), // hypothetical cost
            Constraint::Length(1), // footer
        ])
        .split(area);

    AppLayout {
        header: chunks[0],
        models: chunks[1],
        hardware: chunks[2],
        feed: chunks[3],
        rolling: chunks[4],
        costs: chunks[5],
        footer: chunks[6],
    }
}
