use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout};

use super::AppState;
use super::widgets;

pub fn render(frame: &mut Frame, state: &AppState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // header
            Constraint::Length(6),  // loaded models
            Constraint::Length(4),  // hardware
            Constraint::Min(7),     // live feed (grows on tall terms)
            Constraint::Length(9),  // rolling metrics
            Constraint::Length(7),  // hypothetical cost
            Constraint::Length(1),  // footer
        ])
        .split(area);

    widgets::render_header(frame, chunks[0], state);
    widgets::render_loaded_models(frame, chunks[1], state);
    widgets::render_hardware(frame, chunks[2], state);
    widgets::render_feed(frame, chunks[3], state);
    widgets::render_rolling(frame, chunks[4], state);
    widgets::render_costs(frame, chunks[5], state);
    widgets::render_footer(frame, chunks[6], state);
}
