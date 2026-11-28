//! The `?` screen: every key and mouse action in one place.

use ratatui::Frame;
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};

use super::theme::{self, Hit};

const SECTIONS: &[(&str, &[(&str, &str)])] = &[
    (
        "Anywhere",
        &[
            ("? or F1", "This help (type ? in a text field? use F1)"),
            ("Ctrl+T", "Next theme (or click the theme name)"),
            ("Ctrl+Q", "Quit (stops a running job first)"),
        ],
    ),
    (
        "Setup",
        &[
            ("Tab / Shift+Tab, ↑ ↓", "Move between fields"),
            ("← →, Space", "Change a choice"),
            ("Ctrl+O", "Browse for a file"),
            ("Ctrl+S, Enter on Start", "Start transcribing"),
        ],
    ),
    (
        "File browser",
        &[
            ("↑ ↓, PgUp PgDn, Home End", "Move"),
            ("Enter, →", "Open folder / choose file"),
            ("Backspace, ←", "Up one folder"),
            ("~   /", "Jump to your home folder / the root"),
            ("a-z", "Jump to a name starting with that letter"),
        ],
    ),
    (
        "While transcribing",
        &[
            ("Esc", "Cancel (text so far is kept)"),
            ("N / R / Q", "New / Retry on CPU / Quit, when finished"),
        ],
    ),
    (
        "Mouse",
        &[
            ("Click", "Any button, field, arrow or folder"),
            ("Wheel", "Scroll the file list"),
            ("Shift+drag", "Select text (Option+drag in iTerm2)"),
        ],
    ),
];

fn lines() -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for (title, rows) in SECTIONS {
        if !out.is_empty() {
            out.push(Line::from(""));
        }
        out.push(Line::from(Span::styled(
            *title,
            Style::new()
                .fg(theme::accent())
                .add_modifier(Modifier::BOLD),
        )));
        for (keys, what) in *rows {
            out.push(Line::from(vec![
                Span::styled(format!("  {keys:<26}"), Style::new().fg(theme::accent2())),
                Span::raw(*what),
            ]));
        }
    }
    out
}

/// Draw the help over everything, scrolled down by `scroll` lines.
pub fn render(frame: &mut Frame, area: Rect, scroll: u16) {
    theme::add(area, Hit::HelpClose);
    let all = lines();
    let [popup] = Layout::horizontal([Constraint::Length(area.width.min(84))])
        .flex(Flex::Center)
        .areas(area);
    let height = (all.len() as u16 + 2).min(area.height);
    let [popup] = Layout::vertical([Constraint::Length(height)])
        .flex(Flex::Center)
        .areas(popup);
    theme::clear(frame, popup);
    let block = Block::bordered()
        .border_type(theme::BORDER)
        .border_style(Style::new().fg(theme::accent()))
        .title(Span::styled(
            " Help ",
            Style::new()
                .fg(theme::accent())
                .add_modifier(Modifier::BOLD),
        ))
        .title_bottom(Span::styled(
            " Esc or click to close · ↑ ↓ scroll ",
            theme::dim_style(),
        ));
    let inner = block.inner(popup);
    frame.render_widget(block, popup);
    let max = all.len().saturating_sub(inner.height as usize) as u16;
    frame.render_widget(Paragraph::new(all).scroll((scroll.min(max), 0)), inner);
}
