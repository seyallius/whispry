//! Shown when a job is started but FFmpeg is not installed.

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph};
use whispry_core::config::JobConfig;

use super::run::wrap;

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;
const WARN: Color = Color::Yellow;

#[derive(Debug, PartialEq)]
pub enum DialogAction {
    None,
    /// Look for FFmpeg again.
    Check,
    Close,
}

pub struct FFmpegDialog {
    steps: Vec<String>,
    /// The job to start once FFmpeg is found.
    pub config: JobConfig,
    still_missing: bool,
    focus: usize,
}

const BUTTONS: [&str; 2] = ["Check again", "Close"];

impl FFmpegDialog {
    pub fn new(steps: Vec<String>, config: JobConfig) -> Self {
        FFmpegDialog {
            steps,
            config,
            still_missing: false,
            focus: 0,
        }
    }

    /// FFmpeg was looked for again and still is not there.
    pub fn report_still_missing(&mut self) {
        self.still_missing = true;
    }

    pub fn handle_key(&mut self, key: &KeyEvent) -> DialogAction {
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('c' | 'q') => DialogAction::Close,
                _ => DialogAction::None,
            };
        }
        match key.code {
            KeyCode::Esc => DialogAction::Close,
            KeyCode::Tab | KeyCode::Right | KeyCode::Left | KeyCode::BackTab => {
                self.focus = (self.focus + 1) % BUTTONS.len();
                DialogAction::None
            }
            KeyCode::Char('c') => DialogAction::Check,
            KeyCode::Enter if self.focus == 0 => DialogAction::Check,
            KeyCode::Enter => DialogAction::Close,
            _ => DialogAction::None,
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let [popup] = Layout::horizontal([Constraint::Length(area.width.min(84))])
            .flex(Flex::Center)
            .areas(area);
        let inner_width = popup.width.saturating_sub(4) as usize;

        let mut lines = vec![
            Line::from("Whispry uses FFmpeg to read audio and video files, and it"),
            Line::from("was not found on this computer. To install it:"),
            Line::from(""),
        ];
        for step in &self.steps {
            for (index, line) in wrap(step, inner_width.saturating_sub(4))
                .into_iter()
                .enumerate()
            {
                let prefix = if index == 0 { "  • " } else { "    " };
                lines.push(Line::from(format!("{prefix}{line}")));
            }
        }
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "Then choose Check again. Your choices are kept.",
            Style::new().fg(DIM),
        )));
        if self.still_missing {
            lines.push(Line::from(Span::styled(
                "FFmpeg is still not found. Open a new terminal after installing, so PATH is up to date.",
                Style::new().fg(WARN),
            )));
        }
        lines.push(Line::from(""));
        let mut buttons = Vec::new();
        for (index, label) in BUTTONS.iter().enumerate() {
            let style = if index == self.focus {
                Style::new()
                    .fg(Color::Black)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(ACCENT)
            };
            buttons.push(Span::styled(format!(" {label} "), style));
            buttons.push(Span::raw("  "));
        }
        lines.push(Line::from(buttons));

        let height = (lines.len() as u16 + 2).min(area.height);
        let [popup] = Layout::vertical([Constraint::Length(height)])
            .flex(Flex::Center)
            .areas(popup);
        frame.render_widget(Clear, popup);
        frame.render_widget(
            Paragraph::new(lines).block(
                Block::bordered()
                    .border_style(Style::new().fg(WARN))
                    .title(Span::styled(
                        " FFmpeg is needed ",
                        Style::new().fg(WARN).add_modifier(Modifier::BOLD),
                    ))
                    .padding(ratatui::widgets::Padding::horizontal(1)),
            ),
            popup,
        );
    }
}
