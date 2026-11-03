//! The screen shown while a job runs, and after it ends.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, TryRecvError};
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Gauge, Paragraph};
use whispry_core::cancel::CancelToken;
use whispry_core::config::{DeviceChoice, JobConfig};
use whispry_core::events::{Event, Stage};
use whispry_core::timestamps::format_timestamp;
use whispry_core::transcriber::{JobError, JobSummary};

use super::services::JobMessage;

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;
const OK: Color = Color::Green;
const ERROR: Color = Color::Red;
const ACTIVITY_LINES: usize = 5;
/// Messages handled per redraw, so a burst never freezes the screen.
const MESSAGES_PER_TICK: usize = 500;
/// Speed and time left are not shown until the job has run this long.
const SPEED_AFTER: Duration = Duration::from_secs(3);

#[derive(Clone, Debug, PartialEq)]
pub enum RunStatus {
    Running,
    /// Cancel was requested; waiting for the job to stop.
    Cancelling,
    Finished {
        output: PathBuf,
        audio_duration: f64,
    },
    Cancelled {
        output: Option<PathBuf>,
    },
    Failed {
        message: String,
    },
}

impl RunStatus {
    pub fn is_over(&self) -> bool {
        !matches!(self, RunStatus::Running | RunStatus::Cancelling)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    Cancel,
    RetryOnCpu,
    New,
    Quit,
}

impl Button {
    fn label(self) -> &'static str {
        match self {
            Button::Cancel => "Cancel",
            Button::RetryOnCpu => "Retry on CPU",
            Button::New => "New transcription",
            Button::Quit => "Quit",
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum RunAction {
    None,
    /// Go back to the setup screen.
    New,
    Quit,
    /// Run the same job again with this config.
    Retry(JobConfig),
}

#[derive(Clone, Debug, PartialEq)]
struct ProgressInfo {
    stage: Stage,
    completed: f64,
    total: Option<f64>,
    unit: String,
}

pub struct RunState {
    pub config: JobConfig,
    pub status: RunStatus,
    cancel: CancelToken,
    receiver: Receiver<JobMessage>,
    stage_message: String,
    progress: Option<ProgressInfo>,
    language: Option<(String, f64)>,
    transcript: Vec<String>,
    activity: Vec<String>,
    started: Instant,
    transcribing_since: Option<Instant>,
    ended: Option<Instant>,
    /// Quit as soon as the job has stopped.
    quit_when_stopped: bool,
    focus: usize,
}

/// Break text into lines of at most `width` characters, at spaces if possible.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split(' ') {
        let needed = line.chars().count() + usize::from(!line.is_empty()) + word.chars().count();
        if needed > width && !line.is_empty() {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
        while line.chars().count() > width {
            let rest: String = line.chars().skip(width).collect();
            lines.push(line.chars().take(width).collect());
            line = rest;
        }
    }
    lines.push(line);
    lines
}

/// Cut `text` to `width` characters, ending in "…" if it was shortened.
pub fn truncate(text: &str, width: usize) -> String {
    if text.chars().count() <= width {
        return text.to_string();
    }
    let kept: String = text.chars().take(width.saturating_sub(1)).collect();
    format!("{kept}…")
}

fn format_bytes(bytes: f64) -> String {
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut value = bytes;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{value:.0} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

impl RunState {
    pub fn new(
        config: JobConfig,
        cancel: CancelToken,
        receiver: Receiver<JobMessage>,
        now: Instant,
    ) -> Self {
        RunState {
            config,
            status: RunStatus::Running,
            cancel,
            receiver,
            stage_message: "Starting...".into(),
            progress: None,
            language: None,
            transcript: Vec::new(),
            activity: Vec::new(),
            started: now,
            transcribing_since: None,
            ended: None,
            quit_when_stopped: false,
            focus: 0,
        }
    }

    pub fn quit_when_stopped(&self) -> bool {
        self.quit_when_stopped
    }

    fn log(&mut self, line: impl Into<String>) {
        self.activity.push(line.into());
    }

    /// Take in everything the job has reported so far.
    pub fn poll(&mut self, now: Instant) {
        for _ in 0..MESSAGES_PER_TICK {
            match self.receiver.try_recv() {
                Ok(JobMessage::Event(event)) => self.apply(&event, now),
                Ok(JobMessage::Done(result)) => self.finish(result, now),
                Err(TryRecvError::Empty) => return,
                Err(TryRecvError::Disconnected) => {
                    if !self.status.is_over() {
                        self.finish(
                            Err(JobError::Worker(whispry_core::worker::WorkerError::Exited(
                                String::new(),
                            ))),
                            now,
                        );
                    }
                    return;
                }
            }
        }
    }

    pub fn apply(&mut self, event: &Event, now: Instant) {
        match event {
            Event::StageStarted { stage, message } => {
                self.stage_message = message.trim().to_string();
                self.log(self.stage_message.clone());
                self.progress = None;
                if *stage == Stage::Transcribe {
                    self.transcribing_since = Some(now);
                }
            }
            Event::StageFinished { elapsed, .. } => self.log(format!("Done in {elapsed:.1}s")),
            Event::Progress {
                stage,
                completed,
                total,
                unit,
            } => {
                self.progress = Some(ProgressInfo {
                    stage: *stage,
                    completed: *completed,
                    total: *total,
                    unit: unit.clone(),
                });
            }
            Event::LanguageDetected {
                language,
                probability,
            } => {
                self.language = Some((language.clone(), *probability));
                self.log(format!(
                    "Language: {language} ({:.0}%)",
                    probability * 100.0
                ));
            }
            Event::SegmentReady { start, text, .. } => {
                self.transcript
                    .push(format!("[{}] {text}", format_timestamp(*start)));
            }
            Event::Log(message) => self.log(message.clone()),
            Event::Finished {
                output,
                audio_duration,
                ..
            } => {
                self.status = RunStatus::Finished {
                    output: output.clone(),
                    audio_duration: *audio_duration,
                };
                self.ended.get_or_insert(now);
            }
        }
    }

    fn finish(&mut self, result: Result<JobSummary, JobError>, now: Instant) {
        self.ended.get_or_insert(now);
        match result {
            Ok(summary) => {
                self.status = RunStatus::Finished {
                    output: summary.output,
                    audio_duration: summary.audio_duration,
                };
            }
            Err(JobError::Cancelled(cancelled)) => {
                self.status = RunStatus::Cancelled {
                    output: cancelled.output,
                };
            }
            Err(error) => {
                self.status = RunStatus::Failed {
                    message: error.to_string(),
                };
            }
        }
        self.focus = 0;
    }

    /// Ask the job to stop.
    pub fn request_cancel(&mut self) {
        if self.status == RunStatus::Running {
            self.cancel.cancel();
            self.status = RunStatus::Cancelling;
            self.log("Cancelling...");
        }
    }

    fn retry_config(&self) -> Option<JobConfig> {
        let failed = matches!(self.status, RunStatus::Failed { .. });
        (failed && self.config.device != DeviceChoice::Cpu).then(|| {
            let mut config = self.config.clone();
            config.device = DeviceChoice::Cpu;
            config.compute_type = None;
            config
        })
    }

    fn buttons(&self) -> Vec<Button> {
        match &self.status {
            RunStatus::Running => vec![Button::Cancel],
            RunStatus::Cancelling => vec![],
            RunStatus::Failed { .. } if self.retry_config().is_some() => {
                vec![Button::RetryOnCpu, Button::New, Button::Quit]
            }
            _ => vec![Button::New, Button::Quit],
        }
    }

    fn press(&mut self, button: Button) -> RunAction {
        match button {
            Button::Cancel => {
                self.request_cancel();
                RunAction::None
            }
            Button::RetryOnCpu => self
                .retry_config()
                .map_or(RunAction::None, RunAction::Retry),
            Button::New => RunAction::New,
            Button::Quit => RunAction::Quit,
        }
    }

    pub fn handle_key(&mut self, key: &KeyEvent) -> RunAction {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        if control && matches!(key.code, KeyCode::Char('c' | 'q')) {
            return match self.status {
                RunStatus::Running => {
                    self.quit_when_stopped = true;
                    self.request_cancel();
                    RunAction::None
                }
                RunStatus::Cancelling if self.quit_when_stopped => RunAction::Quit,
                RunStatus::Cancelling => {
                    self.quit_when_stopped = true;
                    RunAction::None
                }
                _ => RunAction::Quit,
            };
        }

        let buttons = self.buttons();
        match key.code {
            KeyCode::Esc if self.status == RunStatus::Running => self.request_cancel(),
            KeyCode::Tab | KeyCode::Right if !buttons.is_empty() => {
                self.focus = (self.focus + 1) % buttons.len();
            }
            KeyCode::BackTab | KeyCode::Left if !buttons.is_empty() => {
                self.focus = (self.focus + buttons.len() - 1) % buttons.len();
            }
            KeyCode::Enter => {
                if let Some(button) = buttons.get(self.focus).copied() {
                    return self.press(button);
                }
            }
            KeyCode::Char(c) if self.status.is_over() => {
                let wanted = match c.to_ascii_lowercase() {
                    'n' => Button::New,
                    'q' => Button::Quit,
                    'r' => Button::RetryOnCpu,
                    _ => return RunAction::None,
                };
                if buttons.contains(&wanted) {
                    return self.press(wanted);
                }
            }
            _ => {}
        }
        RunAction::None
    }

    // ----- numbers shown on screen -----

    /// How far along the current stage is, from 0 to 1, if known.
    pub fn ratio(&self) -> Option<f64> {
        let progress = self.progress.as_ref()?;
        let total = progress.total.filter(|total| *total > 0.0)?;
        Some((progress.completed / total).clamp(0.0, 1.0))
    }

    fn elapsed(&self, now: Instant) -> Duration {
        self.ended
            .unwrap_or(now)
            .saturating_duration_since(self.started)
    }

    /// "Elapsed 00:12 · Audio 05:00 / 10:00 · 3.2x realtime · about 01:30 left"
    pub fn timing_line(&self, now: Instant) -> String {
        let mut parts = vec![format!(
            "Elapsed {}",
            format_timestamp(self.elapsed(now).as_secs_f64())
        )];
        let Some(progress) = &self.progress else {
            return parts.join(" · ");
        };

        if progress.unit == "B" {
            let done = format_bytes(progress.completed);
            parts.push(match progress.total {
                Some(total) => format!("{done} / {}", format_bytes(total)),
                None => done,
            });
            return parts.join(" · ");
        }

        parts.push(match progress.total {
            Some(total) => format!(
                "Audio {} / {}",
                format_timestamp(progress.completed),
                format_timestamp(total)
            ),
            None => format!("Audio {}", format_timestamp(progress.completed)),
        });

        if let Some(since) = self.transcribing_since {
            let spent = self.ended.unwrap_or(now).saturating_duration_since(since);
            if spent >= SPEED_AFTER && progress.completed > 0.0 {
                let speed = progress.completed / spent.as_secs_f64();
                parts.push(format!("{speed:.1}x realtime"));
                if self.status == RunStatus::Running
                    && let Some(total) = progress.total
                {
                    let left = ((total - progress.completed).max(0.0) / speed).round();
                    parts.push(format!("about {} left", format_timestamp(left)));
                }
            }
        }
        parts.join(" · ")
    }

    // ----- drawing -----

    fn summary_line(&self) -> String {
        let name = self.config.audio.file_name().map_or_else(
            || self.config.audio.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        let language = match &self.language {
            Some((code, _)) => code.clone(),
            None => self
                .config
                .language
                .clone()
                .unwrap_or_else(|| "auto-detect".into()),
        };
        format!(
            "{name} · {} model · {language} · device {}",
            self.config.model, self.config.device
        )
    }

    pub fn render(&self, frame: &mut Frame, area: Rect, now: Instant) {
        let width = area.width.min(100);
        let area = Rect::new(
            area.x + (area.width - width) / 2,
            area.y,
            width,
            area.height,
        );
        let result_height = if self.status.is_over() { 4 } else { 0 };
        let [header, progress, transcript, activity, result, buttons] = Layout::vertical([
            Constraint::Length(2),
            Constraint::Length(5),
            Constraint::Min(5),
            Constraint::Length(ACTIVITY_LINES as u16 + 2),
            Constraint::Length(result_height),
            Constraint::Length(1),
        ])
        .areas(area);

        let (title, color) = match &self.status {
            RunStatus::Running => ("Transcribing", ACCENT),
            RunStatus::Cancelling => ("Cancelling", Color::Yellow),
            RunStatus::Finished { .. } => ("Finished", OK),
            RunStatus::Cancelled { .. } => ("Cancelled", Color::Yellow),
            RunStatus::Failed { .. } => ("Failed", ERROR),
        };
        frame.render_widget(
            Paragraph::new(vec![
                Line::from(vec![
                    Span::styled(
                        "Whispry",
                        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw("  "),
                    Span::styled(title, Style::new().fg(color).add_modifier(Modifier::BOLD)),
                ]),
                Line::from(Span::styled(self.summary_line(), Style::new().fg(DIM))),
            ]),
            header,
        );

        self.render_progress(frame, progress, now);
        self.render_log(frame, transcript, " Transcript ", &self.transcript, true);
        let start = self.activity.len().saturating_sub(ACTIVITY_LINES);
        self.render_log(
            frame,
            activity,
            " Activity ",
            &self.activity[start..],
            false,
        );
        self.render_result(frame, result);
        self.render_buttons(frame, buttons);
    }

    fn render_progress(&self, frame: &mut Frame, area: Rect, now: Instant) {
        let block = Block::bordered()
            .border_style(Style::new().fg(DIM))
            .title(" Progress ");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let [stage, bar, timing] = Layout::vertical([Constraint::Length(1); 3]).areas(inner);
        frame.render_widget(Paragraph::new(self.stage_message.clone()), stage);

        let ratio = self.ratio();
        let finished = matches!(self.status, RunStatus::Finished { .. });
        let gauge = Gauge::default()
            .gauge_style(Style::new().fg(if finished { OK } else { ACCENT }))
            .ratio(if finished { 1.0 } else { ratio.unwrap_or(0.0) })
            .label(match ratio {
                _ if finished => "100%".to_string(),
                Some(ratio) => format!("{:.0}%", ratio * 100.0),
                None => "…".to_string(),
            });
        frame.render_widget(gauge, bar);
        frame.render_widget(
            Paragraph::new(Span::styled(self.timing_line(now), Style::new().fg(DIM))),
            timing,
        );
    }

    /// The transcript wraps long lines and follows the newest; the activity
    /// log keeps one line per entry, cut short with "…" if it does not fit.
    fn render_log(
        &self,
        frame: &mut Frame,
        area: Rect,
        title: &str,
        lines: &[String],
        is_transcript: bool,
    ) {
        let block = Block::bordered()
            .border_style(Style::new().fg(DIM))
            .title(title.to_string());
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let width = inner.width as usize;
        let shown_lines: Vec<String> = if is_transcript {
            lines.iter().flat_map(|line| wrap(line, width)).collect()
        } else {
            lines.iter().map(|line| truncate(line, width)).collect()
        };
        let skip = shown_lines.len().saturating_sub(inner.height as usize);
        let style = if is_transcript {
            Style::new()
        } else {
            Style::new().fg(DIM)
        };
        let shown: Vec<Line> = shown_lines[skip..]
            .iter()
            .map(|l| Line::from(Span::styled(l.clone(), style)))
            .collect();
        frame.render_widget(Paragraph::new(shown), inner);
    }

    fn render_result(&self, frame: &mut Frame, area: Rect) {
        let (text, color) = match &self.status {
            RunStatus::Finished {
                output,
                audio_duration,
            } => (
                format!(
                    "Transcribed {} of audio.\nSaved to {}",
                    format_timestamp(*audio_duration),
                    output.display()
                ),
                OK,
            ),
            RunStatus::Cancelled {
                output: Some(output),
            } => (
                format!(
                    "Stopped. Text transcribed so far is saved in\n{}",
                    output.display()
                ),
                Color::Yellow,
            ),
            RunStatus::Cancelled { output: None } => (
                "Stopped before any text was written.".to_string(),
                Color::Yellow,
            ),
            RunStatus::Failed { message } => (message.clone(), ERROR),
            _ => return,
        };
        let lines: Vec<String> = text
            .lines()
            .flat_map(|line| wrap(line, area.width.saturating_sub(2) as usize))
            .collect();
        frame.render_widget(
            Paragraph::new(lines.join("\n"))
                .style(Style::new().fg(color))
                .block(Block::bordered().border_style(Style::new().fg(color))),
            area,
        );
    }

    fn render_buttons(&self, frame: &mut Frame, area: Rect) {
        let buttons = self.buttons();
        if buttons.is_empty() {
            frame.render_widget(
                Paragraph::new(Span::styled(" Stopping, one moment…", Style::new().fg(DIM))),
                area,
            );
            return;
        }
        let mut spans = Vec::new();
        for (index, button) in buttons.iter().enumerate() {
            let style = if index == self.focus {
                Style::new()
                    .fg(Color::Black)
                    .bg(ACCENT)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::new().fg(ACCENT)
            };
            spans.push(Span::styled(format!(" {} ", button.label()), style));
            spans.push(Span::raw("  "));
        }
        if self.status == RunStatus::Running {
            spans.push(Span::styled(
                "Esc cancels · Ctrl+Q cancels and quits",
                Style::new().fg(DIM),
            ));
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), area);
    }
}
