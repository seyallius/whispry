//! The first screen: choose a file, a model, a language and a device.

use std::path::PathBuf;

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use whispry_core::config::{
    DEFAULT_CHUNK_MINUTES, DeviceChoice, JobConfig, LANGUAGES, MODEL_REPOSITORIES,
};

use super::input::TextInput;
use super::services::Services;

const ACCENT: Color = Color::Cyan;
const DIM: Color = Color::DarkGray;
const ERROR: Color = Color::Red;
const OK: Color = Color::Green;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field {
    File,
    Model,
    Language,
    Device,
    Advanced,
    Compute,
    Chunk,
    Start,
}

#[derive(Debug, PartialEq)]
pub enum SetupAction {
    None,
    Quit,
    /// Open the file browser.
    Browse,
    Start(JobConfig),
}

const DEVICE_CHOICES: [DeviceChoice; 3] =
    [DeviceChoice::Auto, DeviceChoice::Cpu, DeviceChoice::Cuda];

/// One line of the device list.
struct DeviceRow {
    label: String,
    note: Option<String>,
    enabled: bool,
}

pub struct SetupState {
    pub file: TextInput,
    pub model: usize,
    /// 0 is auto-detect; the rest follow `LANGUAGES`.
    pub language: usize,
    pub device: usize,
    /// 0 is "best available"; the rest follow the device's compute types.
    pub compute: usize,
    pub chunk: TextInput,
    pub focus: Field,
    pub advanced_open: bool,
    pub errors: Vec<String>,
}

/// Remove the quotes a terminal adds when a file is dragged in, and expand `~`.
pub fn normalize_path(text: &str) -> PathBuf {
    let text = text.trim();
    let unquoted = ["\"", "'"]
        .iter()
        .find_map(|quote| text.strip_prefix(quote)?.strip_suffix(quote))
        .unwrap_or(text);
    if let Some(rest) = unquoted
        .strip_prefix("~/")
        .or_else(|| unquoted.strip_prefix("~\\"))
        && let Some(home) = std::env::home_dir()
    {
        return home.join(rest);
    }
    PathBuf::from(unquoted)
}

fn cycle(index: usize, length: usize, forward: bool) -> usize {
    if forward {
        (index + 1) % length
    } else {
        (index + length - 1) % length
    }
}

fn capitalize(word: &str) -> String {
    let mut chars = word.chars();
    chars.next().map_or_else(String::new, |first| {
        first.to_uppercase().chain(chars).collect()
    })
}

impl SetupState {
    pub fn new(initial_file: Option<&std::path::Path>) -> Self {
        SetupState {
            file: TextInput::new(
                &initial_file.map_or_else(String::new, |p| p.display().to_string()),
            ),
            model: 0,
            language: 0,
            device: 0,
            compute: 0,
            chunk: TextInput::new(&DEFAULT_CHUNK_MINUTES.to_string()),
            focus: Field::File,
            advanced_open: false,
            errors: Vec::new(),
        }
    }

    pub fn device_choice(&self) -> DeviceChoice {
        DEVICE_CHOICES[self.device]
    }

    fn fields(&self) -> Vec<Field> {
        let mut fields = vec![
            Field::File,
            Field::Model,
            Field::Language,
            Field::Device,
            Field::Advanced,
        ];
        if self.advanced_open {
            fields.extend([Field::Compute, Field::Chunk]);
        }
        fields.push(Field::Start);
        fields
    }

    fn move_focus(&mut self, forward: bool) {
        let fields = self.fields();
        let at = fields.iter().position(|f| *f == self.focus).unwrap_or(0);
        self.focus = fields[cycle(at, fields.len(), forward)];
    }

    fn device_rows(&self, services: &dyn Services) -> Vec<DeviceRow> {
        let system = services.system();
        let best = system
            .available_devices()
            .next()
            .map_or("CPU", |d| &d.label);
        let gpu = system.gpu();
        vec![
            DeviceRow {
                label: "Auto".into(),
                note: Some(format!("uses {best}")),
                enabled: true,
            },
            DeviceRow {
                label: "CPU".into(),
                note: None,
                enabled: true,
            },
            DeviceRow {
                label: gpu.label.clone(),
                note: gpu.reason.clone(),
                enabled: gpu.available,
            },
        ]
    }

    /// Compute types the chosen device offers, best first.
    fn compute_types(&self, services: &dyn Services) -> Vec<&'static str> {
        let system = services.system();
        let option = match self.device_choice() {
            DeviceChoice::Auto => system.available_devices().next(),
            DeviceChoice::Cpu => Some(system.device(whispry_core::platform::Device::Cpu)),
            DeviceChoice::Cuda => Some(system.gpu()).filter(|gpu| gpu.available),
        };
        option.map_or_else(Vec::new, |option| option.compute_types.clone())
    }

    fn change_device(&mut self, services: &dyn Services, forward: bool) {
        let rows = self.device_rows(services);
        let mut next = self.device;
        for _ in 0..rows.len() {
            next = cycle(next, rows.len(), forward);
            if rows[next].enabled {
                break;
            }
        }
        if next != self.device {
            self.device = next;
            self.compute = 0;
        }
    }

    /// Everything entered so far as a job, or what is wrong with it.
    pub fn build_config(&self, services: &dyn Services) -> Result<JobConfig, Vec<String>> {
        let mut problems = Vec::new();
        let mut config = JobConfig::new(normalize_path(self.file.text()));
        config.model = MODEL_REPOSITORIES[self.model].0.to_string();
        config.language = self
            .language
            .checked_sub(1)
            .map(|i| LANGUAGES[i].0.to_string());
        config.device = self.device_choice();
        config.compute_type = self
            .compute
            .checked_sub(1)
            .and_then(|i| self.compute_types(services).get(i).map(|t| t.to_string()));

        match self.chunk.text().trim().parse::<f64>() {
            Ok(minutes) => config.chunk_minutes = minutes,
            Err(_) => problems.push("Chunk length must be a number of minutes".to_string()),
        }
        if self.file.text().trim().is_empty() {
            problems.push("Choose a file to transcribe".to_string());
        } else {
            problems.extend(services.validate(&config));
        }
        if problems.is_empty() {
            Ok(config)
        } else {
            Err(problems)
        }
    }

    fn start(&mut self, services: &dyn Services) -> SetupAction {
        match self.build_config(services) {
            Ok(config) => {
                self.errors.clear();
                SetupAction::Start(config)
            }
            Err(problems) => {
                self.errors = problems;
                SetupAction::None
            }
        }
    }

    /// Where the file browser should open: beside the current file, else at home.
    pub fn browse_start(&self) -> PathBuf {
        let current = normalize_path(self.file.text());
        let beside = if current.is_dir() {
            Some(current)
        } else {
            // A bare file name has an empty parent: that means "here".
            current.parent().map(|parent| {
                if parent.as_os_str().is_empty() {
                    PathBuf::from(".")
                } else {
                    parent.to_path_buf()
                }
            })
        };
        beside
            .filter(|dir| dir.is_dir())
            .or_else(std::env::home_dir)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."))
    }

    pub fn choose_file(&mut self, path: &std::path::Path) {
        self.file.set(&path.display().to_string());
        self.errors.clear();
    }

    pub fn handle_paste(&mut self, text: &str) {
        match self.focus {
            Field::File => self.file.insert(text),
            Field::Chunk => self.chunk.insert(text),
            _ => {}
        }
    }

    pub fn handle_key(&mut self, key: &KeyEvent, services: &dyn Services) -> SetupAction {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('q' | 'c') if control => return SetupAction::Quit,
            KeyCode::Char('o') if control => return SetupAction::Browse,
            KeyCode::Char('s') if control => return self.start(services),
            KeyCode::Tab | KeyCode::Down => return self.focus_and_none(true),
            KeyCode::BackTab | KeyCode::Up => return self.focus_and_none(false),
            _ => {}
        }

        match self.focus {
            Field::File => {
                if key.code == KeyCode::Enter {
                    self.move_focus(true);
                } else {
                    self.file.handle_key(key);
                }
            }
            Field::Chunk => {
                if key.code == KeyCode::Enter {
                    self.move_focus(true);
                } else {
                    self.chunk.handle_key(key);
                }
            }
            Field::Model | Field::Language | Field::Device | Field::Compute => {
                self.change_value(key.code, services)
            }
            Field::Advanced => {
                if matches!(key.code, KeyCode::Enter | KeyCode::Char(' ')) {
                    self.advanced_open = !self.advanced_open;
                }
            }
            Field::Start => {
                if key.code == KeyCode::Enter {
                    return self.start(services);
                }
            }
        }
        SetupAction::None
    }

    fn focus_and_none(&mut self, forward: bool) -> SetupAction {
        self.move_focus(forward);
        SetupAction::None
    }

    fn change_value(&mut self, code: KeyCode, services: &dyn Services) {
        let forward = match code {
            KeyCode::Right | KeyCode::Char(' ') => true,
            KeyCode::Left => false,
            KeyCode::Enter => {
                self.move_focus(true);
                return;
            }
            _ => return,
        };
        match self.focus {
            Field::Model => self.model = cycle(self.model, MODEL_REPOSITORIES.len(), forward),
            Field::Language => self.language = cycle(self.language, LANGUAGES.len() + 1, forward),
            Field::Device => self.change_device(services, forward),
            Field::Compute => {
                let options = self.compute_types(services).len() + 1;
                self.compute = cycle(self.compute, options, forward);
            }
            _ => {}
        }
    }

    // ----- drawing -----

    fn field_block(&self, field: Field, title: &str) -> Block<'static> {
        let focused = self.focus == field;
        let style = Style::new().fg(if focused { ACCENT } else { DIM });
        let title = Span::styled(
            format!(" {title} "),
            if focused {
                style.add_modifier(Modifier::BOLD)
            } else {
                style
            },
        );
        Block::bordered().border_style(style).title(title)
    }

    fn select_line(&self, field: Field, text: String) -> Line<'static> {
        if self.focus == field {
            Line::from(vec![
                Span::styled("◀ ", Style::new().fg(ACCENT)),
                Span::styled(text, Style::new().add_modifier(Modifier::BOLD)),
                Span::styled(" ▶", Style::new().fg(ACCENT)),
            ])
        } else {
            Line::from(format!("  {text}"))
        }
    }

    fn model_text(&self, services: &dyn Services) -> String {
        let name = MODEL_REPOSITORIES[self.model].0;
        let state = if services.is_model_downloaded(name) {
            "downloaded"
        } else {
            "downloads on first use"
        };
        format!("{} ({state})", capitalize(name))
    }

    fn language_text(&self) -> String {
        match self.language.checked_sub(1) {
            None => "Auto-detect".to_string(),
            Some(i) => format!("{} ({})", LANGUAGES[i].1, LANGUAGES[i].0),
        }
    }

    fn compute_text(&self, services: &dyn Services) -> String {
        match self.compute.checked_sub(1) {
            None => "Best available".to_string(),
            Some(i) => self
                .compute_types(services)
                .get(i)
                .map_or("Best available", |t| t)
                .to_string(),
        }
    }

    fn render_text_field(
        &self,
        frame: &mut Frame,
        area: Rect,
        field: Field,
        title: &str,
        input: &TextInput,
        hint: &str,
    ) {
        let block = self.field_block(field, title);
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let hint_width = hint.chars().count() as u16;
        let room = (inner.width.saturating_sub(hint_width + 1)) as usize;
        let (shown, cursor) = input.window(room);
        let mut spans = vec![Span::raw(shown)];
        if input.text().is_empty() && self.focus != field {
            spans = vec![Span::styled("(none)", Style::new().fg(DIM))];
        }
        frame.render_widget(Paragraph::new(Line::from(spans)), inner);
        if !hint.is_empty() {
            let x = inner.right().saturating_sub(hint_width);
            let hint_area = Rect::new(x, inner.y, hint_width.min(inner.width), 1);
            frame.render_widget(
                Paragraph::new(Span::styled(hint.to_string(), Style::new().fg(DIM))),
                hint_area,
            );
        }
        if self.focus == field {
            frame.set_cursor_position(Position::new(inner.x + cursor as u16, inner.y));
        }
    }

    pub fn render(&self, frame: &mut Frame, area: Rect, services: &dyn Services) {
        let width = area.width.min(90);
        let area = Rect::new(
            area.x + (area.width - width) / 2,
            area.y,
            width,
            area.height,
        );
        let advanced_height = if self.advanced_open { 4 } else { 1 };
        let error_height = if self.errors.is_empty() {
            0
        } else {
            self.errors.len() as u16 + 2
        };
        let [
            header,
            file,
            pickers,
            device,
            advanced,
            start,
            errors,
            footer,
        ] = Layout::vertical([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(5),
            Constraint::Length(advanced_height),
            Constraint::Length(3),
            Constraint::Length(error_height),
            Constraint::Min(1),
        ])
        .areas(area);

        self.render_header(frame, header, services);
        self.render_text_field(
            frame,
            file,
            Field::File,
            "File",
            &self.file,
            "Ctrl+O browse",
        );

        let [model_area, language_area] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(pickers);
        for (area, field, title, text) in [
            (model_area, Field::Model, "Model", self.model_text(services)),
            (
                language_area,
                Field::Language,
                "Language",
                self.language_text(),
            ),
        ] {
            let block = self.field_block(field, title);
            let inner = block.inner(area);
            frame.render_widget(block, area);
            frame.render_widget(Paragraph::new(self.select_line(field, text)), inner);
        }

        self.render_devices(frame, device, services);
        self.render_advanced(frame, advanced, services);

        let start_style = if self.focus == Field::Start {
            Style::new()
                .fg(Color::Black)
                .bg(ACCENT)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(ACCENT)
        };
        frame.render_widget(
            Paragraph::new(Span::styled("  Start transcription  ", start_style))
                .centered()
                .block(Block::new()),
            Rect::new(start.x, start.y + 1, start.width, 1),
        );

        if !self.errors.is_empty() {
            let lines: Vec<Line> = self
                .errors
                .iter()
                .map(|e| Line::from(format!("• {e}")))
                .collect();
            frame.render_widget(
                Paragraph::new(lines).style(Style::new().fg(ERROR)).block(
                    Block::bordered()
                        .border_style(Style::new().fg(ERROR))
                        .title(" Please fix "),
                ),
                errors,
            );
        }

        let hints = " Tab move · ←/→ change · Ctrl+O browse · Ctrl+S start · Ctrl+Q quit";
        frame.render_widget(
            Paragraph::new(Span::styled(hints, Style::new().fg(DIM))),
            footer,
        );
    }

    fn render_header(&self, frame: &mut Frame, area: Rect, services: &dyn Services) {
        let system = services.system();
        let gpu = if system.gpu().available {
            "GPU ready"
        } else {
            "no GPU"
        };
        let title = Line::from(vec![
            Span::styled(
                "Whispry",
                Style::new().fg(ACCENT).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  Private, local speech-to-text"),
        ]);
        let info = Line::from(Span::styled(
            format!(
                "{} · {} CPU threads · {gpu}",
                system.os_label, system.cpu_threads
            ),
            Style::new().fg(DIM),
        ));
        frame.render_widget(Paragraph::new(vec![title, info]), area);
    }

    fn render_devices(&self, frame: &mut Frame, area: Rect, services: &dyn Services) {
        let block = self.field_block(Field::Device, "Device");
        let inner = block.inner(area);
        frame.render_widget(block, area);
        let lines: Vec<Line> = self
            .device_rows(services)
            .into_iter()
            .enumerate()
            .map(|(index, row)| {
                let selected = index == self.device;
                let marker = if selected { "(●)" } else { "( )" };
                let style = if !row.enabled {
                    Style::new().fg(DIM)
                } else if selected {
                    Style::new().fg(OK).add_modifier(Modifier::BOLD)
                } else {
                    Style::new()
                };
                let mut spans = vec![Span::styled(format!("{marker} {}", row.label), style)];
                if let Some(note) = row.note {
                    spans.push(Span::styled(format!("  {note}"), Style::new().fg(DIM)));
                }
                Line::from(spans)
            })
            .collect();
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn render_advanced(&self, frame: &mut Frame, area: Rect, services: &dyn Services) {
        let focused = self.focus == Field::Advanced;
        let style = if focused {
            Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(DIM)
        };
        let marker = if self.advanced_open { "▾" } else { "▸" };
        let header = Rect::new(area.x, area.y, area.width, 1);
        frame.render_widget(
            Paragraph::new(Span::styled(format!("{marker} Advanced"), style)),
            header,
        );
        if !self.advanced_open {
            return;
        }
        let rest = Rect::new(area.x, area.y + 1, area.width, area.height - 1);
        let [compute, chunk] =
            Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)])
                .areas(rest);
        let compute_block = self.field_block(Field::Compute, "Compute type");
        let inner = compute_block.inner(compute);
        frame.render_widget(compute_block, compute);
        frame.render_widget(
            Paragraph::new(self.select_line(Field::Compute, self.compute_text(services))),
            inner,
        );
        self.render_text_field(
            frame,
            chunk,
            Field::Chunk,
            "Minutes per chunk",
            &self.chunk,
            "",
        );
    }
}
