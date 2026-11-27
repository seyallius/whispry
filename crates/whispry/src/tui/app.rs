//! The terminal UI as a whole: which screen is showing.

use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::{Duration, Instant};

use ratatui::Frame;
use ratatui::crossterm::event::{
    KeyCode, KeyEvent, KeyModifiers, MouseButton, MouseEvent, MouseEventKind,
};
use whispry_core::cancel::CancelToken;
use whispry_core::config::JobConfig;

use super::ffmpeg_dialog::{DialogAction, FFmpegDialog};
use super::file_picker::{PickerAction, PickerState};
use super::help;
use super::run::{RunAction, RunState};
use super::services::Services;
use super::setup::{Field, SetupAction, SetupState};
use super::theme::{self, Hit};

/// Clicks this soon after one that acted are ignored.
const DOUBLE_CLICK: Duration = Duration::from_millis(300);

pub struct App {
    services: Box<dyn Services>,
    setup: SetupState,
    picker: Option<PickerState>,
    ffmpeg_dialog: Option<FFmpegDialog>,
    /// The run screen, once a job has been started.
    run: Option<RunState>,
    quit: bool,
    /// The help screen, with how far it is scrolled.
    help: Option<u16>,
    /// When a click last did something, so a double-click cannot act twice
    /// on whatever the first click uncovered.
    last_act: Option<Instant>,
}

impl App {
    pub fn new(services: Box<dyn Services>, initial_file: Option<PathBuf>) -> Self {
        App {
            services,
            setup: SetupState::new(initial_file.as_deref()),
            picker: None,
            ffmpeg_dialog: None,
            run: None,
            quit: false,
            help: None,
            last_act: None,
        }
    }

    pub fn is_done(&self) -> bool {
        self.quit
    }

    /// Start the job, or ask the user to install FFmpeg first.
    fn start_or_ask_for_ffmpeg(&mut self, config: JobConfig) {
        if self.services.find_ffmpeg().is_some() {
            self.start_job(config);
        } else {
            let steps = self.services.ffmpeg_install_steps();
            self.ffmpeg_dialog = Some(FFmpegDialog::new(steps, config));
        }
    }

    fn start_job(&mut self, config: JobConfig) {
        let (sender, receiver) = channel();
        let cancel = CancelToken::new();
        self.services
            .start_job(config.clone(), cancel.clone(), sender);
        self.run = Some(RunState::new(config, cancel, receiver, Instant::now()));
    }

    /// Take in whatever the running job has reported. Call this often.
    pub fn poll(&mut self) {
        if let Some(run) = &mut self.run {
            run.poll(Instant::now());
            if run.quit_when_stopped() && run.status.is_over() {
                self.quit = true;
            }
        }
    }

    /// Keys that work on every screen. True if the key was used.
    fn global_key(&mut self, key: &KeyEvent) -> bool {
        let control = key.modifiers.contains(KeyModifiers::CONTROL);
        if control && key.code == KeyCode::Char('t') {
            theme::next_theme();
            return true;
        }
        if let Some(scroll) = &mut self.help {
            match key.code {
                KeyCode::Down => *scroll += 1,
                KeyCode::Up => *scroll = scroll.saturating_sub(1),
                KeyCode::PageDown => *scroll += 8,
                KeyCode::PageUp => *scroll = scroll.saturating_sub(8),
                KeyCode::Esc | KeyCode::Enter | KeyCode::F(1) | KeyCode::Char('?' | 'q' | 'c') => {
                    self.help = None;
                }
                _ => {}
            }
            return true;
        }
        // "?" is a letter you can type in a text field; F1 always works.
        let typing = self.run.is_none()
            && self.picker.is_none()
            && self.ffmpeg_dialog.is_none()
            && matches!(self.setup.focus, Field::File | Field::Chunk);
        let opens = key.code == KeyCode::F(1) || (key.code == KeyCode::Char('?') && !typing);
        if opens && !control {
            self.help = Some(0);
        }
        opens && !control
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        if self.global_key(&key) {
            return;
        }
        if let Some(dialog) = &mut self.ffmpeg_dialog {
            match dialog.handle_key(&key) {
                DialogAction::None => {}
                DialogAction::Close => self.ffmpeg_dialog = None,
                DialogAction::Check => {
                    if self.services.find_ffmpeg().is_some() {
                        if let Some(dialog) = self.ffmpeg_dialog.take() {
                            self.start_job(dialog.config);
                        }
                    } else {
                        dialog.report_still_missing();
                    }
                }
            }
            return;
        }
        if let Some(run) = &mut self.run {
            let action = run.handle_key(&key);
            self.apply_run(action);
            return;
        }
        if let Some(picker) = &mut self.picker {
            let action = picker.handle_key(&key);
            self.apply_picker(action);
            return;
        }
        match self.setup.handle_key(&key, self.services.as_ref()) {
            SetupAction::None => {}
            SetupAction::Quit => self.quit = true,
            SetupAction::Browse => {
                self.picker = Some(PickerState::open(&self.setup.browse_start()));
            }
            SetupAction::Start(config) => self.start_or_ask_for_ffmpeg(config),
        }
    }

    fn apply_run(&mut self, action: RunAction) {
        match action {
            RunAction::None => {}
            RunAction::New => self.run = None,
            RunAction::Quit => self.quit = true,
            RunAction::Retry(config) => self.start_or_ask_for_ffmpeg(config),
        }
    }

    fn apply_picker(&mut self, action: PickerAction) {
        match action {
            PickerAction::None => {}
            PickerAction::Cancel => self.picker = None,
            PickerAction::Choose(path) => {
                self.setup.choose_file(&path);
                self.picker = None;
            }
        }
    }

    /// A left click does what the keyboard would: focus the thing, then press it.
    pub fn on_mouse(&mut self, mouse: MouseEvent) {
        self.on_mouse_at(mouse, Instant::now());
    }

    pub fn on_mouse_at(&mut self, mouse: MouseEvent, now: Instant) {
        theme::set_mouse(mouse.column, mouse.row);
        let wheel: isize = match mouse.kind {
            MouseEventKind::ScrollUp => -3,
            MouseEventKind::ScrollDown => 3,
            MouseEventKind::Down(MouseButton::Left) => 0,
            _ => return,
        };
        if wheel != 0 {
            if let Some(scroll) = &mut self.help {
                *scroll = scroll.saturating_add_signed(wheel as i16);
            } else if let Some(picker) = &mut self.picker {
                picker.scroll(wheel);
            }
            return;
        }
        let Some(hit) = theme::at(mouse.column, mouse.row) else {
            return;
        };
        if self
            .last_act
            .is_some_and(|t| now.saturating_duration_since(t) < DOUBLE_CLICK)
        {
            return;
        }
        let acts = match hit {
            Hit::Run(_)
            | Hit::Dialog(_)
            | Hit::Browse
            | Hit::PickerOutside
            | Hit::Help
            | Hit::HelpClose
            | Hit::Theme => true,
            Hit::Setup(field) => matches!(field, Field::Start | Field::Advanced),
            Hit::PickerRow(index) => self.picker.as_ref().is_some_and(|p| p.is_selected(index)),
            _ => false,
        };
        if acts {
            self.last_act = Some(now);
        }
        let enter = KeyEvent::from(KeyCode::Enter);
        match hit {
            Hit::Nothing => {}
            Hit::HelpClose => self.help = None,
            Hit::Help => self.help = Some(0),
            Hit::Theme => theme::next_theme(),
            Hit::PickerCrumb(index) => {
                if let Some(picker) = &mut self.picker {
                    picker.go_to_ancestor(index);
                }
            }
            Hit::Dialog(index) => {
                if let Some(dialog) = &mut self.ffmpeg_dialog {
                    dialog.focus_on(index);
                    self.on_key(enter);
                }
            }
            Hit::Run(button) => {
                if let Some(run) = &mut self.run {
                    let action = run.press(button);
                    self.apply_run(action);
                }
            }
            Hit::PickerRow(index) => {
                if let Some(picker) = &mut self.picker {
                    let action = picker.click(index);
                    self.apply_picker(action);
                }
            }
            Hit::PickerOutside => self.picker = None,
            Hit::Browse => {
                self.picker = Some(PickerState::open(&self.setup.browse_start()));
            }
            Hit::Device(index) => self.setup.pick_device(index, self.services.as_ref()),
            Hit::Arrow(field, forward) => {
                self.setup.focus_on(field);
                self.on_key(KeyEvent::from(if forward {
                    KeyCode::Right
                } else {
                    KeyCode::Left
                }));
            }
            Hit::Setup(field) => {
                self.setup.focus_on(field);
                if matches!(field, Field::Start | Field::Advanced) {
                    self.on_key(enter);
                }
            }
        }
    }

    pub fn on_paste(&mut self, text: &str) {
        if self.run.is_none() && self.picker.is_none() && self.ffmpeg_dialog.is_none() {
            self.setup.handle_paste(text);
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        theme::clear_hits();
        let area = frame.area();
        if let Some(run) = &self.run {
            run.render(frame, area, Instant::now());
        } else {
            self.setup.render(frame, area, self.services.as_ref());
            if let Some(picker) = &self.picker {
                picker.render(frame, area);
            }
        }
        if let Some(dialog) = &self.ffmpeg_dialog {
            dialog.render(frame, area);
        }
        if let Some(scroll) = self.help {
            help::render(frame, area, scroll);
        }
    }
}
