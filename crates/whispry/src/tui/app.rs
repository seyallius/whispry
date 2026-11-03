//! The terminal UI as a whole: which screen is showing.

use std::path::PathBuf;
use std::sync::mpsc::channel;
use std::time::Instant;

use ratatui::Frame;
use ratatui::crossterm::event::KeyEvent;
use whispry_core::cancel::CancelToken;
use whispry_core::config::JobConfig;

use super::ffmpeg_dialog::{DialogAction, FFmpegDialog};
use super::file_picker::{PickerAction, PickerState};
use super::run::{RunAction, RunState};
use super::services::Services;
use super::setup::{SetupAction, SetupState};

pub struct App {
    services: Box<dyn Services>,
    setup: SetupState,
    picker: Option<PickerState>,
    ffmpeg_dialog: Option<FFmpegDialog>,
    /// The run screen, once a job has been started.
    run: Option<RunState>,
    quit: bool,
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

    pub fn on_key(&mut self, key: KeyEvent) {
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
            match run.handle_key(&key) {
                RunAction::None => {}
                RunAction::New => self.run = None,
                RunAction::Quit => self.quit = true,
                RunAction::Retry(config) => self.start_or_ask_for_ffmpeg(config),
            }
            return;
        }
        if let Some(picker) = &mut self.picker {
            match picker.handle_key(&key) {
                PickerAction::None => {}
                PickerAction::Cancel => self.picker = None,
                PickerAction::Choose(path) => {
                    self.setup.choose_file(&path);
                    self.picker = None;
                }
            }
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

    pub fn on_paste(&mut self, text: &str) {
        if self.run.is_none() && self.picker.is_none() && self.ffmpeg_dialog.is_none() {
            self.setup.handle_paste(text);
        }
    }

    pub fn render(&self, frame: &mut Frame) {
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
    }
}
