//! The terminal UI as a whole: which screen is showing and what it returns.

use std::path::PathBuf;

use ratatui::Frame;
use ratatui::crossterm::event::KeyEvent;
use whispry_core::config::JobConfig;

use super::file_picker::{PickerAction, PickerState};
use super::services::Services;
use super::setup::{SetupAction, SetupState};

pub struct App {
    services: Box<dyn Services>,
    setup: SetupState,
    picker: Option<PickerState>,
    outcome: Option<JobConfig>,
    quit: bool,
}

impl App {
    pub fn new(services: Box<dyn Services>, initial_file: Option<PathBuf>) -> Self {
        App {
            services,
            setup: SetupState::new(initial_file.as_deref()),
            picker: None,
            outcome: None,
            quit: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.quit
    }

    /// The job the user chose to start, if they did.
    pub fn into_outcome(self) -> Option<JobConfig> {
        self.outcome
    }

    pub fn on_key(&mut self, key: KeyEvent) {
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
            SetupAction::Start(config) => {
                self.outcome = Some(config);
                self.quit = true;
            }
        }
    }

    pub fn on_paste(&mut self, text: &str) {
        if self.picker.is_none() {
            self.setup.handle_paste(text);
        }
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        self.setup.render(frame, area, self.services.as_ref());
        if let Some(picker) = &self.picker {
            picker.render(frame, area);
        }
    }
}
