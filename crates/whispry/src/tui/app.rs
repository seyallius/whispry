//! The terminal UI as a whole: which screen is showing and what it returns.

use std::path::PathBuf;

use ratatui::Frame;
use ratatui::crossterm::event::KeyEvent;
use whispry_core::config::JobConfig;

use super::services::Services;
use super::setup::{SetupAction, SetupState};

pub struct App {
    services: Box<dyn Services>,
    setup: SetupState,
    outcome: Option<JobConfig>,
    quit: bool,
}

impl App {
    pub fn new(services: Box<dyn Services>, initial_file: Option<PathBuf>) -> Self {
        App {
            services,
            setup: SetupState::new(initial_file.as_deref()),
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
        match self.setup.handle_key(&key, self.services.as_ref()) {
            SetupAction::None => {}
            SetupAction::Quit => self.quit = true,
            SetupAction::Start(config) => {
                self.outcome = Some(config);
                self.quit = true;
            }
        }
    }

    pub fn on_paste(&mut self, text: &str) {
        self.setup.handle_paste(text);
    }

    pub fn render(&self, frame: &mut Frame) {
        let area = frame.area();
        self.setup.render(frame, area, self.services.as_ref());
    }
}
