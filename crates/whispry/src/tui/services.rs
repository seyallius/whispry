//! What the terminal UI asks of the rest of the program.
//!
//! The UI only turns key presses into a job and draws what the job reports.
//! Everything it needs to know or do goes through this trait, so tests can
//! replace it with a fake.

use std::sync::mpsc::Sender;

use whispry_core::cancel::CancelToken;
use whispry_core::config::{JobConfig, validate_config};
use whispry_core::events::Event;
use whispry_core::ffmpeg::{FFmpeg, find_ffmpeg, install_instructions};
use whispry_core::models::find_cached_model;
use whispry_core::platform::{Os, SystemInfo};
use whispry_core::transcriber::{JobEnvironment, JobError, JobSummary, run_job};

/// What a running job sends back to the UI.
pub enum JobMessage {
    Event(Event),
    /// The job is over. This is always the last message.
    Done(Result<JobSummary, JobError>),
}

pub trait Services {
    fn system(&self) -> &SystemInfo;
    /// Whether the worker's Python environment was found.
    fn worker_available(&self) -> bool;
    fn is_model_downloaded(&self, model: &str) -> bool;
    fn validate(&self, config: &JobConfig) -> Vec<String>;
    fn find_ffmpeg(&self) -> Option<FFmpeg>;
    fn ffmpeg_install_steps(&self) -> Vec<String>;
    /// Run a job in the background, sending its events and result to `sender`.
    fn start_job(&self, config: JobConfig, cancel: CancelToken, sender: Sender<JobMessage>);
}

/// The real thing, backed by the core.
pub struct RealServices {
    environment: JobEnvironment,
}

impl RealServices {
    pub fn new() -> Self {
        RealServices {
            environment: JobEnvironment::detect(),
        }
    }
}

impl Services for RealServices {
    fn system(&self) -> &SystemInfo {
        &self.environment.system
    }

    fn worker_available(&self) -> bool {
        self.environment.worker.is_some()
    }

    fn is_model_downloaded(&self, model: &str) -> bool {
        find_cached_model(model, self.environment.model_directory.as_deref()).is_some()
    }

    fn validate(&self, config: &JobConfig) -> Vec<String> {
        validate_config(config)
    }

    fn find_ffmpeg(&self) -> Option<FFmpeg> {
        find_ffmpeg()
    }

    fn ffmpeg_install_steps(&self) -> Vec<String> {
        install_instructions(Os::current())
    }

    fn start_job(&self, config: JobConfig, cancel: CancelToken, sender: Sender<JobMessage>) {
        let environment = self.environment.clone();
        std::thread::spawn(move || {
            let events = sender.clone();
            let result = run_job(
                &config,
                &environment,
                &mut |event| {
                    let _ = events.send(JobMessage::Event(event));
                },
                &cancel,
            );
            let _ = sender.send(JobMessage::Done(result));
        });
    }
}
