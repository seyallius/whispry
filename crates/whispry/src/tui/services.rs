//! What the terminal UI asks of the rest of the program.
//!
//! The UI only turns key presses into a job and draws what the job reports.
//! Everything it needs to know or do goes through this trait, so tests can
//! replace it with a fake.

use whispry_core::config::{JobConfig, validate_config};
use whispry_core::models::find_cached_model;
use whispry_core::platform::SystemInfo;
use whispry_core::transcriber::JobEnvironment;

pub trait Services {
    fn system(&self) -> &SystemInfo;
    /// Whether the worker's Python environment was found.
    fn worker_available(&self) -> bool;
    fn is_model_downloaded(&self, model: &str) -> bool;
    fn validate(&self, config: &JobConfig) -> Vec<String>;
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
}
