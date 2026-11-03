//! All of Whispry's logic: device detection, FFmpeg, audio streaming,
//! chunking, models and the transcription worker.
//!
//! This crate never prints. Front ends (the TUI and the CLI) turn user input
//! into a job and render the events the job emits.

pub mod config;
pub mod events;
pub mod timestamps;

/// The version of Whispry, taken from the workspace manifest.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_not_empty() {
        assert!(!VERSION.is_empty());
    }
}
