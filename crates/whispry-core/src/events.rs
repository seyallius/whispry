//! Events emitted by a transcription job.
//!
//! The core never prints. It reports everything through these events so any
//! front end can render them however it likes.

use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Model,
    Transcribe,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    StageStarted {
        stage: Stage,
        message: String,
    },
    StageFinished {
        stage: Stage,
        elapsed: f64,
    },
    /// Progress within a stage. `total` is `None` when it is unknown.
    Progress {
        stage: Stage,
        completed: f64,
        total: Option<f64>,
        unit: String,
    },
    LanguageDetected {
        language: String,
        probability: f64,
    },
    SegmentReady {
        start: f64,
        end: f64,
        text: String,
    },
    Log(String),
    Finished {
        output: PathBuf,
        audio_duration: f64,
        elapsed: f64,
    },
}

/// Where a job sends its events.
pub type EventSink<'a> = &'a mut dyn FnMut(Event);

/// An event sink that drops everything.
pub fn ignore_event(_event: Event) {}
