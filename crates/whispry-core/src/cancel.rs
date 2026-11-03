//! Cooperative cancellation for jobs running in a background thread.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Set from any thread; checked by the job between units of work. Clones
/// share the same flag.
#[derive(Clone, Debug, Default)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

/// What a job reports once it notices cancellation.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[error("Transcription cancelled")]
pub struct JobCancelled {
    /// The output file, when some text was already written to it.
    pub output: Option<PathBuf>,
    /// How far into the audio the job got, in seconds.
    pub position: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_flag() {
        let token = CancelToken::new();
        let other = token.clone();
        assert!(!other.is_cancelled());
        token.cancel();
        assert!(other.is_cancelled());
    }
}
