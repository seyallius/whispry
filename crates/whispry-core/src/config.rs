//! Job configuration shared by every front end.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

/// Model name and Hugging Face repository, in display order.
pub const MODEL_REPOSITORIES: &[(&str, &str)] = &[
    ("small", "Systran/faster-whisper-small"),
    ("medium", "Systran/faster-whisper-medium"),
];

/// Language code and display name, in display order.
pub const LANGUAGES: &[(&str, &str)] = &[("en", "English"), ("fa", "Persian")];

pub const DEFAULT_MODEL: &str = "small";
pub const DEFAULT_CHUNK_MINUTES: f64 = 10.0;
pub const MAX_CHUNK_MINUTES: f64 = 60.0;
pub const BEAM_SIZE: u32 = 5;
pub const SAMPLE_RATE: u32 = 16_000;

/// The Hugging Face repository for a model name, if the model is known.
pub fn model_repository(model: &str) -> Option<&'static str> {
    MODEL_REPOSITORIES
        .iter()
        .find(|(name, _)| *name == model)
        .map(|(_, repo)| *repo)
}

/// Which compute device the user asked for.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChoice {
    /// Use the GPU when it works, otherwise the CPU.
    #[default]
    Auto,
    Cpu,
    Cuda,
}

impl DeviceChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceChoice::Auto => "auto",
            DeviceChoice::Cpu => "cpu",
            DeviceChoice::Cuda => "cuda",
        }
    }
}

impl fmt::Display for DeviceChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for DeviceChoice {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "auto" => Ok(DeviceChoice::Auto),
            "cpu" => Ok(DeviceChoice::Cpu),
            "cuda" => Ok(DeviceChoice::Cuda),
            other => Err(format!("Unknown device: {other}")),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobConfig {
    pub audio: PathBuf,
    pub model: String,
    /// `None` means detect the language from the audio.
    pub language: Option<String>,
    /// Defaults to a `.txt` file beside the audio.
    pub output: Option<PathBuf>,
    pub device: DeviceChoice,
    pub compute_type: Option<String>,
    /// Audio is processed one window of this length at a time (about 38 MB
    /// per 10 minutes), so RAM use does not depend on the file size.
    pub chunk_minutes: f64,
}

impl JobConfig {
    pub fn new(audio: impl Into<PathBuf>) -> Self {
        JobConfig {
            audio: audio.into(),
            model: DEFAULT_MODEL.to_string(),
            language: None,
            output: None,
            device: DeviceChoice::Auto,
            compute_type: None,
            chunk_minutes: DEFAULT_CHUNK_MINUTES,
        }
    }

    pub fn output_path(&self) -> PathBuf {
        match &self.output {
            Some(output) => output.clone(),
            None => Path::new(&self.audio).with_extension("txt"),
        }
    }
}

/// Human-readable problems with the config (empty when valid).
pub fn validate_config(config: &JobConfig) -> Vec<String> {
    let mut problems = Vec::new();

    if !config.audio.is_file() {
        problems.push(format!("File not found: {}", config.audio.display()));
    }

    if model_repository(&config.model).is_none() {
        problems.push(format!("Unknown model: {}", config.model));
    }

    if let Some(language) = &config.language
        && !LANGUAGES.iter().any(|(code, _)| code == language)
    {
        problems.push(format!("Unsupported language: {language}"));
    }

    if !(1.0..=MAX_CHUNK_MINUTES).contains(&config.chunk_minutes) {
        problems.push(format!(
            "Chunk length must be 1-{MAX_CHUNK_MINUTES} minutes"
        ));
    }

    problems
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_defaults_beside_input() {
        let config = JobConfig::new("/tmp/talk.mp3");
        assert_eq!(config.output_path(), PathBuf::from("/tmp/talk.txt"));
    }

    #[test]
    fn explicit_output_wins() {
        let mut config = JobConfig::new("/tmp/talk.mp3");
        config.output = Some(PathBuf::from("/tmp/notes.txt"));
        assert_eq!(config.output_path(), PathBuf::from("/tmp/notes.txt"));
    }

    #[test]
    fn reports_every_problem() {
        let mut config = JobConfig::new("/nonexistent/missing.wav");
        config.model = "huge".into();
        config.language = Some("xx".into());
        assert_eq!(validate_config(&config).len(), 3);
    }

    #[test]
    fn rejects_out_of_range_chunk_length() {
        let dir = tempfile_dir();
        let audio = dir.join("a.wav");
        std::fs::write(&audio, b"").unwrap();
        let mut config = JobConfig::new(&audio);
        config.chunk_minutes = 0.5;
        assert_eq!(validate_config(&config).len(), 1);
        config.chunk_minutes = 61.0;
        assert_eq!(validate_config(&config).len(), 1);
        config.chunk_minutes = 60.0;
        assert!(validate_config(&config).is_empty());
    }

    #[test]
    fn accepts_valid_config() {
        let dir = tempfile_dir();
        let audio = dir.join("a.wav");
        std::fs::write(&audio, b"").unwrap();
        let mut config = JobConfig::new(&audio);
        config.language = Some("fa".into());
        assert!(validate_config(&config).is_empty());
    }

    #[test]
    fn parses_device_names() {
        assert_eq!("auto".parse(), Ok(DeviceChoice::Auto));
        assert_eq!("cpu".parse(), Ok(DeviceChoice::Cpu));
        assert_eq!("cuda".parse(), Ok(DeviceChoice::Cuda));
        assert!("tpu".parse::<DeviceChoice>().is_err());
        assert_eq!(DeviceChoice::Cuda.to_string(), "cuda");
    }

    /// A unique scratch folder under the system temp dir.
    fn tempfile_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU32, Ordering};
        static NEXT: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "whispry-config-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
