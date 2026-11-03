//! Run a transcription job and report progress through events.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::audio::{AudioDecodeError, stream_audio};
use crate::cancel::{CancelToken, JobCancelled};
use crate::chunking::{AudioWindow, find_cut, iter_windows};
use crate::config::{BEAM_SIZE, JobConfig};
use crate::events::{Event, EventSink, Stage};
use crate::ffmpeg::{FFmpegNotFound, probe_duration, require_ffmpeg};
use crate::models::{ModelError, ensure_model};
use crate::platform::{
    DeviceUnavailable, SystemInfo, detect_system, nvidia_wheel_dirs, resolve_device,
};
use crate::timestamps::format_timestamp;
use crate::worker::{
    Update, WorkerClient, WorkerCommand, WorkerError, find_python, load_model, venv_of,
};

/// Tail of the previous window's text given to the next window as context.
pub const PROMPT_CHARACTERS: usize = 200;

/// How long the search for a quiet point at the end of a window goes on.
const SEARCH_SECONDS: f64 = 30.0;

#[derive(Debug, thiserror::Error)]
pub enum JobError {
    #[error(transparent)]
    Device(#[from] DeviceUnavailable),
    #[error(transparent)]
    FFmpeg(#[from] FFmpegNotFound),
    #[error(transparent)]
    Model(ModelError),
    #[error(transparent)]
    Worker(WorkerError),
    #[error(transparent)]
    Audio(#[from] AudioDecodeError),
    #[error("Could not write the transcription: {0}")]
    Output(#[from] std::io::Error),
    #[error(
        "The transcription worker's Python environment was not found. \
         Run the setup script, or set WHISPRY_PYTHON to a Python with whispry-worker installed."
    )]
    WorkerNotFound,
    #[error(transparent)]
    Cancelled(#[from] JobCancelled),
}

impl From<ModelError> for JobError {
    fn from(error: ModelError) -> Self {
        match error {
            ModelError::Cancelled => JobCancelled {
                output: None,
                position: 0.0,
            }
            .into(),
            other => JobError::Model(other),
        }
    }
}

impl From<WorkerError> for JobError {
    fn from(error: WorkerError) -> Self {
        match error {
            WorkerError::Cancelled => JobCancelled {
                output: None,
                position: 0.0,
            }
            .into(),
            other => JobError::Worker(other),
        }
    }
}

/// What a job runs on: the machine's devices and the transcription worker.
#[derive(Clone, Debug)]
pub struct JobEnvironment {
    pub system: SystemInfo,
    /// `None` when the worker's Python environment was not found.
    pub worker: Option<WorkerCommand>,
    /// Where models are cached; `None` uses the per-user default.
    pub model_directory: Option<PathBuf>,
}

impl JobEnvironment {
    /// Inspect this machine and look for the worker's Python environment.
    pub fn detect() -> Self {
        let os = crate::platform::Os::current();
        let python = find_python(os);
        let library_dirs = python
            .as_deref()
            .and_then(venv_of)
            .map(|venv| nvidia_wheel_dirs(&venv, os))
            .unwrap_or_default();
        JobEnvironment {
            system: detect_system(&library_dirs),
            worker: python.map(WorkerCommand::python_module),
            model_directory: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct JobSummary {
    pub output: PathBuf,
    /// How much audio was transcribed, in seconds.
    pub audio_duration: f64,
    pub elapsed: f64,
}

/// The last `PROMPT_CHARACTERS` characters of `texts` joined by spaces.
fn prompt_tail(texts: &[String]) -> String {
    let joined = texts.join(" ");
    let skip = joined.chars().count().saturating_sub(PROMPT_CHARACTERS);
    joined.chars().skip(skip).collect()
}

/// Transcribe `config.audio` and write timestamped text to the output file.
///
/// Audio is decoded and transcribed one window (`config.chunk_minutes`) at a
/// time, so memory use does not grow with the length of the file.
///
/// Returns [`JobError::Cancelled`] soon after `cancel` is triggered (checked
/// between segments); text transcribed so far stays in the output file.
pub fn run_job(
    config: &JobConfig,
    environment: &JobEnvironment,
    on_event: EventSink,
    cancel: &CancelToken,
) -> Result<JobSummary, JobError> {
    let mut position = 0.0;
    let result = run(config, environment, on_event, cancel, &mut position);
    count_failure_after_cancel_as_cancelled(result, cancel, &config.output_path(), position)
}

/// Once the user has cancelled, the worker or FFmpeg may stop in ways that
/// look like errors. The user asked for the stop, so report a cancellation.
fn count_failure_after_cancel_as_cancelled(
    result: Result<JobSummary, JobError>,
    cancel: &CancelToken,
    output: &Path,
    position: f64,
) -> Result<JobSummary, JobError> {
    match result {
        Err(error) if cancel.is_cancelled() && !matches!(error, JobError::Cancelled(_)) => {
            Err(JobCancelled {
                output: output.exists().then(|| output.to_path_buf()),
                position,
            }
            .into())
        }
        other => other,
    }
}

fn run(
    config: &JobConfig,
    environment: &JobEnvironment,
    on_event: EventSink,
    cancel: &CancelToken,
    position: &mut f64,
) -> Result<JobSummary, JobError> {
    let job_start = Instant::now();
    let output = config.output_path();

    let cancelled = |position: f64| JobCancelled {
        output: output.exists().then(|| output.clone()),
        position,
    };

    // Fail fast on an impossible device or missing FFmpeg before downloading.
    resolve_device(
        config.device,
        &environment.system,
        config.compute_type.as_deref(),
    )?;
    let ffmpeg = require_ffmpeg()?;
    let worker_command = environment
        .worker
        .as_ref()
        .ok_or(JobError::WorkerNotFound)?;
    on_event(Event::Log(format!(
        "Using FFmpeg: {}",
        ffmpeg.ffmpeg.display()
    )));

    let model_dir = ensure_model(
        &config.model,
        on_event,
        cancel,
        environment.model_directory.as_deref(),
    )?;
    if cancel.is_cancelled() {
        return Err(cancelled(*position).into());
    }

    let mut worker = WorkerClient::spawn(worker_command, cancel)?;
    load_model(
        &mut worker,
        &model_dir,
        config,
        &environment.system,
        on_event,
        cancel,
    )?;
    if cancel.is_cancelled() {
        return Err(cancelled(*position).into());
    }

    let duration = probe_duration(&ffmpeg, &config.audio);
    on_event(Event::StageStarted {
        stage: Stage::Transcribe,
        message: "Transcribing...".into(),
    });
    let progress = |completed: f64| Event::Progress {
        stage: Stage::Transcribe,
        completed,
        total: duration,
        unit: "s".into(),
    };
    on_event(progress(0.0));
    let start = Instant::now();

    let mut language = config.language.clone();
    if let Some(language) = &language {
        on_event(Event::LanguageDetected {
            language: language.clone(),
            probability: 1.0,
        });
    }

    let mut prompt: Option<String> = None;
    let mut file = File::create(&output)?;
    let blocks = stream_audio(&config.audio, &ffmpeg, 30.0)?;
    let windows = iter_windows(
        blocks,
        config.chunk_minutes * 60.0,
        SEARCH_SECONDS,
        find_cut,
    );

    for window in windows {
        let window: AudioWindow = window?;
        if cancel.is_cancelled() {
            return Err(cancelled(*position).into());
        }

        let mut texts: Vec<String> = Vec::new();
        let mut detected: Option<(String, f64)> = None;
        let mut write_error: Option<std::io::Error> = None;

        let was_cancelled = worker.transcribe(
            &window.samples,
            language.as_deref(),
            BEAM_SIZE,
            prompt.as_deref(),
            &mut |update| match update {
                Update::Language {
                    language,
                    probability,
                } => {
                    detected = Some((language, probability));
                }
                Update::Segment { start, end, text } => {
                    let segment_start = window.start() + start;
                    let segment_end = window.start() + end.min(window.duration());
                    if write_error.is_none() {
                        let line = format!("[{}] {text}\n", format_timestamp(segment_start));
                        write_error = file
                            .write_all(line.as_bytes())
                            .and_then(|()| file.flush())
                            .err();
                    }
                    texts.push(text.clone());
                    on_event(Event::SegmentReady {
                        start: segment_start,
                        end: segment_end,
                        text,
                    });
                    on_event(progress(segment_end));
                    *position = segment_end;
                }
            },
            cancel,
        );
        let was_cancelled = match was_cancelled {
            Err(WorkerError::Cancelled) => true,
            other => other?,
        };
        if let Some(error) = write_error {
            return Err(error.into());
        }
        if was_cancelled || cancel.is_cancelled() {
            return Err(cancelled(*position).into());
        }

        // Lock the language once speech is found, so windows stay consistent.
        if language.is_none()
            && !texts.is_empty()
            && let Some((found, probability)) = detected
        {
            on_event(Event::LanguageDetected {
                language: found.clone(),
                probability,
            });
            language = Some(found);
        }

        // Carry context across the cut so sentences continue naturally.
        if !texts.is_empty() {
            prompt = Some(prompt_tail(&texts));
        }

        *position = window.start() + window.duration();
        on_event(progress(*position));
    }

    on_event(Event::StageFinished {
        stage: Stage::Transcribe,
        elapsed: start.elapsed().as_secs_f64(),
    });

    let summary = JobSummary {
        output: output.clone(),
        audio_duration: *position,
        elapsed: job_start.elapsed().as_secs_f64(),
    };
    on_event(Event::Finished {
        output,
        audio_duration: summary.audio_duration,
        elapsed: summary.elapsed,
    });
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_prompt_is_the_last_characters_of_the_text() {
        let texts = vec!["a".repeat(150), "b".repeat(150)];
        let prompt = prompt_tail(&texts);
        assert_eq!(prompt.chars().count(), PROMPT_CHARACTERS);
        assert!(prompt.ends_with('b'));
        assert_eq!(prompt_tail(&["hi".into(), "there".into()]), "hi there");
    }

    #[test]
    fn a_failure_after_cancelling_is_reported_as_a_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.txt");
        std::fs::write(&output, "[00:00] hi\n").unwrap();
        let failure = || Err(JobError::Worker(WorkerError::Exited("killed".into())));

        let cancel = CancelToken::new();
        let untouched = count_failure_after_cancel_as_cancelled(failure(), &cancel, &output, 5.0);
        assert!(matches!(untouched, Err(JobError::Worker(_))));

        cancel.cancel();
        let Err(JobError::Cancelled(cancelled)) =
            count_failure_after_cancel_as_cancelled(failure(), &cancel, &output, 5.0)
        else {
            panic!("expected a cancellation");
        };
        assert_eq!((cancelled.output, cancelled.position), (Some(output), 5.0));
    }

    #[test]
    fn the_prompt_never_splits_a_multibyte_character() {
        let texts = vec!["سلام ".repeat(100)];
        assert_eq!(prompt_tail(&texts).chars().count(), PROMPT_CHARACTERS);
    }

    use crate::config::DeviceChoice;
    use crate::events::Event;
    use crate::ffmpeg::find_ffmpeg;
    use crate::models::REQUIRED_FILES;
    use crate::platform::detect_cpu;
    use crate::platform::{Os, cuda_option};
    use crate::test_support::{FakeWorker, require, write_sine_wav};

    /// Everything a job test needs, or `None` when FFmpeg or Python is missing.
    struct Setup {
        _dir: tempfile::TempDir,
        audio: PathBuf,
        fake: FakeWorker,
        environment: JobEnvironment,
    }

    fn setup(seconds: u32, language: &str) -> Option<Setup> {
        require("ffmpeg", find_ffmpeg())?;
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeWorker::new(dir.path(), language)?;

        let models = dir.path().join("models");
        std::fs::create_dir_all(models.join("small")).unwrap();
        for name in REQUIRED_FILES {
            std::fs::write(models.join("small").join(name), "x").unwrap();
        }
        let audio = dir.path().join("talk.wav");
        write_sine_wav(&audio, seconds, 440.0);

        let system = SystemInfo {
            os: Os::Linux,
            os_label: "Linux test".into(),
            cpu_threads: 2,
            devices: vec![cuda_option(Os::Linux, Ok(0), &[]), detect_cpu()],
        };
        let environment = JobEnvironment {
            system,
            worker: Some(fake.command.clone()),
            model_directory: Some(models),
        };
        Some(Setup {
            _dir: dir,
            audio,
            fake,
            environment,
        })
    }

    fn config(setup: &Setup, chunk_minutes: f64) -> JobConfig {
        let mut config = JobConfig::new(&setup.audio);
        config.chunk_minutes = chunk_minutes;
        config
    }

    fn run(setup: &Setup, config: &JobConfig) -> (Result<JobSummary, JobError>, Vec<Event>) {
        let mut events = Vec::new();
        let result = run_job(
            config,
            &setup.environment,
            &mut |e| events.push(e),
            &CancelToken::new(),
        );
        (result, events)
    }

    #[test]
    fn transcribes_window_by_window_with_bounded_memory() {
        let Some(setup) = setup(180, "en") else {
            return;
        };
        let (result, events) = run(&setup, &config(&setup, 1.0));
        let summary = result.unwrap();

        // Windows are at most a minute, so a three-minute file needs several.
        let calls = setup.fake.transcriptions();
        assert!(calls.len() >= 3, "{}", calls.len());
        assert!(
            calls
                .iter()
                .all(|call| call["samples"].as_u64().unwrap() <= 60 * 16_000)
        );
        assert!(
            (summary.audio_duration - 180.0).abs() < 0.1,
            "{}",
            summary.audio_duration
        );

        // Timestamps are shifted by each window's start.
        let starts: Vec<f64> = events
            .iter()
            .filter_map(|e| match e {
                Event::SegmentReady { start, .. } => Some(*start),
                _ => None,
            })
            .collect();
        assert!(starts.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(*starts.last().unwrap() >= 120.0);

        let text = std::fs::read_to_string(setup.audio.with_extension("txt")).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines[0], "[00:00] window 1 at 0");
        assert!(lines.last().unwrap().starts_with("[02:"), "{lines:?}");
        assert!(matches!(events.last(), Some(Event::Finished { .. })));
    }

    #[test]
    fn the_language_is_detected_once_then_locked() {
        let Some(setup) = setup(180, "fa") else {
            return;
        };
        let (result, events) = run(&setup, &config(&setup, 1.0));
        result.unwrap();

        let languages: Vec<Option<String>> = setup
            .fake
            .transcriptions()
            .iter()
            .map(|call| call["language"].as_str().map(str::to_string))
            .collect();
        assert_eq!(languages[0], None);
        assert!(
            languages[1..].iter().all(|l| l.as_deref() == Some("fa")),
            "{languages:?}"
        );
        let detected: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                Event::LanguageDetected { language, .. } => Some(language.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(detected, ["fa"]);
    }

    #[test]
    fn the_previous_text_is_passed_as_context() {
        let Some(setup) = setup(180, "en") else {
            return;
        };
        run(&setup, &config(&setup, 1.0)).0.unwrap();

        let calls = setup.fake.transcriptions();
        assert!(calls[0]["initial_prompt"].is_null());
        assert!(
            calls[1]["initial_prompt"]
                .as_str()
                .unwrap()
                .contains("window 1")
        );
    }

    #[test]
    fn progress_reaches_the_total() {
        let Some(setup) = setup(60, "en") else { return };
        let mut job = config(&setup, 1.0);
        job.language = Some("en".into());
        let (result, events) = run(&setup, &job);
        result.unwrap();

        let last = events
            .iter()
            .rev()
            .find_map(|e| match e {
                Event::Progress {
                    stage: Stage::Transcribe,
                    completed,
                    total,
                    ..
                } => Some((*completed, *total)),
                _ => None,
            })
            .unwrap();
        assert!((last.0 - last.1.unwrap()).abs() < 0.1, "{last:?}");
    }

    #[test]
    fn cancelling_stops_the_job_and_keeps_the_partial_output() {
        let Some(setup) = setup(180, "en") else {
            return;
        };
        let cancel = CancelToken::new();
        let trigger = cancel.clone();

        let result = run_job(
            &config(&setup, 1.0),
            &setup.environment,
            &mut |event| {
                if let Event::SegmentReady { text, .. } = &event
                    && text.contains("window 2")
                {
                    trigger.cancel();
                }
            },
            &cancel,
        );

        let JobError::Cancelled(cancelled) = result.unwrap_err() else {
            panic!("not cancelled")
        };
        assert_eq!(setup.fake.transcriptions().len(), 2);
        assert_eq!(cancelled.output, Some(setup.audio.with_extension("txt")));
        assert!(cancelled.position > 30.0, "{}", cancelled.position);
        let text = std::fs::read_to_string(cancelled.output.unwrap()).unwrap();
        assert!(text.contains("window 2"));
    }

    #[test]
    fn cancelling_before_the_start_never_starts_the_worker() {
        let Some(setup) = setup(30, "en") else { return };
        let cancel = CancelToken::new();
        cancel.cancel();

        let result = run_job(
            &config(&setup, 1.0),
            &setup.environment,
            &mut |_| {},
            &cancel,
        );
        assert!(matches!(result, Err(JobError::Cancelled(_))));
        assert!(setup.fake.calls().is_empty());
    }

    #[test]
    fn an_unavailable_gpu_is_a_clean_error_before_anything_starts() {
        let Some(setup) = setup(1, "en") else { return };
        let mut job = config(&setup, 1.0);
        job.device = DeviceChoice::Cuda;
        let (result, _) = run(&setup, &job);
        assert!(result.unwrap_err().to_string().contains("No NVIDIA GPU"));
        assert!(setup.fake.calls().is_empty());
    }

    #[test]
    fn a_missing_worker_environment_is_reported() {
        let Some(mut setup) = setup(1, "en") else {
            return;
        };
        setup.environment.worker = None;
        let (result, _) = run(&setup, &config(&setup, 1.0));
        assert!(matches!(result, Err(JobError::WorkerNotFound)));
    }

    #[test]
    fn a_file_ffmpeg_cannot_decode_is_reported_by_name() {
        let Some(setup) = setup(1, "en") else { return };
        let broken = setup.audio.with_file_name("broken.mp3");
        std::fs::write(&broken, b"not audio").unwrap();
        let mut job = JobConfig::new(&broken);
        job.chunk_minutes = 1.0;
        let (result, _) = run(&setup, &job);
        assert!(result.unwrap_err().to_string().contains("broken.mp3"));
    }
}
