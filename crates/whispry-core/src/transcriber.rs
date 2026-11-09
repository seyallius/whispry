//! Run a transcription job and report progress through events.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::audio::{AudioDecodeError, stream_audio};
use crate::cancel::{CancelToken, JobCancelled};
use crate::chunking::{AudioWindow, find_cut, iter_windows};
use crate::config::{BEAM_SIZE, JobConfig, SAMPLE_RATE};
use crate::engine::{EngineError, EngineOpener, Update, load_model, real_opener, split_for_engine};
use crate::events::{Event, EventSink, Stage};
use crate::ffmpeg::{FFmpegNotFound, probe_duration, require_ffmpeg};
use crate::models::{ModelError, ensure_model};
use crate::platform::{DeviceUnavailable, SystemInfo, detect_system, resolve_device};
use crate::timestamps::format_timestamp;

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
    Engine(EngineError),
    #[error(transparent)]
    Audio(#[from] AudioDecodeError),
    #[error("Could not write the transcription: {0}")]
    Output(#[from] std::io::Error),
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

impl From<EngineError> for JobError {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::Cancelled => JobCancelled {
                output: None,
                position: 0.0,
            }
            .into(),
            other => JobError::Engine(other),
        }
    }
}

/// What a job runs on: the machine's devices and the speech engine.
#[derive(Clone)]
pub struct JobEnvironment {
    pub system: SystemInfo,
    /// Loads the model into a speech engine ([`real_opener`] in the program).
    pub engine: EngineOpener,
    /// Where models are cached; `None` uses the per-user default.
    pub model_directory: Option<PathBuf>,
}

impl std::fmt::Debug for JobEnvironment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobEnvironment")
            .field("system", &self.system)
            .field("model_directory", &self.model_directory)
            .finish_non_exhaustive()
    }
}

impl JobEnvironment {
    /// Inspect this machine and use the real speech engine.
    pub fn detect() -> Self {
        JobEnvironment {
            system: detect_system(&crate::cuda_libs::installed_dirs()),
            engine: real_opener(),
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

/// Transcribe `config.audio` and write timestamped text to the output file.
///
/// Audio is decoded one window (`config.chunk_minutes`) at a time, so memory
/// use does not grow with the length of the file. Each window is fed to the
/// engine in pieces of at most 30 s, cut in quiet moments.
///
/// Returns [`JobError::Cancelled`] soon after `cancel` is triggered (checked
/// between segments and pieces); text transcribed so far stays in the output
/// file.
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

/// Once the user has cancelled, the engine or FFmpeg may stop in ways that
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

    let (mut engine, _) = load_model(
        &environment.engine,
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

        for piece in split_for_engine(&window.samples) {
            if cancel.is_cancelled() {
                return Err(cancelled(*position).into());
            }
            let piece_start = window.start() + piece.start as f64 / f64::from(SAMPLE_RATE);
            let piece_seconds = piece.len() as f64 / f64::from(SAMPLE_RATE);

            let mut heard_speech = false;
            let mut detected: Option<(String, f64)> = None;
            let mut write_error: Option<std::io::Error> = None;

            let was_cancelled = engine.transcribe(
                &window.samples[piece],
                language.as_deref(),
                BEAM_SIZE,
                &mut |update| match update {
                    Update::Language {
                        language,
                        probability,
                    } => {
                        detected = Some((language, probability));
                    }
                    Update::Segment { start, end, text } => {
                        let segment_start = piece_start + start;
                        let segment_end = piece_start + end.min(piece_seconds);
                        if write_error.is_none() {
                            let line = format!("[{}] {text}\n", format_timestamp(segment_start));
                            write_error = file
                                .write_all(line.as_bytes())
                                .and_then(|()| file.flush())
                                .err();
                        }
                        heard_speech = true;
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
                Err(EngineError::Cancelled) => true,
                other => other?,
            };
            if let Some(error) = write_error {
                return Err(error.into());
            }
            if was_cancelled || cancel.is_cancelled() {
                return Err(cancelled(*position).into());
            }

            // Lock the language once speech is found, so pieces stay consistent.
            if language.is_none()
                && heard_speech
                && let Some((found, probability)) = detected
            {
                on_event(Event::LanguageDetected {
                    language: found.clone(),
                    probability,
                });
                language = Some(found);
            }

            *position = piece_start + piece_seconds;
            on_event(progress(*position));
        }
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
    fn a_failure_after_cancelling_is_reported_as_a_cancellation() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("out.txt");
        std::fs::write(&output, "[00:00] hi\n").unwrap();
        let failure = || Err(JobError::Engine(EngineError::Transcribe("killed".into())));

        let cancel = CancelToken::new();
        let untouched = count_failure_after_cancel_as_cancelled(failure(), &cancel, &output, 5.0);
        assert!(matches!(untouched, Err(JobError::Engine(_))));

        cancel.cancel();
        let Err(JobError::Cancelled(cancelled)) =
            count_failure_after_cancel_as_cancelled(failure(), &cancel, &output, 5.0)
        else {
            panic!("expected a cancellation");
        };
        assert_eq!((cancelled.output, cancelled.position), (Some(output), 5.0));
    }

    use crate::config::DeviceChoice;
    use crate::events::Event;
    use crate::ffmpeg::find_ffmpeg;
    use crate::models::REQUIRED_FILES;
    use crate::platform::detect_cpu;
    use crate::platform::{Os, cuda_option};
    use crate::test_support::{FakeCall, FakeEngines, require, write_sine_wav};

    /// Everything a job test needs, or `None` when FFmpeg is missing.
    struct Setup {
        _dir: tempfile::TempDir,
        audio: PathBuf,
        fake: FakeEngines,
        environment: JobEnvironment,
    }

    /// The samples, language and beam size of each engine call.
    fn calls(setup: &Setup) -> Vec<(usize, Option<String>, u32)> {
        setup
            .fake
            .transcriptions()
            .into_iter()
            .map(|call| match call {
                FakeCall::Transcribe {
                    samples,
                    language,
                    beam_size,
                } => (samples, language, beam_size),
                FakeCall::Load { .. } => unreachable!(),
            })
            .collect()
    }

    fn setup(seconds: u32, language: &str) -> Option<Setup> {
        require("ffmpeg", find_ffmpeg())?;
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeEngines::new(language);

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
            devices: vec![cuda_option(Os::Linux, Ok(0), &[], true), detect_cpu()],
        };
        let environment = JobEnvironment {
            system,
            engine: fake.opener(),
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

        // The engine never gets more than 30 s at once, and a three-minute
        // file needs several pieces.
        let calls = calls(&setup);
        assert!(calls.len() >= 6, "{}", calls.len());
        assert!(calls.iter().all(|(samples, ..)| *samples <= 30 * 16_000));
        assert!(
            (summary.audio_duration - 180.0).abs() < 0.1,
            "{}",
            summary.audio_duration
        );

        // Timestamps are shifted by each piece's start.
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
    fn the_beam_size_is_passed_to_the_engine() {
        let Some(setup) = setup(10, "en") else { return };
        run(&setup, &config(&setup, 1.0)).0.unwrap();
        assert!(calls(&setup).iter().all(|(_, _, beam)| *beam == BEAM_SIZE));
    }

    #[test]
    fn the_language_is_detected_once_then_locked() {
        let Some(setup) = setup(180, "fa") else {
            return;
        };
        let (result, events) = run(&setup, &config(&setup, 1.0));
        result.unwrap();

        let languages: Vec<Option<String>> = calls(&setup).into_iter().map(|(_, l, _)| l).collect();
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
    fn cancelling_before_the_start_never_loads_the_engine() {
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
