//! The speech engine: CTranslate2 running Whisper inside this process.
//!
//! Everything the rest of the program needs is the [`SpeechEngine`] trait and
//! an [`EngineOpener`] that loads a model into one. [`open_ct2`] is the real
//! opener; tests use a fake one.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::cancel::CancelToken;
use crate::chunking::find_cut;
use crate::config::{DeviceChoice, JobConfig, SAMPLE_RATE};
use crate::events::{Event, EventSink, Stage};
use crate::platform::{Device, DeviceOption, SystemInfo, resolve_device};

/// The most audio the engine is given in one call. Whisper looks at 30 s at a
/// time, and a call cannot be interrupted, so this is also how long a
/// cancellation can take to be noticed.
pub const MAX_PIECE_SECONDS: f64 = 30.0;

/// How far back from the 30 s limit the search for a quiet cut point goes.
const PIECE_SEARCH_SECONDS: f64 = 8.0;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("Could not load the model on {device}: {message}")]
    Load { device: String, message: String },
    #[error("Transcription failed: {0}")]
    Transcribe(String),
    #[error("Transcription cancelled")]
    Cancelled,
}

/// What a call to the engine produces while it runs.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    Language {
        language: String,
        probability: f64,
    },
    /// Times are relative to the start of the audio given to the engine.
    Segment {
        start: f64,
        end: f64,
        text: String,
    },
}

/// A loaded speech model.
pub trait SpeechEngine {
    /// Transcribe 16 kHz mono audio, passing each update to `on_update`.
    /// Returns whether the call was cancelled before it finished.
    ///
    /// Callers give at most [`MAX_PIECE_SECONDS`] of audio per call (see
    /// [`split_for_engine`]); `cancel` is only checked between updates.
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        beam_size: u32,
        on_update: &mut dyn FnMut(Update),
        cancel: &CancelToken,
    ) -> Result<bool, EngineError>;
}

/// Loads the model in a folder onto a device with one compute type and a
/// number of CPU threads.
pub type EngineOpener = Arc<
    dyn Fn(&Path, Device, &str, usize) -> Result<Box<dyn SpeechEngine + Send>, EngineError>
        + Send
        + Sync,
>;

/// The opener that runs the real CTranslate2 engine.
pub fn real_opener() -> EngineOpener {
    Arc::new(open_ct2)
}

fn compute_type_of(name: &str) -> Option<ct2rs::ComputeType> {
    use ct2rs::ComputeType;
    Some(match name {
        "default" => ComputeType::DEFAULT,
        "auto" => ComputeType::AUTO,
        "float32" => ComputeType::FLOAT32,
        "int8" => ComputeType::INT8,
        "int8_float32" => ComputeType::INT8_FLOAT32,
        "int8_float16" => ComputeType::INT8_FLOAT16,
        "int8_bfloat16" => ComputeType::INT8_BFLOAT16,
        "int16" => ComputeType::INT16,
        "float16" => ComputeType::FLOAT16,
        "bfloat16" => ComputeType::BFLOAT16,
        _ => return None,
    })
}

/// Load a Whisper model with CTranslate2.
pub fn open_ct2(
    model_dir: &Path,
    device: Device,
    compute_type: &str,
    threads: usize,
) -> Result<Box<dyn SpeechEngine + Send>, EngineError> {
    let load_error = |message: String| EngineError::Load {
        device: device.as_str().into(),
        message,
    };
    let compute = compute_type_of(compute_type)
        .ok_or_else(|| load_error(format!("unknown compute type {compute_type}")))?;
    if device == Device::Cuda && !crate::platform::CUDA_BUILD {
        return Err(load_error(
            "this build of Whispry has no GPU support".to_string(),
        ));
    }

    let config = ct2rs::Config {
        device: match device {
            Device::Cpu => ct2rs::Device::CPU,
            Device::Cuda => ct2rs::Device::CUDA,
        },
        compute_type: compute,
        num_threads_per_replica: threads,
        ..Default::default()
    };
    let whisper =
        ct2rs::Whisper::new(model_dir, config).map_err(|error| load_error(format!("{error:#}")))?;
    Ok(Box::new(Ct2Engine { whisper }))
}

struct Ct2Engine {
    whisper: ct2rs::Whisper,
}

impl SpeechEngine for Ct2Engine {
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        beam_size: u32,
        on_update: &mut dyn FnMut(Update),
        cancel: &CancelToken,
    ) -> Result<bool, EngineError> {
        if cancel.is_cancelled() {
            return Ok(true);
        }
        let options = ct2rs::WhisperOptions {
            beam_size: beam_size as usize,
            ..Default::default()
        };
        let chunks = self
            .whisper
            .generate(samples, language, true, &options)
            .map_err(|error| EngineError::Transcribe(format!("{error:#}")))?;

        let rate = self.whisper.sampling_rate().max(1) as f64;
        let chunk_seconds = self.whisper.n_samples() as f64 / rate;
        let total_seconds = samples.len() as f64 / rate;
        for (index, decoded) in chunks.iter().enumerate() {
            let offset = index as f64 * chunk_seconds;
            let length = (total_seconds - offset).clamp(0.0, chunk_seconds);
            for segment in parse_timestamped(decoded, offset, length) {
                on_update(Update::Segment {
                    start: segment.start,
                    end: segment.end,
                    text: segment.text,
                });
                if cancel.is_cancelled() {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TimedText {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

enum Piece<'a> {
    Timestamp(f64),
    Text(&'a str),
}

/// Split decoded Whisper output into timestamp tokens and text. Other special
/// tokens (`<|endoftext|>`, `<|en|>`, ...) are dropped.
fn pieces(decoded: &str) -> Vec<Piece<'_>> {
    let mut result = Vec::new();
    let mut rest = decoded;
    while !rest.is_empty() {
        let Some(open) = rest.find("<|") else {
            result.push(Piece::Text(rest));
            break;
        };
        let Some(length) = rest[open + 2..].find("|>") else {
            result.push(Piece::Text(rest));
            break;
        };
        if open > 0 {
            result.push(Piece::Text(&rest[..open]));
        }
        let inner = &rest[open + 2..open + 2 + length];
        if inner.starts_with(|c: char| c.is_ascii_digit())
            && let Ok(seconds) = inner.parse::<f64>()
        {
            result.push(Piece::Timestamp(seconds));
        }
        rest = &rest[open + 2 + length + 2..];
    }
    result
}

/// Turn the text of one 30 s chunk, with timestamp tokens such as
/// `<|0.00|> Hello<|3.20|>`, into segments.
///
/// `offset` is where the chunk starts and `length` how long it is, both in
/// seconds. A last segment with no end timestamp ends where the chunk does.
pub fn parse_timestamped(decoded: &str, offset: f64, length: f64) -> Vec<TimedText> {
    let mut segments = Vec::new();
    let mut open: Option<(f64, String)> = None;
    let mut last_end = 0.0;

    let finish = |start: f64, text: &str, end: f64, segments: &mut Vec<TimedText>| {
        let text = text.trim();
        if text.is_empty() {
            return;
        }
        let start = start.clamp(0.0, length);
        let end = end.clamp(start, length);
        segments.push(TimedText {
            start: offset + start,
            end: offset + end,
            text: text.to_string(),
        });
    };

    for piece in pieces(decoded) {
        match piece {
            Piece::Text(text) => open
                .get_or_insert((last_end, String::new()))
                .1
                .push_str(text),
            Piece::Timestamp(time) => match open.take() {
                Some((start, text)) if !text.trim().is_empty() => {
                    finish(start, &text, time, &mut segments);
                    last_end = time;
                }
                _ => open = Some((time, String::new())),
            },
        }
    }
    if let Some((start, text)) = open {
        finish(start, &text, length, &mut segments);
    }
    segments
}

/// Split audio into pieces of at most [`MAX_PIECE_SECONDS`], cutting each in a
/// quiet moment near its end so words are not split where it can be avoided.
pub fn split_for_engine(samples: &[f32]) -> Vec<Range<usize>> {
    let max = (MAX_PIECE_SECONDS * f64::from(SAMPLE_RATE)) as usize;
    let search = (PIECE_SEARCH_SECONDS * f64::from(SAMPLE_RATE)) as usize;
    let mut pieces = Vec::new();
    let mut start = 0;
    while samples.len() - start > max {
        let region = start + max - search..start + max;
        let cut = region.start + find_cut(&samples[region.clone()]);
        let cut = cut.clamp(region.start + 1, region.end);
        pieces.push(start..cut);
        start = cut;
    }
    if start < samples.len() {
        pieces.push(start..samples.len());
    }
    pieces
}

/// Where the model ended up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedModel {
    pub device: Device,
    pub compute_type: String,
}

/// Try each compute type in turn; the first that loads wins. When all fail,
/// the last error is returned (the plainest compute type, so its failure says
/// most about the real cause).
fn load_on(
    opener: &EngineOpener,
    model_dir: &Path,
    device: &DeviceOption,
    chosen: Option<&str>,
    threads: usize,
    cancel: &CancelToken,
) -> Result<(Box<dyn SpeechEngine + Send>, LoadedModel), EngineError> {
    let preferred: Vec<&str> = match chosen {
        Some(chosen) => vec![chosen],
        None => device.compute_types.clone(),
    };
    let mut last = EngineError::Load {
        device: device.device.as_str().into(),
        message: "no compute type to try".into(),
    };
    for compute_type in preferred {
        if cancel.is_cancelled() {
            return Err(EngineError::Cancelled);
        }
        match opener(model_dir, device.device, compute_type, threads) {
            Ok(engine) => {
                let loaded = LoadedModel {
                    device: device.device,
                    compute_type: compute_type.to_string(),
                };
                return Ok((engine, loaded));
            }
            Err(error) => last = error,
        }
    }
    Err(last)
}

/// Load the model on the requested device. When the device is `auto` and the
/// GPU fails to load it, the CPU is used instead.
///
/// Unless the user picked a compute type, the best one the device loads wins.
pub fn load_model(
    opener: &EngineOpener,
    model_dir: &Path,
    config: &JobConfig,
    system: &SystemInfo,
    on_event: EventSink,
    cancel: &CancelToken,
) -> Result<(Box<dyn SpeechEngine + Send>, LoadedModel), EngineError> {
    let (option, _) = resolve_device(config.device, system, config.compute_type.as_deref())
        .map_err(|error| EngineError::Load {
            device: config.device.to_string(),
            message: error.to_string(),
        })?;

    on_event(Event::StageStarted {
        stage: Stage::Model,
        message: format!("Loading model on {}...", option.label),
    });
    let start = Instant::now();
    let chosen = config.compute_type.as_deref();

    let (engine, loaded) = match load_on(
        opener,
        model_dir,
        &option,
        chosen,
        system.cpu_threads,
        cancel,
    ) {
        Ok(loaded) => loaded,
        Err(error @ EngineError::Load { .. })
            if config.device == DeviceChoice::Auto && option.device != Device::Cpu =>
        {
            on_event(Event::Log(format!("{error}. Falling back to CPU.")));
            let (cpu, _) = resolve_device(DeviceChoice::Cpu, system, None).map_err(|error| {
                EngineError::Load {
                    device: "cpu".into(),
                    message: error.to_string(),
                }
            })?;
            load_on(opener, model_dir, &cpu, None, system.cpu_threads, cancel)?
        }
        Err(error) => return Err(error),
    };

    on_event(Event::Log(format!(
        "Running on {} with compute type {}",
        loaded.device.as_str(),
        loaded.compute_type
    )));
    on_event(Event::StageFinished {
        stage: Stage::Model,
        elapsed: start.elapsed().as_secs_f64(),
    });
    Ok((engine, loaded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::platform::{Os, cuda_option, detect_cpu};
    use crate::test_support::{FakeCall, FakeEngines};

    const RATE: usize = SAMPLE_RATE as usize;

    fn segment(start: f64, end: f64, text: &str) -> TimedText {
        TimedText {
            start,
            end,
            text: text.into(),
        }
    }

    #[test]
    fn parses_a_chunk_with_several_segments() {
        let decoded = "<|0.00|> Hello there.<|2.40|><|2.40|> How are you?<|5.00|><|endoftext|>";
        assert_eq!(
            parse_timestamped(decoded, 0.0, 30.0),
            vec![
                segment(0.0, 2.4, "Hello there."),
                segment(2.4, 5.0, "How are you?")
            ]
        );
    }

    #[test]
    fn parses_persian_text() {
        let decoded = "<|0.00|> سلام، حال شما چطور است؟<|3.20|>";
        assert_eq!(
            parse_timestamped(decoded, 0.0, 30.0),
            vec![segment(0.0, 3.2, "سلام، حال شما چطور است؟")]
        );
    }

    #[test]
    fn a_single_timestamp_can_end_one_segment_and_start_the_next() {
        let decoded = "<|0.00|> one<|1.00|> two<|2.00|>";
        assert_eq!(
            parse_timestamped(decoded, 0.0, 30.0),
            vec![segment(0.0, 1.0, "one"), segment(1.0, 2.0, "two")]
        );
    }

    #[test]
    fn an_empty_chunk_has_no_segments() {
        assert!(parse_timestamped("", 0.0, 30.0).is_empty());
        assert!(parse_timestamped("<|endoftext|>", 0.0, 30.0).is_empty());
        assert!(parse_timestamped("<|0.00|><|30.00|>", 0.0, 30.0).is_empty());
        assert!(parse_timestamped("<|0.00|>  <|2.00|>", 0.0, 30.0).is_empty());
    }

    #[test]
    fn a_missing_final_timestamp_ends_at_the_end_of_the_chunk() {
        assert_eq!(
            parse_timestamped("<|0.00|> done<|1.50|><|1.50|> cut off", 0.0, 12.0),
            vec![segment(0.0, 1.5, "done"), segment(1.5, 12.0, "cut off")]
        );
        assert_eq!(
            parse_timestamped("no timestamps at all", 0.0, 7.0),
            vec![segment(0.0, 7.0, "no timestamps at all")]
        );
    }

    #[test]
    fn the_chunk_offset_is_added_and_times_stay_inside_the_chunk() {
        assert_eq!(
            parse_timestamped("<|0.00|> late<|40.00|>", 60.0, 10.0),
            vec![segment(60.0, 70.0, "late")]
        );
        assert_eq!(
            parse_timestamped("<|1.00|> x<|2.00|>", 30.0, 30.0),
            vec![segment(31.0, 32.0, "x")]
        );
    }

    #[test]
    fn other_special_tokens_are_dropped_but_angle_brackets_in_text_stay() {
        assert_eq!(
            parse_timestamped("<|en|><|0.00|> a < b <|1.00|>", 0.0, 30.0),
            vec![segment(0.0, 1.0, "a < b")]
        );
    }

    fn noise(seconds: f64) -> Vec<f32> {
        vec![0.1; (seconds * RATE as f64) as usize]
    }

    #[test]
    fn short_audio_is_one_piece() {
        assert_eq!(split_for_engine(&noise(30.0)), vec![0..30 * RATE]);
        assert_eq!(split_for_engine(&noise(1.0)), vec![0..RATE]);
        assert!(split_for_engine(&[]).is_empty());
    }

    #[test]
    fn long_audio_is_split_into_pieces_of_at_most_thirty_seconds() {
        let samples = noise(200.0);
        let pieces = split_for_engine(&samples);
        assert!(pieces.len() >= 7, "{}", pieces.len());
        assert!(pieces.iter().all(|piece| piece.len() <= 30 * RATE));
        assert_eq!(pieces[0].start, 0);
        assert_eq!(pieces.last().unwrap().end, samples.len());
        assert!(pieces.windows(2).all(|pair| pair[0].end == pair[1].start));
    }

    #[test]
    fn pieces_are_cut_in_a_pause() {
        // Loud audio with a 400 ms pause at 27 s.
        let mut samples: Vec<f32> = (0..45 * RATE)
            .map(|n| if n % 2 == 0 { 0.3 } else { -0.3 })
            .collect();
        let pause = 27 * RATE..27 * RATE + RATE * 2 / 5;
        samples[pause.clone()].fill(0.0);
        let pieces = split_for_engine(&samples);
        assert_eq!(pieces.len(), 2);
        assert!(pause.contains(&pieces[0].end), "{pieces:?}");
    }

    fn system_with_gpu() -> SystemInfo {
        SystemInfo {
            os: Os::Linux,
            os_label: "Linux".into(),
            cpu_threads: 2,
            devices: vec![cuda_option(Os::Linux, Ok(1), &[], true), detect_cpu()],
        }
    }

    fn job(device: DeviceChoice) -> JobConfig {
        let mut config = JobConfig::new("a.wav");
        config.device = device;
        config
    }

    #[test]
    fn the_first_compute_type_that_loads_wins() {
        let fake = FakeEngines::new("en");
        let (_, loaded) = load_model(
            &fake.opener(),
            Path::new("model"),
            &job(DeviceChoice::Cpu),
            &system_with_gpu(),
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        assert_eq!(loaded.compute_type, "int8");
        assert_eq!(fake.loads(), ["cpu int8"]);
    }

    #[test]
    fn a_compute_type_that_does_not_load_makes_the_next_one_run() {
        let fake = FakeEngines::new("en");
        let mut system = system_with_gpu();
        system.devices[1].compute_types = vec!["float16", "int8_float32", "float32"];
        let (_, loaded) = load_model(
            &fake.opener(),
            Path::new("model"),
            &job(DeviceChoice::Cpu),
            &system,
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap();
        // The fake supports int8 and float32 only.
        assert_eq!(loaded.compute_type, "float32");
        assert_eq!(
            fake.loads(),
            ["cpu float16", "cpu int8_float32", "cpu float32"]
        );
    }

    #[test]
    fn a_compute_type_nothing_supports_is_a_load_error() {
        let fake = FakeEngines::new("en");
        let mut system = system_with_gpu();
        system.devices[1].compute_types = vec!["float16"];
        let mut config = job(DeviceChoice::Cpu);
        config.compute_type = Some("float16".into());
        let error = load_model(
            &fake.opener(),
            Path::new("model"),
            &config,
            &system,
            &mut |_| {},
            &CancelToken::new(),
        )
        .err()
        .unwrap();
        assert!(matches!(error, EngineError::Load { message, .. } if message.contains("float16")));
    }

    #[test]
    fn auto_falls_back_to_the_cpu_when_the_gpu_fails_to_load() {
        let fake = FakeEngines::new("en");
        let mut events = Vec::new();

        let (_, loaded) = load_model(
            &fake.opener(),
            Path::new("model"),
            &job(DeviceChoice::Auto),
            &system_with_gpu(),
            &mut |e| events.push(e),
            &CancelToken::new(),
        )
        .unwrap();

        assert_eq!(loaded.device, Device::Cpu);
        let loads = fake.loads();
        assert!(loads[0].starts_with("cuda "), "{loads:?}");
        assert_eq!(loads.last().unwrap(), "cpu int8");
        assert!(
            events
                .iter()
                .any(|e| matches!(e, Event::Log(t) if t.contains("Falling back to CPU")))
        );
        assert!(matches!(
            events.last(),
            Some(Event::StageFinished {
                stage: Stage::Model,
                ..
            })
        ));
    }

    #[test]
    fn an_explicit_gpu_request_does_not_fall_back() {
        let fake = FakeEngines::new("en");
        let error = load_model(
            &fake.opener(),
            Path::new("model"),
            &job(DeviceChoice::Cuda),
            &system_with_gpu(),
            &mut |_| {},
            &CancelToken::new(),
        )
        .err()
        .unwrap();
        assert!(
            error
                .to_string()
                .starts_with("Could not load the model on cuda"),
            "{error}"
        );
        assert!(fake.loads().iter().all(|load| load.starts_with("cuda ")));
    }

    #[test]
    fn cancelling_while_loading_stops_before_the_next_attempt() {
        let fake = FakeEngines::new("en");
        let cancel = CancelToken::new();
        cancel.cancel();
        let error = load_model(
            &fake.opener(),
            Path::new("model"),
            &job(DeviceChoice::Cpu),
            &system_with_gpu(),
            &mut |_| {},
            &cancel,
        )
        .err()
        .unwrap();
        assert!(matches!(error, EngineError::Cancelled));
        assert!(fake.loads().is_empty());
    }

    #[test]
    fn the_fake_engine_reports_updates_in_order() {
        let fake = FakeEngines::new("fa");
        let mut engine = fake.opener()(Path::new("model"), Device::Cpu, "int8", 2).unwrap();
        let mut updates = Vec::new();
        let cancelled = engine
            .transcribe(
                &vec![0.0; 3 * RATE],
                Some("fa"),
                5,
                &mut |u| updates.push(u),
                &CancelToken::new(),
            )
            .unwrap();
        assert!(!cancelled);
        assert_eq!(
            updates,
            vec![
                Update::Language {
                    language: "fa".into(),
                    probability: 0.9
                },
                Update::Segment {
                    start: 0.0,
                    end: 3.0,
                    text: "window 1 at 0".into()
                },
            ]
        );
        assert_eq!(
            fake.transcriptions(),
            vec![FakeCall::Transcribe {
                samples: 3 * RATE,
                language: Some("fa".into()),
                beam_size: 5
            }]
        );
    }
}
