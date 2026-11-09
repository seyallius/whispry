//! Helpers shared by tests.

use std::path::Path;

/// Write a 16 kHz mono 16-bit WAV file holding a sine wave.
pub fn write_sine_wav(path: &Path, seconds: u32, frequency: f64) {
    const RATE: u32 = 16_000;
    let samples = RATE * seconds;
    let mut bytes = Vec::with_capacity(44 + samples as usize * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + samples * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&RATE.to_le_bytes());
    bytes.extend_from_slice(&(RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(samples * 2).to_le_bytes());
    for n in 0..samples {
        let phase = 2.0 * std::f64::consts::PI * frequency * f64::from(n) / f64::from(RATE);
        bytes.extend_from_slice(&((phase.sin() * 8000.0) as i16).to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}

use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::cancel::CancelToken;
use crate::engine::{EngineError, EngineOpener, SpeechEngine, Update};
use crate::platform::Device;

/// Needs from the machine (FFmpeg) that tests skip when missing.
///
/// With `WHISPRY_TEST_STRICT=1`, as set in CI, a missing need fails the test
/// instead, so a green run always means the tests really ran.
pub fn require<T>(what: &str, found: Option<T>) -> Option<T> {
    if found.is_none() {
        let strict = std::env::var("WHISPRY_TEST_STRICT").is_ok_and(|value| value == "1");
        assert!(!strict, "this test needs {what}, which was not found");
        eprintln!("skipped: needs {what}");
    }
    found
}

/// What a fake engine was asked to do.
#[derive(Clone, Debug, PartialEq)]
pub enum FakeCall {
    Load {
        device: String,
        compute_type: String,
    },
    Transcribe {
        samples: usize,
        language: Option<String>,
        beam_size: u32,
    },
}

/// A pure-Rust stand-in for the speech engine that records its calls.
///
/// It loads on the CPU with `int8` or `float32` only (the GPU never loads),
/// and answers every call with one segment per 20 s of audio, 50 ms apart.
#[derive(Clone)]
pub struct FakeEngines {
    language: String,
    calls: Arc<Mutex<Vec<FakeCall>>>,
}

impl FakeEngines {
    pub fn new(language: &str) -> Self {
        FakeEngines {
            language: language.into(),
            calls: Arc::default(),
        }
    }

    pub fn opener(&self) -> EngineOpener {
        let fake = self.clone();
        Arc::new(move |_dir, device, compute_type, _threads| {
            fake.calls.lock().unwrap().push(FakeCall::Load {
                device: device.as_str().into(),
                compute_type: compute_type.into(),
            });
            if device == Device::Cuda {
                return Err(EngineError::Load {
                    device: "cuda".into(),
                    message: "no GPU in the fake engine".into(),
                });
            }
            if !["int8", "float32"].contains(&compute_type) {
                return Err(EngineError::Load {
                    device: device.as_str().into(),
                    message: format!("compute type {compute_type} is not supported"),
                });
            }
            let engine: Box<dyn SpeechEngine + Send> = Box::new(FakeEngine {
                fake: fake.clone(),
                count: 0,
            });
            Ok(engine)
        })
    }

    /// Everything the fake was asked to do, in order.
    pub fn calls(&self) -> Vec<FakeCall> {
        self.calls.lock().unwrap().clone()
    }

    /// Loads as "device compute_type".
    pub fn loads(&self) -> Vec<String> {
        self.calls()
            .into_iter()
            .filter_map(|call| match call {
                FakeCall::Load {
                    device,
                    compute_type,
                } => Some(format!("{device} {compute_type}")),
                FakeCall::Transcribe { .. } => None,
            })
            .collect()
    }

    pub fn transcriptions(&self) -> Vec<FakeCall> {
        self.calls()
            .into_iter()
            .filter(|call| matches!(call, FakeCall::Transcribe { .. }))
            .collect()
    }
}

struct FakeEngine {
    fake: FakeEngines,
    count: usize,
}

impl SpeechEngine for FakeEngine {
    fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        beam_size: u32,
        on_update: &mut dyn FnMut(Update),
        cancel: &CancelToken,
    ) -> Result<bool, EngineError> {
        self.count += 1;
        self.fake.calls.lock().unwrap().push(FakeCall::Transcribe {
            samples: samples.len(),
            language: language.map(str::to_string),
            beam_size,
        });
        on_update(Update::Language {
            language: self.fake.language.clone(),
            probability: 0.9,
        });

        let seconds = samples.len() as f64 / 16_000.0;
        let mut start = 0.0;
        while start + 1.0 < seconds {
            std::thread::sleep(Duration::from_millis(50));
            let end = f64::min(start + 20.0, seconds);
            on_update(Update::Segment {
                start,
                end,
                text: format!("window {} at {start:.0}", self.count),
            });
            if cancel.is_cancelled() {
                return Ok(true);
            }
            start = end;
        }
        Ok(false)
    }
}
