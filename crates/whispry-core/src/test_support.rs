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

use std::path::PathBuf;

use crate::worker::WorkerCommand;

/// Needs from the machine (FFmpeg, Python) that tests skip when missing.
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

/// A Python that can run the worker (has numpy), if this machine has one.
pub fn python() -> Option<PathBuf> {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidates = [
        std::env::var_os("WHISPRY_PYTHON").map(PathBuf::from),
        Some(repo.join(".venv/bin/python")),
        Some(repo.join(".venv/Scripts/python.exe")),
        Some(PathBuf::from("python3")),
        Some(PathBuf::from("python")),
    ];
    candidates.into_iter().flatten().find(|python| {
        crate::process::command(python)
            .args(["-c", "import numpy"])
            .output()
            .is_ok_and(|output| output.status.success())
    })
}

const FAKE_WORKER: &str = r#"
import json, sys, time
from types import SimpleNamespace

sys.path.insert(0, sys.argv[1])
from whispry_worker import server

CALLS, LANGUAGE = sys.argv[2], sys.argv[3]
RATE = 16000


def record(entry):
    with open(CALLS, "a") as file:
        file.write(json.dumps(entry) + "\n")


class FakeModel:
    """One segment per 20 s of audio, 50 ms apart."""

    def __init__(self):
        self.count = 0

    def transcribe(self, samples, **options):
        self.count += 1
        number = self.count
        seconds = len(samples) / RATE
        record({"samples": len(samples), **options})

        def segments():
            start = 0.0
            while start + 1 < seconds:
                time.sleep(0.05)
                end = min(start + 20, seconds)
                yield SimpleNamespace(start=start, end=end, text=f" window {number} at {start:.0f} ")
                start = end

        return segments(), SimpleNamespace(language=LANGUAGE, language_probability=0.9)


def factory(model_dir, device, compute_type, cpu_threads):
    record({"load": device, "compute_type": compute_type})
    if device == "cuda":
        raise RuntimeError("no GPU in the fake worker")
    return FakeModel()


server.serve(sys.stdin.buffer, sys.stdout.buffer, factory, lambda device: {"int8", "float32"})
"#;

/// The real worker server with a fake model, and the file it records calls in.
pub struct FakeWorker {
    pub command: WorkerCommand,
    pub calls: PathBuf,
}

impl FakeWorker {
    /// `None` when no suitable Python is installed (the test is skipped).
    pub fn new(dir: &Path, language: &str) -> Option<FakeWorker> {
        let python = require("Python with numpy", python())?;
        let script = dir.join("fake_worker.py");
        std::fs::write(&script, FAKE_WORKER).unwrap();
        let calls = dir.join("calls.jsonl");
        let worker_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../worker");
        let command = WorkerCommand {
            program: python,
            args: vec![
                script.into_os_string(),
                worker_dir.into_os_string(),
                calls.clone().into_os_string(),
                language.into(),
            ],
        };
        Some(FakeWorker { command, calls })
    }

    /// Everything the fake model was asked to do, in order.
    pub fn calls(&self) -> Vec<serde_json::Value> {
        std::fs::read_to_string(&self.calls)
            .unwrap_or_default()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    pub fn transcriptions(&self) -> Vec<serde_json::Value> {
        self.calls()
            .into_iter()
            .filter(|call| call.get("samples").is_some())
            .collect()
    }
}
