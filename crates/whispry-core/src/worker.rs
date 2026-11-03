//! Talks to the Python transcription worker. See `worker/PROTOCOL.md`.
//!
//! The worker is a child process that keeps the model loaded. Rust sends it
//! audio windows and receives segments as they are ready.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::cancel::CancelToken;
use crate::config::{DeviceChoice, JobConfig};
use crate::events::{Event, EventSink, Stage};
use crate::platform::{Device, DeviceOption, Os, SystemInfo, resolve_device};
use crate::process::command;

/// Starting Python and importing CTranslate2 can take a while on a cold disk.
const READY_TIMEOUT: Duration = Duration::from_secs(120);
/// Loading a model from disk.
const LOAD_TIMEOUT: Duration = Duration::from_secs(600);
/// How long a cancelled window may keep running before the worker is killed.
const CANCEL_GRACE: Duration = Duration::from_secs(30);
const POLL: Duration = Duration::from_millis(50);
const STDERR_LINES_KEPT: usize = 20;
const AUDIO_WRITE_SAMPLES: usize = 16 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum WorkerError {
    #[error("Could not start the transcription worker: {0}")]
    Start(std::io::Error),
    #[error("The transcription worker did not start properly: {0}")]
    NotReady(String),
    #[error("The transcription worker stopped unexpectedly{}", detail(.0))]
    Exited(String),
    #[error("Could not load the model on {device}: {message}")]
    Load { device: String, message: String },
    #[error("Transcription failed: {0}")]
    Transcribe(String),
    #[error("The transcription worker sent something unexpected: {0}")]
    Protocol(String),
    #[error("Transcription cancelled")]
    Cancelled,
}

fn detail(text: &str) -> String {
    if text.is_empty() {
        String::new()
    } else {
        format!(":\n{text}")
    }
}

/// A message from the worker.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkerMessage {
    Ready {
        version: String,
        faster_whisper: String,
    },
    Loaded {
        device: String,
        compute_type: String,
    },
    Language {
        id: u64,
        language: String,
        probability: f64,
    },
    Segment {
        id: u64,
        start: f64,
        end: f64,
        text: String,
    },
    Done {
        id: u64,
        cancelled: bool,
    },
    Error {
        id: Option<u64>,
        stage: String,
        message: String,
    },
}

/// What a window produces while it is transcribed.
#[derive(Clone, Debug, PartialEq)]
pub enum Update {
    Language {
        language: String,
        probability: f64,
    },
    /// Times are relative to the start of the window.
    Segment {
        start: f64,
        end: f64,
        text: String,
    },
}

enum Incoming {
    Message(WorkerMessage),
    /// A line on stdout that is not a protocol message.
    Garbage(String),
    Closed,
}

/// How to start the worker process.
#[derive(Clone, Debug)]
pub struct WorkerCommand {
    pub program: PathBuf,
    pub args: Vec<OsString>,
}

impl WorkerCommand {
    /// Run `python -m whispry_worker` with the given interpreter.
    pub fn python_module(python: impl Into<PathBuf>) -> Self {
        WorkerCommand {
            program: python.into(),
            args: vec!["-m".into(), "whispry_worker".into()],
        }
    }
}

/// The Python interpreter inside a virtual environment folder.
pub fn venv_python(venv: &Path, os: Os) -> PathBuf {
    match os {
        Os::Windows => venv.join("Scripts").join("python.exe"),
        _ => venv.join("bin").join("python"),
    }
}

/// The virtual environment folder an interpreter belongs to.
pub fn venv_of(python: &Path) -> Option<PathBuf> {
    Some(python.parent()?.parent()?.to_path_buf())
}

/// Find the worker's Python: `WHISPRY_PYTHON`, else the `.venv` beside the
/// executable, else one in the current folder or a parent (for developers).
pub fn find_python(os: Os) -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let env = std::env::var_os("WHISPRY_PYTHON").map(PathBuf::from);
    let start = std::env::current_dir().ok();
    find_python_in(os, env, &exe_dir, start.as_deref())
}

fn find_python_in(
    os: Os,
    env: Option<PathBuf>,
    exe_dir: &Path,
    start: Option<&Path>,
) -> Option<PathBuf> {
    if let Some(python) = env.filter(|path| !path.as_os_str().is_empty()) {
        return Some(python);
    }
    let mut folders: Vec<&Path> = vec![exe_dir];
    folders.extend(start.into_iter().flat_map(Path::ancestors));
    folders
        .into_iter()
        .map(|folder| venv_python(&folder.join(".venv"), os))
        .find(|python| python.is_file())
}

/// A running worker process.
pub struct WorkerClient {
    child: Child,
    stdin: ChildStdin,
    incoming: Receiver<Incoming>,
    stderr_tail: Arc<Mutex<VecDeque<String>>>,
    pub version: String,
    next_id: u64,
}

impl WorkerClient {
    /// Start the worker and wait until it says it is ready.
    pub fn spawn(worker: &WorkerCommand, cancel: &CancelToken) -> Result<Self, WorkerError> {
        let mut child = command(&worker.program)
            .args(&worker.args)
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUNBUFFERED", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(WorkerError::Start)?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = child.stderr.take().expect("stderr was piped");

        let stderr_tail = Arc::new(Mutex::new(VecDeque::new()));
        drain_stderr(stderr, stderr_tail.clone());

        let mut client = WorkerClient {
            child,
            stdin,
            incoming: read_messages(stdout),
            stderr_tail,
            version: String::new(),
            next_id: 1,
        };

        match client.wait_for(READY_TIMEOUT, cancel, |message| match message {
            WorkerMessage::Ready { version, .. } => Some(Ok(version)),
            WorkerMessage::Error { message, .. } => Some(Err(WorkerError::NotReady(message))),
            _ => None,
        }) {
            Ok(version) => {
                client.version = version;
                Ok(client)
            }
            Err(WorkerError::Exited(detail)) => Err(WorkerError::NotReady(detail)),
            Err(error) => Err(error),
        }
    }

    fn stderr_text(&self) -> String {
        let lines = self.stderr_tail.lock().unwrap();
        lines.iter().cloned().collect::<Vec<_>>().join("\n")
    }

    fn exited(&mut self) -> WorkerError {
        // Give the stderr thread a moment to read the last lines.
        let _ = self.child.wait();
        std::thread::sleep(Duration::from_millis(50));
        WorkerError::Exited(self.stderr_text())
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    fn send(&mut self, message: &serde_json::Value, audio: &[f32]) -> Result<(), WorkerError> {
        let result = (|| -> std::io::Result<()> {
            writeln!(self.stdin, "{message}")?;
            for chunk in audio.chunks(AUDIO_WRITE_SAMPLES) {
                let mut bytes = Vec::with_capacity(chunk.len() * 4);
                for sample in chunk {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                }
                self.stdin.write_all(&bytes)?;
            }
            self.stdin.flush()
        })();
        result.map_err(|_| self.exited())
    }

    /// Wait for a message `pick` accepts. Cancelling kills the worker.
    fn wait_for<T>(
        &mut self,
        timeout: Duration,
        cancel: &CancelToken,
        mut pick: impl FnMut(WorkerMessage) -> Option<Result<T, WorkerError>>,
    ) -> Result<T, WorkerError> {
        let deadline = Instant::now() + timeout;
        loop {
            if cancel.is_cancelled() {
                self.kill();
                return Err(WorkerError::Cancelled);
            }
            if Instant::now() >= deadline {
                self.kill();
                return Err(WorkerError::NotReady("timed out".into()));
            }
            match self.incoming.recv_timeout(POLL) {
                Ok(Incoming::Message(message)) => {
                    if let Some(result) = pick(message) {
                        return result;
                    }
                }
                Ok(Incoming::Garbage(line)) => return Err(WorkerError::Protocol(line)),
                Ok(Incoming::Closed) | Err(RecvTimeoutError::Disconnected) => {
                    return Err(self.exited());
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }

    /// Load the model on one device. The worker uses the first of
    /// `compute_types` the device supports and returns it. The worker stays
    /// usable after a failure.
    pub fn load(
        &mut self,
        model_dir: &Path,
        device: Device,
        compute_types: &[&str],
        cpu_threads: usize,
        cancel: &CancelToken,
    ) -> Result<String, WorkerError> {
        self.send(
            &serde_json::json!({
                "type": "load",
                "model_dir": model_dir,
                "device": device.as_str(),
                "compute_types": compute_types,
                "cpu_threads": cpu_threads,
            }),
            &[],
        )?;
        self.wait_for(LOAD_TIMEOUT, cancel, |message| match message {
            WorkerMessage::Loaded { compute_type, .. } => Some(Ok(compute_type)),
            WorkerMessage::Error { stage, message, .. } if stage == "load" => {
                Some(Err(WorkerError::Load {
                    device: device.as_str().into(),
                    message,
                }))
            }
            WorkerMessage::Error { message, .. } => Some(Err(WorkerError::Protocol(message))),
            _ => None,
        })
    }

    /// Transcribe one window of 16 kHz mono audio, passing each update to
    /// `on_update` as it arrives. Returns whether the window was cancelled.
    pub fn transcribe(
        &mut self,
        samples: &[f32],
        language: Option<&str>,
        beam_size: u32,
        initial_prompt: Option<&str>,
        on_update: &mut dyn FnMut(Update),
        cancel: &CancelToken,
    ) -> Result<bool, WorkerError> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(
            &serde_json::json!({
                "type": "transcribe",
                "id": id,
                "language": language,
                "beam_size": beam_size,
                "initial_prompt": initial_prompt,
                "audio_bytes": samples.len() * 4,
            }),
            samples,
        )?;

        let mut cancel_sent: Option<Instant> = None;
        loop {
            if cancel.is_cancelled() {
                match cancel_sent {
                    None => {
                        self.send(&serde_json::json!({"type": "cancel", "id": id}), &[])?;
                        cancel_sent = Some(Instant::now());
                    }
                    Some(since) if since.elapsed() > CANCEL_GRACE => {
                        self.kill();
                        return Err(WorkerError::Cancelled);
                    }
                    Some(_) => {}
                }
            }

            match self.incoming.recv_timeout(POLL) {
                Ok(Incoming::Message(message)) => match message {
                    WorkerMessage::Language {
                        id: for_id,
                        language,
                        probability,
                    } if for_id == id => {
                        on_update(Update::Language {
                            language,
                            probability,
                        });
                    }
                    WorkerMessage::Segment {
                        id: for_id,
                        start,
                        end,
                        text,
                    } if for_id == id => {
                        on_update(Update::Segment { start, end, text });
                    }
                    WorkerMessage::Done {
                        id: for_id,
                        cancelled,
                    } if for_id == id => {
                        return Ok(cancelled);
                    }
                    WorkerMessage::Error { message, .. } => {
                        return Err(WorkerError::Transcribe(message));
                    }
                    _ => {}
                },
                Ok(Incoming::Garbage(line)) => return Err(WorkerError::Protocol(line)),
                Ok(Incoming::Closed) | Err(RecvTimeoutError::Disconnected) => {
                    return Err(self.exited());
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
        }
    }

    /// Ask the worker to exit, and stop it if it does not.
    pub fn close(&mut self) {
        let _ = writeln!(self.stdin, r#"{{"type":"shutdown"}}"#);
        let _ = self.stdin.flush();
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if matches!(self.child.try_wait(), Ok(Some(_))) {
                return;
            }
            std::thread::sleep(POLL);
        }
        self.kill();
    }
}

impl Drop for WorkerClient {
    fn drop(&mut self) {
        self.close();
    }
}

fn read_messages(stdout: impl Read + Send + 'static) -> Receiver<Incoming> {
    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            let Ok(line) = line else { break };
            if line.trim().is_empty() {
                continue;
            }
            let incoming = match serde_json::from_str(&line) {
                Ok(message) => Incoming::Message(message),
                Err(_) => Incoming::Garbage(line),
            };
            if sender.send(incoming).is_err() {
                return;
            }
        }
        let _ = sender.send(Incoming::Closed);
    });
    receiver
}

/// Keep reading the worker's stderr, so it can never block on a full pipe,
/// and remember the last lines for error messages.
fn drain_stderr(stderr: impl Read + Send + 'static, tail: Arc<Mutex<VecDeque<String>>>) {
    std::thread::spawn(move || {
        let mut reader = BufReader::new(stderr);
        let mut line = Vec::new();
        while let Ok(count) = reader.read_until(b'\n', &mut line) {
            if count == 0 {
                break;
            }
            let text = String::from_utf8_lossy(&line).trim_end().to_string();
            line.clear();
            let mut lines = tail.lock().unwrap();
            if lines.len() == STDERR_LINES_KEPT {
                lines.pop_front();
            }
            lines.push_back(text);
        }
    });
}

/// Where the model ended up.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedModel {
    pub device: Device,
    pub compute_type: String,
}

/// Load the model on the requested device. When the device is `auto` and the
/// GPU fails to load it, the same worker loads it on the CPU instead.
///
/// Unless the user picked a compute type, the worker chooses the best one
/// the device supports.
pub fn load_model(
    client: &mut WorkerClient,
    model_dir: &Path,
    config: &JobConfig,
    system: &SystemInfo,
    on_event: EventSink,
    cancel: &CancelToken,
) -> Result<LoadedModel, WorkerError> {
    let (option, _) = resolve_device(config.device, system, config.compute_type.as_deref())
        .map_err(|error| WorkerError::Load {
            device: config.device.to_string(),
            message: error.to_string(),
        })?;

    on_event(Event::StageStarted {
        stage: Stage::Model,
        message: format!("Loading model on {}...", option.label),
    });
    let start = Instant::now();

    let attempt = |device: &DeviceOption, client: &mut WorkerClient| {
        let preferred: Vec<&str> = match &config.compute_type {
            Some(chosen) => vec![chosen.as_str()],
            None => device.compute_types.clone(),
        };
        client.load(
            model_dir,
            device.device,
            &preferred,
            system.cpu_threads,
            cancel,
        )
    };

    let loaded = match attempt(&option, client) {
        Ok(compute_type) => LoadedModel {
            device: option.device,
            compute_type,
        },
        Err(error @ WorkerError::Load { .. })
            if config.device == DeviceChoice::Auto && option.device != Device::Cpu =>
        {
            on_event(Event::Log(format!("{error}. Falling back to CPU.")));
            let (cpu, _) = resolve_device(DeviceChoice::Cpu, system, None)
                .map_err(|error| WorkerError::Protocol(error.to_string()))?;
            let compute_type = attempt(&cpu, client)?;
            LoadedModel {
                device: cpu.device,
                compute_type,
            }
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
    Ok(loaded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_every_message_type() {
        let parse = |text: &str| serde_json::from_str::<WorkerMessage>(text).unwrap();
        assert_eq!(
            parse(r#"{"type":"ready","version":"0.3.0","faster_whisper":"1.2.1"}"#),
            WorkerMessage::Ready {
                version: "0.3.0".into(),
                faster_whisper: "1.2.1".into()
            }
        );
        assert_eq!(
            parse(r#"{"type":"segment","id":2,"start":0.5,"end":1.5,"text":"سلام"}"#),
            WorkerMessage::Segment {
                id: 2,
                start: 0.5,
                end: 1.5,
                text: "سلام".into()
            }
        );
        assert_eq!(
            parse(r#"{"type":"done","id":2,"cancelled":true}"#),
            WorkerMessage::Done {
                id: 2,
                cancelled: true
            }
        );
        assert_eq!(
            parse(r#"{"type":"error","id":null,"stage":"load","message":"boom"}"#),
            WorkerMessage::Error {
                id: None,
                stage: "load".into(),
                message: "boom".into()
            }
        );
        assert!(serde_json::from_str::<WorkerMessage>(r#"{"type":"nope"}"#).is_err());
    }

    #[test]
    fn python_lives_in_a_different_place_on_each_os() {
        let venv = Path::new("/env");
        assert_eq!(
            venv_python(venv, Os::Linux),
            PathBuf::from("/env/bin/python")
        );
        assert_eq!(
            venv_python(venv, Os::Windows),
            PathBuf::from("/env").join("Scripts").join("python.exe")
        );
        assert_eq!(
            venv_of(Path::new("/env/bin/python")),
            Some(PathBuf::from("/env"))
        );
    }

    #[test]
    fn finds_the_python_beside_the_executable_before_the_working_folder() {
        let beside = tempfile::tempdir().unwrap();
        let working = tempfile::tempdir().unwrap();
        for root in [beside.path(), working.path()] {
            let python = venv_python(&root.join(".venv"), Os::Linux);
            std::fs::create_dir_all(python.parent().unwrap()).unwrap();
            std::fs::write(&python, "").unwrap();
        }
        assert_eq!(
            find_python_in(Os::Linux, None, beside.path(), Some(working.path())),
            Some(venv_python(&beside.path().join(".venv"), Os::Linux))
        );
    }

    #[test]
    fn finds_the_python_in_a_parent_of_the_working_folder() {
        let root = tempfile::tempdir().unwrap();
        let python = venv_python(&root.path().join(".venv"), Os::Linux);
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, "").unwrap();
        let nested = root.path().join("a/b");
        std::fs::create_dir_all(&nested).unwrap();
        assert_eq!(
            find_python_in(Os::Linux, None, Path::new("/nonexistent"), Some(&nested)),
            Some(python)
        );
    }

    #[test]
    fn the_environment_variable_wins_and_nothing_found_is_none() {
        assert_eq!(
            find_python_in(Os::Linux, Some("/my/python".into()), Path::new("/x"), None),
            Some(PathBuf::from("/my/python"))
        );
        assert_eq!(
            find_python_in(Os::Linux, None, Path::new("/nonexistent"), None),
            None
        );
    }

    use crate::config::JobConfig;
    use crate::platform::{DeviceOption, cuda_option, detect_cpu};
    use crate::test_support::FakeWorker;

    const RATE: usize = 16_000;

    fn started(language: &str) -> Option<(tempfile::TempDir, FakeWorker, WorkerClient)> {
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeWorker::new(dir.path(), language)?;
        let client = WorkerClient::spawn(&fake.command, &CancelToken::new()).unwrap();
        Some((dir, fake, client))
    }

    fn system_with_gpu() -> SystemInfo {
        let devices: Vec<DeviceOption> = vec![cuda_option(Os::Linux, Ok(1), &[]), detect_cpu()];
        SystemInfo {
            os: Os::Linux,
            os_label: "Linux".into(),
            cpu_threads: 2,
            devices,
        }
    }

    #[test]
    fn transcribes_a_window_and_reports_updates_in_order() {
        let Some((_dir, fake, mut client)) = started("fa") else {
            return;
        };
        let cancel = CancelToken::new();
        client
            .load(Path::new("model"), Device::Cpu, &["int8"], 2, &cancel)
            .unwrap();

        let mut updates = Vec::new();
        let cancelled = client
            .transcribe(
                &vec![0.0; 3 * RATE],
                Some("fa"),
                5,
                Some("hello"),
                &mut |u| updates.push(u),
                &cancel,
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
        let call = &fake.transcriptions()[0];
        assert_eq!(call["samples"], 3 * RATE);
        assert_eq!(call["language"], "fa");
        assert_eq!(call["initial_prompt"], "hello");
        assert_eq!(call["beam_size"], 5);
        assert_eq!(call["vad_filter"], true);
    }

    #[test]
    fn the_worker_picks_the_first_compute_type_the_device_supports() {
        let Some((_dir, fake, mut client)) = started("en") else {
            return;
        };
        let used = client
            .load(
                Path::new("model"),
                Device::Cpu,
                &["int8_float32", "int8"],
                2,
                &CancelToken::new(),
            )
            .unwrap();
        // The fake device supports int8 and float32 only.
        assert_eq!(used, "int8");
        assert_eq!(fake.calls()[0]["compute_type"], "int8");
    }

    #[test]
    fn a_compute_type_nothing_supports_is_a_load_error() {
        let Some((_dir, _fake, mut client)) = started("en") else {
            return;
        };
        let error = client
            .load(
                Path::new("model"),
                Device::Cpu,
                &["float16"],
                2,
                &CancelToken::new(),
            )
            .unwrap_err();
        assert!(matches!(error, WorkerError::Load { message, .. } if message.contains("float16")));
    }

    #[test]
    fn a_failed_load_leaves_the_worker_alive_but_without_a_model() {
        let Some((_dir, _fake, mut client)) = started("en") else {
            return;
        };
        let cancel = CancelToken::new();
        let error = client
            .load(Path::new("model"), Device::Cuda, &["int8"], 2, &cancel)
            .unwrap_err();
        assert!(
            matches!(&error, WorkerError::Load { device, message }
            if device == "cuda" && message.contains("no GPU")),
            "{error}"
        );

        let error = client
            .transcribe(&[0.0; 100], None, 5, None, &mut |_| {}, &cancel)
            .unwrap_err();
        assert!(matches!(error, WorkerError::Transcribe(m) if m.contains("No model")));
    }

    #[test]
    fn cancelling_stops_a_window_early() {
        let Some((_dir, _fake, mut client)) = started("en") else {
            return;
        };
        let cancel = CancelToken::new();
        client
            .load(Path::new("model"), Device::Cpu, &["int8"], 2, &cancel)
            .unwrap();

        let mut segments = 0;
        let trigger = cancel.clone();
        // 100 s of audio would give five segments.
        let cancelled = client
            .transcribe(
                &vec![0.0; 100 * RATE],
                None,
                5,
                None,
                &mut |update| {
                    if matches!(update, Update::Segment { .. }) {
                        segments += 1;
                        trigger.cancel();
                    }
                },
                &cancel,
            )
            .unwrap();
        assert!(cancelled);
        assert!((1..5).contains(&segments), "{segments}");
    }

    #[test]
    fn auto_falls_back_to_the_cpu_when_the_gpu_fails_to_load() {
        let Some((_dir, fake, mut client)) = started("en") else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut audio = JobConfig::new(dir.path().join("a.wav"));
        audio.device = DeviceChoice::Auto;
        let mut events = Vec::new();

        load_model(
            &mut client,
            Path::new("model"),
            &audio,
            &system_with_gpu(),
            &mut |e| events.push(e),
            &CancelToken::new(),
        )
        .unwrap();

        let loads: Vec<_> = fake
            .calls()
            .iter()
            .map(|c| c["load"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(loads, ["cuda", "cpu"]);
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
        let Some((_dir, fake, mut client)) = started("en") else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let mut config = JobConfig::new(dir.path().join("a.wav"));
        config.device = DeviceChoice::Cuda;

        let error = load_model(
            &mut client,
            Path::new("model"),
            &config,
            &system_with_gpu(),
            &mut |_| {},
            &CancelToken::new(),
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("Could not load the model on cuda")
        );
        assert_eq!(fake.calls().len(), 1);
    }

    #[test]
    fn a_worker_that_dies_at_start_reports_what_it_printed() {
        let Some(python) = crate::test_support::python() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("dies.py");
        std::fs::write(
            &script,
            "import sys\nsys.stderr.write('boom: no module\\n')\nsys.exit(1)\n",
        )
        .unwrap();
        let command = WorkerCommand {
            program: python,
            args: vec![script.into_os_string()],
        };

        let error = WorkerClient::spawn(&command, &CancelToken::new())
            .err()
            .unwrap();
        assert!(
            matches!(&error, WorkerError::NotReady(m) if m.contains("boom: no module")),
            "{error}"
        );
    }

    #[test]
    fn output_that_is_not_the_protocol_is_reported() {
        let Some(python) = crate::test_support::python() else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("noisy.py");
        std::fs::write(&script, "print('hello')\nimport time\ntime.sleep(5)\n").unwrap();
        let command = WorkerCommand {
            program: python,
            args: vec![script.into_os_string()],
        };

        let error = WorkerClient::spawn(&command, &CancelToken::new())
            .err()
            .unwrap();
        assert!(matches!(error, WorkerError::Protocol(line) if line == "hello"));
    }

    #[test]
    fn a_missing_program_is_a_start_error() {
        let worker = WorkerCommand::python_module("/nonexistent/python");
        let error = WorkerClient::spawn(&worker, &CancelToken::new())
            .err()
            .unwrap();
        assert!(matches!(error, WorkerError::Start(_)));
    }
}
