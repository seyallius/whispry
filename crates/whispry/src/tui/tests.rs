use std::cell::{Cell, RefCell};
use std::path::Path;
use std::sync::mpsc::Sender;
use std::time::Duration;

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use whispry_core::cancel::{CancelToken, JobCancelled};
use whispry_core::config::{DeviceChoice, JobConfig, validate_config};
use whispry_core::events::{Event, Stage};
use whispry_core::ffmpeg::FFmpeg;
use whispry_core::platform::{Os, SystemInfo, cuda_option, detect_cpu};
use whispry_core::transcriber::{JobError, JobSummary};
use whispry_core::worker::WorkerError;

use super::app::App;
use super::services::{JobMessage, Services};
use super::setup::normalize_path;

struct FakeServices {
    system: SystemInfo,
    downloaded: Vec<&'static str>,
    worker: bool,
}

impl FakeServices {
    fn new(gpus: usize) -> Self {
        FakeServices {
            system: SystemInfo {
                os: Os::Linux,
                os_label: "Linux (test)".into(),
                cpu_threads: 4,
                devices: vec![cuda_option(Os::Linux, Ok(gpus), &[]), detect_cpu()],
            },
            downloaded: vec!["small"],
            worker: true,
        }
    }
}

impl Services for FakeServices {
    fn system(&self) -> &SystemInfo {
        &self.system
    }
    fn worker_available(&self) -> bool {
        self.worker
    }
    fn is_model_downloaded(&self, model: &str) -> bool {
        self.downloaded.contains(&model)
    }
    fn validate(&self, config: &JobConfig) -> Vec<String> {
        validate_config(config)
    }
    fn find_ffmpeg(&self) -> Option<FFmpeg> {
        FFMPEG_INSTALLED.get().then(|| FFmpeg {
            ffmpeg: "ffmpeg".into(),
            ffprobe: None,
        })
    }
    fn ffmpeg_install_steps(&self) -> Vec<String> {
        vec![
            "Run:  sudo apt install ffmpeg".into(),
            "Check with:  ffmpeg -version".into(),
        ]
    }
    fn start_job(&self, config: JobConfig, cancel: CancelToken, sender: Sender<JobMessage>) {
        STARTED.with(|started| {
            started.borrow_mut().push(Started {
                config,
                cancel,
                sender,
            })
        });
    }
}

/// A job the app asked the fake to run. The test plays the job's part by
/// sending messages through `sender`.
struct Started {
    config: JobConfig,
    cancel: CancelToken,
    sender: Sender<JobMessage>,
}

thread_local! {
    static FFMPEG_INSTALLED: Cell<bool> = const { Cell::new(true) };
    // Each test runs on its own thread, so each sees only its own jobs.
    static STARTED: RefCell<Vec<Started>> = const { RefCell::new(Vec::new()) };
}

fn jobs_started() -> usize {
    STARTED.with(|started| started.borrow().len())
}

fn last_started() -> JobConfig {
    STARTED.with(|started| {
        started
            .borrow()
            .last()
            .expect("no job was started")
            .config
            .clone()
    })
}

fn last_cancel() -> CancelToken {
    STARTED.with(|started| started.borrow().last().unwrap().cancel.clone())
}

fn send(message: JobMessage) {
    STARTED.with(|started| {
        started
            .borrow()
            .last()
            .unwrap()
            .sender
            .send(message)
            .unwrap()
    });
}

fn send_event(event: Event) {
    send(JobMessage::Event(event));
}

fn press(app: &mut App, code: KeyCode) {
    app.on_key(KeyEvent::new(code, KeyModifiers::NONE));
}

fn ctrl(app: &mut App, c: char) {
    app.on_key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::CONTROL));
}

fn type_text(app: &mut App, text: &str) {
    for c in text.chars() {
        press(app, KeyCode::Char(c));
    }
}

fn screen(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
    terminal.draw(|frame| app.render(frame)).unwrap();
    let buffer = terminal.backend().buffer();
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| buffer[(x, y)].symbol())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// An app with a real (empty) audio file typed into the file field.
fn app_with_file(services: FakeServices) -> (App, tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("talk.mp3");
    std::fs::write(&audio, b"").unwrap();
    (
        App::new(Box::new(services), Some(audio.clone())),
        dir,
        audio,
    )
}

#[test]
fn the_screen_shows_the_machine_and_every_choice() {
    let app = App::new(Box::new(FakeServices::new(0)), None);
    let text = screen(&app, 100, 30);
    for expected in [
        "Whispry",
        "Linux (test)",
        "no GPU",
        "File",
        "Model",
        "Language",
        "Small (downloaded)",
        "Auto-detect",
        "Auto  uses CPU",
        "No NVIDIA GPU or driver found",
        "Start transcription",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
}

#[test]
fn auto_names_the_gpu_when_there_is_one() {
    let app = App::new(Box::new(FakeServices::new(1)), None);
    let text = screen(&app, 100, 30);
    assert!(text.contains("uses NVIDIA GPU (CUDA)"), "{text}");
    assert!(text.contains("GPU ready"), "{text}");
}

#[test]
fn starting_returns_the_defaults_for_a_valid_file() {
    let (mut app, _dir, audio) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 's');
    let config = last_started();
    assert_eq!(config.audio, audio);
    assert_eq!(config.model, "small");
    assert_eq!(config.language, None);
    assert_eq!(config.device, DeviceChoice::Auto);
    assert_eq!(config.compute_type, None);
    assert_eq!(config.chunk_minutes, 10.0);
}

#[test]
fn problems_are_listed_and_the_screen_stays() {
    let mut app = App::new(Box::new(FakeServices::new(0)), None);
    ctrl(&mut app, 's');
    assert_eq!(jobs_started(), 0);
    assert!(screen(&app, 100, 30).contains("Choose a file to transcribe"));

    type_text(&mut app, "/nonexistent/talk.mp3");
    ctrl(&mut app, 's');
    assert_eq!(jobs_started(), 0);
    assert!(screen(&app, 100, 30).contains("File not found"));
}

#[test]
fn a_missing_worker_is_explained() {
    let mut services = FakeServices::new(0);
    services.worker = false;
    let (mut app, _dir, _) = app_with_file(services);
    ctrl(&mut app, 's');
    assert_eq!(jobs_started(), 0);
    assert!(screen(&app, 110, 32).contains("worker was not found"));
}

#[test]
fn choices_are_made_with_tab_and_the_arrow_keys() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    press(&mut app, KeyCode::Tab); // model
    press(&mut app, KeyCode::Right); // medium
    press(&mut app, KeyCode::Tab); // language
    press(&mut app, KeyCode::Right);
    press(&mut app, KeyCode::Right); // Persian
    press(&mut app, KeyCode::Tab); // device
    press(&mut app, KeyCode::Right); // CPU
    ctrl(&mut app, 's');
    let config = last_started();
    assert_eq!(config.model, "medium");
    assert_eq!(config.language.as_deref(), Some("fa"));
    assert_eq!(config.device, DeviceChoice::Cpu);
}

#[test]
fn the_unavailable_gpu_cannot_be_selected() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    for _ in 0..3 {
        press(&mut app, KeyCode::Tab);
    } // device
    press(&mut app, KeyCode::Right); // CPU
    press(&mut app, KeyCode::Right); // would be the GPU; wraps to Auto
    ctrl(&mut app, 's');
    assert_eq!(last_started().device, DeviceChoice::Auto);
}

#[test]
fn the_gpu_can_be_selected_when_available() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(1));
    for _ in 0..3 {
        press(&mut app, KeyCode::Tab);
    }
    press(&mut app, KeyCode::Left); // wraps to the GPU
    ctrl(&mut app, 's');
    assert_eq!(last_started().device, DeviceChoice::Cuda);
}

#[test]
fn advanced_options_open_on_demand_and_set_the_compute_type_and_chunk_length() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    assert!(!screen(&app, 100, 34).contains("Compute type"));
    for _ in 0..4 {
        press(&mut app, KeyCode::Tab);
    } // advanced
    press(&mut app, KeyCode::Enter);
    assert!(screen(&app, 100, 34).contains("Compute type"));
    press(&mut app, KeyCode::Tab); // compute
    press(&mut app, KeyCode::Right); // first real type for the CPU: int8
    press(&mut app, KeyCode::Tab); // chunk
    ctrl(&mut app, 'u');
    type_text(&mut app, "5");
    ctrl(&mut app, 's');
    let config = last_started();
    assert_eq!(config.compute_type.as_deref(), Some("int8"));
    assert_eq!(config.chunk_minutes, 5.0);
}

#[test]
fn a_chunk_length_that_is_not_a_number_is_reported() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    for _ in 0..4 {
        press(&mut app, KeyCode::Tab);
    }
    press(&mut app, KeyCode::Enter);
    press(&mut app, KeyCode::Tab);
    press(&mut app, KeyCode::Tab);
    type_text(&mut app, "x");
    ctrl(&mut app, 's');
    assert_eq!(jobs_started(), 0);
    assert!(screen(&app, 100, 40).contains("must be a number"));
}

#[test]
fn pasting_a_quoted_path_works() {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("my talk.mp3");
    std::fs::write(&audio, b"").unwrap();
    let mut app = App::new(Box::new(FakeServices::new(0)), None);
    app.on_paste(&format!("\"{}\"\n", audio.display()));
    ctrl(&mut app, 's');
    assert_eq!(last_started().audio, audio);
}

#[test]
fn the_start_button_starts_with_enter() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    press(&mut app, KeyCode::BackTab); // wraps to the start button
    press(&mut app, KeyCode::Enter);
    assert_eq!(jobs_started(), 1);
}

#[test]
fn ctrl_q_and_ctrl_c_quit_without_a_job() {
    for key in ['q', 'c'] {
        let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
        ctrl(&mut app, key);
        assert!(app.is_done());
        assert_eq!(jobs_started(), 0);
    }
}

#[test]
fn paths_lose_quotes_and_expand_the_home_folder() {
    assert_eq!(normalize_path("  \"/a b/c.mp3\" "), Path::new("/a b/c.mp3"));
    assert_eq!(normalize_path("'/a/c.mp3'"), Path::new("/a/c.mp3"));
    assert_eq!(normalize_path("/a/c.mp3"), Path::new("/a/c.mp3"));
    if let Some(home) = std::env::home_dir() {
        assert_eq!(normalize_path("~/x.mp3"), home.join("x.mp3"));
    }
}

#[test]
fn browsing_fills_in_the_chosen_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("lecture.mp3"), b"").unwrap();
    std::fs::write(dir.path().join("notes.txt"), b"").unwrap();
    let mut app = App::new(Box::new(FakeServices::new(0)), None);
    app.on_paste(&dir.path().display().to_string());

    ctrl(&mut app, 'o');
    let text = screen(&app, 100, 30);
    assert!(text.contains("Choose a file"), "{text}");
    assert!(
        text.contains("lecture.mp3") && !text.contains("notes.txt"),
        "{text}"
    );

    press(&mut app, KeyCode::Down); // the first entry after ".."
    press(&mut app, KeyCode::Enter);
    assert!(!screen(&app, 100, 30).contains("Choose a file"));
    ctrl(&mut app, 's');
    assert_eq!(last_started().audio, dir.path().join("lecture.mp3"));
}

#[test]
fn cancelling_the_browser_keeps_the_file_field() {
    let (mut app, _dir, audio) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 'o');
    press(&mut app, KeyCode::Esc);
    assert!(!screen(&app, 100, 30).contains("Choose a file"));
    ctrl(&mut app, 's');
    assert_eq!(last_started().audio, audio);
}

#[test]
fn keys_go_to_the_browser_while_it_is_open() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 'o');
    press(&mut app, KeyCode::Tab);
    ctrl(&mut app, 's'); // ignored by the browser, so no job starts
    assert_eq!(jobs_started(), 0);
}

// ----- the run screen -----

fn running_app() -> (App, tempfile::TempDir) {
    let (mut app, dir, _) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 's');
    assert_eq!(jobs_started(), 1);
    (app, dir)
}

fn summary(output: &Path) -> JobSummary {
    JobSummary {
        output: output.to_path_buf(),
        audio_duration: 125.0,
        elapsed: 9.0,
    }
}

fn deliver(app: &mut App) {
    app.poll();
}

#[test]
fn starting_shows_the_run_screen_and_a_cancel_button() {
    let (app, _dir) = running_app();
    let text = screen(&app, 100, 34);
    for expected in [
        "Transcribing",
        "Progress",
        "Transcript",
        "Activity",
        "Cancel",
        "talk.mp3",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
    assert!(!text.contains("Start transcription"));
}

#[test]
fn events_fill_in_the_transcript_the_activity_and_the_progress() {
    let (mut app, _dir) = running_app();
    send_event(Event::StageStarted {
        stage: Stage::Transcribe,
        message: "Transcribing...".into(),
    });
    send_event(Event::LanguageDetected {
        language: "fa".into(),
        probability: 0.93,
    });
    send_event(Event::Progress {
        stage: Stage::Transcribe,
        completed: 30.0,
        total: Some(120.0),
        unit: "s".into(),
    });
    send_event(Event::SegmentReady {
        start: 65.0,
        end: 70.0,
        text: "hello there".into(),
    });
    deliver(&mut app);

    let text = screen(&app, 100, 34);
    assert!(text.contains("[01:05] hello there"), "{text}");
    assert!(text.contains("Language: fa (93%)"), "{text}");
    assert!(text.contains("25%"), "{text}");
    assert!(text.contains("Audio 00:30 / 02:00"), "{text}");
}

#[test]
fn a_finished_job_shows_where_the_text_was_saved() {
    let (mut app, dir) = running_app();
    let output = dir.path().join("talk.txt");
    send(JobMessage::Done(Ok(summary(&output))));
    deliver(&mut app);

    let text = screen(&app, 100, 34);
    assert!(text.contains("Finished"), "{text}");
    assert!(text.contains("Transcribed 02:05 of audio"), "{text}");
    assert!(text.contains("talk.txt"), "{text}");
    assert!(
        text.contains("New transcription") && text.contains("Quit"),
        "{text}"
    );
    assert!(!text.contains("Retry on CPU"));
}

#[test]
fn escape_cancels_and_the_job_reports_where_it_stopped() {
    let (mut app, dir) = running_app();
    press(&mut app, KeyCode::Esc);
    assert!(last_cancel().is_cancelled());
    assert!(screen(&app, 100, 34).contains("Cancelling"));

    let output = dir.path().join("talk.txt");
    send(JobMessage::Done(Err(JobError::Cancelled(JobCancelled {
        output: Some(output.clone()),
        position: 42.0,
    }))));
    deliver(&mut app);
    let text = screen(&app, 100, 34);
    assert!(
        text.contains("Cancelled") && text.contains("saved in"),
        "{text}"
    );
    assert!(!app.is_done());
}

#[test]
fn a_failure_offers_a_retry_on_the_cpu_that_reruns_the_same_job() {
    let (mut app, _dir) = running_app();
    send(JobMessage::Done(Err(JobError::Worker(WorkerError::Load {
        device: "cuda".into(),
        message: "out of memory".into(),
    }))));
    deliver(&mut app);
    let text = screen(&app, 100, 34);
    assert!(
        text.contains("Failed") && text.contains("out of memory"),
        "{text}"
    );
    assert!(text.contains("Retry on CPU"), "{text}");

    press(&mut app, KeyCode::Enter); // the first button
    assert_eq!(jobs_started(), 2);
    let retry = last_started();
    assert_eq!(retry.device, DeviceChoice::Cpu);
    assert_eq!(
        retry.audio,
        STARTED.with(|s| s.borrow()[0].config.audio.clone())
    );
    assert!(screen(&app, 100, 34).contains("Transcribing"));
}

#[test]
fn a_failure_on_the_cpu_has_nothing_to_retry() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    for _ in 0..3 {
        press(&mut app, KeyCode::Tab);
    }
    press(&mut app, KeyCode::Right); // CPU
    ctrl(&mut app, 's');
    send(JobMessage::Done(Err(JobError::Worker(
        WorkerError::Exited("crashed".into()),
    ))));
    deliver(&mut app);
    assert!(!screen(&app, 100, 34).contains("Retry on CPU"));
}

#[test]
fn new_transcription_returns_to_the_setup_screen_with_the_same_choices() {
    let (mut app, dir) = running_app();
    send(JobMessage::Done(Ok(summary(&dir.path().join("talk.txt")))));
    deliver(&mut app);
    press(&mut app, KeyCode::Enter); // "New transcription" is first
    let text = screen(&app, 100, 30);
    assert!(
        text.contains("Start transcription") && text.contains("talk.mp3"),
        "{text}"
    );
}

#[test]
fn quit_ends_the_program_once_the_job_is_over() {
    let (mut app, dir) = running_app();
    send(JobMessage::Done(Ok(summary(&dir.path().join("talk.txt")))));
    deliver(&mut app);
    press(&mut app, KeyCode::Char('q'));
    assert!(app.is_done());
}

#[test]
fn ctrl_q_during_a_job_cancels_it_and_quits_when_it_has_stopped() {
    let (mut app, _dir) = running_app();
    ctrl(&mut app, 'q');
    assert!(last_cancel().is_cancelled());
    assert!(!app.is_done(), "waits for the job to stop");

    send(JobMessage::Done(Err(JobError::Cancelled(JobCancelled {
        output: None,
        position: 0.0,
    }))));
    deliver(&mut app);
    assert!(app.is_done());
}

#[test]
fn a_second_ctrl_q_quits_without_waiting() {
    let (mut app, _dir) = running_app();
    ctrl(&mut app, 'q');
    ctrl(&mut app, 'q');
    assert!(app.is_done());
}

#[test]
fn a_job_thread_that_vanishes_is_reported_as_a_failure() {
    let (mut app, _dir) = running_app();
    STARTED.with(|started| started.borrow_mut().clear()); // drops the sender
    deliver(&mut app);
    assert!(screen(&app, 100, 34).contains("Failed"));
}

#[test]
fn the_time_left_comes_from_the_speed_so_far() {
    use super::run::RunState;
    use std::time::Instant;

    let (sender, receiver) = std::sync::mpsc::channel::<JobMessage>();
    let start = Instant::now();
    let mut run = RunState::new(JobConfig::new("a.mp3"), CancelToken::new(), receiver, start);
    let at = |seconds: u64| start + Duration::from_secs(seconds);

    run.apply(
        &Event::StageStarted {
            stage: Stage::Transcribe,
            message: "Transcribing...".into(),
        },
        at(5),
    );
    run.apply(
        &Event::Progress {
            stage: Stage::Transcribe,
            completed: 40.0,
            total: Some(100.0),
            unit: "s".into(),
        },
        at(25),
    );
    // 40 s of audio in the 20 s since transcription began: 2x, 30 s left.
    assert_eq!(
        run.timing_line(at(25)),
        "Elapsed 00:25 · Audio 00:40 / 01:40 · 2.0x realtime · about 00:30 left"
    );
    // Too early for a trustworthy speed.
    assert_eq!(
        run.timing_line(at(6)),
        "Elapsed 00:06 · Audio 00:40 / 01:40"
    );
    drop(sender);
}

#[test]
fn downloads_show_bytes_instead_of_audio() {
    use super::run::RunState;
    use std::time::Instant;

    let (_sender, receiver) = std::sync::mpsc::channel::<JobMessage>();
    let start = Instant::now();
    let mut run = RunState::new(JobConfig::new("a.mp3"), CancelToken::new(), receiver, start);
    run.apply(
        &Event::Progress {
            stage: Stage::Model,
            completed: 512.0 * 1024.0 * 1024.0,
            total: Some(1024.0 * 1024.0 * 1024.0),
            unit: "B".into(),
        },
        start,
    );
    assert_eq!(run.timing_line(start), "Elapsed 00:00 · 512.0 MB / 1.0 GB");
    assert_eq!(run.ratio(), Some(0.5));
}

#[test]
fn long_lines_wrap_at_the_width() {
    use super::run::wrap;
    assert_eq!(wrap("one two three", 7), ["one two", "three"]);
    assert_eq!(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"]);
    assert_eq!(wrap("", 5), [""]);
}

// ----- the FFmpeg dialog -----

#[test]
fn a_missing_ffmpeg_shows_the_install_steps_instead_of_starting() {
    FFMPEG_INSTALLED.set(false);
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 's');
    assert_eq!(jobs_started(), 0);
    let text = screen(&app, 100, 30);
    for expected in [
        "FFmpeg is needed",
        "sudo apt install ffmpeg",
        "Check again",
        "Close",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in:\n{text}");
    }
}

#[test]
fn checking_again_starts_the_job_once_ffmpeg_is_installed() {
    FFMPEG_INSTALLED.set(false);
    let (mut app, _dir, audio) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 's');

    press(&mut app, KeyCode::Enter); // Check again; still missing
    assert_eq!(jobs_started(), 0);
    assert!(screen(&app, 100, 30).contains("still not found"));

    FFMPEG_INSTALLED.set(true);
    press(&mut app, KeyCode::Enter);
    assert_eq!(jobs_started(), 1);
    assert_eq!(last_started().audio, audio);
    assert!(screen(&app, 100, 30).contains("Transcribing"));
}

#[test]
fn closing_the_dialog_returns_to_the_setup_screen() {
    FFMPEG_INSTALLED.set(false);
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 's');
    press(&mut app, KeyCode::Esc);
    let text = screen(&app, 100, 30);
    assert!(!text.contains("FFmpeg is needed"));
    assert!(text.contains("Start transcription"));
    assert_eq!(jobs_started(), 0);
}

#[test]
fn the_close_button_works_with_tab_and_enter() {
    FFMPEG_INSTALLED.set(false);
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 's');
    press(&mut app, KeyCode::Tab);
    press(&mut app, KeyCode::Enter);
    assert!(!screen(&app, 100, 30).contains("FFmpeg is needed"));
}

#[test]
fn a_retry_on_the_cpu_also_checks_for_ffmpeg() {
    let (mut app, _dir) = running_app();
    send(JobMessage::Done(Err(JobError::Worker(
        WorkerError::Exited("x".into()),
    ))));
    deliver(&mut app);
    FFMPEG_INSTALLED.set(false);
    press(&mut app, KeyCode::Enter); // Retry on CPU
    assert_eq!(jobs_started(), 1);
    assert!(screen(&app, 100, 30).contains("FFmpeg is needed"));
}

#[test]
fn long_activity_lines_are_cut_short_not_wrapped() {
    use super::run::truncate;
    assert_eq!(truncate("short", 10), "short");
    assert_eq!(truncate("a very long line", 8), "a very …");

    let (mut app, _dir) = running_app();
    send_event(Event::Log(format!(
        "Using cached model: {}",
        "x".repeat(200)
    )));
    deliver(&mut app);
    let text = screen(&app, 100, 34);
    assert!(text.contains("Using cached model: xxx"), "{text}");
    assert!(text.contains('…'), "{text}");
}

#[test]
fn browsing_a_bare_file_name_opens_the_current_folder_not_home() {
    let mut app = App::new(Box::new(FakeServices::new(0)), None);
    app.on_paste("Cargo.toml"); // exists in the crate folder the tests run in
    ctrl(&mut app, 'o');
    let cwd = std::env::current_dir().unwrap();
    let text = screen(&app, 140, 30);
    assert!(text.contains(&cwd.display().to_string()), "{text}");
}

#[test]
fn a_finished_job_no_longer_says_it_is_transcribing() {
    let (mut app, dir) = running_app();
    send_event(Event::StageStarted {
        stage: Stage::Transcribe,
        message: "Transcribing...".into(),
    });
    deliver(&mut app);
    assert!(screen(&app, 100, 34).contains("Transcribing..."));
    send(JobMessage::Done(Ok(summary(&dir.path().join("talk.txt")))));
    deliver(&mut app);
    let text = screen(&app, 100, 34);
    // The stage line is the first row inside the progress box.
    let stage_line = text.lines().nth(3).unwrap();
    assert!(stage_line.contains("Done."), "{text}");
    assert!(!stage_line.contains("Transcribing"), "{text}");
}
