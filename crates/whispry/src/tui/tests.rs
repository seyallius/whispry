use std::path::Path;

use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use whispry_core::config::{DeviceChoice, JobConfig, validate_config};
use whispry_core::platform::{Os, SystemInfo, cuda_option, detect_cpu};

use super::app::App;
use super::services::Services;
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
    assert!(app.is_done());
    let config = app.into_outcome().unwrap();
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
    assert!(!app.is_done());
    assert!(screen(&app, 100, 30).contains("Choose a file to transcribe"));

    type_text(&mut app, "/nonexistent/talk.mp3");
    ctrl(&mut app, 's');
    assert!(!app.is_done());
    assert!(screen(&app, 100, 30).contains("File not found"));
}

#[test]
fn a_missing_worker_is_explained() {
    let mut services = FakeServices::new(0);
    services.worker = false;
    let (mut app, _dir, _) = app_with_file(services);
    ctrl(&mut app, 's');
    assert!(!app.is_done());
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
    let config = app.into_outcome().unwrap();
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
    assert_eq!(app.into_outcome().unwrap().device, DeviceChoice::Auto);
}

#[test]
fn the_gpu_can_be_selected_when_available() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(1));
    for _ in 0..3 {
        press(&mut app, KeyCode::Tab);
    }
    press(&mut app, KeyCode::Left); // wraps to the GPU
    ctrl(&mut app, 's');
    assert_eq!(app.into_outcome().unwrap().device, DeviceChoice::Cuda);
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
    let config = app.into_outcome().unwrap();
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
    assert!(!app.is_done());
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
    assert_eq!(app.into_outcome().unwrap().audio, audio);
}

#[test]
fn the_start_button_starts_with_enter() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    press(&mut app, KeyCode::BackTab); // wraps to the start button
    press(&mut app, KeyCode::Enter);
    assert!(app.into_outcome().is_some());
}

#[test]
fn ctrl_q_and_ctrl_c_quit_without_a_job() {
    for key in ['q', 'c'] {
        let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
        ctrl(&mut app, key);
        assert!(app.is_done());
        assert!(app.into_outcome().is_none());
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
    assert_eq!(
        app.into_outcome().unwrap().audio,
        dir.path().join("lecture.mp3")
    );
}

#[test]
fn cancelling_the_browser_keeps_the_file_field() {
    let (mut app, _dir, audio) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 'o');
    press(&mut app, KeyCode::Esc);
    assert!(!screen(&app, 100, 30).contains("Choose a file"));
    ctrl(&mut app, 's');
    assert_eq!(app.into_outcome().unwrap().audio, audio);
}

#[test]
fn keys_go_to_the_browser_while_it_is_open() {
    let (mut app, _dir, _) = app_with_file(FakeServices::new(0));
    ctrl(&mut app, 'o');
    press(&mut app, KeyCode::Tab);
    ctrl(&mut app, 's'); // ignored by the browser, so no job starts
    assert!(!app.is_done());
}
