//! Runs the real binary for the parts that need no model.

use std::process::{Command, Output};

fn whispry(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_whispry"))
        .args(args)
        .env("WHISPRY_PYTHON", "")
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn prints_its_version() {
    let output = whispry(&["--version"]);
    assert!(output.status.success());
    assert!(text(&output.stdout).starts_with("whispry 0."));
}

#[test]
fn devices_lists_the_cpu_and_the_gpu() {
    let output = whispry(&["devices"]);
    let out = text(&output.stdout);
    assert!(output.status.success());
    assert!(out.contains("CPU threads:"), "{out}");
    assert!(out.contains("CPU: int8"), "{out}");
    assert!(out.contains("NVIDIA GPU (CUDA)"), "{out}");
    assert!(out.contains("FFmpeg:"), "{out}");
}

#[test]
fn a_missing_file_is_a_usage_error() {
    let output = whispry(&["transcribe", "/nonexistent/missing.wav"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("File not found"));
}

#[test]
fn unknown_models_and_languages_are_rejected() {
    for args in [
        ["transcribe", "a.wav", "--model", "huge"],
        ["transcribe", "a.wav", "--language", "xx"],
        ["transcribe", "a.wav", "--device", "tpu"],
    ] {
        assert_eq!(whispry(&args).status.code(), Some(2), "{args:?}");
    }
}

#[test]
fn an_out_of_range_chunk_length_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("a.wav");
    std::fs::write(&audio, b"").unwrap();
    let output = whispry(&[
        "transcribe",
        audio.to_str().unwrap(),
        "--chunk-minutes",
        "99",
    ]);
    assert_eq!(output.status.code(), Some(2));
    assert!(text(&output.stderr).contains("Chunk length"));
}

#[test]
fn an_unavailable_gpu_is_a_clean_error() {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("a.wav");
    std::fs::write(&audio, b"").unwrap();
    // Only meaningful on machines without a usable GPU, like CI runners.
    if text(&whispry(&["devices"]).stdout).contains("[available]   NVIDIA") {
        return;
    }
    let output = whispry(&["transcribe", audio.to_str().unwrap(), "--device", "cuda"]);
    assert_eq!(output.status.code(), Some(1));
    // macOS says CUDA is not supported there; other systems say no GPU was found.
    assert!(
        text(&output.stderr).contains("No NVIDIA GPU")
            || text(&output.stderr).contains("not supported on macOS"),
        "{}",
        text(&output.stderr)
    );
}
