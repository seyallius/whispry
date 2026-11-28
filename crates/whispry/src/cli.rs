//! Command-line front end: parses arguments and renders core events as text.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::builder::PossibleValuesParser;
use clap::error::ErrorKind;
use clap::{CommandFactory, Parser, Subcommand};
use whispry_core::cancel::CancelToken;
use whispry_core::config::{
    DEFAULT_CHUNK_MINUTES, DEFAULT_MODEL, DeviceChoice, JobConfig, LANGUAGES, MODEL_REPOSITORIES,
    validate_config,
};
use whispry_core::ffmpeg::{find_ffmpeg, install_instructions};
use whispry_core::models::ensure_model;
use whispry_core::platform::Os;
use whispry_core::transcriber::{JobEnvironment, JobError, run_job};

use crate::console::ConsoleRenderer;

#[derive(Parser)]
#[command(name = "whispry", version, about = "Private, local speech-to-text.")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand)]
pub enum Command {
    /// Open the terminal UI (this is what runs without arguments)
    Tui {
        /// Fill in this file
        audio: Option<PathBuf>,
    },
    /// Transcribe a file without the terminal UI
    Transcribe(TranscribeArgs),
    /// Show the OS and which devices can run Whisper
    Devices,
    /// Show whether the GPU can be used (downloads nothing unless --install)
    Gpu {
        /// Download the GPU libraries (asks first, shows the size)
        #[arg(long)]
        install: bool,
        /// Do not ask before downloading (needs --install)
        #[arg(long, requires = "install")]
        yes: bool,
    },
    /// Check that FFmpeg is installed and show how to get it if not
    Ffmpeg,
    /// Download a model for offline use
    Download {
        #[arg(value_parser = model_names())]
        model: String,
    },
}

#[derive(clap::Args)]
pub struct TranscribeArgs {
    /// Audio or video file
    pub audio: PathBuf,
    /// Language code; omit to auto-detect
    #[arg(long, value_parser = language_codes())]
    pub language: Option<String>,
    /// Whisper model size
    #[arg(long, default_value = DEFAULT_MODEL, value_parser = model_names())]
    pub model: String,
    /// Compute device; auto uses the GPU when available
    #[arg(long, default_value = "auto", value_parser = ["auto", "cpu", "cuda"])]
    pub device: String,
    /// CTranslate2 compute type; defaults to the device's best
    #[arg(long)]
    pub compute_type: Option<String>,
    /// Process audio this many minutes at a time (bounds RAM use)
    #[arg(long, default_value_t = DEFAULT_CHUNK_MINUTES)]
    pub chunk_minutes: f64,
    /// Output text file (default: beside the input, .txt)
    #[arg(short, long)]
    pub output: Option<PathBuf>,
}

fn model_names() -> PossibleValuesParser {
    PossibleValuesParser::new(MODEL_REPOSITORIES.iter().map(|(name, _)| *name))
}

fn language_codes() -> PossibleValuesParser {
    PossibleValuesParser::new(LANGUAGES.iter().map(|(code, _)| *code))
}

/// Exit code for a run stopped with Ctrl+C.
const INTERRUPTED: u8 = 130;

fn usage_error(message: String) -> ! {
    Cli::command()
        .error(ErrorKind::ValueValidation, message)
        .exit()
}

fn fail(message: impl std::fmt::Display) -> ExitCode {
    eprintln!("\nError: {message}");
    ExitCode::FAILURE
}

/// Print how to install FFmpeg.
fn print_ffmpeg_help() {
    println!("FFmpeg is required to decode audio, but it was not found.\n");
    for step in install_instructions(Os::current()) {
        println!("  - {step}");
    }
    println!("\nRun Whispry again once FFmpeg is installed.");
}

/// Cancel the job when the user presses Ctrl+C.
fn cancel_on_ctrl_c() -> CancelToken {
    let token = CancelToken::new();
    let handler = token.clone();
    // A failure only means Ctrl+C will stop the program the usual way.
    let _ = ctrlc::set_handler(move || handler.cancel());
    token
}

fn show_devices() -> ExitCode {
    let environment = JobEnvironment::detect();
    let system = &environment.system;
    println!("{}", system.os_label);
    println!("CPU threads: {}", system.cpu_threads);
    for option in &system.devices {
        if option.available {
            println!(
                "  [available]   {}: {}",
                option.label,
                option.compute_types.join(", ")
            );
        } else {
            let reason = option.reason.as_deref().unwrap_or("unavailable");
            println!("  [unavailable] {}: {reason}", option.label);
        }
    }
    if system.gpu().gpu_hint.is_some() {
        println!("For GPU support, run `whispry gpu`.");
    }
    match find_ffmpeg() {
        Some(tool) => println!("FFmpeg: {}", tool.ffmpeg.display()),
        None => println!("FFmpeg: not found"),
    }
    ExitCode::SUCCESS
}

fn check_ffmpeg() -> ExitCode {
    match find_ffmpeg() {
        Some(tool) => {
            println!("FFmpeg found: {}", tool.ffmpeg.display());
            ExitCode::SUCCESS
        }
        None => {
            print_ffmpeg_help();
            ExitCode::FAILURE
        }
    }
}

fn download(model: &str) -> ExitCode {
    let cancel = cancel_on_ctrl_c();
    let mut renderer = ConsoleRenderer::new();
    match ensure_model(model, &mut |event| renderer.handle(&event), &cancel, None) {
        Ok(path) => {
            println!("{}", path.display());
            ExitCode::SUCCESS
        }
        Err(whispry_core::models::ModelError::Cancelled) => {
            eprintln!("\nInterrupted by user.");
            ExitCode::from(INTERRUPTED)
        }
        Err(error) => fail(error),
    }
}

fn transcribe(args: TranscribeArgs) -> ExitCode {
    let config = JobConfig {
        audio: args.audio,
        model: args.model,
        language: args.language,
        output: args.output,
        device: args
            .device
            .parse::<DeviceChoice>()
            .expect("clap checked the device"),
        compute_type: args.compute_type,
        chunk_minutes: args.chunk_minutes,
    };
    if let Some(problem) = validate_config(&config).into_iter().next() {
        usage_error(problem);
    }

    let cancel = cancel_on_ctrl_c();
    let mut renderer = ConsoleRenderer::new();
    let environment = JobEnvironment::detect();

    match run_job(
        &config,
        &environment,
        &mut |event| renderer.handle(&event),
        &cancel,
    ) {
        Ok(_) => ExitCode::SUCCESS,
        Err(JobError::FFmpeg(_)) => {
            print_ffmpeg_help();
            ExitCode::FAILURE
        }
        Err(JobError::Cancelled(cancelled)) => {
            eprintln!("\nInterrupted by user.");
            if let Some(output) = cancelled.output {
                eprintln!("Text transcribed so far is in: {}", output.display());
            }
            ExitCode::from(INTERRUPTED)
        }
        Err(error) => fail(error),
    }
}

fn open_tui(audio: Option<PathBuf>) -> ExitCode {
    use std::io::IsTerminal;

    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        eprintln!(
            "The terminal UI needs an interactive terminal. Use `whispry transcribe` instead."
        );
        return ExitCode::FAILURE;
    }
    match crate::tui::run(audio) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => fail(error),
    }
}

pub fn run(cli: Cli) -> ExitCode {
    match cli.command {
        None => open_tui(None),
        Some(Command::Tui { audio }) => open_tui(audio),
        Some(Command::Transcribe(args)) => transcribe(args),
        Some(Command::Devices) => show_devices(),
        Some(Command::Gpu { install, yes }) => {
            if install {
                crate::gpu::install(yes)
            } else {
                crate::gpu::show_status()
            }
        }
        Some(Command::Ffmpeg) => check_ffmpeg(),
        Some(Command::Download { model }) => download(&model),
    }
}
