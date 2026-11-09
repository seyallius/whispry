//! `whispry gpu`: show whether the GPU can be used, and (only when asked)
//! download the NVIDIA libraries it needs.

use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use whispry_core::cancel::CancelToken;
use whispry_core::cuda_libs::{self, CudaLibsError};
use whispry_core::platform::{CUDA_BUILD, Device, Os, SystemInfo, detect_system};

/// Shown when the size cannot be looked up (for instance when offline).
const FALLBACK_SIZE: &str = "about 550 MB";

/// Exit code for a request that is refused (not a failure to carry it out).
const REFUSED: u8 = 2;

/// Exit code for a run stopped with Ctrl+C.
const INTERRUPTED: u8 = 130;

/// Where this machine stands with the GPU. Only [`GpuStatus::NeedsLibraries`]
/// can lead to a download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuStatus {
    /// No NVIDIA GPU, or an OS without CUDA. Nothing to offer.
    NoGpu { reason: String },
    /// A GPU is present, but this is the CPU edition.
    NeedsCudaEdition,
    /// The GPU edition and its libraries are all in place.
    Ready { libraries: Option<PathBuf> },
    /// The GPU edition, a GPU, but no libraries yet.
    NeedsLibraries,
}

/// Decide the status from what was detected. Pure: no I/O.
pub fn gpu_action(system: &SystemInfo, cuda_build: bool, existing: Option<PathBuf>) -> GpuStatus {
    let gpu = system.device(Device::Cuda);
    let gpu_present = gpu.available || gpu.gpu_hint.is_some();
    if !gpu_present || matches!(system.os, Os::MacOs | Os::Other) {
        return GpuStatus::NoGpu {
            reason: gpu
                .reason
                .clone()
                .unwrap_or_else(|| "No NVIDIA GPU found".into()),
        };
    }
    if !cuda_build {
        return GpuStatus::NeedsCudaEdition;
    }
    match existing {
        Some(dir) => GpuStatus::Ready {
            libraries: Some(dir),
        },
        None if gpu.available => GpuStatus::Ready { libraries: None },
        None => GpuStatus::NeedsLibraries,
    }
}

fn cuda_edition_text() -> String {
    format!(
        "GPU use needs the CUDA edition of this release (asset name \
         `whispry-<version>-<target>-cuda`), from\n  {}/releases\n\
         Nothing is downloaded by this command. Whispry keeps working on your CPU meanwhile.",
        env!("CARGO_PKG_REPOSITORY")
    )
}

fn no_gpu_text() -> &'static str {
    "No NVIDIA GPU found. Whispry runs on your CPU; nothing to install."
}

fn size_text(os: Os) -> String {
    match cuda_libs::plan(os) {
        Ok(plan) => cuda_libs::size_label(plan.size),
        Err(_) => FALLBACK_SIZE.into(),
    }
}

fn current_status() -> (SystemInfo, GpuStatus) {
    let system = detect_system(&cuda_libs::installed_dirs());
    let existing = cuda_libs::find_existing(system.os, &[]);
    let status = gpu_action(&system, CUDA_BUILD, existing);
    (system, status)
}

pub fn show_status() -> ExitCode {
    let (system, status) = current_status();
    match status {
        GpuStatus::NoGpu { .. } => println!("{}", no_gpu_text()),
        GpuStatus::NeedsCudaEdition => println!("NVIDIA GPU found.\n{}", cuda_edition_text()),
        GpuStatus::Ready { libraries } => {
            println!("GPU ready.");
            if let Some(dir) = libraries {
                println!("GPU libraries found in: {}", dir.display());
            }
            println!("No download is needed.");
        }
        GpuStatus::NeedsLibraries => {
            println!("NVIDIA GPU found, but the GPU libraries (cuBLAS) are missing.");
            println!("They are a download of {}.", size_text(system.os));
            println!(
                "They would be saved in: {}",
                cuda_libs::cache_dir().display()
            );
            println!("Nothing is downloaded until you run `whispry gpu --install`.");
        }
    }
    ExitCode::SUCCESS
}

fn refuse(message: &str) -> ExitCode {
    eprintln!("{message}");
    ExitCode::from(REFUSED)
}

/// Whether the user agrees: a "y"/"yes" answer. Anything else is "no".
fn is_yes(answer: &str) -> bool {
    matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes")
}

fn confirm(question: &str) -> bool {
    print!("{question} [y/N] ");
    let _ = std::io::stdout().flush();
    let mut answer = String::new();
    std::io::stdin().lock().read_line(&mut answer).is_ok() && is_yes(&answer)
}

pub fn install(yes: bool) -> ExitCode {
    let (system, status) = current_status();
    match status {
        GpuStatus::NoGpu { .. } => return refuse(no_gpu_text()),
        GpuStatus::NeedsCudaEdition => {
            return refuse(&format!(
                "This is the CPU edition of Whispry.\n{}",
                cuda_edition_text()
            ));
        }
        GpuStatus::Ready { libraries } => {
            match libraries {
                Some(dir) => println!("The GPU libraries are already installed: {}", dir.display()),
                None => println!("The GPU libraries are already installed."),
            }
            println!("Nothing was downloaded.");
            return ExitCode::SUCCESS;
        }
        GpuStatus::NeedsLibraries => {}
    }

    let plan = match cuda_libs::plan(system.os) {
        Ok(plan) => plan,
        Err(error) => {
            eprintln!("\nError: {error}");
            return ExitCode::FAILURE;
        }
    };
    let dest = cuda_libs::cache_dir();
    let size = cuda_libs::size_label(plan.size);
    println!("Whispry needs NVIDIA's cuBLAS libraries to use your GPU.");
    println!("They will be saved in: {}", dest.display());
    if !yes {
        if !std::io::stdin().is_terminal() {
            return refuse(&format!(
                "Not downloading {size}: run `whispry gpu --install --yes` to confirm without a prompt."
            ));
        }
        if !confirm(&format!("Download {size} now?")) {
            println!("Nothing was downloaded.");
            return ExitCode::SUCCESS;
        }
    }

    let cancel = CancelToken::new();
    let handler = cancel.clone();
    let _ = ctrlc::set_handler(move || handler.cancel());

    let bar = ProgressBar::new(plan.size);
    bar.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
    bar.set_style(
        ProgressStyle::with_template(
            "{prefix:>10} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
        )
        .expect("the template is valid")
        .progress_chars("=> "),
    );
    bar.set_prefix("GPU libs");
    let result = cuda_libs::install(&plan, &dest, &mut |done, _| bar.set_position(done), &cancel);
    bar.finish_and_clear();
    match result {
        Ok(path) => {
            println!("Installed the GPU libraries in: {}", path.display());
            println!("Whispry will use the GPU automatically.");
            ExitCode::SUCCESS
        }
        Err(CudaLibsError::Cancelled) => {
            eprintln!("\nInterrupted by user.");
            ExitCode::from(INTERRUPTED)
        }
        Err(error) => {
            eprintln!("\nError: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use whispry_core::platform::{cuda_option, detect_cpu};

    fn system(os: Os, gpus: usize, missing: &[&str], cuda_build: bool) -> SystemInfo {
        SystemInfo {
            os,
            os_label: "test".into(),
            cpu_threads: 4,
            devices: vec![cuda_option(os, Ok(gpus), missing, cuda_build), detect_cpu()],
        }
    }

    fn dir() -> Option<PathBuf> {
        Some(PathBuf::from("/libs"))
    }

    #[test]
    fn no_gpu_or_macos_never_offers_anything() {
        for (os, gpus) in [
            (Os::Linux, 0),
            (Os::Windows, 0),
            (Os::MacOs, 0),
            (Os::MacOs, 1),
            (Os::Other, 1),
        ] {
            for cuda_build in [false, true] {
                for existing in [None, dir()] {
                    let status =
                        gpu_action(&system(os, gpus, &[], cuda_build), cuda_build, existing);
                    assert!(
                        matches!(status, GpuStatus::NoGpu { .. }),
                        "{os:?} {gpus} {cuda_build}: {status:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn no_gpu_reason_mentions_no_download() {
        let GpuStatus::NoGpu { reason } = gpu_action(&system(Os::Linux, 0, &[], true), true, None)
        else {
            panic!("expected NoGpu");
        };
        assert_eq!(reason, "No NVIDIA GPU found");
        assert!(!no_gpu_text().to_lowercase().contains("download"));
        assert!(!no_gpu_text().contains("CUDA edition"));
    }

    #[test]
    fn cpu_edition_with_a_gpu_needs_the_cuda_edition() {
        let info = system(Os::Linux, 1, &[], false);
        assert_eq!(gpu_action(&info, false, None), GpuStatus::NeedsCudaEdition);
        assert_eq!(gpu_action(&info, false, dir()), GpuStatus::NeedsCudaEdition);
        assert!(cuda_edition_text().contains("whispry-<version>-<target>-cuda"));
        assert!(cuda_edition_text().contains("https://github.com/seyallius/whispry/releases"));
    }

    #[test]
    fn cuda_edition_with_libraries_is_ready() {
        let found = system(Os::Linux, 1, &[], true);
        assert_eq!(
            gpu_action(&found, true, dir()),
            GpuStatus::Ready { libraries: dir() }
        );
        assert_eq!(
            gpu_action(&found, true, None),
            GpuStatus::Ready { libraries: None }
        );
    }

    #[test]
    fn cuda_edition_without_libraries_can_download() {
        let missing = system(Os::Windows, 1, &["cublas64_12.dll"], true);
        assert_eq!(gpu_action(&missing, true, None), GpuStatus::NeedsLibraries);
        // Libraries found elsewhere win over the detection result.
        assert_eq!(
            gpu_action(&missing, true, dir()),
            GpuStatus::Ready { libraries: dir() }
        );
    }

    #[test]
    fn only_one_state_can_download() {
        let cases = [
            gpu_action(&system(Os::Linux, 0, &[], true), true, None),
            gpu_action(&system(Os::MacOs, 1, &[], true), true, None),
            gpu_action(&system(Os::Linux, 1, &[], false), false, None),
            gpu_action(&system(Os::Linux, 1, &[], true), true, dir()),
        ];
        assert!(cases.iter().all(|s| *s != GpuStatus::NeedsLibraries));
    }

    #[test]
    fn answers_default_to_no() {
        assert!(is_yes("y\n"));
        assert!(is_yes(" YES "));
        assert!(!is_yes("\n"));
        assert!(!is_yes("n"));
        assert!(!is_yes("maybe"));
    }
}
