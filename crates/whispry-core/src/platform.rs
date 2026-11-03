//! Detect the operating system and which compute devices can run Whisper.
//!
//! This is a quick check done with no Python: it asks `nvidia-smi` whether a
//! GPU exists and looks for the CUDA libraries on disk. The transcription
//! worker has the last word, because it actually loads the model.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::config::DeviceChoice;
use crate::process::command;

/// Best choice first. The first one the device supports becomes the default.
pub const CUDA_COMPUTE_TYPES: &[&str] = &["int8_float16", "float16", "int8", "float32"];
pub const CPU_COMPUTE_TYPES: &[&str] = &["int8", "int8_float32", "float32"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Linux,
    Windows,
    MacOs,
    Other,
}

impl Os {
    pub fn current() -> Os {
        match std::env::consts::OS {
            "linux" => Os::Linux,
            "windows" => Os::Windows,
            "macos" => Os::MacOs,
            _ => Os::Other,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Os::Linux => "Linux",
            Os::Windows => "Windows",
            Os::MacOs => "macOS",
            Os::Other => std::env::consts::OS,
        }
    }

    /// Libraries CTranslate2 loads at runtime for CUDA. On Linux it only
    /// needs cuBLAS; the Windows build also uses cuDNN.
    pub fn cuda_libraries(self) -> &'static [&'static str] {
        match self {
            Os::Linux => &["libcublas.so.12"],
            Os::Windows => &["cublas64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"],
            Os::MacOs | Os::Other => &[],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Cuda => "cuda",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceOption {
    pub device: Device,
    pub label: String,
    pub available: bool,
    /// Why the device can't be used, when it can't.
    pub reason: Option<String>,
    pub compute_types: Vec<&'static str>,
}

impl DeviceOption {
    pub fn default_compute_type(&self) -> Option<&'static str> {
        self.compute_types.first().copied()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SystemInfo {
    pub os: Os,
    pub os_label: String,
    pub cpu_threads: usize,
    pub devices: Vec<DeviceOption>,
}

impl SystemInfo {
    pub fn device(&self, device: Device) -> &DeviceOption {
        self.devices
            .iter()
            .find(|option| option.device == device)
            .expect("every system lists both a CPU and a GPU option")
    }

    pub fn available_devices(&self) -> impl Iterator<Item = &DeviceOption> {
        self.devices.iter().filter(|option| option.available)
    }

    pub fn gpu(&self) -> &DeviceOption {
        self.device(Device::Cuda)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct DeviceUnavailable(pub String);

impl fmt::Display for DeviceUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DeviceUnavailable {}

pub fn detect_cpu() -> DeviceOption {
    DeviceOption {
        device: Device::Cpu,
        label: "CPU".into(),
        available: true,
        reason: None,
        compute_types: CPU_COMPUTE_TYPES.to_vec(),
    }
}

/// Number of NVIDIA GPUs in the output of `nvidia-smi -L`.
pub fn count_gpus(listing: &str) -> usize {
    listing
        .lines()
        .filter(|line| line.trim_start().starts_with("GPU "))
        .count()
}

fn query_gpu_count() -> Result<usize, String> {
    let output = command("nvidia-smi")
        .arg("-L")
        .output()
        .map_err(|error| error.to_string())?;
    if !output.status.success() {
        return Err(format!("nvidia-smi exited with {}", output.status));
    }
    Ok(count_gpus(&String::from_utf8_lossy(&output.stdout)))
}

/// Folders where pip-installed NVIDIA libraries live inside a Python
/// environment (`lib` on Linux, `bin` on Windows).
pub fn nvidia_wheel_dirs(venv: &Path, os: Os) -> Vec<PathBuf> {
    let (site_packages, folder): (Vec<PathBuf>, &str) = match os {
        Os::Windows => (vec![venv.join("Lib").join("site-packages")], "bin"),
        _ => {
            let lib = venv.join("lib");
            let found = std::fs::read_dir(&lib)
                .into_iter()
                .flatten()
                .flatten()
                .map(|entry| entry.path().join("site-packages"))
                .collect();
            (found, "lib")
        }
    };

    let mut dirs = Vec::new();
    for packages in site_packages {
        let Ok(entries) = std::fs::read_dir(packages.join("nvidia")) else {
            continue;
        };
        for entry in entries.flatten() {
            let candidate = entry.path().join(folder);
            if candidate.is_dir() {
                dirs.push(candidate);
            }
        }
    }
    dirs.sort();
    dirs
}

/// Folders the OS loader searches by default, plus common CUDA installs.
pub fn system_library_dirs(os: Os) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let search_path = match os {
        Os::Windows => "PATH",
        _ => "LD_LIBRARY_PATH",
    };
    if let Some(value) = std::env::var_os(search_path) {
        dirs.extend(std::env::split_paths(&value));
    }

    match os {
        Os::Linux => dirs.extend(
            [
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib64",
                "/usr/lib",
                "/usr/local/cuda/lib64",
                "/opt/cuda/lib64",
            ]
            .map(PathBuf::from),
        ),
        Os::Windows => {
            if let Some(cuda) = std::env::var_os("CUDA_PATH") {
                dirs.push(PathBuf::from(cuda).join("bin"));
            }
        }
        Os::MacOs | Os::Other => {}
    }
    dirs
}

/// CUDA libraries for this OS that are not in any of `dirs`.
pub fn missing_cuda_libraries(os: Os, dirs: &[PathBuf]) -> Vec<&'static str> {
    os.cuda_libraries()
        .iter()
        .copied()
        .filter(|name| !dirs.iter().any(|dir| dir.join(name).is_file()))
        .collect()
}

/// Decide whether the GPU is usable, from what the checks found.
pub fn cuda_option(
    os: Os,
    gpus: Result<usize, String>,
    missing_libraries: &[&str],
) -> DeviceOption {
    let unavailable = |reason: String| DeviceOption {
        device: Device::Cuda,
        label: "NVIDIA GPU (CUDA)".into(),
        available: false,
        reason: Some(reason),
        compute_types: Vec::new(),
    };

    if os.cuda_libraries().is_empty() {
        return unavailable(format!("CUDA is not supported on {}", os.name()));
    }

    let count = match gpus {
        Ok(count) => count,
        Err(_) => return unavailable("No NVIDIA GPU or driver found".into()),
    };
    if count == 0 {
        return unavailable("No NVIDIA GPU or driver found".into());
    }

    if !missing_libraries.is_empty() {
        return unavailable(format!(
            "NVIDIA GPU found, but CUDA libraries are missing ({}). \
             Run setup again with --gpu to install them.",
            missing_libraries.join(", ")
        ));
    }

    let label = if count == 1 {
        "NVIDIA GPU (CUDA)".to_string()
    } else {
        format!("NVIDIA GPU (CUDA, {count} devices)")
    };
    DeviceOption {
        device: Device::Cuda,
        label,
        available: true,
        reason: None,
        compute_types: CUDA_COMPUTE_TYPES.to_vec(),
    }
}

/// Look for a GPU and the CUDA libraries. `extra_library_dirs` are searched
/// first, so pass the worker environment's NVIDIA wheel folders there.
pub fn detect_cuda(os: Os, extra_library_dirs: &[PathBuf]) -> DeviceOption {
    let gpus = query_gpu_count();
    let missing = match gpus {
        Ok(count) if count > 0 => {
            let mut dirs = extra_library_dirs.to_vec();
            dirs.extend(system_library_dirs(os));
            missing_cuda_libraries(os, &dirs)
        }
        _ => Vec::new(),
    };
    cuda_option(os, gpus, &missing)
}

/// Inspect this machine.
pub fn detect_system(extra_library_dirs: &[PathBuf]) -> SystemInfo {
    let os = Os::current();
    SystemInfo {
        os,
        os_label: format!("{} ({})", os.name(), std::env::consts::ARCH),
        cpu_threads: std::thread::available_parallelism().map_or(1, |n| n.get()),
        devices: vec![detect_cuda(os, extra_library_dirs), detect_cpu()],
    }
}

/// Pick the device and compute type to use for a job.
///
/// `Auto` prefers the GPU and falls back to the CPU.
pub fn resolve_device(
    requested: DeviceChoice,
    system: &SystemInfo,
    compute_type: Option<&str>,
) -> Result<(DeviceOption, String), DeviceUnavailable> {
    let option = match requested {
        DeviceChoice::Auto => system
            .available_devices()
            .next()
            .ok_or_else(|| DeviceUnavailable("No usable device found".into()))?,
        DeviceChoice::Cpu | DeviceChoice::Cuda => {
            let device = if requested == DeviceChoice::Cpu {
                Device::Cpu
            } else {
                Device::Cuda
            };
            let option = system.device(device);
            if !option.available {
                return Err(DeviceUnavailable(format!(
                    "{} is unavailable: {}",
                    option.label,
                    option.reason.as_deref().unwrap_or("unknown reason")
                )));
            }
            option
        }
    };

    let compute_type = match compute_type {
        None => option
            .default_compute_type()
            .ok_or_else(|| DeviceUnavailable(format!("{} has no compute types", option.label)))?
            .to_string(),
        Some(requested) if option.compute_types.contains(&requested) => requested.to_string(),
        Some(requested) => {
            return Err(DeviceUnavailable(format!(
                "{} does not support compute type {requested}. Supported: {}",
                option.label,
                option.compute_types.join(", ")
            )));
        }
    };

    Ok((option.clone(), compute_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> DeviceOption {
        cuda_option(Os::Linux, Ok(1), &[])
    }

    fn no_gpu() -> DeviceOption {
        cuda_option(Os::Linux, Ok(0), &[])
    }

    fn system(gpu: DeviceOption) -> SystemInfo {
        SystemInfo {
            os: Os::Linux,
            os_label: "Linux".into(),
            cpu_threads: 4,
            devices: vec![gpu, detect_cpu()],
        }
    }

    #[test]
    fn auto_prefers_gpu() {
        let (option, compute) = resolve_device(DeviceChoice::Auto, &system(gpu()), None).unwrap();
        assert_eq!(
            (option.device, compute.as_str()),
            (Device::Cuda, "int8_float16")
        );
    }

    #[test]
    fn auto_falls_back_to_cpu() {
        let (option, compute) =
            resolve_device(DeviceChoice::Auto, &system(no_gpu()), None).unwrap();
        assert_eq!((option.device, compute.as_str()), (Device::Cpu, "int8"));
    }

    #[test]
    fn explicit_unavailable_gpu_explains_why() {
        let error = resolve_device(DeviceChoice::Cuda, &system(no_gpu()), None).unwrap_err();
        assert!(error.to_string().contains("No NVIDIA GPU"));
    }

    #[test]
    fn rejects_unsupported_compute_type() {
        let error =
            resolve_device(DeviceChoice::Cpu, &system(gpu()), Some("int8_float16")).unwrap_err();
        assert!(error.to_string().contains("does not support"));
    }

    #[test]
    fn accepts_supported_compute_type() {
        let (_, compute) =
            resolve_device(DeviceChoice::Cpu, &system(gpu()), Some("float32")).unwrap();
        assert_eq!(compute, "float32");
    }

    #[test]
    fn gpu_with_missing_libraries_names_them() {
        let option = cuda_option(Os::Windows, Ok(1), &["cudnn_ops64_9.dll"]);
        assert!(!option.available);
        assert!(option.reason.unwrap().contains("cudnn_ops64_9.dll"));
    }

    #[test]
    fn gpu_with_libraries_is_available_with_default_compute_type() {
        let option = cuda_option(Os::Windows, Ok(2), &[]);
        assert!(option.available);
        assert_eq!(option.label, "NVIDIA GPU (CUDA, 2 devices)");
        assert_eq!(option.default_compute_type(), Some("int8_float16"));
    }

    #[test]
    fn missing_driver_and_zero_gpus_look_the_same_to_the_user() {
        let failed = cuda_option(Os::Linux, Err("not found".into()), &[]);
        assert_eq!(failed.reason, no_gpu().reason);
    }

    #[test]
    fn cuda_is_unsupported_on_macos() {
        let option = cuda_option(Os::MacOs, Ok(1), &[]);
        assert!(option.reason.unwrap().contains("not supported"));
    }

    #[test]
    fn counts_gpus_in_nvidia_smi_listing() {
        let listing = "GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-abc)\n\
                       GPU 1: NVIDIA GeForce RTX 3060 (UUID: GPU-def)\n";
        assert_eq!(count_gpus(listing), 2);
        assert_eq!(count_gpus(""), 0);
        assert_eq!(count_gpus("No devices were found"), 0);
    }

    #[test]
    fn finds_missing_libraries_across_folders() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        std::fs::write(first.path().join("cublas64_12.dll"), "").unwrap();
        std::fs::write(second.path().join("cudnn_ops64_9.dll"), "").unwrap();
        let dirs = vec![first.path().to_path_buf(), second.path().to_path_buf()];
        assert_eq!(
            missing_cuda_libraries(Os::Windows, &dirs),
            vec!["cudnn_cnn64_9.dll"]
        );
    }

    #[test]
    fn finds_wheel_folders_on_linux_and_windows() {
        let linux = tempfile::tempdir().unwrap();
        let cublas = linux
            .path()
            .join("lib/python3.12/site-packages/nvidia/cublas/lib");
        std::fs::create_dir_all(&cublas).unwrap();
        assert_eq!(nvidia_wheel_dirs(linux.path(), Os::Linux), vec![cublas]);

        let windows = tempfile::tempdir().unwrap();
        let cudnn = windows.path().join("Lib/site-packages/nvidia/cudnn/bin");
        std::fs::create_dir_all(&cudnn).unwrap();
        assert_eq!(nvidia_wheel_dirs(windows.path(), Os::Windows), vec![cudnn]);
    }

    #[test]
    fn no_wheel_folders_without_a_venv() {
        assert!(nvidia_wheel_dirs(Path::new("/nonexistent"), Os::Linux).is_empty());
    }

    #[test]
    fn detects_this_machine_without_panicking() {
        let info = detect_system(&[]);
        assert!(info.cpu_threads >= 1);
        assert!(info.device(Device::Cpu).available);
        assert!(!info.gpu().label.is_empty());
    }
}
