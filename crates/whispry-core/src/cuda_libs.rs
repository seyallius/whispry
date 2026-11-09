//! The NVIDIA libraries the GPU edition loads at run time (opt-in download).
//!
//! The CUDA edition needs cuBLAS only. Nothing is fetched unless the user
//! asks: [`plan`] says what would be downloaded, [`install`] does it. The
//! libraries come out of NVIDIA's `nvidia-cublas-cu12` wheel on PyPI.

use std::ffi::OsString;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::cancel::CancelToken;
use crate::platform::{Os, nvidia_wheel_dirs, system_library_dirs};

const PYPI_URL: &str = "https://pypi.org/pypi/nvidia-cublas-cu12/json";

/// Set on the re-run copy of the program so it does not re-run again.
pub const RELAUNCH_MARKER: &str = "WHISPRY_RELAUNCHED";

/// Progress is reported at most once per this many bytes.
const PROGRESS_STEP: u64 = 1 << 20;

#[derive(Debug, thiserror::Error)]
pub enum CudaLibsError {
    #[error("The GPU libraries are not available for {0}")]
    Unsupported(String),
    #[error("Could not download the GPU libraries: {0}")]
    Network(String),
    #[error("The downloaded GPU libraries are damaged (checksum mismatch). Please try again.")]
    HashMismatch,
    #[error("The downloaded package does not contain {0}")]
    MissingInWheel(String),
    #[error("The downloaded package could not be read: {0}")]
    BadArchive(String),
    #[error("Could not save the GPU libraries: {0}")]
    Io(#[from] std::io::Error),
    #[error("Download cancelled")]
    Cancelled,
}

fn network_error(error: impl std::fmt::Display) -> CudaLibsError {
    CudaLibsError::Network(error.to_string())
}

/// What [`install`] would download.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DownloadPlan {
    pub os: Os,
    pub version: String,
    pub url: String,
    pub sha256: String,
    pub size: u64,
}

/// The library files the engine loads.
pub fn required_files(os: Os) -> &'static [&'static str] {
    match os {
        Os::Linux => &["libcublas.so.12", "libcublasLt.so.12"],
        Os::Windows => &["cublas64_12.dll", "cublasLt64_12.dll"],
        Os::MacOs | Os::Other => &[],
    }
}

/// Where the libraries are downloaded to, following each OS's convention.
pub fn cache_dir() -> PathBuf {
    cache_dir_for(
        Os::current(),
        |name| std::env::var_os(name),
        std::env::home_dir(),
    )
}

fn cache_dir_for(os: Os, env: impl Fn(&str) -> Option<OsString>, home: Option<PathBuf>) -> PathBuf {
    let home = home.unwrap_or_else(|| PathBuf::from("."));
    let base = match os {
        Os::Windows => env("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join("AppData").join("Local")),
        Os::MacOs => home.join("Library").join("Caches"),
        Os::Linux | Os::Other => env("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".cache")),
    };
    base.join("whispry").join("cuda")
}

fn holds_all(os: Os, dir: &Path) -> bool {
    let files = required_files(os);
    !files.is_empty() && files.iter().all(|name| dir.join(name).is_file())
}

/// Folders that hold downloaded CUDA libraries, to search before the system's.
pub fn installed_dirs() -> Vec<PathBuf> {
    installed_dirs_in(Os::current(), &cache_dir())
}

fn installed_dirs_in(os: Os, cache: &Path) -> Vec<PathBuf> {
    if holds_all(os, cache) {
        vec![cache.to_path_buf()]
    } else {
        Vec::new()
    }
}

/// Virtual environments (`.venv`, `venv`) in `starts` and their ancestors.
fn venv_dirs(starts: &[PathBuf]) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    for start in starts {
        for dir in start.ancestors() {
            for name in [".venv", "venv"] {
                let venv = dir.join(name);
                if venv.is_dir() && !found.contains(&venv) {
                    found.push(venv);
                }
            }
        }
    }
    found
}

/// cuBLAS folders of pip-installed NVIDIA wheels inside those virtual envs.
fn venv_cublas_dirs(os: Os, starts: &[PathBuf]) -> Vec<PathBuf> {
    venv_dirs(starts)
        .iter()
        .flat_map(|venv| nvidia_wheel_dirs(venv, os))
        .filter(|dir| {
            dir.parent()
                .and_then(Path::file_name)
                .is_some_and(|name| name == "cublas")
        })
        .collect()
}

/// A folder that already holds every required library, so no download is
/// needed: the cache, the system's library folders, a Python virtual
/// environment near here or the executable, then `extra_dirs`.
pub fn find_existing(os: Os, extra_dirs: &[PathBuf]) -> Option<PathBuf> {
    let mut starts = Vec::new();
    starts.extend(std::env::current_dir().ok());
    starts.extend(
        std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(Path::to_path_buf)),
    );
    find_existing_in(
        os,
        &cache_dir(),
        &system_library_dirs(os),
        &starts,
        extra_dirs,
    )
}

fn find_existing_in(
    os: Os,
    cache: &Path,
    system_dirs: &[PathBuf],
    venv_starts: &[PathBuf],
    extra_dirs: &[PathBuf],
) -> Option<PathBuf> {
    std::iter::once(cache.to_path_buf())
        .chain(system_dirs.iter().cloned())
        .chain(venv_cublas_dirs(os, venv_starts))
        .chain(extra_dirs.iter().cloned())
        .find(|dir| holds_all(os, dir))
}

/// "553 MB" (decimal megabytes, as download sizes are usually quoted).
pub fn size_label(bytes: u64) -> String {
    const MB: u64 = 1_000_000;
    if bytes >= 1000 * MB {
        format!("{:.1} GB", bytes as f64 / 1e9)
    } else {
        format!("{} MB", ((bytes + MB / 2) / MB).max(1))
    }
}

// ---- Choosing what to download -------------------------------------------

fn http_agent() -> ureq::Agent {
    ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .build()
        .into()
}

/// Ask PyPI which cuBLAS package to download.
pub fn plan(os: Os) -> Result<DownloadPlan, CudaLibsError> {
    plan_from(&http_agent(), PYPI_URL, os)
}

fn plan_from(agent: &ureq::Agent, url: &str, os: Os) -> Result<DownloadPlan, CudaLibsError> {
    check_supported(os)?;
    let text = agent
        .get(url)
        .call()
        .map_err(network_error)?
        .body_mut()
        .read_to_string()
        .map_err(network_error)?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(network_error)?;
    choose_wheel(&json, os)
}

fn check_supported(os: Os) -> Result<(), CudaLibsError> {
    let arch_ok = std::env::consts::ARCH == "x86_64";
    if matches!(os, Os::Linux | Os::Windows) && arch_ok {
        Ok(())
    } else {
        Err(CudaLibsError::Unsupported(format!(
            "{} ({})",
            os.name(),
            std::env::consts::ARCH
        )))
    }
}

/// Numeric parts of "12.9.1.4"; `None` for pre-releases or other oddities.
fn parse_version(version: &str) -> Option<Vec<u64>> {
    version
        .split('.')
        .map(|part| part.parse::<u64>().ok())
        .collect()
}

/// Accepted releases: 12.4 up to, not including, 13.
fn version_in_range(parts: &[u64]) -> bool {
    parts.first() == Some(&12) && parts.get(1).copied().unwrap_or(0) >= 4
}

fn wheel_matches(filename: &str, os: Os) -> bool {
    match os {
        Os::Linux => {
            filename.ends_with(".whl")
                && filename.contains("manylinux")
                && filename.ends_with("x86_64.whl")
        }
        Os::Windows => filename.ends_with("win_amd64.whl"),
        Os::MacOs | Os::Other => false,
    }
}

/// Pick the newest acceptable wheel from PyPI's JSON description of the
/// package (`releases`: version -> files).
pub fn choose_wheel(json: &serde_json::Value, os: Os) -> Result<DownloadPlan, CudaLibsError> {
    if required_files(os).is_empty() {
        return Err(CudaLibsError::Unsupported(os.name().to_string()));
    }
    let releases = json["releases"]
        .as_object()
        .ok_or_else(|| network_error("unexpected answer from the package index"))?;

    let mut best: Option<(Vec<u64>, DownloadPlan)> = None;
    for (version, files) in releases {
        let Some(parts) = parse_version(version).filter(|parts| version_in_range(parts)) else {
            continue;
        };
        if best.as_ref().is_some_and(|(newest, _)| *newest >= parts) {
            continue;
        }
        for file in files.as_array().into_iter().flatten() {
            let usable = file["yanked"].as_bool() != Some(true)
                && file["packagetype"]
                    .as_str()
                    .is_none_or(|t| t == "bdist_wheel");
            let (Some(name), Some(url), Some(sha256), Some(size)) = (
                file["filename"].as_str(),
                file["url"].as_str(),
                file["digests"]["sha256"].as_str(),
                file["size"].as_u64(),
            ) else {
                continue;
            };
            if usable && wheel_matches(name, os) {
                best = Some((
                    parts.clone(),
                    DownloadPlan {
                        os,
                        version: version.clone(),
                        url: url.to_string(),
                        sha256: sha256.to_ascii_lowercase(),
                        size,
                    },
                ));
                break;
            }
        }
    }
    best.map(|(_, plan)| plan).ok_or_else(|| {
        CudaLibsError::Unsupported(format!("{}: no matching package was found", os.name()))
    })
}

// ---- Downloading ----------------------------------------------------------

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// Download the package, check it, and put the library files in `dest`.
///
/// Nothing is left behind on failure or cancellation, and `dest` only
/// appears once it is complete.
pub fn install(
    plan: &DownloadPlan,
    dest: &Path,
    on_progress: &mut dyn FnMut(u64, u64),
    cancel: &CancelToken,
) -> Result<PathBuf, CudaLibsError> {
    install_with(&http_agent(), plan, dest, on_progress, cancel)
}

fn install_with(
    agent: &ureq::Agent,
    plan: &DownloadPlan,
    dest: &Path,
    on_progress: &mut dyn FnMut(u64, u64),
    cancel: &CancelToken,
) -> Result<PathBuf, CudaLibsError> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let wheel = dest.with_extension("whl.part");
    let staging = dest.with_extension("part");

    let result = download_wheel(agent, plan, &wheel, on_progress, cancel)
        .and_then(|()| extract_libraries(plan.os, &wheel, &staging, cancel))
        .and_then(|()| publish(&staging, dest));
    let _ = std::fs::remove_file(&wheel);
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result?;
    on_progress(plan.size, plan.size);
    Ok(dest.to_path_buf())
}

fn download_wheel(
    agent: &ureq::Agent,
    plan: &DownloadPlan,
    wheel: &Path,
    on_progress: &mut dyn FnMut(u64, u64),
    cancel: &CancelToken,
) -> Result<(), CudaLibsError> {
    let mut response = agent.get(&plan.url).call().map_err(network_error)?;
    let mut source = response.body_mut().as_reader();
    let mut file = File::create(wheel)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 256 * 1024];
    let (mut done, mut reported) = (0u64, 0u64);
    on_progress(0, plan.size);
    loop {
        if cancel.is_cancelled() {
            return Err(CudaLibsError::Cancelled);
        }
        let count = source.read(&mut buffer).map_err(network_error)?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
        file.write_all(&buffer[..count])?;
        done += count as u64;
        if done - reported >= PROGRESS_STEP {
            reported = done;
            on_progress(done, plan.size);
        }
    }
    file.flush()?;
    if hex(&hasher.finalize()) != plan.sha256.to_ascii_lowercase() {
        return Err(CudaLibsError::HashMismatch);
    }
    Ok(())
}

/// Copy only the required libraries out of the wheel into `staging`.
///
/// Files are looked up by their exact expected path in the archive and
/// written under their own base name, so no archive path can escape.
fn extract_libraries(
    os: Os,
    wheel: &Path,
    staging: &Path,
    cancel: &CancelToken,
) -> Result<(), CudaLibsError> {
    let folder = match os {
        Os::Windows => "bin",
        _ => "lib",
    };
    let mut archive = zip::ZipArchive::new(File::open(wheel)?)
        .map_err(|error| CudaLibsError::BadArchive(error.to_string()))?;

    let _ = std::fs::remove_dir_all(staging);
    std::fs::create_dir_all(staging)?;
    let mut buffer = vec![0u8; 256 * 1024];
    for name in required_files(os) {
        let member = format!("nvidia/cublas/{folder}/{name}");
        let mut entry = archive
            .by_name(&member)
            .map_err(|_| CudaLibsError::MissingInWheel((*name).to_string()))?;
        if !entry.is_file() {
            return Err(CudaLibsError::MissingInWheel((*name).to_string()));
        }
        let mut out = File::create(staging.join(name))?;
        loop {
            if cancel.is_cancelled() {
                return Err(CudaLibsError::Cancelled);
            }
            let count = entry
                .read(&mut buffer)
                .map_err(|error| CudaLibsError::BadArchive(error.to_string()))?;
            if count == 0 {
                break;
            }
            out.write_all(&buffer[..count])?;
        }
        out.flush()?;
    }
    Ok(())
}

fn publish(staging: &Path, dest: &Path) -> Result<(), CudaLibsError> {
    if dest.exists() {
        std::fs::remove_dir_all(dest)?;
    }
    Ok(std::fs::rename(staging, dest)?)
}

// ---- Putting the libraries on the loader path -------------------------------

/// The environment variable to set for the re-run, or `None` when the
/// program can carry on as it is.
///
/// `dirs` are the library folders, `current_env` reads the present
/// environment. Nothing is needed when the folders are already on the
/// loader path, when this is already the re-run, or on other systems.
pub fn relaunch_environment(
    os: Os,
    dirs: &[PathBuf],
    current_env: &dyn Fn(&str) -> Option<OsString>,
) -> Option<(String, OsString)> {
    let var = match os {
        Os::Linux => "LD_LIBRARY_PATH",
        Os::Windows => "PATH",
        Os::MacOs | Os::Other => return None,
    };
    if dirs.is_empty() || current_env(RELAUNCH_MARKER).is_some() {
        return None;
    }
    let existing = current_env(var).unwrap_or_default();
    let present: Vec<PathBuf> = std::env::split_paths(&existing)
        .filter(|path| !path.as_os_str().is_empty())
        .collect();
    if dirs.iter().all(|dir| present.contains(dir)) {
        return None;
    }
    let combined = std::env::join_paths(dirs.iter().cloned().chain(present)).ok()?;
    Some((var.to_string(), combined))
}

/// For the CUDA edition: run this program again with the downloaded
/// libraries on the loader path (glibc ignores changes made after start-up).
/// Returns the exit code to quit with, or `None` to carry on in this process.
#[allow(unexpected_cfgs)]
pub fn relaunch_if_needed() -> Option<i32> {
    if !cfg!(feature = "cuda") {
        return None;
    }
    let (var, value) = relaunch_environment(Os::current(), &installed_dirs(), &|name| {
        std::env::var_os(name)
    })?;
    let exe = std::env::current_exe().ok()?;
    let status = std::process::Command::new(exe)
        .args(std::env::args_os().skip(1))
        .env(var, value)
        .env(RELAUNCH_MARKER, "1")
        .status()
        .ok()?;
    // No code means the child was killed by a signal.
    Some(status.code().unwrap_or(130))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Cursor};
    use std::net::TcpListener;

    fn sha256_hex(bytes: &[u8]) -> String {
        hex(&Sha256::digest(bytes))
    }

    fn file(name: &str, extra: serde_json::Value) -> serde_json::Value {
        let mut value = serde_json::json!({
            "filename": name,
            "url": format!("https://files.example/{name}"),
            "digests": {"sha256": "ABCD"},
            "size": 42,
            "yanked": false,
            "packagetype": "bdist_wheel",
        });
        for (key, v) in extra.as_object().into_iter().flatten() {
            value[key] = v.clone();
        }
        value
    }

    fn fixture() -> serde_json::Value {
        serde_json::json!({
            "releases": {
                "12.4.5.8": [
                    file("n-12.4.5.8-py3-none-manylinux2014_x86_64.whl", serde_json::json!({})),
                    file("n-12.4.5.8-py3-none-win_amd64.whl", serde_json::json!({})),
                ],
                "12.9.1.4": [
                    file("n-12.9.1.4-py3-none-manylinux_2_27_aarch64.whl", serde_json::json!({})),
                    file("n-12.9.1.4-py3-none-manylinux_2_27_x86_64.whl", serde_json::json!({})),
                    file("n-12.9.1.4-py3-none-win_amd64.whl", serde_json::json!({})),
                ],
                "12.9.3.13": [
                    file("n-12.9.3.13-py3-none-manylinux_2_27_x86_64.whl",
                        serde_json::json!({"yanked": true})),
                ],
                "12.10.0.1": [
                    file("n-12.10.0.1-py3-none-win_amd64.whl", serde_json::json!({})),
                ],
                "12.3.0.1": [
                    file("n-12.3.0.1-py3-none-win_amd64.whl", serde_json::json!({})),
                ],
                "13.0.0.1": [
                    file("n-13.0.0.1-py3-none-manylinux_2_27_x86_64.whl", serde_json::json!({})),
                    file("n-13.0.0.1-py3-none-win_amd64.whl", serde_json::json!({})),
                ],
                "12.9.9rc1": [
                    file("n-12.9.9rc1-py3-none-win_amd64.whl", serde_json::json!({})),
                ],
            }
        })
    }

    #[test]
    fn chooses_the_newest_matching_wheel_in_range() {
        let linux = choose_wheel(&fixture(), Os::Linux).unwrap();
        // 12.9.3.13 is yanked and 13.x is out of range.
        assert_eq!(linux.version, "12.9.1.4");
        assert!(linux.url.ends_with("manylinux_2_27_x86_64.whl"));
        assert_eq!(linux.sha256, "abcd");
        assert_eq!(linux.size, 42);
        assert_eq!(linux.os, Os::Linux);

        // 12.10 sorts above 12.9 numerically, 12.3 is too old, rc is skipped.
        let windows = choose_wheel(&fixture(), Os::Windows).unwrap();
        assert_eq!(windows.version, "12.10.0.1");
        assert!(windows.url.ends_with("win_amd64.whl"));
    }

    #[test]
    fn no_matching_wheel_is_an_error() {
        let only_13 = serde_json::json!({"releases": {"13.0.0.1": [
            file("n-13.0.0.1-py3-none-win_amd64.whl", serde_json::json!({}))
        ]}});
        assert!(matches!(
            choose_wheel(&only_13, Os::Windows),
            Err(CudaLibsError::Unsupported(_))
        ));
        assert!(matches!(
            choose_wheel(&fixture(), Os::MacOs),
            Err(CudaLibsError::Unsupported(_))
        ));
        assert!(matches!(
            choose_wheel(&serde_json::json!({}), Os::Linux),
            Err(CudaLibsError::Network(_))
        ));
    }

    #[test]
    fn skips_files_without_a_checksum() {
        let json = serde_json::json!({"releases": {"12.5.0.0": [
            file("n-12.5.0.0-py3-none-win_amd64.whl", serde_json::json!({"digests": {}}))
        ]}});
        assert!(choose_wheel(&json, Os::Windows).is_err());
    }

    #[test]
    fn formats_sizes() {
        assert_eq!(size_label(553_153_899), "553 MB");
        assert_eq!(size_label(581_242_350), "581 MB");
        assert_eq!(size_label(1_500_000_000), "1.5 GB");
        assert_eq!(size_label(10), "1 MB");
    }

    #[test]
    fn required_files_per_os() {
        assert_eq!(
            required_files(Os::Linux),
            ["libcublas.so.12", "libcublasLt.so.12"]
        );
        assert_eq!(
            required_files(Os::Windows),
            ["cublas64_12.dll", "cublasLt64_12.dll"]
        );
        assert!(required_files(Os::MacOs).is_empty());
        assert!(required_files(Os::Other).is_empty());
    }

    #[test]
    fn cache_dir_follows_conventions() {
        let none = |_: &str| None;
        let home = Some(PathBuf::from("/h"));
        assert_eq!(
            cache_dir_for(Os::Linux, none, home.clone()),
            PathBuf::from("/h/.cache/whispry/cuda")
        );
        assert_eq!(
            cache_dir_for(
                Os::Linux,
                |n| (n == "XDG_CACHE_HOME").then(|| "/x".into()),
                home.clone()
            ),
            PathBuf::from("/x/whispry/cuda")
        );
        assert_eq!(
            cache_dir_for(
                Os::Windows,
                |n| (n == "LOCALAPPDATA").then(|| "/l".into()),
                home
            ),
            PathBuf::from("/l/whispry/cuda")
        );
    }

    fn touch_all(os: Os, dir: &Path) {
        std::fs::create_dir_all(dir).unwrap();
        for name in required_files(os) {
            std::fs::write(dir.join(name), b"x").unwrap();
        }
    }

    #[test]
    fn installed_dirs_needs_every_file() {
        let temp = tempfile::tempdir().unwrap();
        assert!(installed_dirs_in(Os::Linux, temp.path()).is_empty());
        std::fs::write(temp.path().join("libcublas.so.12"), b"x").unwrap();
        assert!(installed_dirs_in(Os::Linux, temp.path()).is_empty());
        touch_all(Os::Linux, temp.path());
        assert_eq!(
            installed_dirs_in(Os::Linux, temp.path()),
            vec![temp.path().to_path_buf()]
        );
        assert!(installed_dirs_in(Os::MacOs, temp.path()).is_empty());
    }

    #[test]
    fn finds_existing_in_order() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let cache = root.join("cache");
        let system = root.join("system");
        let extra = root.join("extra");
        let none: &[PathBuf] = &[];

        assert_eq!(find_existing_in(Os::Linux, &cache, none, none, none), None);

        touch_all(Os::Linux, &extra);
        let extras = [extra.clone()];
        assert_eq!(
            find_existing_in(Os::Linux, &cache, none, none, &extras),
            Some(extra.clone())
        );

        touch_all(Os::Linux, &system);
        assert_eq!(
            find_existing_in(
                Os::Linux,
                &cache,
                std::slice::from_ref(&system),
                none,
                &extras
            ),
            Some(system.clone())
        );

        touch_all(Os::Linux, &cache);
        assert_eq!(
            find_existing_in(Os::Linux, &cache, &[system], none, &extras),
            Some(cache)
        );
    }

    #[test]
    fn finds_libraries_in_a_virtualenv_of_an_ancestor() {
        let temp = tempfile::tempdir().unwrap();
        let project = temp.path().join("project");
        let deep = project.join("a").join("b");
        std::fs::create_dir_all(&deep).unwrap();
        let lib = project.join(".venv/lib/python3.12/site-packages/nvidia/cublas/lib");
        touch_all(Os::Linux, &lib);
        // Another NVIDIA package folder must not be mistaken for cuBLAS.
        touch_all(
            Os::Linux,
            &project.join(".venv/lib/python3.12/site-packages/nvidia/other/lib"),
        );

        let cache = temp.path().join("cache");
        let found = find_existing_in(Os::Linux, &cache, &[], &[deep], &[]);
        assert_eq!(found, Some(lib));
    }

    #[test]
    fn finds_windows_libraries_in_a_virtualenv() {
        let temp = tempfile::tempdir().unwrap();
        let bin = temp.path().join("venv/Lib/site-packages/nvidia/cublas/bin");
        touch_all(Os::Windows, &bin);
        let found = find_existing_in(
            Os::Windows,
            &temp.path().join("cache"),
            &[],
            &[temp.path().to_path_buf()],
            &[],
        );
        assert_eq!(found, Some(bin));
    }

    #[test]
    fn partial_virtualenv_libraries_are_not_enough() {
        let temp = tempfile::tempdir().unwrap();
        let lib = temp
            .path()
            .join(".venv/lib/python3.12/site-packages/nvidia/cublas/lib");
        std::fs::create_dir_all(&lib).unwrap();
        std::fs::write(lib.join("libcublas.so.12"), b"x").unwrap();
        let found = find_existing_in(
            Os::Linux,
            &temp.path().join("cache"),
            &[],
            &[temp.path().to_path_buf()],
            &[],
        );
        assert_eq!(found, None);
    }

    // ---- relaunch ----

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(*value))
        }
    }

    #[test]
    fn relaunch_prepends_to_the_loader_path() {
        let dirs = [PathBuf::from("/c/cuda")];
        let (var, value) =
            relaunch_environment(Os::Linux, &dirs, &env_of(&[("LD_LIBRARY_PATH", "/a:/b")]))
                .unwrap();
        assert_eq!(var, "LD_LIBRARY_PATH");
        assert_eq!(value, OsString::from("/c/cuda:/a:/b"));

        let (var, value) = relaunch_environment(Os::Linux, &dirs, &env_of(&[])).unwrap();
        assert_eq!(var, "LD_LIBRARY_PATH");
        assert_eq!(value, OsString::from("/c/cuda"));
    }

    #[test]
    fn relaunch_uses_path_on_windows() {
        // Path syntax differs per host, so build the expectation the same way.
        let dirs = [PathBuf::from("cuda")];
        let existing = std::env::join_paths(["a", "b"]).unwrap();
        let existing = existing.to_str().unwrap();
        let (var, value) =
            relaunch_environment(Os::Windows, &dirs, &env_of(&[("PATH", existing)])).unwrap();
        assert_eq!(var, "PATH");
        assert_eq!(value, std::env::join_paths(["cuda", "a", "b"]).unwrap());
    }

    #[test]
    fn relaunch_is_skipped_when_not_needed() {
        let dirs = [PathBuf::from("/c/cuda")];
        // Already the re-run.
        assert_eq!(
            relaunch_environment(Os::Linux, &dirs, &env_of(&[(RELAUNCH_MARKER, "1")])),
            None
        );
        // Already on the path.
        assert_eq!(
            relaunch_environment(
                Os::Linux,
                &dirs,
                &env_of(&[("LD_LIBRARY_PATH", "/x:/c/cuda")])
            ),
            None
        );
        // Nothing installed.
        assert_eq!(relaunch_environment(Os::Linux, &[], &env_of(&[])), None);
        // Other systems.
        assert_eq!(relaunch_environment(Os::MacOs, &dirs, &env_of(&[])), None);
        assert_eq!(relaunch_environment(Os::Other, &dirs, &env_of(&[])), None);
    }

    // ---- install ----

    fn wheel(os: Os, members: &[(&str, &[u8])], skip: Option<&str>) -> Vec<u8> {
        let folder = if os == Os::Windows { "bin" } else { "lib" };
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated);
        for name in required_files(os) {
            if skip == Some(*name) {
                continue;
            }
            writer
                .start_file(format!("nvidia/cublas/{folder}/{name}"), options)
                .unwrap();
            writer
                .write_all(format!("contents of {name}").as_bytes())
                .unwrap();
        }
        writer
            .start_file(format!("nvidia/cublas/{folder}/libnvblas.so.12"), options)
            .unwrap();
        writer.write_all(b"not needed").unwrap();
        for (name, data) in members {
            writer.start_file(*name, options).unwrap();
            writer.write_all(data).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }

    fn serve(body: Vec<u8>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/wheel.whl", listener.local_addr().unwrap());
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let body = body.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(&stream);
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line.trim().is_empty() {
                            break;
                        }
                    }
                    let mut stream = &stream;
                    let head = format!(
                        "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&body);
                });
            }
        });
        url
    }

    fn plain_agent() -> ureq::Agent {
        // Tests talk to localhost, so ignore any proxy from the environment.
        ureq::Agent::config_builder().proxy(None).build().into()
    }

    fn plan_for(os: Os, url: String, bytes: &[u8]) -> DownloadPlan {
        DownloadPlan {
            os,
            version: "12.9.1.4".into(),
            url,
            sha256: sha256_hex(bytes),
            size: bytes.len() as u64,
        }
    }

    fn leftovers(dir: &Path) -> Vec<String> {
        std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect()
    }

    #[test]
    fn installs_only_the_required_libraries() {
        for os in [Os::Linux, Os::Windows] {
            let bytes = wheel(os, &[], None);
            let plan = plan_for(os, serve(bytes.clone()), &bytes);
            let temp = tempfile::tempdir().unwrap();
            let dest = temp.path().join("cuda");

            let mut last = (0, 0);
            let mut calls = 0;
            let installed = install_with(
                &plain_agent(),
                &plan,
                &dest,
                &mut |done, total| {
                    last = (done, total);
                    calls += 1;
                },
                &CancelToken::new(),
            )
            .unwrap();

            assert_eq!(installed, dest);
            let mut names = leftovers(&dest);
            names.sort();
            let mut expected: Vec<String> =
                required_files(os).iter().map(|s| s.to_string()).collect();
            expected.sort();
            assert_eq!(names, expected);
            let first = required_files(os)[0];
            assert_eq!(
                std::fs::read_to_string(dest.join(first)).unwrap(),
                format!("contents of {first}")
            );
            assert_eq!(last, (plan.size, plan.size));
            assert!(calls >= 2);
            // Only the finished folder remains beside it.
            assert_eq!(leftovers(temp.path()), vec!["cuda".to_string()]);
            assert!(holds_all(os, &dest));
        }
    }

    #[test]
    fn replaces_an_existing_install() {
        let bytes = wheel(Os::Linux, &[], None);
        let plan = plan_for(Os::Linux, serve(bytes.clone()), &bytes);
        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("cuda");
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("stale"), b"x").unwrap();
        install_with(
            &plain_agent(),
            &plan,
            &dest,
            &mut |_, _| {},
            &CancelToken::new(),
        )
        .unwrap();
        assert!(!dest.join("stale").exists());
        assert!(holds_all(Os::Linux, &dest));
    }

    #[test]
    fn wrong_hash_is_rejected_and_cleaned_up() {
        let bytes = wheel(Os::Linux, &[], None);
        let mut plan = plan_for(Os::Linux, serve(bytes.clone()), &bytes);
        plan.sha256 = sha256_hex(b"something else");
        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("cuda");
        let error = install_with(
            &plain_agent(),
            &plan,
            &dest,
            &mut |_, _| {},
            &CancelToken::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CudaLibsError::HashMismatch));
        assert!(leftovers(temp.path()).is_empty());
    }

    #[test]
    fn missing_member_is_reported() {
        let bytes = wheel(Os::Linux, &[], Some("libcublasLt.so.12"));
        let plan = plan_for(Os::Linux, serve(bytes.clone()), &bytes);
        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("cuda");
        let error = install_with(
            &plain_agent(),
            &plan,
            &dest,
            &mut |_, _| {},
            &CancelToken::new(),
        )
        .unwrap_err();
        match error {
            CudaLibsError::MissingInWheel(name) => assert_eq!(name, "libcublasLt.so.12"),
            other => panic!("unexpected {other:?}"),
        }
        assert!(leftovers(temp.path()).is_empty());
    }

    #[test]
    fn not_a_zip_is_reported() {
        let bytes = b"definitely not a zip".to_vec();
        let plan = plan_for(Os::Linux, serve(bytes.clone()), &bytes);
        let temp = tempfile::tempdir().unwrap();
        let error = install_with(
            &plain_agent(),
            &plan,
            &temp.path().join("cuda"),
            &mut |_, _| {},
            &CancelToken::new(),
        )
        .unwrap_err();
        assert!(matches!(error, CudaLibsError::BadArchive(_)));
        assert!(leftovers(temp.path()).is_empty());
    }

    #[test]
    fn zip_slip_names_never_escape() {
        let evil: &[(&str, &[u8])] = &[
            ("../evil.txt", b"x"),
            ("nvidia/cublas/lib/../../../evil2.txt", b"x"),
            ("/abs-evil.txt", b"x"),
        ];
        let bytes = wheel(Os::Linux, evil, None);
        let plan = plan_for(Os::Linux, serve(bytes.clone()), &bytes);
        let temp = tempfile::tempdir().unwrap();
        let outer = temp.path().join("outer");
        let dest = outer.join("cuda");
        install_with(
            &plain_agent(),
            &plan,
            &dest,
            &mut |_, _| {},
            &CancelToken::new(),
        )
        .unwrap();

        assert_eq!(leftovers(&outer), vec!["cuda".to_string()]);
        assert_eq!(leftovers(temp.path()), vec!["outer".to_string()]);
        assert_eq!(leftovers(&dest).len(), 2);
    }

    #[test]
    fn cancelling_leaves_nothing_behind() {
        // Large enough to need several reads.
        // Incompressible, so the download takes several reads.
        let mut seed = 0x2545_f491_u32;
        let big: Vec<u8> = (0..3 << 20)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 24) as u8
            })
            .collect();
        let bytes = wheel(Os::Linux, &[("padding.bin", &big)], None);
        let plan = plan_for(Os::Linux, serve(bytes.clone()), &bytes);
        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("cuda");
        let cancel = CancelToken::new();
        let trigger = cancel.clone();
        let error = install_with(
            &plain_agent(),
            &plan,
            &dest,
            &mut |done, _| {
                if done > 0 {
                    trigger.cancel();
                }
            },
            &cancel,
        )
        .unwrap_err();
        assert!(matches!(error, CudaLibsError::Cancelled));
        assert!(leftovers(temp.path()).is_empty());
    }

    #[test]
    fn a_previous_install_survives_a_failed_one() {
        let bytes = wheel(Os::Linux, &[], None);
        let mut plan = plan_for(Os::Linux, serve(bytes.clone()), &bytes);
        plan.sha256 = "0".repeat(64);
        let temp = tempfile::tempdir().unwrap();
        let dest = temp.path().join("cuda");
        touch_all(Os::Linux, &dest);
        assert!(
            install_with(
                &plain_agent(),
                &plan,
                &dest,
                &mut |_, _| {},
                &CancelToken::new()
            )
            .is_err()
        );
        assert!(holds_all(Os::Linux, &dest));
    }

    #[test]
    fn plan_from_reads_a_package_index() {
        let json = fixture().to_string().into_bytes();
        let url = serve(json);
        let os = Os::current();
        let result = plan_from(&plain_agent(), &url, os);
        if matches!(os, Os::Linux | Os::Windows) && std::env::consts::ARCH == "x86_64" {
            assert_eq!(result.unwrap().os, os);
        } else {
            assert!(matches!(result, Err(CudaLibsError::Unsupported(_))));
        }
    }

    #[test]
    fn errors_read_well() {
        assert_eq!(CudaLibsError::Cancelled.to_string(), "Download cancelled");
        assert!(
            CudaLibsError::HashMismatch
                .to_string()
                .contains("try again")
        );
        assert!(
            CudaLibsError::MissingInWheel("x".into())
                .to_string()
                .contains('x')
        );
    }
}
