//! Locate FFmpeg on any OS, or explain how to get it.

use std::ffi::OsStr;
use std::path::{Path, PathBuf};

use crate::platform::Os;
use crate::process::command;

#[derive(Debug, thiserror::Error)]
#[error("FFmpeg was not found. Install it and add it to PATH.")]
pub struct FFmpegNotFound;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FFmpeg {
    pub ffmpeg: PathBuf,
    pub ffprobe: Option<PathBuf>,
}

/// Find `name` in the folders of a PATH-style variable. On Windows the
/// `.exe` suffix is added when missing.
pub fn find_in_path(name: &str, search_path: &OsStr, os: Os) -> Option<PathBuf> {
    let file_name = if os == Os::Windows && !name.to_lowercase().ends_with(".exe") {
        format!("{name}.exe")
    } else {
        name.to_string()
    };
    std::env::split_paths(search_path)
        .map(|dir| dir.join(&file_name))
        .find(|candidate| candidate.is_file())
}

fn find_from_environment(os: Os) -> Option<FFmpeg> {
    let search_path = std::env::var_os("PATH")?;
    let ffmpeg = find_in_path("ffmpeg", &search_path, os)?;
    let ffprobe = find_in_path("ffprobe", &search_path, os);
    Some(FFmpeg { ffmpeg, ffprobe })
}

/// FFmpeg from PATH, if installed.
pub fn find_ffmpeg() -> Option<FFmpeg> {
    find_from_environment(Os::current())
}

pub fn require_ffmpeg() -> Result<FFmpeg, FFmpegNotFound> {
    find_ffmpeg().ok_or(FFmpegNotFound)
}

fn linux_distribution_ids() -> Vec<String> {
    let Ok(text) = std::fs::read_to_string("/etc/os-release") else {
        return Vec::new();
    };
    let mut ids = Vec::new();
    for line in text.lines() {
        if let Some((key, value)) = line.split_once('=')
            && (key == "ID" || key == "ID_LIKE")
        {
            ids.extend(
                value
                    .trim()
                    .trim_matches('"')
                    .split_whitespace()
                    .map(str::to_string),
            );
        }
    }
    ids
}

/// Steps to install FFmpeg system-wide and put it on PATH.
pub fn install_instructions(os: Os) -> Vec<String> {
    instructions_for(os, &linux_distribution_ids())
}

fn instructions_for(os: Os, distribution_ids: &[String]) -> Vec<String> {
    match os {
        Os::Windows => vec![
            "Open PowerShell and run:  winget install --id Gyan.FFmpeg -e   \
             (or, if you use them: choco install ffmpeg / scoop install ffmpeg)"
                .into(),
            "Manual alternative: download a release build from \
             https://www.gyan.dev/ffmpeg/builds/, extract it (e.g. to C:\\ffmpeg), \
             then add C:\\ffmpeg\\bin to PATH via Settings > System > About > \
             Advanced system settings > Environment Variables > Path > Edit > New."
                .into(),
            "Close and reopen your terminal, then check with:  ffmpeg -version".into(),
        ],
        Os::MacOs => vec![
            "Install Homebrew from https://brew.sh if you don't have it.".into(),
            "Run:  brew install ffmpeg".into(),
            "Check with:  ffmpeg -version".into(),
        ],
        Os::Linux | Os::Other => {
            let has = |names: &[&str]| {
                distribution_ids
                    .iter()
                    .any(|id| names.contains(&id.as_str()))
            };
            let command = if has(&["debian", "ubuntu"]) {
                "sudo apt install ffmpeg"
            } else if has(&["fedora", "rhel", "centos"]) {
                "sudo dnf install ffmpeg-free   (or ffmpeg from RPM Fusion)"
            } else if has(&["arch"]) {
                "sudo pacman -S ffmpeg"
            } else if has(&["opensuse", "suse"]) {
                "sudo zypper install ffmpeg"
            } else {
                "install the 'ffmpeg' package with your distribution's package manager"
            };
            vec![
                format!("Run:  {command}"),
                "Check with:  ffmpeg -version".into(),
            ]
        }
    }
}

/// Read the `Duration: HH:MM:SS.ss` line FFmpeg prints when probing a file.
pub fn parse_duration(text: &str) -> Option<f64> {
    let rest = text.split("Duration:").nth(1)?.trim_start();
    let clock = rest.split([',', ' ']).next()?;
    let mut parts = clock.split(':');
    let hours: f64 = parts.next()?.parse().ok()?;
    let minutes: f64 = parts.next()?.parse().ok()?;
    let seconds: f64 = parts.next()?.parse().ok()?;
    Some(hours * 3600.0 + minutes * 60.0 + seconds)
}

/// The media duration in seconds, or `None` if it can't be read.
pub fn probe_duration(tool: &FFmpeg, audio: &Path) -> Option<f64> {
    if let Some(ffprobe) = &tool.ffprobe
        && let Ok(output) = command(ffprobe)
            .args(["-v", "error", "-show_entries", "format=duration"])
            .args(["-of", "default=noprint_wrappers=1:nokey=1"])
            .arg(audio)
            .output()
        && let Ok(seconds) = String::from_utf8_lossy(&output.stdout).trim().parse()
    {
        return Some(seconds);
    }

    // Without ffprobe, FFmpeg prints the duration when it opens the input.
    let output = command(&tool.ffmpeg)
        .args(["-hide_banner", "-nostdin", "-i"])
        .arg(audio)
        .output()
        .ok()?;
    parse_duration(&String::from_utf8_lossy(&output.stderr))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{require, write_sine_wav};

    #[test]
    fn finds_tools_in_path_folders() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ffmpeg"), "").unwrap();
        let path = std::env::join_paths([Path::new("/nonexistent"), dir.path()]).unwrap();
        assert_eq!(
            find_in_path("ffmpeg", &path, Os::Linux),
            Some(dir.path().join("ffmpeg"))
        );
        assert_eq!(find_in_path("ffprobe", &path, Os::Linux), None);
    }

    #[test]
    fn adds_exe_suffix_on_windows() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("ffmpeg.exe"), "").unwrap();
        let path = std::env::join_paths([dir.path()]).unwrap();
        assert_eq!(
            find_in_path("ffmpeg", &path, Os::Windows),
            Some(dir.path().join("ffmpeg.exe"))
        );
    }

    #[test]
    fn windows_and_mac_instructions_name_the_package_manager() {
        assert!(
            install_instructions(Os::Windows)
                .iter()
                .any(|step| step.contains("winget install"))
        );
        assert!(
            install_instructions(Os::MacOs)
                .iter()
                .any(|step| step.contains("brew install ffmpeg"))
        );
    }

    #[test]
    fn linux_instructions_follow_the_distribution() {
        let ids = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert!(
            instructions_for(Os::Linux, &ids(&["ubuntu", "debian"]))[0].contains("apt install")
        );
        assert!(instructions_for(Os::Linux, &ids(&["arch"]))[0].contains("pacman"));
        assert!(instructions_for(Os::Linux, &[])[0].contains("package manager"));
    }

    #[test]
    fn parses_the_duration_line() {
        let text = "  Duration: 01:02:03.50, start: 0.000000, bitrate: 256 kb/s";
        assert_eq!(parse_duration(text), Some(3723.5));
        assert_eq!(parse_duration("no duration here"), None);
        assert_eq!(parse_duration("Duration: N/A, start: 0"), None);
    }

    #[test]
    fn probes_duration_with_and_without_ffprobe() {
        let Some(system) = require("ffmpeg", find_ffmpeg()) else {
            return;
        };
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("sine.wav");
        write_sine_wav(&wav, 3, 440.0);

        let seconds = probe_duration(&system, &wav).unwrap();
        assert!((seconds - 3.0).abs() < 0.05, "{seconds}");

        let without_probe = FFmpeg {
            ffmpeg: system.ffmpeg.clone(),
            ffprobe: None,
        };
        let seconds = probe_duration(&without_probe, &wav).unwrap();
        assert!((seconds - 3.0).abs() < 0.05, "{seconds}");
    }
}
