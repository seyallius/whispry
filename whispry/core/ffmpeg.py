"""Locate FFmpeg on any OS, or explain how to get it.

Lookup order: FFmpeg on PATH, then the bundled build from the optional
`imageio-ffmpeg` package. The bundled build is never installed without
the user choosing it; front ends call install_bundled_ffmpeg() only after
asking.
"""

from __future__ import annotations

import importlib
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BUNDLED_PACKAGE = "imageio-ffmpeg"
BUNDLED_DOWNLOAD_SIZE = "about 30-40 MB download, 70-170 MB on disk"

SYSTEM = "system"
BUNDLED = "bundled"


class FFmpegNotFound(RuntimeError):
    def __init__(self):
        super().__init__(
            "FFmpeg was not found. Install it and add it to PATH, "
            f"or install the bundled build ({BUNDLED_PACKAGE})."
        )


@dataclass(frozen=True)
class FFmpeg:
    ffmpeg: str
    ffprobe: str | None
    source: str


def subprocess_flags() -> int:
    """Keep FFmpeg from opening a console window on Windows."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def _bundled_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError:
        return None


def find_ffmpeg() -> FFmpeg | None:
    system = shutil.which("ffmpeg")
    if system:
        return FFmpeg(system, shutil.which("ffprobe"), SYSTEM)

    bundled = _bundled_ffmpeg()
    if bundled:
        # The bundled package ships ffmpeg only, no ffprobe.
        return FFmpeg(bundled, None, BUNDLED)

    return None


def require_ffmpeg() -> FFmpeg:
    tool = find_ffmpeg()
    if tool is None:
        raise FFmpegNotFound()
    return tool


def install_bundled_ffmpeg() -> subprocess.CompletedProcess:
    """pip-install the bundled FFmpeg into the running Python environment."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", BUNDLED_PACKAGE],
        capture_output=True,
        text=True,
        creationflags=subprocess_flags(),
    )
    # Let this already-running process import the newly installed package.
    importlib.invalidate_caches()
    return result


def _linux_distribution_ids() -> set[str]:
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return set()
    ids = set()
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key in ("ID", "ID_LIKE"):
            ids.update(value.strip().strip('"').split())
    return ids


def install_instructions(os_name: str | None = None) -> list[str]:
    """Steps to install FFmpeg system-wide and put it on PATH."""
    os_name = os_name or platform.system()

    if os_name == "Windows":
        return [
            "Open PowerShell and run:  winget install --id Gyan.FFmpeg -e   "
            "(or, if you use them: choco install ffmpeg / scoop install ffmpeg)",
            "Manual alternative: download a release build from "
            "https://www.gyan.dev/ffmpeg/builds/, extract it (e.g. to C:\\ffmpeg), "
            "then add C:\\ffmpeg\\bin to PATH via Settings > System > About > "
            "Advanced system settings > Environment Variables > Path > Edit > New.",
            "Close and reopen your terminal, then check with:  ffmpeg -version",
        ]

    if os_name == "Darwin":
        return [
            "Install Homebrew from https://brew.sh if you don't have it.",
            "Run:  brew install ffmpeg",
            "Check with:  ffmpeg -version",
        ]

    ids = _linux_distribution_ids()
    if ids & {"debian", "ubuntu"}:
        command = "sudo apt install ffmpeg"
    elif ids & {"fedora", "rhel", "centos"}:
        command = "sudo dnf install ffmpeg-free   (or ffmpeg from RPM Fusion)"
    elif ids & {"arch"}:
        command = "sudo pacman -S ffmpeg"
    elif ids & {"opensuse", "suse"}:
        command = "sudo zypper install ffmpeg"
    else:
        command = "install the 'ffmpeg' package with your distribution's package manager"

    return [
        f"Run:  {command}",
        "Check with:  ffmpeg -version",
    ]


_DURATION = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def probe_duration(tool: FFmpeg, audio: Path) -> float | None:
    """Return the media duration in seconds, or None if it can't be read."""
    if tool.ffprobe:
        result = subprocess.run(
            [
                tool.ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio),
            ],
            capture_output=True,
            text=True,
            creationflags=subprocess_flags(),
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass

    # No ffprobe (bundled build): FFmpeg prints the duration when probing input.
    result = subprocess.run(
        [tool.ffmpeg, "-hide_banner", "-nostdin", "-i", str(audio)],
        capture_output=True,
        text=True,
        errors="replace",
        creationflags=subprocess_flags(),
    )
    match = _DURATION.search(result.stderr)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
