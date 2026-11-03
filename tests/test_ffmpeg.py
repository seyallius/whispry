import shutil
from pathlib import Path

import pytest

from whispry.core import ffmpeg as tools


def test_prefers_system_ffmpeg(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(tools, "_bundled_ffmpeg", lambda: "/bundled/ffmpeg")
    found = tools.find_ffmpeg()
    assert (found.ffmpeg, found.ffprobe, found.source) == ("/usr/bin/ffmpeg", "/usr/bin/ffprobe", "system")


def test_uses_bundled_only_when_installed(monkeypatch):
    monkeypatch.setattr(tools.shutil, "which", lambda name: None)
    monkeypatch.setattr(tools, "_bundled_ffmpeg", lambda: "/bundled/ffmpeg")
    assert tools.find_ffmpeg() == tools.FFmpeg("/bundled/ffmpeg", None, "bundled")

    monkeypatch.setattr(tools, "_bundled_ffmpeg", lambda: None)
    assert tools.find_ffmpeg() is None
    with pytest.raises(tools.FFmpegNotFound):
        tools.require_ffmpeg()


@pytest.mark.parametrize("os_name, expected", [
    ("Windows", "winget install"),
    ("Darwin", "brew install ffmpeg"),
])
def test_install_instructions(os_name, expected):
    assert any(expected in step for step in tools.install_instructions(os_name))


def test_linux_instructions_follow_distribution(monkeypatch):
    monkeypatch.setattr(tools, "_linux_distribution_ids", lambda: {"ubuntu", "debian"})
    assert "apt install" in tools.install_instructions("Linux")[0]
    monkeypatch.setattr(tools, "_linux_distribution_ids", lambda: {"arch"})
    assert "pacman" in tools.install_instructions("Linux")[0]


needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


@needs_ffmpeg
def test_probe_duration_with_and_without_ffprobe(sine_wav: Path):
    system = tools.find_ffmpeg()
    assert system.ffprobe
    assert tools.probe_duration(system, sine_wav) == pytest.approx(3.0, abs=0.05)

    no_probe = tools.FFmpeg(system.ffmpeg, None, "bundled")
    assert tools.probe_duration(no_probe, sine_wav) == pytest.approx(3.0, abs=0.05)


@needs_ffmpeg
def test_stream_audio_yields_bounded_blocks(sine_wav: Path):
    from whispry.core.audio import stream_audio
    from whispry.core.config import SAMPLE_RATE

    blocks = list(stream_audio(sine_wav, tools.find_ffmpeg(), block_seconds=1.0))
    assert all(len(b) <= SAMPLE_RATE for b in blocks)
    assert sum(len(b) for b in blocks) == pytest.approx(3 * SAMPLE_RATE, abs=SAMPLE_RATE // 100)


@needs_ffmpeg
def test_stream_audio_reports_decode_errors(tmp_path: Path):
    from whispry.core.audio import AudioDecodeError, stream_audio

    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio")
    with pytest.raises(AudioDecodeError, match="broken.mp3"):
        list(stream_audio(broken, tools.find_ffmpeg()))


@needs_ffmpeg
def test_closing_stream_early_stops_ffmpeg(tmp_path: Path):
    from conftest import make_audio
    from whispry.core.audio import stream_audio

    long_audio = make_audio(tmp_path / "long.wav", "sine=duration=120")
    stream = stream_audio(long_audio, tools.find_ffmpeg(), block_seconds=1.0)
    next(stream)
    stream.close()  # must not hang or leave FFmpeg running
