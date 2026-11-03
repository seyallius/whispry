import pytest

from conftest import make_audio
from fakes import CPU_SYSTEM, install_fake_model
from whispry import cli


def test_no_arguments_opens_tui(monkeypatch):
    opened = []
    monkeypatch.setattr("whispry.tui.main", lambda audio=None: opened.append(audio))
    cli.main([])
    assert opened == [None]


def test_transcribe_end_to_end_with_fake_model(monkeypatch, tmp_path, capsys):
    from whispry.core import transcriber

    install_fake_model(monkeypatch)
    monkeypatch.setattr(transcriber, "detect_system", lambda: CPU_SYSTEM)
    audio = make_audio(tmp_path / "talk.wav", "sine=duration=30")
    output = tmp_path / "out" / "result.txt"
    output.parent.mkdir()

    cli.main(["transcribe", str(audio), "--language", "en", "--output", str(output)])

    assert output.read_text(encoding="utf-8").startswith("[00:00] window 1")
    assert f"Saved transcription to: {output}" in capsys.readouterr().out


def test_transcribe_rejects_missing_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as raised:
        cli.main(["transcribe", str(tmp_path / "missing.wav")])
    assert raised.value.code == 2
    assert "File not found" in capsys.readouterr().err


def test_unavailable_gpu_is_a_clean_error(monkeypatch, tmp_path, capsys):
    from whispry.core import transcriber

    monkeypatch.setattr(transcriber, "detect_system", lambda: CPU_SYSTEM)
    audio = make_audio(tmp_path / "talk.wav", "sine=duration=1")
    with pytest.raises(SystemExit) as raised:
        cli.main(["transcribe", str(audio), "--device", "cuda"])
    assert raised.value.code == 1
    assert "No NVIDIA GPU" in capsys.readouterr().err


def test_devices_lists_each_device(monkeypatch, capsys):
    monkeypatch.setattr(cli, "detect_system", lambda: CPU_SYSTEM)
    cli.main(["devices"])
    out = capsys.readouterr().out
    assert "[unavailable] NVIDIA GPU (CUDA): No NVIDIA GPU" in out
    assert "[available]   CPU: int8, float32" in out
