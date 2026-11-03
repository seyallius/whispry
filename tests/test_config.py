from pathlib import Path

from whispry.core.config import JobConfig, validate_config
from whispry.core.timestamps import format_timestamp


def test_format_timestamp():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(65.9) == "01:05"
    assert format_timestamp(3725) == "01:02:05"


def test_output_defaults_beside_input(tmp_path: Path):
    config = JobConfig(audio=tmp_path / "talk.mp3")
    assert config.output_path == tmp_path / "talk.txt"


def test_validate_config_reports_problems(tmp_path: Path):
    config = JobConfig(audio=tmp_path / "missing.wav", model="huge", language="xx")
    problems = validate_config(config)
    assert len(problems) == 3


def test_validate_config_accepts_valid(tmp_path: Path):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"")
    assert validate_config(JobConfig(audio=audio, language="fa")) == []
