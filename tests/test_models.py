from pathlib import Path

from whispry.core import models


def _fake_model(directory: Path, name: str) -> None:
    target = directory / name
    target.mkdir(parents=True)
    for file in models.REQUIRED_FILES:
        (target / file).write_text("x")


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("WHISPRY_MODEL_DIR", str(tmp_path))
    assert models.default_model_directory() == tmp_path


def test_finds_cached_model(tmp_path):
    _fake_model(tmp_path, "small")
    assert models.find_cached_model("small", tmp_path) == tmp_path / "small"
    assert models.find_cached_model("medium", tmp_path) is None


def test_falls_back_to_legacy_directory(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy"
    _fake_model(legacy, "small")
    monkeypatch.setenv("WHISPRY_MODEL_DIR", str(tmp_path / "new"))
    monkeypatch.setattr(models, "LEGACY_MODEL_DIRECTORY", legacy)
    assert models.find_cached_model("small") == legacy / "small"
