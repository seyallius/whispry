"""Job configuration shared by every front end."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_REPOSITORIES = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}

LANGUAGES = {
    "en": "English",
    "fa": "Persian",
}

DEFAULT_MODEL = "small"
BEAM_SIZE = 5
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class JobConfig:
    audio: Path
    model: str = DEFAULT_MODEL
    language: str | None = None
    output: Path | None = None

    @property
    def output_path(self) -> Path:
        return self.output or self.audio.with_suffix(".txt")


def validate_config(config: JobConfig) -> list[str]:
    """Return human-readable problems with the config (empty when valid)."""
    problems = []

    if not config.audio.is_file():
        problems.append(f"File not found: {config.audio}")

    if config.model not in MODEL_REPOSITORIES:
        problems.append(f"Unknown model: {config.model}")

    if config.language is not None and config.language not in LANGUAGES:
        problems.append(f"Unsupported language: {config.language}")

    return problems
