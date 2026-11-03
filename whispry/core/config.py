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
DEFAULT_CHUNK_MINUTES = 10
MAX_CHUNK_MINUTES = 60
BEAM_SIZE = 5
SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class JobConfig:
    audio: Path
    model: str = DEFAULT_MODEL
    language: str | None = None
    output: Path | None = None
    device: str = "auto"
    compute_type: str | None = None
    # Audio is processed one window of this length at a time (~38 MB per
    # 10 minutes), so RAM use does not depend on the file size.
    chunk_minutes: float = DEFAULT_CHUNK_MINUTES

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

    if not 1 <= config.chunk_minutes <= MAX_CHUNK_MINUTES:
        problems.append(f"Chunk length must be 1-{MAX_CHUNK_MINUTES} minutes")

    if config.device not in ("auto", "cpu", "cuda"):
        problems.append(f"Unknown device: {config.device}")

    return problems
