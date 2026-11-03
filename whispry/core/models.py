"""Locate cached Whisper models or download them."""

from __future__ import annotations

from pathlib import Path

from whispry.core.config import MODEL_REPOSITORIES
from whispry.core.events import EventSink, Log, ignore_event

MODEL_FILES = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
    "preprocessor_config.json",
]

REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json")

MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models"


def model_path(model_name: str, directory: Path = MODEL_DIRECTORY) -> Path:
    return directory / model_name


def is_model_cached(model_name: str, directory: Path = MODEL_DIRECTORY) -> bool:
    destination = model_path(model_name, directory)
    return all((destination / name).is_file() for name in REQUIRED_FILES)


def ensure_model(
    model_name: str,
    on_event: EventSink = ignore_event,
    directory: Path = MODEL_DIRECTORY,
) -> Path:
    """Return the local model directory, downloading the model if missing."""
    destination = model_path(model_name, directory)

    if is_model_cached(model_name, directory):
        on_event(Log(f"Using cached model: {destination}"))
        return destination

    # Imported lazily: it is slow to import and only needed for downloads.
    from huggingface_hub import snapshot_download

    on_event(Log(f"Downloading model: {model_name}"))

    snapshot_download(
        repo_id=MODEL_REPOSITORIES[model_name],
        local_dir=str(destination),
        allow_patterns=MODEL_FILES,
    )

    return destination
