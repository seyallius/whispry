"""Locate cached Whisper models or download them."""

from __future__ import annotations

import os
import sys
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

# Where models lived before Whispry was a package. Still honored when present.
LEGACY_MODEL_DIRECTORY = Path(__file__).resolve().parents[2] / "models"


def default_model_directory() -> Path:
    """Per-user model cache, following each OS's convention.

    WHISPRY_MODEL_DIR overrides it.
    """
    override = os.environ.get("WHISPRY_MODEL_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Caches"
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")

    return base / "whispry" / "models"


def is_model_cached(model_name: str, directory: Path) -> bool:
    destination = directory / model_name
    return all((destination / name).is_file() for name in REQUIRED_FILES)


def find_cached_model(model_name: str, directory: Path | None = None) -> Path | None:
    """Return the directory of an already-downloaded model, if any."""
    candidates = [directory or default_model_directory()]
    if directory is None:
        candidates.append(LEGACY_MODEL_DIRECTORY)

    for candidate in candidates:
        if is_model_cached(model_name, candidate):
            return candidate / model_name

    return None


def ensure_model(
    model_name: str,
    on_event: EventSink = ignore_event,
    directory: Path | None = None,
) -> Path:
    """Return the local model directory, downloading the model if missing."""
    cached = find_cached_model(model_name, directory)
    if cached:
        on_event(Log(f"Using cached model: {cached}"))
        return cached

    # Imported lazily: it is slow to import and only needed for downloads.
    from huggingface_hub import snapshot_download

    destination = (directory or default_model_directory()) / model_name
    on_event(Log(f"Downloading model {model_name} to {destination}"))

    snapshot_download(
        repo_id=MODEL_REPOSITORIES[model_name],
        local_dir=str(destination),
        allow_patterns=MODEL_FILES,
    )

    return destination
