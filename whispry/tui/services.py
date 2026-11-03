"""The core functions the TUI calls, gathered in one replaceable object.

The UI never imports transcription logic directly; tests swap these out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from whispry.core import ffmpeg, models, platform, transcriber
from whispry.core.config import validate_config


@dataclass(frozen=True)
class Services:
    detect_system: Callable = platform.detect_system
    resolve_device: Callable = platform.resolve_device
    validate_config: Callable = validate_config
    is_model_downloaded: Callable[[str], bool] = (
        lambda name: models.find_cached_model(name) is not None
    )
    run_job: Callable = transcriber.run_job
    find_ffmpeg: Callable = ffmpeg.find_ffmpeg
    install_bundled_ffmpeg: Callable = ffmpeg.install_bundled_ffmpeg
    ffmpeg_install_instructions: Callable = ffmpeg.install_instructions
