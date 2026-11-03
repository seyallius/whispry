"""Events emitted by a transcription job.

The core never prints. It reports everything through these events so any
front end can render them however it likes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Union


class Stage(str, Enum):
    MODEL = "model"
    TRANSCRIBE = "transcribe"


@dataclass(frozen=True)
class StageStarted:
    stage: Stage
    message: str


@dataclass(frozen=True)
class StageFinished:
    stage: Stage
    elapsed: float


@dataclass(frozen=True)
class Progress:
    """Progress within a stage. `total` is None when it is unknown."""

    stage: Stage
    completed: float
    total: float | None
    unit: str


@dataclass(frozen=True)
class LanguageDetected:
    language: str
    probability: float


@dataclass(frozen=True)
class SegmentReady:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Log:
    message: str


@dataclass(frozen=True)
class Finished:
    output: Path
    audio_duration: float
    elapsed: float


Event = Union[
    StageStarted,
    StageFinished,
    Progress,
    LanguageDetected,
    SegmentReady,
    Log,
    Finished,
]

EventSink = Callable[[Event], None]


def ignore_event(event: Event) -> None:
    """An event sink that drops everything."""
