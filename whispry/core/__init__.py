"""Transcription logic. Nothing in this package prints or imports a UI.

Front ends (TUI, CLI) build a JobConfig, call run_job(), and render the
events it emits.
"""

from whispry.core.config import JobConfig
from whispry.core.events import Event, EventSink
from whispry.core.transcriber import run_job

__all__ = ["Event", "EventSink", "JobConfig", "run_job"]
