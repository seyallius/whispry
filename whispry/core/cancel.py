"""Cooperative cancellation for jobs running in a background thread."""

from __future__ import annotations

import threading
from pathlib import Path


class JobCancelled(Exception):
    """Raised inside run_job() once cancellation is noticed."""

    def __init__(self, output: Path | None = None, position: float = 0.0):
        super().__init__("Transcription cancelled")
        self.output = output
        self.position = position


class CancelToken:
    """Set from any thread; checked by the job between units of work."""

    def __init__(self):
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()
