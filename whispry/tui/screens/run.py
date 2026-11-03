"""Second screen: run a job and show its progress live."""

from __future__ import annotations

import time
from dataclasses import replace

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import Button, Footer, Header, Label, ProgressBar, RichLog, Static

from whispry.core.audio import AudioDecodeError
from whispry.core.cancel import CancelToken, JobCancelled
from whispry.core.config import JobConfig
from whispry.core.events import (
    Finished,
    LanguageDetected,
    Log,
    Progress,
    SegmentReady,
    StageFinished,
    StageStarted,
)
from whispry.core.ffmpeg import FFmpegNotFound
from whispry.core.platform import CPU, DeviceUnavailable
from whispry.core.timestamps import format_timestamp
from whispry.core.transcriber import ModelLoadError

# Errors the user can act on; anything else is shown with its type name.
EXPECTED_ERRORS = (AudioDecodeError, DeviceUnavailable, FFmpegNotFound, ModelLoadError)


class RunScreen(Screen):
    DEFAULT_CSS = """
    RunScreen #run { padding: 1 2; }
    RunScreen #summary { color: $text-muted; }
    RunScreen #stage { margin-top: 1; text-style: bold; }
    RunScreen ProgressBar { margin: 1 0 0 0; }
    RunScreen #timing { color: $text-muted; }
    RunScreen #transcript { height: 1fr; border: round $primary; margin-top: 1; }
    RunScreen #activity { height: 5; border: round $panel; color: $text-muted; }
    RunScreen #result { margin-top: 1; height: auto; }
    RunScreen #result.error { color: $error; }
    RunScreen #result.success { color: $success; }
    RunScreen #buttons { height: auto; margin-top: 1; }
    RunScreen #buttons Button { margin-right: 2; }
    """

    BINDINGS = [
        ("escape", "cancel", "Cancel"),
        ("n", "new", "New"),
    ]

    def __init__(self, config: JobConfig):
        super().__init__()
        self.config = config
        self.token = CancelToken()
        self.started = 0.0
        self.position = 0.0
        self.total: float | None = None
        self.running = False

    def compose(self) -> ComposeResult:
        language = self.config.language or "auto-detect"
        device = self.config.device
        yield Header()
        with Vertical(id="run"):
            yield Static(
                f"{self.config.audio.name} · model {self.config.model} · "
                f"device {device} · language {language}",
                id="summary",
            )
            yield Label("Starting...", id="stage")
            yield ProgressBar(total=None, show_eta=True, id="progress")
            yield Static("", id="timing")
            yield RichLog(wrap=True, id="transcript")
            yield RichLog(wrap=True, max_lines=200, id="activity")
            yield Static("", id="result")
            with Horizontal(id="buttons"):
                yield Button("Cancel", variant="error", id="cancel")
                yield Button("Retry on CPU", id="retry-cpu")
                yield Button("New transcription", variant="primary", id="new", disabled=True)
                yield Button("Quit", id="quit")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#retry-cpu").display = False
        self.query_one("#transcript", RichLog).border_title = "Transcript"
        self.query_one("#activity", RichLog).border_title = "Activity"
        self.set_interval(1.0, self._update_timing)
        self._start()

    # ----- job lifecycle -----

    def _start(self) -> None:
        self.running = True
        self.started = time.monotonic()
        self.run_job()

    @work(thread=True, exclusive=True)
    def run_job(self) -> None:
        services = self.app.services
        try:
            result = services.run_job(
                self.config,
                lambda event: self.app.call_from_thread(self.handle_event, event),
                services.detect_system(),
                self.token,
            )
        except JobCancelled as cancelled:
            self.app.call_from_thread(self._cancelled, cancelled)
        except BaseException as error:  # Shown to the user, never swallowed silently.
            self.app.call_from_thread(self._failed, error)
        else:
            self.app.call_from_thread(self._finished, result)

    def handle_event(self, event) -> None:
        """Render one core event. Runs on the UI thread."""
        if isinstance(event, StageStarted):
            self.query_one("#stage", Label).update(event.message)
        elif isinstance(event, StageFinished):
            self._activity(f"{event.stage.value} finished in {event.elapsed:.1f}s")
        elif isinstance(event, Progress):
            self.position, self.total = event.completed, event.total
            self.query_one("#progress", ProgressBar).update(
                total=event.total, progress=event.completed
            )
        elif isinstance(event, LanguageDetected):
            self._activity(f"Language: {event.language} ({event.probability:.0%})")
        elif isinstance(event, SegmentReady):
            self.query_one("#transcript", RichLog).write(
                f"[{format_timestamp(event.start)}] {event.text}"
            )
        elif isinstance(event, Log):
            self._activity(event.message)

    def _activity(self, message: str) -> None:
        self.query_one("#activity", RichLog).write(message)

    def _update_timing(self) -> None:
        if not self.running:
            return
        elapsed = time.monotonic() - self.started
        text = f"Elapsed {format_timestamp(elapsed)} · Audio {format_timestamp(self.position)}"
        if self.total:
            text += f" / {format_timestamp(self.total)}"
            if self.position > 0:
                speed = self.position / max(elapsed, 1e-6)
                text += f" · {speed:.1f}x realtime"
        self.query_one("#timing", Static).update(text)

    def _done(self, message: str, style: str) -> None:
        self._update_timing()
        self.running = False
        result = self.query_one("#result", Static)
        result.set_classes(style)
        result.update(message)
        self.query_one("#cancel", Button).disabled = True
        self.query_one("#new", Button).disabled = False

    def _finished(self, result: Finished) -> None:
        self.query_one("#stage", Label).update("Done")
        progress = self.query_one("#progress", ProgressBar)
        progress.update(total=result.audio_duration or 1, progress=result.audio_duration or 1)
        self._done(
            f"Transcribed {format_timestamp(result.audio_duration)} of audio in "
            f"{format_timestamp(result.elapsed)}.\nSaved to: {result.output}",
            "success",
        )

    def _cancelled(self, cancelled: JobCancelled) -> None:
        self.query_one("#stage", Label).update("Cancelled")
        message = "Cancelled."
        if cancelled.output:
            message += (
                f" Text up to {format_timestamp(cancelled.position)} "
                f"was saved to: {cancelled.output}"
            )
        self._done(message, "error")

    def _failed(self, error: BaseException) -> None:
        self.query_one("#stage", Label).update("Failed")
        detail = str(error) if isinstance(error, EXPECTED_ERRORS) else (
            f"{type(error).__name__}: {error}"
        )
        self._done(f"Error: {detail}", "error")
        if isinstance(error, ModelLoadError) and error.device != CPU:
            self.query_one("#retry-cpu").display = True

    # ----- buttons -----

    @on(Button.Pressed, "#cancel")
    def action_cancel(self) -> None:
        if self.running:
            self.token.cancel()
            self.query_one("#stage", Label).update("Cancelling after the current segment...")
            self.query_one("#cancel", Button).disabled = True

    @on(Button.Pressed, "#retry-cpu")
    def _retry_on_cpu(self) -> None:
        self.app.switch_screen(RunScreen(replace(self.config, device=CPU, compute_type=None)))

    @on(Button.Pressed, "#new")
    def action_new(self) -> None:
        if not self.running:
            self.app.pop_screen()

    @on(Button.Pressed, "#quit")
    def _quit(self) -> None:
        self.token.cancel()
        self.app.exit()

    def on_unmount(self) -> None:
        # Stop the background job if the screen goes away mid-run.
        self.token.cancel()
