import asyncio
import threading
from pathlib import Path

from textual.widgets import Button, RichLog, Static

from test_tui_setup import services
from whispry.core.cancel import JobCancelled
from whispry.core.config import JobConfig
from whispry.core.events import Finished, Progress, SegmentReady, Stage, StageStarted
from whispry.core.transcriber import ModelLoadError
from whispry.tui.app import WhispryApp
from whispry.tui.screens.run import RunScreen

CONFIG = JobConfig(Path("talk.mp3"), device="cuda")


def run_screen(run_job, interact, config=CONFIG):
    app = WhispryApp(services(run_job=run_job))

    async def main():
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause()
            app.start_job(config)
            await pilot.pause()
            await interact(app, pilot)

    asyncio.run(main())


def text(app, selector):
    return str(app.screen.query_one(selector, Static).render())


def log_lines(app, selector):
    return [strip.text for strip in app.screen.query_one(selector, RichLog).lines]


def test_shows_progress_transcript_and_result():
    def fake_run_job(config, on_event, system, cancel):
        on_event(StageStarted(Stage.TRANSCRIBE, "Transcribing..."))
        on_event(Progress(Stage.TRANSCRIBE, 30, 60, "s"))
        on_event(SegmentReady(65, 70, "hello world"))
        return Finished(Path("talk.txt"), 60, 5)

    async def check(app, pilot):
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "[01:05] hello world" in log_lines(app, "#transcript")
        assert "Saved to: talk.txt" in text(app, "#result")
        assert app.screen.query_one("#cancel", Button).disabled
        assert not app.screen.query_one("#new", Button).disabled

    run_screen(fake_run_job, check)


def test_cancel_button_stops_job():
    started = threading.Event()

    def fake_run_job(config, on_event, system, cancel):
        started.set()
        while not cancel.cancelled:
            threading.Event().wait(0.01)
        raise JobCancelled(Path("talk.txt"), 42)

    async def check(app, pilot):
        await asyncio.get_running_loop().run_in_executor(None, started.wait, 5)
        await pilot.click("#cancel")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "Cancelled" in text(app, "#result")
        assert "00:42" in text(app, "#result")

    run_screen(fake_run_job, check)


def test_gpu_load_failure_offers_cpu_retry():
    calls = []

    def fake_run_job(config, on_event, system, cancel):
        calls.append(config.device)
        if config.device == "cuda":
            raise ModelLoadError("cuda", RuntimeError("out of memory"))
        return Finished(Path("talk.txt"), 1, 1)

    async def check(app, pilot):
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "out of memory" in text(app, "#result")
        retry = app.screen.query_one("#retry-cpu", Button)
        assert retry.display
        await pilot.click("#retry-cpu")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert isinstance(app.screen, RunScreen)
        assert "Saved to" in text(app, "#result")

    run_screen(fake_run_job, check)
    assert calls == ["cuda", "cpu"]


def test_unexpected_errors_are_shown_not_swallowed():
    def fake_run_job(config, on_event, system, cancel):
        raise ValueError("weird")

    async def check(app, pilot):
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "ValueError: weird" in text(app, "#result")

    run_screen(fake_run_job, check)
