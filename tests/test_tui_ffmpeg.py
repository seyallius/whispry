import asyncio
import subprocess
from pathlib import Path

from textual.widgets import Input, Static

from test_tui_setup import services
from whispry.core.events import Finished
from whispry.core.ffmpeg import FFmpeg
from whispry.tui.app import WhispryApp
from whispry.tui.screens.ffmpeg import FFmpegScreen


def run_setup(tmp_path, interact, **overrides):
    audio = tmp_path / "talk.mp3"
    audio.write_bytes(b"")
    jobs = []

    def run_job(config, on_event, system, cancel):
        jobs.append(config)
        return Finished(config.output_path, 0, 0)

    app = WhispryApp(services(run_job=run_job, **overrides))

    async def main():
        async with app.run_test(size=(110, 45)) as pilot:
            await pilot.pause()
            app.screen.query_one("#file", Input).value = str(audio)
            await pilot.press("ctrl+s")
            await pilot.pause()
            await interact(app, pilot)

    asyncio.run(main())
    return jobs


def test_missing_ffmpeg_asks_and_shows_install_steps(tmp_path):
    async def interact(app, pilot):
        assert isinstance(app.screen, FFmpegScreen)
        await pilot.click("#steps")
        await pilot.pause()
        body = str(app.screen.query_one("#body", Static).render())
        assert "1. step one" in body
        await pilot.click("#cancel")
        await pilot.pause()
        assert not isinstance(app.screen, FFmpegScreen)

    jobs = run_setup(
        tmp_path,
        interact,
        find_ffmpeg=lambda: None,
        ffmpeg_install_instructions=lambda: ["step one", "step two"],
    )
    assert jobs == []


def test_bundled_ffmpeg_is_installed_only_when_chosen_then_job_starts(tmp_path):
    installed = []

    def install():
        installed.append(True)
        return subprocess.CompletedProcess([], 0, "", "")

    def find():
        return FFmpeg("/bundled/ffmpeg", None, "bundled") if installed else None

    async def interact(app, pilot):
        assert installed == []
        await pilot.click("#bundled")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()

    jobs = run_setup(tmp_path, interact, find_ffmpeg=find, install_bundled_ffmpeg=install)
    assert installed == [True]
    assert len(jobs) == 1
