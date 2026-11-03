import asyncio
from dataclasses import replace

from textual.widgets import Input, RadioButton, Select, Static

from fakes import CPU_SYSTEM
from whispry.core.events import Finished
from whispry.core.ffmpeg import FFmpeg
from whispry.core.platform import CUDA, DeviceOption, SystemInfo
from whispry.tui.app import WhispryApp
from whispry.tui.services import Services

GPU_SYSTEM = SystemInfo(
    "Windows",
    "Windows 11 (AMD64)",
    8,
    (
        DeviceOption(CUDA, "NVIDIA GPU (CUDA)", True, compute_types=("int8_float16", "float16")),
        CPU_SYSTEM.device("cpu"),
    ),
)


def services(system=CPU_SYSTEM, **overrides):
    defaults = {
        "detect_system": lambda: system,
        "is_model_downloaded": lambda name: name == "small",
        "find_ffmpeg": lambda: FFmpeg("ffmpeg", "ffprobe", "system"),
    }
    return replace(Services(), **{**defaults, **overrides})


def run(app, interact):
    async def main():
        async with app.run_test(size=(100, 50)) as pilot:
            await pilot.pause()
            await interact(app, pilot)
    asyncio.run(main())
    return app.return_value


def test_gpu_option_disabled_with_reason_when_missing():
    async def check(app, pilot):
        screen = app.screen
        assert screen.query_one("#device-cuda", RadioButton).disabled
        reason = screen.query_one("#gpu-reason", Static).render()
        assert "No NVIDIA GPU" in str(reason)
        assert "uses CPU" in str(screen.query_one("#device-auto", RadioButton).label)

    run(WhispryApp(services()), check)


def test_gpu_option_enabled_and_compute_types_follow_device():
    async def check(app, pilot):
        screen = app.screen
        assert not screen.query_one("#device-cuda", RadioButton).disabled
        assert not screen.query("#gpu-reason")
        compute = screen.query_one("#compute", Select)
        assert "int8_float16" in str(compute._options)

        await pilot.click("#device-cpu")
        await pilot.pause()
        assert "int8_float16" not in str(compute._options)

    run(WhispryApp(services(GPU_SYSTEM)), check)


def test_marks_downloaded_models():
    async def check(app, pilot):
        assert "small (downloaded)" in str(app.screen.query_one("#model", Select)._options)

    run(WhispryApp(services()), check)


def test_start_validates_then_runs_job(tmp_path):
    audio = tmp_path / "talk.mp3"
    started = []

    def fake_run_job(config, on_event, system, cancel):
        started.append(config)
        return Finished(config.output_path, 0.0, 0.0)

    async def interact(app, pilot):
        app.screen.query_one("#file", Input).value = str(audio)
        await pilot.click("#start")
        await pilot.pause()
        assert "File not found" in str(app.screen.query_one("#errors", Static).render())

        audio.write_bytes(b"")
        app.screen.query_one("#language", Select).value = "fa"
        await pilot.press("ctrl+s")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert type(app.screen).__name__ == "RunScreen"

    run(WhispryApp(services(run_job=fake_run_job)), interact)
    [config] = started
    assert (config.audio, config.language, config.device, config.compute_type) == (audio, "fa", "auto", None)
