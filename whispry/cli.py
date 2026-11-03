"""Command-line front end: parses arguments and renders core events as text."""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from whispry import __version__
from whispry.core import JobConfig, run_job
from whispry.core import ffmpeg as ffmpeg_tools
from whispry.core.audio import AudioDecodeError
from whispry.core.config import (
    DEFAULT_CHUNK_MINUTES,
    DEFAULT_MODEL,
    LANGUAGES,
    MODEL_REPOSITORIES,
    validate_config,
)
from whispry.core.events import (
    Finished,
    LanguageDetected,
    Log,
    Progress,
    SegmentReady,
    Stage,
    StageFinished,
    StageStarted,
)
from whispry.core.models import ensure_model
from whispry.core.platform import DeviceUnavailable, detect_system
from whispry.core.timestamps import format_timestamp
from whispry.core.transcriber import ModelLoadError


class ConsoleRenderer:
    """Turns core events into tqdm progress bars and printed lines."""

    def __init__(self):
        self.bars: dict[Stage, tqdm] = {}

    def __call__(self, event):
        if isinstance(event, StageStarted):
            print(f"\n{event.message}", flush=True)
        elif isinstance(event, StageFinished):
            bar = self.bars.pop(event.stage, None)
            if bar:
                bar.close()
            print(f"Done in {event.elapsed:.1f}s", flush=True)
        elif isinstance(event, Progress):
            self._progress(event)
        elif isinstance(event, LanguageDetected):
            print(f"Language: {event.language} ({event.probability:.0%})")
        elif isinstance(event, SegmentReady):
            tqdm.write(f"[{format_timestamp(event.start)}] {event.text}")
        elif isinstance(event, Log):
            tqdm.write(event.message)
        elif isinstance(event, Finished):
            print(f"\nCompleted in {event.elapsed:.1f}s")
            print(f"Saved transcription to: {event.output}")

    def _progress(self, event: Progress):
        bar = self.bars.get(event.stage)
        if bar is None:
            bar = tqdm(
                total=event.total,
                desc=event.stage.value.capitalize(),
                unit=event.unit,
                unit_scale=event.unit == "B",
                dynamic_ncols=True,
            )
            self.bars[event.stage] = bar
        bar.update(event.completed - bar.n)


def ask_for_ffmpeg() -> bool:
    """Ask how to get FFmpeg. Return True once it is available."""
    print("FFmpeg is required to decode audio, but it was not found.\n")
    print("  1) Show me how to install FFmpeg and add it to PATH")
    print(f"  2) Use the bundled FFmpeg ({ffmpeg_tools.BUNDLED_DOWNLOAD_SIZE})")
    print("  3) Cancel\n")

    if not sys.stdin.isatty():
        print("Not running interactively; choose by installing FFmpeg or running:")
        print(f"  {sys.executable} -m pip install {ffmpeg_tools.BUNDLED_PACKAGE}")
        return False

    choice = input("Choose [1/2/3]: ").strip()

    if choice == "1":
        print()
        for step in ffmpeg_tools.install_instructions():
            print(f"  - {step}")
        print("\nRun Whispry again once FFmpeg is installed.")
        return False

    if choice == "2":
        print(f"Installing {ffmpeg_tools.BUNDLED_PACKAGE}...", flush=True)
        result = ffmpeg_tools.install_bundled_ffmpeg()
        if result.returncode != 0:
            print(result.stdout[-2000:], result.stderr[-2000:], file=sys.stderr)
            return False
        return ffmpeg_tools.find_ffmpeg() is not None

    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whispry",
        description="Private, local speech-to-text. Run without arguments to open the TUI.",
    )
    parser.add_argument("--version", action="version", version=f"whispry {__version__}")
    commands = parser.add_subparsers(dest="command")

    tui = commands.add_parser("tui", help="Open the terminal UI (default)")
    tui.add_argument("audio", type=Path, nargs="?", help="Pre-fill this file")

    run = commands.add_parser("transcribe", help="Transcribe a file without the TUI")
    run.add_argument("audio", type=Path, help="Audio or video file")
    run.add_argument("--language", choices=LANGUAGES, default=None,
                     help="Language code; omit to auto-detect")
    run.add_argument("--model", choices=MODEL_REPOSITORIES, default=DEFAULT_MODEL,
                     help="Whisper model size")
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                     help="Compute device; auto uses the GPU when available")
    run.add_argument("--compute-type", default=None,
                     help="CTranslate2 compute type; defaults to the device's best")
    run.add_argument("--chunk-minutes", type=float, default=DEFAULT_CHUNK_MINUTES,
                     help="Process audio this many minutes at a time (bounds RAM use)")
    run.add_argument("--output", "-o", type=Path, default=None,
                     help="Output text file (default: beside the input, .txt)")

    commands.add_parser("devices", help="Show the OS and which devices can run Whisper")

    download = commands.add_parser("download", help="Download a model for offline use")
    download.add_argument("model", choices=MODEL_REPOSITORIES)

    return parser


def show_devices() -> None:
    system = detect_system()
    print(system.os_label)
    print(f"CPU threads: {system.cpu_threads}")
    for option in system.devices:
        if option.available:
            print(f"  [available]   {option.label}: {', '.join(option.compute_types)}")
        else:
            print(f"  [unavailable] {option.label}: {option.reason}")
    tool = ffmpeg_tools.find_ffmpeg()
    print(f"FFmpeg: {tool.ffmpeg} ({tool.source})" if tool else "FFmpeg: not found")


def transcribe(parser: argparse.ArgumentParser, args) -> None:
    config = JobConfig(
        audio=args.audio,
        model=args.model,
        language=args.language,
        output=args.output,
        device=args.device,
        compute_type=args.compute_type,
        chunk_minutes=args.chunk_minutes,
    )
    for problem in validate_config(config):
        parser.error(problem)

    if ffmpeg_tools.find_ffmpeg() is None and not ask_for_ffmpeg():
        sys.exit(1)

    run_job(config, ConsoleRenderer())


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command in (None, "tui"):
            from whispry.tui import main as tui_main

            tui_main(getattr(args, "audio", None))
        elif args.command == "transcribe":
            transcribe(parser, args)
        elif args.command == "devices":
            show_devices()
        elif args.command == "download":
            print(ensure_model(args.model, ConsoleRenderer()))
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except (
        AudioDecodeError,
        DeviceUnavailable,
        ModelLoadError,
        ffmpeg_tools.FFmpegNotFound,
    ) as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
