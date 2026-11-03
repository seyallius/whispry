"""Console front end: parses arguments and renders core events."""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

from whispry.core import JobConfig, run_job
from whispry.core.config import DEFAULT_MODEL, LANGUAGES, MODEL_REPOSITORIES
from whispry.core.config import validate_config
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
from whispry.core import ffmpeg as ffmpeg_tools
from whispry.core.platform import DeviceUnavailable
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


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio locally.")
    parser.add_argument("audio", type=Path, help="Audio or video file")
    parser.add_argument("--language", choices=LANGUAGES, default=None,
                        help="Language code; omit to auto-detect")
    parser.add_argument("--model", choices=MODEL_REPOSITORIES, default=DEFAULT_MODEL,
                        help="Whisper model size")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                        help="Compute device; auto uses the GPU when available")
    parser.add_argument("--compute-type", default=None,
                        help="CTranslate2 compute type; defaults to the device's best")
    args = parser.parse_args()

    config = JobConfig(
        audio=args.audio,
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
    )
    for problem in validate_config(config):
        parser.error(problem)

    if ffmpeg_tools.find_ffmpeg() is None and not ask_for_ffmpeg():
        sys.exit(1)

    try:
        run_job(config, ConsoleRenderer())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except (DeviceUnavailable, ModelLoadError, ffmpeg_tools.FFmpegNotFound) as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
