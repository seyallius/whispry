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
from whispry.core.timestamps import format_timestamp


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


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio locally.")
    parser.add_argument("audio", type=Path, help="Audio or video file")
    parser.add_argument("--language", choices=LANGUAGES, default=None,
                        help="Language code; omit to auto-detect")
    parser.add_argument("--model", choices=MODEL_REPOSITORIES, default=DEFAULT_MODEL,
                        help="Whisper model size")
    args = parser.parse_args()

    config = JobConfig(audio=args.audio, model=args.model, language=args.language)
    for problem in validate_config(config):
        parser.error(problem)

    try:
        run_job(config, ConsoleRenderer())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
