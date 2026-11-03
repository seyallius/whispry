"""Run a transcription job and report progress through events."""

from __future__ import annotations

import time

from whispry.core.audio import decode_audio
from whispry.core.config import BEAM_SIZE, JobConfig
from whispry.core.events import (
    EventSink,
    Finished,
    LanguageDetected,
    Progress,
    SegmentReady,
    Stage,
    StageFinished,
    StageStarted,
    ignore_event,
)
from whispry.core.models import ensure_model
from whispry.core.timestamps import format_timestamp

COMPUTE_TYPE = "int8_float16"


def run_job(config: JobConfig, on_event: EventSink = ignore_event) -> Finished:
    """Transcribe `config.audio` and write timestamped text to the output file."""
    from faster_whisper import WhisperModel

    job_start = time.perf_counter()

    model_dir = ensure_model(config.model, on_event)

    on_event(StageStarted(Stage.MODEL, "Loading model onto NVIDIA GPU..."))
    start = time.perf_counter()
    model = WhisperModel(
        str(model_dir),
        device="cuda",
        compute_type=COMPUTE_TYPE,
        local_files_only=True,
    )
    on_event(StageFinished(Stage.MODEL, time.perf_counter() - start))

    on_event(StageStarted(Stage.DECODE, "Decoding audio with FFmpeg..."))
    start = time.perf_counter()
    samples = decode_audio(config.audio, on_event)
    segments, info = model.transcribe(
        samples,
        language=config.language,
        beam_size=BEAM_SIZE,
        vad_filter=True,
    )
    on_event(StageFinished(Stage.DECODE, time.perf_counter() - start))
    on_event(LanguageDetected(info.language, info.language_probability))

    on_event(StageStarted(Stage.TRANSCRIBE, "Transcribing..."))
    start = time.perf_counter()
    output = config.output_path
    duration = info.duration

    with output.open("w", encoding="utf-8") as file:
        for segment in segments:
            text = segment.text.strip()
            file.write(f"[{format_timestamp(segment.start)}] {text}\n")
            file.flush()

            on_event(SegmentReady(segment.start, segment.end, text))
            on_event(
                Progress(Stage.TRANSCRIBE, min(segment.end, duration), duration, "s")
            )

    on_event(Progress(Stage.TRANSCRIBE, duration, duration, "s"))
    on_event(StageFinished(Stage.TRANSCRIBE, time.perf_counter() - start))

    finished = Finished(output, duration, time.perf_counter() - job_start)
    on_event(finished)
    return finished
