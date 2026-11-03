"""Run a transcription job and report progress through events."""

from __future__ import annotations

import time
from contextlib import closing

from whispry.core.audio import stream_audio
from whispry.core.chunking import iter_windows
from whispry.core.config import BEAM_SIZE, JobConfig
from whispry.core.cuda_libs import prepare_cuda_libraries
from whispry.core.events import (
    EventSink,
    Finished,
    LanguageDetected,
    Log,
    Progress,
    SegmentReady,
    Stage,
    StageFinished,
    StageStarted,
    ignore_event,
)
from whispry.core.ffmpeg import probe_duration, require_ffmpeg
from whispry.core.models import ensure_model
from whispry.core.platform import (
    AUTO,
    CPU,
    SystemInfo,
    detect_system,
    resolve_device,
)
from whispry.core.timestamps import format_timestamp

# Tail of the previous window's text given to the next window as context.
PROMPT_CHARACTERS = 200


class ModelLoadError(RuntimeError):
    """The model could not be loaded on the requested device."""

    def __init__(self, device: str, error: Exception):
        super().__init__(f"Could not load the model on {device}: {error}")
        self.device = device


def _load_model(model_dir, device: str, compute_type: str, system: SystemInfo):
    prepare_cuda_libraries(system.os_name)
    from faster_whisper import WhisperModel

    try:
        return WhisperModel(
            str(model_dir),
            device=device,
            compute_type=compute_type,
            cpu_threads=system.cpu_threads if device == CPU else 0,
            local_files_only=True,
        )
    except Exception as error:
        raise ModelLoadError(device, error) from error


def load_model(
    model_dir,
    config: JobConfig,
    system: SystemInfo,
    on_event: EventSink = ignore_event,
):
    """Load the model, falling back from GPU to CPU when the device is `auto`."""
    option, compute_type = resolve_device(config.device, system, config.compute_type)

    on_event(StageStarted(Stage.MODEL, f"Loading model on {option.label} ({compute_type})..."))
    start = time.perf_counter()

    try:
        model = _load_model(model_dir, option.name, compute_type, system)
    except ModelLoadError as error:
        if config.device != AUTO or option.name == CPU:
            raise
        on_event(Log(f"{error}. Falling back to CPU."))
        option, compute_type = resolve_device(CPU, system)
        model = _load_model(model_dir, option.name, compute_type, system)

    on_event(StageFinished(Stage.MODEL, time.perf_counter() - start))
    return model


def run_job(
    config: JobConfig,
    on_event: EventSink = ignore_event,
    system: SystemInfo | None = None,
) -> Finished:
    """Transcribe `config.audio` and write timestamped text to the output file.

    Audio is decoded and transcribed one window (`config.chunk_minutes`) at a
    time, so memory use does not grow with the length of the file.
    """
    job_start = time.perf_counter()
    system = system or detect_system()

    # Fail fast on an impossible device or missing FFmpeg before downloading.
    resolve_device(config.device, system, config.compute_type)
    ffmpeg = require_ffmpeg()
    on_event(Log(f"Using {ffmpeg.source} FFmpeg: {ffmpeg.ffmpeg}"))

    model_dir = ensure_model(config.model, on_event)
    model = load_model(model_dir, config, system, on_event)

    duration = probe_duration(ffmpeg, config.audio)
    on_event(StageStarted(Stage.TRANSCRIBE, "Transcribing..."))
    on_event(Progress(Stage.TRANSCRIBE, 0.0, duration, "s"))
    start = time.perf_counter()

    language = config.language
    if language:
        on_event(LanguageDetected(language, 1.0))

    prompt = None
    position = 0.0
    output = config.output_path

    with output.open("w", encoding="utf-8") as file, closing(
        stream_audio(config.audio, ffmpeg)
    ) as blocks:
        for window in iter_windows(blocks, config.chunk_minutes * 60):
            texts = []
            segments, info = model.transcribe(
                window.samples,
                language=language,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                initial_prompt=prompt,
            )

            for segment in segments:
                text = segment.text.strip()
                segment_start = window.start + segment.start
                segment_end = window.start + min(segment.end, window.duration)

                file.write(f"[{format_timestamp(segment_start)}] {text}\n")
                file.flush()
                texts.append(text)

                on_event(SegmentReady(segment_start, segment_end, text))
                on_event(Progress(Stage.TRANSCRIBE, segment_end, duration, "s"))

            # Lock the language once speech is found, so windows stay consistent.
            if language is None and texts:
                language = info.language
                on_event(LanguageDetected(language, info.language_probability))

            # Carry context across the cut so sentences continue naturally.
            if texts:
                prompt = " ".join(texts)[-PROMPT_CHARACTERS:]

            position = window.start + window.duration
            on_event(Progress(Stage.TRANSCRIBE, position, duration, "s"))

    on_event(StageFinished(Stage.TRANSCRIBE, time.perf_counter() - start))

    finished = Finished(output, position, time.perf_counter() - job_start)
    on_event(finished)
    return finished
