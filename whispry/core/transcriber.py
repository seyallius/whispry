"""Run a transcription job and report progress through events."""

from __future__ import annotations

import time

from whispry.core.audio import decode_audio
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
from whispry.core.ffmpeg import require_ffmpeg
from whispry.core.models import ensure_model
from whispry.core.platform import (
    AUTO,
    CPU,
    SystemInfo,
    detect_system,
    resolve_device,
)
from whispry.core.timestamps import format_timestamp


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
    """Transcribe `config.audio` and write timestamped text to the output file."""
    job_start = time.perf_counter()
    system = system or detect_system()

    # Fail fast on an impossible device or missing FFmpeg before downloading.
    resolve_device(config.device, system, config.compute_type)
    ffmpeg = require_ffmpeg()
    on_event(Log(f"Using {ffmpeg.source} FFmpeg: {ffmpeg.ffmpeg}"))

    model_dir = ensure_model(config.model, on_event)
    model = load_model(model_dir, config, system, on_event)

    on_event(StageStarted(Stage.DECODE, "Decoding audio with FFmpeg..."))
    start = time.perf_counter()
    samples = decode_audio(config.audio, ffmpeg, on_event)
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
