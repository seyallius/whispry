"""Decode audio with FFmpeg."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from whispry.core.config import SAMPLE_RATE
from whispry.core.events import EventSink, Progress, Stage, ignore_event

BYTES_PER_SAMPLE = 4
READ_SIZE = 1024 * 1024


def get_audio_duration(audio: Path) -> float:
    """Return the media duration in seconds using FFprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def decode_audio(audio: Path, on_event: EventSink = ignore_event) -> np.ndarray:
    """Decode the whole file to 16 kHz mono float32 samples."""
    duration = get_audio_duration(audio)
    expected_bytes = int(duration * SAMPLE_RATE * BYTES_PER_SAMPLE)

    command = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", str(audio),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    chunks = []
    decoded_bytes = 0

    try:
        while True:
            chunk = process.stdout.read(READ_SIZE)

            if not chunk:
                break

            chunks.append(chunk)
            decoded_bytes += len(chunk)
            on_event(Progress(Stage.DECODE, decoded_bytes, expected_bytes, "B"))

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(f"FFmpeg decoding failed with exit code {return_code}")

        return np.frombuffer(b"".join(chunks), dtype=np.float32)

    except BaseException:
        process.kill()
        process.wait()
        raise

    finally:
        if process.stdout:
            process.stdout.close()
