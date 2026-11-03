"""Decode audio with FFmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

from whispry.core.config import SAMPLE_RATE
from whispry.core.events import EventSink, Progress, Stage, ignore_event
from whispry.core.ffmpeg import FFmpeg, probe_duration, subprocess_flags

BYTES_PER_SAMPLE = 4
READ_SIZE = 1024 * 1024


def decode_audio(
    audio: Path,
    tool: FFmpeg,
    on_event: EventSink = ignore_event,
) -> np.ndarray:
    """Decode the whole file to 16 kHz mono float32 samples."""
    duration = probe_duration(tool, audio)
    expected_bytes = int(duration * SAMPLE_RATE * BYTES_PER_SAMPLE) if duration else None

    command = [
        tool.ffmpeg,
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
        creationflags=subprocess_flags(),
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


class AudioDecodeError(RuntimeError):
    pass


def _ffmpeg_decode_command(tool: FFmpeg, audio: Path) -> list[str]:
    return [
        tool.ffmpeg,
        "-nostdin",
        "-v", "error",
        "-i", str(audio),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]


def stream_audio(
    audio: Path,
    tool: FFmpeg,
    block_seconds: float = 30.0,
) -> Iterator[np.ndarray]:
    """Yield 16 kHz mono float32 blocks of `block_seconds` (the last may be shorter).

    Only one block is held in memory at a time. Closing the generator early
    stops FFmpeg.
    """
    block_bytes = max(1, int(block_seconds * SAMPLE_RATE)) * BYTES_PER_SAMPLE

    # Stderr goes to a temp file so a chatty FFmpeg can never block on a full pipe.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            _ffmpeg_decode_command(tool, audio),
            stdout=subprocess.PIPE,
            stderr=errors,
            creationflags=subprocess_flags(),
        )
        try:
            leftover = b""
            while True:
                data = process.stdout.read(block_bytes)
                if not data:
                    break
                data = leftover + data
                usable = len(data) - len(data) % BYTES_PER_SAMPLE
                leftover = data[usable:]
                if usable:
                    yield np.frombuffer(data[:usable], dtype=np.float32)

            if process.wait() != 0:
                errors.seek(0)
                detail = errors.read().decode("utf-8", "replace").strip()[-1000:]
                raise AudioDecodeError(
                    f"FFmpeg could not decode {audio.name} "
                    f"(exit code {process.returncode}): {detail}"
                )
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            process.stdout.close()
