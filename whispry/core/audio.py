"""Decode audio with FFmpeg."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

from whispry.core.config import SAMPLE_RATE
from whispry.core.ffmpeg import FFmpeg, subprocess_flags

BYTES_PER_SAMPLE = 4


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
