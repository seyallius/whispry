"""Split a stream of audio blocks into windows that end in silence.

Each window is transcribed independently, so memory use is bounded by the
window length instead of the file length. Cutting inside a pause keeps
words from being split across two windows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np

from whispry.core.config import SAMPLE_RATE

# Returns (start, end) sample ranges that contain speech.
SpeechDetector = Callable[[np.ndarray], list[tuple[int, int]]]

FRAME = SAMPLE_RATE // 50  # 20 ms


@dataclass(frozen=True)
class AudioWindow:
    offset: int
    samples: np.ndarray

    @property
    def start(self) -> float:
        return self.offset / SAMPLE_RATE

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


def vad_speech_regions(samples: np.ndarray) -> list[tuple[int, int]]:
    """Speech regions found by faster-whisper's Silero VAD."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    options = VadOptions(min_silence_duration_ms=300, speech_pad_ms=0)
    return [(r["start"], r["end"]) for r in get_speech_timestamps(samples, options)]


def quietest_point(samples: np.ndarray) -> int:
    """Middle of the lowest-energy 20 ms frame. Fallback when there is no pause."""
    frames = len(samples) // FRAME
    if frames == 0:
        return len(samples)
    energy = np.square(samples[: frames * FRAME].reshape(frames, FRAME)).mean(axis=1)
    return int(np.argmin(energy)) * FRAME + FRAME // 2


def find_cut(samples: np.ndarray, detect_speech: SpeechDetector) -> int:
    """Pick where to end a window inside `samples` (the search region).

    Cuts in the middle of the longest pause. Without speech, cuts at the end;
    with nonstop speech, cuts at the quietest moment.
    """
    regions = detect_speech(samples)
    if not regions:
        return len(samples)

    gaps = []
    previous_end = 0
    for start, end in regions:
        gaps.append((previous_end, start))
        previous_end = end
    gaps.append((previous_end, len(samples)))

    # Longest gap wins; on ties prefer the later one (longer window).
    start, end = max(gaps, key=lambda gap: (gap[1] - gap[0], gap[0]))
    if end - start < FRAME:
        return quietest_point(samples)
    return (start + end) // 2


def iter_windows(
    blocks: Iterable[np.ndarray],
    window_seconds: float,
    search_seconds: float = 30.0,
    detect_speech: SpeechDetector = vad_speech_regions,
) -> Iterator[AudioWindow]:
    """Group decoded blocks into windows of about `window_seconds`.

    Each window ends somewhere in its last `search_seconds`, at a pause.
    At most one window plus one block is held in memory.
    """
    window = int(window_seconds * SAMPLE_RATE)
    search = min(int(search_seconds * SAMPLE_RATE), window // 2)

    buffer = np.empty(0, dtype=np.float32)
    offset = 0

    for block in blocks:
        buffer = np.concatenate((buffer, block))

        while len(buffer) >= window:
            region_start = window - search
            cut = region_start + find_cut(buffer[region_start:window], detect_speech)
            cut = max(cut, 1)
            yield AudioWindow(offset, buffer[:cut])
            # Copy so the yielded window's memory can be released independently.
            buffer = buffer[cut:].copy()
            offset += cut

    if len(buffer):
        yield AudioWindow(offset, buffer)
