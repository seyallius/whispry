import numpy as np

from whispry.core.chunking import AudioWindow, find_cut, iter_windows, quietest_point
from whispry.core.config import SAMPLE_RATE


def blocks_of(samples: np.ndarray, size: int):
    for start in range(0, len(samples), size):
        yield samples[start:start + size]


def no_speech(samples):
    return []


def test_windows_cover_audio_exactly_once():
    audio = np.arange(25 * SAMPLE_RATE, dtype=np.float32)
    windows = list(iter_windows(blocks_of(audio, 3 * SAMPLE_RATE), 10, 2, no_speech))

    assert [w.duration for w in windows] == [10, 10, 5]
    assert [w.start for w in windows] == [0, 10, 20]
    np.testing.assert_array_equal(np.concatenate([w.samples for w in windows]), audio)


def test_window_memory_is_bounded_by_window_size():
    block = SAMPLE_RATE
    audio = np.zeros(120 * SAMPLE_RATE, dtype=np.float32)
    for window in iter_windows(blocks_of(audio, block), 10, 2, no_speech):
        assert len(window.samples) <= 10 * SAMPLE_RATE


def test_cut_lands_in_longest_pause():
    # Search region of 10 s: speech 0-3 s, pause 3-7 s, speech 7-9 s.
    regions = [(0, 3 * SAMPLE_RATE), (7 * SAMPLE_RATE, 9 * SAMPLE_RATE)]
    samples = np.zeros(10 * SAMPLE_RATE, dtype=np.float32)
    assert find_cut(samples, lambda s: regions) == 5 * SAMPLE_RATE


def test_cut_falls_back_to_quietest_point_during_nonstop_speech():
    samples = np.ones(SAMPLE_RATE, dtype=np.float32)
    samples[8000:8320] = 0.0  # one quiet 20 ms frame
    assert find_cut(samples, lambda s: [(0, len(s))]) == quietest_point(samples)
    assert 8000 <= quietest_point(samples) <= 8320


def test_windows_cut_at_detected_pauses():
    # A pause around 9 s in every 10 s of audio.
    def detector(region):
        # Region is the last 4 s of a 10 s window (6-10 s); pause at 2.5-3.5 s of it.
        return [(0, int(2.5 * SAMPLE_RATE)), (int(3.5 * SAMPLE_RATE), len(region))]

    audio = np.zeros(30 * SAMPLE_RATE, dtype=np.float32)
    windows = list(iter_windows(blocks_of(audio, SAMPLE_RATE), 10, 4, detector))
    assert windows[0].duration == 9.0
    assert windows[1].start == 9.0


def test_audio_window_properties():
    window = AudioWindow(SAMPLE_RATE * 2, np.zeros(SAMPLE_RATE // 2, dtype=np.float32))
    assert (window.start, window.duration) == (2.0, 0.5)


def test_real_vad_runs_and_finds_no_speech_in_silence():
    from whispry.core.chunking import vad_speech_regions

    assert vad_speech_regions(np.zeros(5 * SAMPLE_RATE, dtype=np.float32)) == []
