import pytest

from conftest import make_audio
from fakes import CPU_SYSTEM, FakeModel, install_fake_model
from whispry.core.config import SAMPLE_RATE, JobConfig
from whispry.core.events import Finished, LanguageDetected, Progress, SegmentReady
from whispry.core.transcriber import run_job


@pytest.fixture
def three_minutes(tmp_path):
    return make_audio(tmp_path / "talk.wav", "sine=duration=180")


def test_transcribes_window_by_window_with_bounded_memory(monkeypatch, three_minutes):
    model = install_fake_model(monkeypatch)
    events = []

    result = run_job(JobConfig(three_minutes, chunk_minutes=1), events.append, CPU_SYSTEM)

    # Three one-minute windows, none larger than the configured length.
    assert len(model.calls) == 3
    assert all(call["samples"] <= 60 * SAMPLE_RATE for call in model.calls)
    assert result.audio_duration == pytest.approx(180, abs=0.1)

    # Timestamps are shifted by each window's start.
    starts = [e.start for e in events if isinstance(e, SegmentReady)]
    assert starts == sorted(starts)
    assert starts[-1] >= 120

    lines = three_minutes.with_suffix(".txt").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "[00:00] window 1 at 0"
    assert lines[-1].startswith("[02:")


def test_language_is_detected_once_then_locked(monkeypatch, three_minutes):
    model = install_fake_model(monkeypatch, FakeModel(language="fa"))
    events = []

    run_job(JobConfig(three_minutes, chunk_minutes=1), events.append, CPU_SYSTEM)

    assert [call["language"] for call in model.calls] == [None, "fa", "fa"]
    assert [e.language for e in events if isinstance(e, LanguageDetected)] == ["fa"]


def test_previous_text_is_passed_as_context(monkeypatch, three_minutes):
    model = install_fake_model(monkeypatch)
    run_job(JobConfig(three_minutes, chunk_minutes=1), lambda e: None, CPU_SYSTEM)

    assert model.calls[0]["initial_prompt"] is None
    assert "window 1" in model.calls[1]["initial_prompt"]


def test_progress_reaches_total(monkeypatch, three_minutes):
    install_fake_model(monkeypatch)
    events = []
    run_job(JobConfig(three_minutes, language="en", chunk_minutes=1), events.append, CPU_SYSTEM)

    progress = [e for e in events if isinstance(e, Progress)]
    assert progress[-1].completed == pytest.approx(progress[-1].total, abs=0.1)
    assert isinstance(events[-1], Finished)


def test_cancel_stops_job_and_keeps_partial_output(monkeypatch, three_minutes):
    from whispry.core.cancel import CancelToken, JobCancelled

    model = install_fake_model(monkeypatch)
    token = CancelToken()

    def on_event(event):
        if isinstance(event, SegmentReady) and event.start >= 60:
            token.cancel()

    with pytest.raises(JobCancelled) as raised:
        run_job(JobConfig(three_minutes, chunk_minutes=1), on_event, CPU_SYSTEM, token)

    assert len(model.calls) == 2
    assert raised.value.output == three_minutes.with_suffix(".txt")
    assert raised.value.position > 60
    assert "window 2" in raised.value.output.read_text(encoding="utf-8")


def test_cancel_before_start_skips_model_loading(monkeypatch, three_minutes):
    from whispry.core.cancel import CancelToken, JobCancelled

    model = install_fake_model(monkeypatch)
    token = CancelToken()
    token.cancel()

    with pytest.raises(JobCancelled):
        run_job(JobConfig(three_minutes), lambda e: None, CPU_SYSTEM, token)
    assert model.calls == []
