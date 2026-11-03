import threading

import numpy as np

from helpers import LOAD, FakeModel, Session, audio, encode, run_serve, transcribe_msg
from whispry_worker import __version__


def types(messages):
    return [m["type"] for m in messages]


def test_ready_is_first():
    out = run_serve(b"", lambda *a: FakeModel())
    assert out[0]["type"] == "ready"
    assert out[0]["version"] == __version__
    assert "faster_whisper" in out[0]
    assert len(out) == 1


def test_load_success_echoes_settings():
    seen = []

    def factory(*args):
        seen.append(args)
        return FakeModel()

    out = run_serve(encode(LOAD), factory)
    assert out[1] == {"type": "loaded", "device": "cpu", "compute_type": "int8"}
    assert seen == [("m", "cpu", "int8", 2)]


def test_load_failure_keeps_worker_alive():
    def factory(*args):
        raise RuntimeError("bad model")

    data = encode(LOAD) + encode(transcribe_msg(7), audio(4))
    out = run_serve(data, factory)
    assert out[1] == {"type": "error", "id": None, "stage": "load", "message": "bad model"}
    assert out[2] == {"type": "error", "id": 7, "stage": "transcribe", "message": "No model is loaded"}
    assert types(out) == ["ready", "error", "error"]


def test_transcribe_before_load():
    out = run_serve(encode(transcribe_msg(3), audio(8)), lambda *a: FakeModel())
    assert types(out) == ["ready", "error"]
    assert out[1]["id"] == 3 and out[1]["stage"] == "transcribe"


def test_transcribe_full_flow():
    model = FakeModel()
    msg = transcribe_msg(5, language="en", beam_size=3, initial_prompt="hi")
    session = Session(lambda *a: model)
    session.send(LOAD)
    session.send(msg, audio(100))
    session.wait_for("done")
    out = session.close()
    assert types(out) == ["ready", "loaded", "language", "segment", "segment", "segment", "done"]
    assert out[2] == {"type": "language", "id": 5, "language": "en", "probability": 0.9}
    segs = out[3:6]
    assert [s["text"] for s in segs] == ["one", "two", "three"]
    assert [s["start"] for s in segs] == [0.0, 1.0, 2.0]
    assert out[6] == {"type": "done", "id": 5, "cancelled": False}
    samples, options = model.calls[0]
    assert len(samples) == 100
    assert np.array_equal(samples, np.arange(100, dtype="<f4"))
    assert options == {"language": "en", "beam_size": 3, "vad_filter": True, "initial_prompt": "hi"}


def test_model_exception_keeps_worker_alive():
    class Broken(FakeModel):
        def transcribe(self, samples, **options):
            raise ValueError("boom")

    data = encode(LOAD) + encode(transcribe_msg(1), audio(4)) + encode(transcribe_msg(2), audio(4))
    out = run_serve(data, lambda *a: Broken())
    assert types(out) == ["ready", "loaded", "error", "error"]
    assert out[2]["message"] == "boom" and out[3]["id"] == 2


def test_cancel_mid_transcription():
    gate, started = threading.Event(), threading.Event()
    model = FakeModel(texts=("a", "b", "c", "d"), gate=gate, started=started)
    session = Session(lambda *a: model)
    session.send(LOAD)
    session.send(transcribe_msg(9), audio(10))
    assert started.wait(5)
    session.send({"type": "cancel", "id": 9})
    threading.Event().wait(0.2)  # let the reader thread process the cancel
    gate.set()
    out = session.close()
    segments = [m for m in out if m["type"] == "segment"]
    assert len(segments) < 4
    assert out[-1] == {"type": "done", "id": 9, "cancelled": True}


def test_cancel_unknown_id_is_ignored():
    data = encode({"type": "cancel", "id": 42}) + encode(LOAD)
    out = run_serve(data, lambda *a: FakeModel())
    assert types(out) == ["ready", "loaded"]


def test_eof_exits_cleanly():
    session = Session(lambda *a: FakeModel())
    out = session.close()
    assert types(out) == ["ready"]


def test_shutdown_exits():
    out = run_serve(encode({"type": "shutdown"}) + encode(LOAD), lambda *a: FakeModel())
    assert types(out) == ["ready"]


def test_unknown_message_type_is_protocol_error():
    out = run_serve(encode({"type": "nope"}), lambda *a: FakeModel())
    assert types(out) == ["ready", "error"]
    assert out[1]["stage"] == "protocol" and out[1]["id"] is None


def test_malformed_line_is_protocol_error():
    out = run_serve(b"not json\n", lambda *a: FakeModel())
    assert out[1]["stage"] == "protocol"


def test_short_audio_is_protocol_error():
    data = encode(transcribe_msg(1), audio(10))[:-5]
    out = run_serve(data, lambda *a: FakeModel())
    assert out[-1]["stage"] == "protocol"
