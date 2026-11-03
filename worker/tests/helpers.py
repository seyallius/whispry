import io
import json
import os
import threading
from types import SimpleNamespace

import numpy as np

from whispry_worker.server import serve


class FakeModel:
    """Stands in for WhisperModel; records calls and yields the given segments."""

    def __init__(self, texts=("  one ", "two", " three  "), gate=None, started=None):
        self.texts = texts
        self.gate = gate
        self.started = started
        self.calls = []

    def transcribe(self, samples, **options):
        self.calls.append((samples, options))
        info = SimpleNamespace(language="en", language_probability=0.9)
        return self._segments(), info

    def _segments(self):
        for i, text in enumerate(self.texts):
            if self.gate is not None and i > 0:
                self.started.set()
                self.gate.wait(5)
            yield SimpleNamespace(start=float(i), end=i + 1.0, text=text)


def encode(message, audio=b""):
    if audio:
        message = dict(message, audio_bytes=len(audio))
    return json.dumps(message).encode() + b"\n" + audio


def transcribe_msg(id=1, **extra):
    return {"type": "transcribe", "id": id, "language": None, "beam_size": 5,
            "initial_prompt": None, **extra}


LOAD = {"type": "load", "model_dir": "m", "device": "cpu", "compute_type": "int8", "cpu_threads": 2}


def run_serve(input_bytes, factory):
    """Run serve() over fixed input; returns the parsed output messages."""
    out = io.BytesIO()
    serve(io.BytesIO(input_bytes), out, factory)
    return parse(out.getvalue())


def parse(data):
    return [json.loads(line) for line in data.splitlines()]


class Session:
    """Runs serve() on a thread with a pipe for stdin, so tests can send messages live."""

    def __init__(self, factory):
        read_fd, self.write_fd = os.pipe()
        self.out = io.BytesIO()
        self.thread = threading.Thread(
            target=serve, args=(os.fdopen(read_fd, "rb"), self.out, factory), daemon=True
        )
        self.thread.start()

    def send(self, message, audio=b""):
        os.write(self.write_fd, encode(message, audio))

    def wait_for(self, kind, timeout=5):
        """Block until an output message of this type has been written."""
        for _ in range(int(timeout / 0.01)):
            if f'"type":"{kind}"'.encode() in self.out.getvalue():
                return
            threading.Event().wait(0.01)
        raise TimeoutError(kind)

    def close(self):
        os.close(self.write_fd)
        self.thread.join(5)
        assert not self.thread.is_alive()
        return parse(self.out.getvalue())


def audio(n):
    return np.arange(n, dtype="<f4").tobytes()
