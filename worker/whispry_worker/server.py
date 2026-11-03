"""The worker: reads commands on stdin, writes results on stdout."""

import os
import platform
import queue
import sys
import threading

from whispry_worker import __version__
from whispry_worker.cuda_libs import prepare_cuda_libraries
from whispry_worker.protocol import read_message, write_message

_STOP = object()


def load_whisper_model(model_dir, device, compute_type, cpu_threads):
    """Load a faster-whisper model from a local directory."""
    prepare_cuda_libraries(platform.system())
    from faster_whisper import WhisperModel

    return WhisperModel(
        str(model_dir),
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads if device == "cpu" else 0,
        local_files_only=True,
    )


def supported_compute_types(device):
    """Compute types this machine can run on `device`."""
    # The pip-installed CUDA libraries must be loaded before CTranslate2 is
    # asked anything about the GPU.
    prepare_cuda_libraries(platform.system())
    import ctranslate2

    return set(ctranslate2.get_supported_compute_types(device))


def _faster_whisper_version():
    try:
        import faster_whisper

        return faster_whisper.__version__
    except ImportError:
        return "unavailable"


class _Reader(threading.Thread):
    """Owns stdin: parses every message and feeds the work queue."""

    def __init__(self, stdin, stdout, work):
        super().__init__(daemon=True)
        self.stdin = stdin
        self.stdout = stdout
        self.work = work
        self.cancels = {}
        self.lock = threading.Lock()

    def run(self):
        try:
            self._loop()
        except (ValueError, EOFError) as exc:
            write_message(
                self.stdout, {"type": "error", "id": None, "stage": "protocol", "message": str(exc)}
            )
        finally:
            with self.lock:
                for event in self.cancels.values():
                    event.set()
            self.work.put(_STOP)

    def _loop(self):
        while True:
            item = read_message(self.stdin)
            if item is None:
                return
            message, audio = item
            kind = message.get("type")
            if kind == "shutdown":
                return
            if kind == "cancel":
                with self.lock:
                    event = self.cancels.get(message.get("id"))
                if event:
                    event.set()
            elif kind == "load":
                self.work.put((message, None, None))
            elif kind == "transcribe":
                event = threading.Event()
                with self.lock:
                    self.cancels[message.get("id")] = event
                self.work.put((message, audio or b"", event))
            else:
                raise ValueError(f"unknown message type: {kind!r}")

    def forget(self, request_id):
        with self.lock:
            self.cancels.pop(request_id, None)


def serve(stdin, stdout, model_factory, supported_types=supported_compute_types):
    """Run the worker until shutdown, EOF or a protocol error."""
    import numpy

    write_message(
        stdout,
        {"type": "ready", "version": __version__, "faster_whisper": _faster_whisper_version()},
    )
    work = queue.Queue()
    reader = _Reader(stdin, stdout, work)
    reader.start()
    model = None

    while True:
        item = work.get()
        if item is _STOP:
            return
        message, audio, cancel = item
        if message["type"] == "load":
            model = _load(stdout, model_factory, supported_types, message)
            continue
        request_id = message.get("id")
        try:
            if model is None:
                raise RuntimeError("No model is loaded")
            samples = numpy.frombuffer(audio, dtype="<f4")
            _transcribe(stdout, model, message, samples, cancel)
        except Exception as exc:
            write_message(
                stdout,
                {"type": "error", "id": request_id, "stage": "transcribe", "message": str(exc)},
            )
        finally:
            reader.forget(request_id)


def _load(stdout, model_factory, supported_types, message):
    try:
        compute_type = _choose_compute_type(
            message["device"], message["compute_types"], supported_types
        )
        model = model_factory(
            message["model_dir"], message["device"], compute_type, message["cpu_threads"]
        )
    except Exception as exc:
        write_message(stdout, {"type": "error", "id": None, "stage": "load", "message": str(exc)})
        return None
    write_message(
        stdout, {"type": "loaded", "device": message["device"], "compute_type": compute_type}
    )
    return model


def _choose_compute_type(device, preferred, supported_types):
    """The first preferred compute type that `device` supports."""
    supported = supported_types(device)
    for compute_type in preferred:
        if compute_type in supported:
            return compute_type
    raise RuntimeError(
        f"None of the compute types {', '.join(preferred)} is supported on {device} "
        f"(supported: {', '.join(sorted(supported))})"
    )


def _transcribe(stdout, model, message, samples, cancel):
    request_id = message["id"]
    segments, info = model.transcribe(
        samples,
        language=message.get("language"),
        beam_size=message["beam_size"],
        vad_filter=True,
        initial_prompt=message.get("initial_prompt"),
    )
    write_message(
        stdout,
        {
            "type": "language",
            "id": request_id,
            "language": info.language,
            "probability": info.language_probability,
        },
    )
    cancelled = False
    for segment in segments:
        write_message(
            stdout,
            {
                "type": "segment",
                "id": request_id,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            },
        )
        if cancel.is_set():
            cancelled = True
            break
    write_message(stdout, {"type": "done", "id": request_id, "cancelled": cancelled})


def protect_stdout():
    """Return a private writer for the real stdout; route everything else to stderr."""
    writer = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return writer


def main():
    stdout = protect_stdout()
    serve(sys.stdin.buffer, stdout, model_factory=load_whisper_model)
