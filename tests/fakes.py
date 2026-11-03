"""Stand-ins for the Whisper model so the pipeline can be tested offline."""

from types import SimpleNamespace

from whispry.core.config import SAMPLE_RATE
from whispry.core.platform import CPU, CUDA, DeviceOption, SystemInfo

CPU_SYSTEM = SystemInfo(
    "Linux",
    "Linux test",
    2,
    (
        DeviceOption(CUDA, "NVIDIA GPU (CUDA)", False, reason="No NVIDIA GPU or driver found"),
        DeviceOption(CPU, "CPU", True, compute_types=("int8", "float32")),
    ),
)


class FakeModel:
    """Emits one segment per 20 s of audio and records every call."""

    def __init__(self, language="en"):
        self.calls = []
        self.language = language

    def transcribe(self, samples, **options):
        self.calls.append({"samples": len(samples), **options})
        seconds = len(samples) / SAMPLE_RATE
        number = len(self.calls)

        def segments():
            start = 0.0
            while start + 1 < seconds:
                end = min(start + 20, seconds)
                yield SimpleNamespace(start=start, end=end, text=f" window {number} at {start:.0f} ")
                start = end

        return segments(), SimpleNamespace(language=self.language, language_probability=0.9)


def install_fake_model(monkeypatch, model=None):
    from whispry.core import transcriber

    model = model or FakeModel()
    monkeypatch.setattr(transcriber, "ensure_model", lambda name, on_event: "model-dir")
    monkeypatch.setattr(transcriber, "_load_model", lambda *args: model)
    return model
