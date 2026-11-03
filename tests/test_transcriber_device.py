import pytest

from whispry.core import transcriber
from whispry.core.config import JobConfig
from whispry.core.events import Log
from whispry.core.platform import CPU, CUDA, DeviceOption, SystemInfo

SYSTEM = SystemInfo(
    "Linux",
    "Linux",
    4,
    (
        DeviceOption(CUDA, "GPU", True, compute_types=("int8_float16",)),
        DeviceOption(CPU, "CPU", True, compute_types=("int8",)),
    ),
)


def fake_loader(fail_on):
    calls = []

    def load(model_dir, device, compute_type, system):
        calls.append((device, compute_type))
        if device in fail_on:
            raise transcriber.ModelLoadError(device, RuntimeError("boom"))
        return f"model-on-{device}"

    return load, calls


def test_auto_falls_back_to_cpu_when_gpu_load_fails(monkeypatch, tmp_path):
    load, calls = fake_loader(fail_on={CUDA})
    monkeypatch.setattr(transcriber, "_load_model", load)
    events = []
    model = transcriber.load_model("m", JobConfig(tmp_path), SYSTEM, events.append)
    assert model == "model-on-cpu"
    assert calls == [(CUDA, "int8_float16"), (CPU, "int8")]
    assert any(isinstance(e, Log) and "Falling back" in e.message for e in events)


def test_explicit_gpu_does_not_fall_back(monkeypatch, tmp_path):
    load, _ = fake_loader(fail_on={CUDA})
    monkeypatch.setattr(transcriber, "_load_model", load)
    with pytest.raises(transcriber.ModelLoadError):
        transcriber.load_model("m", JobConfig(tmp_path, device=CUDA), SYSTEM)
