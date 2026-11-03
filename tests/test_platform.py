import sys
import types

import pytest

from whispry.core import platform as plat
from whispry.core.platform import (
    CPU,
    CUDA,
    DeviceOption,
    DeviceUnavailable,
    SystemInfo,
    resolve_device,
)

GPU = DeviceOption(CUDA, "GPU", True, compute_types=("int8_float16", "float16"))
NO_GPU = DeviceOption(CUDA, "GPU", False, reason="No NVIDIA GPU or driver found")
CPU_OPTION = DeviceOption(CPU, "CPU", True, compute_types=("int8", "float32"))


def system(gpu: DeviceOption) -> SystemInfo:
    return SystemInfo("Linux", "Linux", 4, (gpu, CPU_OPTION))


def test_auto_prefers_gpu():
    option, compute = resolve_device("auto", system(GPU))
    assert (option.name, compute) == (CUDA, "int8_float16")


def test_auto_falls_back_to_cpu():
    option, compute = resolve_device("auto", system(NO_GPU))
    assert (option.name, compute) == (CPU, "int8")


def test_explicit_unavailable_gpu_raises_with_reason():
    with pytest.raises(DeviceUnavailable, match="No NVIDIA GPU"):
        resolve_device("cuda", system(NO_GPU))


def test_rejects_unsupported_compute_type():
    with pytest.raises(DeviceUnavailable, match="does not support"):
        resolve_device("cpu", system(GPU), "int8_float16")


@pytest.fixture
def fake_ctranslate2(monkeypatch):
    module = types.SimpleNamespace(
        cuda_count=0,
        get_cuda_device_count=lambda: module.cuda_count,
        get_supported_compute_types=lambda device: {
            "cpu": {"int8", "float32", "int8_float32"},
            "cuda": {"float16", "int8_float16", "int8", "float32"},
        }[device],
    )
    monkeypatch.setitem(sys.modules, "ctranslate2", module)
    return module


def test_detect_cuda_without_gpu(fake_ctranslate2):
    assert not plat.detect_cuda("Linux").available


def test_detect_cuda_with_gpu_and_libraries(fake_ctranslate2, monkeypatch):
    fake_ctranslate2.cuda_count = 1
    monkeypatch.setattr(plat, "_missing_cuda_libraries", lambda os_name: [])
    option = plat.detect_cuda("Windows")
    assert option.available
    assert option.default_compute_type == "int8_float16"


def test_detect_cuda_with_gpu_but_missing_libraries(fake_ctranslate2, monkeypatch):
    fake_ctranslate2.cuda_count = 1
    monkeypatch.setattr(plat, "_missing_cuda_libraries", lambda os_name: ["cudnn64_9.dll"])
    option = plat.detect_cuda("Windows")
    assert not option.available
    assert "cudnn64_9.dll" in option.reason


def test_detect_cuda_unsupported_on_macos(fake_ctranslate2):
    assert "not supported" in plat.detect_cuda("Darwin").reason


def test_detect_cpu_orders_compute_types(fake_ctranslate2):
    assert plat.detect_cpu().compute_types == ("int8", "int8_float32", "float32")
