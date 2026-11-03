"""Detect the operating system and which compute devices can run Whisper."""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass, field
from functools import lru_cache

CPU = "cpu"
CUDA = "cuda"
AUTO = "auto"

# Best choice first. The first one the device supports becomes the default.
PREFERRED_COMPUTE_TYPES = {
    CUDA: ("int8_float16", "float16", "int8", "float32"),
    CPU: ("int8", "int8_float32", "float32"),
}

# Libraries CTranslate2 needs at runtime for CUDA, per OS.
CUDA_LIBRARIES = {
    "Linux": ("libcublas.so.12", "libcudnn.so.9"),
    "Windows": ("cublas64_12.dll", "cudnn64_9.dll"),
}


@dataclass(frozen=True)
class DeviceOption:
    name: str
    label: str
    available: bool
    reason: str | None = None
    compute_types: tuple[str, ...] = ()

    @property
    def default_compute_type(self) -> str | None:
        return self.compute_types[0] if self.compute_types else None


@dataclass(frozen=True)
class SystemInfo:
    os_name: str
    os_label: str
    cpu_threads: int
    devices: tuple[DeviceOption, ...] = field(default_factory=tuple)

    def device(self, name: str) -> DeviceOption:
        for option in self.devices:
            if option.name == name:
                return option
        raise KeyError(name)

    @property
    def available_devices(self) -> tuple[DeviceOption, ...]:
        return tuple(option for option in self.devices if option.available)

    @property
    def gpu(self) -> DeviceOption:
        return self.device(CUDA)


class DeviceUnavailable(RuntimeError):
    pass


def _ordered(device: str, supported: set[str]) -> tuple[str, ...]:
    preferred = [t for t in PREFERRED_COMPUTE_TYPES[device] if t in supported]
    return tuple(preferred)


def _missing_cuda_libraries(os_name: str) -> list[str]:
    missing = []
    for name in CUDA_LIBRARIES.get(os_name, ()):
        try:
            ctypes.CDLL(name)
        except OSError:
            missing.append(name)
    return missing


def detect_cpu() -> DeviceOption:
    import ctranslate2

    return DeviceOption(
        name=CPU,
        label="CPU",
        available=True,
        compute_types=_ordered(CPU, set(ctranslate2.get_supported_compute_types(CPU))),
    )


def detect_cuda(os_name: str) -> DeviceOption:
    import ctranslate2

    def unavailable(reason: str) -> DeviceOption:
        return DeviceOption(CUDA, "NVIDIA GPU (CUDA)", False, reason)

    if os_name not in CUDA_LIBRARIES:
        return unavailable(f"CUDA is not supported on {os_name}")

    try:
        count = ctranslate2.get_cuda_device_count()
    except Exception as error:  # Driver problems surface as generic errors.
        return unavailable(f"Could not query CUDA: {error}")

    if count == 0:
        return unavailable("No NVIDIA GPU or driver found")

    missing = _missing_cuda_libraries(os_name)
    if missing:
        return unavailable(
            "NVIDIA GPU found, but CUDA libraries are missing "
            f"({', '.join(missing)}). Install GPU support: pip install 'whispry[gpu]'"
        )

    supported = set(ctranslate2.get_supported_compute_types(CUDA))
    label = "NVIDIA GPU (CUDA)" if count == 1 else f"NVIDIA GPU (CUDA, {count} devices)"
    return DeviceOption(CUDA, label, True, compute_types=_ordered(CUDA, supported))


@lru_cache(maxsize=1)
def detect_system() -> SystemInfo:
    """Inspect this machine once. Cached because CUDA probing is slow-ish."""
    os_name = platform.system()
    os_label = f"{os_name} {platform.release()} ({platform.machine()})"

    return SystemInfo(
        os_name=os_name,
        os_label=os_label,
        cpu_threads=os.cpu_count() or 1,
        devices=(detect_cuda(os_name), detect_cpu()),
    )


def resolve_device(
    requested: str,
    system: SystemInfo,
    compute_type: str | None = None,
) -> tuple[DeviceOption, str]:
    """Pick the device and compute type to use for a job.

    `auto` prefers the GPU and falls back to the CPU.
    """
    if requested == AUTO:
        option = system.available_devices[0]
    else:
        try:
            option = system.device(requested)
        except KeyError:
            raise DeviceUnavailable(f"Unknown device: {requested}") from None
        if not option.available:
            raise DeviceUnavailable(f"{option.label} is unavailable: {option.reason}")

    if compute_type is None:
        compute_type = option.default_compute_type
    elif compute_type not in option.compute_types:
        raise DeviceUnavailable(
            f"{option.label} does not support compute type {compute_type}. "
            f"Supported: {', '.join(option.compute_types)}"
        )

    return option, compute_type
