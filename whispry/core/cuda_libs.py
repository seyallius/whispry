"""Make NVIDIA runtime libraries installed from pip visible to CTranslate2.

The `[gpu]` extra installs cuBLAS (and cuDNN on Windows) as Python wheels
under site-packages/nvidia/*. CTranslate2 loads them by name at runtime,
which only works if the OS loader can find them. This module loads them
by full path first, so later lookups by name resolve to the loaded copy.
It replaces the old LD_LIBRARY_PATH wrapper script, and works on Windows.
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

# Only these libraries are preloaded; everything else is left alone.
LIBRARY_PREFIXES = ("libcublas", "libcudnn", "cublas", "cudnn")


def nvidia_library_dirs() -> list[Path]:
    """Return lib/ (Linux) and bin/ (Windows) folders of installed NVIDIA wheels."""
    try:
        import nvidia
    except ImportError:
        return []

    dirs = []
    for root in getattr(nvidia, "__path__", []):
        for package in sorted(Path(root).iterdir()):
            for name in ("lib", "bin"):
                candidate = package / name
                if candidate.is_dir():
                    dirs.append(candidate)
    return dirs


def _library_files(dirs: list[Path], os_name: str) -> list[Path]:
    pattern = "*.dll" if os_name == "Windows" else "*.so*"
    return [
        path
        for directory in dirs
        for path in sorted(directory.glob(pattern))
        if path.name.lower().startswith(LIBRARY_PREFIXES)
    ]


def _preload(files: list[Path], loader) -> list[Path]:
    """Load files, retrying until no progress, so dependency order doesn't matter."""
    pending = list(files)
    loaded = []
    while pending:
        failed = []
        for path in pending:
            try:
                loader(path)
                loaded.append(path)
            except OSError:
                failed.append(path)
        if len(failed) == len(pending):
            break
        pending = failed
    return loaded


@lru_cache(maxsize=None)
def prepare_cuda_libraries(os_name: str) -> tuple[Path, ...]:
    """Expose pip-installed CUDA libraries to this process. Safe to call often.

    Returns the libraries that were loaded (empty when none are installed).
    """
    dirs = nvidia_library_dirs()
    if not dirs:
        return ()

    files = _library_files(dirs, os_name)

    if os_name == "Windows":
        for directory in dirs:
            os.add_dll_directory(str(directory))
        # Delay-loaded DLLs use the classic search order, which includes PATH.
        os.environ["PATH"] = os.pathsep.join(
            [*map(str, dirs), os.environ.get("PATH", "")]
        )
        loaded = _preload(files, lambda path: ctypes.WinDLL(str(path)))
    elif os_name == "Linux":
        loaded = _preload(
            files, lambda path: ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        )
    else:
        loaded = []

    return tuple(loaded)
