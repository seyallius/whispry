"""Make NVIDIA runtime libraries installed from pip wheels loadable in-process."""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

_PREFIXES = ("libcublas", "libcudnn", "cublas", "cudnn")


def nvidia_library_dirs():
    """Return the lib/ and bin/ folders of every package in the `nvidia` namespace."""
    try:
        import nvidia
    except ImportError:
        return []
    dirs = []
    for root in getattr(nvidia, "__path__", []):
        for package in sorted(Path(root).iterdir()):
            for name in ("lib", "bin"):
                folder = package / name
                if folder.is_dir():
                    dirs.append(folder)
    return dirs


def _library_files(dirs, os_name):
    """Pick the cuBLAS/cuDNN shared libraries from the given folders."""
    files = []
    for folder in dirs:
        found = []
        for path in Path(folder).iterdir():
            if not path.is_file():
                continue
            name = path.name.lower()
            if os_name == "Windows":
                wanted = name.endswith(".dll")
            else:
                wanted = ".so" in name and name.startswith(_PREFIXES)
            if wanted:
                found.append(path)
        files.extend(sorted(found))
    return files


def _preload(files, loader):
    """Load files, retrying until no more progress (libraries depend on each other)."""
    pending = list(files)
    loaded = []
    progress = True
    while pending and progress:
        progress = False
        for path in list(pending):
            try:
                loader(path)
            except OSError:
                continue
            pending.remove(path)
            loaded.append(path)
            progress = True
    return loaded


@lru_cache
def prepare_cuda_libraries(os_name):
    """Preload pip-installed CUDA libraries; returns the tuple of loaded files."""
    dirs = nvidia_library_dirs()
    if not dirs:
        return ()
    files = _library_files(dirs, os_name)
    if os_name == "Windows":
        for folder in dirs:
            os.add_dll_directory(str(folder))
        os.environ["PATH"] = os.pathsep.join(
            [str(d) for d in dirs] + [os.environ.get("PATH", "")]
        )
        return tuple(_preload(files, lambda p: ctypes.WinDLL(str(p))))
    if os_name == "Linux":
        return tuple(
            _preload(files, lambda p: ctypes.CDLL(str(p), mode=ctypes.RTLD_GLOBAL))
        )
    return ()
