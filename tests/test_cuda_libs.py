import sys
import types
from pathlib import Path

from whispry.core import cuda_libs


def _fake_nvidia(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "nvidia"
    for package, folder, files in [
        ("cublas", "lib", ["libcublas.so.12", "libcublasLt.so.12", "libnvblas.so.12"]),
        ("cudnn", "bin", ["cudnn_ops64_9.dll", "readme.txt"]),
    ]:
        directory = root / package / folder
        directory.mkdir(parents=True)
        for name in files:
            (directory / name).write_text("")
    monkeypatch.setitem(sys.modules, "nvidia", types.SimpleNamespace(__path__=[str(root)]))
    return root


def test_finds_wheel_library_dirs(tmp_path, monkeypatch):
    root = _fake_nvidia(tmp_path, monkeypatch)
    assert cuda_libs.nvidia_library_dirs() == [root / "cublas" / "lib", root / "cudnn" / "bin"]


def test_selects_only_cuda_libraries_for_os(tmp_path, monkeypatch):
    _fake_nvidia(tmp_path, monkeypatch)
    dirs = cuda_libs.nvidia_library_dirs()
    linux = [p.name for p in cuda_libs._library_files(dirs, "Linux")]
    windows = [p.name for p in cuda_libs._library_files(dirs, "Windows")]
    assert linux == ["libcublas.so.12", "libcublasLt.so.12"]
    assert windows == ["cudnn_ops64_9.dll"]


def test_preload_retries_until_dependencies_are_loaded():
    loaded = set()

    def loader(path):
        # "b" depends on "a" being loaded first.
        if path.name == "b" and "a" not in loaded:
            raise OSError("missing dependency")
        loaded.add(path.name)

    result = cuda_libs._preload([Path("b"), Path("a")], loader)
    assert [p.name for p in result] == ["a", "b"]


def test_preload_gives_up_on_unloadable_files():
    def loader(path):
        raise OSError("broken")

    assert cuda_libs._preload([Path("x")], loader) == []
