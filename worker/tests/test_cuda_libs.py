from whispry_worker.cuda_libs import _library_files, _preload


def make_tree(tmp_path):
    lib = tmp_path / "nvidia" / "cublas" / "lib"
    bin_ = tmp_path / "nvidia" / "cudnn" / "bin"
    lib.mkdir(parents=True)
    bin_.mkdir(parents=True)
    for name in ("libcublas.so.12", "libcublasLt.so.12", "libnvblas.so.12"):
        (lib / name).touch()
    for name in ("cudnn_ops64_9.dll", "readme.txt"):
        (bin_ / name).touch()
    return [lib, bin_]


def test_linux_selection(tmp_path):
    files = _library_files(make_tree(tmp_path), "Linux")
    assert [f.name for f in files] == ["libcublas.so.12", "libcublasLt.so.12"]


def test_windows_selection(tmp_path):
    files = _library_files(make_tree(tmp_path), "Windows")
    assert [f.name for f in files] == ["cudnn_ops64_9.dll"]


def test_preload_retries_until_dependencies_load():
    loaded = []

    def loader(name):
        if name == "b" and "a" not in loaded:
            raise OSError("needs a")
        loaded.append(name)

    assert _preload(["b", "a"], loader) == ["a", "b"]


def test_preload_gives_up_on_unloadable():
    def loader(name):
        raise OSError("no")

    assert _preload(["x", "y"], loader) == []
