import importlib.util
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("whispry_setup", Path(__file__).with_name("setup.py"))
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def make_worker(folder: Path) -> Path:
    (folder / "worker").mkdir(parents=True)
    (folder / "worker" / "pyproject.toml").write_text("")
    return folder


def test_finds_the_worker_next_to_the_script_in_a_release_archive(tmp_path):
    root = make_worker(tmp_path / "whispry-1.0")
    assert setup.find_root(root) == root


def test_finds_the_worker_one_level_up_in_a_source_checkout(tmp_path):
    root = make_worker(tmp_path / "checkout")
    scripts = root / "scripts"
    scripts.mkdir()
    assert setup.find_root(scripts) == root


def test_no_worker_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        setup.find_root(tmp_path)


def test_python_lives_in_a_different_place_on_each_os():
    venv = Path("env")
    assert setup.venv_python(venv, "linux") == venv / "bin" / "python"
    assert setup.venv_python(venv, "win32") == venv / "Scripts" / "python.exe"


def test_gpu_libraries_exist_only_for_linux_and_windows():
    assert setup.gpu_supported("linux") and setup.gpu_supported("win32")
    assert not setup.gpu_supported("darwin")


def test_counts_gpus_in_nvidia_smi_output():
    listing = "GPU 0: NVIDIA GeForce RTX 3060 (UUID: GPU-a)\nGPU 1: NVIDIA RTX A2000 (UUID: GPU-b)\n"
    assert setup.count_gpus(listing) == 2
    assert setup.count_gpus("") == 0
    assert setup.count_gpus("No devices were found") == 0


@pytest.mark.parametrize("gpu, dev, expected", [
    (False, False, ""), (True, False, "gpu"), (False, True, "dev"), (True, True, "gpu,dev"),
])
def test_extras(gpu, dev, expected):
    assert setup.extras(gpu, dev) == expected


def test_install_target():
    worker = Path("w")
    assert setup.install_target(worker, False, False) == ["w"]
    assert setup.install_target(worker, True, False) == ["w[gpu]"]
    assert setup.install_target(worker, True, True) == ["-e", "w[gpu,dev]"]


def test_a_missing_nvidia_smi_means_no_gpu(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(setup.subprocess, "run", missing)
    assert setup.detect_gpu() is False


def test_cuda_libraries_already_on_the_system_are_found(tmp_path):
    for name in setup.required_cuda_libraries("win32"):
        (tmp_path / name).write_text("")
    assert setup.system_has_cuda_libraries("win32", {"PATH": str(tmp_path)})


def test_a_partly_installed_cuda_is_not_enough(tmp_path):
    (tmp_path / "cublas64_12.dll").write_text("")
    assert not setup.system_has_cuda_libraries("win32", {"PATH": str(tmp_path)})
