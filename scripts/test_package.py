import importlib.util
import tarfile
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("whispry_package", Path(__file__).with_name("package.py"))
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    (root / "worker").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "worker" / "server.py").write_text("")
    (root / "scripts" / "setup.py").write_text("# setup")
    (root / "LICENSE").write_text("license")
    (root / "README.md").write_text("readme")
    binary = tmp_path / "whispry"
    binary.write_text("program")
    binary.chmod(0o755)
    return root, binary


def folder_entries(folder, names):
    return sorted([folder] + [f"{folder}/{n}" for n in names])


def test_archive_holds_only_the_program_and_docs(repo, tmp_path):
    root, binary = repo
    archive = package.package(root, binary, "1.2.3", "macos-arm64", tmp_path / "dist")

    assert archive.name == "whispry-1.2.3-macos-arm64.tar.gz"
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == folder_entries(
            "whispry-1.2.3-macos-arm64", ["LICENSE", "README.md", "whispry"])


def test_worker_and_setup_script_are_never_included(repo, tmp_path):
    root, binary = repo
    archive = package.package(root, binary, "1.2.3", "linux-x86_64", tmp_path / "dist")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert not any("worker" in n or "setup.py" in n for n in names)


def test_cuda_edition_target_name(repo, tmp_path):
    root, binary = repo
    archive = package.package(root, binary, "1.2.3", "linux-x86_64-cuda", tmp_path / "dist")
    assert archive.name == "whispry-1.2.3-linux-x86_64-cuda.tar.gz"
    with tarfile.open(archive) as tar:
        assert "whispry-1.2.3-linux-x86_64-cuda/whispry" in tar.getnames()


def test_windows_archive_is_a_zip(repo, tmp_path):
    root, binary = repo
    exe = binary.with_name("whispry.exe")
    binary.rename(exe)
    archive = package.package(root, exe, "1.2.3", "windows-x86_64", tmp_path / "dist")

    assert archive.name == "whispry-1.2.3-windows-x86_64.zip"
    with zipfile.ZipFile(archive) as zip_file:
        names = zip_file.namelist()
    assert "whispry-1.2.3-windows-x86_64/whispry.exe" in names
    assert "whispry-1.2.3-windows-x86_64/LICENSE" in names


def test_windows_cuda_archive_is_a_zip(repo, tmp_path):
    root, binary = repo
    exe = binary.with_name("whispry.exe")
    binary.rename(exe)
    archive = package.package(root, exe, "1.2.3", "windows-x86_64-cuda", tmp_path / "dist")
    assert archive.name == "whispry-1.2.3-windows-x86_64-cuda.zip"


def test_a_missing_program_is_an_error(repo, tmp_path):
    root, _ = repo
    with pytest.raises(SystemExit):
        package.package(root, tmp_path / "nothing", "1.0.0", "linux-x86_64", tmp_path / "dist")


def test_the_real_repository_packages(tmp_path):
    binary = tmp_path / "whispry"
    binary.write_text("program")
    archive = package.package(package.ROOT, binary, "0.0.0", "linux-x86_64", tmp_path / "dist")
    with tarfile.open(archive) as tar:
        names = tar.getnames()
    assert "whispry-0.0.0-linux-x86_64/whispry" in names
    assert "whispry-0.0.0-linux-x86_64/LICENSE" in names
    assert not any("worker" in name for name in names)
