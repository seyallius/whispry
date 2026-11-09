import importlib.util
import sys
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
    (root / "worker" / "whispry_worker" / "__pycache__").mkdir(parents=True)
    (root / "worker" / "tests").mkdir()
    (root / "scripts").mkdir()
    (root / "worker" / "pyproject.toml").write_text("[project]")
    (root / "worker" / "whispry_worker" / "server.py").write_text("")
    (root / "worker" / "whispry_worker" / "__pycache__" / "server.pyc").write_text("")
    (root / "worker" / "tests" / "test_server.py").write_text("")
    (root / "scripts" / "setup.py").write_text("# setup")
    (root / "LICENSE").write_text("license")
    binary = tmp_path / "whispry"
    binary.write_text("program")
    binary.chmod(0o755)
    return root, binary


def test_archive_holds_only_the_program_and_docs_without_worker_or_setup(repo, tmp_path):
    root, binary = repo
    import shutil
    shutil.rmtree(root / "worker")
    (root / "scripts" / "setup.py").unlink()
    (root / "README.md").write_text("readme")
    archive = package.package(root, binary, "1.2.3", "macos-arm64", tmp_path / "dist")

    assert archive.name == "whispry-1.2.3-macos-arm64.tar.gz"
    folder = "whispry-1.2.3-macos-arm64"
    with tarfile.open(archive) as tar:
        assert sorted(tar.getnames()) == sorted(
            [folder] + [f"{folder}/{n}" for n in ["LICENSE", "README.md", "whispry"]])


def test_linux_archive_includes_worker_and_setup_script_when_present(repo, tmp_path):
    root, binary = repo
    archive = package.package(root, binary, "1.2.3", "linux-x86_64", tmp_path / "dist")

    assert archive.name == "whispry-1.2.3-linux-x86_64.tar.gz"
    with tarfile.open(archive) as tar:
        names = sorted(tar.getnames())
        folder = "whispry-1.2.3-linux-x86_64"
        assert names == sorted(
            [folder] + [f"{folder}/{name}" for name in [
                "LICENSE", "setup.py", "whispry", "worker", "worker/pyproject.toml",
                "worker/whispry_worker", "worker/whispry_worker/server.py"]]
        )
        if sys.platform != "win32":  # Windows files have no executable bit
            assert tar.getmember("whispry-1.2.3-linux-x86_64/whispry").mode & 0o111


def test_windows_archive_is_a_zip(repo, tmp_path):
    root, binary = repo
    exe = binary.with_name("whispry.exe")
    binary.rename(exe)
    archive = package.package(root, exe, "1.2.3", "windows-x86_64", tmp_path / "dist")

    assert archive.name == "whispry-1.2.3-windows-x86_64.zip"
    with zipfile.ZipFile(archive) as zip_file:
        names = zip_file.namelist()
    assert "whispry-1.2.3-windows-x86_64/whispry.exe" in names
    assert "whispry-1.2.3-windows-x86_64/worker/whispry_worker/server.py" in names
    assert not any("tests" in name or "__pycache__" in name for name in names)


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
    assert not any("/tests" in name for name in names)
