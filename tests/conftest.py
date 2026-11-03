import shutil
import subprocess
from pathlib import Path

import pytest


def make_audio(path: Path, filter_graph: str) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", filter_graph, str(path)],
        check=True,
    )
    return path


@pytest.fixture
def sine_wav(tmp_path: Path) -> Path:
    return make_audio(tmp_path / "sine.wav", "sine=frequency=440:duration=3")
