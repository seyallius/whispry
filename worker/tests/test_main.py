import json
import os
import subprocess
import sys
from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent.parent


def env():
    return {**os.environ, "PYTHONPATH": str(WORKER_DIR)}


def test_python_m_whispry_worker_sends_ready():
    proc = subprocess.Popen(
        [sys.executable, "-m", "whispry_worker"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env(),
    )
    first = json.loads(proc.stdout.readline())
    assert first["type"] == "ready"
    assert "faster_whisper" in first
    proc.stdin.write(b'{"type":"shutdown"}\n')
    proc.stdin.flush()
    assert proc.wait(10) == 0
    assert proc.stdout.read() == b""


def test_stray_prints_never_reach_stdout():
    script = (
        "import os\n"
        "from whispry_worker.server import protect_stdout\n"
        "from whispry_worker.protocol import write_message\n"
        "out = protect_stdout()\n"
        "print('noise')\n"
        "os.write(1, b'raw noise\\n')\n"
        "write_message(out, {'type': 'ready'})\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, env=env(), timeout=30)
    assert result.stdout == b'{"type":"ready"}\n'
    assert b"noise" in result.stderr
