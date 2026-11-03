#!/usr/bin/env python3
"""Set up Whispry's transcription worker.

Run this once after downloading Whispry (it only needs Python 3.10 or newer):

    python3 setup.py          (Windows: py setup.py)

It creates a private Python environment in the ".venv" folder next to the
whispry program and installs the worker into it, plus the NVIDIA CUDA
libraries when it finds an NVIDIA GPU. Nothing is installed anywhere else.

Options:
    --gpu    install the GPU libraries even if no GPU was detected
    --cpu    skip the GPU libraries even if a GPU was detected
    --dev    for a source checkout: editable install with the test tools
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

MINIMUM_PYTHON = (3, 10)
HERE = Path(__file__).resolve().parent


def find_root(here: Path = HERE) -> Path:
    """The folder holding worker/ and, beside it, .venv and the program.

    A release archive has worker/ next to this script. In a source checkout
    this script lives in scripts/ and worker/ is one level up.
    """
    for candidate in (here, here.parent):
        if (candidate / "worker" / "pyproject.toml").is_file():
            return candidate
    sys.exit("Could not find the worker folder next to this script.")


def venv_python(venv: Path, platform: str = sys.platform) -> Path:
    if platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def gpu_supported(platform: str = sys.platform) -> bool:
    """The NVIDIA libraries are only published for Linux and Windows."""
    return platform.startswith("linux") or platform == "win32"


def count_gpus(listing: str) -> int:
    """Number of GPUs in the output of `nvidia-smi -L`."""
    return sum(1 for line in listing.splitlines() if line.strip().startswith("GPU "))


def detect_gpu() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and count_gpus(result.stdout) > 0


def extras(gpu: bool, dev: bool) -> str:
    return ",".join(name for name, wanted in (("gpu", gpu), ("dev", dev)) if wanted)


def install_target(worker: Path, gpu: bool, dev: bool) -> list[str]:
    """The pip arguments that install the worker."""
    spec = str(worker)
    chosen = extras(gpu, dev)
    if chosen:
        spec += f"[{chosen}]"
    return ["-e", spec] if dev else [spec]


def run(command: list[str], **options) -> subprocess.CompletedProcess:
    return subprocess.run(command, **options)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Set up Whispry's transcription worker.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--gpu", action="store_true", help="install GPU libraries")
    group.add_argument("--cpu", action="store_true", help="skip GPU libraries")
    parser.add_argument("--dev", action="store_true", help="editable install with test tools")
    args = parser.parse_args(argv)

    if sys.version_info < MINIMUM_PYTHON:
        found = ".".join(map(str, sys.version_info[:3]))
        wanted = ".".join(map(str, MINIMUM_PYTHON))
        print(f"Python {wanted} or newer is needed (this is {found}).", file=sys.stderr)
        return 1

    root = find_root()
    venv = root / ".venv"
    python = venv_python(venv)

    if not gpu_supported():
        gpu = False
    elif args.gpu:
        gpu = True
    elif args.cpu:
        gpu = False
    else:
        gpu = detect_gpu()
    print(f"NVIDIA GPU libraries: {'yes' if gpu else 'no'}"
          + ("" if args.gpu or args.cpu else " (detected automatically; use --gpu or --cpu to override)"))

    if not python.is_file():
        print(f"Creating the Python environment in {venv} ...")
        if run([sys.executable, "-m", "venv", str(venv)]).returncode != 0:
            print("Could not create the environment. On Debian or Ubuntu, "
                  "install the python3-venv package first.", file=sys.stderr)
            return 1

    print("Installing the transcription worker (this can take a few minutes) ...")
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check",
               *install_target(root / "worker", gpu, args.dev)]
    if run(command).returncode != 0:
        print("Installing the worker failed. Check your internet connection and try again.",
              file=sys.stderr)
        return 1

    check = run([str(python), "-c", "import faster_whisper, ctranslate2"], capture_output=True, text=True)
    if check.returncode != 0:
        print("The worker was installed but cannot start:\n" + check.stderr, file=sys.stderr)
        return 1

    program = find_program(root)
    if program is None:
        print("\nWorker ready. The whispry program was not found next to this script"
              + (" (build it with: cargo build --release)." if args.dev else "."))
        return 0

    print()
    for subcommand in ("ffmpeg", "devices"):
        run([str(program), subcommand], env={**os.environ, "WHISPRY_PYTHON": str(python)})
        print()
    print(f"All set. Start Whispry with: {program}")
    return 0


def find_program(root: Path) -> Path | None:
    name = "whispry.exe" if sys.platform == "win32" else "whispry"
    candidates = [root / name, root / "target" / "release" / name, root / "target" / "debug" / name]
    return next((path for path in candidates if path.is_file()), None)


if __name__ == "__main__":
    sys.exit(main())
