#!/usr/bin/env python3
"""Set up Whispry in a local virtual environment on Linux, Windows, or macOS.

    python3 scripts/setup.py          (Windows: py scripts\\setup.py)

Detects an NVIDIA GPU and installs GPU support only when one is present,
then checks for FFmpeg and asks how to get it if it is missing.
Uses only the standard library, so it runs before anything is installed.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
MIN_PYTHON = (3, 10)


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def venv_command(name: str) -> Path:
    if platform.system() == "Windows":
        return VENV / "Scripts" / f"{name}.exe"
    return VENV / "bin" / name


def has_nvidia_gpu() -> bool:
    if platform.system() not in ("Linux", "Windows"):
        return False
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        result = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def run(*command) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument("--gpu", action="store_true", help="Install GPU support even if no GPU is detected")
    gpu.add_argument("--cpu", action="store_true", help="Skip GPU support even if a GPU is detected")
    parser.add_argument("--dev", action="store_true", help="Also install development tools (pytest)")
    parser.add_argument("--skip-ffmpeg", action="store_true", help="Don't check for FFmpeg")
    args = parser.parse_args()

    if sys.version_info < MIN_PYTHON:
        sys.exit(f"Whispry needs Python {'.'.join(map(str, MIN_PYTHON))} or newer; "
                 f"this is {platform.python_version()}.")

    print(f"System: {platform.system()} {platform.release()} ({platform.machine()}), "
          f"Python {platform.python_version()}")

    if args.gpu:
        use_gpu = True
        print("GPU support: requested with --gpu")
    elif args.cpu:
        use_gpu = False
        print("GPU support: skipped with --cpu")
    else:
        use_gpu = has_nvidia_gpu()
        print("GPU support: NVIDIA GPU detected, installing CUDA libraries" if use_gpu
              else "GPU support: no NVIDIA GPU detected, installing CPU-only")

    if not venv_python().exists():
        print(f"Creating virtual environment in {VENV}")
        venv.create(VENV, with_pip=True)

    extras = [name for name, wanted in (("gpu", use_gpu), ("dev", args.dev)) if wanted]
    target = f".[{','.join(extras)}]" if extras else "."

    python = venv_python()
    run(python, "-m", "pip", "install", "--upgrade", "pip")
    run(python, "-m", "pip", "install", "-e", target)

    if not args.skip_ffmpeg:
        # Reuses the app's own check, which asks before installing anything.
        subprocess.run([str(python), "-m", "whispry", "ffmpeg"], cwd=ROOT)

    print()
    run(python, "-m", "whispry", "devices")
    print()
    print("Whispry is ready. Start it with:")
    print(f"  {venv_command('whispry')}")
    print("or activate the environment first and then just run `whispry`:")
    if platform.system() == "Windows":
        print(r"  .venv\Scripts\activate")
    else:
        print("  source .venv/bin/activate")


if __name__ == "__main__":
    main()
