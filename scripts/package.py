#!/usr/bin/env python3
"""Build a release archive: the program, LICENSE and README.md.

    python3 scripts/package.py --binary target/release/whispry \
        --version 0.3.0 --target linux-x86_64 --out dist

Windows targets get a .zip, everything else a .tar.gz. Unzipping gives one
folder, whispry-<version>-<target>. The target may end in -cuda for the CUDA
edition, for example linux-x86_64-cuda.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def build_folder(root: Path, binary: Path, destination: Path) -> None:
    """Lay out the release folder."""
    destination.mkdir(parents=True)
    shutil.copy2(binary, destination / binary.name)
    for name in ("LICENSE", "README.md"):
        if (root / name).is_file():
            shutil.copy2(root / name, destination / name)


def make_archive(folder: Path, out: Path, windows: bool) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    if windows:
        archive = out / f"{folder.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for path in sorted(folder.rglob("*")):
                zip_file.write(path, path.relative_to(folder.parent))
    else:
        archive = out / f"{folder.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(folder, arcname=folder.name)
    return archive


def package(root: Path, binary: Path, version: str, target: str, out: Path) -> Path:
    if not binary.is_file():
        sys.exit(f"The program was not found: {binary}")
    name = f"whispry-{version}-{target}"
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch) / name
        build_folder(root, binary, folder)
        return make_archive(folder, out, windows=target.startswith("windows"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a release archive.")
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--target", required=True, help="for example linux-x86_64")
    parser.add_argument("--out", type=Path, default=Path("dist"))
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    print(package(args.root, args.binary, args.version, args.target, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
