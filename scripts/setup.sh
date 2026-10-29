#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

python3 -m venv .venv

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

# CUDA 12 runtime libraries used by CTranslate2.
.venv/bin/python -m pip install \
  nvidia-cublas-cu12 \
  'nvidia-cudnn-cu12==9.*'

echo
echo "Environment ready."
echo "Run: just check"
