#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment missing. Run: just setup" >&2
  exit 1
fi

CUDA_LIBS="$("$PYTHON" - <<'PY'
import os
import nvidia.cublas
import nvidia.cudnn

paths = [
    os.path.join(nvidia.cublas.__path__[0], "lib"),
    os.path.join(nvidia.cudnn.__path__[0], "lib"),
]

print(":".join(paths))
PY
)"

export LD_LIBRARY_PATH="${CUDA_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd "$ROOT"
exec "$PYTHON" "$@"
