#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" nas_memory/burnin/collector.py \
  --duration-hours 72 \
  --mode mixed \
  --gate strict

