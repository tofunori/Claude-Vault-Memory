#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="$ROOT_DIR/.venv/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

exec "$PYTHON_BIN" -m uvicorn nas_memory.api:app --host "${MEMORY_API_HOST:-0.0.0.0}" --port "${MEMORY_API_PORT:-8766}"
