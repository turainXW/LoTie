#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_ROOT="${EVAL90_STATE_ROOT:-$ROOT/.codeagent/eval90}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.10 || true)}"
HARNESS_PYTHON="${HARNESS_PYTHON:-$ROOT/.venv/bin/python}"
COMMAND="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -x "$HARNESS_PYTHON" ]]; then
  echo "Harness venv missing; creating $ROOT/.venv"
  python3 -m venv "$ROOT/.venv"
  "$ROOT/.venv/bin/python" -m pip install --upgrade pip
  "$ROOT/.venv/bin/python" -m pip install -e "$ROOT[dev,swegym]"
fi

if [[ -z "$PYTHON_BIN" || ! -x "$PYTHON_BIN" ]]; then
  echo "Python 3.10 is required. Set PYTHON_BIN=/absolute/path/to/python3.10" >&2
  exit 2
fi

exec "$HARNESS_PYTHON" "$ROOT/scripts/deploy_eval90.py" "$COMMAND" \
  --state-root "$STATE_ROOT" \
  --python "$PYTHON_BIN" \
  "$@"
