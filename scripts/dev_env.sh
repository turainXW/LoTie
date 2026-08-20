#!/usr/bin/env bash
# Source this file from the project root:
#   source scripts/dev_env.sh

if [ -n "${BASH_SOURCE:-}" ]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
else
  SCRIPT_PATH="${(%):-%x}"
fi

ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT/.venv}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Missing virtual environment: $VENV_DIR" >&2
  echo "Run: bash scripts/install_dev.sh" >&2
  return 1 2>/dev/null || exit 1
fi

source "$VENV_DIR/bin/activate"

if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
fi

export PYTHONPATH="$ROOT/src:$ROOT/benchmarks/leetcode_top10${PYTHONPATH:+:$PYTHONPATH}"
export MSWEA_GLOBAL_CONFIG_DIR="${MSWEA_GLOBAL_CONFIG_DIR:-$ROOT/.mswea}"
export MSWEA_SILENT_STARTUP="${MSWEA_SILENT_STARTUP:-1}"

mkdir -p "$MSWEA_GLOBAL_CONFIG_DIR"

echo "LoTie dev env active"
echo "python=$(command -v python)"
echo "lotie=$(command -v lotie)"
echo "minicoder=$(command -v minicoder)"
echo "codeagent=$(command -v codeagent) (legacy alias)"
