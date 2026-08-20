#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

EXTRAS="${1:-dev,swegym}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel

if [ "$EXTRAS" = "base" ] || [ -z "$EXTRAS" ]; then
  python -m pip install -e .
else
  python -m pip install -e ".[$EXTRAS]"
fi

# iCloud-backed folders can mark generated .pth files as hidden. Python skips
# those files, which breaks editable imports and installed CLI entrypoints.
if command -v chflags >/dev/null 2>&1; then
  SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  find "$SITE_PACKAGES" -maxdepth 1 -type f -name '*.pth' -exec chflags nohidden {} +
fi

python -c 'import code_agent_baseline'
lotie --help >/dev/null

mkdir -p .codeagent/memory
codeagent repo-map --repo . --task "initialize repository context" >/dev/null
codeagent skills --config .codeagent/skills.json init >/dev/null
codeagent memory --memory-dir .codeagent/memory init >/dev/null

if [ ! -f .env ]; then
  cp .env.example .env
fi

echo "Installed LoTie development environment."
echo "Virtual environment: $PWD/$VENV_DIR"
echo "Installed extras: $EXTRAS"
echo "Next: edit .env, then run: source $VENV_DIR/bin/activate"
echo "CLI: lotie --help (compatibility aliases: minicoder, codeagent)"
echo "Smoke test: PYTHONPATH=src:benchmarks/leetcode_top10 python -m unittest discover -s benchmarks -p 'test_*.py'"
