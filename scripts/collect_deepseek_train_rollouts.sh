#!/bin/zsh
set -euo pipefail

ROOT=${0:A:h:h}
RUN_DIR=${1:-$ROOT/outputs/training_runs/deepseek_v4_flash_train300_rollout4_v1}

mkdir -p "$RUN_DIR"
exec >> "$RUN_DIR/collection.log" 2>&1

if [[ -z ${DEEPSEEK_API_KEY:-} ]]; then
  print -u2 -- "DEEPSEEK_API_KEY is not available in the login shell"
  exit 1
fi

print -r -- $$ > "$RUN_DIR/collector.pid"
print -r -- "collector_started pid=$$ time=$(date '+%Y-%m-%dT%H:%M:%S%z')"

exec "$ROOT/.venv/bin/python" "$ROOT/scripts/run_pass_at_k_experiment.py" \
  --selection "$ROOT/data/dataset_splits_v1/selection.json" \
  --split train \
  --samples 4 \
  --workers 1 \
  --output-dir "$RUN_DIR" \
  --protocol-retry-temperature 0.0
