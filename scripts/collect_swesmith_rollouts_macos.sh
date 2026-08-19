#!/bin/zsh
set -euo pipefail

ROOT=${0:A:h:h}
SELECTION=${SELECTION:-$ROOT/data/swesmith_train_extension_v1/selection.json}
SOURCE_TASKS=${SOURCE_TASKS:-$ROOT/data/swesmith_train_extension_v1/tasks_runnable.jsonl}
RUNTIME_DIR=${RUNTIME_DIR:-$ROOT/data/runtime/swesmith_train_extension_v1}
RUNTIME_TASKS=${RUNTIME_TASKS:-$RUNTIME_DIR/tasks_runtime.jsonl}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/.codeagent/swesmith_train_extension_v1_portable}
RUN_DIR=${RUN_DIR:-$ROOT/outputs/training_runs/deepseek_v4_flash_swesmith_extension100_rollout4_v1}
MODEL=${MODEL:-deepseek-v4-flash}
MODEL_URL=${MODEL_URL:-https://api.deepseek.com/chat/completions}
API_KEY_ENV=${API_KEY_ENV:-DEEPSEEK_API_KEY}
SAMPLES=${SAMPLES:-4}
WORKERS=${WORKERS:-4}
SETUP_WORKERS=${SETUP_WORKERS:-4}
PYTHON310=${PYTHON310:-}
BOOTSTRAP=${BOOTSTRAP:-1}
BOOTSTRAP_PROJECT=${BOOTSTRAP_PROJECT:-1}
PROJECT_PYTHON=${PROJECT_PYTHON:-}
USE_CAFFEINATE=${USE_CAFFEINATE:-1}

if [[ ! -x $ROOT/.venv/bin/python ]]; then
  if [[ $BOOTSTRAP_PROJECT != 1 ]]; then
    print -u2 -- "Project venv missing: $ROOT/.venv/bin/python"
    print -u2 -- "Set BOOTSTRAP_PROJECT=1 or run: bash scripts/install_dev.sh"
    exit 2
  fi
  if [[ -z $PROJECT_PYTHON ]]; then
    PROJECT_PYTHON=$(command -v python3.10 || command -v python3 || true)
  fi
  if [[ -z $PROJECT_PYTHON ]]; then
    print -u2 -- "No Python interpreter found for the project venv"
    exit 2
  fi
  print -r -- "Creating project venv with $PROJECT_PYTHON"
  env PYTHON_BIN="$PROJECT_PYTHON" VENV_DIR="$ROOT/.venv" \
    bash "$ROOT/scripts/install_dev.sh" 'dev,swegym'
fi

if [[ -z ${(P)API_KEY_ENV:-} ]]; then
  print -u2 -- "$API_KEY_ENV is not available in the current shell"
  exit 2
fi

if (( WORKERS < 1 || SETUP_WORKERS < 1 || SAMPLES < 1 )); then
  print -u2 -- "WORKERS, SETUP_WORKERS, and SAMPLES must be positive"
  exit 2
fi

mkdir -p "$RUNTIME_DIR" "$RUN_DIR"

if [[ $BOOTSTRAP == 1 || ! -s $RUNTIME_TASKS ]]; then
  bootstrap=(
    "$ROOT/.venv/bin/python" "$ROOT/scripts/bootstrap_swesmith_venvs.py"
    --tasks "$SOURCE_TASKS"
    --cache-root "$CACHE_ROOT"
    --output-dir "$RUNTIME_DIR"
    --workers "$SETUP_WORKERS"
    --smoke-per-repo 1
  )
  if [[ -n $PYTHON310 ]]; then
    bootstrap+=(--python "$PYTHON310")
  fi
  "${bootstrap[@]}"
fi

PID_FILE=$RUN_DIR/collector.pid
if [[ -s $PID_FILE ]]; then
  previous_pid=$(<"$PID_FILE")
  if kill -0 "$previous_pid" 2>/dev/null; then
    print -u2 -- "Collector already running: pid=$previous_pid"
    exit 3
  fi
fi

print -r -- $$ > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT INT TERM

runner=(
  "$ROOT/.venv/bin/python" "$ROOT/scripts/run_pass_at_k_experiment.py"
  --selection "$SELECTION"
  --split train
  --evalplus-tasks "$ROOT/data/evalplus_local/tasks_agent.jsonl"
  --swesmith-tasks "$RUNTIME_TASKS"
  --dataset swesmith_py
  --samples "$SAMPLES"
  --workers "$WORKERS"
  --model "$MODEL"
  --model-url "$MODEL_URL"
  --api-key-env "$API_KEY_ENV"
  --temperature 1.0
  --top-p 0.95
  --protocol-retry-temperature 0.0
  --max-tokens 4000
  --thinking-mode enabled
  --reasoning-effort max
  --repo-max-steps 40
  --context-max-tokens 64000
  --model-timeout-sec 180
  --repo-verify-timeout-sec 600
  --max-infra-attempts 3
  --output-dir "$RUN_DIR"
)

print -r -- "collector_started pid=$$ workers=$WORKERS samples=$SAMPLES time=$(date '+%Y-%m-%dT%H:%M:%S%z')" \
  | tee -a "$RUN_DIR/collection.log"
print -r -- "runtime_tasks=$RUNTIME_TASKS" | tee -a "$RUN_DIR/collection.log"

if [[ $USE_CAFFEINATE == 1 ]] && command -v caffeinate >/dev/null 2>&1; then
  caffeinate -dimsu "${runner[@]}" 2>&1 | tee -a "$RUN_DIR/collection.log"
else
  "${runner[@]}" 2>&1 | tee -a "$RUN_DIR/collection.log"
fi
