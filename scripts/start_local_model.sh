#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PYTHON="${MODEL_PYTHON:-python3}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a local Hugging Face checkpoint directory}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}" 
MODEL_HOST="${MODEL_HOST:-127.0.0.1}"
MODEL_PORT="${MODEL_PORT:-8000}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/local_model}"

mkdir -p "$LOG_DIR"

"$MODEL_PYTHON" "$ROOT/scripts/serve_local_transformers.py" \
  --model-path "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --host "$MODEL_HOST" \
  --port "$MODEL_PORT" \
  >"$LOG_DIR/server.log" 2>&1 &

pid=$!
echo "$pid" >"$LOG_DIR/server.pid"
echo "Local model starting: pid=$pid log=$LOG_DIR/server.log"
echo "Endpoint: http://$MODEL_HOST:$MODEL_PORT/v1/chat/completions"
