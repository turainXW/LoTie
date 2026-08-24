#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to Qwen3-4B-Instruct-2507 or a local path}"
ACTOR_ADAPTER="${ACTOR_ADAPTER:?Set ACTOR_ADAPTER to the Epoch3 actor adapter}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/unpacked/critic_coldstart_ep3_qwen_mixed_r1_r3_v1}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-${ROOT}/data/unpacked/critic_coldstart_ep3_train398_v3}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs}"
PLAN_OUTPUT="${PLAN_OUTPUT:-${RUN_ROOT}/qwen3_4b_value_model_v1_plan}"
TRAIN_OUTPUT="${TRAIN_OUTPUT:-${RUN_ROOT}/qwen3_4b_value_model_v1}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$RUN_ROOT"

for path in "$PLAN_OUTPUT" "$TRAIN_OUTPUT"; do
  if [[ -e "$path" ]]; then
    echo "Refusing to overwrite existing output: $path" >&2
    exit 2
  fi
done

for data_root in "$DATA_DIR" "$VALIDATION_DATA_DIR"; do
  test -s "$data_root/input_ids.npy"
  test -s "$data_root/action_mask.npy"
  test -s "$data_root/samples.jsonl"
  test -s "$data_root/dataset_manifest.json"
done

COMMON_ARGS=(
  --data-dir "$DATA_DIR"
  --validation-data-dir "$VALIDATION_DATA_DIR"
  --base-model "$BASE_MODEL"
  --actor-adapter "$ACTOR_ADAPTER"
  --train-selection all
  --max-train-len 65536
  --batch-profile critic_96gb_verified
  --target-trajectories-per-update 8
  --epochs 1
  --critic-lr 5e-6
  --value-head-lr 5e-5
  --warmup-steps 10
  --save-every-steps 0
  --seed 20260823
)

"$PYTHON_BIN" "$ROOT/scripts/train_value_model.py" \
  "${COMMON_ARGS[@]}" \
  --output-dir "$PLAN_OUTPUT" \
  --selection-only

"$PYTHON_BIN" "$ROOT/scripts/train_value_model.py" \
  "${COMMON_ARGS[@]}" \
  --output-dir "$TRAIN_OUTPUT"

CHECKPOINT="$TRAIN_OUTPUT/smoke_checkpoint"
test -s "$CHECKPOINT/critic_adapter/adapter_model.safetensors"
test -s "$CHECKPOINT/value_head.safetensors"
test -s "$CHECKPOINT/optimizer.pt"
test -s "$CHECKPOINT/scheduler.pt"
test -s "$CHECKPOINT/rng_state.pt"
test -s "$CHECKPOINT/trainer_state.json"
test -s "$CHECKPOINT/checkpoint_manifest.json"
test -s "$TRAIN_OUTPUT/result.json"

echo "Completed: $TRAIN_OUTPUT"
