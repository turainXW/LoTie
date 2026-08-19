#!/usr/bin/env bash
set -uo pipefail

ROOT="${LOTTIE_SFT_ROOT:-/root/autodl-tmp/lottie_sft}"
PYTHON="${LOTTIE_SFT_PYTHON:-/root/autodl-tmp/toolopd_grpo/.venv/bin/python}"
MODEL="${LOTTIE_MODEL_PATH:-/root/autodl-tmp/model_cache/Qwen/Qwen3-4B-Instruct-2507}"
DATASET="${LOTTIE_DATASET_DIR:-$ROOT/tokenized/qwen3_4b_instruct_2507_primary_32k_maskv2/dataset}"
PROFILE="${LOTTIE_BATCH_PROFILE:-rtx_pro_6000_verified}"
SMOKE_ROOT="$ROOT/smoke/bucket_profile_${PROFILE}"
SUMMARY="$SMOKE_ROOT/summary.tsv"

mkdir -p "$SMOKE_ROOT"
printf 'bucket\treturncode\tresult\n' > "$SUMMARY"

for bucket in 4096 8192 16384 24576; do
  output_dir="$SMOKE_ROOT/bucket_${bucket}"
  log_file="$SMOKE_ROOT/bucket_${bucket}.log"
  rm -rf "$output_dir"
  "$PYTHON" "$ROOT/train_qwen_lottie_lora.py" \
    --dataset-dir "$DATASET" \
    --model-path "$MODEL" \
    --output-dir "$output_dir" \
    --smoke-bucket "$bucket" \
    --epochs 1 \
    --max-steps 1 \
    --gradient-accumulation-steps 1 \
    --batch-profile "$PROFILE" \
    --attn-implementation sdpa \
    --no-save-final \
    >"$log_file" 2>&1
  returncode=$?
  if [[ -f "$output_dir/result.json" ]]; then
    result="$output_dir/result.json"
  else
    result="$log_file"
  fi
  printf '%s\t%s\t%s\n' "$bucket" "$returncode" "$result" >> "$SUMMARY"
done

cat "$SUMMARY"
