#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESTINATION="${1:-${ROOT}/data/unpacked}"

mkdir -p "$DESTINATION"
tar -xzf "$ROOT/data/critic_ep3_qwen_mixed_r1_r3_training_data_v1.tar.gz" \
  -C "$DESTINATION"
tar -xzf "$ROOT/data/critic_ep3_fixed_holdout40_v1.tar.gz" \
  -C "$DESTINATION"

echo "Unpacked datasets under: $DESTINATION"
