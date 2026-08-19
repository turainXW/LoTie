#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-v1}"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
PACKAGE_NAME="trace-repair-agent-macos-trajectories-$VERSION"
STAGING="$DIST_DIR/$PACKAGE_NAME"
ARCHIVE="$DIST_DIR/$PACKAGE_NAME.tar.gz"

rm -rf "$STAGING" "$ARCHIVE" "$ARCHIVE.sha256"
mkdir -p "$STAGING" "$DIST_DIR"

copy_path() {
  local relative="$1"
  if [[ -e "$ROOT/$relative" ]]; then
    mkdir -p "$STAGING/$(dirname "$relative")"
    cp -R "$ROOT/$relative" "$STAGING/$relative"
  fi
}

for path in \
  src scripts tools benchmarks configs docs examples \
  pyproject.toml requirements-dev.txt requirements-full.txt requirements-local-llm.txt \
  requirements-mini.txt requirements-swegym.txt \
  README.md README_zh.md DEPLOY_zh.md DEPLOY_EVAL90_zh.md OPEN_SOURCE_CHECKLIST_zh.md \
  Dockerfile.runtime docker-compose.runtime.yml .env.example .gitignore harness_versions.json; do
  copy_path "$path"
done

copy_path data/dataset_splits_v1
copy_path data/eval90
copy_path data/evalplus_local/tasks_agent.jsonl
copy_path data/evalplus_local/tasks_runnable.manifest.json
copy_path data/swesmith_local/tasks_runnable.jsonl
copy_path data/swesmith_local/tasks_runnable.manifest.json
copy_path data/swesmith_train_extension_v1/README.md
copy_path data/swesmith_train_extension_v1/selection.json
copy_path data/swesmith_train_extension_v1/tasks_runnable.jsonl
copy_path data/swesmith_train_extension_v1/audit_report.json

copy_path reports/Lottie_Code_Agent_FULL_PROJECT_REPORT_ZH.md
copy_path reports/lottie_train1200_cleaning_report_zh.md
copy_path reports/qwen_sft_mask_v2_20260815/README_zh.md
copy_path reports/qwen_sft_mask_v2_20260815/experiment_config_e2_v1.json
copy_path reports/qwen_v3_eval_20260816/README.md
copy_path reports/qwen_v3_eval_20260816/DETAILED_ANALYSIS_ZH.md
copy_path reports/qwen_v3_eval_20260816/metrics.csv
copy_path reports/qwen_v3_eval_20260816/chart_data.json
copy_path reports/qwen_v3_eval_20260816/detailed_analysis_data.json
copy_path reports/qwen_v3_eval_20260816/charts
copy_path reports/qwen_v3_eval_20260816/plot_v3_eval_results.py
copy_path reports/qwen_v3_eval_20260816/build_detailed_analysis.py

CLEANED="$ROOT/outputs/training_packages/lottie_train1200_cleaned_complete_v1"
TRAJECTORY_OUT="$STAGING/data/trajectories/lottie_train1200_cleaned_complete_v1"
mkdir -p "$TRAJECTORY_OUT/data" "$TRAJECTORY_OUT/audit" "$TRAJECTORY_OUT/metadata" "$TRAJECTORY_OUT/report"

for path in \
  data/sft_primary.jsonl data/sft_optional_tier_b.jsonl data/rl_rollouts.jsonl \
  audit/repair_audit.jsonl audit/trajectory_audit.jsonl audit/salvage_audit_summary.json \
  audit/sft_exclusions.jsonl audit/sft_manual_review.jsonl \
  metadata/clean_manifest.json metadata/strict_manifest.json metadata/salvage_manifest.json \
  metadata/qwen3_4b_token_lengths.json metadata/pass_at_k_summary.json \
  report/CLEANING_REPORT_ZH.md manifest.json; do
  if [[ -f "$CLEANED/$path" ]]; then
    mkdir -p "$TRAJECTORY_OUT/$(dirname "$path")"
    cp "$CLEANED/$path" "$TRAJECTORY_OUT/$path"
  fi
done

find "$STAGING" -type f \( -name '*.json' -o -name '*.jsonl' \) \
  -exec python3 "$STAGING/scripts/sanitize_release_data.py" \
    --replace-prefix "$ROOT=<project_root>" \
    --replace-prefix "$HOME=<home>" \
    --replace-prefix '/private/tmp=<tmp>' \
    --replace-prefix '/tmp=<tmp>' \
    --replace-prefix '/root/autodl-tmp=<remote_root>' \
    {} +

for path in "$TRAJECTORY_OUT"/data/*.jsonl; do
  gzip -9 "$path"
done

find "$STAGING" -type d \( \
  -name .git -o -name .codeagent -o -name .venv -o -name venv -o -name __pycache__ -o \
  -name .pytest_cache -o -name .mypy_cache -o -name '*.egg-info' -o \
  -name outputs -o -name dist -o -name runtime \
\) -prune -exec rm -rf {} +
find "$STAGING" -type f \( \
  -name '.env' -o -name '*.pyc' -o -name '*.log' -o -name '*.pid' -o \
  -name '*.dmg' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o \
  -name '.DS_Store' \
\) -delete

SECRET_PATTERN='(sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY|password[[:space:]]*=[[:space:]]*[^[:space:]#]{8,})'
if LC_ALL=C grep -R -nE "$SECRET_PATTERN" "$STAGING" --exclude='.env.example' >"$DIST_DIR/secret_scan.txt"; then
  echo "Potential secret material found; refusing to package:" >&2
  cat "$DIST_DIR/secret_scan.txt" >&2
  exit 2
fi
rm -f "$DIST_DIR/secret_scan.txt"

if LC_ALL=C grep -R -nF "$HOME" "$STAGING" >"$DIST_DIR/machine_path_scan.txt" || \
   LC_ALL=C grep -R -nE 'connect\.[A-Za-z0-9.-]*seetacloud\.com' "$STAGING" >>"$DIST_DIR/machine_path_scan.txt"; then
  echo "Machine-local paths or remote hosts found; refusing to package:" >&2
  cat "$DIST_DIR/machine_path_scan.txt" >&2
  exit 2
fi
rm -f "$DIST_DIR/machine_path_scan.txt"

if find "$STAGING" -type f -size +95M -print -quit | grep -q .; then
  echo "A staging file exceeds 95 MiB; use Git LFS or GitHub Release instead:" >&2
  find "$STAGING" -type f -size +95M -print >&2
  exit 2
fi

(
  cd "$STAGING"
  find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 shasum -a 256 > SHA256SUMS
)

if command -v xattr >/dev/null 2>&1; then
  xattr -cr "$STAGING"
fi
COPYFILE_DISABLE=1 tar --no-xattrs -C "$DIST_DIR" -czf "$ARCHIVE" "$PACKAGE_NAME"
(
  cd "$DIST_DIR"
  shasum -a 256 "$(basename "$ARCHIVE")" >"$(basename "$ARCHIVE").sha256"
)

echo "staging=$STAGING"
echo "archive=$ARCHIVE"
echo "checksum=$ARCHIVE.sha256"
du -sh "$STAGING" "$ARCHIVE"
