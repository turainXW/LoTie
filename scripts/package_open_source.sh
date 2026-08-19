#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-$(date +%Y%m%d)}"
DIST_DIR="${DIST_DIR:-$ROOT/dist}"
PACKAGE_NAME="lottie-agent-open-source-$VERSION"
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
  src scripts benchmarks configs examples \
  pyproject.toml requirements.txt requirements-dev.txt \
  README.md README_zh.md DEPLOY_zh.md DEPLOY_EVAL90_zh.md \
  OPEN_SOURCE_CHECKLIST_zh.md \
  Dockerfile.runtime docker-compose.runtime.yml .env.example .gitignore; do
  copy_path "$path"
done

copy_path data/dataset_splits_v1
copy_path data/eval90
copy_path data/evalplus_local/tasks.jsonl
copy_path data/swesmith_local/tasks.jsonl

python3 "$STAGING/scripts/sanitize_release_data.py" \
  "$STAGING/data/evalplus_local/tasks.jsonl" \
  "$STAGING/data/swesmith_local/tasks.jsonl"

find "$STAGING" -type d \( \
  -name .git -o -name .codeagent -o -name .venv -o -name venv -o -name __pycache__ -o \
  -name .pytest_cache -o -name .mypy_cache -o -name '*.egg-info' -o \
  -name outputs -o -name dist \
\) -prune -exec rm -rf {} +
find "$STAGING" -type f \( \
  -name '.env' -o -name '*.pyc' -o -name '*.log' -o -name '*.pid' -o \
  -name '*.dmg' -o -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \
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
  echo "Machine-local paths found; refusing to package:" >&2
  cat "$DIST_DIR/machine_path_scan.txt" >&2
  exit 2
fi
rm -f "$DIST_DIR/machine_path_scan.txt"

if command -v xattr >/dev/null 2>&1; then
  xattr -cr "$STAGING"
fi
COPYFILE_DISABLE=1 tar --no-xattrs -C "$DIST_DIR" -czf "$ARCHIVE" "$PACKAGE_NAME"
(
  cd "$DIST_DIR"
  shasum -a 256 "$(basename "$ARCHIVE")" >"$(basename "$ARCHIVE").sha256"
)

echo "archive=$ARCHIVE"
echo "checksum=$ARCHIVE.sha256"
du -h "$ARCHIVE"
