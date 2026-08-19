#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${CODEAGENT_DOCKER_IMAGE:-lottie-codeagent-runtime:latest}"
REPO="${CODEAGENT_DOCKER_REPO:-$PROJECT_ROOT}"

if [[ $# -eq 0 ]]; then
  echo 'usage: scripts/codeagent_docker.sh "你的任务"'
  echo "optional: CODEAGENT_DOCKER_REPO=/path/to/repo"
  exit 2
fi

env_args=()
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  env_args+=(--env-file "$PROJECT_ROOT/.env")
fi

docker run --rm -it \
  "${env_args[@]}" \
  -v "$REPO:/workspace/repo" \
  -w /workspace/repo \
  "$IMAGE" \
  --repo /workspace/repo \
  "$@"
