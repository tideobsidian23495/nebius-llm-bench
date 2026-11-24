#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-nebius-llm-bench}"
PORT="${PORT:-8080}"

docker build -t "$IMAGE_NAME" .
docker run --rm --gpus all \
  -p "$PORT:8080" \
  -v "$(pwd)/results:/app/results" \
  "$IMAGE_NAME" "$@"
