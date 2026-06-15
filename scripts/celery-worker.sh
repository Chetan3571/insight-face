#!/usr/bin/env bash
# Celery worker for local + staging (embeddings + default queues).
# Sources .env so CELERY_WORKER_CONCURRENCY is respected.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -x "$ROOT/env/bin/celery" ]]; then
  echo "ERROR: venv not found at $ROOT/env — run: python3 -m venv env && pip install -r requirements.txt"
  exit 1
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# CPU staging: use 1 (each worker loads ~500MB AdaFace models).
# GPU with enough VRAM/RAM: set CELERY_WORKER_CONCURRENCY=2 or 4 in .env
CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-1}"

exec "$ROOT/env/bin/celery" -A config worker \
  -Q embeddings,default \
  -l info \
  --concurrency="$CONCURRENCY" \
  --prefetch-multiplier=1
