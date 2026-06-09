#!/usr/bin/env bash
# Full project health check (Docker + Django).
set -euo pipefail

cd "$(cd "$(dirname "$0")/.." && pwd)"

FULL=false
SKIP_DOCKER=false
for arg in "$@"; do
  case "$arg" in
    --full) FULL=true ;;
    --skip-docker) SKIP_DOCKER=true ;;
  esac
done

echo "=== Docker services ==="
if [[ "$SKIP_DOCKER" == false ]] && [[ -f deploy/docker.sh ]]; then
  ./deploy/docker.sh ps || true
  echo ""
fi

echo "=== Django project check ==="
source env/bin/activate
ARGS=()
[[ "$FULL" == true ]] && ARGS+=(--full)
[[ "$SKIP_DOCKER" == true ]] && ARGS+=(--skip-docker)
python manage.py check_project "${ARGS[@]}"
