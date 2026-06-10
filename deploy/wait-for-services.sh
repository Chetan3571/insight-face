#!/usr/bin/env bash
# Wait for Docker PostgreSQL and Redis (sources .env from project root).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DOCKER="$ROOT/deploy/docker.sh"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found in $ROOT"
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

chmod +x "$DOCKER"

echo "==> Starting PostgreSQL + Redis"
"$DOCKER" up -d db redis

echo "==> Waiting for database at ${DB_HOST:-localhost}:${DB_PORT:-5433}"
ready=false
for _ in $(seq 1 60); do
  if "$DOCKER" exec -T db pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "ERROR: Database not ready after 60s."
  echo "  Check: $DOCKER ps"
  echo "  Logs:  $DOCKER logs db --tail 50"
  exit 1
fi
echo "    Database ready."

echo "==> Waiting for Redis"
ready=false
for _ in $(seq 1 30); do
  if "$DOCKER" exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    ready=true
    break
  fi
  sleep 1
done
if [[ "$ready" != true ]]; then
  echo "ERROR: Redis not ready after 30s."
  echo "  Check: $DOCKER logs redis --tail 50"
  exit 1
fi
echo "    Redis ready."
