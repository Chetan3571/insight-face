#!/usr/bin/env bash
# Local dev: start Docker services and run migrations.
# Usage:  ./scripts/dev-up.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
DOCKER="$ROOT/deploy/docker.sh"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — set DB_PASSWORD, then run again."
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ "${DB_PORT}" == "5432" ]]; then
  echo "ERROR: DB_PORT=5432 points at system PostgreSQL (no pgvector)."
  echo "       Set DB_PORT=5433 in .env to use the Docker pgvector database."
  exit 1
fi

chmod +x "$DOCKER"

echo "==> Starting PostgreSQL (pgvector) + Redis"
if ! "$DOCKER" up -d db redis; then
  echo ""
  echo "Docker failed. Ensure Docker Desktop/daemon is running, then retry."
  exit 1
fi

echo "==> Waiting for database on ${DB_HOST}:${DB_PORT}"
for _ in $(seq 1 30); do
  if "$DOCKER" exec -T db pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! "$DOCKER" exec -T db pg_isready -U "${DB_USER}" -d "${DB_NAME}" >/dev/null 2>&1; then
  echo "Database did not become ready. Check: $DOCKER ps && $DOCKER logs db"
  exit 1
fi

echo "==> Waiting for Redis"
for _ in $(seq 1 30); do
  if "$DOCKER" exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    break
  fi
  sleep 1
done

if [[ ! -d env ]]; then
  echo "==> Creating Python venv"
  python3 -m venv env
fi
# shellcheck disable=SC1091
source env/bin/activate

echo "==> Running migrations"
python manage.py migrate

echo ""
echo "Done. Database: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME}"
