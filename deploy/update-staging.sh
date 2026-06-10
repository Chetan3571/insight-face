#!/usr/bin/env bash
# Redeploy after git pull on staging.
set -euo pipefail

APP_DIR="/home/ubuntu/repos/insight-face"
cd "$APP_DIR"

if [[ ! -f .env ]]; then
  echo "ERROR: .env not found. Copy .env.example to .env and configure it."
  exit 1
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

if [[ "${DB_PORT:-}" == "5432" ]]; then
  echo "ERROR: DB_PORT=5432 usually points at system PostgreSQL, not Docker pgvector."
  echo "       Set DB_PORT=5433 in .env"
  exit 1
fi

chmod +x deploy/docker.sh deploy/wait-for-services.sh
./deploy/wait-for-services.sh

if [[ ! -d env ]]; then
  python3 -m venv env
fi
# shellcheck disable=SC1091
source env/bin/activate

echo "==> Installing dependencies"
pip install -r requirements.txt

echo "==> Migrations (DB: ${DB_USER}@${DB_HOST}:${DB_PORT}/${DB_NAME})"
python manage.py migrate

echo "==> Collecting static files"
python manage.py collectstatic --noinput

mkdir -p logs media

echo "==> Restarting app processes"
sudo supervisorctl restart insight-face-gunicorn insight-face-celery || \
  sudo supervisorctl start insight-face-gunicorn insight-face-celery

echo ""
echo "Done. Verify:"
echo "  curl -s https://newmodel.snapdme.com/api/health/"
echo "  sudo supervisorctl status"
echo "  tail -f logs/django.log"
echo "  tail -f logs/celery.log"
