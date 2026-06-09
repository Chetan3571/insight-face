#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/ubuntu/repos/insight-face"
cd "$APP_DIR"
DOCKER="./deploy/docker.sh"

echo "==> Checking .env"
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example — edit it before continuing."
  exit 1
fi

set -a
source .env
set +a

chmod +x "$DOCKER"

echo "==> Starting PostgreSQL + Redis via Docker"
$DOCKER up -d db redis

echo "==> Waiting for database"
for i in $(seq 1 30); do
  if $DOCKER exec -T db pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

echo "==> Waiting for Redis"
for i in $(seq 1 30); do
  if $DOCKER exec -T redis redis-cli ping 2>/dev/null | grep -q PONG; then
    break
  fi
  sleep 1
done

echo "==> Python venv + dependencies"
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Django migrations"
python manage.py migrate

echo "==> Supervisor"
sudo cp deploy/supervisor/insight-face.conf /etc/supervisor/conf.d/
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart insight-face-gunicorn insight-face-celery || \
  sudo supervisorctl start insight-face-gunicorn insight-face-celery

echo "==> Nginx"
sudo cp deploy/nginx/newmodel.snapdme.com.conf /etc/nginx/sites-available/insight-face
sudo ln -sf /etc/nginx/sites-available/insight-face /etc/nginx/sites-enabled/insight-face
sudo nginx -t
sudo systemctl reload nginx

echo ""
echo "Done. Verify:"
echo "  curl -I http://newmodel.snapdme.com/api/upload/"
echo "  sudo supervisorctl status"
echo "  $DOCKER ps"
echo "  $DOCKER exec redis redis-cli ping"
