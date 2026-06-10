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

chmod +x deploy/wait-for-services.sh
./deploy/wait-for-services.sh

echo "==> Python venv + dependencies"
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Django migrations"
python manage.py migrate

echo "==> Collecting static files into staticfiles/"
python manage.py collectstatic --noinput

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
echo "  tail -f logs/django.log"
echo "  curl -I http://newmodel.snapdme.com/static/admin/css/base.css"
echo "  curl -I http://newmodel.snapdme.com/api/upload/"
echo "  sudo supervisorctl status"
echo "  $DOCKER ps"
echo "  $DOCKER exec redis redis-cli ping"
