#!/usr/bin/env bash
# Redeploy after git pull on staging.
set -euo pipefail

APP_DIR="/home/ubuntu/repos/insight-face"
cd "$APP_DIR"

source env/bin/activate
pip install -r requirements.txt

python manage.py migrate
python manage.py collectstatic --noinput

sudo supervisorctl restart insight-face-gunicorn insight-face-celery

echo "Done."
