#!/usr/bin/env bash
# Run on staging:  ./deploy/troubleshoot-502.sh
set -euo pipefail

APP=/home/ubuntu/repos/insight-face
cd "$APP"

echo "=== 1. Supervisor ==="
sudo supervisorctl status insight-face-gunicorn insight-face-celery || true

echo ""
echo "=== 2. Port 8010 listening? ==="
ss -tlnp | grep 8010 || echo "NOTHING listening on 8010 — gunicorn is down"

echo ""
echo "=== 3. Curl Gunicorn directly (bypass nginx) ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:8010/api/health/ || echo "curl failed"

echo ""
echo "=== 4. Curl via nginx ==="
curl -s -o /dev/null -w "HTTP %{http_code}\n" -H "Host: newmodel.snapdme.com" http://127.0.0.1/api/health/ || echo "curl failed"

echo ""
echo "=== 5. Gunicorn log (last 20 lines) ==="
tail -20 "$APP/logs/gunicorn-error.log" 2>/dev/null || \
  sudo tail -20 /var/log/supervisor/insight-face-gunicorn.log 2>/dev/null || \
  echo "no gunicorn log found"

echo ""
echo "=== 6. Nginx error log ==="
sudo tail -10 /var/log/nginx/error.log 2>/dev/null || echo "no nginx error log"

echo ""
echo "=== 7. Memory ==="
free -h
