#!/usr/bin/env bash
# Wrapper: uses sudo when the current user cannot access Docker.
set -euo pipefail

if docker info >/dev/null 2>&1; then
  exec docker compose "$@"
fi

if sudo docker info >/dev/null 2>&1; then
  exec sudo docker compose "$@"
fi

echo "Docker is not running or not accessible. Try:"
echo "  sudo usermod -aG docker \$USER   # then log out and back in"
echo "  sudo systemctl start docker"
exit 1
