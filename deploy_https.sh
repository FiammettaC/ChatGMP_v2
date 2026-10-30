#!/bin/bash
set -e

echo "Ensuring swap and overcommit are configured..."
if [ ! -f /swapfile ]; then
  sudo fallocate -l 4G /swapfile
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
  echo 'vm.overcommit_memory=1' | sudo tee -a /etc/sysctl.conf
  echo 1 | sudo tee /proc/sys/vm/overcommit_memory
  echo "Swap configured."
else
  echo "Swap already exists, skipping."
fi

echo "Pulling latest image..."
docker pull gcr.io/chatgmp/chatgmp:latest

echo "Stopping and removing existing container (if any)..."
docker stop chatgmp-app 2>/dev/null || true
docker rm chatgmp-app 2>/dev/null || true

echo "Running container with HTTPS on port 8443..."
mkdir -p /home/fiacac/chatgmp-data/audits
docker run -d \
  --name chatgmp-app \
  --runtime nvidia \
  -p 443:8443 \
  --restart always \
  -v /etc/letsencrypt/live/chatgmp.dk/fullchain.pem:/app/certs/cert.pem:ro \
  -v /etc/letsencrypt/live/chatgmp.dk/privkey.pem:/app/certs/key.pem:ro \
  -v /home/fiacac/chatgmp-data/audits:/app/data/audits \
  gcr.io/chatgmp/chatgmp:latest \
  gunicorn --bind 0.0.0.0:8443 --workers 1 --worker-class gevent --worker-connections 50 \
    --timeout 300 --graceful-timeout 120 \
    --certfile /app/certs/cert.pem --keyfile /app/certs/key.pem \
    --access-logfile - --error-logfile - main:app

echo "Container started successfully!"
docker ps
