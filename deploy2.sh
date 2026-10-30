#!/bin/bash
set -e
echo 'Getting GCP metadata token...'
TOKEN=$(curl -sf -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
echo "Token first 20: ${TOKEN:0:20}..."
echo "$TOKEN" | docker login -u oauth2accesstoken --password-stdin gcr.io
echo 'Pulling image...'
docker pull gcr.io/chatgmp/chatgmp:latest
echo 'Pull done! Starting container...'
docker rm -f chatgmp-app 2>/dev/null || true
docker run -d --name chatgmp-app --runtime nvidia -p 443:8443 --restart always \
  -v /etc/letsencrypt/live/chatgmp.dk/fullchain.pem:/app/certs/cert.pem:ro \
  -v /etc/letsencrypt/live/chatgmp.dk/privkey.pem:/app/certs/key.pem:ro \
  -v /home/fiacac/chatgmp-data/audits:/app/data/audits \
  -v /home/fiacac/chatgmp-data/instance:/app/instance \
  -v /home/fiacac/chatgmp-data/huggingface-cache:/root/.cache/huggingface \
  -v /home/fiacac/chatgmp-templates/templates:/app/templates \
  gcr.io/chatgmp/chatgmp:latest \
  gunicorn --bind 0.0.0.0:8443 --workers 1 --worker-class gevent --worker-connections 50 \
  --timeout 300 --graceful-timeout 120 \
  --max-requests 500 --max-requests-jitter 50 \
  --certfile /app/certs/cert.pem --keyfile /app/certs/key.pem \
  --access-logfile - --error-logfile - main:app
echo 'DEPLOY_COMPLETE'
