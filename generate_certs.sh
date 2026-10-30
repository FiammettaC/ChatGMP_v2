#!/bin/bash
mkdir -p ~/certs
cd ~/certs
openssl req -x509 -newkey rsa:4096 -nodes -out cert.pem -keyout key.pem -days 365 -subj "/CN=34.61.233.89"
echo "Certificates generated successfully"
ls -la
