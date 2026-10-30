#!/bin/bash
set -e

# Generate self-signed SSL certificate before starting Gunicorn
python - <<'EOF'
import os, datetime
from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa

cert_dir = '/app/certs'
cert_file = os.path.join(cert_dir, 'cert.pem')
key_file = os.path.join(cert_dir, 'key.pem')

if not (os.path.exists(cert_file) and os.path.exists(key_file)):
    os.makedirs(cert_dir, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, u"chatgmp.dk")])
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject).issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.utcnow())
        .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
        .sign(private_key, hashes.SHA256())
    )
    with open(cert_file, 'wb') as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_file, 'wb') as f:
        f.write(private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption()
        ))
    print("SSL certificate generated.")
else:
    print("SSL certificate already exists.")
EOF

exec gunicorn \
    --bind 0.0.0.0:8443 \
    --workers 1 \
    --worker-class gevent \
    --worker-connections 50 \
    --timeout 300 \
    --graceful-timeout 120 \
    --max-requests 500 \
    --max-requests-jitter 50 \
    --certfile /app/certs/cert.pem \
    --keyfile /app/certs/key.pem \
    --access-logfile - \
    --error-logfile - \
    main:app
