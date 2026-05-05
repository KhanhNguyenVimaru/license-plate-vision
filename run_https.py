"""Run Flask app with self-signed HTTPS certificate.

This allows camera access on mobile browsers which require HTTPS for getUserMedia.
"""

import os
import sys

from werkzeug.serving import make_ssl_devcert

# Create self-signed cert if not exists
cert_path = os.path.join(os.path.dirname(__file__), "cert")
if not os.path.exists(f"{cert_path}.key"):
    print("Generating self-signed SSL certificate for HTTPS...")
    make_ssl_devcert(cert_path, host="localhost")
    print(f"Certificate created: {cert_path}.crt")

# Import and run app
from app import app

if __name__ == "__main__":
    print("=" * 50)
    print("Starting HTTPS server on https://0.0.0.0:5000")
    print("=" * 50)
    print("\nAccess URLs:")
    print("  Local:    https://localhost:5000")
    print("  Network:  https://<your-ip>:5000")
    print("\nNOTE: Browser will show a security warning because")
    print("this uses a self-signed certificate. Click 'Advanced'")
    print("and then 'Proceed' to continue.")
    print("=" * 50 + "\n")

    app.run(
        host="0.0.0.0",
        port=5000,
        ssl_context=(f"{cert_path}.crt", f"{cert_path}.key"),
        debug=False,
    )
