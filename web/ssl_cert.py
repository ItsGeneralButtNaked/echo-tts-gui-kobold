"""
web/ssl_cert.py — Self-signed TLS certificate generation for Ecko.

Writes cert.pem / key.pem into <project>/ssl/.
Set EXTRA_CERT_IPS env var (comma-separated) to include Tailscale / LAN IPs.
"""

import datetime
import ipaddress
import os
import socket
import sys

from core.logger import log


def ensure_ssl_cert(cert_dir: str, cert_file: str, key_file: str) -> None:
    """Generate a self-signed cert if one doesn't already exist."""
    os.makedirs(cert_dir, exist_ok=True)
    if os.path.exists(cert_file) and os.path.exists(key_file):
        return

    print("[SSL] Generating self-signed certificate…")
    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.x509 import DNSName, IPAddress, SubjectAlternativeName
        from cryptography.x509.oid import NameOID

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except Exception as e:
            log.debug("[SSL] local IP probe fell back to 127.0.0.1: %s", e)
            local_ip = "127.0.0.1"

        extra_ips = []
        for raw in os.environ.get("EXTRA_CERT_IPS", "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                extra_ips.append(ipaddress.IPv4Address(raw))
            except ValueError:
                print(f"[SSL] Ignoring invalid IP: {raw!r}")

        all_ips = list(dict.fromkeys([
            ipaddress.IPv4Address("127.0.0.1"),
            ipaddress.IPv4Address(local_ip),
            *extra_ips,
        ]))
        san_entries = [DNSName("localhost")] + [IPAddress(ip) for ip in all_ips]
        subject = issuer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, local_ip)])
        san = SubjectAlternativeName(san_entries)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject).issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
            .add_extension(san, critical=False)
            .sign(key, hashes.SHA256())
        )
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            ))
        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        extra_str = ", ".join(str(ip) for ip in extra_ips)
        print(f"[SSL] Certificate written — local: {local_ip}"
              + (f", extra: {extra_str}" if extra_str else ""))

    except ImportError:
        print("[SSL] ERROR: 'cryptography' not installed. Run: pip install cryptography")
        sys.exit(1)
