"""
ecko_web.py — Ecko web server entry point.

Wiring and services live in web/app.py.
This file handles paths, one-time migrations, and app.run().

Run:
    python ecko_web.py
    EXTRA_CERT_IPS=100.64.1.2 python ecko_web.py   # Tailscale IPs in cert
"""

import os
import shutil
import socket
import threading

from core.logger import log, install_print_hook
from web.app     import create_app

install_print_hook()   # capture all print() calls to logs/ecko.log

# ── Paths ─────────────────────────────────────────────────────────────────────

WEB_PORT  = 5050
_HERE     = os.path.dirname(os.path.abspath(__file__))

CERT_DIR  = os.path.join(_HERE, "ssl")
CERT_FILE = os.path.join(CERT_DIR, "cert.pem")
KEY_FILE  = os.path.join(CERT_DIR, "key.pem")

RAG_DIR       = os.path.join(_HERE, "rag")
RAG_EXTRA_DIR = os.path.join(RAG_DIR, "extra")
RAG_CONV_DIR  = os.path.join(RAG_DIR, "conversations")
SAFETY_DIR    = os.path.join(_HERE, "safety")
SESSION_FILE  = os.path.join(_HERE, "ecko_session.json")
MEMORY_DIR    = os.path.join(_HERE, "memories")
ASCII_ART_DIR = os.path.join(_HERE, "ascii_art")


# ── One-time migration: move existing RAG files into subdirs ──────────────────

def _migrate_rag_dirs():
    os.makedirs(RAG_EXTRA_DIR, exist_ok=True)
    os.makedirs(RAG_CONV_DIR,  exist_ok=True)
    for fname in os.listdir(RAG_DIR):
        src = os.path.join(RAG_DIR, fname)
        if not os.path.isfile(src) or not fname.lower().endswith(".txt"):
            continue
        dst = os.path.join(
            RAG_CONV_DIR  if fname.endswith("_conversation.txt") else RAG_EXTRA_DIR,
            fname,
        )
        if not os.path.exists(dst):
            shutil.move(src, dst)
            sub = "conversations" if fname.endswith("_conversation.txt") else "extra"
            print(f"[RAG] Migrated {fname} → {sub}/")

_migrate_rag_dirs()


# ── Build app ─────────────────────────────────────────────────────────────────

app, _services = create_app({
    "session_file":   SESSION_FILE,
    "rag_extra_dir":  RAG_EXTRA_DIR,
    "rag_conv_dir":   RAG_CONV_DIR,
    "safety_dir":     SAFETY_DIR,
    "memory_dir":     MEMORY_DIR,
    "ascii_art_dir":  ASCII_ART_DIR,
    "guest_cfg_path": os.path.join(_HERE, "guest_config.json"),
})


# ── Startup ───────────────────────────────────────────────────────────────────

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        log.debug("[STARTUP] _get_local_ip() fell back to 127.0.0.1: %s", e)
        return "127.0.0.1"


if __name__ == "__main__":
    from web.ssl_cert import ensure_ssl_cert
    ensure_ssl_cert(CERT_DIR, CERT_FILE, KEY_FILE)

    local_ip = _get_local_ip()
    print()
    print("╔══════════════════════════════════════════╗")
    print("║           ECKO  WEB  SERVER               ║")
    print("╠══════════════════════════════════════════╣")
    print(f"║  Local:   https://localhost:{WEB_PORT}         ║")
    print(f"║  Phone:   https://{local_ip}:{WEB_PORT}  ║")
    print("║                                          ║")
    print("║  Android: Advanced → Proceed past cert   ║")
    print("╚══════════════════════════════════════════╝")
    print()

    from core.stt import get_whisper
    threading.Thread(target=get_whisper, daemon=True).start()

    app.run(
        host="0.0.0.0",
        port=WEB_PORT,
        ssl_context=(CERT_FILE, KEY_FILE),
        threaded=True,
        debug=False,
    )
