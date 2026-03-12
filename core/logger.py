"""
core/logger.py — Centralised logging for Ecko.

Provides a single shared logger that writes to both console and a rotating
log file.  All existing [BRACKET] prefixes in print() calls are preserved
as-is — this module is purely additive.

Usage:
    from core.logger import log
    log.info("[RAG] Loaded 42 chunks")
    log.warning("[SAFETY] Score threshold reached")
    log.error("[LLM] API call failed: %s", err)

The log file path defaults to  logs/ecko.log  relative to the project root.
Override by setting the ECKO_LOG_FILE environment variable before import.

File rotation: 5 MB per file, 5 backups kept → max ~25 MB on disk.
Log level:     DEBUG to file, INFO to console (set ECKO_LOG_LEVEL to override).
"""

import logging
import logging.handlers
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
_LOG_DIR    = os.path.join(_HERE, "logs")
_LOG_FILE   = os.environ.get("ECKO_LOG_FILE", os.path.join(_LOG_DIR, "ecko.log"))
_LOG_LEVEL  = os.environ.get("ECKO_LOG_LEVEL", "DEBUG").upper()

_FILE_MAX_BYTES = 5 * 1024 * 1024   # 5 MB
_FILE_BACKUPS   = 5                  # keep ecko.log.1 … ecko.log.5

# ─────────────────────────────────────────────────────────────────────────────
# FORMATTER
# ─────────────────────────────────────────────────────────────────────────────

_FILE_FMT    = "%(asctime)s  %(levelname)-8s  %(message)s"
_CONSOLE_FMT = "%(message)s"          # keep console output identical to old print() style
_DATE_FMT    = "%Y-%m-%d %H:%M:%S"

# ─────────────────────────────────────────────────────────────────────────────
# BUILD LOGGER  (runs once on first import)
# ─────────────────────────────────────────────────────────────────────────────

def _build_logger() -> logging.Logger:
    logger = logging.getLogger("ecko")
    if logger.handlers:
        return logger   # already initialised (e.g. reloaded module)

    level = getattr(logging, _LOG_LEVEL, logging.DEBUG)
    logger.setLevel(logging.DEBUG)   # accept everything; handlers filter independently

    # ── file handler (rotating) ───────────────────────────────────────────────
    try:
        os.makedirs(_LOG_DIR, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            _LOG_FILE,
            maxBytes=_FILE_MAX_BYTES,
            backupCount=_FILE_BACKUPS,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(_FILE_FMT, datefmt=_DATE_FMT))
        logger.addHandler(fh)
    except Exception as e:
        print(f"[LOGGER] Could not open log file {_LOG_FILE!r}: {e}", file=sys.stderr)

    # ── console handler ───────────────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(_CONSOLE_FMT))
    logger.addHandler(ch)

    # Prevent log records from propagating to the root logger (avoids duplicates
    # if Flask/Werkzeug has also configured the root logger).
    logger.propagate = False

    logger.info(
        "[LOGGER] Logging started — file: %s  level: %s",
        _LOG_FILE, _LOG_LEVEL,
    )
    return logger


log: logging.Logger = _build_logger()


# ─────────────────────────────────────────────────────────────────────────────
# PRINT REDIRECTOR  (optional — call install_print_hook() to capture legacy prints)
# ─────────────────────────────────────────────────────────────────────────────

class _PrintToLog:
    """
    Redirect sys.stdout so that existing print() calls are captured by the
    logger and therefore also written to the log file.

    Call install_print_hook() once at startup to activate.
    The original stdout is preserved for the console handler above.
    """

    def __init__(self, original_stdout):
        self._orig = original_stdout

    def write(self, text: str):
        text = text.rstrip("\n")
        if text:
            log.info(text)
        # Also write to original stdout so the console handler still fires
        # via the logging StreamHandler — no double-print needed.

    def flush(self):
        self._orig.flush()

    def isatty(self):
        return False


_hook_installed = False

def install_print_hook():
    """
    Redirect all print() output through the logger so it appears in the log
    file.  Safe to call multiple times — only installs once.

    Call this once near the top of ecko_web.py after the logger is imported.
    """
    global _hook_installed
    if _hook_installed:
        return
    sys.stdout = _PrintToLog(sys.__stdout__)
    _hook_installed = True
    log.info("[LOGGER] print() hook installed — all stdout captured to log file")
