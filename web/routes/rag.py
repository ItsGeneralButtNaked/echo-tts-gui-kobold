"""
web/routes/rag.py — Retrieval-Augmented Generation routes.

Blueprint: rag_bp
Routes:
  GET  /rag/status
  POST /rag/load
  POST /rag/clear
  POST /rag/save
  GET  /rag/files
"""

import os
import re

from flask import Blueprint, jsonify, request

rag_bp = Blueprint("rag", __name__)


def _safe_rag_path(rag_dir: str, filename: str) -> str:
    """Resolve a filename inside rag_dir, rejecting any path traversal attempt."""
    safe = os.path.basename(filename)          # strip any directory component
    full = os.path.realpath(os.path.join(rag_dir, safe))
    root = os.path.realpath(rag_dir)
    if not full.startswith(root + os.sep):
        raise ValueError(f"Path traversal rejected: {filename!r}")
    return full

# Wired at startup
_get_rag     = None   # () -> RAGMemory
_get_session = None   # () -> Session
_rag_dir     = None   # str path


def wire(*, get_rag, get_session, rag_dir):
    global _get_rag, _get_session, _rag_dir
    _get_rag     = get_rag
    _get_session = get_session
    _rag_dir     = rag_dir


@rag_bp.route("/rag/status", methods=["GET"])
def rag_status():
    return jsonify(_get_rag().status)


@rag_bp.route("/rag/load", methods=["POST", "OPTIONS"])
def rag_load():
    if request.method == "OPTIONS":
        return "", 204
    rag     = _get_rag()
    SESSION = _get_session()
    data = request.get_json(force=True)

    filenames = data.get("filenames") or []
    if not filenames and data.get("filename"):
        filenames = [data["filename"].strip()]
    filenames = [f.strip() for f in filenames if f and f.strip()]
    if not filenames:
        return jsonify({"error": "no filename"}), 400

    os.makedirs(_rag_dir, exist_ok=True)
    paths = []
    for filename in filenames:
        try:
            full = _safe_rag_path(_rag_dir, filename)
        except ValueError:
            return jsonify({"error": f"Invalid filename: {filename}"}), 400
        if not os.path.exists(full):
            return jsonify({"error": f"File not found: {filename}"}), 404
        paths.append(full)

    try:
        semantic = data.get("semantic", False)
        if semantic and not rag._use_semantic:
            rag._use_semantic = True
        rag.load_multiple(paths)
        rag.enabled = True
        SESSION.tts.extra["rag_file"]     = ",".join(filenames)
        SESSION.tts.extra["rag_semantic"] = bool(rag._index is not None)
        SESSION.save_persistent()
        return jsonify({"ok": True, "chunks": len(rag.chunks), "filenames": filenames,
                        "semantic": rag._index is not None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rag_bp.route("/rag/clear", methods=["POST", "OPTIONS"])
def rag_clear():
    if request.method == "OPTIONS":
        return "", 204
    rag     = _get_rag()
    SESSION = _get_session()
    rag.clear()
    SESSION.tts.extra.pop("rag_file", None)
    SESSION.tts.extra.pop("rag_semantic", None)
    SESSION.save_persistent()
    return jsonify({"ok": True})


@rag_bp.route("/rag/save", methods=["POST", "OPTIONS"])
def rag_save():
    """Save the current conversation history as a plain-text RAG file."""
    if request.method == "OPTIONS":
        return "", 204
    SESSION = _get_session()
    data     = request.get_json(force=True)
    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "no filename"}), 400

    safe = re.sub(r"[^\w\-.]", "_", filename)
    if not safe.endswith(".txt"):
        safe += ".txt"

    os.makedirs(_rag_dir, exist_ok=True)
    path = os.path.join(_rag_dir, safe)

    lines = []
    for msg in SESSION.chat_history:
        role    = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
        lines.append(f"[{role}]\n{content}\n")

    if not lines:
        return jsonify({"error": "no conversation to save"}), 400
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[RAG] Saved conversation to '{path}' ({len(lines)} turns)")
        return jsonify({"ok": True, "filename": safe, "turns": len(lines)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@rag_bp.route("/rag/files", methods=["GET"])
def rag_files():
    os.makedirs(_rag_dir, exist_ok=True)
    files = sorted(
        f for f in os.listdir(_rag_dir)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(_rag_dir, f))
    )
    return jsonify(files)


@rag_bp.route("/rag/add", methods=["POST", "OPTIONS"])
def rag_add():
    if request.method == "OPTIONS":
        return "", 204
    rag     = _get_rag()
    SESSION = _get_session()
    data    = request.get_json(force=True)

    filenames = data.get("filenames") or []
    if not filenames and data.get("filename"):
        filenames = [data["filename"].strip()]
    filenames = [f.strip() for f in filenames if f and f.strip()]
    if not filenames:
        return jsonify({"error": "no filename"}), 400

    os.makedirs(_rag_dir, exist_ok=True)
    new_paths = []
    for filename in filenames:
        try:
            full = _safe_rag_path(_rag_dir, filename)
        except ValueError:
            return jsonify({"error": f"Invalid filename: {filename}"}), 400
        if not os.path.exists(full):
            return jsonify({"error": f"File not found: {filename}"}), 404
        new_paths.append(full)

    try:
        semantic = data.get("semantic", False)
        if semantic and not rag._use_semantic:
            rag._use_semantic = True

        existing_paths = list(getattr(rag, "_loaded_paths", []))
        all_paths = existing_paths + [p for p in new_paths if p not in existing_paths]

        rag.load_multiple(all_paths)
        rag.enabled = True

        existing_names = [f.strip() for f in SESSION.tts.extra.get("rag_file", "").split(",") if f.strip()]
        combined = existing_names + [fn for fn in filenames if fn not in existing_names]
        SESSION.tts.extra["rag_file"]     = ",".join(combined)
        SESSION.tts.extra["rag_semantic"] = bool(rag._index is not None)
        SESSION.save_persistent()
        return jsonify({"ok": True, "chunks": len(rag.chunks), "filenames": combined,
                        "semantic": rag._index is not None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /conv_rag ─────────────────────────────────────────────────────────────────
# Separate wire target for ConvRAG — set after rag_bp.wire() in ecko_web.py

_get_conv_rag = None   # () -> ConvRAG


def wire_conv_rag(*, get_conv_rag):
    global _get_conv_rag
    _get_conv_rag = get_conv_rag


@rag_bp.route("/conv_rag/status", methods=["GET"])
def conv_rag_status():
    return jsonify(_get_conv_rag().status())


@rag_bp.route("/conv_rag/set", methods=["POST", "OPTIONS"])
def conv_rag_set():
    """Enable/disable auto-flush and configure threshold."""
    if request.method == "OPTIONS":
        return "", 204
    conv  = _get_conv_rag()
    data  = request.get_json(force=True)
    if "enabled" in data:
        conv.enabled = bool(data["enabled"])
    if "threshold" in data:
        val = int(data["threshold"])
        conv.threshold = max(6, min(val, 200))   # clamp 6–200 messages
    SESSION = _get_session()
    SESSION.tts.extra["conv_rag_enabled"]   = conv.enabled
    SESSION.tts.extra["conv_rag_threshold"] = conv.threshold
    SESSION.save_persistent()
    return jsonify({"ok": True, **conv.status()})


@rag_bp.route("/conv_rag/clear", methods=["POST", "OPTIONS"])
def conv_rag_clear():
    """Delete the conversation RAG file for the current character."""
    if request.method == "OPTIONS":
        return "", 204
    _get_conv_rag().clear_file()
    return jsonify({"ok": True})
