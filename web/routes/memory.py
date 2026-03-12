"""
web/routes/memory.py — Structured memory store routes.

Blueprint: memory_bp
Routes:
  GET  /memory
  POST /memory/enable
  POST /memory/add
  POST /memory/update
  POST /memory/delete
  POST /memory/clear
  POST /memory/promote
  POST /memory/restore
  POST /memory/decay
  GET  /memory/export
  POST /memory/import   body: {mode: "replace"|"merge", entries:[...], archived:[...]}
"""

import json
import os
from datetime import datetime

from flask import Blueprint, jsonify, request

memory_bp = Blueprint("memory", __name__)

# Wired at startup
_get_memory  = None   # () -> MemoryStore
_get_session = None   # () -> Session
_memory_dir  = None   # str path — needed for backup file writes


def wire(*, get_memory, get_session, memory_dir=None):
    global _get_memory, _get_session, _memory_dir
    _get_memory  = get_memory
    _get_session = get_session
    _memory_dir  = memory_dir


@memory_bp.route("/memory", methods=["GET"])
def memory_viewer():
    _memory = _get_memory()
    _memory.recompute_scores()
    return jsonify(_memory.to_viewer_dict())


@memory_bp.route("/memory/enable", methods=["POST"])
def memory_enable():
    _memory = _get_memory()
    SESSION = _get_session()
    data = request.get_json(force=True)
    _memory.enabled = bool(data.get("enabled", True))
    _memory.save()
    SESSION.tts.extra["memory_enabled"] = _memory.enabled
    SESSION.save_persistent()
    return jsonify({"ok": True, "enabled": _memory.enabled})


@memory_bp.route("/memory/add", methods=["POST"])
def memory_add():
    _memory = _get_memory()
    data    = request.get_json(force=True)
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "no content"}), 400
    content = content[:2000]   # cap single-entry content length
    entry = _memory.add_entry(
        content  = content,
        category = data.get("category", "fact"),
        score    = float(data.get("score", 0.5)),
        global_  = bool(data.get("global", False)),
    )
    _memory.save()
    return jsonify({"ok": True, "entry": entry})


@memory_bp.route("/memory/update", methods=["POST"])
def memory_update():
    _memory = _get_memory()
    data    = request.get_json(force=True)
    mem_id  = data.pop("id", None)
    if not mem_id:
        return jsonify({"error": "no id"}), 400
    ok = _memory.update_entry(mem_id, **data)
    if ok:
        _memory.save()
    return jsonify({"ok": ok})


@memory_bp.route("/memory/delete", methods=["POST"])
def memory_delete():
    _memory = _get_memory()
    data    = request.get_json(force=True)
    mem_id  = data.get("id")
    if not mem_id:
        return jsonify({"error": "no id"}), 400
    ok = _memory.delete_entry(mem_id)
    if ok:
        _memory.save()
    return jsonify({"ok": ok})


@memory_bp.route("/memory/count", methods=["GET"])
def memory_count():
    mem = _get_memory()
    return jsonify({"count": len(mem.entries)})


@memory_bp.route("/memory/clear", methods=["POST"])
def memory_clear():
    mem = _get_memory()
    was_enabled = mem.enabled
    mem.clear_all()
    mem.enabled = was_enabled  # clear_all may reset enabled; restore it
    return jsonify({"ok": True, "cleared": True})


@memory_bp.route("/memory/promote", methods=["POST"])
def memory_promote():
    _memory = _get_memory()
    data    = request.get_json(force=True)
    mem_id  = data.get("id")
    if not mem_id:
        return jsonify({"ok": False})
    for e in _memory.entries:
        if e["id"] == mem_id:
            e["global"] = not e.get("global", False)
            _memory.save()
            return jsonify({"ok": True, "global": e["global"]})
    return jsonify({"ok": False})


@memory_bp.route("/memory/restore", methods=["POST"])
def memory_restore():
    _memory = _get_memory()
    data    = request.get_json(force=True)
    mem_id  = data.get("id")
    ok = _memory.restore_archived(mem_id) if mem_id else False
    if ok:
        _memory.save()
    return jsonify({"ok": ok})


@memory_bp.route("/memory/decay", methods=["POST"])
def memory_decay():
    _memory = _get_memory()
    _memory.decay_and_archive()
    _memory.save()
    return jsonify({"ok": True,
                    "active":   len(_memory.entries),
                    "archived": len(_memory.archived)})


# ── /memory/export ────────────────────────────────────────────────────────────

@memory_bp.route("/memory/export", methods=["GET"])
def memory_export():
    """Return the full memory bank as a downloadable JSON file."""
    _memory = _get_memory()
    _memory.recompute_scores()
    payload = {
        "character": _memory.character,
        "exported":  datetime.utcnow().isoformat(),
        "entries":   _memory.entries,
        "archived":  _memory.archived,
    }
    filename = f"{_memory.character}_memory_{datetime.utcnow().strftime('%Y%m%d')}.json"
    return (
        json.dumps(payload, indent=2),
        200,
        {
            "Content-Type":        "application/json",
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


# ── /memory/import ────────────────────────────────────────────────────────────

@memory_bp.route("/memory/import", methods=["POST", "OPTIONS"])
def memory_import():
    """
    Import a memory bank from a JSON payload.

    Body:
      {
        "mode":     "replace" | "merge",
        "entries":  [ ... ],
        "archived": [ ... ]           # optional, only used on replace
      }

    replace — backs up the current bank to
              memories/<character>_backup_<timestamp>.json
              then replaces entries + archived wholesale.

    merge   — appends incoming entries that don't already exist
              (dedup by exact content match, case-insensitive trim).
    """
    if request.method == "OPTIONS":
        return "", 204

    _memory = _get_memory()
    data    = request.get_json(force=True)

    mode              = data.get("mode", "merge")
    incoming_entries  = data.get("entries",  [])
    incoming_archived = data.get("archived", [])

    if not isinstance(incoming_entries, list):
        return jsonify({"error": "entries must be a list"}), 400

    # Guard against unbounded imports
    MAX_IMPORT_ENTRIES = 2000
    MAX_CONTENT_LEN    = 2000
    if len(incoming_entries) > MAX_IMPORT_ENTRIES:
        return jsonify({"error": f"Too many entries (max {MAX_IMPORT_ENTRIES})"}), 400

    def _sanitise_entry(e: dict) -> dict:
        return {**e, "content": str(e.get("content", ""))[:MAX_CONTENT_LEN]}

    incoming_entries  = [_sanitise_entry(e) for e in incoming_entries if isinstance(e, dict)]
    if isinstance(incoming_archived, list):
        incoming_archived = [_sanitise_entry(e) for e in incoming_archived if isinstance(e, dict)]

    if mode == "replace":
        backup_path = _write_backup(_memory)
        print(f"[MEMORY] Backup written to: {backup_path}")

        _memory.entries  = incoming_entries
        _memory.archived = incoming_archived if isinstance(incoming_archived, list) else []
        _memory.save()

        return jsonify({
            "ok":      True,
            "mode":    "replace",
            "active":  len(_memory.entries),
            "archived": len(_memory.archived),
            "backup":  os.path.basename(backup_path) if backup_path else None,
        })

    else:
        # Merge — append entries not already present (dedup by content)
        existing_contents = {
            e.get("content", "").strip().lower()
            for e in _memory.entries
        }

        added = 0
        for entry in incoming_entries:
            content_key = entry.get("content", "").strip().lower()
            if not content_key or content_key in existing_contents:
                continue
            _memory.entries.append(entry)
            existing_contents.add(content_key)
            added += 1

        if added:
            _memory.save()

        return jsonify({
            "ok":      True,
            "mode":    "merge",
            "added":   added,
            "skipped": len(incoming_entries) - added,
            "active":  len(_memory.entries),
        })


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_backup(_memory) -> str | None:
    """Serialise current memory bank to a timestamped backup file."""
    try:
        mem_dir = _memory_dir
        if not mem_dir:
            # Fall back to the directory the memory file lives in
            mem_dir = os.path.dirname(getattr(_memory, "_path", "") or "memories")
        os.makedirs(mem_dir, exist_ok=True)
        ts      = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path    = os.path.join(mem_dir, f"{_memory.character}_backup_{ts}.json")
        payload = {
            "character": _memory.character,
            "backed_up": datetime.utcnow().isoformat(),
            "entries":   _memory.entries,
            "archived":  _memory.archived,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path
    except Exception as e:
        print(f"[MEMORY] Backup write failed: {e}")
        return None
