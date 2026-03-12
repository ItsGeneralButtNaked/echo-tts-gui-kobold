"""
web/routes/fx.py — Visual effects trigger endpoint for Ecko.

Blueprint: fx_bp
Routes:
  POST /fx/trigger   — Broadcast a named visual effect to all connected clients
  GET  /fx/list      — Return list of available effects
"""

from flask import Blueprint, jsonify, request
from web.fx import fx_payload, EFFECTS, random_effect

fx_bp = Blueprint("fx", __name__)

_push_raw_payload = None  # wired at startup


def wire(push_fn):
    global _push_raw_payload
    _push_raw_payload = push_fn


@fx_bp.route("/fx/trigger", methods=["POST"])
def trigger_fx():
    data = request.get_json(silent=True) or {}
    effect = data.get("effect", "random")
    duration_ms = int(data.get("duration_ms", 0))

    if effect == "random":
        mood = data.get("mood", "random")
        effect = random_effect(mood)

    if effect not in EFFECTS:
        return jsonify({"error": f"Unknown effect: {effect}"}), 400

    if _push_raw_payload:
        _push_raw_payload(fx_payload(effect, duration_ms))

    return jsonify({"ok": True, "effect": effect})


@fx_bp.route("/fx/list", methods=["GET"])
def list_fx():
    return jsonify({"effects": EFFECTS})
