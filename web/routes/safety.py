"""
web/routes/safety.py — Safety layer management routes.

Blueprint: safety_bp
Routes:
  GET  /safety/status
  GET  /safety/rules
  POST /safety/rules
  POST /safety/reset
  POST /safety/defaults
  POST /safety/settings
  POST /safety/clear-flags
"""

from flask import Blueprint, jsonify, request

safety_bp = Blueprint("safety", __name__)

_get_safety  = None   # () -> SafetyLayer  — wired at startup
_get_session = None   # () -> Session      — wired at startup


def wire(*, get_safety, get_session):
    global _get_safety, _get_session
    _get_safety  = get_safety
    _get_session = get_session


@safety_bp.route("/safety/status", methods=["GET"])
def safety_status():
    return jsonify(_get_safety().status())


@safety_bp.route("/safety/rules", methods=["GET"])
def safety_get_rules():
    return jsonify(_get_safety().get_rules())


@safety_bp.route("/safety/rules", methods=["POST", "OPTIONS"])
def safety_set_rules():
    if request.method == "OPTIONS":
        return "", 204
    rules = request.get_json(force=True)
    if not isinstance(rules, list):
        return jsonify({"error": "expected array"}), 400
    _get_safety().set_rules(rules)
    return jsonify({"ok": True, "count": len(rules)})


@safety_bp.route("/safety/reset", methods=["POST", "OPTIONS"])
def safety_reset_score():
    if request.method == "OPTIONS":
        return "", 204
    _get_safety().reset_score()
    return jsonify({"ok": True})


@safety_bp.route("/safety/defaults", methods=["POST", "OPTIONS"])
def safety_reset_defaults():
    if request.method == "OPTIONS":
        return "", 204
    _get_safety().reset_to_defaults()
    return jsonify({"ok": True, "count": len(_get_safety().get_rules())})


@safety_bp.route("/safety/settings", methods=["POST", "OPTIONS"])
def safety_settings():
    if request.method == "OPTIONS":
        return "", 204
    data   = request.get_json(force=True)
    safety = _get_safety()
    session = _get_session()
    changed = False
    if "layer1_enabled" in data:
        safety.layer1_enabled = bool(data["layer1_enabled"])
        session.tts.extra["safety_layer1_enabled"] = safety.layer1_enabled
        changed = True
    if "layer2_enabled" in data:
        safety.layer2_enabled = bool(data["layer2_enabled"])
        session.tts.extra["safety_layer2_enabled"] = safety.layer2_enabled
        changed = True
    if changed:
        session.save_persistent()
    return jsonify({"ok": True})


@safety_bp.route("/safety/clear-flags", methods=["POST", "OPTIONS"])
def safety_clear_flags():
    if request.method == "OPTIONS":
        return "", 204
    _get_safety().clear_flags()
    return jsonify({"ok": True})
