"""
web/app.py — Ecko Flask application factory.

Constructs and wires the Flask app: services, blueprints, CORS, and the
initiative/session broadcast patches. ecko_web.py calls create_app() and
then runs the returned app.

Keeping wiring here (not in ecko_web.py) means the entry point stays thin
and the factory can be imported by tests or alternative entry points without
side-effects.
"""

import json
import logging
import os

from flask import Flask, Response, jsonify, request as flask_request

import core.characters as chars_mod
from core.context_mode import MODES, DEFAULT_MODE, get_mode as _get_mode_by_name
from core.llm          import LLMCaller
from core.memory       import MemoryStore
from core.rag          import RAGMemory
from core.safety       import SafetyLayer
from core.session      import Session
from core.tts          import TTS_PROVIDER_REGISTRY  # noqa: imported for route wiring

from web.conv_rag        import ConvRAG
from web.initiative      import Initiative, SW_JS
from web.session_restore import restore_session_state


# ── Silence noisy-but-harmless Werkzeug broken-pipe / SSL EOF errors ──────────
_wz_log = logging.getLogger("werkzeug")
_wz_orig_error = _wz_log.error

def _wz_filtered_error(msg, *args, **kwargs):
    text = str(msg % args) if args else str(msg)
    if "BrokenPipeError" in text or "UNEXPECTED_EOF" in text or "Broken pipe" in text:
        return
    _wz_orig_error(msg, *args, **kwargs)

_wz_log.error = _wz_filtered_error


def create_app(config: dict) -> tuple[Flask, dict]:
    """
    Build and return (flask_app, services).

    config keys:
        session_file   : path to ecko_session.json
        rag_extra_dir  : path to rag/extra/
        rag_conv_dir   : path to rag/conversations/
        safety_dir     : path to safety/
        memory_dir     : path to memories/
        guest_cfg_path : path to guest_config.json (optional)

    services dict contains all runtime singletons so ecko_web.py can pass
    them to app.run() helpers (e.g. for Whisper pre-warm).
    """
    chars_mod.SURFACE = "web"

    # ── Core services ─────────────────────────────────────────────────────────
    session = Session(session_file=config["session_file"])
    rag     = RAGMemory()

    memory = MemoryStore(character="global", memory_dir=config["memory_dir"])
    memory.load()

    safety = SafetyLayer(safety_dir=config["safety_dir"])
    safety.load_score("global")
    safety.memory_hook = memory.add_entry

    # ── Context mode ──────────────────────────────────────────────────────────
    _active_mode_name = [session.tts.extra.get("context_mode", DEFAULT_MODE)]
    if _active_mode_name[0] not in MODES:
        _active_mode_name[0] = DEFAULT_MODE

    def get_active_mode():
        return _get_mode_by_name(_active_mode_name[0])

    def set_active_mode(name: str):
        if name not in MODES:
            print(f"[CONTEXT MODE] Unknown mode {name!r}, ignoring")
            return
        _active_mode_name[0] = name
        session.tts.extra["context_mode"] = name
        session.save_persistent()
        print(f"[CONTEXT MODE] Switched to {name!r} "
              f"({MODES[name].emoji} {MODES[name].label})")

    # ── Injection chain: RAG → memory → LLM ──────────────────────────────────
    def _base_inject(system_prompt: str, user_text: str) -> str:
        return rag.inject(system_prompt, user_text, mode=get_active_mode())

    session.llm.memory_inject_fn = memory.make_inject_fn(
        upstream_fn=_base_inject,
        get_mode=get_active_mode,
    )

    # ── Isolated-session support ───────────────────────────────────────────────
    session_mode   = ["shared"]         # mutable ref
    isolated_llm: dict[str, LLMCaller] = {}

    def get_client_llm():
        ip = flask_request.remote_addr or "unknown"
        if ip not in isolated_llm:
            c = LLMCaller()
            c.from_dict(session.llm.to_dict())
            c.memory_inject_fn = session.llm.memory_inject_fn
            isolated_llm[ip] = c
        return isolated_llm[ip]

    # ── Conversation RAG ──────────────────────────────────────────────────────
    conv_rag = ConvRAG(rag_dir=config["rag_conv_dir"])
    conv_rag.enabled   = bool(session.tts.extra.get("conv_rag_enabled",   False))
    conv_rag.threshold = int(session.tts.extra.get("conv_rag_threshold",  ConvRAG.DEFAULT_THRESHOLD))

    # ── Initiative engine ─────────────────────────────────────────────────────
    initiative = Initiative()
    initiative.wire(
        get_busy     = lambda: session.busy,
        set_busy     = lambda v: setattr(session, "busy", v),
        get_context  = lambda: (session.chat_history, memory, session.llm),
        save_fn      = lambda: session.save_persistent() if session_mode[0] != "isolated" else None,
        broadcast_fn = None,    # patched below after chat routes are imported
        strip_fn     = None,
    )

    # ── Session state restore ─────────────────────────────────────────────────
    # Inline setter used only during restore — replaces the global memory ref
    # and re-wires the inject chain so the restored character's bank is live
    # before any request arrives.
    def _restore_set_memory(new_mem):
        nonlocal memory
        memory = new_mem
        safety.memory_hook = memory.add_entry
        session.llm.memory_inject_fn = memory.make_inject_fn(
            upstream_fn=_base_inject,
            get_mode=get_active_mode,
        )

    restore_session_state(
        session             = session,
        memory              = memory,
        safety              = safety,
        initiative          = initiative,
        rag                 = rag,
        rag_extra_dir       = config["rag_extra_dir"],
        memory_dir          = config["memory_dir"],
        active_mode_name_ref = _active_mode_name,
        set_memory          = _restore_set_memory,
    )

    # ── Guest / kiosk mode ────────────────────────────────────────────────────
    guest_mode   = False
    guest_config = {}
    gcfg_path    = config.get("guest_cfg_path", "")
    if gcfg_path and os.path.exists(gcfg_path):
        try:
            with open(gcfg_path, encoding="utf-8") as _f:
                guest_config = json.load(_f)
            guest_mode = True
            print(f"[ECKO] Guest mode active — config: {guest_config}")
        except Exception as _ge:
            print(f"[ECKO] guest_config.json load error: {_ge}")

    # ── Flask app ─────────────────────────────────────────────────────────────
    app = Flask(__name__)

    # Allowed CORS origins — localhost only by default.
    # For LAN / remote access, add explicit origins here rather than using wildcard.
    _CORS_ALLOWED = {
        "http://localhost", "http://127.0.0.1",
        "https://localhost", "https://127.0.0.1",
    }

    @app.after_request
    def add_cors(response):
        origin = flask_request.headers.get("Origin", "")
        # Allow any localhost port (e.g. http://localhost:5000)
        if origin in _CORS_ALLOWED or (
            origin.startswith(("http://localhost:", "http://127.0.0.1:",
                               "https://localhost:", "https://127.0.0.1:"))
        ):
            response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

    @app.route("/sw.js")
    def service_worker():
        return Response(SW_JS, mimetype="application/javascript",
                        headers={"Service-Worker-Allowed": "/"})

    # ── Initiative routes ─────────────────────────────────────────────────────
    @app.route("/initiative/status")
    def initiative_status():
        return jsonify(initiative.status())

    @app.route("/initiative/set", methods=["POST"])
    def initiative_set():
        data    = flask_request.get_json(force=True)
        enabled = data.get("enabled", False)
        mode    = data.get("mode", "light")
        if enabled:
            initiative.start(mode)
        else:
            initiative.stop()
        return jsonify(initiative.status())

    @app.route("/initiative/reschedule", methods=["POST"])
    def initiative_reschedule():
        initiative.reschedule()
        return "", 204

    # ── Blueprints ────────────────────────────────────────────────────────────
    import web.routes.chat as _chat_mod
    from web.sanitise  import strip_leaked_context
    from web.frontend  import FRONTEND_HTML

    _chat_mod.wire(
        get_session       = lambda: session,
        get_safety        = lambda: safety,
        get_memory        = lambda: memory,
        get_client_llm    = get_client_llm,
        get_session_mode  = lambda: session_mode[0],
        get_initiative    = lambda: initiative,
        get_conv_rag      = lambda: conv_rag,
        get_rag           = lambda: rag,
        get_frontend_html = lambda: FRONTEND_HTML,
    )
    app.register_blueprint(_chat_mod.chat_bp)

    # Patch initiative broadcast now that push_raw_payload is available
    initiative._broadcast_fn = _chat_mod.push_raw_payload
    initiative._strip_fn     = strip_leaked_context

    import web.routes.safety as _safety_mod
    _safety_mod.wire(get_safety=lambda: safety, get_session=lambda: session)
    app.register_blueprint(_safety_mod.safety_bp)

    import web.routes.settings as _settings_mod
    _settings_mod.wire(
        get_session      = lambda: session,
        get_rag          = lambda: rag,
        get_memory       = lambda: memory,
        get_safety       = lambda: safety,
        get_initiative   = lambda: initiative,
        get_conv_rag     = lambda: conv_rag,
        get_session_mode = lambda: session_mode[0],
        set_session_mode = lambda v: session_mode.__setitem__(0, v),
        get_isolated_llm = lambda: isolated_llm,
        get_chars_mod    = lambda: chars_mod,
        get_guest_config = lambda: (guest_mode, guest_config),
        get_active_mode  = get_active_mode,
        set_active_mode  = set_active_mode,
    )
    app.register_blueprint(_settings_mod.settings_bp)

    import web.routes.rag as _rag_mod
    _rag_mod.wire(
        get_rag     = lambda: rag,
        get_session = lambda: session,
        rag_dir     = config["rag_extra_dir"],
    )
    app.register_blueprint(_rag_mod.rag_bp)
    _rag_mod.wire_conv_rag(get_conv_rag=lambda: conv_rag)

    import web.routes.memory as _memory_mod
    _memory_mod.wire(
        get_memory  = lambda: memory,
        get_session = lambda: session,
        memory_dir  = config["memory_dir"],
    )
    app.register_blueprint(_memory_mod.memory_bp)

    import web.routes.characters as _chars_mod

    def _set_memory(new_mem: MemoryStore):
        nonlocal memory
        memory = new_mem
        safety.memory_hook = memory.add_entry
        _memory_mod.wire(get_memory=lambda: memory, get_session=lambda: session)

    _chars_mod.wire(
        get_session     = lambda: session,
        get_rag         = lambda: rag,
        get_safety      = lambda: safety,
        get_memory      = lambda: memory,
        set_memory      = _set_memory,
        get_initiative  = lambda: initiative,
        get_chars_mod   = lambda: chars_mod,
        get_base_inject = lambda: _base_inject,
        get_conv_rag    = lambda: conv_rag,
        get_mode        = get_active_mode,
        memory_dir      = config["memory_dir"],
        rag_dir         = config["rag_extra_dir"],
    )
    app.register_blueprint(_chars_mod.characters_bp)

    import web.routes.fx as _fx_mod
    _fx_mod.wire(push_fn=_chat_mod.push_raw_payload)
    app.register_blueprint(_fx_mod.fx_bp)

    services = {
        "session":    session,
        "memory":     memory,
        "safety":     safety,
        "rag":        rag,
        "conv_rag":   conv_rag,
        "initiative": initiative,
    }

    return app, services
