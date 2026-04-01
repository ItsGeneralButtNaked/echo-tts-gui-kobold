"""
web/routes/settings.py — Session state, settings, and reset routes.

Blueprint: settings_bp
Routes:
  GET  /state
  POST /settings
  POST /reset
  GET  /guest_config
"""

import os
from flask import Blueprint, jsonify, request, send_from_directory

from core.llm import PROVIDER_REGISTRY
from core.tts import TTS_PROVIDER_REGISTRY

settings_bp = Blueprint("settings", __name__)

# Wired at startup
_get_session      = None
_get_rag          = None
_get_memory       = None
_get_safety       = None
_get_initiative   = None
_get_conv_rag     = None
_get_session_mode = None
_set_session_mode = None
_get_isolated_llm = None
_get_chars_mod    = None
_get_guest_config = None  # () -> (GUEST_MODE, GUEST_CONFIG)
_get_active_mode  = None  # () -> ContextMode
_set_active_mode  = None  # (name: str) -> None
_get_art_lib      = None
_art_lib_path     = ""
_get_image_lib    = None
_image_dir        = ""
_get_video_lib    = None
_video_dir        = ""


def wire(*, get_session, get_rag, get_memory, get_safety, get_initiative, get_conv_rag,
         get_session_mode, set_session_mode, get_isolated_llm,
         get_chars_mod, get_guest_config, get_active_mode=None, set_active_mode=None,
         get_art_lib=None, art_lib_path="",
         get_image_lib=None, image_dir="",
         get_video_lib=None, video_dir=""):
    global _get_session, _get_rag, _get_memory, _get_safety, _get_initiative, _get_conv_rag
    global _get_session_mode, _set_session_mode, _get_isolated_llm
    global _get_chars_mod, _get_guest_config, _get_active_mode, _set_active_mode
    global _get_art_lib, _art_lib_path, _get_image_lib, _image_dir, _get_video_lib, _video_dir
    _get_session      = get_session
    _get_rag          = get_rag
    _get_memory       = get_memory
    _get_safety       = get_safety
    _get_initiative   = get_initiative
    _get_conv_rag     = get_conv_rag
    _get_session_mode = get_session_mode
    _set_session_mode = set_session_mode
    _get_isolated_llm = get_isolated_llm
    _get_chars_mod    = get_chars_mod
    _get_guest_config = get_guest_config
    _get_active_mode  = get_active_mode
    _set_active_mode  = set_active_mode
    _get_art_lib      = get_art_lib or (lambda: None)
    _art_lib_path     = art_lib_path
    _get_image_lib    = get_image_lib or (lambda: None)
    _image_dir        = image_dir
    _get_video_lib    = get_video_lib or (lambda: None)
    _video_dir        = video_dir


# ── /state ────────────────────────────────────────────────────────────────────

@settings_bp.route("/state", methods=["GET"])
def get_state():
    SESSION      = _get_session()
    rag          = _get_rag()
    _memory      = _get_memory()
    _initiative  = _get_initiative()
    SESSION_MODE = _get_session_mode()
    chars_mod    = _get_chars_mod()
    # _get_conv_rag() called inline in response dict

    voices               = SESSION.tts.list_voices()
    tts_online, tts_label = SESSION.tts.ping()
    llm_online, llm_label = SESSION.llm.ping()
    prov = PROVIDER_REGISTRY.get(SESSION.llm.provider_id, {})

    return jsonify({
        # LLM
        "provider_id":       SESSION.llm.provider_id,
        "provider_label":    prov.get("label", SESSION.llm.provider_id),
        "provider_registry": {k: {"label": v["label"], "base_url": v["base_url"],
                                   "needs_api_key": v["needs_api_key"],
                                   "needs_agent_id": v["needs_agent_id"],
                                   "needs_model": v["needs_model"]}
                               for k, v in PROVIDER_REGISTRY.items()},
        "base_url":          SESSION.llm.base_url,
        "api_key":           bool(SESSION.llm.api_key),    # masked — never expose key to browser
        "agent_id":          SESSION.llm.agent_id,
        "model":             SESSION.llm.model,
        "system_prompt":     SESSION.llm.system_prompt,
        "first_message":     SESSION.tts.extra.get("first_message", ""),
        "first_message_tts": SESSION.tts.extra.get("first_message_tts", False),
        "max_reply_tokens":  SESSION.llm.max_reply_tokens,
        "max_history":       SESSION.llm.max_history,
        "llm_label":         llm_label,
        "llm_online":        llm_online,
        # TTS
        "tts_provider_id":   SESSION.tts.provider_id,
        "tts_provider_registry": {k: {"label": v["label"], "base_url": v["base_url"],
                                       "needs_api_key": v["needs_api_key"]}
                                   for k, v in TTS_PROVIDER_REGISTRY.items()},
        "tts_base_url":      SESSION.tts.base_url,
        "tts_api_key":       bool(SESSION.tts.api_key),    # masked
        "voice":             SESSION.tts.voice,
        "voices":            voices,
        "tts_online":        tts_online,
        "tts_label":         tts_label,
        # Auto-continue
        "auto_continue_enabled": SESSION.tts.extra.get("ac_enabled", SESSION.auto_continue_enabled),
        "auto_continue_mode":    SESSION.tts.extra.get("ac_mode",    SESSION.auto_continue_mode),
        # Initiative
        "initiative_enabled":    SESSION.tts.extra.get("initiative_enabled", _initiative.enabled),
        "initiative_mode":       SESSION.tts.extra.get("initiative_mode",    _initiative.mode),
        "initiative_next_secs":  max(0, _initiative.status()["secs_remaining"]),
        "initiative_fx_chance":      SESSION.tts.extra.get("initiative_fx_chance",      _initiative.fx_chance),
        "initiative_img_chance":     SESSION.tts.extra.get("initiative_img_chance",     _initiative.img_chance),
        "initiative_video_chance":   SESSION.tts.extra.get("initiative_video_chance",   _initiative.video_chance),
        "initiative_ascii_chance":   SESSION.tts.extra.get("initiative_ascii_chance",   _initiative.ascii_chance),
        "initiative_terminal_chance":SESSION.tts.extra.get("initiative_terminal_chance",_initiative.terminal_chance),
        "initiative_glitch_chance":  SESSION.tts.extra.get("initiative_glitch_chance",  _initiative.glitch_chance),
        "initiative_in_sleep":   _initiative.status()["in_sleep_window"],
        "sleep_timer_enabled":   SESSION.tts.extra.get("sleep_timer_enabled", _initiative.sleep_timer_enabled),
        "sleep_start":           SESSION.tts.extra.get("sleep_start", _initiative.sleep_start),
        "sleep_end":             SESSION.tts.extra.get("sleep_end",   _initiative.sleep_end),
        # Audio FX
        "kv_scale":        SESSION.tts.extra.get("kv_scale"),
        "kv_min_t":        SESSION.tts.extra.get("kv_min_t", 0.9),
        "kv_max_layers":   SESSION.tts.extra.get("kv_max_layers", 24),
        "fx_enabled":      SESSION.tts.extra.get("fx_enabled", False),
        "reverb_wet":      SESSION.tts.extra.get("reverb_wet", 0.25),
        "reverb_predelay": SESSION.tts.extra.get("reverb_predelay", 20),
        "delay_wet":       SESSION.tts.extra.get("delay_wet", 0.25),
        "delay_time":      SESSION.tts.extra.get("delay_time", 0),
        "delay_feedback":  SESSION.tts.extra.get("delay_feedback", 0.35),
        "chorus_wet":      SESSION.tts.extra.get("chorus_wet", 0),
        "chorus_depth":    SESSION.tts.extra.get("chorus_depth", 0.005),
        "chorus_rate":     SESSION.tts.extra.get("chorus_rate", 1.2),
        "ringmod_wet":     SESSION.tts.extra.get("ringmod_wet", 0),
        "ringmod_freq":    SESSION.tts.extra.get("ringmod_freq", 120),
        "crush_wet":       SESSION.tts.extra.get("crush_wet", 0),
        "crush_bits":      SESSION.tts.extra.get("crush_bits", 8),
        "crush_sr":        SESSION.tts.extra.get("crush_sr", 4),
        "dist_wet":        SESSION.tts.extra.get("dist_wet", 0),
        "dist_drive":      SESSION.tts.extra.get("dist_drive", 20),
        "revgate_wet":     SESSION.tts.extra.get("revgate_wet", 0),
        "revgate_length":  SESSION.tts.extra.get("revgate_length", 400),
        "ir_b64":          SESSION.tts.extra.get("ir_b64", None),
        "ir_name":         SESSION.tts.extra.get("ir_name", ""),
        "master_gain":     SESSION.tts.extra.get("master_gain", 1.5),
        "session_mode":    SESSION_MODE,
        "char_mode":       chars_mod.CHARACTER_MODE,
        # Avatar display settings — echoed back from char file if present
        **{k: SESSION.tts.extra[k] for k in SESSION.tts.extra if k.startswith("av_")},
        # Wave display state
        "wave_mode":           SESSION.tts.extra.get("wave_mode", "ribbon"),
        "main_wave_visible":   SESSION.tts.extra.get("main_wave_visible", True),
        "avatar_wave_visible": SESSION.tts.extra.get("avatar_wave_visible", True),
        "visual_fx_enabled":   SESSION.tts.extra.get("visual_fx_enabled", False),
        "mood_fx_enabled":     SESSION.tts.extra.get("mood_fx_enabled",   False),
        "barge_in_enabled":    SESSION.tts.extra.get("barge_in_enabled",  True),
        "sub_speed":           SESSION.tts.extra.get("sub_speed", 11),
        "ui_hue":              SESSION.tts.extra.get("ui_hue", 140),
        "char_name":           os.path.splitext(os.path.basename(
                                   SESSION.tts.extra.get("loaded_char_path", "")
                               ))[0] or "",
        "loaded_char_path":    SESSION.tts.extra.get("loaded_char_path", ""),
        "safety_indicator_visible": SESSION.tts.extra.get("safety_indicator_visible", True),
        "wave_amp":  SESSION.tts.extra.get("wave_amp", 1.0),
        "wave_fade": SESSION.tts.extra.get("wave_fade", 0.25),
        "has_conversation": bool(SESSION.chat_history),
        "chat_history":     [{"role": m["role"], "content": m["content"],
                               "gen_images":  m.get("gen_images", []),
                               "user_image":  m.get("user_image")}
                              for m in SESSION.chat_history],
        # Extra RAG
        "rag_enabled":  rag.enabled,
        "rag_file":     SESSION.tts.extra.get("rag_file", ""),
        "rag_chunks":   len(rag.chunks),
        "rag_semantic": rag._use_semantic and rag._index is not None,
        "rag_cuda":     SESSION.tts.extra.get("rag_cuda", False),
        # STT settings
        "stt_provider": SESSION.tts.extra.get("stt_provider", "faster-whisper"),
        "stt_model":    SESSION.tts.extra.get("stt_model", "base"),
        "stt_cuda":     SESSION.tts.extra.get("stt_cuda", False),
        # Conversation RAG
        "conv_rag_enabled":   _get_conv_rag().enabled,
        "conv_rag_threshold": _get_conv_rag().threshold,
        "conv_rag_file":      _get_conv_rag().filename,
        "conv_rag_exists":    _get_conv_rag().status()["exists"],
        # ASCII art library
        "art_lib_count":   (_get_art_lib().count   if _get_art_lib()   else 0),
        "image_lib_count": (_get_image_lib().count if _get_image_lib() else 0),
        "video_lib_count": (_get_video_lib().count if _get_video_lib() else 0),
        # Memory
        "memory_enabled": SESSION.tts.extra.get("memory_enabled", _memory.enabled),
        "memory_count":   len(_memory.entries),
        # Safety layers
        "safety_layer1_enabled": _get_safety().layer1_enabled,
        "safety_layer2_enabled": _get_safety().layer2_enabled,
        # Context mode
        "context_mode": SESSION.tts.extra.get("context_mode", "standard"),
        # Web search
        "websearch_enabled":      SESSION.tts.extra.get("websearch_enabled", False),
        "websearch_api_key":      bool(SESSION.tts.extra.get("websearch_api_key", "")),  # masked
        "websearch_result_count": SESSION.tts.extra.get("websearch_result_count", 3),
        # Vision
        "vision_enabled":         SESSION.tts.extra.get("vision_enabled", False),
    })


# ── /settings ────────────────────────────────────────────────────────────────

_FX_KEYS = (
    "fx_enabled", "reverb_wet", "reverb_predelay",
    "delay_wet", "delay_time", "delay_feedback",
    "crush_wet", "crush_bits", "crush_sr",
    "chorus_wet", "chorus_depth", "chorus_rate",
    "ringmod_wet", "ringmod_freq",
    "ir_b64", "ir_name",
    "dist_wet", "dist_drive",
    "revgate_wet", "revgate_length",
)


@settings_bp.route("/settings", methods=["POST"])
def update_settings():
    SESSION      = _get_session()
    _initiative  = _get_initiative()
    chars_mod    = _get_chars_mod()
    data = request.get_json(force=True)
    changed_provider = False

    # LLM
    if "provider_id" in data:
        new_pid = data["provider_id"]
        if new_pid in PROVIDER_REGISTRY and new_pid != SESSION.llm.provider_id:
            SESSION.llm.provider_id = new_pid
            SESSION.llm.base_url = PROVIDER_REGISTRY[new_pid]["base_url"]
            changed_provider = True
    for field in ("base_url", "agent_id", "model", "system_prompt"):
        if field in data:
            setattr(SESSION.llm, field, data[field])
    if "first_message" in data:
        SESSION.tts.extra["first_message"] = data["first_message"]
    if "first_message_tts" in data:
        SESSION.tts.extra["first_message_tts"] = bool(data["first_message_tts"])
    # Only update api_key when a real key string is supplied.
    # Guard against the masked boolean ("true"/"false") being echoed back by
    # loadState → applySettings when the user opens settings without re-entering
    # their key — that would silently overwrite the real key with "true".
    if "api_key" in data and isinstance(data["api_key"], str) and data["api_key"] and data["api_key"] not in ("true", "false"):
        SESSION.llm.api_key = data["api_key"]
    if "max_reply_tokens" in data:
        val = data["max_reply_tokens"]
        SESSION.llm.max_reply_tokens = int(val) if val else 300
    if "max_history" in data:
        val = data["max_history"]
        SESSION.llm.max_history = int(val) if val else 10
    if changed_provider:
        SESSION.reset()

    # TTS
    if "tts_provider_id" in data:
        new_tpid = data["tts_provider_id"]
        if new_tpid in TTS_PROVIDER_REGISTRY:
            SESSION.tts.provider_id = new_tpid
            SESSION.tts.base_url = TTS_PROVIDER_REGISTRY[new_tpid]["base_url"]
    if "tts_base_url" in data:
        SESSION.tts.base_url = data["tts_base_url"]
    # Only update tts_api_key when a real key string is supplied.
    if "tts_api_key" in data and isinstance(data["tts_api_key"], str) and data["tts_api_key"] and data["tts_api_key"] not in ("true", "false"):
        SESSION.tts.api_key = data["tts_api_key"]
    if "voice" in data:
        import re as _re
        raw_voice = data["voice"]
        # Reject data URLs and base64 blobs — never valid voice IDs
        if raw_voice and (raw_voice.startswith("data:") or len(raw_voice) > 256):
            print(f"[TTS] Rejected invalid voice value (len={len(raw_voice)})")
        else:
            # Strip display suffixes added by list_voices:
            #   ElevenLabs: "Name (voice_id)"
            #   Hume:       "Name [Hume] (uuid)" or "Name [Custom] (uuid)"
            _el_match = _re.match(r'^.+?\s*(?:\[.*?\]\s*)?\(([A-Za-z0-9\-]+)\)\s*$', raw_voice)
            SESSION.tts.voice = _el_match.group(1) if _el_match else raw_voice
    if "kv_scale" in data:
        val = data["kv_scale"]
        SESSION.tts.extra["kv_scale"] = float(val) if val not in (None, "", "null") else None
    if "kv_min_t" in data:
        SESSION.tts.extra["kv_min_t"] = float(data["kv_min_t"])
    if "kv_max_layers" in data:
        SESSION.tts.extra["kv_max_layers"] = int(data["kv_max_layers"])

    for fx_key in _FX_KEYS:
        if fx_key in data:
            SESSION.tts.extra[fx_key] = data[fx_key]

    if "master_gain" in data:
        val = data["master_gain"]
        SESSION.tts.extra["master_gain"] = float(val) if val not in (None, "", "null") else 1.5

    # Avatar display settings
    for k, v in data.items():
        if k.startswith("av_"):
            SESSION.tts.extra[k] = v

    # Vision
    if "vision_enabled" in data:
        SESSION.tts.extra["vision_enabled"] = bool(data["vision_enabled"])

    # Web search
    for k in ("websearch_enabled", "websearch_api_key", "websearch_result_count"):
        if k in data:
            v = data[k]
            if k == "websearch_enabled":
                v = bool(v)
            elif k == "websearch_api_key":
                # Only update when a real string key is supplied (masked state returns bool)
                if not isinstance(v, str) or not v:
                    continue
            elif k == "websearch_result_count":
                v = int(v) if v else 3
            SESSION.tts.extra[k] = v

    # Wave display state + FX toggles
    for k in ("wave_mode", "main_wave_visible", "avatar_wave_visible",
              "ui_hue", "safety_indicator_visible", "wave_amp", "wave_fade",
              "visual_fx_enabled", "mood_fx_enabled", "barge_in_enabled"):
        if k in data:
            SESSION.tts.extra[k] = data[k]

    if "sub_speed" in data:
        SESSION.tts.extra["sub_speed"] = max(4, min(30, int(data["sub_speed"])))

    # Auto-continue
    if "auto_continue_enabled" in data:
        SESSION.auto_continue_enabled = bool(data["auto_continue_enabled"])
        SESSION.tts.extra["ac_enabled"] = SESSION.auto_continue_enabled
        if SESSION.auto_continue_enabled:
            SESSION.start_ac_timer()
        else:
            SESSION.stop_ac_timer()
    if "auto_continue_mode" in data:
        SESSION.auto_continue_mode = data["auto_continue_mode"]
        SESSION.tts.extra["ac_mode"] = SESSION.auto_continue_mode

    # Initiative
    if "initiative_enabled" in data or "initiative_mode" in data:
        ini_enabled = data.get("initiative_enabled", _initiative.enabled)
        ini_mode    = data.get("initiative_mode",    _initiative.mode)
        SESSION.tts.extra["initiative_enabled"] = ini_enabled
        SESSION.tts.extra["initiative_mode"]    = ini_mode
        if ini_enabled != _initiative.enabled or ini_mode != _initiative.mode:
            if ini_enabled:
                _initiative.start(ini_mode)
            else:
                _initiative.stop()

    if "initiative_fx_chance" in data:
        val = int(data["initiative_fx_chance"])
        _initiative.fx_chance = max(0, min(100, val))
        SESSION.tts.extra["initiative_fx_chance"] = _initiative.fx_chance
    if "initiative_img_chance" in data:
        val = int(data["initiative_img_chance"])
        _initiative.img_chance = max(0, min(100, val))
        SESSION.tts.extra["initiative_img_chance"] = _initiative.img_chance
    if "initiative_video_chance" in data:
        val = int(data["initiative_video_chance"])
        _initiative.video_chance = max(0, min(100, val))
        SESSION.tts.extra["initiative_video_chance"] = _initiative.video_chance
    if "initiative_ascii_chance" in data:
        val = int(data["initiative_ascii_chance"])
        _initiative.ascii_chance = max(0, min(100, val))
        SESSION.tts.extra["initiative_ascii_chance"] = _initiative.ascii_chance
    if "initiative_terminal_chance" in data:
        val = int(data["initiative_terminal_chance"])
        _initiative.terminal_chance = max(0, min(100, val))
        SESSION.tts.extra["initiative_terminal_chance"] = _initiative.terminal_chance
    if "initiative_glitch_chance" in data:
        val = int(data["initiative_glitch_chance"])
        _initiative.glitch_chance = max(0, min(100, val))
        SESSION.tts.extra["initiative_glitch_chance"] = _initiative.glitch_chance

    # Sleep timer (shared by initiative + auto-continue)
    if "sleep_timer_enabled" in data:
        v = bool(data["sleep_timer_enabled"])
        _initiative.sleep_timer_enabled = v
        SESSION.ac_sleep_timer_enabled  = v
        SESSION.tts.extra["sleep_timer_enabled"] = v
    if "sleep_start" in data:
        v = int(data["sleep_start"]) % 24
        _initiative.sleep_start = v
        SESSION.ac_sleep_start  = v
        SESSION.tts.extra["sleep_start"] = v
    if "sleep_end" in data:
        v = int(data["sleep_end"]) % 24
        _initiative.sleep_end = v
        SESSION.ac_sleep_end  = v
        SESSION.tts.extra["sleep_end"] = v

    # RAG CUDA device toggle — clear embedder so it reloads on next index build
    if "rag_cuda" in data:
        rag = _get_rag()
        new_cuda = bool(data["rag_cuda"])
        old_cuda = SESSION.tts.extra.get("rag_cuda", False)
        SESSION.tts.extra["rag_cuda"] = new_cuda
        rag.use_cuda = new_cuda
    # STT settings
    _stt_changed = False
    if "stt_provider" in data:
        SESSION.tts.extra["stt_provider"] = str(data["stt_provider"])
        _stt_changed = True
    if "stt_model" in data:
        SESSION.tts.extra["stt_model"] = str(data["stt_model"])
        _stt_changed = True
    if "stt_api_key" in data:
        SESSION.tts.extra["stt_api_key"] = str(data["stt_api_key"])
    if "stt_api_url" in data:
        SESSION.tts.extra["stt_api_url"] = str(data["stt_api_url"])
    if "stt_cuda" in data:
        new_cuda = bool(data["stt_cuda"])
        SESSION.tts.extra["stt_cuda"] = new_cuda
        _stt_changed = True
    if _stt_changed:
        import core.stt as _stt_mod
        _stt_mod._whisper_model = None
        _model  = SESSION.tts.extra.get("stt_model", "base")
        _device = "cuda" if SESSION.tts.extra.get("stt_cuda", False) else "cpu"
        print(f"[STT] Settings changed — will reload as {_model!r} on {_device}")
        import threading as _t
        _t.Thread(target=_stt_mod.get_whisper,
                  kwargs={"model": _model, "device": _device},
                  daemon=True).start()

    # Conversation RAG
    if "conv_rag_enabled" in data or "conv_rag_threshold" in data:
        conv = _get_conv_rag()
        if "conv_rag_enabled" in data:
            conv.enabled = bool(data["conv_rag_enabled"])
            SESSION.tts.extra["conv_rag_enabled"] = conv.enabled
        if "conv_rag_threshold" in data:
            conv.threshold = max(6, min(int(data["conv_rag_threshold"]), 200))
            SESSION.tts.extra["conv_rag_threshold"] = conv.threshold

    # Context mode preset
    if "context_mode" in data and callable(_set_active_mode):
        _set_active_mode(data["context_mode"])

    # Session / character mode
    if "session_mode" in data:
        _set_session_mode(data["session_mode"])
        print(f"[SESSION] Mode: {data['session_mode']}")
    if "char_mode" in data and data["char_mode"] in ("shared", "isolated"):
        chars_mod.CHARACTER_MODE = data["char_mode"]
        print(f"[CHARACTERS] Mode: {chars_mod.CHARACTER_MODE}")

    SESSION.save_persistent()
    return jsonify({"ok": True})


# ── /clear_all ───────────────────────────────────────────────────────────────
# Nuclear reset: clears conv_rag file, extra RAG index, all memories, and
# the conversation history — giving the agent a completely fresh slate.

@settings_bp.route("/clear_all", methods=["POST"])
def clear_all():
    SESSION  = _get_session()
    rag      = _get_rag()
    memory   = _get_memory()
    conv_rag = _get_conv_rag()

    # 1. Conversation history
    SESSION.reset()

    # 2. Extra RAG (in-memory index only — doesn't delete files on disk)
    rag.clear()

    # 3. Conversation RAG file
    conv_rag.clear_file()

    # 4. Memory entries (keeps enabled state)
    was_enabled = memory.enabled
    memory.clear_all()
    memory.enabled = was_enabled
    memory.save()

    # Re-inject first message now that history is empty
    first_msg = SESSION.tts.extra.get("first_message", "").strip()
    if first_msg:
        SESSION.chat_history = [{"role": "assistant", "content": first_msg}]
        SESSION.save_persistent()
        _char_path = SESSION.tts.extra.get("loaded_char_path", "")
        if _char_path:
            import os as _os, json as _json
            _char_name = _os.path.splitext(_os.path.basename(_char_path))[0]
            _mem_dir = getattr(memory, "_dir", None)
            if _mem_dir:
                _bubbles_path = _os.path.join(_mem_dir, f"{_char_name}_bubbles.json")
                try:
                    with open(_bubbles_path, "w", encoding="utf-8") as _bf:
                        _json.dump(SESSION.chat_history, _bf, ensure_ascii=False)
                except Exception as _be:
                    print(f"[BUBBLES] Clear-all bubble-save failed: {_be}")
        print("[FIRST_MSG] Re-injected opening message after clear_all")
        return jsonify({"ok": True, "first_message": first_msg,
                        "first_message_tts": SESSION.tts.extra.get("first_message_tts", False)})

    return jsonify({"ok": True})


# ── /reset ───────────────────────────────────────────────────────────────────

@settings_bp.route("/reset", methods=["POST"])
def reset_conv():
    SESSION      = _get_session()
    SESSION_MODE = _get_session_mode()
    if SESSION_MODE == "isolated":
        ip = request.remote_addr or "unknown"
        isolated = _get_isolated_llm()
        if ip in isolated:
            isolated[ip].reset_conv()
    else:
        SESSION.reset()

    # Re-inject first message on a now-empty history
    first_msg = SESSION.tts.extra.get("first_message", "").strip()
    if first_msg:
        SESSION.chat_history = [{"role": "assistant", "content": first_msg}]
        SESSION.save_persistent()
        # Keep the per-character bubble file in sync so restarts restore correctly
        _char_path = SESSION.tts.extra.get("loaded_char_path", "")
        if _char_path:
            import os as _os, json as _json
            _char_name = _os.path.splitext(_os.path.basename(_char_path))[0]
            _mem_dir = getattr(_get_memory(), "_dir", None)
            if _mem_dir:
                _bubbles_path = _os.path.join(_mem_dir, f"{_char_name}_bubbles.json")
                try:
                    with open(_bubbles_path, "w", encoding="utf-8") as _bf:
                        _json.dump(SESSION.chat_history, _bf, ensure_ascii=False)
                except Exception as _be:
                    print(f"[BUBBLES] Reset bubble-save failed: {_be}")
        print("[FIRST_MSG] Re-injected opening message after reset")
        return jsonify({"ok": True, "first_message": first_msg,
                        "first_message_tts": SESSION.tts.extra.get("first_message_tts", False)})

    return jsonify({"ok": True})


# ── /guest_config ─────────────────────────────────────────────────────────────

@settings_bp.route("/guest_config")
def guest_config():
    guest_mode, guest_cfg = _get_guest_config()
    return jsonify({
        "guest_mode": guest_mode,
        "title":      guest_cfg.get("title", ""),
        "character":  guest_cfg.get("character", ""),
    })


# ── /images/<path> ───────────────────────────────────────────────────────────
# Serve image library files by relative path so gen_images URLs are lightweight
# strings rather than base64 blobs.  Path is validated to stay inside _image_dir.

@settings_bp.route("/images/<path:rel_path>")
def serve_image(rel_path):
    if not _image_dir:
        return "Image library not configured", 404
    safe_root = os.path.realpath(_image_dir)
    target    = os.path.realpath(os.path.join(_image_dir, rel_path))
    if not target.startswith(safe_root + os.sep):
        return "Forbidden", 403
    directory = os.path.dirname(target)
    filename  = os.path.basename(target)
    return send_from_directory(directory, filename)


@settings_bp.route("/videos/<path:rel_path>")
def serve_video(rel_path):
    if not _video_dir:
        return "Video library not configured", 404
    safe_root = os.path.realpath(_video_dir)
    target    = os.path.realpath(os.path.join(_video_dir, rel_path))
    if not target.startswith(safe_root + os.sep):
        return "Forbidden", 403
    directory = os.path.dirname(target)
    filename  = os.path.basename(target)
    return send_from_directory(directory, filename)


# ── /ascii_art/reload ─────────────────────────────────────────────────────────

@settings_bp.route("/ascii_art/reload", methods=["POST"])
def ascii_art_reload():
    lib = _get_art_lib()
    if lib is None:
        return jsonify({"ok": False, "error": "art library not configured"}), 500
    count = lib.reload(_art_lib_path)
    print(f"[ASCII ART] Reloaded — {count} piece(s) from {_art_lib_path!r}")
    return jsonify({"ok": True, "count": count})


# ── /image_lib/reload ─────────────────────────────────────────────────────────

@settings_bp.route("/image_lib/reload", methods=["POST"])
def image_lib_reload():
    lib = _get_image_lib()
    if lib is None:
        return jsonify({"ok": False, "error": "image library not configured"}), 500
    data = request.get_json(force=True) or {}
    char_name = data.get("char_name", "").strip()
    count = lib.reload(char_name) if char_name else lib.reload()
    print(f"[IMAGE_LIB] Reloaded — {count} image(s)" +
          (f" for '{char_name}'" if char_name else f" from {_image_dir!r}"))
    return jsonify({"ok": True, "count": count})


# ── /video_lib/reload ─────────────────────────────────────────────────────────

@settings_bp.route("/video_lib/reload", methods=["POST"])
def video_lib_reload():
    lib = _get_video_lib()
    if lib is None:
        return jsonify({"ok": False, "error": "video library not configured"}), 500
    data = request.get_json(force=True) or {}
    char_name = data.get("char_name", "").strip()
    count = lib.reload(char_name) if char_name else lib.reload()
    print(f"[VIDEO_LIB] Reloaded — {count} video(s)" +
          (f" for '{char_name}'" if char_name else f" from {_video_dir!r}"))
    return jsonify({"ok": True, "count": count})
