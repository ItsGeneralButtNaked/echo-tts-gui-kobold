"""
web/routes/characters.py — Character preset load/save routes.

Blueprint: characters_bp
Routes:
  GET  /characters
  POST /characters/load
  POST /characters/save
"""

import json
import os

from core.logger import log
from flask import Blueprint, jsonify, request

from core.llm import PROVIDER_REGISTRY
from core.memory import MemoryStore
from core.tts import TTS_PROVIDER_REGISTRY

characters_bp = Blueprint("characters", __name__)

_FX_KEYS_LOAD = (
    "fx_enabled", "reverb_wet", "reverb_predelay",
    "delay_wet", "delay_time", "delay_feedback",
    "crush_wet", "crush_bits", "crush_sr",
    "chorus_wet", "chorus_depth", "chorus_rate",
    "ringmod_wet", "ringmod_freq",
    "ir_b64", "ir_name",
)

# Wired at startup
_get_session     = None
_get_rag         = None
_get_safety      = None
_get_memory      = None
_set_memory      = None   # (MemoryStore) -> None — replaces the global _memory ref
_get_initiative  = None
_get_conv_rag    = None
_get_chars_mod   = None
_get_base_inject = None   # () -> fn(system_prompt, user_text) -> str
_get_mode        = None   # () -> ContextMode — active context mode preset
_memory_dir      = None
_rag_dir         = None


def wire(*, get_session, get_rag, get_safety, get_memory, set_memory,
         get_initiative, get_conv_rag, get_chars_mod, get_base_inject,
         get_mode=None, memory_dir, rag_dir):
    global _get_session, _get_rag, _get_safety, _get_memory, _set_memory
    global _get_initiative, _get_conv_rag, _get_chars_mod, _get_base_inject
    global _get_mode, _memory_dir, _rag_dir
    _get_session     = get_session
    _get_rag         = get_rag
    _get_safety      = get_safety
    _get_memory      = get_memory
    _set_memory      = set_memory
    _get_initiative  = get_initiative
    _get_conv_rag    = get_conv_rag
    _get_chars_mod   = get_chars_mod
    _get_base_inject = get_base_inject
    _get_mode        = get_mode
    _memory_dir      = memory_dir
    _rag_dir         = rag_dir


@characters_bp.route("/characters", methods=["GET"])
def list_characters():
    return jsonify(_get_chars_mod().list_characters())


@characters_bp.route("/characters/adjacent", methods=["GET"])
def characters_adjacent():
    chars_mod = _get_chars_mod()
    current   = request.args.get("path", "")
    direction = request.args.get("dir", "right")

    all_chars = chars_mod.list_characters()
    paths     = [c["path"] for c in all_chars]
    if not paths:
        return jsonify({"ok": False, "error": "no characters"}), 404

    try:
        idx = paths.index(current)
    except ValueError:
        idx = 0

    next_idx = (idx - 1) % len(paths) if direction == "left" else (idx + 1) % len(paths)
    target   = all_chars[next_idx]
    return jsonify({"ok": True, "path": target["path"], "label": target["label"]})


def _bubble_history_path(char_name: str) -> str:
    """Return path to the per-character bubble history JSON file."""
    return os.path.join(_memory_dir, f"{char_name}_bubbles.json")


def _save_bubble_history(char_name: str, chat_history: list) -> None:
    """Persist current chat bubble history for a character."""
    if not char_name or not _memory_dir:
        return
    path = _bubble_history_path(char_name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[BUBBLES] Failed to save bubble history for '{char_name}': {e}")


def _load_bubble_history(char_name: str) -> list:
    """Load saved bubble history for a character, or [] if none."""
    if not char_name or not _memory_dir:
        return []
    path = _bubble_history_path(char_name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"[BUBBLES] Failed to load bubble history for '{char_name}': {e}")
        return []



@characters_bp.route("/characters/load", methods=["POST"])
def load_character():
    SESSION     = _get_session()
    rag         = _get_rag()
    _safety     = _get_safety()
    _initiative = _get_initiative()
    chars_mod   = _get_chars_mod()

    data = request.get_json(force=True)
    rel  = data.get("path", "")
    char = chars_mod.load_character(rel)
    if char is None:
        return jsonify({"error": "not found"}), 404

    # Save outgoing character's bubble history before wiping chat
    _outgoing_char = os.path.splitext(os.path.basename(
        SESSION.tts.extra.get("loaded_char_path", "")))[0]
    if _outgoing_char:
        _save_bubble_history(_outgoing_char, SESSION.chat_history)

    # LLM
    new_pid = char.get("provider_id", SESSION.llm.provider_id)
    changed = new_pid != SESSION.llm.provider_id
    SESSION.llm.provider_id   = new_pid
    SESSION.llm.base_url      = char.get("base_url", PROVIDER_REGISTRY.get(new_pid, {}).get("base_url", SESSION.llm.base_url))
    SESSION.llm.api_key       = char.get("api_key",       SESSION.llm.api_key)
    SESSION.llm.agent_id      = char.get("agent_id",      SESSION.llm.agent_id)
    SESSION.llm.model         = char.get("model",         SESSION.llm.model)
    from datetime import datetime as _dt
    _now = _dt.now()
    _time_str     = _now.strftime("%H:%M")
    _datetime_str = _now.strftime("%A %d %B %Y %H:%M")
    def _sub_time(s: str) -> str:
        return s.replace("{{local_time}}", _time_str).replace("{{local_datetime}}", _datetime_str)
    SESSION.llm.system_prompt = _sub_time(char.get("system_prompt", SESSION.llm.system_prompt))
    if "max_reply_tokens" in char:
        SESSION.llm.max_reply_tokens = int(char["max_reply_tokens"])
    if "max_history" in char:
        SESSION.llm.max_history = int(char["max_history"])

    if changed:
        SESSION.reset()
    else:
        SESSION.llm.reset_conv()

    # TTS
    if char.get("tts_provider_id") and char["tts_provider_id"] in TTS_PROVIDER_REGISTRY:
        SESSION.tts.provider_id = char["tts_provider_id"]
        SESSION.tts.base_url    = char.get("tts_base_url", TTS_PROVIDER_REGISTRY[char["tts_provider_id"]]["base_url"])
        SESSION.tts.api_key     = char.get("tts_api_key") or SESSION.tts.api_key
    if char.get("voice"):
        SESSION.tts.voice = char["voice"]

    kv = char.get("kv_scale_value")
    SESSION.tts.extra["kv_scale"] = float(kv) if char.get("kv_scale_enabled") and kv else None
    if char.get("kv_min_t") is not None:
        SESSION.tts.extra["kv_min_t"] = float(char["kv_min_t"])
    if char.get("kv_max_layers") is not None:
        SESSION.tts.extra["kv_max_layers"] = int(char["kv_max_layers"])
    if char.get("master_gain") is not None:
        SESSION.tts.extra["master_gain"] = float(char["master_gain"])

    for fx_key in _FX_KEYS_LOAD:
        if fx_key in char:
            SESSION.tts.extra[fx_key] = char[fx_key]

    for k, v in char.items():
        if k.startswith("av_"):
            if v is None:
                SESSION.tts.extra.pop(k, None)
            else:
                SESSION.tts.extra[k] = v
    if "ui_hue" in char:
        SESSION.tts.extra["ui_hue"] = int(char["ui_hue"])

    # Load character-specific memory store
    char_name = os.path.splitext(os.path.basename(rel))[0]
    new_memory = MemoryStore(character=char_name, memory_dir=_memory_dir)
    new_memory.load()
    _set_memory(new_memory)
    SESSION.llm.memory_inject_fn = new_memory.make_inject_fn(
        upstream_fn=_get_base_inject(),
        get_mode=_get_mode,
    )

    # Behaviour settings
    if "auto_continue_enabled" in char:
        SESSION.auto_continue_enabled = bool(char["auto_continue_enabled"])
        SESSION.tts.extra["ac_enabled"] = SESSION.auto_continue_enabled
        if SESSION.auto_continue_enabled:
            SESSION.start_ac_timer()
        else:
            SESSION.stop_ac_timer()
    if "auto_continue_mode" in char:
        SESSION.auto_continue_mode = char["auto_continue_mode"]
        SESSION.tts.extra["ac_mode"] = SESSION.auto_continue_mode

    if "initiative_enabled" in char or "initiative_mode" in char:
        ini_en   = bool(char.get("initiative_enabled", False))
        ini_mode = char.get("initiative_mode", "light")
        SESSION.tts.extra["initiative_enabled"] = ini_en
        SESSION.tts.extra["initiative_mode"]    = ini_mode
        if ini_en != _initiative.enabled or ini_mode != _initiative.mode:
            if ini_en:
                _initiative.start(ini_mode)
            else:
                _initiative.stop()

    if "memory_enabled" in char:
        new_memory.enabled = bool(char["memory_enabled"])
        SESSION.tts.extra["memory_enabled"] = new_memory.enabled

    _safety = _get_safety()
    if "safety_layer1_enabled" in char:
        _safety.layer1_enabled = bool(char["safety_layer1_enabled"])
    if "safety_layer2_enabled" in char:
        _safety.layer2_enabled = bool(char["safety_layer2_enabled"])

    # Always clear extra RAG first so previous character's files don't persist
    rag.clear()
    SESSION.tts.extra.pop("rag_file", None)
    SESSION.tts.extra.pop("rag_semantic", None)

    if char.get("rag_file"):
        # rag_file may be a comma-joined list of filenames
        _rag_filenames = [f.strip() for f in char["rag_file"].split(",") if f.strip()]
        _rag_paths = []
        for fn in _rag_filenames:
            try:
                full = os.path.realpath(os.path.join(_rag_dir, os.path.basename(fn)))
                root = os.path.realpath(_rag_dir)
                if full.startswith(root + os.sep) and os.path.exists(full):
                    _rag_paths.append(full)
            except Exception as e:
                log.debug("[CHARACTERS] RAG path resolution skipped for %r: %s", fn, e)
        if _rag_paths:
            try:
                if char.get("rag_semantic") and not rag._use_semantic:
                    rag._use_semantic = True
                if len(_rag_paths) == 1:
                    rag.load(_rag_paths[0])
                else:
                    rag.load_multiple(_rag_paths)
                rag.enabled = True
                SESSION.tts.extra["rag_file"]     = ",".join(_rag_filenames)
                SESSION.tts.extra["rag_semantic"] = char.get("rag_semantic", False)
            except Exception as e:
                print(f"[RAG] Failed to restore from character: {e}")

    for wk in ("wave_mode", "main_wave_visible", "avatar_wave_visible"):
        if wk in char:
            SESSION.tts.extra[wk] = char[wk]

    # Switch conv_rag to new character's file and restore its settings
    _conv_rag = _get_conv_rag()
    _conv_rag.set_character(char_name)
    if "conv_rag_enabled" in char:
        _conv_rag.enabled = bool(char["conv_rag_enabled"])
        SESSION.tts.extra["conv_rag_enabled"] = _conv_rag.enabled
    if "conv_rag_threshold" in char:
        _conv_rag.threshold = int(char["conv_rag_threshold"])
        SESSION.tts.extra["conv_rag_threshold"] = _conv_rag.threshold

    # Restore bubble history for the incoming character
    incoming_bubbles = _load_bubble_history(char_name)
    SESSION.chat_history = incoming_bubbles
    print(f"[BUBBLES] Loaded {len(incoming_bubbles)} bubble(s) for '{char_name}'")

    _safety.load_score(char_name)
    _safety.memory_hook = new_memory.add_entry

    SESSION.tts.extra["loaded_char_path"] = rel
    SESSION.save_persistent()
    print(f"[CHARACTER] Loaded '{rel}' — provider: {SESSION.llm.provider_id}, voice: {SESSION.tts.voice}")
    return jsonify({
        "ok":            True,
        "provider_id":   SESSION.llm.provider_id,
        "voice":         SESSION.tts.voice,
        "tts_provider_id": SESSION.tts.provider_id,
        "master_gain":   SESSION.tts.extra.get("master_gain", 1.5),
        "char_name":     char_name,
        "ui_hue":        SESSION.tts.extra.get("ui_hue", 140),
        "chat_history":  [{"role": m["role"], "content": m["content"],
                           "gen_images":  m.get("gen_images", []),
                           "user_image":  m.get("user_image")}
                          for m in SESSION.chat_history],
    })


@characters_bp.route("/characters/delete", methods=["POST"])
def delete_character():
    SESSION   = _get_session()
    chars_mod = _get_chars_mod()

    data = request.get_json(force=True)
    rel  = data.get("path", "").strip()
    if not rel:
        return jsonify({"error": "no path"}), 400

    # Refuse to delete the currently loaded character
    if SESSION.tts.extra.get("loaded_char_path", "") == rel:
        return jsonify({"error": "cannot delete the currently loaded character"}), 400

    ok = chars_mod.delete_character(rel)
    if not ok:
        return jsonify({"error": "not found"}), 404
    return jsonify({"ok": True})


@characters_bp.route("/characters/save", methods=["POST"])
def save_character():
    SESSION     = _get_session()
    _memory     = _get_memory()
    _initiative = _get_initiative()
    chars_mod   = _get_chars_mod()

    data = request.get_json(force=True)
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "no name"}), 400

    mg = data.get("master_gain")
    if mg is not None:
        SESSION.tts.extra["master_gain"] = float(mg)
    if "ui_hue" in data:
        SESSION.tts.extra["ui_hue"] = int(data["ui_hue"])
    kv = data.get("kv_scale")
    if kv is not None:
        SESSION.tts.extra["kv_scale"] = float(kv) if kv not in ("", "null") else None
    if "kv_min_t" in data:
        SESSION.tts.extra["kv_min_t"] = float(data["kv_min_t"])
    if "kv_max_layers" in data:
        SESSION.tts.extra["kv_max_layers"] = int(data["kv_max_layers"])

    char_data = {
        "provider_id":      SESSION.llm.provider_id,
        "base_url":         SESSION.llm.base_url,
        "api_key":          SESSION.llm.api_key,
        "agent_id":         SESSION.llm.agent_id,
        "model":            SESSION.llm.model,
        "system_prompt":    SESSION.llm.system_prompt,
        "max_reply_tokens": SESSION.llm.max_reply_tokens,
        "max_history":      SESSION.llm.max_history,
        "tts_provider_id":  SESSION.tts.provider_id,
        "tts_base_url":     SESSION.tts.base_url,
        "tts_api_key":      SESSION.tts.api_key,
        "voice":            SESSION.tts.voice,
        "kv_scale_enabled": SESSION.tts.extra.get("kv_scale") is not None,
        "kv_scale_value":   str(SESSION.tts.extra.get("kv_scale") or "1.25"),
        "kv_min_t":         SESSION.tts.extra.get("kv_min_t", 0.9),
        "kv_max_layers":    SESSION.tts.extra.get("kv_max_layers", 24),
        "master_gain":      SESSION.tts.extra.get("master_gain", 1.5),
        "fx_enabled":       SESSION.tts.extra.get("fx_enabled", False),
        "reverb_wet":       SESSION.tts.extra.get("reverb_wet", 0.25),
        "reverb_predelay":  SESSION.tts.extra.get("reverb_predelay", 20),
        "delay_wet":        SESSION.tts.extra.get("delay_wet", 0.25),
        "delay_time":       SESSION.tts.extra.get("delay_time", 0),
        "delay_feedback":   SESSION.tts.extra.get("delay_feedback", 0.35),
        "chorus_wet":       SESSION.tts.extra.get("chorus_wet", 0),
        "chorus_depth":     SESSION.tts.extra.get("chorus_depth", 0.005),
        "chorus_rate":      SESSION.tts.extra.get("chorus_rate", 1.2),
        "ringmod_wet":      SESSION.tts.extra.get("ringmod_wet", 0),
        "ringmod_freq":     SESSION.tts.extra.get("ringmod_freq", 120),
        "crush_wet":        SESSION.tts.extra.get("crush_wet", 0),
        "crush_bits":       SESSION.tts.extra.get("crush_bits", 8),
        "crush_sr":         SESSION.tts.extra.get("crush_sr", 4),
        "ir_b64":           SESSION.tts.extra.get("ir_b64", None),
        "ir_name":          SESSION.tts.extra.get("ir_name", ""),
        # UI
        "ui_hue": SESSION.tts.extra.get("ui_hue", 140),
        # Safety layers
        "safety_layer1_enabled": _get_safety().layer1_enabled,
        "safety_layer2_enabled": _get_safety().layer2_enabled,
        # Conv RAG
        "conv_rag_enabled":   SESSION.tts.extra.get("conv_rag_enabled", False),
        "conv_rag_threshold": SESSION.tts.extra.get("conv_rag_threshold", 20),
        # Behaviour
        "auto_continue_enabled": SESSION.tts.extra.get("ac_enabled", SESSION.auto_continue_enabled),
        "auto_continue_mode":    SESSION.tts.extra.get("ac_mode",    SESSION.auto_continue_mode),
        "initiative_enabled":    SESSION.tts.extra.get("initiative_enabled", _initiative.enabled),
        "initiative_mode":       SESSION.tts.extra.get("initiative_mode",    _initiative.mode),
        "memory_enabled":        SESSION.tts.extra.get("memory_enabled", _memory.enabled),
        "rag_file":              SESSION.tts.extra.get("rag_file", ""),
        "rag_semantic":          SESSION.tts.extra.get("rag_semantic", False),
        "wave_mode":             SESSION.tts.extra.get("wave_mode", "ribbon"),
        "main_wave_visible":     SESSION.tts.extra.get("main_wave_visible", True),
        "avatar_wave_visible":   SESSION.tts.extra.get("avatar_wave_visible", True),
    }
    for k, v in data.items():
        if k.startswith("av_"):
            char_data[k] = v

    # Ensure all avatar image slots are explicitly written — slots absent from
    # the payload mean the user cleared them, so write null to overwrite any
    # previously saved image for that slot.
    for _slot in ('idle', 'talk', 'blink-closed', 'blink-talk', 'scream', 'sleep'):
        _key = f'av_img_{_slot}'
        if _key not in char_data:
            char_data[_key] = None

    subfolder = data.get("subfolder", "").strip()
    path = chars_mod.save_character(name, char_data, subfolder=subfolder)
    return jsonify({"ok": True, "path": os.path.relpath(path, chars_mod.characters_dir())})
