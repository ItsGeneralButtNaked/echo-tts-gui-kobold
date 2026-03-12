"""
web/session_restore.py — Session state restoration for Ecko web server.

Reads tts.extra (persisted to disk via session.json) and re-applies all
runtime settings so a server restart leaves the UI in exactly the state
the user last set it.

Called once from web/app.py during app construction.
"""

import json
import os

from core.memory import MemoryStore


def restore_session_state(
    *,
    session,
    memory,
    safety,
    initiative,
    rag,
    rag_extra_dir: str,
    memory_dir: str,
    active_mode_name_ref: list,   # single-element list used as a mutable ref
    set_memory=None,              # optional callable(MemoryStore) — replaces global ref
):
    """
    Re-apply all persisted runtime state from session.tts.extra.

    Parameters are passed explicitly so this module stays decoupled from
    the global singletons in app.py.

    active_mode_name_ref : [str] — single-element list; element 0 is read
        and written so the caller's variable is updated.
    set_memory : if provided, called with a freshly-loaded character
        MemoryStore so the global ref in app.py is replaced (same path
        as character reload).  When None, memory.enabled is still toggled
        but the bank content won't be available until a character is loaded.
    """
    ex = session.tts.extra

    # ── memory ────────────────────────────────────────────────────────────────
    if "memory_enabled" in ex:
        memory.enabled = bool(ex["memory_enabled"])
        print(f"[RESTORE] memory.enabled = {memory.enabled}")

    # ── safety layers ─────────────────────────────────────────────────────────
    if "safety_layer1_enabled" in ex:
        safety.layer1_enabled = bool(ex["safety_layer1_enabled"])
        print(f"[RESTORE] safety.layer1_enabled = {safety.layer1_enabled}")
    if "safety_layer2_enabled" in ex:
        safety.layer2_enabled = bool(ex["safety_layer2_enabled"])
        print(f"[RESTORE] safety.layer2_enabled = {safety.layer2_enabled}")

    # ── initiative ────────────────────────────────────────────────────────────
    if "initiative_fx_chance" in ex:
        initiative.fx_chance = max(0, min(100, int(ex["initiative_fx_chance"])))
    if "sleep_timer_enabled" in ex:
        v = bool(ex["sleep_timer_enabled"])
        initiative.sleep_timer_enabled = v
        session.ac_sleep_timer_enabled = v
    if "sleep_start" in ex:
        v = int(ex["sleep_start"]) % 24
        initiative.sleep_start = v
        session.ac_sleep_start = v
    if "sleep_end" in ex:
        v = int(ex["sleep_end"]) % 24
        initiative.sleep_end = v
        session.ac_sleep_end = v
    if ex.get("initiative_enabled"):
        ini_mode = ex.get("initiative_mode", "calm")
        initiative.start(ini_mode)
        print(f"[RESTORE] initiative started  mode={ini_mode!r}")

    # ── RAG extra files ───────────────────────────────────────────────────────
    if "rag_cuda" in ex:
        rag.use_cuda = bool(ex["rag_cuda"])
        print(f"[RESTORE] rag.use_cuda = {rag.use_cuda}")
    rag_file = ex.get("rag_file", "")
    if rag_file:
        _rag_filenames = [f.strip() for f in rag_file.split(",") if f.strip()]
        _rag_paths = [
            os.path.join(rag_extra_dir, fn) for fn in _rag_filenames
            if os.path.exists(os.path.join(rag_extra_dir, fn))
        ]
        if _rag_paths:
            if ex.get("rag_semantic") and not rag._use_semantic:
                rag._use_semantic = True
            if len(_rag_paths) == 1:
                rag.load(_rag_paths[0])
            else:
                rag.load_multiple(_rag_paths)
            rag.enabled = True
            print(f"[RESTORE] RAG loaded {len(_rag_paths)} file(s): {_rag_filenames}")
        else:
            print(f"[RESTORE] RAG file(s) not found, skipping: {_rag_filenames}")

    # ── loaded character ──────────────────────────────────────────────────────
    loaded_char = ex.get("loaded_char_path", "")
    if loaded_char:
        print(f"[RESTORE] Last loaded character: {loaded_char}")
        char_name    = os.path.splitext(os.path.basename(loaded_char))[0]

        # Re-load the character's memory bank so the inject chain has content
        # on first use — identical to what characters.py does on character load.
        new_memory = MemoryStore(character=char_name, memory_dir=memory_dir)
        new_memory.load()
        new_memory.enabled = bool(ex.get("memory_enabled", memory.enabled))
        if set_memory is not None:
            set_memory(new_memory)
            memory = new_memory   # keep local ref in sync for the summary log below
        else:
            # Fallback: at least copy entries into the existing object so the
            # inject fn sees them even if the global ref isn't swappable.
            memory.entries   = new_memory.entries
            memory.archived  = new_memory.archived
            memory.character = new_memory.character
            memory.enabled   = new_memory.enabled
        print(f"[RESTORE] Memory loaded: {len(memory.entries)} entries for '{char_name}'")

        bubbles_path = os.path.join(memory_dir, f"{char_name}_bubbles.json")
        if os.path.exists(bubbles_path):
            try:
                with open(bubbles_path, encoding="utf-8") as _bf:
                    _bubbles = json.load(_bf)
                if isinstance(_bubbles, list) and _bubbles:
                    session.chat_history = _bubbles
                    print(f"[RESTORE] Loaded {len(_bubbles)} bubble(s) for '{char_name}'")
            except Exception as _be:
                print(f"[RESTORE] Failed to load bubble history for '{char_name}': {_be}")

    print(
        f"[RESTORE] Session state restored — "
        f"memory={memory.enabled}  "
        f"safety_l1={safety.layer1_enabled}  "
        f"safety_l2={safety.layer2_enabled}  "
        f"context_mode={active_mode_name_ref[0]!r}"
    )
