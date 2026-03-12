"""
core/characters.py — Character preset load / save.

Supports two storage modes:
  "shared"   — all UIs (desktop + web) read/write the same characters/ folder.
  "isolated" — desktop reads from characters_desktop/, web from characters_web/.

The active mode is controlled by the CHARACTER_MODE module variable and can be
changed at runtime (e.g. from a settings panel).
"""

import os
import json
from typing import Optional

from core.llm import PROVIDER_REGISTRY

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DIRS: dict[str, str] = {
    "shared":   os.path.join(_BASE, "characters"),
    "desktop":  os.path.join(_BASE, "characters_desktop"),
    "web":      os.path.join(_BASE, "characters_web"),
}

# Which surface we're running on ("desktop" | "web")
SURFACE: str = "desktop"

# "shared" uses the combined characters/ folder.
# "isolated" uses characters_desktop/ or characters_web/ depending on SURFACE.
CHARACTER_MODE: str = "shared"


def characters_dir() -> str:
    """Return the active characters directory path, creating it if needed."""
    key = "shared" if CHARACTER_MODE == "shared" else SURFACE
    d = _DIRS.get(key, _DIRS["shared"])
    os.makedirs(d, exist_ok=True)
    return d


# ─────────────────────────────────────────────────────────────────────────────
# CHARACTER SCHEMA
# ─────────────────────────────────────────────────────────────────────────────

def empty_character() -> dict:
    """Return a blank character dict with all expected keys."""
    return {
        # LLM
        "provider_id":      "koboldcpp",
        "base_url":         PROVIDER_REGISTRY["koboldcpp"]["base_url"],
        "api_key":          "",
        "agent_id":         "",
        "model":            "",
        "system_prompt":    "",
        # TTS
        "tts_provider_id":  "alltalk",
        "tts_base_url":     "http://localhost:8000",
        "tts_api_key":      "",
        "voice":            "",
        "kv_scale_enabled": False,
        "kv_scale_value":   "1.25",
        "master_gain":      1.5,
        # Auto-continue
        "auto_continue_enabled": True,
        "auto_continue_mode":    "standard",
    }


# ─────────────────────────────────────────────────────────────────────────────
# LIST
# ─────────────────────────────────────────────────────────────────────────────

def list_characters() -> list[dict]:
    """
    Return a list of {"label": str, "path": str} dicts for all .json files
    found recursively in the active characters directory.
    Path is relative to characters_dir().
    """
    base = characters_dir()
    results = []

    def scan(folder: str, prefix: str = ""):
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if os.path.isdir(full) and name.lower() != "unsorted":
                scan(full, prefix + name + "/")
            elif name.lower().endswith(".json"):
                rel   = os.path.relpath(full, base)
                label = prefix + os.path.splitext(name)[0]
                results.append({"label": label, "path": rel})

    scan(base)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_character(rel_path: str) -> Optional[dict]:
    """
    Load a character by its relative path (as returned by list_characters()).
    Returns the character dict, or None if not found.
    """
    full = os.path.join(characters_dir(), rel_path)
    if not os.path.exists(full):
        return None
    with open(full, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Back-fill any missing keys with defaults so callers can rely on schema
    merged = empty_character()
    merged.update(data)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────────────────────────────────────

def save_character(name: str, data: dict, subfolder: str = "") -> str:
    """
    Save a character JSON file.

    name      : filename without extension (e.g. "aria")
    data      : character dict (will be merged with empty_character defaults)
    subfolder : optional sub-directory inside characters_dir (e.g. "robots")

    Returns the absolute path of the saved file.
    """
    base = characters_dir()
    if subfolder:
        base = os.path.join(base, subfolder)
        os.makedirs(base, exist_ok=True)

    merged = empty_character()
    merged.update(data)

    path = os.path.join(base, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[CHARACTERS] Saved '{name}' → {path} (mode={CHARACTER_MODE})")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# DELETE
# ─────────────────────────────────────────────────────────────────────────────

def delete_character(rel_path: str) -> bool:
    """Delete a character file by relative path. Returns True on success."""
    full = os.path.join(characters_dir(), rel_path)
    if os.path.exists(full):
        os.remove(full)
        print(f"[CHARACTERS] Deleted '{rel_path}'")
        return True
    return False
