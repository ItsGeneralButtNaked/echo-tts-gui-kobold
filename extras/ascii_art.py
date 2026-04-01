"""
extras/ascii_art.py — ASCII art detection, extraction, and local library.

Folder layout for AsciiArtLibrary:
    ascii_art/Default/       — fallback pool used when no character pool exists
    ascii_art/<CharName>/    — character-specific pool, e.g. ascii_art/Makima/
                               files can be named anything (e.g. Makima-1.txt)

Pick priority: character pool → Default pool → root-level .txt files → None
"""

import os
import random
import re

_ART_CHARS = set(r'|\\/\-+=#*@^~<>[]{}')
_ART_DENSITY_THRESHOLD = 0.35
_ART_MIN_LINES = 3
_MULTI_PIECE_DELIM = re.compile(r'^\s*---+\s*$', re.MULTILINE)


def is_ascii_art(text: str) -> bool:
    lines = [l for l in text.strip().splitlines() if l.strip()]
    if len(lines) < _ART_MIN_LINES:
        return False
    art_line_count = 0
    for line in lines:
        if not line:
            continue
        art_chars = sum(1 for c in line if c in _ART_CHARS)
        if len(line) > 0 and (art_chars / len(line)) >= _ART_DENSITY_THRESHOLD:
            art_line_count += 1
    return art_line_count >= _ART_MIN_LINES


def extract_fenced_art(text: str) -> str | None:
    match = re.search(r'```(?:ascii)?\s*\n([\s\S]*?)```', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def split_art_from_text(text: str) -> tuple[str, str | None]:
    art = extract_fenced_art(text)
    if art is None:
        return text, None
    prose = re.sub(r'```(?:ascii)?\s*\n[\s\S]*?```', '', text, flags=re.IGNORECASE).strip()
    return prose, art


# ── Local art library ─────────────────────────────────────────────────────────

class AsciiArtLibrary:
    """
    Load ASCII art pieces from a root ascii_art/ directory.

    Folder layout:
        ascii_art/Default/       -- general fallback pieces (any .txt)
        ascii_art/<CharName>/    -- character-specific pieces, e.g. ascii_art/Makima/
                                    files may be named anything (Makima-1.txt etc.)

    Single file paths are still supported for backwards compat.
    Multi-piece files: pieces separated by lines of 3+ dashes (---).

    pick(char_name)        -- char pool first; falls back to Default; then general
    pick_fenced(char_name) -- same, wrapped in ``` fences
    """

    def __init__(self):
        self._root: str = ""
        self._char_cache: dict[str, list[str]] = {}   # lower-case name -> pieces
        self._default: list[str] = []                  # ascii_art/Default/
        self._general: list[str] = []                  # root-level .txt files
        self._last: dict[str, int] = {}                # repeat-avoidance per pool

    # ── loading ───────────────────────────────────────────────────────────────

    def load(self, path: str) -> int:
        if not path or not os.path.exists(path):
            return 0
        if os.path.isdir(path):
            return self._load_root(path)
        return self._load_file_into(path, self._general)

    def _load_root(self, dirpath: str) -> int:
        self._root = dirpath
        self._char_cache.clear()
        self._default.clear()
        self._general.clear()
        self._last.clear()
        total = 0
        for entry in sorted(os.listdir(dirpath)):
            full = os.path.join(dirpath, entry)
            if os.path.isdir(full):
                pool: list[str] = []
                for fname in sorted(os.listdir(full)):
                    if fname.lower().endswith('.txt'):
                        total += self._load_file_into(os.path.join(full, fname), pool)
                if pool:
                    key = entry.lower()
                    if key == 'default':
                        self._default = pool
                        print(f"[ASCII ART] Default pool — {len(pool)} piece(s)")
                    else:
                        self._char_cache[key] = pool
                        print(f"[ASCII ART] '{entry}' pool — {len(pool)} piece(s)")
            elif entry.lower().endswith('.txt'):
                total += self._load_file_into(full, self._general)
        char_summary = ", ".join(f"'{k}' {len(v)}" for k, v in self._char_cache.items())
        print(f"[ASCII ART] Ready — {total} total | "
              f"default={len(self._default)} general={len(self._general)}"
              + (f" | {char_summary}" if char_summary else ""))
        return total

    def _load_file_into(self, filepath: str, pool: list) -> int:
        try:
            with open(filepath, encoding='utf-8') as f:
                raw = f.read()
        except Exception as e:
            print(f"[ASCII ART] Failed to read {filepath}: {e}")
            return 0
        pieces = [p.strip() for p in _MULTI_PIECE_DELIM.split(raw)]
        pieces = [p for p in pieces if p]
        pool.extend(pieces)
        print(f"[ASCII ART] Loaded {len(pieces)} piece(s) from {os.path.basename(filepath)}")
        return len(pieces)

    def reload(self, path: str = "") -> int:
        """Clear and reload. Uses stored root if path omitted."""
        return self.load(path or self._root)

    def reload_char(self, char_name: str) -> int:
        """Reload only one character's subfolder (e.g. after adding new files)."""
        if not self._root or not char_name:
            return 0
        key = char_name.lower()
        subdir = ""
        for entry in os.listdir(self._root):
            if entry.lower() == key and os.path.isdir(os.path.join(self._root, entry)):
                subdir = os.path.join(self._root, entry)
                break
        pool: list[str] = []
        if subdir:
            for fname in sorted(os.listdir(subdir)):
                if fname.lower().endswith('.txt'):
                    self._load_file_into(os.path.join(subdir, fname), pool)
        if key == 'default':
            self._default = pool
        else:
            self._char_cache[key] = pool
        self._last.pop(key, None)
        return len(pool)

    # ── picking ───────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return (len(self._general) + len(self._default)
                + sum(len(v) for v in self._char_cache.values()))

    def _pick_from(self, pool: list[str], pool_key: str) -> str | None:
        if not pool:
            return None
        if len(pool) == 1:
            return pool[0]
        last = self._last.get(pool_key, -1)
        candidates = [i for i in range(len(pool)) if i != last]
        idx = random.choice(candidates)
        self._last[pool_key] = idx
        return pool[idx]

    def pick(self, char_name: str = "") -> str | None:
        """Priority: character pool -> Default pool -> general root files -> None."""
        if char_name:
            key = char_name.lower()
            pool = self._char_cache.get(key)
            if pool:
                return self._pick_from(pool, key)
        if self._default:
            return self._pick_from(self._default, "default")
        if self._general:
            return self._pick_from(self._general, "general")
        return None

    def pick_fenced(self, char_name: str = "") -> str | None:
        """Return a random piece wrapped in ``` fences, or None if empty."""
        art = self.pick(char_name)
        if art is None:
            return None
        return f"```\n{art}\n```"
