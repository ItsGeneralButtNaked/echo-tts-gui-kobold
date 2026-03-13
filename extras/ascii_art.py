"""
extras/ascii_art.py — ASCII art detection, extraction, and local library.

The primary ASCII art rendering lives in the frontend JS (avatar code
renderer). This module provides server-side helpers for:
  - Detecting whether a reply is/contains ASCII art before sending
  - Extracting fenced ASCII art blocks from LLM replies
  - AsciiArtLibrary: load and pick from local .txt files / multi-piece files
    so initiative/AC openers never have to ask the LLM to generate art
"""

import os
import random
import re

# Characters commonly found in ASCII art but rarely in natural prose
_ART_CHARS = set(r'|\\/\-+=#*@^~<>[]{}')

# Minimum density of art characters per line to be considered art
_ART_DENSITY_THRESHOLD = 0.35

# Minimum number of lines that must look like art
_ART_MIN_LINES = 3

# Delimiter used to separate multiple pieces inside one file
_MULTI_PIECE_DELIM = re.compile(r'^\s*---+\s*$', re.MULTILINE)


def is_ascii_art(text: str) -> bool:
    """
    Heuristic: return True if the text looks like ASCII art.

    Mirrors the _avIsAsciiArt() logic in frontend.js so the server can
    make the same determination without a round-trip.
    """
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
    """
    Extract the content of the first fenced ASCII art block.

    Looks for:
        ```
        <art content>
        ```
    or:
        ```ascii
        <art content>
        ```

    Returns the art content (without fences) or None if not found.
    """
    match = re.search(r'```(?:ascii)?\s*\n([\s\S]*?)```', text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def split_art_from_text(text: str) -> tuple[str, str | None]:
    """
    Split a reply into (prose, art_block).

    If the reply contains a fenced art block, returns the prose before it
    and the art content separately. Otherwise returns (text, None).

    Useful for sending prose to TTS while displaying art on the avatar canvas.
    """
    art = extract_fenced_art(text)
    if art is None:
        return text, None

    # Strip the fenced block from the prose
    prose = re.sub(r'```(?:ascii)?\s*\n[\s\S]*?```', '', text,
                   flags=re.IGNORECASE).strip()
    return prose, art


# ── Local art library ─────────────────────────────────────────────────────────

class AsciiArtLibrary:
    """
    Load ASCII art pieces from a directory and/or multi-piece files.

    Directory mode: each .txt file is one piece.
    Multi-piece file: pieces separated by lines of three or more dashes (---).

    Usage:
        lib = AsciiArtLibrary()
        lib.load('/app/ascii_art')          # directory of .txt files
        lib.load('/app/ascii_art/pack.txt') # single multi-piece file
        art = lib.pick()                    # random piece, or None if empty
    """

    def __init__(self):
        self._pieces: list[str] = []
        self._last_index: int = -1

    # ── loading ──────────────────────────────────────────────────────────────

    def load(self, path: str) -> int:
        """
        Load art from a file or directory.  Returns number of new pieces added.
        Silently skips if path doesn't exist so the app starts cleanly with no
        art files present.
        """
        if not path or not os.path.exists(path):
            return 0
        if os.path.isdir(path):
            return self._load_dir(path)
        return self._load_file(path)

    def _load_dir(self, dirpath: str) -> int:
        added = 0
        for fname in sorted(os.listdir(dirpath)):
            if fname.lower().endswith('.txt'):
                added += self._load_file(os.path.join(dirpath, fname))
        return added

    def _load_file(self, filepath: str) -> int:
        try:
            with open(filepath, encoding='utf-8') as f:
                raw = f.read()
        except Exception as e:
            print(f"[ASCII ART] Failed to read {filepath}: {e}")
            return 0

        pieces = [p.strip() for p in _MULTI_PIECE_DELIM.split(raw)]
        pieces = [p for p in pieces if p]  # drop empty
        self._pieces.extend(pieces)
        print(f"[ASCII ART] Loaded {len(pieces)} piece(s) from {os.path.basename(filepath)}")
        return len(pieces)

    def reload(self, path: str) -> int:
        """Clear and reload from path."""
        self._pieces.clear()
        self._last_index = -1
        return self.load(path)

    # ── picking ──────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        return len(self._pieces)

    def pick(self) -> str | None:
        """Return a random piece, avoiding immediate repeats. None if library empty."""
        if not self._pieces:
            return None
        if len(self._pieces) == 1:
            return self._pieces[0]
        candidates = [i for i in range(len(self._pieces)) if i != self._last_index]
        idx = random.choice(candidates)
        self._last_index = idx
        return self._pieces[idx]

    def pick_fenced(self) -> str | None:
        """Return a random piece wrapped in ```ascii fences, or None if empty."""
        art = self.pick()
        if art is None:
            return None
        return f"```\n{art}\n```"
