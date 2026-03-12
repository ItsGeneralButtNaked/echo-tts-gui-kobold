"""
extras/ascii_art.py — ASCII art detection and helper utilities.

The primary ASCII art rendering lives in the frontend JS (avatar code
renderer). This module provides server-side helpers for:
  - Detecting whether a reply is/contains ASCII art before sending
  - Extracting fenced ASCII art blocks from LLM replies
  - (Future) Generating ASCII art via LLM prompt templates

These are intentionally thin — the heavy lifting stays client-side.
"""

import re

# Characters commonly found in ASCII art but rarely in natural prose
_ART_CHARS = set(r'|\/\-+=#*@^~<>[]{}')

# Minimum density of art characters per line to be considered art
_ART_DENSITY_THRESHOLD = 0.35

# Minimum number of lines that must look like art
_ART_MIN_LINES = 3


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
