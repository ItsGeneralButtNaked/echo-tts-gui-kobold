"""
extras/python_code.py — Python / code-block detection helpers.

The primary code rendering lives in the frontend JS (avatar code renderer,
_avExtractCodeBlock / _avTokenisePython). This module provides the matching
server-side helpers for:
  - Detecting whether a reply contains a Python / code block
  - Extracting the code content (fenced or heuristic)
  - Stripping code blocks from text before sending to TTS

Detection priority mirrors _avExtractCodeBlock() in frontend.js:
  1. Closed fenced block  (``` ... ```)
  2. Unclosed fenced block
  3. ANSI escape sequences (terminal / agent output)
  4. Structural Python heuristic
"""

import re

# ── Structural Python heuristics ─────────────────────────────────────────────

# Patterns that strongly suggest a line is Python source
_PY_STRUCTURAL = [
    re.compile(r'^\s*(def |class |import |from .+ import|async def |@\w)'),
    re.compile(r'^\s*[\w.\[\]]+\s*=\s*[^=]'),           # assignment
    re.compile(r'^\s+(if |elif |else:|for |while |try:|except|return |yield |pass$|break$)'),
    re.compile(r':\s*$'),                                 # line ending with colon
]

# Minimum structural lines and density to qualify as Python
_PY_MIN_STRUCTURAL = 2
_PY_MIN_DENSITY    = 0.40   # fraction of non-blank lines that look structural

# Minimum length of an unclosed-fence body to bother rendering
_UNCLOSED_MIN_LEN = 30

# ANSI escape pattern
_ANSI_RE = re.compile(r'\x1b\[')


def is_python_code(text: str) -> bool:
    """
    Return True if *text* looks like Python / code (no fences needed).

    Uses the same structural heuristic as _avExtractCodeBlock() in the JS.
    Does NOT match fenced blocks — use extract_code_block() for those.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    if _ANSI_RE.search(text):
        return True
    structural = sum(
        1 for l in lines
        if any(p.search(l) for p in _PY_STRUCTURAL)
    )
    return structural >= _PY_MIN_STRUCTURAL and (structural / len(lines)) > _PY_MIN_DENSITY


def extract_code_block(text: str) -> dict | None:
    """
    Attempt to extract a code block from *text*.

    Returns a dict  ``{"code": str, "type": "code"}``  or ``None``.

    Detection order (mirrors frontend _avExtractCodeBlock):
      1. Closed fenced block  ```[python|py]  ...  ```
      2. Unclosed fenced block (body > _UNCLOSED_MIN_LEN chars)
      3. ANSI escape sequences present → whole text is code
      4. Structural Python heuristic
    """
    # 1. Closed fenced block
    m = re.search(r'```(?:python|py)?\s*\n?([\s\S]*?)```', text, re.IGNORECASE)
    if m:
        return {"code": m.group(1).strip(), "type": "code"}

    # 2. Unclosed fenced block
    m = re.search(r'```(?:python|py)?\s*\n?([\s\S]+)$', text, re.IGNORECASE)
    if m and len(m.group(1).strip()) > _UNCLOSED_MIN_LEN:
        return {"code": m.group(1).strip(), "type": "code"}

    # 3. ANSI terminal output
    if _ANSI_RE.search(text):
        return {"code": text, "type": "code"}

    # 4. Structural heuristic
    if is_python_code(text):
        return {"code": text, "type": "code"}

    return None


def split_code_from_text(text: str) -> tuple[str, str | None]:
    """
    Split a reply into ``(prose, code_block)``.

    If a fenced code block is found, returns the surrounding prose and the
    extracted code separately. Otherwise returns ``(text, None)``.

    Useful for routing prose to TTS while sending code to the avatar renderer.
    """
    result = extract_code_block(text)
    if result is None:
        return text, None

    # Only strip fenced blocks from prose — heuristic / ANSI matches use whole text
    prose = re.sub(
        r'```(?:python|py)?\s*\n?[\s\S]*?```', '', text,
        flags=re.IGNORECASE,
    ).strip()

    # If nothing was stripped (heuristic match) the whole text is code
    if prose == text.strip():
        return "", result["code"]

    return prose, result["code"]


def strip_code_for_tts(text: str) -> str:
    """
    Remove code blocks and ANSI sequences from *text* before sending to TTS.

    Mirrors _stripCodeForTTS() in frontend.js:
      - Removes fenced code blocks entirely
      - Removes inline backtick code
      - Strips ANSI escape sequences
      - Collapses excess blank lines
    """
    # Fenced blocks
    out = re.sub(r'```[\s\S]*?```', ' ', text)
    # Inline code
    out = re.sub(r'`[^`]+`', ' ', out)
    # ANSI escapes
    out = re.sub(r'\x1b\[[0-9;]*m', '', out)
    # Collapse excess whitespace
    out = re.sub(r'\n{3,}', '\n\n', out).strip()
    return out
