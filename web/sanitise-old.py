"""
web/sanitise.py — Reply sanitisation helpers for Ecko.

Strips injected context that the LLM accidentally echoes back.
"""

import re as _re


def strip_leaked_context(text: str) -> str:
    """
    Remove injected context that the LLM accidentally echoed back.

    Two known leak patterns:
      1. A bare JSON array/object block (agent context or memory JSON).
         e.g.  [ {"content": "...", "category": "preference", ...} ]
      2. A completely empty reply that is just "[]".

    Strips any leading/trailing JSON block and leaves only the prose reply.
    Returns "..." rather than empty if nothing remains after stripping.
    """
    if not text:
        return "..."

    stripped = text.strip()

    if stripped in ("[]", "{}", "[ ]"):
        print("[CHAT] Sanitised: reply was empty JSON placeholder")
        return "..."

    def _find_json_block_bounds(s: str):
        """Return (start, end) of the first top-level JSON array/object, or None."""
        for b_open, b_close in (("[", "]"), ("{", "}")):
            idx = s.find(b_open)
            if idx == -1:
                continue
            depth = 0
            in_str = False
            escape = False
            for i, ch in enumerate(s[idx:], idx):
                if escape:
                    escape = False; continue
                if ch == "\\" and in_str:
                    escape = True; continue
                if ch == '"':
                    in_str = not in_str; continue
                if in_str:
                    continue
                if ch == b_open:
                    depth += 1
                elif ch == b_close:
                    depth -= 1
                    if depth == 0:
                        return (idx, i)
        return None

    bounds = _find_json_block_bounds(stripped)
    if bounds is None:
        return stripped

    start, end = bounds
    before = stripped[:start].strip()
    after  = stripped[end + 1:].strip()

    if start <= 30 and after:
        print("[CHAT] Sanitised: stripped leaked JSON context from reply start")
        return after
    if (len(stripped) - end - 1) <= 30 and before:
        print("[CHAT] Sanitised: stripped leaked JSON context from reply end")
        return before

    return stripped
