"""
web/sanitise.py — Reply sanitisation helpers for Ecko.

Strips injected context that the LLM accidentally echoes back.
"""

import re as _re


# ── Injection block headers used by rag.inject() and memory.make_inject_fn() ──
# Any line that starts with one of these signals the start of a leaked block.
_BLOCK_HEADERS = (
    "[Character Profile]",
    "[Relevant Past Interactions]",
    "[Memory]",
    "[RECENT CONVERSATION]",
    "[System:",
    "[---",             # conv-rag auto-flush markers
    "[WEB SEARCH",      # web search results block
    "[END WEB SEARCH",  # closing tag
)

# Category labels emitted by memory inject: [FACT], [PREFERENCE], etc.
_CATEGORY_RE = _re.compile(
    r'^\[(FACT|PREFERENCE|EMOTION|RELATIONSHIP|TOPIC|EVENT|OBJECTIVE|'
    r'CONFUSION|RECENT CONVERSATION)\]',
    _re.IGNORECASE,
)

# Inline context tags that can appear mid-sentence after real reply content.
# Truncate the reply at the first match and keep only what precedes it.
_INLINE_CONTEXT_RE = _re.compile(
    r'\[(?:'
    r'END OF [A-Z ]{1,30}'          # [END OF MEMORY], [END OF CONTEXT], etc.
    r'|BEGIN(?:\s+OF)?\s+[A-Z ]+'   # [BEGIN MEMORY], [BEGIN OF CONTEXT], etc.
    r'|START(?:\s+OF)?\s+[A-Z ]+'   # [START OF ...]
    r'|You are [A-Z]'               # [You are Seraphina...]
    r'|SYSTEM\b'                    # [SYSTEM]
    r'|INST\b'                      # [INST]
    r'|\/INST\b'                    # [/INST]
    r'|THINKING\b'                  # [THINKING] reasoning token leak
    r'|SEARCH\b'                    # [SEARCH] model simulating tool call
    r'|SEARCH RESULTS'              # [SEARCH RESULTS]
    r')',
    _re.IGNORECASE,
)


def strip_leaked_context(text: str) -> str:
    """
    Remove injected context that the LLM accidentally echoed back.

    Handles three leak patterns:
      1. Leaked JSON array/object blocks (agent context or memory JSON).
      2. Injection block headers ([Character Profile], [Memory], etc.)
         and everything that follows them on subsequent lines.
      3. Inline [CATEGORY] label lines echoed mid-reply by small models.

    Returns "..." rather than empty if nothing remains after stripping.
    """
    if not text:
        return "..."

    stripped = text.strip()

    if stripped in ("[]", "{}", "[ ]"):
        print("[CHAT] Sanitised: reply was empty JSON placeholder")
        return "..."

    # ── Pre-pass: strip search pre-fill prefix if model echoed it ─────────────
    # When search results are injected as "Here's what I found: <results>\n\nAssistant:"
    # the model sometimes starts its reply by repeating that prefix.
    import re as _re2
    _prefill_re = _re2.compile(
        r"^Here'?s what I found[:\s]+.{10,800}?\n{1,2}(?=\S)",
        _re2.DOTALL | _re2.IGNORECASE,
    )
    prefill_match = _prefill_re.match(stripped)
    if prefill_match:
        stripped = stripped[prefill_match.end():].strip()
        print("[CHAT] Sanitised: stripped echoed search pre-fill prefix")

    # ── Pass 1: strip injection block headers and everything after them ────────
    # Split into lines, find the first line that looks like a block header or
    # a [CATEGORY] label, and truncate there.
    lines = stripped.splitlines()
    clean_lines = []
    found_leak = False
    for line in lines:
        stripped_line = line.strip()
        if any(stripped_line.startswith(h) for h in _BLOCK_HEADERS):
            found_leak = True
            break
        if _CATEGORY_RE.match(stripped_line):
            found_leak = True
            break
        clean_lines.append(line)

    if found_leak:
        result = "\n".join(clean_lines).strip()
        print(f"[CHAT] Sanitised: stripped leaked injection block from reply")
        stripped = result if result else "..."
        if stripped == "...":
            return "..."

    # ── Pass 2: strip leaked JSON blocks ──────────────────────────────────────
    def _find_json_block_bounds(s: str):
        """Return (start, end) of the first top-level JSON array/object, or None."""
        for b_open, b_close in ("[", "]"), ("{", "}"):
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
    if bounds is not None:
        start, end = bounds
        before = stripped[:start].strip()
        after  = stripped[end + 1:].strip()

        if start <= 30 and after:
            print("[CHAT] Sanitised: stripped leaked JSON context from reply start")
            stripped = after
        elif (len(stripped) - end - 1) <= 30 and before:
            print("[CHAT] Sanitised: stripped leaked JSON context from reply end")
            stripped = before

    # ── Pass 3: truncate at inline context tags ────────────────────────────────
    # Runs unconditionally — Pass 2 may leave the text intact if the JSON bounds
    # don't match the trim conditions, and [SYSTEM] can appear after valid content
    # on the same line (the model echoes the injected block mid-sentence).
    inline_match = _INLINE_CONTEXT_RE.search(stripped)
    if inline_match and inline_match.start() > 0:
        before_tag = stripped[:inline_match.start()].strip()
        if before_tag:
            print("[CHAT] Sanitised: stripped inline context tag from reply")
            stripped = before_tag

    # ── Pass 4: strip leaked code fences ──────────────────────────────────────
    # Only strip if the fence is *unclosed* — a closing ``` means it's a
    # legitimate fenced block (code, ASCII art, haiku, etc.) that the model
    # intentionally produced.  An unclosed fence almost always means the model
    # started echoing back an injected context block mid-reply.
    fence_match = _re.search(r'\n```', stripped)
    if fence_match and fence_match.start() > 0:
        after_open = stripped[fence_match.end():]
        has_close = _re.search(r'\n```', after_open) is not None
        if not has_close:
            before_fence = stripped[:fence_match.start()].strip()
            if before_fence:
                print("[CHAT] Sanitised: stripped unclosed code fence from reply")
                return before_fence
    return stripped
