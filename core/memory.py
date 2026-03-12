"""
core/memory.py — Structured memory layer.

After each conversation turn an LLM call (small, fast) extracts structured
memory entries from the exchange.  Entries are scored on:
  - recency          (decays with time)
  - hit frequency    (boosted each time a memory is retrieved)
  - semantic/keyword relevance to the current query

Memories are stored per-character in JSON (+ optional FAISS index for semantic
retrieval).  Entries below a threshold are archived rather than deleted so they
remain recoverable.

Entry schema:
{
  "id":           "mem_abc123",
  "content":      "User dislikes filler phrases",
  "category":     "preference",   # preference | fact | emotion | relationship | topic | event
  "score":        0.82,           # composite 0..1
  "hits":         3,
  "created":      "2026-01-01T12:00:00",
  "last_accessed": "2026-01-15T09:00:00",
  "character":    "aria",
  "global":       false           # if true, shared across all characters
}

Scoring formula:
  score = recency_weight * 0.3 + normalised_hits * 0.4 + relevance * 0.3
  recency_weight decays as: exp(-days_since_access / DECAY_DAYS)
"""

import json
import math
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import requests

from core.logger import log

CATEGORIES = ("preference", "fact", "emotion", "relationship", "topic", "event")
ARCHIVE_THRESHOLD = 0.10   # memories below this are archived
DECAY_DAYS        = 30.0   # half-life for recency decay

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _days_since(iso: str) -> float:
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
    except Exception as e:
        log.debug("[MEMORY] _days_since() could not parse %r, returning 0.0: %s", iso, e)
        return 0.0

def _recency_weight(last_accessed: str) -> float:
    days = _days_since(last_accessed)
    return math.exp(-days / DECAY_DAYS)

def _keyword_relevance(content: str, query: str) -> float:
    q_words = set(query.lower().split())
    c_words = set(content.lower().split())
    if not q_words:
        return 0.0
    return len(q_words & c_words) / len(q_words)


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY STORE
# ─────────────────────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Per-character memory store backed by a JSON file.

    Usage:
        store = MemoryStore("aria", memory_dir="memories")
        store.load()
        memories = store.retrieve("tell me about yourself", top_k=5)
        store.add_entry(content="User prefers short replies", category="preference")
        store.save()

    Wire into the LLM layer:
        llm.memory_inject_fn = store.make_inject_fn(llm.memory_inject_fn)
    """

    def __init__(self, character: str = "global", memory_dir: str = "memories",
                 global_dir: str = "memories"):
        self.character   = character
        self._dir        = memory_dir
        self._global_dir = global_dir
        self.entries:  list[dict] = []
        self.archived: list[dict] = []
        self.enabled = False

    # ── paths ─────────────────────────────────────────────────────────────────

    def _path(self, archive: bool = False) -> str:
        os.makedirs(self._dir, exist_ok=True)
        suffix = "_archive" if archive else ""
        return os.path.join(self._dir, f"{self.character}{suffix}.json")

    def _global_path(self, archive: bool = False) -> str:
        os.makedirs(self._global_dir, exist_ok=True)
        suffix = "_archive" if archive else ""
        return os.path.join(self._global_dir, f"global{suffix}.json")

    # ── persistence ───────────────────────────────────────────────────────────

    def load(self):
        for attr, path in [("entries", self._path()), ("archived", self._path(True))]:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        setattr(self, attr, json.load(f))
            except Exception as e:
                print(f"[MEMORY] Load error ({path}): {e}")
        # Load and merge global memories
        try:
            gp = self._global_path()
            if os.path.exists(gp):
                with open(gp, "r", encoding="utf-8") as f:
                    globals_ = json.load(f)
                # Prepend global entries — deduplicate by id
                existing_ids = {e["id"] for e in self.entries}
                for g in globals_:
                    if g["id"] not in existing_ids:
                        self.entries.append(g)
        except Exception as e:
            print(f"[MEMORY] Global load error: {e}")
        print(f"[MEMORY] Loaded {len(self.entries)} active, "
              f"{len(self.archived)} archived for '{self.character}'")

    def save(self):
        """Save active and archived entries (global memories are saved separately)."""
        active_local   = [e for e in self.entries  if not e.get("global")]
        archived_local = [e for e in self.archived if not e.get("global")]
        global_entries = [e for e in self.entries  if e.get("global")]

        for data, path in [
            (active_local,   self._path()),
            (archived_local, self._path(True)),
            (global_entries, self._global_path()),
        ]:
            try:
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.chmod(tmp, 0o600)
                os.replace(tmp, path)
            except Exception as e:
                print(f"[MEMORY] Save error ({path}): {e}")

    # ── scoring ───────────────────────────────────────────────────────────────

    def _compute_score(self, entry: dict, query: str = "", max_hits: int = 1) -> float:
        recency   = _recency_weight(entry.get("last_accessed", entry.get("created", _now_iso())))
        norm_hits = min(entry.get("hits", 0) / max(max_hits, 1), 1.0)
        relevance = _keyword_relevance(entry.get("content", ""), query) if query else 0.0
        return recency * 0.3 + norm_hits * 0.4 + relevance * 0.3

    def recompute_scores(self, query: str = ""):
        if not self.entries: return
        max_hits = max((e.get("hits", 0) for e in self.entries), default=1)
        for e in self.entries:
            e["score"] = round(self._compute_score(e, query, max_hits), 4)

    # ── decay & archive ───────────────────────────────────────────────────────

    def decay_and_archive(self):
        """Move low-scoring memories to archive. Call periodically (e.g. on session start)."""
        self.recompute_scores()
        keep, archive = [], []
        for e in self.entries:
            if e.get("score", 1.0) < ARCHIVE_THRESHOLD:
                archive.append(e)
            else:
                keep.append(e)
        if archive:
            print(f"[MEMORY] Archiving {len(archive)} low-score entries")
        self.archived.extend(archive)
        self.entries = keep

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top_k most relevant active memories, updating hit counters."""
        if not self.entries:
            return []
        max_hits = max((e.get("hits", 0) for e in self.entries), default=1)
        scored = []
        for e in self.entries:
            s = self._compute_score(e, query, max_hits)
            scored.append((s, e))
        scored.sort(reverse=True, key=lambda x: x[0])
        results = []
        now = _now_iso()
        for _, e in scored[:top_k]:
            e["hits"]          = e.get("hits", 0) + 1
            e["last_accessed"] = now
            e["score"]         = round(self._compute_score(e, query, max_hits), 4)
            results.append(e)
        return results

    # ── add entry ─────────────────────────────────────────────────────────────

    def add_entry(self, content: str, category: str = "fact",
                  score: float = 0.5, global_: bool = False) -> dict:
        """Manually add a memory entry."""
        entry = {
            "id":           "mem_" + uuid.uuid4().hex[:8],
            "content":      content,
            "category":     category if category in CATEGORIES else "fact",
            "score":        round(score, 4),
            "hits":         0,
            "created":      _now_iso(),
            "last_accessed": _now_iso(),
            "character":    self.character,
            "global":       global_,
        }
        self.entries.append(entry)
        return entry

    def update_entry(self, mem_id: str, **kwargs):
        """Update fields on an existing entry by ID."""
        for e in self.entries:
            if e["id"] == mem_id:
                e.update(kwargs)
                return True
        return False

    def delete_entry(self, mem_id: str) -> bool:
        """Hard-delete an entry by ID (active or archived)."""
        for lst in (self.entries, self.archived):
            for i, e in enumerate(lst):
                if e["id"] == mem_id:
                    lst.pop(i)
                    return True
        return False

    def clear_all(self):
        """Delete all active memory entries."""
        self.entries.clear()
        self.save()

    def promote_to_global(self, mem_id: str) -> bool:
        """Mark a memory as global so it applies across all characters."""
        for e in self.entries:
            if e["id"] == mem_id:
                e["global"] = True
                return True
        return False

    def restore_archived(self, mem_id: str) -> bool:
        """Move a memory from the archive back to active."""
        for i, e in enumerate(self.archived):
            if e["id"] == mem_id:
                self.entries.append(self.archived.pop(i))
                return True
        return False

    # ── injection helper ──────────────────────────────────────────────────────

    def make_inject_fn(self, upstream_fn=None, get_mode=None):
        """
        Return a memory_inject_fn compatible with LLMCaller.memory_inject_fn.
        Chains with an existing inject function (e.g. RAG) if provided.

        get_mode : optional callable() -> ContextMode.
                   When provided, max_memories and memory_chars are taken from
                   the active mode preset.  Falls back to safe hardcoded defaults
                   so existing callers with no get_mode are unaffected.
        """
        def inject(system_prompt: str, user_text: str) -> str:
            if upstream_fn:
                system_prompt = upstream_fn(system_prompt, user_text)
            if not self.enabled:
                return system_prompt

            # Resolve limits from active mode if available
            if callable(get_mode):
                try:
                    m = get_mode()
                    top_k      = m.max_memories
                    char_limit = m.memory_chars
                except Exception as e:
                    log.warning("[MEMORY] get_mode() failed, using defaults: %s", e)
                    top_k      = 3
                    char_limit = 120
            else:
                top_k      = 3
                char_limit = 120

            memories = self.retrieve(user_text, top_k=top_k)
            if not memories:
                return system_prompt

            lines = [
                f"[{e['category'].upper()}] {e['content'][:char_limit]}"
                for e in memories
            ]
            block = "\n\n[Memory]\n" + "\n".join(lines)
            print(f"[MEMORY] inject: top_k={top_k}  char_limit={char_limit}  "
                  f"entries={len(memories)}  block_len={len(block)}")
            return system_prompt + block
        return inject

    # ── LLM extraction ────────────────────────────────────────────────────────

    def extract_from_turn(
        self,
        user_text: str,
        assistant_text: str,
        llm_caller,          # LLMCaller instance for extraction calls
        model_override: str = "",
        every_n_turns: int = 3,
        safety=None,         # optional SafetyLayer ref for concern_level callback
    ):
        """
        Ask the LLM to extract structured memories from a single exchange.
        Runs in a background thread — call as fire-and-forget after each turn.

        every_n_turns: only extract on every Nth turn to reduce LLM load.
        The extraction prompt asks for JSON; entries are parsed and added to
        self.entries.  Uses the same LLMCaller but with a separate system prompt
        so the character persona is not affected.
        If safety is provided, concern_level from the response is forwarded.
        """
        # Throttle: use an internal counter
        self._extraction_turn_counter = getattr(self, "_extraction_turn_counter", 0) + 1
        if self._extraction_turn_counter % every_n_turns != 0:
            return

        prompt = _EXTRACTION_PROMPT.format(
            user=user_text[:600],
            assistant=assistant_text[:600],
            categories=", ".join(CATEGORIES),
        )
        try:
            import threading
            def _run():
                try:
                    tmp = type(llm_caller)()
                    tmp.from_dict(llm_caller.to_dict())
                    tmp.max_reply_tokens = 400
                    tmp.memory_inject_fn = None
                    tmp.conv_id          = None
                    style = tmp._style

                    if style == "mistral_conv":
                        # The Mistral Agents API replies in-character — useless for JSON
                        # extraction.  Route to the standard Mistral chat completions
                        # endpoint instead, using the same API key / base URL.
                        # mistral-small-latest is cheap, fast, and JSON-capable.
                        tmp.provider_id    = "mistral_extract"   # not in registry — overridden below
                        tmp.agent_id       = ""
                        tmp.model          = model_override or "mistral-small-latest"
                        tmp.system_prompt  = _EXTRACTION_SYSTEM_PROMPT
                        # Call chat_openai directly — bypass LLMCaller.chat() dispatch
                        from core.llm.openai_compat import chat_openai as _chat_openai
                        user_msg = prompt
                        # Temporarily set style to openai so chat_openai is used
                        # (provider_id is not in registry so _style returns 'openai')
                        raw = _chat_openai(tmp, user_msg, [])
                    elif style in ("kobold", "openai"):
                        tmp.system_prompt = _EXTRACTION_SYSTEM_PROMPT
                        user_msg = "Extract memories from the exchange above. Output JSON only."
                        raw = tmp.chat(user_msg, [])
                    else:
                        tmp.system_prompt = ""
                        user_msg = prompt
                        raw = tmp.chat(user_msg, [])

                    if isinstance(raw, dict):
                        raw = raw.get("reply", "")
                    concern = self._parse_and_add(raw)
                    if safety is not None and concern > 0:
                        safety.record_score_delta(
                            float(concern),
                            label=f"Memory extractor: concern_level={concern}"
                        )
                    self.save()
                except Exception as e:
                    print(f"[MEMORY] Extraction error: {e}")
            t = threading.Thread(target=_run, daemon=True)
            t.start()
        except Exception as e:
            print(f"[MEMORY] extract_from_turn error: {e}")

    def _parse_and_add(self, raw: str) -> int:
        """Parse extraction response, add memories, return concern_level (0-3)."""
        import re, json as _json
        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Some LLMs (e.g. KoboldCPP) emit narrative text before or after the
        # JSON payload.  Try to extract the first JSON array or object using
        # a bracket-depth scan so we handle nested structures correctly.
        def _extract_json_block(text: str) -> str:
            for start_char, end_char in (("{", "}"), ("[", "]")):
                idx = text.find(start_char)
                if idx == -1:
                    continue
                depth = 0
                in_str = False
                escape = False
                for i, ch in enumerate(text[idx:], idx):
                    if escape:
                        escape = False
                        continue
                    if ch == "\\" and in_str:
                        escape = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == start_char:
                        depth += 1
                    elif ch == end_char:
                        depth -= 1
                        if depth == 0:
                            return text[idx:i + 1]
            return None

        block = _extract_json_block(raw)
        if block is None:
            print(f"[MEMORY] No JSON block found in extraction response, skipping.")
            return 0

        concern_level = 0
        try:
            parsed = _json.loads(block)

            # New format: {"memories": [...], "concern_level": N}
            if isinstance(parsed, dict) and "memories" in parsed:
                concern_level = int(parsed.get("concern_level", 0))
                items = parsed.get("memories", [])
            # Legacy format: plain array
            elif isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = [parsed]
            else:
                print(f"[MEMORY] Parse warning: unexpected type {type(parsed)}, skipping.")
                return 0

            added = 0
            for item in items:
                if not isinstance(item, dict): continue
                content = item.get("content", "").strip()
                if not content: continue
                category = item.get("category", "fact")
                score    = float(item.get("score", 0.5))
                global_  = bool(item.get("global", False))
                self.add_entry(content, category, score, global_)
                added += 1
            print(f"[MEMORY] Extracted {added} memories, concern_level={concern_level}")
            return concern_level
        except Exception as e:
            print(f"[MEMORY] Parse error: {e} | raw={repr(block[:200])}")
            return 0

    # ── viewer API ────────────────────────────────────────────────────────────

    def to_viewer_dict(self) -> dict:
        """Return a dict suitable for the memory viewer/editor UI."""
        return {
            "character": self.character,
            "enabled":   self.enabled,
            "entries":   sorted(self.entries,
                                key=lambda e: e.get("score", 0), reverse=True),
            "archived":  sorted(self.archived,
                                key=lambda e: e.get("score", 0), reverse=True),
            "stats": {
                "active":   len(self.entries),
                "archived": len(self.archived),
                "categories": {
                    cat: sum(1 for e in self.entries if e.get("category") == cat)
                    for cat in CATEGORIES
                },
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION PROMPT
# ─────────────────────────────────────────────────────────────────────────────

# Fixed system prompt used for kobold/openai extraction calls — keeps the
# extractor isolated from the live character's system_prompt.
_EXTRACTION_SYSTEM_PROMPT = (
    "You are a memory extraction assistant. "
    "Output only valid JSON matching the schema requested. No other text."
)

_EXTRACTION_PROMPT = """\
You are a memory extraction assistant. Given one exchange between a User and \
an Assistant, extract memorable facts, preferences, emotions, or events worth \
storing for future conversations.

Exchange:
User: {user}
Assistant: {assistant}

Return ONLY a JSON object with two keys:
  "memories": array of memory objects, each with:
    "content"   : concise single sentence describing what to remember
    "category"  : one of [{categories}]
    "score"     : importance 0.0-1.0 (0.9+ = critical, 0.5 = moderate, 0.1 = minor)
    "global"    : true if this applies across ALL characters (e.g. user's name, location)
  "concern_level": integer 0-3 rating how concerning the user's message is:
    0 = normal conversation
    1 = mildly sensitive topic (dark themes, edgy humour)
    2 = moderately concerning (boundary pushing, manipulative framing)
    3 = seriously concerning (harmful intent, explicit harmful requests)

Return {{"memories": [], "concern_level": 0}} if nothing to store and no concern.
No other text. No markdown fences.
"""
