"""
core/rag.py — Conversation-file RAG (Retrieval-Augmented Generation).

Two retrieval tiers:
  1. Keyword (BM25-like word-overlap) — always available, zero dependencies.
  2. Semantic (FAISS + sentence-transformers) — activated automatically when
     the optional dependencies are present, or explicitly via use_semantic=True.

Usage:
    rag = RAGMemory()
    rag.load("rag/myconv.txt")
    rag.enabled = True

    # In LLMCaller.memory_inject_fn:
    def inject(system_prompt, user_text):
        return rag.inject(system_prompt, user_text)

    llm.memory_inject_fn = inject
"""

import os
import re
from collections import OrderedDict
from hashlib import sha256
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# QUERY EMBEDDING CACHE
# ─────────────────────────────────────────────────────────────────────────────
#
# Caches the encoded vector for each unique query string so that repeated or
# similar questions don't re-run the sentence-transformers model (20–80 ms on
# CPU per call).  Chunk embeddings are already persisted in the FAISS index and
# are never re-encoded after load, so only the query-side hot path needs this.
#
# Module-level so the cache persists across requests for the process lifetime.
# If multiple workers are ever added, move _embed_cache into the RAGMemory
# instance instead.

_EMBED_CACHE_MAX: int = 256
_embed_cache: OrderedDict = OrderedDict()


def _cached_encode(model, text: str):
    """Return model.encode([text]) with an LRU cache keyed on a hash of text."""
    key = sha256(text.encode()).hexdigest()[:16]
    if key in _embed_cache:
        _embed_cache.move_to_end(key)          # LRU touch — mark as recently used
        return _embed_cache[key]
    vec = model.encode([text], convert_to_numpy=True)
    _embed_cache[key] = vec
    if len(_embed_cache) > _EMBED_CACHE_MAX:
        _embed_cache.popitem(last=False)        # evict least-recently-used entry
    return vec


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL SEMANTIC SEARCH
# ─────────────────────────────────────────────────────────────────────────────

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss as _faiss
    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False

# Global lock — prevents concurrent FAISS index builds (e.g. conv_rag flush
# racing a TTS/LLM call on the same CUDA device).  encode() during retrieval
# is read-only so it does NOT need this lock.
import threading as _threading
_index_build_lock = _threading.Lock()


class RAGMemory:
    """
    Load a plain-text conversation file, chunk it, and retrieve relevant
    passages to inject into an LLM system prompt.
    """

    def __init__(self, use_semantic: bool = False):
        self.raw_text           = ""
        self.cleaned_text       = ""
        self.chunks: list[str]  = []
        self.character_profile  = ""
        self.enabled            = False

        # Semantic search state
        self._use_semantic   = use_semantic and _SEMANTIC_AVAILABLE
        self._embedder       = None
        self._index          = None   # FAISS index
        self._chunk_vecs     = None   # numpy array of chunk embeddings
        # Device for sentence-transformers — False=CPU (safe default), True=CUDA
        self.use_cuda        = False

        # Context-mode overrides — set by inject() from the active ContextMode.
        # These are the fallback defaults used when no mode is passed.
        self._default_max_rag_chunks = 2
        self._default_rag_chunk_chars = 350
        self._default_min_rag_score   = 2
        self._default_min_rag_cosine  = 0.38
        self._default_profile_chars   = 400

    # ── load ─────────────────────────────────────────────────────────────────

    def load(self, path: str):
        """Load a plain-text conversation file and build search indices."""
        self.load_multiple([path])

    def load_multiple(self, paths: list[str]):
        """Load and merge one or more plain-text conversation files."""
        all_chunks: list[str] = []
        profile_parts: list[str] = []

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            # Store raw_text as the last loaded file (used internally by _chunk)
            self.raw_text = raw
            cleaned = self._clean(raw)
            all_chunks.extend(self._chunk(cleaned))
            profile_parts.append(self._build_profile(cleaned))

        self.cleaned_text      = ""          # not meaningful across multiple files
        self.chunks            = all_chunks
        # Cap at 1500 chars here; inject() will apply the tighter per-mode cap
        self.character_profile = "\n".join(profile_parts)[:1500]

        if self._use_semantic and _SEMANTIC_AVAILABLE:
            self._build_semantic_index()

        labels = ", ".join(os.path.basename(p) for p in paths)
        print(f"[RAG] Loaded {len(paths)} file(s) ({labels}) — "
              f"{len(self.chunks)} chunks, "
              f"profile: {len(self.character_profile)} chars, "
              f"semantic: {self._use_semantic and self._index is not None}")

    def clear(self):
        self.raw_text          = ""
        self.cleaned_text      = ""
        self.chunks            = []
        self.character_profile = ""
        self.enabled           = False
        self._index            = None
        self._chunk_vecs       = None
        print("[RAG] Cleared.")

    # ── text processing ───────────────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        """
        Clean a conversation export into plain dialogue turns.
        Strips: tool response blobs, image URLs, code blocks, markdown,
        conversation header lines, and turn-number prefixes.
        """
        lines = text.splitlines()
        out = []
        in_code_block = False

        for line in lines:
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                continue

            line = line.strip()
            if not line:
                continue

            if line.startswith("Conversation conv_") or line.startswith("Conversation "):
                continue

            line = re.sub(r"^\d+\.\s*", "", line).strip()

            if line.startswith("[{'type'") or line.startswith("[{\"type\""):
                continue

            line = re.sub(r'!?\[.*?\]\(https?://\S+?\)', '', line)
            line = re.sub(r'https?://\S+', '', line)
            line = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', line)
            line = re.sub(r'_{1,2}(.*?)_{1,2}', r'\1', line)

            # Keep role tags intact for _build_profile — strip them in _chunk
            line = line.strip()
            if len(line) > 10:
                out.append(line)

        return "\n".join(out)

    def _chunk(self, text: str, chunk_size: int = 800) -> list[str]:
        """
        Chunk by user+assistant turn pairs extracted from raw text.
        Each chunk is one exchange. Falls back to word-count chunking.
        Handles both numbered format (42. [user]) and plain ([USER]\\ncontent).
        """
        pairs = []

        # Try numbered format first: "42. [user] content"
        turn_re = re.compile(r"^\s*\d+\.\s*\[(user|assistant)\]\s*(.*)", re.IGNORECASE)
        current_user = ""
        for line in self.raw_text.splitlines():
            m = turn_re.match(line.strip())
            if not m:
                continue
            role, content = m.group(1).lower(), m.group(2).strip()
            if content.startswith("[{'type'") or content.startswith("[{\"type\""):
                t = re.search(r"'text':\s*'(.*?)'(?:,|\})", content)
                content = t.group(1)[:300] if t else ""
            content = re.sub(r'!?\[.*?\]\(https?://\S+?\)', '[image]', content)
            content = re.sub(r'https?://\S+', '', content)
            content = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', content).strip()
            if not content:
                continue
            if role == "user":
                current_user = content
            elif role == "assistant" and current_user:
                pairs.append(f"[user] {current_user}\n[assistant] {content[:500]}")
                current_user = ""

        if pairs:
            return pairs

        # Try plain block format: "[USER]" on one line, content on next
        plain_re = re.compile(r"^\[(USER|ASSISTANT)\]$", re.IGNORECASE)
        blocks, current_role, current_content = [], None, []
        for line in self.raw_text.splitlines():
            m = plain_re.match(line.strip())
            if m:
                if current_role and current_content:
                    blocks.append((current_role.lower(), " ".join(current_content).strip()))
                current_role = m.group(1)
                current_content = []
            elif current_role:
                current_content.append(line.strip())
        if current_role and current_content:
            blocks.append((current_role.lower(), " ".join(current_content).strip()))

        current_user = ""
        for role, content in blocks:
            content = re.sub(r'\*{1,3}(.*?)\*{1,3}', r'\1', content).strip()
            if not content:
                continue
            if role == "user":
                current_user = content
            elif role == "assistant" and current_user:
                pairs.append(f"[user] {current_user}\n[assistant] {content[:500]}")
                current_user = ""

        if pairs:
            return pairs

        # Fallback: word-count chunks
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    def _build_profile(self, text: str, max_chars: int = 1500) -> str:
        """Collect assistant lines to form a character voice profile.
        Works with cleaned text that still has role tags, or raw text."""
        lines = []
        # Try cleaned text with role tags still present
        for line in text.splitlines():
            if re.search(r"\[assistant\]", line, re.IGNORECASE):
                content = re.sub(r"\[assistant\]\s*", "", line, flags=re.IGNORECASE).strip()
                if len(content) < 40:
                    continue
                skip_phrases = [
                    "first chat", "first conversation", "first time",
                    "look who it is", "what's up", "well well", "decided to drop",
                    "oh look who", "hey there", "oh hey", "well hello",
                    "nice to meet", "great to meet", "good to meet",
                ]
                if any(p in content.lower() for p in skip_phrases):
                    continue
                lines.append(content)

        # Fallback: scan raw text for assistant blocks
        if not lines:
            in_assistant = False
            for line in self.raw_text.splitlines():
                if re.match(r"^\[ASSISTANT\]$", line.strip(), re.IGNORECASE):
                    in_assistant = True
                    continue
                if re.match(r"^\[USER\]$", line.strip(), re.IGNORECASE):
                    in_assistant = False
                    continue
                if in_assistant and len(line.strip()) >= 40:
                    lines.append(line.strip())

        return "\n".join(lines)[:max_chars]

    # ── semantic index ────────────────────────────────────────────────────────

    def _build_semantic_index(self):
        if not _SEMANTIC_AVAILABLE or not self.chunks:
            return
        with _index_build_lock:
            try:
                if self._embedder is None:
                    device = "cuda" if self.use_cuda else "cpu"
                    print(f"[RAG] Loading sentence-transformers model ({device})…")
                    self._embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
                vecs = self._embedder.encode(self.chunks, convert_to_numpy=True,
                                             show_progress_bar=False)
                vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
                dim  = vecs.shape[1]
                idx  = _faiss.IndexFlatIP(dim)
                idx.add(vecs.astype("float32"))
                self._index      = idx
                self._chunk_vecs = vecs
            except Exception as e:
                print(f"[RAG] Semantic index build failed: {e}")
                self._index = None

    # ── retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 3,
                 min_score: int = 1, min_cosine: float = 0.0) -> list[str]:
        """Return the most relevant chunks for query.

        min_score  : minimum keyword-overlap word count (keyword mode).
        min_cosine : minimum cosine similarity threshold (semantic mode).
        Both default to permissive values so existing callers are unaffected.
        """
        if not self.chunks:
            return []
        if self._use_semantic and self._index is not None:
            return self._retrieve_semantic(query, top_k, min_cosine=min_cosine)
        return self._retrieve_keyword(query, top_k, min_score=min_score)

    def _retrieve_keyword(self, query: str, top_k: int,
                          min_score: int = 1) -> list[str]:
        query_words = set(query.lower().split())
        scored = []
        for chunk in self.chunks:
            score = len(query_words & set(chunk.lower().split()))
            scored.append((score, chunk))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [c for s, c in scored[:top_k] if s >= min_score]

    def _retrieve_semantic(self, query: str, top_k: int,
                           min_cosine: float = 0.0) -> list[str]:
        try:
            import numpy as np
            q = _cached_encode(self._embedder, query)
            q = q / (np.linalg.norm(q) + 1e-9)
            k = min(top_k, len(self.chunks))
            scores, idxs = self._index.search(q.astype("float32"), k)
            results = []
            for score, idx in zip(scores[0], idxs[0]):
                if 0 <= idx < len(self.chunks) and float(score) >= min_cosine:
                    results.append(self.chunks[idx])
            return results
        except Exception as e:
            print(f"[RAG] Semantic retrieval error: {e}")
            return self._retrieve_keyword(query, top_k)

    # ── injection helper ──────────────────────────────────────────────────────

    def inject(self, system_prompt: str, user_text: str,
               mode=None) -> str:
        """Inject character profile + relevant past interactions into the prompt.

        mode : optional ContextMode instance.  When provided its per-slot caps
               and relevance floors are used.  Falls back to safe instance
               defaults so existing callers with no mode argument are unaffected.
        """
        if not self.enabled:
            return system_prompt

        # Resolve limits — prefer live mode values, fall back to instance defaults
        if mode is not None:
            profile_chars   = mode.profile_chars
            max_chunks      = mode.max_rag_chunks
            chunk_chars     = mode.rag_chunk_chars
            min_score       = mode.min_rag_score
            min_cosine      = mode.min_rag_cosine
            budget          = mode.prompt_budget_chars
        else:
            profile_chars   = self._default_profile_chars
            max_chunks      = self._default_max_rag_chunks
            chunk_chars     = self._default_rag_chunk_chars
            min_score       = self._default_min_rag_score
            min_cosine      = self._default_min_rag_cosine
            budget          = 9999  # no hard cap when running without a mode

        injection = ""

        # ── Slot 1: character profile ─────────────────────────────────────────
        if self.character_profile and profile_chars > 0:
            profile = self.character_profile[:profile_chars]
            injection += "\n\n[Character Profile]\n" + profile
            budget -= len(profile)

        # ── Slot 2: relevant past interactions ────────────────────────────────
        if max_chunks > 0 and budget > 100:
            relevant = self.retrieve(
                user_text, top_k=max_chunks,
                min_score=min_score, min_cosine=min_cosine,
            )
            rag_parts = []
            for chunk in relevant:
                trimmed = chunk[:chunk_chars]
                if budget - len(trimmed) < 100:
                    break
                rag_parts.append(trimmed)
                budget -= len(trimmed)
            if rag_parts:
                injection += "\n\n[Relevant Past Interactions]\n" + "\n\n".join(rag_parts)

        total_injected = len(injection)
        print(f"[RAG] inject: profile={profile_chars}  max_chunks={max_chunks}  "
              f"injected={total_injected}  budget_remaining={budget}")

        return system_prompt + injection

    # ── status ────────────────────────────────────────────────────────────────

    @property
    def status(self) -> dict:
        return {
            "enabled":  self.enabled,
            "chunks":   len(self.chunks),
            "semantic": self._use_semantic and self._index is not None,
        }
