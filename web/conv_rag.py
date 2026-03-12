"""
web/conv_rag.py — Automatic conversation RAG manager for Ecko.

ConvRAG handles per-character rolling conversation context:
  - Appends flushed turns to  rag/<character>_conversation.txt
  - Reloads the RAG index after each flush
  - Triggers automatically after assistant replies once turn count
    reaches the configured threshold
  - The frontend keeps its own bubble list so the chat UI is unaffected

Usage (wired in ecko_web.py):
    conv_rag = ConvRAG(rag_dir=RAG_DIR)
    conv_rag.set_character("maya")          # called on character load
    conv_rag.maybe_flush(session, rag)      # called after each assistant reply
"""

import os
import threading
from datetime import datetime


class ConvRAG:
    """Manages automatic flushing of old chat turns into a character RAG file."""

    DEFAULT_THRESHOLD  = 20   # total messages (10 exchanges) before a flush
    DEFAULT_KEEP       = 6    # messages to keep in live history after flush
    MAX_ARCHIVE_LINES  = 600  # ~60-80 exchanges; older lines are rotated out

    def __init__(self, rag_dir: str):
        self.rag_dir   = rag_dir
        self.character = "global"
        self.enabled   = False
        self.threshold = self.DEFAULT_THRESHOLD
        self.keep      = self.DEFAULT_KEEP
        self._lock     = threading.Lock()
        self._semantic = False   # mirrors rag._use_semantic at flush time

    # ── public API ────────────────────────────────────────────────────────────

    def set_character(self, character: str) -> None:
        self.character = character

    @property
    def filename(self) -> str:
        return f"{self.character}_conversation.txt"

    @property
    def filepath(self) -> str:
        return os.path.join(self.rag_dir, self.filename)

    def status(self) -> dict:
        exists = os.path.exists(self.filepath)
        size   = os.path.getsize(self.filepath) if exists else 0
        return {
            "enabled":   self.enabled,
            "threshold": self.threshold,
            "keep":      self.keep,
            "character": self.character,
            "filename":  self.filename,
            "exists":    exists,
            "size_bytes": size,
        }

    def maybe_flush(self, session, rag) -> bool:
        """
        Called after each assistant reply.
        If history is at or over threshold, flush the oldest turns.
        Returns True if a flush occurred.
        """
        if not self.enabled:
            return False
        if len(session.chat_history) < self.threshold:
            return False

        with self._lock:
            # Re-check inside lock in case another thread already flushed
            if len(session.chat_history) < self.threshold:
                return False
            # Conv RAG always uses semantic retrieval if available;
            # snapshot from rag object but default to True
            self._semantic = True
            self._flush(session, rag)
            return True

    def clear_file(self) -> None:
        """Delete the conversation RAG file for the current character."""
        try:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
                print(f"[CONV RAG] Cleared {self.filename}")
        except Exception as e:
            print(f"[CONV RAG] Clear error: {e}")

    # ── internals ─────────────────────────────────────────────────────────────

    def _flush(self, session, rag) -> None:
        """
        Move all but the last `keep` messages out of live history and into
        the conversation RAG file, then reload the RAG index.
        """
        history = session.chat_history
        n_flush = len(history) - self.keep
        if n_flush <= 0:
            return

        to_flush = history[:n_flush]
        session.chat_history = history[n_flush:]

        # Serialise to plain text — same format as /rag/save
        lines = []
        for msg in to_flush:
            role    = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            lines.append(f"[{role}]\n{content}\n")

        block = (
            f"\n[--- auto-flush {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC ---]\n"
            + "\n".join(lines)
        )

        os.makedirs(self.rag_dir, exist_ok=True)
        try:
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(block)
            print(
                f"[CONV RAG] Flushed {len(to_flush)} turns → {self.filename} "
                f"(kept {len(session.chat_history)})"
            )
        except Exception as e:
            print(f"[CONV RAG] Write error: {e}")
            # Restore history if write failed — don't silently lose turns
            session.chat_history = to_flush + session.chat_history
            return

        # Rotate archive — trim to MAX_ARCHIVE_LINES so the file never grows
        # unboundedly.  Oldest lines are dropped; recent history is preserved.
        self._trim_archive()

        # Reload RAG so the flushed turns are immediately searchable
        self._reload_rag(rag)

    def _trim_archive(self) -> None:
        """Trim the archive file to MAX_ARCHIVE_LINES, dropping the oldest lines."""
        try:
            if not os.path.exists(self.filepath):
                return
            with open(self.filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if len(lines) <= self.MAX_ARCHIVE_LINES:
                return
            trimmed = lines[-self.MAX_ARCHIVE_LINES:]
            with open(self.filepath, "w", encoding="utf-8") as f:
                f.writelines(trimmed)
            print(
                f"[CONV RAG] Archive trimmed: {len(lines)} → {len(trimmed)} lines "
                f"({self.filename})"
            )
        except Exception as e:
            print(f"[CONV RAG] Trim error: {e}")

    def _reload_rag(self, rag) -> None:
        """Reload the conversation RAG file into the active RAG index."""
        try:
            if not os.path.exists(self.filepath):
                return

            # If extra RAG files are also loaded, we need to keep them.
            # RAGMemory.load_multiple replaces all chunks, so collect existing
            # extra paths (anything that isn't our conv file) and merge.
            extra_paths = []
            if hasattr(rag, "_loaded_paths"):
                extra_paths = [
                    p for p in rag._loaded_paths
                    if os.path.basename(p) != self.filename
                ]

            all_paths = [self.filepath] + extra_paths

            # Preserve semantic mode — don't downgrade if it was already on
            if hasattr(rag, "_use_semantic") and self._semantic:
                rag._use_semantic = True

            if hasattr(rag, "load_multiple"):
                rag.load_multiple(all_paths)
            else:
                rag.load(self.filepath)

            rag.enabled = True
            print(f"[CONV RAG] RAG index reloaded ({len(rag.chunks)} chunks)")
        except Exception as e:
            print(f"[CONV RAG] Reload error: {e}")
