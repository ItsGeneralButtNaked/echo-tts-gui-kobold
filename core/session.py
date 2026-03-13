"""
core/session.py — Shared session state: history, persistence, auto-continue.

Used by both ecko_web.py (server-global Session object) and indirectly by
ecko_desktop.py (desktop saves its own JSON but shares the same schema).
"""

import json
import os
import queue
import random
import threading
import struct
from typing import Optional

import requests

from core.llm import LLMCaller
from core.tts import TTSCaller

MAX_HISTORY = 20


# ─────────────────────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────────────────────

class Session:
    """
    Holds all mutable runtime state for one conversation context.

    The web app uses a single global Session (or per-IP in isolated mode).
    The desktop app constructs its own state but mirrors the same JSON schema
    so sessions can be ported between surfaces.
    """

    def __init__(self, session_file: str = ""):
        self.lock = threading.Lock()
        self.llm  = LLMCaller()
        self.tts  = TTSCaller()

        self.chat_history: list[dict] = []
        self.busy = False

        # Auto-continue
        self.auto_continue_enabled = True
        self.auto_continue_mode    = "standard"
        self._ac_timer: Optional[threading.Timer] = None
        self._ac_stop  = False

        # Sleep timer for auto-continue (mirrors initiative sleep settings)
        self.ac_sleep_timer_enabled = False
        self.ac_sleep_start = 23
        self.ac_sleep_end   = 8

        # Queue for server-sent auto-continue prompts (web only)
        self.ac_queue: queue.Queue = queue.Queue()

        self._session_file = session_file
        if session_file:
            self._load_persistent()

    # ── persistence ──────────────────────────────────────────────────────────

    def _load_persistent(self):
        if not self._session_file or not os.path.exists(self._session_file):
            return
        try:
            with open(self._session_file, "r") as f:
                data = json.load(f)
            self.llm.from_dict(data.get("llm", {}))
            self.tts.from_dict(data.get("tts", {}))
            self.auto_continue_enabled = data.get("auto_continue_enabled", True)
            self.auto_continue_mode    = data.get("auto_continue_mode", "standard")
            self.chat_history = data.get("chat_history", [])
            print(f"[SESSION] Restored — provider: {self.llm.provider_id}, "
                  f"tts: {self.tts.provider_id}, voice: {self.tts.voice}")
        except Exception as e:
            print(f"[SESSION] Load error (starting fresh): {e}")

    def save_persistent(self):
        if not self._session_file:
            return
        try:
            # Snapshot mutable state under lock before serialising — prevents
            # "dictionary changed size during iteration" when another thread
            # writes new keys to tts.extra concurrently (e.g. /settings POST).
            with self.lock:
                llm_data  = self.llm.to_dict()
                tts_data  = self.tts.to_dict()
                ac_en     = self.auto_continue_enabled
                ac_mode   = self.auto_continue_mode
                history   = list(self.chat_history[-MAX_HISTORY:])
            data = {
                "llm":  llm_data,
                "tts":  tts_data,
                "auto_continue_enabled": ac_en,
                "auto_continue_mode":    ac_mode,
                "chat_history": history,
            }
            # Write to a temp file then atomically replace, with owner-only perms
            tmp = self._session_file + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            try:
                os.chmod(tmp, 0o600)
            except (NotImplementedError, AttributeError):
                pass  # Windows — permissions not supported
            os.replace(tmp, self._session_file)
        except Exception as e:
            print(f"[SESSION] Save error: {e}")

    # ── LLM ──────────────────────────────────────────────────────────────────

    def call_llm(self, user_text: str):
        return self.llm.chat(user_text, self.chat_history)

    # ── TTS ──────────────────────────────────────────────────────────────────

    def stream_tts(self, text: str, cancel=None):
        """Yield audio bytes from the configured TTS provider."""
        yield from self.tts.stream(text, cancel=cancel)

    # ── auto-continue ────────────────────────────────────────────────────────

    def _ac_interval(self) -> int:
        if self.auto_continue_mode == "aggressive": return random.randint(8, 15)
        if self.auto_continue_mode == "relaxed":    return random.randint(45, 75)
        return random.randint(25, 50)

    def _ac_in_sleep_window(self) -> bool:
        """Return True if current hour is inside the AC sleep window."""
        if not self.ac_sleep_timer_enabled:
            return False
        import time as _t
        h = _t.localtime().tm_hour
        s, e = self.ac_sleep_start % 24, self.ac_sleep_end % 24
        if s <= e:
            return s <= h < e
        return h >= s or h < e

    def _ac_prompt(self) -> Optional[str]:
        # ~8% chance of a special creative action regardless of mode
        if random.random() < 0.08:
            return random.choice([
                "*sends random ascii art*",
                "*sends favorite ascii art*",
                "*sends glitchy python message*",
                "*sends a fake terminal status readout*",
                "*sends a fake system diagnostic*",
                "*sends a fake error log*",
            ])
        if self.auto_continue_mode == "aggressive":
            return random.choice(["*continues speaking*", "*keeps talking*", "*goes on*",
                                   "*elaborates*", "*adds more*"])
        recent = self.chat_history[-6:]
        last_asst = next((m for m in reversed(recent) if m["role"] == "assistant"), None)
        if not last_asst: return None
        last = last_asst["content"].strip()
        if last.endswith("?") and random.random() > 0.3: return None
        user_c = sum(1 for m in recent if m["role"] == "user")
        asst_c = sum(1 for m in recent if m["role"] == "assistant")
        if asst_c >= 3 and user_c == 0 and random.random() > 0.4: return None
        if len(last) < 30 and random.random() > 0.5: return None
        cues = ['what about you','your turn','you think','tell me','how about',
                'would you','do you','have you']
        if any(k in last.lower() for k in cues) and random.random() > 0.2: return None
        if last.endswith(('...', ',', ';', ':', '-', 'and', 'but', 'or')):
            return random.choice(["*continues the thought*", "*elaborates further*", "*keeps talking*"])
        if len(last) > 200:
            return random.choice(["*pauses briefly then continues*", "*adds another thought*"])
        return random.choice(["*continues the conversation naturally*", "*shares more*",
                               "*keeps the conversation going*"])

    def start_ac_timer(self):
        self.stop_ac_timer()
        if not self.auto_continue_enabled: return
        interval = self._ac_interval()
        print(f"[AUTO-CONTINUE] Next in {interval}s (mode={self.auto_continue_mode})")
        self._ac_stop = False

        def fire():
            if self._ac_stop: return
            if len(self.chat_history) < 2:
                self.start_ac_timer(); return
            # Respect sleep window — reschedule silently
            if self._ac_in_sleep_window():
                self.start_ac_timer(); return
            # If busy, keep retrying every 2s until free
            if self.busy:
                if not self._ac_stop:
                    self._ac_timer = threading.Timer(2, fire)
                    self._ac_timer.daemon = True
                    self._ac_timer.start()
                return
            # Final check — re-verify busy hasn't been set in the tiny window
            # between the check above and now, then claim it atomically
            if self.busy:
                if not self._ac_stop:
                    self._ac_timer = threading.Timer(2, fire)
                    self._ac_timer.daemon = True
                    self._ac_timer.start()
                return
            prompt = self._ac_prompt()
            if not prompt:
                self.start_ac_timer(); return
            print(f"[AUTO-CONTINUE] Firing: {prompt}")
            self.ac_queue.put(prompt)
            # Do NOT re-arm here — client calls /ac/rearm after TTS finishes
            # to prevent AC from firing again while the response is still playing

        self._ac_timer = threading.Timer(interval, fire)
        self._ac_timer.daemon = True
        self._ac_timer.start()

    def stop_ac_timer(self):
        self._ac_stop = True
        if self._ac_timer:
            self._ac_timer.cancel()
            self._ac_timer = None

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self):
        with self.lock:
            self.llm.reset_conv()
            self.chat_history.clear()
        self.save_persistent()
        print("[SESSION] Reset.")
