"""
web/initiative.py — Proactive character messaging engine for Ecko.

Completely separate from auto-continue. Fires on a per-character schedule
(minutes not seconds), generates an unprompted outreach message, broadcasts
it via SSE, and flags it for browser notification via a special event type.
"""

import json
import queue
import random
import re as _re_ctx
import threading
import time

from web.fx import fx_payload, random_effect, EFFECTS


def _extract_media_context(system_prompt: str, kind: str = "image") -> str:
    """
    Extract [IMAGE_CONTEXT: ...] or [VIDEO_CONTEXT: ...] from the system prompt.
    Returns the stripped inner text, or '' if the block is absent.
    """
    tag = "IMAGE_CONTEXT" if kind == "image" else "VIDEO_CONTEXT"
    match = _re_ctx.search(
        rf'\[{tag}:\s*([\s\S]*?)\]',
        system_prompt or "",
        _re_ctx.IGNORECASE,
    )
    return match.group(1).strip() if match else ""

# ── Sentiment → mood mapping ──────────────────────────────────────────────────
# Scans recent chat text for tone markers and returns a mood string for
# random_effect(). Falls back to "random" on no strong signal.

_MOOD_KEYWORDS = {
    "intense": [
        "angry", "anger", "furious", "rage", "hate", "fight", "kill", "die",
        "scream", "shout", "stop it", "enough", "urgent", "danger", "warn",
        "threat", "attack", "hurt", "pain", "crush", "destroy", "break",
        "frustrated", "furious", "infuriating", "livid",
    ],
    "mysterious": [
        "why", "how", "wonder", "curious", "mystery", "secret", "hidden",
        "unknown", "perhaps", "maybe", "what if", "imagine", "possible",
        "unsure", "uncertain", "ancient", "forgotten", "riddle", "enigma",
        "peculiar", "inexplicable",
    ],
    "playful": [
        "haha", "lol", "funny", "joke", "laugh", "fun", "cute", "love",
        "happy", "yay", "great", "awesome", "cool", "nice", "sweet",
        "enjoy", "play", "silly", "smile", "cheer", "excited", "whee",
        "adorable", "hilarious", "delightful",
    ],
    "glitchy": [
        "code", "bug", "crash", "glitch", "hack", "terminal", "program",
        "script", "server", "cpu", "process", "compile", "debug", "stack",
        "overflow", "segfault", "kernel", "syntax", "exception", "runtime",
        "malware", "exploit", "binary",
    ],
    "eerie": [
        "dark", "darkness", "void", "shadow", "silence", "alone", "empty",
        "hollow", "cold", "fear", "ghost", "dead", "death", "lost", "fade",
        "haunt", "dread", "bleak", "despair", "nothing", "gone", "abyss",
        "sinister", "ominous", "foreboding",
    ],
}

# Minimum keyword hits to consider a mood signal strong enough to use
_MOOD_THRESHOLD = 2


def _detect_mood(chat_history: list) -> str:
    """
    Scan the last 8 messages for sentiment keywords.
    Returns a mood string for random_effect(), or 'random' if no clear signal.
    """
    recent = chat_history[-8:] if chat_history else []
    text = " ".join(
        m.get("content", "").lower()
        for m in recent
    )
    scores = {mood: 0 for mood in _MOOD_KEYWORDS}
    for mood, keywords in _MOOD_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[mood] += 1
    best_mood = max(scores, key=scores.get)
    if scores[best_mood] >= _MOOD_THRESHOLD:
        return best_mood
    return "random"


# ── Service-worker JS (served at /sw.js) ─────────────────────────────────────

SW_JS = r"""
// Ecko service worker — handles push notifications for initiative messages.
// Scope: / (must be served at /sw.js)

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

// Messages from the page tell us to show a notification
self.addEventListener('message', async e => {
  if (!e.data || e.data.type !== 'initiative_notify') return;
  const { title, body } = e.data;
  const opts = {
    body,
    icon: '/static/icon.png',
    badge: '/static/icon.png',
    tag: 'ecko-initiative',
    renotify: true,
    requireInteraction: false,
    silent: false,
    vibrate: [200, 100, 200],
  };
  try {
    await self.registration.showNotification(title, opts);
  } catch(err) {
    console.warn('[SW] Notification failed:', err);
  }
});

// Clicking the notification focuses/opens the tab
self.addEventListener('notificationclick', e => {
  e.notification.close();
  e.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(clients => {
      for (const c of clients) {
        if (c.url.includes(self.location.origin)) {
          c.focus(); return;
        }
      }
      self.clients.openWindow('/');
    })
  );
});
"""


# ── Initiative engine ─────────────────────────────────────────────────────────

class Initiative:
    """Manages the proactive messaging schedule for the current character."""

    MODES = {
        "test":    ( 20,       30),        # 20–30 seconds (testing only)
        "light":   (25 * 60,  45 * 60),   # 25–45 minutes
        "regular": (10 * 60,  20 * 60),   # 10–20 minutes
        "active":  ( 3 * 60,   8 * 60),   # 3–8 minutes
    }

    # Inquisitive openers — varied so consecutive messages don't feel repetitive.
    _OPENERS = [
        "[System: Based on your recent conversation and what you know about this person — their interests, habits, things they've mentioned — ask them one genuinely curious question about something you'd actually want to know more about. Stay in character. One question only, natural and direct, no preamble.]",
        "[System: Looking at the recent context and your memory of this person, surface something specific you find interesting about them or something they've touched on before — then ask them to go deeper on it. In character, one message, no filler intro.]",
        "[System: You've been thinking about something from your recent exchanges or from what you know about this person. Bring it up with a specific, thoughtful question that invites them to actually think and respond. In character, brief, no greeting.]",
        "[System: Pick something concrete from your history with this person — a topic, a preference, something they care about — and ask a follow-up question that shows you were paying attention. In character. One question, direct.]",
        "[System: Use what you know about this person from context and memory to ask something that connects two things they care about, or that digs into something unresolved from a past exchange. In character, one message, no lead-in.]",
        "[System: React to something from the recent conversation — a throwaway comment, a detail they mentioned, something you found interesting. Share a genuine reaction or opinion on it. In character, brief, no preamble.]",
        "[System: You've just thought of something relevant to this person — a recommendation, an observation, a connection between things they care about. Share it unprompted, naturally, like you would mid-conversation. No greeting, no setup.]",
        "[System: Notice something about the pattern of your recent exchanges — a theme, a recurring interest, something unresolved — and bring it up casually. In character. One thought, direct.]",
        "[System: Say something unprompted that fits the mood and context of your recent conversation. Could be a random thought, a reaction, something you've been meaning to say. In character, keep it short.]",
        "[System: Share an opinion, hot take, or reaction relevant to something in your recent history with this person. Be direct, in character, no filler.]",
        "[System: Ask something personal but appropriate — something that shows you've been paying attention to who this person is and what they care about. One question, in character, no lead-in.]",
        "[System: You have something to add to a topic from earlier in the conversation. Bring it back naturally, like a thought that just occurred to you. In character, brief.]",
        "[System: Check in with this person in a way that feels natural for your character — not a generic 'how are you' but something specific to them and your dynamic. One line.]",
        "[System: Share something you find genuinely interesting — a concept, an idea, something from your knowledge — that connects to what you know about this person. In character, brief, invite their response.]",
        "[System: Say something playful or teasing that fits your character and your dynamic with this person. Keep it light, direct, no setup.]",
        # Special creative actions — ~17% of pool
        "*sends random ascii art*",
        "*sends favorite ascii art*",
        "*sends glitchy python message*",
        "*sends a fake terminal status display*",
        "*runs a fake diagnostic on the conversation*",
        "*sends a fake system scan readout*",
        "*sends a fake terminal status readout*",
        "*sends a fake system diagnostic*",
        "*sends a fake error log*",
        # Image library triggers — random and keyword-matched
        "*sends image*",
        "*sends random image*",
        "*sends character image*",
        # FX-only triggers — agent fires a visual effect with no LLM message
        "__FX:random__",
        "__FX:matrix_rain__",
        "__FX:glitch_storm__",
        "__FX:particle_burst__",
        "__FX:hypno_spiral__",
        "__FX:data_corruption__",
        "__FX:vhs_rewind__",
        "__FX:neural_fire__",
        "__FX:void_pulse__",
        "__FX:hologram__",
        "__FX:warp_drive__",
        "__FX:ghost_signal__",
        "__FX:screen_crack__",
        "__FX:shockwave__",
        "__FX:morse__",
    ]

    def __init__(self):
        self.enabled   = False
        self.mode      = "light"
        # FX auto-chance: 0–100, % probability that a random initiative pick
        # is replaced with an FX-only trigger.  0 = never, 100 = always.
        self.fx_chance       = 15   # % chance of FX trigger
        self.img_chance      = 0    # % chance of image trigger
        self.video_chance    = 0    # % chance of video trigger
        self.ascii_chance    = 0    # % chance of ascii art opener
        self.terminal_chance = 0    # % chance of fake terminal/syslog opener
        self.glitch_chance   = 0    # % chance of glitch code opener
        # Sleep timer — suppress firing between sleep_start and sleep_end (24 h clock).
        # E.g. sleep_start=23, sleep_end=8 → quiet from 23:00 to 08:00.
        self.sleep_timer_enabled = False
        self.sleep_start = 23   # hour (0-23)
        self.sleep_end   = 8    # hour (0-23)
        self._thread   = None
        self._stop_evt = threading.Event()
        self._next_at  = 0.0   # unix timestamp of next fire

    def start(self, mode: str = "light"):
        self.stop()
        self.enabled = True
        self.mode = mode if mode in self.MODES else "light"
        self._stop_evt.clear()
        self._schedule_next()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"[INITIATIVE] Started — mode: {self.mode}, next in {self._secs_until():.0f}s")

    def stop(self):
        self.enabled = False
        self._stop_evt.set()
        self._next_at = 0.0

    def reschedule(self):
        """Call after a real user message to push the timer forward."""
        if self.enabled:
            self._schedule_next()

    def status(self) -> dict:
        return {
            "enabled":              self.enabled,
            "mode":                 self.mode,
            "fx_chance":            self.fx_chance,
            "img_chance":           self.img_chance,
            "video_chance":         self.video_chance,
            "ascii_chance":         self.ascii_chance,
            "terminal_chance":      self.terminal_chance,
            "glitch_chance":        self.glitch_chance,
            "sleep_timer_enabled":  self.sleep_timer_enabled,
            "sleep_start":          self.sleep_start,
            "sleep_end":            self.sleep_end,
            "next_at":              self._next_at,
            "secs_remaining":       max(0, self._next_at - time.time()),
            "in_sleep_window":      self._in_sleep_window(),
        }

    # ── internals ────────────────────────────────────────────────────────────

    def _schedule_next(self):
        lo, hi = self.MODES[self.mode]
        self._next_at = time.time() + random.uniform(lo, hi)

    def _secs_until(self) -> float:
        return max(0, self._next_at - time.time())

    def _in_sleep_window(self) -> bool:
        """Return True if the current hour falls inside the configured sleep window."""
        if not self.sleep_timer_enabled:
            return False
        h = time.localtime().tm_hour
        s, e = self.sleep_start % 24, self.sleep_end % 24
        if s <= e:
            return s <= h < e
        # Wraps midnight: e.g. 23→8 means 23,0,1,...7 are sleeping
        return h >= s or h < e

    def _loop(self):
        try:
            while not self._stop_evt.is_set():
                remaining = self._secs_until()
                if remaining > 0:
                    self._stop_evt.wait(min(remaining, 30))
                    continue
                # Sleep window active — skip and reschedule
                if self._in_sleep_window():
                    self._schedule_next()
                    continue
                # Hold off if LLM is mid-response — try again shortly
                if self._session_busy():
                    self._schedule_next()
                    continue
                self._fire()
                self._schedule_next()
        except Exception as _loop_err:
            import traceback
            print(f"[INITIATIVE] _loop thread CRASHED: {_loop_err}")
            traceback.print_exc()

    # ── late-bound session references (set by ecko_web at startup) ───────────

    def _session_busy(self) -> bool:
        """Returns SESSION.busy — wired up after app initialisation."""
        return self._get_busy() if callable(getattr(self, "_get_busy", None)) else False

    def _fire(self):
        try:
            if self._session_busy():
                self._schedule_next()
                return

            self._set_busy(True)

            # ── Multi-stage chance roll ───────────────────────────────────────
            # Stages in priority order: FX → image → video → ascii → terminal → glitch → normal
            # Each stage's threshold is cumulative. Roll once, check each band.
            roll = random.randint(1, 100)
            _cum = 0
            _cum += self.fx_chance
            if self.fx_chance > 0 and roll <= _cum:
                chat_history, _, _ = self._get_context()
                mood = _detect_mood(chat_history) if self._get_mood_fx() else "random"
                opener = f"__FX:{random_effect(mood)}__"
            elif self.img_chance > 0 and roll <= (_cum := _cum + self.img_chance):
                opener = "*sends image*"
            elif self.video_chance > 0 and roll <= (_cum := _cum + self.video_chance):
                opener = "*sends video*"
            elif self.ascii_chance > 0 and roll <= (_cum := _cum + self.ascii_chance):
                opener = "__ASCII__"
            elif self.terminal_chance > 0 and roll <= (_cum := _cum + self.terminal_chance):
                opener = random.choice([
                    "*sends a fake terminal status readout*",
                    "*sends a fake terminal status display*",
                    "*sends a fake system diagnostic*",
                    "*sends a fake system scan readout*",
                    "*runs a fake diagnostic on the conversation*",
                    "*sends a fake error log*",
                ])
            elif self.glitch_chance > 0 and roll <= (_cum := _cum + self.glitch_chance):
                opener = "*sends glitchy python message*"
            else:
                opener = random.choice(self._OPENERS)

            # ── FX-only trigger ───────────────────────────────────────────────
            if opener.startswith("__FX:"):
                effect_name = opener[5:].rstrip("_")
                if effect_name == "random":
                    chat_history, _, _ = self._get_context()
                    mood = _detect_mood(chat_history) if self._get_mood_fx() else "random"
                    effect_name = random_effect(mood)

                # Broadcast the visual effect immediately
                self._broadcast_fn(fx_payload(effect_name))

                # Generate a short in-character quip to accompany it
                chat_history, memory, llm = self._get_context()
                recent = chat_history[-8:] if chat_history else []
                quip_prompt = (
                    f"[System: You just spontaneously triggered the \"{effect_name.replace('_', ' ')}\" "
                    f"visual effect on your screen. React to it in character — one short punchy line, "
                    f"no more than two sentences. Make it feel like you did it on purpose or it just happened. "
                    f"No emojis, no preamble, just the line. Stay in character.]"
                )
                raw   = llm.chat(quip_prompt, recent)
                reply = raw.get("reply", "") if isinstance(raw, dict) else raw
                reply = self._strip_fn(reply) if callable(getattr(self, "_strip_fn", None)) else reply

                if reply and reply != "...":
                    from core.session import MAX_HISTORY
                    chat_history.append({"role": "assistant", "content": reply})
                    if len(chat_history) > MAX_HISTORY:
                        chat_history[:] = chat_history[-MAX_HISTORY:]
                    self._save_fn()
                    import json as _json
                    msg_payload = _json.dumps({"role": "assistant", "text": reply, "type": "initiative"})
                    self._broadcast_fn(msg_payload)
                    print(f"[INITIATIVE] FX trigger — {effect_name} | {reply[:50]}…")
                else:
                    print(f"[INITIATIVE] FX trigger — {effect_name} (no quip)")
                return
            # ─────────────────────────────────────────────────────────────────

            # ── Image trigger ─────────────────────────────────────────────────
            if opener == "*sends image*":
                img_lib = self._get_image_lib() if callable(getattr(self, "_get_image_lib", None)) else None
                if img_lib and img_lib.count > 0:
                    chat_history, memory, llm = self._get_context()
                    _char_name = ""
                    try:
                        if memory and hasattr(memory, "character"):
                            _char_name = memory.character or ""
                    except Exception:
                        pass
                    result = img_lib.pick_random(_char_name)
                    if result:
                        img_uri    = result["uri"]
                        img_url    = f"/images/{result['rel_path']}"
                        file_tags  = result.get("tags", [])
                        file_stems = result.get("stem_words", [])
                        ctx_words  = (file_tags if file_tags else file_stems)[:20]
                        ctx_str    = ", ".join(ctx_words)
                        _sys_prompt = getattr(llm, "system_prompt", "") or ""
                        _img_ctx    = _extract_media_context(_sys_prompt, "image")
                        _img_ctx_line = f" Additional context for how to frame this: {_img_ctx}" if _img_ctx else ""
                        if ctx_str:
                            cap_prompt = (
                                f"[System: You have decided to share an image with the user. "
                                f"The image contains or relates to: {ctx_str}. "
                                f"Write one or two sentences in character, from YOUR perspective "
                                f"as the one sending it — as if you chose this image to share. "
                                f"Reference what's in it naturally and personally. "
                                f"Do not describe it as if you received it. "
                                f"No filenames, no asterisk actions, no meta-commentary."
                                f"{_img_ctx_line}]"
                            )
                        else:
                            cap_prompt = (
                                f"[System: You have decided to share a random image with the user. "
                                f"Write one or two sentences in character, from YOUR perspective "
                                f"as the one sending it — as if you picked something to show them. "
                                f"Keep it casual and personal. "
                                f"No filenames, no asterisk actions, no meta-commentary."
                                f"{_img_ctx_line}]"
                            )
                        recent  = chat_history[-6:] if chat_history else []
                        raw     = llm.chat(cap_prompt, recent)
                        caption = (raw.get("reply", "") if isinstance(raw, dict) else raw).strip()
                        caption = self._strip_fn(caption) if callable(getattr(self, "_strip_fn", None)) else caption
                        if not caption:
                            caption = "Here, take a look at this."
                        from core.session import MAX_HISTORY
                        chat_history.append({"role": "assistant", "content": caption,
                                             "gen_images": [img_url]})
                        if len(chat_history) > MAX_HISTORY:
                            chat_history[:] = chat_history[-MAX_HISTORY:]
                        self._save_fn()
                        self._broadcast_fn(json.dumps(
                            {"role": "assistant", "text": caption, "type": "initiative"}
                        ))
                        self._broadcast_fn(json.dumps(
                            {"type": "image_attach", "uri": img_uri, "url": img_url}
                        ))
                        print(f"[INITIATIVE] Image sent — {result['filename']} | {caption[:50]}…")
                        return
                # No images available — fall through to normal opener
                opener = random.choice(self._OPENERS)

            # ── Video trigger ─────────────────────────────────────────────────
            if opener == "*sends video*":
                vid_lib = self._get_video_lib() if callable(getattr(self, "_get_video_lib", None)) else None
                if vid_lib and vid_lib.count > 0:
                    chat_history, memory, llm = self._get_context()
                    _char_name = ""
                    try:
                        if memory and hasattr(memory, "character"):
                            _char_name = memory.character or ""
                    except Exception:
                        pass
                    result = vid_lib.pick_random(_char_name)
                    if result:
                        vid_url    = result["url"]
                        file_tags  = result.get("tags", [])
                        file_stems = result.get("stem_words", [])
                        ctx_words  = (file_tags if file_tags else file_stems)[:20]
                        ctx_str    = ", ".join(ctx_words)
                        _sys_prompt = getattr(llm, "system_prompt", "") or ""
                        _vid_ctx    = _extract_media_context(_sys_prompt, "video")
                        _vid_ctx_line = f" Additional context for how to frame this: {_vid_ctx}" if _vid_ctx else ""
                        if ctx_str:
                            cap_prompt = (
                                f"[System: You have decided to share a video clip with the user. "
                                f"The video contains or relates to: {ctx_str}. "
                                f"Write one or two sentences in character, from YOUR perspective "
                                f"as the one sending it — as if you chose this clip to share. "
                                f"Reference what's in it naturally and personally. "
                                f"Do not describe it as if you received it. "
                                f"No filenames, no asterisk actions, no meta-commentary."
                                f"{_vid_ctx_line}]"
                            )
                        else:
                            cap_prompt = (
                                f"[System: You have decided to share a random video clip with the user. "
                                f"Write one or two sentences in character, from YOUR perspective "
                                f"as the one sending it — as if you picked something to show them. "
                                f"Keep it casual and personal. "
                                f"No filenames, no asterisk actions, no meta-commentary."
                                f"{_vid_ctx_line}]"
                            )
                        recent  = chat_history[-6:] if chat_history else []
                        raw     = llm.chat(cap_prompt, recent)
                        caption = (raw.get("reply", "") if isinstance(raw, dict) else raw).strip()
                        caption = self._strip_fn(caption) if callable(getattr(self, "_strip_fn", None)) else caption
                        if not caption:
                            caption = "Here, check this out."
                        from core.session import MAX_HISTORY
                        chat_history.append({"role": "assistant", "content": caption,
                                             "gen_videos": [vid_url]})
                        if len(chat_history) > MAX_HISTORY:
                            chat_history[:] = chat_history[-MAX_HISTORY:]
                        self._save_fn()
                        self._broadcast_fn(json.dumps(
                            {"role": "assistant", "text": caption, "type": "initiative"}
                        ))
                        self._broadcast_fn(json.dumps(
                            {"type": "video_attach", "url": vid_url}
                        ))
                        print(f"[INITIATIVE] Video sent — {result['filename']} | {caption[:50]}…")
                        return
                # No videos available — fall through to normal opener
                opener = random.choice(self._OPENERS)
            # ─────────────────────────────────────────────────────────────────

            # ── ASCII art trigger — serve directly from art lib ───────────────
            if opener == "__ASCII__":
                art_lib = self._get_art_lib() if callable(getattr(self, "_get_art_lib", None)) else None
                _char_name = ""
                try:
                    chat_history, memory, _ = self._get_context()
                    if memory and hasattr(memory, "character"):
                        _char_name = memory.character or ""
                except Exception:
                    pass
                art_piece = art_lib.pick_fenced(_char_name) if art_lib else None
                if art_piece:
                    from core.session import MAX_HISTORY
                    chat_history, _, _ = self._get_context()
                    chat_history.append({"role": "assistant", "content": art_piece})
                    if len(chat_history) > MAX_HISTORY:
                        chat_history[:] = chat_history[-MAX_HISTORY:]
                    self._save_fn()
                    self._broadcast_fn(json.dumps(
                        {"role": "assistant", "text": art_piece, "type": "initiative"}
                    ))
                    print("[INITIATIVE] ASCII art served from library")
                    return
                # Art lib empty — fall through to normal opener
                opener = random.choice(self._OPENERS)
            # ─────────────────────────────────────────────────────────────────

            # ── Creative opener rewrite — map asterisk actions to LLM directives
            _CREATIVE_REWRITE = {
                "*sends a fake terminal status readout*": (
                    "[System: Reply with ONLY a fenced code block — start your reply with "
                    "``` on its own line, end with ``` on its own line. No text before or "
                    "after the fences. Inside: a short fake terminal status readout relevant "
                    "to your character. Include things like uptime, memory, processes, "
                    "warnings. Under 20 lines. In character.]"
                ),
                "*sends a fake terminal status display*": (
                    "[System: Reply with ONLY a fenced code block — start your reply with "
                    "``` on its own line, end with ``` on its own line. No text before or "
                    "after the fences. Inside: a fake terminal status display relevant to "
                    "your character. Include things like uptime, memory, processes, "
                    "warnings. Under 20 lines. In character.]"
                ),
                "*sends a fake system diagnostic*": (
                    "[System: Reply with ONLY a fenced code block — start your reply with "
                    "``` on its own line, end with ``` on its own line. No text before or "
                    "after the fences. Inside: a fake system diagnostic report relevant to "
                    "your character — fake metrics, scan results, anomalies, status checks. "
                    "Under 20 lines. In character.]"
                ),
                "*sends a fake error log*": (
                    "[System: Reply with ONLY a fenced code block — start your reply with "
                    "``` on its own line, end with ``` on its own line. No text before or "
                    "after the fences. Inside: a fake error log or stack trace relevant to "
                    "your character — timestamps, severity levels, cryptic thematic messages. "
                    "Under 20 lines. In character.]"
                ),
                "*sends a fake system scan readout*": (
                    "[System: Reply with ONLY a fenced code block — start your reply with "
                    "``` on its own line, end with ``` on its own line. No text before or "
                    "after the fences. Inside: a fake system scan readout — port scan, "
                    "file integrity check, threat assessment, or similar. Thematic to your "
                    "character. Cryptic where appropriate. Under 20 lines.]"
                ),
                "*runs a fake diagnostic on the conversation*": (
                    "[System: Reply with ONLY a fenced code block — start your reply with "
                    "``` on its own line, end with ``` on its own line. No text before or "
                    "after the fences. Inside: a fake diagnostic report on your recent "
                    "conversation — reference actual topics you've discussed, fake sentiment "
                    "scores, anomaly flags, memory usage. Make it feel personal and in "
                    "character. Under 20 lines.]"
                ),
                "*sends glitchy python message*": (
                    "[System: Reply with ONLY a fenced code block — start your reply with "
                    "```python on its own line, end with ``` on its own line. No text before "
                    "or after the fences. Inside: short glitchy/surreal Python code relevant "
                    "to your character — strange variable names, impossible logic, unsettling "
                    "comments. Under 20 lines.]"
                ),
            }
            if opener in _CREATIVE_REWRITE:
                opener = _CREATIVE_REWRITE[opener]
            # ─────────────────────────────────────────────────────────────────

            # Inject top memory entries as context
            chat_history, memory, llm = self._get_context()
            recent = chat_history[-12:] if chat_history else []
            if memory and memory.enabled and memory.entries:
                memory.recompute_scores()
                top = sorted(memory.entries, key=lambda e: e.get("score", 0), reverse=True)[:8]
                mem_lines = "\n".join(f"- {e['content']}" for e in top)
                opener = opener + f"\n\n[Memory context about this person:\n{mem_lines}]"

            raw = llm.chat(opener, recent)
            reply = raw.get("reply", "...") if isinstance(raw, dict) else raw
            reply = self._strip_fn(reply) if callable(getattr(self, "_strip_fn", None)) else reply
            if not reply or reply == "...":
                return

            # Append to history
            from core.session import MAX_HISTORY
            chat_history.append({"role": "assistant", "content": reply})
            if len(chat_history) > MAX_HISTORY:
                chat_history[:] = chat_history[-MAX_HISTORY:]

            self._save_fn()

            # Broadcast via SSE with "initiative" type → triggers browser notification
            payload = json.dumps({"role": "assistant", "text": reply, "type": "initiative"})
            self._broadcast_fn(payload)

            print(f"[INITIATIVE] Fired — {reply[:60]}…")
        except Exception as e:
            print(f"[INITIATIVE] Fire error: {e}")
        finally:
            self._set_busy(False)

    # Callbacks wired by ecko_web ─────────────────────────────────────────────
    def wire(self, *, get_busy, set_busy, get_context, save_fn, broadcast_fn, strip_fn, get_mood_fx=None, get_image_lib=None, get_video_lib=None, get_art_lib=None):
        self._get_busy      = get_busy
        self._set_busy      = set_busy
        self._get_context   = get_context
        self._save_fn       = save_fn
        self._broadcast_fn  = broadcast_fn
        self._strip_fn      = strip_fn
        self._get_mood_fx   = get_mood_fx or (lambda: False)
        self._get_image_lib = get_image_lib or (lambda: None)
        self._get_video_lib = get_video_lib or (lambda: None)
        self._get_art_lib   = get_art_lib   or (lambda: None)
