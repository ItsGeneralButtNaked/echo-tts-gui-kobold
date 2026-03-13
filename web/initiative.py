"""
web/initiative.py — Proactive character messaging engine for Ecko.

Completely separate from auto-continue. Fires on a per-character schedule
(minutes not seconds), generates an unprompted outreach message, broadcasts
it via SSE, and flags it for browser notification via a special event type.
"""

import json
import queue
import random
import threading
import time

from web.fx import fx_payload, random_effect, EFFECTS


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
        "*sends a fake terminal status readout*",
        "*sends a fake system diagnostic*",
        "*sends a fake error log*",
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
        self.fx_chance = 15   # sensible default: 15 %
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
            "enabled":             self.enabled,
            "mode":                self.mode,
            "fx_chance":           self.fx_chance,
            "sleep_timer_enabled": self.sleep_timer_enabled,
            "sleep_start":         self.sleep_start,
            "sleep_end":           self.sleep_end,
            "next_at":             self._next_at,
            "secs_remaining":      max(0, self._next_at - time.time()),
            "in_sleep_window":     self._in_sleep_window(),
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

            # ── FX chance override: roll dice first, bypassing the opener pool ──
            # fx_chance=0 → never auto-fire FX; fx_chance=100 → always FX
            if self.fx_chance > 0 and random.randint(1, 100) <= self.fx_chance:
                effect_name = random_effect()
                opener = f"__FX:{effect_name}__"
            else:
                opener = random.choice(self._OPENERS)

            # ── FX-only trigger — broadcast a visual effect + in-character quip ──
            if opener.startswith("__FX:"):
                effect_name = opener[5:].rstrip("_")
                if effect_name == "random":
                    effect_name = random_effect()

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
    def wire(self, *, get_busy, set_busy, get_context, save_fn, broadcast_fn, strip_fn):
        self._get_busy    = get_busy
        self._set_busy    = set_busy
        self._get_context = get_context   # () -> (chat_history, memory, llm)
        self._save_fn     = save_fn       # () -> None
        self._broadcast_fn = broadcast_fn # (payload_str) -> None
        self._strip_fn    = strip_fn      # (text) -> text
