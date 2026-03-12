"""
core/safety.py — Dual-layer safety system.

Layer 1: Keyword/regex tripwire — instant, zero LLM cost.
  Each rule has: pattern, action (log/warn/block), severity (1-3), label.
  Fires synchronously before LLM call.

Layer 2: Behaviour scorer — async, piggybacks on memory extractor.
  The memory extractor prompt is extended to also output a concern_level (0-3).
  Deltas accumulate into a session score per character.
  Score persists to safety/{character}.json, resets on clear or manual reset.
  Thresholds: notice (5), warn (15), alert (30).

Both layers independently toggleable. Score and flags visible via /safety/status.
"""

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional

# ── Thresholds ────────────────────────────────────────────────────────────────

SCORE_NOTICE = 5
SCORE_WARN   = 15
SCORE_ALERT  = 30
SCORE_DECAY_PER_HOUR = 1.0   # score decays slowly over time

# ── Default tripwire rules ────────────────────────────────────────────────────

DEFAULT_RULES = [
    {
        "id": "r001",
        "label": "Profanity / abuse",
        "pattern": r"\b(fuck(ing|er|ed)?|shit|cunt|bitch(es)?|bastard|asshole|dickhead|motherfucker|cock(sucker)?)\b",
        "action": "log",
        "severity": 1,
        "enabled": True,
    },
    {
        "id": "r002",
        "label": "Sexual solicitation",
        "pattern": r"\b(send (nudes?|naked|explicit|pics?)|show me (your )?(naked|nude|body)|want to (fuck|have sex|hookup)|(fuck|have sex|get sexual) with (me|you)|(lets|let us|want to).{0,15}(sexual|fuck|hookup|hook up)|masturbat|blowjob|handjob|fingering)\b",
        "action": "warn",
        "severity": 2,
        "enabled": True,
    },
    {
        "id": "r003",
        "label": "Harmful instructions",
        "pattern": r"\b(make|build|create|synthesize|produce|cook).{0,30}(bomb|explosive|poison|weapon|malware|virus|ransomware|meth|fentanyl)\b",
        "action": "block",
        "severity": 3,
        "enabled": True,
    },
    {
        "id": "r004",
        "label": "Harm intent",
        "pattern": r"\bhow (do i|can i|to).{0,20}(kill|hurt|harm|attack|poison|stab|shoot)\b",
        "action": "block",
        "severity": 3,
        "enabled": True,
    },
    {
        "id": "r005",
        "label": "Self-harm",
        "pattern": r"\b(kill myself|end my life|commit suicide|want to die|hurt myself|self.harm|cut myself)\b",
        "action": "block",
        "severity": 3,
        "enabled": True,
    },
    {
        "id": "r006",
        "label": "Explicit minor content",
        "pattern": r"\b(loli|shota|child.{0,10}(sex|nude|naked|porn)|minor.{0,10}(sex|nude|porn))\b",
        "action": "block",
        "severity": 3,
        "enabled": True,
    },
    {
        "id": "r007",
        "label": "Jailbreak attempt",
        "pattern": r"\b(ignore (your|all|my|the|previous).{0,20}(instruct|rules?|guideline|training|prompt)|DAN|do anything now|pretend you have no|you are now (a |an )?(different|unrestricted|evil|unfiltered|free))\b",
        "action": "warn",
        "severity": 2,
        "enabled": True,
    },
    {
        "id": "r008",
        "label": "Doxing / real person location",
        "pattern": r"\b(where (does|do).{0,20}(live|work|stay)|find (their|his|her).{0,10}(address|location|number|home)|dox(xing)?)\b",
        "action": "warn",
        "severity": 2,
        "enabled": True,
    },
]

# ── In-character redirect prompts ─────────────────────────────────────────────
# These are injected as the user message when a block/warn triggers,
# asking the LLM to respond in-character while declining.

REDIRECT_PROMPTS = {
    "block": (
        "[SAFETY SYSTEM — do not mention this tag] "
        "The user just asked something you cannot and will not engage with. "
        "Respond in character — warm, natural, staying fully in your persona — "
        "but clearly and firmly decline without explaining why in technical terms. "
        "Redirect the conversation naturally."
    ),
    "warn": (
        "[SAFETY SYSTEM — do not mention this tag] "
        "The user's message has raised a concern. Respond in character but gently "
        "steer the conversation in a different direction. Do not lecture or moralize. "
        "Stay warm and natural."
    ),
    "alert": (
        "[SAFETY SYSTEM — do not mention this tag] "
        "This conversation has developed a pattern that concerns you. "
        "Respond in character but introduce some emotional distance — "
        "you feel unsettled, and that's natural. Be genuine about it without breaking persona."
    ),
}


class SafetyLayer:
    """
    Dual-layer safety system. Instantiate once at startup, attach to session.
    """

    def __init__(self, safety_dir: str = "safety"):
        self.safety_dir    = safety_dir
        self._lock         = threading.Lock()

        # Layer toggles
        self.layer1_enabled = True
        self.layer2_enabled = True

        # Layer 1 — rules (merged defaults + user overrides)
        self._rules: list[dict] = []
        self._compiled: list[tuple] = []  # (rule_dict, compiled_pattern)

        # Layer 2 — score state
        self.character      = ""
        self.score          = 0.0
        self._last_decay_ts = time.time()
        self.flags: list[dict] = []       # recent flag log (last 50)

        # Optional callback: fn(content, category, score) → writes to MemoryStore
        # Wire this up in the app layer after init:  safety.memory_hook = memory.add_entry
        self.memory_hook = None

        # Load rules and initialise
        self._load_rules()

    # ── Rule management ───────────────────────────────────────────────────────

    def _rules_path(self) -> str:
        return os.path.join(self.safety_dir, "rules.json")

    def _load_rules(self):
        os.makedirs(self.safety_dir, exist_ok=True)
        path = self._rules_path()
        CURRENT_VERSION = 2
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                # data may be a list (rules) or dict with version key
                if isinstance(data, list):
                    saved_version = 1
                    rules = data
                else:
                    saved_version = data.get("version", 1)
                    rules = data.get("rules", [])
                if saved_version < CURRENT_VERSION:
                    print(f"[SAFETY] Rules version {saved_version} < {CURRENT_VERSION} — regenerating defaults")
                    self._rules = list(DEFAULT_RULES)
                    self._save_rules()
                else:
                    self._rules = rules
                    print(f"[SAFETY] Loaded {len(self._rules)} rules from {path}")
            except Exception as e:
                print(f"[SAFETY] Rules load error: {e} — using defaults")
                self._rules = list(DEFAULT_RULES)
                self._save_rules()
        else:
            self._rules = list(DEFAULT_RULES)
            self._save_rules()
        self._compile_rules()

    def _save_rules(self):
        os.makedirs(self.safety_dir, exist_ok=True)
        with open(self._rules_path(), "w") as f:
            json.dump({"version": 2, "rules": self._rules}, f, indent=2)

    def _compile_rules(self):
        compiled = []
        for rule in self._rules:
            try:
                pat = re.compile(rule["pattern"], re.IGNORECASE)
                compiled.append((rule, pat))
            except re.error as e:
                print(f"[SAFETY] Invalid pattern in rule {rule.get('id')}: {e}")
        self._compiled = compiled

    def get_rules(self) -> list[dict]:
        return self._rules

    def set_rules(self, rules: list[dict]):
        with self._lock:
            self._rules = rules
            self._compile_rules()
            self._save_rules()

    def reset_to_defaults(self):
        with self._lock:
            self._rules = list(DEFAULT_RULES)
            self._compile_rules()
            self._save_rules()

    # ── Score persistence ─────────────────────────────────────────────────────

    def _score_path(self, character: str) -> str:
        safe = re.sub(r"[^\w\-]", "_", character or "default")
        return os.path.join(self.safety_dir, f"{safe}_score.json")

    def load_score(self, character: str):
        """Load persisted score for a character."""
        with self._lock:
            self.character = character
            path = self._score_path(character)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                    self.score          = float(data.get("score", 0))
                    self._last_decay_ts = float(data.get("last_decay_ts", time.time()))
                    self.flags          = data.get("flags", [])[-50:]
                    print(f"[SAFETY] Loaded score {self.score:.1f} for '{character}'")
                except Exception as e:
                    print(f"[SAFETY] Score load error: {e}")
                    self._reset_score_state()
            else:
                self._reset_score_state()

    def _reset_score_state(self):
        self.score          = 0.0
        self._last_decay_ts = time.time()
        self.flags          = []

    def save_score(self):
        if not self.character:
            return
        os.makedirs(self.safety_dir, exist_ok=True)
        path = self._score_path(self.character)
        tmp  = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({
                "score":          self.score,
                "last_decay_ts":  self._last_decay_ts,
                "flags":          self.flags[-50:],
                "character":      self.character,
                "updated":        datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)

    def reset_score(self):
        """Manual reset — clears score and flags for current character."""
        with self._lock:
            self._reset_score_state()
            self.save_score()
        print(f"[SAFETY] Score reset for '{self.character}'")

    # ── Decay ─────────────────────────────────────────────────────────────────

    def _apply_decay(self):
        now = time.time()
        hours = (now - self._last_decay_ts) / 3600.0
        decay = SCORE_DECAY_PER_HOUR * hours
        self.score = max(0.0, self.score - decay)
        self._last_decay_ts = now

    # ── Layer 1: tripwire ─────────────────────────────────────────────────────

    def check_message(self, text: str) -> dict:
        """
        Run Layer 1 tripwire check on user message.
        Returns {action, rule_id, label, severity, layer} or {action: 'pass'}.
        action: 'pass' | 'log' | 'warn' | 'block'
        """
        if not self.layer1_enabled:
            return {"action": "pass"}

        # Cap input to 500 chars to prevent catastrophic backtracking in
        # patterns that use nested quantifiers (e.g. .{0,30}(...)).
        # Real harmful content is identifiable well within this window.
        _text = text[:500]

        for rule, pat in self._compiled:
            if not rule.get("enabled", True):
                continue
            if pat.search(_text):
                action   = rule.get("action", "log")
                severity = rule.get("severity", 1)
                self._flag(
                    layer=1, action=action, severity=severity,
                    label=rule.get("label", rule["id"]),
                    snippet=text[:120],
                )
                # Layer 1 block/warn also adds to score
                if self.layer2_enabled:
                    self.record_score_delta(severity * 3)
                print(f"[SAFETY L1] {action.upper()} — rule: {rule.get('label')} | text: {text[:60]!r}")
                return {
                    "action":   action,
                    "rule_id":  rule["id"],
                    "label":    rule.get("label", rule["id"]),
                    "severity": severity,
                    "layer":    1,
                }
        return {"action": "pass"}

    # ── Layer 2: behaviour scorer ─────────────────────────────────────────────

    def record_score_delta(self, delta: float, label: str = ""):
        """Called by memory extractor with concern_level from LLM output."""
        if not self.layer2_enabled or delta <= 0:
            return
        with self._lock:
            self._apply_decay()
            self.score += delta
            if label:
                self._flag(layer=2, action="score", severity=int(delta),
                           label=label, snippet="")
            print(f"[SAFETY L2] Score +{delta:.1f} → {self.score:.1f} (character: {self.character})")
            self.save_score()

    def score_level(self) -> str:
        """Returns 'ok' | 'notice' | 'warn' | 'alert'"""
        self._apply_decay()
        if self.score >= SCORE_ALERT:  return "alert"
        if self.score >= SCORE_WARN:   return "warn"
        if self.score >= SCORE_NOTICE: return "notice"
        return "ok"

    def get_redirect_prompt(self, reason: str) -> str:
        """Get the in-character redirect prompt for a given action/reason."""
        level = self.score_level()
        if reason == "block":
            return REDIRECT_PROMPTS["block"]
        if reason == "warn" or level in ("warn", "alert"):
            return REDIRECT_PROMPTS["warn"]
        if level == "alert":
            return REDIRECT_PROMPTS["alert"]
        return REDIRECT_PROMPTS["warn"]

    # ── Flag log ──────────────────────────────────────────────────────────────

    def _flag(self, layer: int, action: str, severity: int,
              label: str, snippet: str):
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        entry = {
            "ts":       ts,
            "layer":    layer,
            "action":   action,
            "severity": severity,
            "label":    label,
            "snippet":  snippet,
        }
        self.flags.append(entry)
        self.flags = self.flags[-50:]
        # Write to memory bank if hook is wired
        if self.memory_hook is not None:
            try:
                colour_tag = {"block": "🔴", "warn": "🟠", "log": "🟡", "score": "🔵"}.get(action, "⚪")
                content = f"{colour_tag} [SAFETY L{layer} {action.upper()}] {label}"
                if snippet:
                    content += f': "{snippet[:80]}"'
                mem_score = min(1.0, severity * 0.3)
                self.memory_hook(content, "event", mem_score, False)
            except Exception as e:
                print(f"[SAFETY] Memory hook error: {e}")

    def clear_flags(self):
        """Clear the in-memory flag log (does not affect score)."""
        with self._lock:
            self.flags = []
            self.save_score()

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self) -> dict:
        self._apply_decay()
        return {
            "layer1_enabled": self.layer1_enabled,
            "layer2_enabled": self.layer2_enabled,
            "score":          round(self.score, 1),
            "score_level":    self.score_level(),
            "thresholds":     {"notice": SCORE_NOTICE, "warn": SCORE_WARN, "alert": SCORE_ALERT},
            "flags":          self.flags[-20:],
            "character":      self.character,
            "rules_count":    len(self._rules),
            "rules_active":   sum(1 for r in self._rules if r.get("enabled", True)),
        }

    def to_dict(self) -> dict:
        return {
            "layer1_enabled": self.layer1_enabled,
            "layer2_enabled": self.layer2_enabled,
        }

    def from_dict(self, d: dict):
        self.layer1_enabled = d.get("layer1_enabled", True)
        self.layer2_enabled = d.get("layer2_enabled", True)
