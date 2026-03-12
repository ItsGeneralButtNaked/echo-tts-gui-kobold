"""
web/fx.py — Visual effects engine for Ecko.

Provides a registry of named screen effects the agent or system can trigger.
Effects are broadcast as SSE payloads with type="fx" and an effect name.

Agent-triggered via reply tag:  *fx: matrix_rain*
  Place anywhere in a reply — the tag fires the effect and is stripped from
  the displayed bubble text. Use aliases: matrix, glitch, static, particles,
  scanlines, corrupt, heartbeat, hypno.

User-triggered via !fx command (avatar must be open):
  !fx              — random effect + agent quip
  !fx matrix       — specific effect + agent quip
  !fx list         — show available effects

Initiative auto-trigger via __FX:name__ opener in _OPENERS pool.

Available effects:
  matrix_rain     — Tinted Katakana/hex character cascade
  glitch_storm    — Rapid horizontal tear/offset glitch bars
  signal_static   — Tinted white-noise TV static burst
  particle_burst  — Radial neon particle explosion from centre
  scanline_warp   — CRT scanline geometry distortion
  data_corruption — Random character substitution cascade
  heartbeat       — Pulsing EKG-style line sweep
  hypno_spiral    — Rotating tinted colour spiral
"""

import json
import random

# ── Effect registry ───────────────────────────────────────────────────────────

EFFECTS = [
    "matrix_rain",
    "glitch_storm",
    "signal_static",
    "particle_burst",
    "scanline_warp",
    "data_corruption",
    "heartbeat",
    "hypno_spiral",
]

# Effects that pair well with unsettling/intense conversational moments
MOOD_EFFECTS = {
    "intense":    ["glitch_storm", "data_corruption", "signal_static"],
    "mysterious": ["matrix_rain", "hypno_spiral", "scanline_warp"],
    "playful":    ["particle_burst", "heartbeat"],
    "random":     EFFECTS,
}


def fx_payload(effect: str, duration_ms: int = 0) -> str:
    """
    Build a JSON SSE payload that triggers a named visual effect on the frontend.
    duration_ms=0 means use the effect's own default duration.
    """
    return json.dumps({
        "type": "fx",
        "effect": effect,
        "duration_ms": duration_ms,
    })


def random_effect(mood: str = "random") -> str:
    """Return a random effect name, optionally filtered by mood."""
    pool = MOOD_EFFECTS.get(mood, EFFECTS)
    return random.choice(pool)
