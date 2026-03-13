"""
web/fx.py — Visual effects engine for Ecko.

Provides a registry of named screen effects the agent or system can trigger.
Effects are broadcast as SSE payloads with type="fx" and an effect name.

Agent-triggered via reply tag:  *fx: matrix_rain*
User-triggered via !fx command (avatar must be open).
Initiative auto-trigger via __FX:name__ opener in _OPENERS pool.
"""

import json
import random

EFFECTS = [
    # Original effects
    "matrix_rain",
    "glitch_storm",
    "signal_static",
    "particle_burst",
    "scanline_warp",
    "data_corruption",
    "heartbeat",
    "hypno_spiral",
    "heart_pulse",
    "heart_scatter",
    # New effects
    "vhs_rewind",
    "neural_fire",
    "pixel_melt",
    "void_pulse",
    "static_burst",
    "cascade",
    "chromatic_bloom",
    "screen_crack",
    "ekg_flatline",
    "binary_rain",
    "warp_drive",
    "acid_wash",
    "ghost_signal",
    "memory_leak",
    "hologram",
    "shockwave",
    "morse",
    "thermal_vision",
    "digital_rain_color",
]

MOOD_EFFECTS = {
    "intense":    ["glitch_storm", "data_corruption", "signal_static",
                   "static_burst", "screen_crack", "shockwave", "ekg_flatline"],
    "mysterious": ["matrix_rain", "hypno_spiral", "scanline_warp",
                   "void_pulse", "ghost_signal", "hologram", "morse"],
    "playful":    ["particle_burst", "heartbeat", "heart_pulse", "heart_scatter",
                   "chromatic_bloom", "warp_drive"],
    "glitchy":    ["vhs_rewind", "pixel_melt", "memory_leak", "neural_fire",
                   "binary_rain", "cascade", "digital_rain_color"],
    "eerie":      ["void_pulse", "ghost_signal", "ekg_flatline", "thermal_vision",
                   "acid_wash", "morse"],
    "random":     EFFECTS,
}


def fx_payload(effect: str, duration_ms: int = 0) -> str:
    return json.dumps({
        "type": "fx",
        "effect": effect,
        "duration_ms": duration_ms,
    })


def random_effect(mood: str = "random") -> str:
    pool = MOOD_EFFECTS.get(mood, EFFECTS)
    return random.choice(pool)
