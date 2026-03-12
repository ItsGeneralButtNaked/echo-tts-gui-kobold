"""
core/context_mode.py — Context budget presets for prompt assembly.

Five named modes control how much context is injected per turn.
All modes are NON-DESTRUCTIVE — switching modes never deletes stored
memories, RAG chunks, or conversation archives.  They only change what
is retrieved and injected at inference time.

Mode ladder (fastest → smartest):
  voice_fast      ⚡  Absolute minimum — real-time voice, lowest latency
  voice_balanced  🎙️  Light memory recall — voice with some context
  standard        💬  Default text chat — balanced speed / quality
  deep_recall     🧠  More RAG + memories — slower, richer context
  full_context    📚  Maximum — all available memory and history

Usage:
    from core.context_mode import MODES, DEFAULT_MODE
    m = MODES["standard"]
    print(m.prompt_budget_chars)   # 1800
"""

from dataclasses import dataclass


@dataclass
class ContextMode:
    name:               str
    label:              str
    description:        str
    emoji:              str

    # ── total injected context cap ────────────────────────────────────────────
    prompt_budget_chars: int   # hard ceiling on total chars injected (profile + rag + memory)

    # ── RAG / conversation archive ────────────────────────────────────────────
    max_rag_chunks:     int    # number of past-interaction passages to retrieve
    rag_chunk_chars:    int    # max chars per RAG passage
    min_rag_score:      int    # minimum keyword overlap score (keyword retrieval)
    min_rag_cosine:     float  # minimum cosine similarity (semantic retrieval)

    # ── structured memory ─────────────────────────────────────────────────────
    max_memories:       int    # number of scored memory entries to inject
    memory_chars:       int    # max chars per memory entry

    # ── character profile ─────────────────────────────────────────────────────
    profile_chars:      int    # max chars of the character voice profile

    # ── live conversation history (Kobold / OpenAI-compat paths only) ─────────
    max_history:        int    # number of recent messages sent as chat turns


# ─────────────────────────────────────────────────────────────────────────────
# PRESETS
# ─────────────────────────────────────────────────────────────────────────────

MODES: dict[str, ContextMode] = {

    "voice_fast": ContextMode(
        name                = "voice_fast",
        label               = "Voice Fast",
        description         = "Absolute minimum context. Lowest latency, best for real-time voice.",
        emoji               = "⚡",
        prompt_budget_chars = 400,
        max_rag_chunks      = 0,
        rag_chunk_chars     = 0,
        min_rag_score       = 99,   # effectively disabled
        min_rag_cosine      = 0.99,
        max_memories        = 2,
        memory_chars        = 80,
        profile_chars       = 200,
        max_history         = 4,
    ),

    "voice_balanced": ContextMode(
        name                = "voice_balanced",
        label               = "Voice Balanced",
        description         = "Light RAG + memories. Voice with occasional memory recall.",
        emoji               = "🎙️",
        prompt_budget_chars = 900,
        max_rag_chunks      = 1,
        rag_chunk_chars     = 250,
        min_rag_score       = 3,
        min_rag_cosine      = 0.45,
        max_memories        = 3,
        memory_chars        = 100,
        profile_chars       = 300,
        max_history         = 6,
    ),

    "standard": ContextMode(
        name                = "standard",
        label               = "Standard",
        description         = "Default. Balanced for text chat with moderate context.",
        emoji               = "💬",
        prompt_budget_chars = 1800,
        max_rag_chunks      = 2,
        rag_chunk_chars     = 350,
        min_rag_score       = 2,
        min_rag_cosine      = 0.38,
        max_memories        = 4,
        memory_chars        = 120,
        profile_chars       = 400,
        max_history         = 10,
    ),

    "deep_recall": ContextMode(
        name                = "deep_recall",
        label               = "Deep Recall",
        description         = "More RAG + memories. Slower but more contextually aware.",
        emoji               = "🧠",
        prompt_budget_chars = 3000,
        max_rag_chunks      = 3,
        rag_chunk_chars     = 450,
        min_rag_score       = 1,
        min_rag_cosine      = 0.32,
        max_memories        = 6,
        memory_chars        = 150,
        profile_chars       = 600,
        max_history         = 14,
    ),

    "full_context": ContextMode(
        name                = "full_context",
        label               = "Full Context",
        description         = "Maximum context. Uses all available memory and history.",
        emoji               = "📚",
        prompt_budget_chars = 5000,
        max_rag_chunks      = 5,
        rag_chunk_chars     = 500,
        min_rag_score       = 1,
        min_rag_cosine      = 0.28,
        max_memories        = 8,
        memory_chars        = 180,
        profile_chars       = 1000,
        max_history         = 20,
    ),
}

# Ordered list for UI rendering (slider / toggle)
MODE_ORDER: list[str] = [
    "voice_fast",
    "voice_balanced",
    "standard",
    "deep_recall",
    "full_context",
]

DEFAULT_MODE = "standard"


def get_mode(name: str) -> ContextMode:
    """Return a ContextMode by name, falling back to DEFAULT_MODE."""
    return MODES.get(name, MODES[DEFAULT_MODE])
