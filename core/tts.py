"""
core/tts.py — Unified TTS provider registry and streaming caller.

Supports AllTalk/EchoTTS, Kokoro, Piper, Coqui, ElevenLabs, and OpenAI TTS.
All providers expose a stream_audio(text, voice, **opts) generator that yields
raw PCM bytes (44100 Hz, 16-bit mono) unless noted otherwise.
"""

import struct
import threading
import requests
from typing import Generator, Iterator

from core.logger import log


# ─────────────────────────────────────────────────────────────────────────────
# PROVIDER REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

TTS_PROVIDER_REGISTRY: dict[str, dict] = {
    "alltalk": {
        "label":          "EchoTTS (local)",
        "base_url":       "http://localhost:8000",
        "needs_api_key":  False,
        "needs_voice":    True,         # voice list fetched from /v1/voices
        "api_style":      "openai_pcm", # OpenAI-compatible /v1/audio/speech → PCM stream
        "output_format":  "pcm",
    },
    # "kokoro": {
    #     "label":          "Kokoro (local, low VRAM)",
    #     "base_url":       "http://localhost:8880",
    #     "needs_api_key":  False,
    #     "needs_voice":    True,
    #     "api_style":      "openai_pcm",
    #     "output_format":  "pcm",
    # },
    # "piper": {
    #     "label":          "Piper (local)",
    #     "base_url":       "http://localhost:5000",
    #     "needs_api_key":  False,
    #     "needs_voice":    True,         # voice = model name passed as query param
    #     "api_style":      "piper",
    #     "output_format":  "wav",
    # },
    # "coqui": {
    #     "label":          "Coqui TTS (local)",
    #     "base_url":       "http://localhost:5002",
    #     "needs_api_key":  False,
    #     "needs_voice":    True,         # speaker_id / style
    #     "api_style":      "coqui",
    #     "output_format":  "wav",
    # },
    "elevenlabs": {
        "label":          "ElevenLabs (API)",
        "base_url":       "https://api.elevenlabs.io",
        "needs_api_key":  True,
        "needs_voice":    True,         # voice_id from ElevenLabs
        "api_style":      "elevenlabs",
        "output_format":  "mp3",
    },
    "openai_tts": {
        "label":          "OpenAI TTS (API)",
        "base_url":       "https://api.openai.com",
        "needs_api_key":  True,
        "needs_voice":    True,         # alloy, echo, nova, onyx, shimmer, fable
        "api_style":      "openai_pcm",
        "output_format":  "pcm",
    },
}

# Voices available for providers that have a fixed list rather than a server endpoint
BUILTIN_VOICES: dict[str, list[str]] = {
    "openai_tts":  ["alloy", "echo", "nova", "onyx", "shimmer", "fable"],
    "elevenlabs":  [],   # fetched from API at runtime
}


def _wav_header(sample_rate: int = 44100) -> bytes:
    """Return a streaming WAV header (data length = 0xFFFFFFFF)."""
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16,
        b"data", 0xFFFFFFFF,
    )


# ─────────────────────────────────────────────────────────────────────────────
# TTS CALLER
# ─────────────────────────────────────────────────────────────────────────────

class TTSCaller:
    """
    Unified TTS streaming layer.

    Usage:
        tts = TTSCaller()
        for chunk in tts.stream(text):
            audio_player.write(chunk)          # raw PCM bytes
    """

    def __init__(self):
        self.provider_id = "alltalk"
        self.base_url    = TTS_PROVIDER_REGISTRY["alltalk"]["base_url"]
        self.api_key     = ""
        self.voice       = ""
        # Extra params (KV scale for AllTalk/EchoTTS, style for Coqui, etc.)
        self.extra: dict = {}

    # ── serialise / deserialise ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "base_url":    self.base_url,
            "api_key":     self.api_key,
            "voice":       self.voice,
            "extra":       self.extra,
        }

    def from_dict(self, d: dict):
        self.provider_id = d.get("provider_id", self.provider_id)
        self.base_url    = d.get("base_url",    self.base_url)
        self.api_key     = d.get("api_key",     self.api_key)
        self.voice       = d.get("voice",       self.voice)
        self.extra       = d.get("extra",       self.extra)

    @property
    def _style(self) -> str:
        return TTS_PROVIDER_REGISTRY.get(self.provider_id, {}).get("api_style", "openai_pcm")

    # ── voice list ───────────────────────────────────────────────────────────

    def list_voices(self) -> list[str]:
        """
        Return available voice IDs for the current provider.
        Returns [] on failure.
        """
        style = self._style
        try:
            if style == "openai_pcm":
                if self.provider_id in BUILTIN_VOICES:
                    return BUILTIN_VOICES[self.provider_id]
                r = requests.get(f"{self.base_url}/v1/voices", timeout=5)
                return sorted(v["id"] for v in r.json().get("data", []))
            elif style == "piper":
                r = requests.get(f"{self.base_url}/api/voices", timeout=5)
                return [v.get("key") or v.get("name") for v in r.json()]
            elif style == "coqui":
                r = requests.get(f"{self.base_url}/api/speakers", timeout=5)
                return [s.get("name", s) if isinstance(s, dict) else s for s in r.json()]
            elif style == "elevenlabs":
                hdrs = {"xi-api-key": self.api_key} if self.api_key else {}
                r = requests.get(f"{self.base_url}/v1/voices", headers=hdrs, timeout=5)
                voices = r.json().get("voices", [])
                # Return "Name (voice_id)" so the dropdown is human-readable.
                # The settings route strips the suffix back to just the id when saving.
                return [f"{v.get('name','?')} ({v['voice_id']})" for v in voices]
        except Exception as e:
            print(f"[TTS] list_voices failed ({self.provider_id}): {e}")
        return []

    def ping(self) -> tuple[bool, str]:
        """Return (online: bool, label: str)"""
        try:
            style = self._style
            if style in ("openai_pcm",):
                if self.provider_id in BUILTIN_VOICES:
                    # API providers — do a lightweight models check
                    hdrs = {}
                    if self.api_key:
                        hdrs["Authorization"] = f"Bearer {self.api_key}"
                    requests.get(f"{self.base_url}/v1/models", headers=hdrs, timeout=4)
                else:
                    requests.get(f"{self.base_url}/v1/voices", timeout=4)
            elif style == "piper":
                requests.get(f"{self.base_url}/api/voices", timeout=4)
            elif style == "coqui":
                requests.get(f"{self.base_url}/api/speakers", timeout=4)
            elif style == "elevenlabs":
                hdrs = {"xi-api-key": self.api_key} if self.api_key else {}
                requests.get(f"{self.base_url}/v1/user", headers=hdrs, timeout=4)
            label = TTS_PROVIDER_REGISTRY.get(self.provider_id, {}).get("label", self.provider_id)
            return True, label
        except Exception as e:
            log.debug("[TTS] ping failed (%s): %s", self.provider_id, e)
            return False, TTS_PROVIDER_REGISTRY.get(self.provider_id, {}).get("label", self.provider_id)

    # ── streaming ─────────────────────────────────────────────────────────────

    def stream(self, text: str, cancel: "threading.Event | None" = None) -> Iterator[bytes]:
        style = self._style
        if style == "openai_pcm":
            yield from self._stream_openai_pcm(text, cancel)
        elif style == "piper":
            yield from self._stream_piper(text)
        elif style == "coqui":
            yield from self._stream_coqui(text)
        elif style == "elevenlabs":
            yield from self._stream_elevenlabs(text, cancel)
        else:
            raise ValueError(f"Unknown TTS api_style: {style!r}")

    def _stream_openai_pcm(self, text: str, cancel=None) -> Iterator[bytes]:
        """AllTalk / EchoTTS / Kokoro / OpenAI TTS — OpenAI-compatible PCM stream."""
        yield _wav_header()

        payload: dict = {
            "input":           text,
            "voice":           self.voice,
            "stream":          True,
            "response_format": "pcm",
        }

        if self.provider_id in ("alltalk", "kokoro") and self.extra.get("kv_scale") is not None:
            payload["extra_body"] = {
                "speaker_kv_scale":     self.extra["kv_scale"],
                "speaker_kv_min_t":     self.extra.get("kv_min_t", 0.9),
                "speaker_kv_max_layers": self.extra.get("kv_max_layers", 24),
            }

        headers: dict = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.provider_id == "openai_tts":
            payload["model"] = "tts-1"

        # Use a session so we can access the underlying socket and close it
        # immediately from a cancel thread without waiting for the next chunk
        sess = requests.Session()
        resp = sess.post(
            f"{self.base_url}/v1/audio/speech",
            json=payload, headers=headers, stream=True, timeout=120,
        )
        resp.raise_for_status()

        # Register socket closer on the cancel event so it fires immediately
        # rather than waiting for the next iter_content chunk to unblock
        _raw = None
        try:
            _raw = resp.raw._fp.fp.raw._sock  # urllib3 socket
        except Exception as e:
            log.debug("[TTS] socket grab skipped (non-fatal): %s", e)

        def _close_on_cancel():
            if cancel:
                cancel.wait()           # blocks until cancel.set() is called
                try:
                    if _raw: _raw.close()
                except Exception as e:
                    log.debug("[TTS] cancel socket close skipped: %s", e)
                try:
                    resp.close()
                    sess.close()
                except Exception as e:
                    log.debug("[TTS] cancel session close skipped: %s", e)

        if cancel:
            t = threading.Thread(target=_close_on_cancel, daemon=True)
            t.start()

        try:
            for chunk in resp.iter_content(4096):
                if cancel and cancel.is_set():
                    return
                if chunk:
                    if cancel and cancel.is_set():  # check again after receiving, before yielding
                        return
                    yield chunk
        except Exception as e:
            # Socket closed by cancel thread mid-read — expected, not an error
            if cancel and cancel.is_set():
                return
            raise
        finally:
            try:
                resp.close()
            except Exception as e:
                log.debug("[TTS] resp.close() skipped: %s", e)
            try:
                sess.close()
            except Exception as e:
                log.debug("[TTS] sess.close() skipped: %s", e)

    def _stream_piper(self, text: str) -> Iterator[bytes]:
        """Piper REST — /api/tts returns WAV bytes."""
        params = {"text": text}
        if self.voice:
            params["voice"] = self.voice
        r = requests.get(f"{self.base_url}/api/tts", params=params, timeout=60)
        r.raise_for_status()
        yield r.content

    def _stream_coqui(self, text: str) -> Iterator[bytes]:
        """Coqui TTS REST — /api/tts returns WAV bytes."""
        params = {"text": text, "speaker_id": self.voice or ""}
        if self.extra.get("style"):
            params["style"] = self.extra["style"]
        r = requests.get(f"{self.base_url}/api/tts", params=params, timeout=60)
        r.raise_for_status()
        yield r.content

    def _stream_elevenlabs(self, text: str, cancel=None) -> Iterator[bytes]:
        """ElevenLabs streaming — yields MP3 chunks.
        Tries the streaming endpoint first; falls back to the standard endpoint
        on 402 (free-tier accounts cannot use /stream).
        """
        if not self.api_key:
            raise ValueError("ElevenLabs API key not set — enter your key in TTS settings and hit APPLY")
        voice_id = self.voice or "21m00Tcm4TlvDq8ikWAM"  # default: Rachel
        headers = {
            "xi-api-key":    self.api_key,
            "Content-Type":  "application/json",
            "Accept":        "audio/mpeg",
        }
        payload = {
            "text":              text,
            "model_id":          self.extra.get("model_id", "eleven_multilingual_v2"),
            "voice_settings":    self.extra.get("voice_settings", {
                "stability": 0.5, "similarity_boost": 0.75,
            }),
        }
        # Try streaming endpoint first
        resp = requests.post(
            f"{self.base_url}/v1/text-to-speech/{voice_id}/stream",
            headers=headers, json=payload, stream=True, timeout=120,
        )
        if resp.status_code == 402:
            # Free-tier: streaming not available — fall back to standard endpoint
            print("[TTS/ElevenLabs] Streaming endpoint requires paid plan, falling back to standard endpoint")
            resp.close()
            resp = requests.post(
                f"{self.base_url}/v1/text-to-speech/{voice_id}",
                headers=headers, json=payload, timeout=120,
            )
        resp.raise_for_status()
        for chunk in resp.iter_content(4096):
            if cancel and cancel.is_set():
                break
            if chunk:
                yield chunk
