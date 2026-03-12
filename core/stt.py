"""
core/stt.py — Speech-to-text: Whisper model loader, PTT engine, VAD open-mic.

Desktop uses STTEngine (PTT) and VADEngine (open-mic).
Web uses get_whisper() + transcribe_audio() for server-side file transcription.
"""

import os
import queue
import subprocess
import tempfile
import threading
import numpy as np

from faster_whisper import WhisperModel
from core.logger import log


# ─────────────────────────────────────────────────────────────────────────────
# SHARED WHISPER SINGLETON (web / server-side)
# ─────────────────────────────────────────────────────────────────────────────

_whisper_model = None
_whisper_lock  = threading.Lock()

def get_whisper(model: str = "base", device: str = "cpu") -> WhisperModel:
    """Lazy-load and return a shared WhisperModel instance.

    Uses double-checked locking: the fast path (already loaded) returns
    without acquiring the lock at all.  The slow path (first load) serialises
    on _whisper_lock and re-checks inside to handle the race where two threads
    both see None before either acquires the lock.
    """
    global _whisper_model
    if _whisper_model is not None:          # fast path — no lock needed
        return _whisper_model
    with _whisper_lock:                     # slow path — serialise first load
        if _whisper_model is None:          # re-check: another thread may have loaded while we waited
            log.info("[STT] Loading Whisper model %r on %s…", model, device)
            _whisper_model = WhisperModel(
                model, device=device,
                compute_type="float16" if device != "cpu" else "int8",
            )
            log.info("[STT] Whisper ready.")
    return _whisper_model


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm") -> str:
    """
    Transcribe raw audio bytes (from a web upload).
    Converts to WAV via ffmpeg if available, then runs Whisper.
    """
    suffix = ".webm" if "webm" in mime_type else ".ogg" if "ogg" in mime_type else ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    wav_path = tmp_path + ".wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True,
        )
        model = get_whisper()
        segs, _ = model.transcribe(wav_path, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segs).strip()
    except FileNotFoundError:
        try:
            model = get_whisper()
            segs, _ = model.transcribe(tmp_path, language="en", vad_filter=True)
            return " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""
    except Exception as e:
        print(f"[STT] Error: {e}")
        return ""
    finally:
        for p in [tmp_path, wav_path]:
            try:
                os.unlink(p)
            except Exception as e:
                log.debug("[STT] temp file cleanup skipped (%s): %s", p, e)


# ─────────────────────────────────────────────────────────────────────────────
# PRESS-TO-TALK ENGINE  (desktop)
# ─────────────────────────────────────────────────────────────────────────────

STT_SAMPLE_RATE = 16000
STT_CHANNELS    = 1
STT_BLOCKSIZE   = 1024


class STTEngine:
    """Press-to-talk STT via faster-whisper + sounddevice."""

    def __init__(self, model: str = "base", device: str = "cpu"):
        self.model   = WhisperModel(model, device=device,
                                    compute_type="float16" if device != "cpu" else "int8")
        self.audio_q = queue.Queue()
        self.recording = False
        self.stream    = None

    def start(self):
        if self.stream:
            return
        import sounddevice as sd
        self.recording = True
        self.audio_q.queue.clear()
        self.stream = sd.InputStream(
            samplerate=STT_SAMPLE_RATE, channels=STT_CHANNELS, dtype="float32",
            blocksize=STT_BLOCKSIZE, callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, frames, time_, status):
        if self.recording:
            self.audio_q.put(indata.copy())

    def transcribe(self) -> str:
        chunks = []
        while not self.audio_q.empty():
            chunks.append(self.audio_q.get())
        if not chunks:
            return ""
        audio = np.concatenate(chunks, axis=0).flatten()
        segs, _ = self.model.transcribe(audio, language="en", vad_filter=True)
        return " ".join(s.text.strip() for s in segs)


# ─────────────────────────────────────────────────────────────────────────────
# VAD OPEN-MIC ENGINE  (desktop)
# ─────────────────────────────────────────────────────────────────────────────

class VADEngine:
    """
    Always-on microphone with speech detection.
    Primary:  Silero-VAD (neural, accurate) — requires torch.
    Fallback: Energy-based VAD (pure numpy, always works).
    Runs in its own background thread.
    """

    def __init__(self, stt_model, on_speech_start, on_speech_end, on_state_change=None):
        """
        stt_model       : WhisperModel instance
        on_speech_start : called when speech begins (enables barge-in)
        on_speech_end   : called with transcribed text when speech ends
        on_state_change : optional; 'listening'|'speaking'|'processing'
        """
        self._stt_model       = stt_model
        self._on_speech_start = on_speech_start
        self._on_speech_end   = on_speech_end
        self._on_state_change = on_state_change
        self._running         = False
        self._thread          = None
        self._stream          = None

        # VAD state
        self._VAD_SR         = 16000
        self._frame_ms       = 30
        self._frame_size     = self._VAD_SR * self._frame_ms // 1000
        self._vad_model      = None
        self._use_silero     = False
        self._speech_buf     = []
        self._in_speech      = False
        self._silence_frames = 0
        self._SILENCE_THRESH = 12        # ~360 ms silence → end utterance
        self._SPEECH_THRESH  = 3
        self._speech_frames  = 0

        # Energy VAD
        self._ENERGY_SPEECH  = 0.012
        self._ENERGY_SILENCE = 0.007
        self._noise_floor    = 0.005
        self._energy_history = []

    # ── start / stop ─────────────────────────────────────────────────────────

    def start(self):
        self._load_vad()
        self._running = True
        self._thread  = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                log.warning("[STT] VADEngine stream close failed: %s", e)
            self._stream = None

    def _load_vad(self):
        try:
            import torch
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad",
                force_reload=False, onnx=False, trust_repo=True,
            )
            self._vad_model  = model
            self._use_silero = True
            print("[VAD] Silero loaded.")
        except Exception as e:
            print(f"[VAD] Silero not available ({e}), using energy VAD.")
            self._use_silero = False

    def _run(self):
        import sounddevice as sd
        self._stream = sd.InputStream(
            samplerate=self._VAD_SR, channels=1, dtype="float32",
            blocksize=self._frame_size, callback=self._cb,
        )
        with self._stream:
            while self._running:
                import time; time.sleep(0.05)

    def _cb(self, indata, frames, time_, status):
        if not self._running:
            return
        frame = indata[:, 0].copy()
        is_speech = self._detect(frame)

        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            if not self._in_speech and self._speech_frames >= self._SPEECH_THRESH:
                self._in_speech = True
                self._speech_buf.clear()
                if self._on_speech_start:
                    self._on_speech_start()
                if self._on_state_change:
                    self._on_state_change("speaking")
            if self._in_speech:
                self._speech_buf.append(frame)
        else:
            if self._in_speech:
                self._speech_buf.append(frame)
                self._silence_frames += 1
                if self._silence_frames >= self._SILENCE_THRESH:
                    self._in_speech   = False
                    self._speech_frames = 0
                    audio = np.concatenate(self._speech_buf)
                    self._speech_buf.clear()
                    if self._on_state_change:
                        self._on_state_change("processing")
                    threading.Thread(target=self._transcribe, args=(audio,),
                                     daemon=True).start()
            else:
                self._speech_frames = max(0, self._speech_frames - 1)
                self._silence_frames = 0

    def _detect(self, frame: np.ndarray) -> bool:
        if self._use_silero and self._vad_model is not None:
            try:
                import torch
                t = torch.from_numpy(frame).float()
                prob = self._vad_model(t, self._VAD_SR).item()
                return prob > 0.5
            except Exception as e:
                log.warning("[STT] Silero VAD failed, falling back to energy VAD: %s", e)
        # Energy fallback
        rms = float(np.sqrt(np.mean(frame ** 2)))
        self._energy_history.append(rms)
        if len(self._energy_history) > 50:
            self._noise_floor = min(np.percentile(self._energy_history[-50:], 20), 0.02)
            self._energy_history = self._energy_history[-50:]
        thresh = max(self._noise_floor * 2.5, self._ENERGY_SPEECH)
        return rms > thresh

    def _transcribe(self, audio: np.ndarray):
        try:
            segs, _ = self._stt_model.transcribe(audio, language="en", vad_filter=True)
            text = " ".join(s.text.strip() for s in segs).strip()
            if text and self._on_speech_end:
                self._on_speech_end(text)
            if self._on_state_change:
                self._on_state_change("listening")
        except Exception as e:
            print(f"[VAD] Transcription error: {e}")
            if self._on_state_change:
                self._on_state_change("listening")
