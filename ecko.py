import os
import sys
import threading
import requests
import numpy as np
import sounddevice as sd
import time
import json
import random

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QPushButton, QComboBox, QLabel, QSlider,
    QSizePolicy, QFileDialog, QMessageBox, QLineEdit
)
from PySide6.QtCore import Qt, QTimer, QRectF, QPointF, QSettings, QEvent, Signal, QObject
from PySide6.QtGui import QPainter, QColor, QPen, QFont, QTextCursor, QPainterPath, QLinearGradient, QBrush

# ======================
# CONFIG
# ======================

KOBOLD_BASE = "http://localhost:5001"
TTS_BASE = "http://localhost:8000"

SAMPLE_RATE = 44100
CHANNELS = 1

WAVE_SAMPLES = 2048
WAVE_TIMER_MS = 30
MAX_HISTORY_MESSAGES = 20  # 10 turns

# ======================
# STT CONFIG
# ======================

STT_SAMPLE_RATE = 16000
STT_CHANNELS = 1
STT_BLOCKSIZE = 1024

LEFT_ALT_SCANCODE = 0x38  # Windows left alt

# ======================
# WAVE DISPLAY CONFIG
# ======================
# Options: 'wave', 'lips', 'circles', 'ribbon', 'radial', 'particles', 'wave_grid'
WAVE_DISPLAY_MODE = 'ribbon'

# ======================
# AUDIO PLAYER
# ======================

class PCMPlayer:
    def __init__(self):
        self.stream = sd.RawOutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=1024
        )
        self.stream.start()

        self.last_audio = np.zeros(WAVE_SAMPLES, dtype=np.int16)
        self.last_rms = 0.0

        self.auto_gain = False
        self.agc_mode = "rms"  # or "peak"
        self.agc_target_rms = 0.1
        self.agc_gain = 1.0
        self.agc_max_gain = 1.6
        self.agc_smoothing = 0.1

    def play(self, pcm_bytes, gain=1.5, limit=0.95):
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self.agc_mode == "rms":
            level = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
        else:  # peak
            level = float(np.max(np.abs(audio)) + 1e-12)

        self.last_rms = 0.97 * self.last_rms + 0.03 * level

        if self.auto_gain:
            desired = self.agc_target_rms / max(self.last_rms, 1e-9)
            desired = min(desired, self.agc_max_gain)
            self.agc_gain += (desired - self.agc_gain) * self.agc_smoothing
            gain *= self.agc_gain

        audio *= gain
        audio = np.clip(audio, -limit, limit)
        out = (audio * 32767).astype(np.int16)

        if len(out) >= WAVE_SAMPLES:
            self.last_audio = out[-WAVE_SAMPLES:]
        else:
            keep = WAVE_SAMPLES - len(out)
            self.last_audio = np.concatenate([self.last_audio[-keep:], out])

        self.stream.write(out.tobytes())

    def stop_playback(self):
        """Immediately stop audio playback and clear buffer"""
        # Clear the stream buffer
        try:
            self.stream.abort()  # Stop immediately
            self.stream.start()  # Restart for next playback
            self.last_audio = np.zeros(WAVE_SAMPLES, dtype=np.int16)
            self.last_rms = 0.0
        except Exception as e:
            print(f"[AUDIO] Error stopping playback: {e}")

    def close(self):
        self.stream.stop()
        self.stream.close()

# ======================
# STT ENGINE
# ======================

from faster_whisper import WhisperModel
import queue

class STTEngine:
    def __init__(self, model="base", device="auto"):
        self.model = WhisperModel(
            model,
            device=device,
            compute_type="float16" if device != "cpu" else "int8"
        )

        self.audio_q = queue.Queue()
        self.recording = False
        self.stream = None

    def start(self):
        if self.stream:
            return

        self.recording = True
        self.audio_q.queue.clear()

        self.stream = sd.InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=STT_CHANNELS,
            dtype="float32",
            blocksize=STT_BLOCKSIZE,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def _callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_q.put(indata.copy())

    def transcribe(self):
        chunks = []
        while not self.audio_q.empty():
            chunks.append(self.audio_q.get())

        if not chunks:
            return ""

        audio = np.concatenate(chunks, axis=0).flatten()
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            vad_filter=True
        )

        return " ".join(seg.text.strip() for seg in segments)


# ======================
# STATUS LED
# ======================

class StatusLED(QWidget):
    def __init__(self, diameter=10):
        super().__init__()
        self._color = QColor("#2a2a2a")
        self._diameter = diameter
        self.setFixedSize(diameter, diameter)

    def set_color(self, color_hex):
        self._color = QColor(color_hex)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(self._color)
        p.drawEllipse(0, 0, self._diameter, self._diameter)


# ======================
# FLEXIBLE WAVE DISPLAY
# ======================

class WaveDisplay(QWidget):
    def __init__(self, player, app_ref, mode=None):
        """
        Flexible audio display widget.
        
        Parameters:
        -----------
        player : object
            Audio player object providing waveform data
        app_ref : object
            Reference to main application (if needed)
        mode : str, optional
            Display mode: 'wave', 'lips', 'circles', 'ribbon', 'radial', 'particles', 'wave_grid'.
            If None, falls back to the global WAVE_DISPLAY_MODE.
        """
        super().__init__()
        self.player = player
        self.app_ref = app_ref

        # Use global default if no mode provided
        if mode is None:
            mode = WAVE_DISPLAY_MODE

        self.mode = mode

        # Widget setup
        self.setMinimumHeight(220)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Timer for refreshing the display
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(WAVE_TIMER_MS)

        # Smoothing buffer
        self._smoothed = np.zeros(WAVE_SAMPLES, dtype=np.float32)
        self._smooth_alpha = 0.25
        
        # Amplitude scale (adjustable)
        self.amplitude_scale = 1.0

    def set_mode(self, mode):
        """Switch display mode at runtime."""
        if mode in ('wave', 'lips', 'circles', 'ribbon', 'radial', 'particles', 'wave_grid'):
            self.mode = mode
            self.update()

    def set_amplitude_scale(self, scale):
        """Adjust amplitude multiplier (0.1 to 2.0)"""
        self.amplitude_scale = max(0.1, min(2.0, scale))

    def paintEvent(self, event):
        """Main paint handler."""
        if self.mode == 'wave':
            self._paint_wave()
        elif self.mode == 'lips':
            self._paint_lips()
        elif self.mode == 'circles':
            self._paint_circles()
        elif self.mode == 'ribbon':
            self._paint_ribbon()
        elif self.mode == 'radial':
            self._paint_radial()
        elif self.mode == 'particles':
            self._paint_particles()
        elif self.mode == 'wave_grid':
            self._paint_wave_grid()
        else:
            # fallback: clear background
            self.fillBackground(event)

    def fillBackground(self, event):
        """Optional: clear the widget if mode is unknown."""
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

    # ======================
    # Original waveform
    # ======================
    def _paint_wave(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Colors
        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")
        dim = QColor("#2f9d57")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        # Audio
        audio = self.player.last_audio.astype(np.float32) / 32768.0
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        mid_y = r.center().y()
        p.setPen(QPen(dim, 1))
        p.drawLine(int(r.left()) + 10, int(mid_y), int(r.right()) - 10, int(mid_y))

        left, right = r.left() + 10, r.right() - 10
        amp = 0.42 * (r.height() - 20) * self.amplitude_scale

        xs = np.linspace(left, right, len(self._smoothed))
        ys = mid_y - (self._smoothed * amp)

        step = max(1, len(xs) // max(600, self.width()))
        p.setPen(QPen(neon, 2))
        for i in range(step, len(xs), step):
            p.drawLine(QPointF(xs[i - step], ys[i - step]), QPointF(xs[i], ys[i]))

        # Debug
        if self.app_ref.debug_enabled:
            self._draw_debug(p, r)

    # ======================
    # Talking lips
    # ======================
    def _paint_lips(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Colors
        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2*pad, self.height() - 2*pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        # Audio (envelope)
        audio = self.player.last_audio.astype(np.float32) / 32768.0
        audio = np.abs(audio)
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        mid_y = r.center().y()
        left, right = r.left() + 22, r.right() - 22
        max_open = 0.38 * r.height() * self.amplitude_scale
        base_open = 6

        xs = np.linspace(left, right, len(self._smoothed))
        opens = base_open + self._smoothed * max_open

        # Lips
        top = QPainterPath()
        bottom = QPainterPath()
        top.moveTo(xs[0], mid_y - opens[0])
        bottom.moveTo(xs[0], mid_y + opens[0])
        curvature = 0.18

        for i in range(1, len(xs)):
            x = xs[i]
            curve = 1.0 - curvature * np.cos(i / len(xs) * np.pi)
            o = opens[i] * curve
            top.lineTo(x, mid_y - o)
            bottom.lineTo(x, mid_y + o)

        # Fill mouth interior
        mouth = QPainterPath(top)
        mouth.connectPath(bottom.toReversed())
        fill = QColor(neon)
        fill.setAlpha(60)
        p.setBrush(fill)
        p.setPen(Qt.NoPen)
        p.drawPath(mouth)

        # Lip outlines
        p.setPen(QPen(neon, 2.5))
        p.setBrush(Qt.NoBrush)
        p.drawPath(top)
        p.drawPath(bottom)

        # Debug
        if self.app_ref.debug_enabled:
            self._draw_debug(p, r)

    # ======================
    # Neon Circles
    # ======================
    def _paint_circles(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Colors
        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        # Audio
        audio = self.player.last_audio.astype(np.float32) / 32768.0
        audio = np.abs(audio)
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        # Layout
        circle_count = 16
        left, right = r.left() + 20, r.right() - 20
        xs = np.linspace(left, right, circle_count)
        mid_y = r.center().y()

        max_radius = 0.35 * r.height() * self.amplitude_scale

        p.setPen(Qt.NoPen)
        p.setBrush(neon)

        for i, x in enumerate(xs):
            idx = int(i * len(self._smoothed) / circle_count)
            radius = 6 + self._smoothed[idx] * max_radius
            p.drawEllipse(QPointF(x, mid_y), radius, radius)
            
        # Debug
        if self.app_ref.debug_enabled:
            self._draw_debug(p, r)

    # ======================
    # Ribbon Wave 
    # ======================
    def _paint_ribbon(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Colors
        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")
        dim = QColor("#2f9d57")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        # Audio
        audio = self.player.last_audio.astype(np.float32) / 32768.0
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        left, right = r.left() + 10, r.right() - 10
        mid_y = r.center().y()
        amp = 0.42 * (r.height() - 20) * self.amplitude_scale

        xs = np.linspace(left, right, len(self._smoothed))

        path = QPainterPath()
        path.moveTo(xs[0], mid_y)

        for i in range(len(xs)):
            path.lineTo(xs[i], mid_y - self._smoothed[i] * amp)

        for i in reversed(range(len(xs))):
            path.lineTo(xs[i], mid_y + self._smoothed[i] * amp)

        path.closeSubpath()

        grad = QLinearGradient(r.left(), r.top(), r.right(), r.bottom())
        grad.setColorAt(0.0, neon)
        grad.setColorAt(1.0, dim)

        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(grad))
        p.drawPath(path)
        
        # Debug
        if self.app_ref.debug_enabled:
            self._draw_debug(p, r)
        
    # ======================
    # Radial Bars 
    # ======================
    def _paint_radial(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Colors
        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        # Audio
        audio = self.player.last_audio.astype(np.float32) / 32768.0
        audio = np.abs(audio)
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        # Geometry
        cx, cy = r.center().x(), r.center().y()
        r_base = min(r.width(), r.height()) * 0.28

        spike_count = 64
        angles = np.linspace(0, 2 * np.pi, spike_count, endpoint=False)

        p.setPen(QPen(neon, 2))

        for i, angle in enumerate(angles):
            idx = int(i * len(self._smoothed) / spike_count)
            amp = self._smoothed[idx]

            r_outer = r_base + amp * r_base * self.amplitude_scale

            x1 = cx + r_base * np.cos(angle)
            y1 = cy + r_base * np.sin(angle)
            x2 = cx + r_outer * np.cos(angle)
            y2 = cy + r_outer * np.sin(angle)

            p.drawLine(QPointF(x1, y1), QPointF(x2, y2))
            
        # Debug
        if self.app_ref.debug_enabled:
            self._draw_debug(p, r)
            
    # ======================
    # Particles
    # ======================
    def _paint_particles(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Colors
        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        # Audio
        audio = self.player.last_audio.astype(np.float32) / 32768.0
        audio = np.abs(audio)
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        # Particles
        particle_count = 32
        xs = np.linspace(r.left() + 20, r.right() - 20, particle_count)
        mid_y = r.center().y()

        max_y_offset = 0.35 * r.height() * self.amplitude_scale
        max_radius = 12 * self.amplitude_scale

        p.setPen(Qt.NoPen)
        p.setBrush(neon)

        for i, x in enumerate(xs):
            idx = int(i * len(self._smoothed) / particle_count)
            amp = self._smoothed[idx]

            y = mid_y + amp * max_y_offset
            radius = 4 + amp * max_radius

            p.drawEllipse(QPointF(x, y), radius, radius)
            
        # Debug
        if self.app_ref.debug_enabled:
            self._draw_debug(p, r)

    # ======================
    # Wave Grid
    # ======================
    def _paint_wave_grid(self):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Colors
        bg = QColor("#141414")
        panel = QColor("#1a1a1a")
        border = QColor("#2e7d4f")
        neon = QColor("#4cff7a")

        p.fillRect(self.rect(), bg)

        pad = 14
        r = QRectF(pad, pad, self.width() - 2 * pad, self.height() - 2 * pad)
        p.setPen(QPen(border, 2))
        p.setBrush(panel)
        p.drawRoundedRect(r, 16, 16)

        # Audio
        audio = self.player.last_audio.astype(np.float32) / 32768.0
        self._smoothed = (1 - self._smooth_alpha) * self._smoothed + self._smooth_alpha * audio

        # Grid waves
        wave_count = 5
        left, right = r.left() + 10, r.right() - 10
        xs = np.linspace(left, right, len(self._smoothed))

        base_amp = 0.18 * r.height() * self.amplitude_scale

        for j in range(wave_count):
            t = j / max(1, wave_count - 1)
            offset_y = r.top() + r.height() * (0.2 + t * 0.6)

            # Slight depth fade
            color = QColor(neon)
            color.setAlpha(int(220 - t * 120))

            p.setPen(QPen(color, 1.8 - t * 0.8))

            for i in range(1, len(xs)):
                y1 = offset_y - self._smoothed[i - 1] * base_amp * (j + 1)
                y2 = offset_y - self._smoothed[i] * base_amp * (j + 1)
                p.drawLine(QPointF(xs[i - 1], y1), QPointF(xs[i], y2))
                
        # Debug
        if self.app_ref.debug_enabled:
            self._draw_debug(p, r)

    # ======================
    # Debug overlay
    # ======================
    def _draw_debug(self, p, r):
        neon = QColor("#4cff7a")
        p.setPen(neon)
        p.setFont(QFont("", 10))
        y = int(r.top()) + 22
        for line in (
            f"BUSY: {self.app_ref.busy}",
            f"HISTORY: {len(self.app_ref.chat_history)} msgs",
            f"KOBOLD: {self.app_ref.ollama_tps:.1f} tk/s" if self.app_ref.ollama_tps else "KOBOLD: --",
            f"TTS TTFB: {int(self.app_ref.tts_ttfb * 1000)} ms" if self.app_ref.tts_ttfb else "TTS TTFB: --",
        ):
            p.drawText(int(r.left()) + 16, y, line)
            y += 16


# ======================
# TEXT EDIT
# ======================

class SendTextEdit(QTextEdit):
    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.parent_app = parent_app


# ======================
# MAIN APP
# ======================

class App(QWidget):
    # Signal to restart timer from background thread
    restart_auto_continue_signal = Signal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ecko v0.3")
        self.resize(860, 700)

        self.settings = QSettings("Ecko", "Ecko-GUI-v0.3")

        self.player = PCMPlayer()
        self.user_gain = 1.5
        self.busy = False
        self.debug_enabled = False

        self.ollama_tps = 0.0
        self.tts_ttfb = 0.0

        self.pipeline_sem = threading.Semaphore(1)
        self.chat_history = []

        self._stt_text = ""
        self._stt_index = 0
        
        # Interruption tracking
        self._current_generation = None  # Text being generated
        self._tts_interrupted = False
        self._interrupt_point = 0  # How much was spoken before interrupt
        self._stop_tts = False  # Flag to stop TTS streaming
        
        # Thread control for interruption
        self._pipeline_thread = None
        
        # Auto-continue modes
        self.auto_continue_mode = "standard"  # 'standard', 'aggressive', 'relaxed'
        
        # ======================
        # AUTO-CONTINUE TIMER
        # ======================
        self.auto_continue_enabled = True
        self._last_auto_continue = 0.0
        self.auto_continue_timer = QTimer(self)
        self.auto_continue_timer.setSingleShot(True)  # Changed to single-shot
        self.auto_continue_timer.timeout.connect(self.auto_continue)
        
        # Randomize initial interval
        self._set_random_auto_continue_interval()

        self.stt = STTEngine(model="base", device="cpu")

        self.mic_enabled = False
        self.ptt_enabled = False
        self.ptt_active = False

        # Connect restart signal
        self.restart_auto_continue_signal.connect(self._restart_auto_continue_timer)

        self.build_ui()
        self.initial_load()
        
        # Start auto-continue timer after initialization
        if self.auto_continue_enabled:
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()

    # =========================
    # AUTO-CONTINUE (FIXED)
    # =========================
    def _set_random_auto_continue_interval(self):
        """Set random interval based on current mode"""
        if self.auto_continue_mode == "aggressive":
            interval = random.randint(10_000, 20_000)  # 10-20 seconds
        elif self.auto_continue_mode == "relaxed":
            interval = random.randint(45_000, 75_000)  # 45-75 seconds
        else:  # standard
            interval = random.randint(30_000, 60_000)  # 30-60 seconds
        
        self.auto_continue_timer.setInterval(interval)
        print(f"[AUTO-CONTINUE] Mode: {self.auto_continue_mode}, Next trigger in {interval/1000:.1f} seconds")

    def _restart_auto_continue_timer(self):
        """Restart timer - must be called from main GUI thread"""
        print("[AUTO-CONTINUE] _restart_auto_continue_timer executing on main thread...")
        if self.auto_continue_enabled:
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()
            print(f"[AUTO-CONTINUE] Timer restarted. Active: {self.auto_continue_timer.isActive()}, Interval: {self.auto_continue_timer.interval()/1000:.1f}s")
        else:
            print("[AUTO-CONTINUE] Not restarting - feature disabled")

    def auto_continue(self):
        """
        Enhanced context-aware auto-continue with intelligent triggering.
        Uses mode-based intervals and context analysis.
        """
        print(f"\n[AUTO-CONTINUE] ===== TIMER FIRED =====")
        print(f"[AUTO-CONTINUE] Mode: {self.auto_continue_mode}")
        print(f"[AUTO-CONTINUE] Enabled: {self.auto_continue_enabled}")
        print(f"[AUTO-CONTINUE] Busy: {self.busy}")
        print(f"[AUTO-CONTINUE] History length: {len(self.chat_history)}")
        print(f"[AUTO-CONTINUE] Text box empty: {not self.text_box.toPlainText().strip()}")
        
        # Don't continue if disabled
        if not self.auto_continue_enabled:
            print("[AUTO-CONTINUE] SKIPPED: Feature disabled")
            return

        # Don't continue if no conversation exists
        if len(self.chat_history) < 2:
            print("[AUTO-CONTINUE] SKIPPED: Insufficient history")
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()
            return

        # Don't continue if currently busy
        if self.busy:
            print("[AUTO-CONTINUE] SKIPPED: System busy")
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()
            return

        # Look at the last few messages for context
        recent = self.chat_history[-6:]  # Increased from 4 to 6 for better context
        last_assistant = next((m for m in reversed(recent) if m["role"] == "assistant"), None)

        if not last_assistant:
            print("[AUTO-CONTINUE] SKIPPED: No assistant message found")
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()
            return

        # Analyze the last message for context
        last_text = last_assistant["content"].strip()
        print(f"[AUTO-CONTINUE] Last message (last 80 chars): ...{last_text[-80:]}")

        # Don't trigger if user has typed
        if self.text_box.toPlainText().strip():
            print("[AUTO-CONTINUE] SKIPPED: User has text in box")
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()
            return

        # === CONTEXT ANALYSIS ===
        # Aggressive mode: skip context analysis, always continue
        if self.auto_continue_mode == "aggressive":
            print("[AUTO-CONTINUE] Aggressive mode - skipping context analysis")
            continuation_prompt = random.choice([
                "*continues speaking*",
                "*keeps talking*",
                "*goes on*"
            ])
        else:
            # Standard/Relaxed: use context analysis
            continuation_prompt = self._analyze_context_and_build_prompt(recent, last_text)
            
            if continuation_prompt is None:
                print("[AUTO-CONTINUE] SKIPPED: Context analysis suggests not continuing")
                self._set_random_auto_continue_interval()
                self.auto_continue_timer.start()
                return

        print(f"[AUTO-CONTINUE] Using continuation prompt: {continuation_prompt}")

        # Try to acquire semaphore - if busy, just skip this cycle
        if not self.pipeline_sem.acquire(blocking=False):
            print("[AUTO-CONTINUE] SKIPPED: Couldn't acquire semaphore")
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()
            return

        self._last_auto_continue = time.time()
        
        print("[AUTO-CONTINUE] ✓✓✓ EXECUTING CONTINUATION ✓✓✓")

        # Run continuation in background thread
        def safe_continue():
            try:
                print("[AUTO-CONTINUE] Calling run_pipeline...")
                self.run_pipeline(continuation_prompt)
                print("[AUTO-CONTINUE] run_pipeline returned")
            except Exception as e:
                print(f"[AUTO-CONTINUE] ERROR: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print("[AUTO-CONTINUE] In finally block, releasing semaphore...")
                self.pipeline_sem.release()
                print(f"[AUTO-CONTINUE] Semaphore released. Auto-continue enabled: {self.auto_continue_enabled}")
                
                # Emit signal to restart timer on main thread
                if self.auto_continue_enabled:
                    print("[AUTO-CONTINUE] Emitting restart signal...")
                    self.restart_auto_continue_signal.emit()
                else:
                    print("[AUTO-CONTINUE] Not restarting - feature disabled")

        threading.Thread(target=safe_continue, daemon=True).start()

    def _analyze_context_and_build_prompt(self, recent_messages, last_text):
        """
        Analyze conversation context to determine if and how to continue.
        Returns a continuation prompt or None if shouldn't continue.
        """
        # === CHECK 1: Message completeness ===
        # Questions or statements ending with question marks suggest waiting for user
        if last_text.endswith('?'):
            print("[AUTO-CONTINUE] Context: Last message is a question - might wait for user")
            # Only continue 30% of the time for questions
            if random.random() > 0.3:
                return None
        
        # === CHECK 2: Conversation flow patterns ===
        # Count recent turns
        user_count = sum(1 for m in recent_messages if m["role"] == "user")
        assistant_count = sum(1 for m in recent_messages if m["role"] == "assistant")
        
        print(f"[AUTO-CONTINUE] Recent turns - User: {user_count}, Assistant: {assistant_count}")
        
        # If assistant has spoken 3+ times in a row, maybe give user a chance
        if assistant_count >= 3 and user_count == 0:
            print("[AUTO-CONTINUE] Context: Assistant dominating conversation")
            # Only continue 40% of the time
            if random.random() > 0.4:
                return None
        
        # === CHECK 3: Message length analysis ===
        # Very short messages might be waiting for user response
        if len(last_text) < 30:
            print(f"[AUTO-CONTINUE] Context: Short message ({len(last_text)} chars)")
            # Only continue 50% of the time for short messages
            if random.random() > 0.5:
                return None
        
        # === CHECK 4: Conversation energy/engagement ===
        # Look for engagement cues
        engagement_keywords = ['what about you', 'your turn', 'you think', 'tell me', 
                               'how about', 'would you', 'do you', 'have you']
        
        last_text_lower = last_text.lower()
        has_engagement_cue = any(keyword in last_text_lower for keyword in engagement_keywords)
        
        if has_engagement_cue:
            print("[AUTO-CONTINUE] Context: Engagement cue detected - inviting user response")
            # Only continue 20% of the time when inviting user input
            if random.random() > 0.2:
                return None
        
        # === CHECK 5: Ellipsis or trailing thoughts ===
        # Messages ending with ... or , suggest incomplete thought
        seems_incomplete = last_text.endswith(('...', ',', ';', ':', '-', 'and', 'but', 'or'))
        
        if seems_incomplete:
            print("[AUTO-CONTINUE] Context: Message seems incomplete - good to continue")
        
        # === BUILD CONTEXT-AWARE PROMPT ===
        # Choose prompt based on context
        prompts = []
        
        if seems_incomplete:
            prompts.extend([
                "*continues the thought*",
                "*elaborates further*",
                "*keeps talking*"
            ])
        elif len(last_text) > 200:
            prompts.extend([
                "*pauses briefly then continues*",
                "*adds another thought*",
                "*continues speaking*"
            ])
        else:
            prompts.extend([
                "*continues the conversation naturally*",
                "*shares more*",
                "*keeps the conversation going*",
                "*continues talking*"
            ])
        
        # Return a random appropriate prompt
        return random.choice(prompts)

    def finish_stt(self):
        """Called after PTT ends to gradually type the transcribed text and auto-send."""
        self._stt_text = self.stt.transcribe()
        self._stt_index = 0
        if not self._stt_text:
            return

        def type_step():
            if self._stt_index < len(self._stt_text):
                self.text_box.moveCursor(QTextCursor.End)
                self.text_box.insertPlainText(self._stt_text[self._stt_index])
                self._stt_index += 1
                QTimer.singleShot(30, type_step)
            else:
                self.text_box.moveCursor(QTextCursor.End)
                QTimer.singleShot(50, self.on_send)

        type_step()

    def get_character_dir(self):
        path = os.path.join(os.getcwd(), "characters")
        os.makedirs(path, exist_ok=True)
        return path

    def on_character_selected(self, index):
        if index <= 0:
            return

        rel_path = self.char_box.itemData(index)
        path = os.path.join(self.get_character_dir(), rel_path)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.apply_character_data(data)
            self.reset_chat()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_wave_mode_changed(self, index):
        """Update wave display mode when dropdown changes."""
        modes = ['wave', 'lips', 'circles', 'ribbon', 'radial', 'particles', 'wave_grid']
        if 0 <= index < len(modes):
            self.wave.set_mode(modes[index])
            self.settings.setValue("wave_mode", modes[index])

    def on_wave_amplitude_changed(self, value):
        """Update wave amplitude scale (slider 10-200 -> scale 0.1-2.0)"""
        scale = value / 100.0
        self.wave.set_amplitude_scale(scale)
        self.settings.setValue("wave_amplitude", value)

    def on_auto_continue_mode_changed(self, mode_text):
        """Update auto-continue mode"""
        self.auto_continue_mode = mode_text.lower()
        self.settings.setValue("auto_continue_mode", self.auto_continue_mode)
        
        # Update timer with new interval
        if self.auto_continue_enabled and self.auto_continue_timer.isActive():
            self._set_random_auto_continue_interval()
            # Don't need to restart - it will use new interval on next trigger
        
        print(f"[AUTO-CONTINUE] Mode changed to: {self.auto_continue_mode}")

    def load_characters(self):
        self.char_box.blockSignals(True)
        self.char_box.clear()
        self.char_box.addItem("— Select Character —")

        char_dir = self.get_character_dir()

        def add_folder(folder, depth=0):
            """Recursively add folder headers and JSON files"""
            folder_name = os.path.basename(folder)
            
            # Skip 'unsorted' folders
            if folder_name.lower() == "unsorted":
                return

            # Add folder as a header (skip root)
            if depth > 0:
                display_name = "    " * (depth - 1) + folder_name
                self.char_box.addItem(display_name)
                index = self.char_box.count() - 1
                item = self.char_box.model().item(index)
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)

            # Add JSON files and recurse
            for f in sorted(os.listdir(folder)):
                path = os.path.join(folder, f)
                if os.path.isdir(path):
                    add_folder(path, depth + 1)
                elif f.lower().endswith(".json"):
                    rel_path = os.path.relpath(path, char_dir)
                    display_name = "    " * depth + os.path.splitext(f)[0]
                    self.char_box.addItem(display_name, rel_path)

        add_folder(char_dir)
        self.char_box.blockSignals(False)
    
    def build_prompt(self):
        prompt = ""

        for msg in self.chat_history:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                prompt += f"[SYSTEM]\n{content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"

        prompt += "Assistant:"
        return prompt

    def get_kv_scaling(self):
        if not self.kv_scale_btn.isChecked():
            return None

        try:
            val = float(self.kv_scale_input.text())
            return val
        except ValueError:
            return None

    def toggle_mic(self, enabled):
        self.mic_enabled = enabled

        if not enabled:
            self.ptt_active = False
            self.stt.stop()
            self.stt.audio_q.queue.clear()
            self.ptt_btn.setChecked(False)

    def toggle_ptt(self, enabled):
        # PTT button is now purely indicator; do not change anything manually
        pass

    def toggle_auto_continue(self, enabled):
        self.auto_continue_enabled = enabled
        if enabled:
            print(f"[AUTO-CONTINUE] Enabling feature...")
            self._set_random_auto_continue_interval()
            self.auto_continue_timer.start()
            print(f"[AUTO-CONTINUE] Timer started with {self.auto_continue_timer.interval()/1000:.1f}s interval")
            print(f"[AUTO-CONTINUE] Timer active: {self.auto_continue_timer.isActive()}")
        else:
            print("[AUTO-CONTINUE] Disabling feature...")
            self.auto_continue_timer.stop()
            print(f"[AUTO-CONTINUE] Timer stopped. Active: {self.auto_continue_timer.isActive()}")
        
    def eventFilter(self, obj, event):
        # ------------------ ENTER TO SEND ------------------
        if (
            event.type() == QEvent.KeyPress
            and obj is self.text_box
            and self.enter_send_btn.isChecked()
            and event.key() in (Qt.Key_Return, Qt.Key_Enter)
            and not (event.modifiers() & Qt.ShiftModifier)
        ):
            event.accept()
            self.on_send()
            return True

        # ------------------ INTERRUPT WITH ESC - DISABLED ------------------
        # ESC interrupt behavior is disabled
        # if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Escape:
        #     if self.busy:
        #         print("[INTERRUPT] ESC pressed - interrupting current speech")
        #         self.interrupt_speech()
        #         return True

        # ------------------ PTT (ALT) - ENABLED via keyboard only ------------------
        # PTT button is hidden but Alt key functionality is preserved
        if self.mic_enabled:
            if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Alt and not self.ptt_active:
                self.ptt_active = True
                self.ptt_btn.setChecked(True)
                self.stt.start()

            elif event.type() == QEvent.KeyRelease and event.key() == Qt.Key_Alt and self.ptt_active:
                self.ptt_active = False
                self.ptt_btn.setChecked(False)
                self.stt.stop()
                self.finish_stt()
                return True

        return super().eventFilter(obj, event)

    def interrupt_speech(self):
        """Interrupt current TTS playback - simplified version"""
        if not self.busy:
            print("[INTERRUPT] Not currently speaking")
            return
        
        print("[INTERRUPT] Setting interrupt flags...")
        self._stop_tts = True
        self._tts_interrupted = True
        
        # Stop audio immediately
        self.player.stop_playback()
        
        # Calculate how much was spoken (estimate based on time)
        if self._current_generation:
            elapsed = time.time() - self._generation_start_time
            chars_spoken = int(elapsed * 12.5)
            self._interrupt_point = min(chars_spoken, len(self._current_generation))
            
            print(f"[INTERRUPT] Estimated {self._interrupt_point}/{len(self._current_generation)} chars spoken")
        
        print("[INTERRUPT] Interrupt signal sent")

    # ---------- UI ----------

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # === TOP ROW ===
        top = QHBoxLayout()
        self.ollama_led = StatusLED()
        self.tts_led = StatusLED()
        
        top.addWidget(self.tts_led)
        top.addWidget(QLabel("Echo-TTS"))
        top.addSpacing(10)
        top.addWidget(self.ollama_led)
        top.addWidget(QLabel("KoboldCPP"))
        top.addStretch(1)

        self.agc_mode_box = QComboBox()
        self.agc_mode_box.addItems(["RMS", "Peak"])
        self.agc_mode_box.setFixedWidth(80)
        top.addWidget(self.agc_mode_box)

        self.agc_mode_box.currentTextChanged.connect(
            lambda t: setattr(self.player, "agc_mode", t.lower())
        )

        self.agc_btn = QPushButton("Gain")
        self.agc_btn.setCheckable(True)
        self.agc_btn.toggled.connect(lambda s: setattr(self.player, "auto_gain", s))
        top.addWidget(self.agc_btn)

        top.addWidget(QLabel("VOL"))
        self.vol_slider = QSlider(Qt.Horizontal)
        self.vol_slider.setRange(50, 300)
        self.vol_slider.setValue(150)
        self.vol_slider.setFixedWidth(160)
        self.vol_slider.valueChanged.connect(lambda v: setattr(self, "user_gain", v / 100))
        top.addWidget(self.vol_slider)

        self.debug_btn = QPushButton("Debug")
        self.debug_btn.setCheckable(True)
        self.debug_btn.toggled.connect(lambda s: setattr(self, "debug_enabled", s))
        top.addWidget(self.debug_btn)

        # ---- KV Scaling controls ----
        self.kv_scale_btn = QPushButton("KV Scale")
        self.kv_scale_btn.setCheckable(True)
        top.addWidget(self.kv_scale_btn)

        self.kv_scale_input = QLineEdit("1.25")
        self.kv_scale_input.setFixedWidth(60)
        self.kv_scale_input.setAlignment(Qt.AlignCenter)
        self.kv_scale_input.setToolTip("speaker_kv_scale")
        top.addWidget(self.kv_scale_input)
        self.kv_scale_input.setEnabled(False)
        self.kv_scale_btn.toggled.connect(self.kv_scale_input.setEnabled)

        self.model_box = QComboBox()
        self.voice_box = QComboBox()
        self.model_box.setFixedWidth(180)
        self.voice_box.setFixedWidth(140)

        self.model_box.currentTextChanged.connect(lambda t: self.settings.setValue("last_model", t))
        self.voice_box.currentTextChanged.connect(lambda t: self.settings.setValue("last_voice", t))

        self.model_box.setEditable(True)
        self.model_box.lineEdit().setReadOnly(True)
        
        top.addWidget(QLabel("Model"))
        top.addWidget(self.model_box)
        top.addWidget(QLabel("Voice"))
        top.addWidget(self.voice_box)

        root.addLayout(top)

        # === WAVE CONTROLS ROW ===
        wave_controls = QHBoxLayout()
        
        wave_controls.addWidget(QLabel("Wave Mode:"))
        self.wave_mode_box = QComboBox()
        self.wave_mode_box.addItems(['Wave', 'Lips', 'Circles', 'Ribbon', 'Radial', 'Particles', 'Wave Grid'])
        self.wave_mode_box.setFixedWidth(120)
        self.wave_mode_box.currentIndexChanged.connect(self.on_wave_mode_changed)
        wave_controls.addWidget(self.wave_mode_box)
        
        wave_controls.addSpacing(10)
        wave_controls.addWidget(QLabel("Wave Amp:"))
        self.wave_amp_slider = QSlider(Qt.Horizontal)
        self.wave_amp_slider.setRange(10, 200)  # 0.1x to 2.0x
        self.wave_amp_slider.setValue(100)  # 1.0x default
        self.wave_amp_slider.setFixedWidth(120)
        self.wave_amp_slider.valueChanged.connect(self.on_wave_amplitude_changed)
        wave_controls.addWidget(self.wave_amp_slider)
        
        self.wave_amp_label = QLabel("1.0x")
        self.wave_amp_label.setFixedWidth(40)
        wave_controls.addWidget(self.wave_amp_label)
        
        # Update label when slider changes
        self.wave_amp_slider.valueChanged.connect(
            lambda v: self.wave_amp_label.setText(f"{v/100:.1f}x")
        )
        
        wave_controls.addStretch(1)
        root.addLayout(wave_controls)

        self.wave = WaveDisplay(self.player, self)
        root.addWidget(self.wave, 1)

        self.text_box = SendTextEdit(self)
        self.text_box.setPlaceholderText("Type text to speak…")
        self.text_box.setMinimumHeight(120)
        self.text_box.installEventFilter(self)
        QApplication.instance().installEventFilter(self)

        self.ptt_btn = QPushButton("PTT")
        self.ptt_btn.setCheckable(True)
        self.ptt_btn.setEnabled(False)  # DISABLED
        self.ptt_btn.setVisible(False)  # HIDDEN
        
        root.addWidget(self.text_box)

        # ---- bottom bar (2 rows) ----
        bottom_layout = QVBoxLayout()

        # --- Row 1: Chat / Character controls ---
        row1 = QHBoxLayout()
        self.enter_send_btn = QPushButton("Enter to Send")
        self.enter_send_btn.setCheckable(True)
        row1.addWidget(self.enter_send_btn)

        row1.addSpacing(12)

        self.save_char_btn = QPushButton("Save Character")
        self.save_char_btn.clicked.connect(self.save_character)
        row1.addWidget(self.save_char_btn)

        row1.addWidget(QLabel("Character"))
        self.char_box = QComboBox()
        self.char_box.setFixedWidth(200)
        self.char_box.currentIndexChanged.connect(self.on_character_selected)
        row1.addWidget(self.char_box)

        self.reset_chat_btn = QPushButton("Reset Chat")
        self.reset_chat_btn.clicked.connect(self.reset_chat)
        row1.addWidget(self.reset_chat_btn)

        row1.addStretch(1)
        bottom_layout.addLayout(row1)

        # --- Row 2: Mic / PTT / Speak ---
        row2 = QHBoxLayout()
        self.mic_btn = QPushButton("🎤 Mic")
        self.mic_btn.setCheckable(True)
        self.mic_btn.toggled.connect(self.toggle_mic)
        row2.addWidget(self.mic_btn)

        row2.addWidget(self.ptt_btn)

        row2.addStretch(1)
        
        # Auto-continue controls
        self.auto_continue_btn = QPushButton("Auto ▶")
        self.auto_continue_btn.setCheckable(True)
        self.auto_continue_btn.setChecked(True)
        self.auto_continue_btn.toggled.connect(self.toggle_auto_continue)
        row2.addWidget(self.auto_continue_btn)
        
        self.auto_continue_mode_box = QComboBox()
        self.auto_continue_mode_box.addItems(['Standard', 'Aggressive', 'Relaxed'])
        self.auto_continue_mode_box.setFixedWidth(100)
        self.auto_continue_mode_box.currentTextChanged.connect(self.on_auto_continue_mode_changed)
        row2.addWidget(self.auto_continue_mode_box)
        
        row2.addSpacing(10)
        
        self.interrupt_btn = QPushButton("⏹ Stop (ESC)")
        self.interrupt_btn.clicked.connect(self.interrupt_speech)
        self.interrupt_btn.setEnabled(False)  # DISABLED
        self.interrupt_btn.setVisible(False)  # HIDDEN
        row2.addWidget(self.interrupt_btn)
        
        self.send_btn = QPushButton("Speak ▶")
        self.send_btn.clicked.connect(self.on_send)
        row2.addWidget(self.send_btn)

        bottom_layout.addLayout(row2)
        root.addLayout(bottom_layout)

        # --- System prompt toggle ---
        self.sys_toggle = QPushButton("System Prompt ▼")
        self.sys_toggle.setCheckable(True)
        self.sys_toggle.toggled.connect(self.toggle_system_prompt)
        root.addWidget(self.sys_toggle)

        self.system_prompt = QTextEdit()
        self.system_prompt.setVisible(False)
        self.system_prompt.setMinimumHeight(80)
        root.addWidget(self.system_prompt)

        self.setStyleSheet("""
            QWidget { background-color: #141414; color: #4cff7a; }
            QTextEdit, QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #2e7d4f;
                border-radius: 10px;
                padding: 6px 8px;
            }
            QPushButton {
                background-color: #1e1e1e;
                border: 1px solid #2e7d4f;
                border-radius: 10px;
                padding: 8px 12px;
            }
            QPushButton:checked {
                background-color: #234b34;
                border: 1px solid #4cff7a;
            }
            QSlider::groove:horizontal {
                border: 1px solid #2e7d4f;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #4cff7a;
                border-radius: 3px;
            }
            QSlider::add-page:horizontal {
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #4cff7a;
                border: 1px solid #2e7d4f;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """)

    def toggle_system_prompt(self, expanded):
        self.system_prompt.setVisible(expanded)
        self.sys_toggle.setText("System Prompt ▲" if expanded else "System Prompt ▼")

    # ---------- LOAD ----------

    def initial_load(self):
        self.load_kobold_status()
        self.load_voices()
        self.load_characters()

        self.vol_slider.setValue(int(self.settings.value("volume", 150)))
        self.user_gain = self.vol_slider.value() / 100
        self.agc_btn.setChecked(self.settings.value("agc", False, type=bool))
        self.debug_btn.setChecked(self.settings.value("debug", False, type=bool))
        self.enter_send_btn.setChecked(self.settings.value("enter_send", False, type=bool))
        self.system_prompt.setPlainText(self.settings.value("system_prompt", ""))

        # Load wave display settings
        saved_mode = self.settings.value("wave_mode", "ribbon")
        mode_index = ['wave', 'lips', 'circles', 'ribbon', 'radial', 'particles', 'wave_grid'].index(saved_mode) if saved_mode in ['wave', 'lips', 'circles', 'ribbon', 'radial', 'particles', 'wave_grid'] else 3
        self.wave_mode_box.setCurrentIndex(mode_index)
        
        saved_amp = int(self.settings.value("wave_amplitude", 100))
        self.wave_amp_slider.setValue(saved_amp)
        self.wave.set_amplitude_scale(saved_amp / 100.0)
        
        # Load auto-continue mode
        saved_mode = self.settings.value("auto_continue_mode", "standard")
        mode_index = ['standard', 'aggressive', 'relaxed'].index(saved_mode) if saved_mode in ['standard', 'aggressive', 'relaxed'] else 0
        self.auto_continue_mode_box.setCurrentIndex(mode_index)
        self.auto_continue_mode = saved_mode

        size = self.settings.value("window_size")
        if size:
            self.resize(size)

    def load_kobold_status(self):
        try:
            r = requests.get(f"{KOBOLD_BASE}/api/v1/model", timeout=3)
            model_name = r.json().get("result", "Unknown model")

            self.model_box.blockSignals(True)
            self.model_box.clear()
            self.model_box.addItem(model_name)
            self.model_box.blockSignals(False)

            self.ollama_led.set_color("#4cff7a")
        except Exception:
            self.ollama_led.set_color("#3a1414")

    def load_voices(self):
        self.voice_box.blockSignals(True)
        self.voice_box.clear()
        try:
            r = requests.get(f"{TTS_BASE}/v1/voices", timeout=5)
            voices = sorted(v["id"] for v in r.json().get("data", []))
            self.voice_box.addItems(voices)
            last = self.settings.value("last_voice", "")
            if last in voices:
                self.voice_box.setCurrentText(last)
            self.tts_led.set_color("#4cff7a")
        except Exception:
            self.tts_led.set_color("#3a1414")
        finally:
            self.voice_box.blockSignals(False)

    # =========================
    # CHARACTER PRESETS
    # =========================

    def get_character_data(self):
        return {
            "system_prompt": self.system_prompt.toPlainText(),
            "voice": self.voice_box.currentText(),
            "volume": self.vol_slider.value(),
            "agc": self.agc_btn.isChecked(),
            "agc_mode": self.agc_mode_box.currentText(),
            "kv_scale_enabled": self.kv_scale_btn.isChecked(),
            "kv_scale_value": self.kv_scale_input.text(),
        }

    def apply_character_data(self, data):
        if "system_prompt" in data:
            self.system_prompt.setPlainText(data["system_prompt"])

        if "voice" in data:
            idx = self.voice_box.findText(data["voice"])
            if idx != -1:
                self.voice_box.setCurrentIndex(idx)

        if "volume" in data:
            self.vol_slider.setValue(int(data["volume"]))

        if "agc" in data:
            self.agc_btn.setChecked(bool(data["agc"]))
            
        if "agc_mode" in data:
            self.agc_mode_box.setCurrentText(data["agc_mode"])

        if "kv_scale_enabled" in data:
            self.kv_scale_btn.setChecked(bool(data["kv_scale_enabled"]))

        if "kv_scale_value" in data:
            self.kv_scale_input.setText(str(data["kv_scale_value"]))

    def save_character(self):
        base_dir = os.path.join(os.getcwd(), "characters")
        os.makedirs(base_dir, exist_ok=True)

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Character",
            base_dir,
            "Character Preset (*.json)"
        )
        if not path:
            return

        if not path.lower().endswith(".json"):
            path += ".json"

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.get_character_data(), f, indent=2)
            self.load_characters()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def reset_chat(self):
        if QMessageBox.question(
            self,
            "Reset Chat",
            "Clear conversation history?",
            QMessageBox.Yes | QMessageBox.No
        ) == QMessageBox.Yes:
            self.chat_history.clear()
            self.ollama_tps = 0.0
            self.tts_ttfb = 0.0

    # ---------- ACTION ----------
    def on_send(self):
        """
        Fixed version: properly handles semaphore acquisition/release and timer restart
        Interrupt behavior DISABLED
        """
        print(f"[ON_SEND] User sending message...")
        
        # INTERRUPT BEHAVIOR DISABLED - Code preserved but commented out
        # Check if we're interrupting current speech
        # was_interrupted = False
        # if self.busy:
        #     print("[ON_SEND] Currently speaking - interrupting...")
        #     was_interrupted = True
        #     self.interrupt_speech()
        #     
        #     # Wait for pipeline to notice interrupt and exit
        #     for i in range(30):  # 3 seconds max
        #         if not self.busy:
        #             print(f"[ON_SEND] Pipeline stopped after {i * 0.1:.1f}s")
        #             break
        #         time.sleep(0.1)
        #     else:
        #         print("[ON_SEND] Warning: Pipeline still busy after 3s, continuing anyway")
        
        # Try to acquire semaphore
        acquired = False
        for attempt in range(30):  # 3 seconds total
            if self.pipeline_sem.acquire(blocking=False):
                print(f"[ON_SEND] Semaphore acquired (attempt {attempt + 1})")
                acquired = True
                break
            time.sleep(0.1)
        
        if not acquired:
            print(f"[ON_SEND] Failed to acquire semaphore after 3 seconds")
            print(f"[ON_SEND] Busy state: {self.busy}, forcing acquisition...")
            # Force acquire - this is safe because we waited 3 seconds
            self.pipeline_sem.acquire(blocking=True)
            print(f"[ON_SEND] Forced semaphore acquisition")

        text = self.text_box.toPlainText().strip()
        if not text:
            self.pipeline_sem.release()
            print(f"[ON_SEND] No text, ignoring")
            return

        self.text_box.clear()
        
        # Reset interruption flags for new pipeline
        self._stop_tts = False
        self._tts_interrupted = False
        
        # Stop auto-continue timer while processing
        self.auto_continue_timer.stop()

        def safe_pipeline():
            try:
                # INTERRUPT CONTEXT DISABLED - just run normally
                # if was_interrupted and self._current_generation:
                #     context_text = self._build_interruption_context(text)
                #     self.run_pipeline(context_text)
                # else:
                self.run_pipeline(text)
            except Exception as e:
                print(f"[ON_SEND] Pipeline error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                print(f"[ON_SEND] Pipeline thread finishing, releasing semaphore...")
                try:
                    self.pipeline_sem.release()
                    print(f"[ON_SEND] Semaphore released successfully")
                except Exception as e:
                    print(f"[ON_SEND] Error releasing semaphore: {e}")
                
                # Restart auto-continue timer on main thread after completion
                if self.auto_continue_enabled:
                    self.restart_auto_continue_signal.emit()

        self._pipeline_thread = threading.Thread(target=safe_pipeline, daemon=True)
        self._pipeline_thread.start()

    def _build_interruption_context(self, user_text):
        """Build context-aware message when user interrupts"""
        if not self._current_generation:
            return user_text
        
        interrupted_text = self._current_generation
        spoken_part = interrupted_text[:self._interrupt_point]
        unspoken_part = interrupted_text[self._interrupt_point:]
        
        print(f"[INTERRUPT] Building context...")
        print(f"[INTERRUPT] Spoken: '{spoken_part[-50:]}'")
        print(f"[INTERRUPT] Unspoken: '{unspoken_part[:50]}'")
        
        # Build a context-aware interruption message
        context_parts = []
        
        # Add what was being said
        if len(spoken_part) > 20:
            # Extract last sentence or phrase
            last_sentence = spoken_part.split('.')[-1].strip()
            if len(last_sentence) > 10:
                context_parts.append(f"[You were saying: \"{last_sentence[-60:]}...\"]")
        
        # Add interruption acknowledgment
        interruption_phrases = [
            "[User interrupts]",
            "[User cuts in]",
            "[Interrupted mid-sentence]"
        ]
        context_parts.append(random.choice(interruption_phrases))
        
        # Add user's actual message
        context_parts.append(f"User says: {user_text}")
        
        # Optional: hint about unspoken content if significant
        if len(unspoken_part) > 50:
            context_parts.append("[Note: Was about to continue but user interrupted]")
        
        final_context = " ".join(context_parts)
        print(f"[INTERRUPT] Final context: {final_context}")
        
        # Reset interruption flags
        self._tts_interrupted = False
        self._current_generation = None
        self._interrupt_point = 0
        
        return final_context

    def run_pipeline(self, text):
        """
        Fixed version: cleaner error handling and timer management
        """
        print(f"[PIPELINE] Starting pipeline with text: {text[:50]}...")
        self.busy = True
        self.ollama_tps = 0.0
        self.tts_ttfb = 0.0

        try:
            if not self.chat_history:
                sys_text = self.system_prompt.toPlainText().strip()
                if sys_text:
                    self.chat_history.append({"role": "system", "content": sys_text})

            self.chat_history.append({"role": "user", "content": text})

            if len(self.chat_history) > MAX_HISTORY_MESSAGES:
                self.chat_history = self.chat_history[-MAX_HISTORY_MESSAGES:]

            # ===== KOBOLDCPP GENERATION =====
            prompt = self.build_prompt()

            start = time.perf_counter()
            r = requests.post(
                f"{KOBOLD_BASE}/api/v1/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": 300,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop_sequence": ["User:"]
                },
                timeout=120
            )

            elapsed = time.perf_counter() - start
            reply = r.json()["results"][0]["text"].strip()
            tokens_est = max(1, len(reply) // 4)
            self.ollama_tps = tokens_est / max(elapsed, 1e-6)

            self.chat_history.append({"role": "assistant", "content": reply})

            # ===== TTS =====
            tts_start = time.perf_counter()
            first = True

            tts_payload = {
                "input": reply,
                "voice": self.voice_box.currentText(),
                "stream": True
            }

            kv_scale = self.get_kv_scaling()
            if kv_scale is not None:
                tts_payload["extra_body"] = {
                    "speaker_kv_scale": kv_scale,
                    "speaker_kv_min_t": 0.9,
                    "speaker_kv_max_layers": 24
                }

            with requests.post(
                f"{TTS_BASE}/v1/audio/speech",
                json=tts_payload,
                stream=True,
                timeout=120
            ) as resp:
                for chunk in resp.iter_content(4096):
                    if chunk:
                        if first:
                            self.tts_ttfb = time.perf_counter() - tts_start
                            first = False
                        self.player.play(chunk, self.user_gain)

        finally:
            self.busy = False
            print(f"[PIPELINE] Pipeline complete. Busy now: {self.busy}")

    def save_settings(self):
        self.settings.setValue("volume", self.vol_slider.value())
        self.settings.setValue("agc", self.agc_btn.isChecked())
        self.settings.setValue("debug", self.debug_btn.isChecked())
        self.settings.setValue("enter_send", self.enter_send_btn.isChecked())
        self.settings.setValue("system_prompt", self.system_prompt.toPlainText())
        self.settings.setValue("last_model", self.model_box.currentText())
        self.settings.setValue("last_voice", self.voice_box.currentText())
        self.settings.setValue("window_size", self.size())

    def closeEvent(self, event):
        self.save_settings()
        self.player.close()
        event.accept()


# ======================
# ENTRY
# ======================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())
