"""
Capture audio continue + segmentation en phrases/segments par détection de silence.

Cross-platform (sounddevice/PortAudio) : fonctionne à l'identique sous
Windows et macOS, seuls les noms des périphériques diffèrent (WASAPI vs
CoreAudio), géré automatiquement par sounddevice/PortAudio.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000  # attendu par Whisper
BLOCK_DURATION = 0.5  # secondes par bloc lu depuis le micro
SILENCE_RMS_THRESHOLD = 0.01  # à ajuster selon le micro / la salle
SILENCE_DURATION_TO_CUT = 1.2  # secondes de silence avant de couper un segment
MAX_SEGMENT_DURATION = 25.0  # sécurité : on coupe même sans silence au-delà


@dataclass
class AudioSegment:
    audio: np.ndarray
    sample_rate: int
    started_at: float  # timestamp epoch du début du segment


def list_input_devices() -> list[dict]:
    """Retourne la liste des micros disponibles (nom + index), pour peuplement UI."""
    devices = sd.query_devices()
    return [
        {"index": i, "name": d["name"]}
        for i, d in enumerate(devices)
        if d["max_input_channels"] > 0
    ]


class AudioRecorder:
    """
    Enregistre en continu depuis un micro donné et pousse des AudioSegment
    complets (phrase/segment de parole coupé sur silence) dans une queue,
    consommée ensuite par le thread de transcription.
    """

    def __init__(self, device_index: int | None, on_segment_ready):
        self.device_index = device_index
        self.on_segment_ready = on_segment_ready  # callback(AudioSegment)
        self._stream: sd.InputStream | None = None
        self._running = False

        self._buffer: list[np.ndarray] = []
        self._segment_started_at: float | None = None
        self._silence_accum = 0.0

    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device_index,
            blocksize=int(SAMPLE_RATE * BLOCK_DURATION),
            callback=self._on_audio_block,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._flush_segment(force=True)

    def _on_audio_block(self, indata, frames, time_info, status):
        if status:
            # xrun ou autre souci audio : on log plutôt que de planter
            print(f"[audio] status: {status}")

        block = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(block**2)))

        if rms >= SILENCE_RMS_THRESHOLD:
            # bloc "parlé"
            if self._segment_started_at is None:
                self._segment_started_at = time.time()
            self._buffer.append(block)
            self._silence_accum = 0.0
        else:
            # bloc silencieux
            if self._segment_started_at is not None:
                self._buffer.append(block)  # garde un peu de silence de fin (naturel)
                self._silence_accum += BLOCK_DURATION
                if self._silence_accum >= SILENCE_DURATION_TO_CUT:
                    self._flush_segment()

        # sécurité anti-segment trop long (le prof qui parle sans pause)
        if self._segment_started_at is not None:
            duration = len(self._buffer) * BLOCK_DURATION
            if duration >= MAX_SEGMENT_DURATION:
                self._flush_segment()

    def _flush_segment(self, force: bool = False):
        if not self._buffer or self._segment_started_at is None:
            self._buffer = []
            self._segment_started_at = None
            self._silence_accum = 0.0
            return

        audio = np.concatenate(self._buffer)
        segment = AudioSegment(
            audio=audio,
            sample_rate=SAMPLE_RATE,
            started_at=self._segment_started_at,
        )
        self._buffer = []
        self._segment_started_at = None
        self._silence_accum = 0.0

        # évite de transcrire des micro-segments (toux, bruit bref)
        if len(audio) / SAMPLE_RATE >= 0.4 or force:
            self.on_segment_ready(segment)


class TranscriptionWorker:
    """
    Thread séparé qui consomme les AudioSegment depuis une queue et appelle
    le backend de transcription, pour ne jamais bloquer le callback audio
    (qui doit rester très rapide) ni l'UI.
    """

    def __init__(self, backend, on_text_ready):
        self.backend = backend
        self.on_text_ready = on_text_ready  # callback(text: str, started_at: float)
        self._queue: queue.Queue[AudioSegment] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._running = False

    def submit(self, segment: AudioSegment):
        self._queue.put(segment)

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run(self):
        while self._running or not self._queue.empty():
            try:
                segment = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text = self.backend.transcribe(segment.audio, segment.sample_rate)
            except Exception as exc:  # on ne veut jamais planter le worker
                print(f"[transcription] erreur: {exc}")
                text = ""
            if text:
                self.on_text_ready(text, segment.started_at)
