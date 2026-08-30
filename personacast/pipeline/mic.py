"""
always on VAD driven mic capture 
"""

from __future__ import annotations

import io
import threading
import wave
from collections import deque
from typing import Callable

import numpy as np

from .. import config


class MicState:
    IDLE = "idle"
    SPEAKING = "speaking"
    DISARMED = "disarmed"


def _pcm16_to_wav_bytes(frames: list[np.ndarray], sample_rate: int) -> bytes:

    pcm = np.concatenate(frames) if frames else np.array([], dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


_silero_model = None


def _default_vad_fn(frame: np.ndarray, sample_rate: int) -> float:
    """
    lazy load silero-vad on first use, returning voice speed popropability in 0 to 1 interval for each frame
    """
    global _silero_model
    import torch

    if _silero_model is None:
        import os
        try:
            import certifi
            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        except ImportError:
            pass

        _silero_model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", trust_repo=True,
        )

    audio = torch.from_numpy(frame.astype(np.float32) / 32768.0)
    with torch.no_grad():
        return float(_silero_model(audio, sample_rate).item())


### slero vad needs minimum chunk length per call 
_SILERO_CHUNK_SAMPLES = {16000: 512, 8000: 256}


class MicSession:
    def __init__(self, on_speech_start: Callable[[], None],
                 on_utterance: Callable[[bytes], None], *,
                 sample_rate: int = config.STT_INPUT_SAMPLE_RATE,
                 onset_ms: int = config.VAD_ONSET_MS,
                 silence_ms: int = config.VAD_SILENCE_MS,
                 preroll_ms: int = config.VAD_PREROLL_MS,
                 min_utterance_ms: int = config.MIC_MIN_UTTERANCE_MS,
                 max_utterance_ms: int = config.MIC_MAX_UTTERANCE_MS,
                 aggressiveness: float = config.VAD_AGGRESSIVENESS,
                 vad_fn: Callable[[np.ndarray, int], float] | None = None):

        ### VAD settings
        self.on_speech_start = on_speech_start
        self.on_utterance = on_utterance
        self.sample_rate = sample_rate
        self.onset_ms = onset_ms
        self.silence_ms = silence_ms
        self.preroll_ms = preroll_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.threshold = aggressiveness

        ## lazy loading
        self._vad_fn = vad_fn or _default_vad_fn

        ## min chunk size to consider it speaking
        self._vad_chunk_samples = _SILERO_CHUNK_SAMPLES.get(sample_rate, 512) if vad_fn is None else None
        self._vad_raw_buffer = np.array([], dtype=np.int16)

        self._state = MicState.DISARMED
        self._ring: deque[tuple[np.ndarray, float]] = deque()
        self._ring_ms = 0.0
        self._utterance: list[np.ndarray] = []
        self._utterance_ms = 0.0
        self._voiced_ms = 0.0
        self._voiced_total_ms = 0.0
        self._silence_ms_run = 0.0

        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def _reset_locked(self, state: str) -> None:
        self._state = state
        self._utterance = []
        self._utterance_ms = 0.0
        self._voiced_ms = 0.0
        self._voiced_total_ms = 0.0
        self._silence_ms_run = 0.0

    def arm(self) -> None:
        with self._lock:
            self._reset_locked(MicState.IDLE)

    def disarm(self) -> None:
        with self._lock:
            self._reset_locked(MicState.DISARMED)

    def _ring_push(self, frame: np.ndarray, frame_ms: float) -> None:
        self._ring.append((frame, frame_ms))
        self._ring_ms += frame_ms
        while self._ring_ms > self.preroll_ms and len(self._ring) > 1:
            _, popped_ms = self._ring.popleft()
            self._ring_ms -= popped_ms

    def push_frame(self, frame: np.ndarray) -> None:
        """
        feed frame from our audio thread 
        """
        if self._vad_chunk_samples is None:
            self._handle_chunk(frame)
            return

        self._vad_raw_buffer = np.concatenate([self._vad_raw_buffer, frame])
        while len(self._vad_raw_buffer) >= self._vad_chunk_samples:
            chunk, self._vad_raw_buffer = (
                self._vad_raw_buffer[: self._vad_chunk_samples],
                self._vad_raw_buffer[self._vad_chunk_samples :],
            )
            self._handle_chunk(chunk)

    def _handle_chunk(self, frame: np.ndarray) -> None:
        """
        advance state machine
        """
        events = self._score_and_handle(frame)
        for kind, payload in events:
            if kind == "speech_start":
                self.on_speech_start()
            elif kind == "utterance":
                self.on_utterance(payload)

    def _score_and_handle(self, frame: np.ndarray) -> list[tuple[str, bytes | None]]:
        frame_ms = len(frame) / float(self.sample_rate) * 1000.0

        with self._lock:
            if self._state == MicState.DISARMED:
                self._ring_push(frame, frame_ms)
                return []

        voiced = self._vad_fn(frame, self.sample_rate) >= self.threshold

        with self._lock:
            if self._state == MicState.DISARMED:
                self._ring_push(frame, frame_ms)
                return []

            if self._state == MicState.IDLE:
                self._ring_push(frame, frame_ms)
                if voiced:
                    self._voiced_ms += frame_ms
                    if self._voiced_ms >= self.onset_ms:
                        self._state = MicState.SPEAKING
                        self._utterance = [f for f, _ in self._ring]
                        self._utterance_ms = self._ring_ms
                        self._voiced_total_ms = self._voiced_ms
                        self._silence_ms_run = 0.0
                        return [("speech_start", None)]
                else:
                    self._voiced_ms = 0.0
                return []

            self._utterance.append(frame)
            self._utterance_ms += frame_ms
            if voiced:
                self._voiced_total_ms += frame_ms
                self._silence_ms_run = 0.0
            else:
                self._silence_ms_run += frame_ms

            if (self._silence_ms_run >= self.silence_ms
                    or self._utterance_ms >= self.max_utterance_ms > 0):
                return self._close_utterance_locked()
            return []

    def _close_utterance_locked(self) -> list[tuple[str, bytes | None]]:
        frames = self._utterance
        voiced_ms = self._voiced_total_ms

        self._reset_locked(MicState.IDLE)
        self._ring.clear()
        self._ring_ms = 0.0

        if voiced_ms < self.min_utterance_ms:
            return []

        return [("utterance", _pcm16_to_wav_bytes(frames, self.sample_rate))]
