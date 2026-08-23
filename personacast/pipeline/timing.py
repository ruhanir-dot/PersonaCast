
from __future__ import annotations

import time
from contextlib import contextmanager

STT = "stt"
OPENER_SELECT= "opener_select"
FIRST_AUDIO= "time_to_first_audio"
INTERPRET = "interpret"
WEB = "web"
WEB_ANSWER = "web_answer"
GENERATE= "generate"
TTS= "tts"
OPENERS= "openers"


class Timer:

    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.marks:dict[str, float] = {}
        self._t0 =time.perf_counter()

    @contextmanager 
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start)

    def mark(self, name: str) -> float:
        elapsed = time.perf_counter() - self._t0
        self.marks[name] = elapsed
        return elapsed

    @property
    def total(self) -> float:
        return time.perf_counter() - self._t0

    def note(self, **fields) -> None:
        self.marks.update(fields)

    def as_dict(self) -> dict:
        return {
            "stages": {k: round(v, 3) for k, v in self.stages.items()},
            "marks": {k: round(v, 3) for k, v in self.marks.items()},
            "total": round(self.total, 3),
        }

    def summary(self) -> str:
        first = self.marks.get(FIRST_AUDIO)
        parts = [f"{k}={v:.1f}s" for k, v in sorted(self.stages.items(), key=lambda kv: -kv[1])]
        head = f"first audio {first:.2f}s · " if first is not None else ""
        return head + " ".join(parts)
