"""Per-file cost and timing log.

Prices are USD list prices checked on 2026-10-29 (assemblyai.com/pricing,
ai.google.dev/gemini-api/docs/pricing). Several Gemini prices rise on
2026-12-01 / 2027-01-01, so update this table when they change. Free-tier keys
are not billed; the figures then show what the same run would cost on paid tier.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

# per hour of audio
ASSEMBLYAI_PER_HOUR = {"universal-3-5-pro": 0.21, "universal-2": 0.15}
ASSEMBLYAI_DIARIZATION_PER_HOUR = 0.02

# per 1M tokens: (input, output incl. thinking)
GEMINI_PER_MTOK = {
    "gemini-4-argon": (2.00, 10.00),
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.6-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.5-transcribe": (2.00, 12.00),
}


@dataclass
class CostTracker:
    items: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def add_assemblyai(self, model: str, audio_seconds: float, diarization: bool) -> None:
        rate = ASSEMBLYAI_PER_HOUR.get(model, max(ASSEMBLYAI_PER_HOUR.values()))
        if diarization:
            rate += ASSEMBLYAI_DIARIZATION_PER_HOUR
        self.items.append(
            {"stage": "stt", "provider": "assemblyai", "model": model,
             "audio_seconds": round(audio_seconds, 1), "usd": round(rate * audio_seconds / 3600, 5)}
        )

    def add_gemini(self, stage: str, model: str, usage) -> None:
        if usage is None:
            return
        tin = usage.prompt_token_count or 0
        tout = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        pin, pout = GEMINI_PER_MTOK.get(model, (None, None))
        usd = None if pin is None else round((tin * pin + tout * pout) / 1e6, 5)
        self.items.append(
            {"stage": stage, "provider": "gemini", "model": model,
             "input_tokens": tin, "output_tokens": tout, "usd": usd}
        )

    @contextmanager
    def timer(self, stage: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.timings[stage] = round(self.timings.get(stage, 0) + time.perf_counter() - t0, 2)

    @property
    def total_usd(self) -> float:
        return round(sum(i["usd"] or 0 for i in self.items), 4)

    def summary(self) -> dict:
        by_stage: dict[str, float] = {}
        for i in self.items:
            by_stage[i["stage"]] = round(by_stage.get(i["stage"], 0) + (i["usd"] or 0), 5)
        unknown = sorted({i["model"] for i in self.items if i["usd"] is None})
        return {"total_usd": self.total_usd, "by_stage_usd": by_stage, "unpriced_models": unknown,
                "seconds": self.timings, "calls": self.items}
