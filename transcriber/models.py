"""Core data types shared by every stage of the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Word:
    text: str
    start: float  # seconds
    end: float
    confidence: float | None = None
    speaker: str | None = None


@dataclass
class Segment:
    """One speaker turn. `id` is stable across the pipeline (used to merge chunks)."""

    id: int
    speaker: str
    start: float
    end: float
    text: str
    confidence: float | None = None
    words: list[Word] = field(default_factory=list)
    raw_text: str | None = None  # STT text before LLM cleanup
    flags: list[str] = field(default_factory=list)  # reasons this segment needs human review


@dataclass
class Transcript:
    segments: list[Segment]
    duration: float
    language: str | None = None
    provider: str = ""
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
