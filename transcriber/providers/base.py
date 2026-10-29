"""The single interface every STT provider implements."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..config import Settings
from ..cost import CostTracker
from ..models import Segment, Transcript, Word


class STTProvider(ABC):
    name: str = ""

    def __init__(self, settings: Settings, cost: CostTracker):
        self.settings = settings
        self.cost = cost

    @abstractmethod
    async def transcribe(self, audio: Path, duration: float) -> Transcript:
        """Transcribe with diarization + word timestamps. Times in seconds."""


SENTENCE_END = (".", "?", "!", "।", "|")


def words_to_segments(words: list[Word], max_gap: float = 1.5, soft_max_len: float = 30.0) -> list[Segment]:
    """Group a flat word list into turns (for providers without utterances).

    A new segment starts on a speaker change, a pause > max_gap, or at a sentence
    end once the current segment is longer than soft_max_len seconds.
    """
    segs: list[Segment] = []
    for w in words:
        prev = segs[-1] if segs else None
        if (
            prev is None
            or prev.speaker != (w.speaker or "?")
            or w.start - prev.end > max_gap
            or (prev.end - prev.start > soft_max_len and prev.text.endswith(SENTENCE_END))
        ):
            segs.append(Segment(id=len(segs), speaker=w.speaker or "?", start=w.start, end=w.end, text=""))
        s = segs[-1]
        s.words.append(w)
        s.end = w.end
        s.text = f"{s.text} {w.text}".strip()
    for s in segs:
        confs = [w.confidence for w in s.words if w.confidence is not None]
        s.confidence = round(sum(confs) / len(confs), 3) if confs else None
    return segs
