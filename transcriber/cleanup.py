"""LLM cleanup pass: Gemini listens to the audio and corrects the raw STT text.

Chunking: every segment has a stable id and is *owned* by exactly one chunk
(the one whose core window contains its midpoint). The overlap around a chunk
is sent as read-only context, so the model hears/sees the boundary but only
returns its owned ids. Merging is then a lookup by id: no duplicate lines, and
timestamps / speaker turns are never touched by the model.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from google import genai
from google.genai import types

from .config import ConfigError, Settings
from .cost import CostTracker
from .models import Segment
from .preprocess import cut_clip
from .retry import APIError, Transient, with_retry

log = logging.getLogger(__name__)

UNCLEAR = "[unclear]"
# Reject a cleaned segment whose word count drifts this far from the raw one:
# that means the model summarised, dropped or invented speech.
MIN_RATIO, MAX_RATIO, MIN_WORDS_FOR_RATIO = 0.6, 1.5, 4


@dataclass
class Chunk:
    index: int
    owned: list[Segment]
    context: list[Segment]
    audio_start: float
    audio_end: float


def plan_chunks(segments: list[Segment], duration: float, threshold: float, length: float,
                overlap: float) -> list[Chunk]:
    if not segments:
        return []
    if duration <= threshold:
        return [Chunk(0, list(segments), [], 0.0, duration)]
    n = math.ceil(duration / length)
    chunks: list[Chunk] = []
    for k in range(n):
        core_start, core_end = k * length, (k + 1) * length
        owned = [s for s in segments
                 if core_start <= (s.start + s.end) / 2 < core_end or (k == n - 1 and (s.start + s.end) / 2 >= core_end)]
        if not owned:
            continue
        a0 = max(0.0, min(core_start, owned[0].start) - overlap)
        a1 = min(duration, max(min(core_end, duration), owned[-1].end) + overlap)
        owned_ids = {s.id for s in owned}
        context = [s for s in segments if s.id not in owned_ids and s.end > a0 and s.start < a1]
        chunks.append(Chunk(len(chunks), owned, context, a0, a1))
    return chunks


SCRIPT_RULES = {
    "roman": (
        "Write Hindi/Urdu words in Roman script the way Hinglish is normally typed in India "
        "(e.g. 'kya', 'hai', 'nahi', 'theek hai', 'bol raha hoon'). Write English words in normal English spelling."
    ),
    "devanagari": (
        "Write Hindi/Urdu words in Devanagari. Write English words in Latin script with normal English spelling "
        "- never transliterate English words into Devanagari."
    ),
}

SYSTEM = """You are a meticulous verbatim transcription editor for Hindi-English (Hinglish) phone calls.
You receive an audio clip and a machine transcript of it, split into segments with ids, speakers and
times relative to the start of the clip. Listen to the audio and correct the transcript.

RULES - follow all of them:
1. VERBATIM. Never add, remove, reorder, summarise, paraphrase or translate what was said. Keep fillers,
   repetitions, false starts and grammatical mistakes exactly as spoken. Do not abbreviate, expand or
   "improve" words (keep "kilo" as "kilo", "mins" as "mins").
2. Only fix misheard words, names, numbers and spellings when the AUDIO clearly supports the fix.
   If the audio does not clearly support a change, keep the machine text.
3. If a word or phrase is genuinely unintelligible, replace just that part with [unclear]. Never guess.
4. Script: {script_rule}
5. Write numbers, amounts, dates and codes as digits/symbols when the speaker says them as such
   (e.g. "invoice 4572", "Rs 38,500", "22 October").
6. Keep the given segmentation: return exactly one entry per segment id listed under SEGMENTS TO EDIT,
   with the same id. Do not merge, split or move text between segments, and do not change speakers.
7. CONTEXT segments are only there for continuity at the clip edges - do not return them.
8. Add normal punctuation and capitalisation; nothing else.
Return JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
            },
        }
    },
    "required": ["segments"],
}


def _fmt(segs: list[Segment], offset: float) -> str:
    return "\n".join(
        json.dumps({"id": s.id, "speaker": s.speaker, "start": round(s.start - offset, 2),
                    "end": round(s.end - offset, 2), "text": s.text}, ensure_ascii=False)
        for s in segs
    )


def build_prompt(chunk: Chunk) -> str:
    parts = []
    if chunk.context:
        parts.append("CONTEXT (read-only, do not return):\n" + _fmt(chunk.context, chunk.audio_start))
    parts.append("SEGMENTS TO EDIT:\n" + _fmt(chunk.owned, chunk.audio_start))
    return "\n\n".join(parts)


def apply_edits(owned: list[Segment], edits: dict[int, str]) -> None:
    """Write cleaned text into segments, with guards against non-verbatim edits."""
    for s in owned:
        s.raw_text = s.text
        new = edits.get(s.id)
        if new is None or not new.strip():
            s.flags.append("cleanup_missing: kept raw STT text")
            continue
        new = new.strip()
        raw_n, new_n = len(s.text.split()), len(new.replace(UNCLEAR, "x").split())
        if raw_n >= MIN_WORDS_FOR_RATIO and not (MIN_RATIO <= new_n / raw_n <= MAX_RATIO):
            s.flags.append(f"cleanup_rejected: word count {raw_n}->{new_n}, kept raw STT text")
            continue
        s.text = new


class Cleaner:
    def __init__(self, settings: Settings, cost: CostTracker):
        if not settings.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set (.env) - needed for the cleanup pass (or use --no-cleanup)")
        if settings.hindi_script not in SCRIPT_RULES:
            raise ConfigError(f"HINDI_SCRIPT must be one of {sorted(SCRIPT_RULES)}")
        self.s = settings
        self.cost = cost
        self.client = genai.Client(api_key=settings.gemini_api_key,
                                   http_options=types.HttpOptions(timeout=900_000))
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM.format(script_rule=SCRIPT_RULES[settings.hindi_script]),
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            max_output_tokens=65536,
        )

    async def _clean_chunk(self, chunk: Chunk, audio: Path, work: Path, sem: asyncio.Semaphore) -> None:
        async with sem:
            clip = await cut_clip(audio, chunk.audio_start, chunk.audio_end, work / f"chunk{chunk.index:03d}.ogg")
            contents = [types.Part.from_bytes(data=clip.read_bytes(), mime_type="audio/ogg"), build_prompt(chunk)]

            async def call(model: str) -> dict[int, str]:
                resp = await self.client.aio.models.generate_content(
                    model=model, contents=contents, config=self.config)
                self.cost.add_gemini("cleanup", model, resp.usage_metadata)
                try:
                    data = json.loads(resp.text)
                    return {int(e["id"]): e["text"] for e in data["segments"]}
                except (TypeError, ValueError, KeyError) as exc:
                    reason = resp.candidates[0].finish_reason if resp.candidates else "no candidates"
                    raise Transient(f"unparseable cleanup output (finish_reason={reason})") from exc

            models = [m for m in (self.s.cleanup_model, self.s.cleanup_fallback_model) if m]
            edits = None
            for model in dict.fromkeys(models):
                try:
                    edits = await with_retry(f"Cleanup chunk {chunk.index} ({model})", call, model, attempts=6)
                    break
                except APIError as exc:
                    log.warning("%s", exc)
            if edits is None:
                log.error("cleanup chunk %d failed on %s - keeping raw STT text for %d segments",
                          chunk.index, models, len(chunk.owned))
                for seg in chunk.owned:
                    seg.raw_text = seg.text
                    seg.flags.append("cleanup_failed: kept raw STT text")
                return
            unknown = set(edits) - {s.id for s in chunk.owned}
            if unknown:
                log.debug("chunk %d: ignoring %d non-owned ids", chunk.index, len(unknown))
            apply_edits(chunk.owned, edits)
            log.info("cleanup chunk %d (%.0f-%.0fs, %d segs) done",
                     chunk.index, chunk.audio_start, chunk.audio_end, len(chunk.owned))

    async def clean(self, segments: list[Segment], duration: float, audio: Path, work: Path) -> int:
        s = self.s
        chunks = plan_chunks(segments, duration, s.chunk_threshold_s, s.chunk_len_s, s.chunk_overlap_s)
        sem = asyncio.Semaphore(s.max_concurrency)
        await asyncio.gather(*(self._clean_chunk(c, audio, work, sem) for c in chunks))
        return len(chunks)
