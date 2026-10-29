"""Gemini dedicated transcription model (gemini-3.5-transcribe).

Docs: https://ai.google.dev/gemini-api/docs/generate-content/transcribe
With diarization or word timestamps enabled the model accepts at most 30 min,
so longer audio is split into ~25 min pieces. Speaker labels are per request,
so across pieces they are reconciled only by the cleanup LLM (best effort).
No per-word confidence is returned.
"""
from __future__ import annotations

import logging
from pathlib import Path

from google import genai
from google.genai import types

from ..config import ConfigError
from ..models import Transcript, Word
from ..preprocess import cut_clip
from ..retry import with_retry
from .base import STTProvider, words_to_segments

log = logging.getLogger(__name__)
PIECE_S = 25 * 60


def _secs(v: str | None) -> float:
    return float(v.rstrip("s")) if v else 0.0


class GeminiProvider(STTProvider):
    name = "gemini"

    async def transcribe(self, audio: Path, duration: float) -> Transcript:
        s = self.settings
        if not s.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set (.env)")
        client = genai.Client(api_key=s.gemini_api_key, http_options=types.HttpOptions(timeout=900_000))
        cfg = types.GenerateContentConfig(
            audio_transcription_config=types.AudioTranscriptionConfig(
                diarization=True,
                word_timestamp=True,
                language_codes=[],  # empty = auto-detect, handles code-switching
            )
        )
        words: list[Word] = []
        starts = [i * PIECE_S for i in range(int(duration // PIECE_S) + 1) if i * PIECE_S < duration]
        for n, start in enumerate(starts):
            end = min(start + PIECE_S, duration)
            if len(starts) == 1:
                path, mime = audio, "audio/flac"
            else:
                path = await cut_clip(audio, start, end, audio.with_suffix(f".stt{n}.ogg"))
                mime = "audio/ogg"
            part = types.Part.from_bytes(data=path.read_bytes(), mime_type=mime)
            resp = await with_retry(
                "Gemini transcription", client.aio.models.generate_content,
                model=s.gemini_stt_model, contents=[part], config=cfg,
            )
            self.cost.add_gemini("stt", s.gemini_stt_model, resp.usage_metadata)
            for p in resp.candidates[0].content.parts:
                at = p.audio_transcription
                if not at:
                    continue
                spk = (at.speaker_label or "?").replace("spk:", "S")
                for w in at.words or []:
                    words.append(Word(text=w.word, start=start + _secs(w.start_offset),
                                      end=start + _secs(w.end_offset), speaker=spk))
            log.info("gemini stt piece %d/%d done", n + 1, len(starts))

        segs = words_to_segments(words)
        return Transcript(segments=segs, duration=duration, provider=self.name, model=s.gemini_stt_model,
                          meta={"pieces": len(starts)})
