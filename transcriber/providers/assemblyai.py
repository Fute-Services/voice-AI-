"""AssemblyAI pre-recorded transcription (REST, async httpx).

Docs: https://www.assemblyai.com/docs/pre-recorded-audio/code-switching
Universal-3.5 Pro follows English<->Hindi switches mid-sentence when given
language_codes=["en", "hi"]; Universal-2 is listed as fallback.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx

from ..config import ConfigError
from ..models import Segment, Transcript, Word
from ..retry import APIError, with_retry
from .base import STTProvider

log = logging.getLogger(__name__)
BASE = "https://api.assemblyai.com"
POLL_S = 5
MAX_WAIT_S = 3600


class AssemblyAIProvider(STTProvider):
    name = "assemblyai"

    def _client(self) -> httpx.AsyncClient:
        if not self.settings.assemblyai_api_key:
            raise ConfigError("ASSEMBLYAI_API_KEY is not set (.env)")
        return httpx.AsyncClient(
            base_url=BASE,
            headers={"authorization": self.settings.assemblyai_api_key},
            timeout=httpx.Timeout(60, write=600),
        )

    async def transcribe(self, audio: Path, duration: float) -> Transcript:
        s = self.settings
        async with self._client() as client:
            async def upload() -> str:
                r = await client.post("/v2/upload", content=audio.read_bytes())
                r.raise_for_status()
                return r.json()["upload_url"]

            upload_url = await with_retry("AssemblyAI upload", upload)
            body: dict = {
                "audio_url": upload_url,
                "speech_models": s.assemblyai_models,
                "language_codes": s.language_codes,
                "speaker_labels": True,
                "punctuate": True,
                "format_text": True,
            }
            if s.speakers_expected:
                body["speakers_expected"] = s.speakers_expected
            if s.keyterms:
                body["keyterms_prompt"] = s.keyterms

            async def submit() -> str:
                r = await client.post("/v2/transcript", json=body)
                r.raise_for_status()
                return r.json()["id"]

            tid = await with_retry("AssemblyAI submit", submit)
            log.info("AssemblyAI job %s submitted", tid)

            async def poll() -> dict:
                r = await client.get(f"/v2/transcript/{tid}")
                r.raise_for_status()
                return r.json()

            waited = 0
            while True:
                data = await with_retry("AssemblyAI poll", poll)
                if data["status"] == "completed":
                    break
                if data["status"] == "error":
                    raise APIError(f"AssemblyAI transcription failed: {data.get('error')}")
                if waited > MAX_WAIT_S:
                    raise APIError(f"AssemblyAI job {tid} still {data['status']} after {MAX_WAIT_S}s")
                await asyncio.sleep(POLL_S)
                waited += POLL_S

        model = data.get("speech_model_used") or data.get("speech_model") or s.assemblyai_models[0]
        audio_s = float(data.get("audio_duration") or duration)
        self.cost.add_assemblyai(model, audio_s, diarization=True)
        return Transcript(
            segments=parse_utterances(data),
            duration=audio_s,
            language=data.get("language_code"),
            provider=self.name,
            model=model,
            meta={"job_id": tid, "language_confidence": data.get("language_confidence")},
        )


def _word(w: dict) -> Word:
    return Word(text=w["text"], start=w["start"] / 1000, end=w["end"] / 1000,
                confidence=w.get("confidence"), speaker=w.get("speaker"))


def parse_utterances(data: dict) -> list[Segment]:
    utts = data.get("utterances") or []
    if not utts:  # no diarization result: single pseudo-speaker
        words = [_word(w) for w in data.get("words") or []]
        if not words:
            return []
        utts = [{"speaker": "A", "start": data["words"][0]["start"], "end": data["words"][-1]["end"],
                 "text": data.get("text", ""), "confidence": data.get("confidence"), "words": data["words"]}]
    return [
        Segment(id=i, speaker=u.get("speaker") or "?", start=u["start"] / 1000, end=u["end"] / 1000,
                text=u["text"], confidence=u.get("confidence"), words=[_word(w) for w in u.get("words", [])])
        for i, u in enumerate(utts)
    ]
