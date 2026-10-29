"""Orchestrates: preprocess -> STT -> LLM cleanup -> outputs."""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .cleanup import Cleaner
from .config import Settings
from .cost import CostTracker
from .output import relabel_speakers, review_items, wer, write_outputs
from .preprocess import check_ffmpeg, preprocess, probe_duration
from .providers import get_provider

log = logging.getLogger(__name__)


@dataclass
class Result:
    paths: dict[str, Path]
    review: list[dict]
    cost: dict
    wer: dict | None
    seconds: float


async def run(src: Path, out_dir: Path, settings: Settings, reference: Path | None = None) -> Result:
    t0 = time.perf_counter()
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")
    check_ffmpeg()
    cost = CostTracker()
    stem = src.stem
    work = out_dir / f"{stem}.work"
    work.mkdir(parents=True, exist_ok=True)

    with cost.timer("preprocess"):
        duration = await probe_duration(src)
        wav, flac = await preprocess(src, work, denoise=settings.denoise)
        shutil.copy2(src, work / f"original{src.suffix.lower()}")
    log.info("preprocessed %s (%.1fs audio)", src.name, duration)

    provider = get_provider(settings, cost)
    with cost.timer("stt"):
        transcript = await provider.transcribe(flac, duration)
    relabel_speakers(transcript)
    log.info("stt done: %s/%s, %d segments", transcript.provider, transcript.model, len(transcript.segments))
    raw_text = "\n".join(s.text for s in transcript.segments)

    if settings.cleanup and transcript.segments:
        with cost.timer("cleanup"):
            n = await Cleaner(settings, cost).clean(transcript.segments, duration, flac, work)
        transcript.meta.update(cleanup_model=settings.cleanup_model, cleanup_chunks=n,
                               hindi_script=settings.hindi_script)

    review = review_items(transcript, settings.low_confidence)
    wer_scores = None
    if reference:
        ref = reference.read_text(encoding="utf-8")
        final = "\n".join(s.text for s in transcript.segments)
        wer_scores = {"raw_stt": round(wer(ref, raw_text), 4), "final": round(wer(ref, final), 4)}

    cost.timings["total"] = round(time.perf_counter() - t0, 2)
    summary = cost.summary()
    paths = write_outputs(transcript, out_dir, stem, review, summary | ({"wer": wer_scores} if wer_scores else {}))
    log.info("cost $%.4f, time %.1fs", summary["total_usd"], cost.timings["total"])
    return Result(paths, review, summary, wer_scores, cost.timings["total"])
