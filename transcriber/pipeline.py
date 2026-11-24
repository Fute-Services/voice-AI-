"""Orchestrates: preprocess -> STT -> LLM cleanup -> call analysis -> outputs."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .analysis import Analyzer
from .cleanup import Cleaner
from .config import Settings
from .cost import CostTracker
from .metrics import call_metrics
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
    analysis: dict | None = None


async def run(src: Path, out_dir: Path, settings: Settings, reference: Path | None = None,
              on_event: Callable[[str, dict], None] | None = None) -> Result:
    """Run the whole pipeline. `on_event(name, data)` is told about each stage as it starts ("stage") and gets
    the finished transcript ("transcript") before the analysis runs, so a UI can show it without waiting."""
    t0 = time.perf_counter()
    emit = on_event or (lambda name, data: None)
    if not src.is_file():
        raise FileNotFoundError(f"Input file not found: {src}")
    check_ffmpeg()
    cost = CostTracker()
    stem = src.stem
    work = out_dir / f"{stem}.work"
    work.mkdir(parents=True, exist_ok=True)

    emit("stage", {"stage": "preprocess"})
    with cost.timer("preprocess"):
        duration = await probe_duration(src)
        flac = await preprocess(src, work, denoise=settings.denoise)
    log.info("preprocessed %s (%.1fs audio)", src.name, duration)

    provider = get_provider(settings, cost)
    emit("stage", {"stage": "stt", "duration": duration})
    with cost.timer("stt"):
        transcript = await provider.transcribe(flac, duration)
    relabel_speakers(transcript)
    log.info("stt done: %s/%s, %d segments", transcript.provider, transcript.model, len(transcript.segments))
    raw_text = "\n".join(s.text for s in transcript.segments)

    if settings.cleanup and transcript.segments:
        emit("stage", {"stage": "cleanup"})
        with cost.timer("cleanup"):
            n = await Cleaner(settings, cost).clean(transcript.segments, duration, flac, work)
        transcript.meta.update(cleanup_model=settings.cleanup_model, cleanup_chunks=n,
                               hindi_script=settings.hindi_script)

    review = review_items(transcript, settings.low_confidence)
    analysis = None
    if settings.analysis:
        emit("transcript", {**transcript.to_dict(), "review": review})
        emit("stage", {"stage": "analysis"})
        with cost.timer("analysis"):
            analysis = await Analyzer(settings, cost).analyze(transcript.segments)
        log.info("analysis: %s (%s lead, %s confidence)", analysis["outcome"], analysis["lead_temperature"],
                 analysis["confidence"])

    # Measured talk-time analytics; per role (agent / customer) once the analysis has told them apart.
    roles = analysis or {}
    transcript.meta["metrics"] = call_metrics(transcript.segments, duration, roles.get("agent_speaker") or "",
                                              roles.get("customer_speaker") or "")

    wer_scores = None
    if reference:
        ref = reference.read_text(encoding="utf-8")
        final = "\n".join(s.text for s in transcript.segments)
        wer_scores = {"raw_stt": round(wer(ref, raw_text), 4), "final": round(wer(ref, final), 4)}

    cost.timings["total"] = round(time.perf_counter() - t0, 2)
    summary = cost.summary()
    paths = write_outputs(transcript, out_dir, stem, review, summary | ({"wer": wer_scores} if wer_scores else {}),
                          analysis)
    log.info("cost $%.4f, time %.1fs", summary["total_usd"], cost.timings["total"])
    return Result(paths, review, summary, wer_scores, cost.timings["total"], analysis)
