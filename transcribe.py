"""High-accuracy Hinglish transcription.

    python transcribe.py path/to/file.mp3 --out out/
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from transcriber.config import ConfigError, Settings
from transcriber.pipeline import run
from transcriber.preprocess import AudioError
from transcriber.retry import APIError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe audio to a speaker-labelled, timestamped transcript.")
    p.add_argument("file", type=Path, help="audio file (mp3/wav/m4a/ogg/...)")
    p.add_argument("--out", type=Path, default=Path("out"), help="output directory (default: out/)")
    p.add_argument("--provider", choices=["auto", "assemblyai", "gemini"], help="STT provider (default: auto)")
    p.add_argument("--script", choices=["roman", "devanagari"], help="script for Hindi words (default: roman)")
    p.add_argument("--cleanup-model", help="Gemini model for the cleanup pass")
    p.add_argument("--no-cleanup", action="store_true", help="skip the LLM cleanup pass")
    p.add_argument("--no-denoise", action="store_true", help="skip noise reduction (still normalises volume)")
    p.add_argument("--speakers", type=int, help="expected number of speakers (AssemblyAI)")
    p.add_argument("--keyterms", help="comma-separated names/jargon to bias STT, e.g. 'Sharma Logistics,IFSC'")
    p.add_argument("--no-analysis", action="store_true", help="skip the call outcome analysis")
    p.add_argument("--analysis-model", help="Gemini model for call analysis (default: cleanup model)")
    p.add_argument("--context", help="business context for analysis, e.g. 'Projects: Green Valley, Sector 150 Noida'")
    p.add_argument("--reference", type=Path, help="reference transcript .txt to compute WER")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def build_settings(a: argparse.Namespace) -> Settings:
    s = Settings()
    if a.provider:
        s.stt_provider = a.provider
    if a.script:
        s.hindi_script = a.script
    if a.cleanup_model:
        s.cleanup_model = a.cleanup_model
    if a.no_cleanup:
        s.cleanup = False
    if a.no_denoise:
        s.denoise = False
    if a.speakers:
        s.speakers_expected = a.speakers
    if a.no_analysis:
        s.analysis = False
    if a.analysis_model:
        s.analysis_model = a.analysis_model
    if a.context:
        s.analysis_context = a.context
    if a.keyterms:
        s.keyterms = [t.strip() for t in a.keyterms.split(",") if t.strip()]
    return s


def print_analysis(a: dict) -> None:
    print(f"\nOUTCOME: {a['outcome_label'].upper()}  ({a['lead_temperature']} lead, {a['confidence']} confidence)"
          + ("  ** needs human review **" if a.get("needs_review") else ""))
    print(f"  Why:  {a.get('reason', '')}")
    print(f"  Next: {a.get('next_action', '')}" + (f"  [{a['follow_up_when']}]" if a.get("follow_up_when") else ""))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        stream.reconfigure(encoding="utf-8", errors="replace")
    a = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    try:
        res = asyncio.run(run(a.file, a.out, build_settings(a), a.reference))
    except (ConfigError, AudioError, APIError, FileNotFoundError) as exc:
        logging.error("%s", exc)
        return 1

    print("\nOutputs:")
    for kind, path in res.paths.items():
        print(f"  {kind:7s} {path}")
    print(f"\nCost: ${res.cost['total_usd']:.4f}  {res.cost['by_stage_usd']}")
    print(f"Time: {res.cost['seconds']}")
    if res.wer:
        print(f"WER:  raw STT {res.wer['raw_stt']:.1%}  ->  final {res.wer['final']:.1%}")
    print(f"Review: {len(res.review)} segment(s) flagged -> {res.paths['review']}")
    if res.analysis:
        print_analysis(res.analysis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
