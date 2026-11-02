"""Re-run the call outcome analysis on an existing transcript (no new transcription cost).

    python analyze.py out/call.transcript.json
    python analyze.py out/*.transcript.json --context "Projects: Green Valley (Sector 150, Noida)"
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import sys
from pathlib import Path

from transcribe import print_analysis
from transcriber.analysis import Analyzer, to_markdown
from transcriber.config import ConfigError, Settings
from transcriber.cost import CostTracker
from transcriber.models import Segment
from transcriber.retry import APIError


async def analyze_file(path: Path, settings: Settings) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = [Segment(id=s["id"], speaker=s["speaker"], start=s["start"], end=s["end"], text=s["text"])
            for s in data["segments"]]
    cost = CostTracker()
    result = await Analyzer(settings, cost).analyze(segs)
    stem = path.name.removesuffix(".transcript.json")
    data["analysis"] = result
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (path.parent / f"{stem}.analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (path.parent / f"{stem}.analysis.md").write_text(to_markdown(result, stem), encoding="utf-8")
    print(f"\n== {stem}  (cost ${cost.total_usd:.4f})")
    print_analysis(result)
    return result


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Decide the customer's outcome from existing transcript(s).")
    p.add_argument("files", nargs="+", help="*.transcript.json files (globs ok)")
    p.add_argument("--model", help="Gemini model (default: ANALYSIS_MODEL or CLEANUP_MODEL)")
    p.add_argument("--context", help="business context: project names, locations, agent names")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpcore", "google_genai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    s = Settings()
    if a.model:
        s.analysis_model = a.model
    if a.context:
        s.analysis_context = a.context
    files = [m for f in a.files for m in (sorted(map(Path, glob.glob(f))) if glob.has_magic(f) else [Path(f)])]
    if not files:
        logging.error("no files matched %s", a.files)
        return 1
    try:
        for f in files:
            asyncio.run(analyze_file(f, s))
    except (ConfigError, APIError, FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        logging.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
