"""Write transcript.json / .md / .txt and the human-review list."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .cleanup import UNCLEAR
from .models import Transcript


def ts(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def relabel_speakers(t: Transcript) -> None:
    """Provider labels (A/B, spk:0...) -> 'Speaker 1', 'Speaker 2' in order of first appearance."""
    mapping: dict[str, str] = {}
    for s in t.segments:
        s.speaker = mapping.setdefault(s.speaker, f"Speaker {len(mapping) + 1}")
        for w in s.words:
            if w.speaker is not None:
                w.speaker = mapping.setdefault(w.speaker, f"Speaker {len(mapping) + 1}")


def review_items(t: Transcript, low_conf: float, low_word_conf: float = 0.4) -> list[dict]:
    items = []
    for s in t.segments:
        reasons = list(s.flags)
        n_unclear = s.text.count(UNCLEAR)
        if n_unclear:
            reasons.append(f"{n_unclear} x {UNCLEAR}")
        if s.confidence is not None and s.confidence < low_conf:
            reasons.append(f"low STT confidence {s.confidence:.2f}")
        weak = [w for w in s.words if w.confidence is not None and w.confidence < low_word_conf]
        if weak:
            reasons.append("low-confidence words: " + ", ".join(f"'{w.text}' ({w.confidence:.2f})" for w in weak[:8]))
        if reasons:
            items.append({"id": s.id, "start": s.start, "end": s.end, "time": ts(s.start), "speaker": s.speaker,
                          "text": s.text, "raw_text": s.raw_text, "reasons": reasons})
    return items


def write_outputs(t: Transcript, out_dir: Path, stem: str, review: list[dict], cost: dict,
                  analysis: dict | None = None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {k: out_dir / f"{stem}.{ext}" for k, ext in
             (("json", "transcript.json"), ("md", "transcript.md"), ("txt", "transcript.txt"), ("review", "review.md"))}

    data = t.to_dict()
    data["review"] = review
    data["cost"] = cost
    if analysis is not None:
        data["analysis"] = analysis
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    header = [f"# Transcript: {stem}", "",
              f"- Duration: {ts(t.duration)}  ",
              f"- STT: {t.provider} / {t.model}" + (f"  \n- Cleanup: {t.meta['cleanup_model']}" if t.meta.get("cleanup_model") else ""),
              f"- Segments: {len(t.segments)}, flagged for review: {len(review)}", ""]
    lines = [f"**[{ts(s.start)}] {s.speaker}:** {s.text}" for s in t.segments]
    paths["md"].write_text("\n".join(header) + "\n" + "\n\n".join(lines) + "\n", encoding="utf-8")
    paths["txt"].write_text("\n".join(f"[{ts(s.start)}] {s.speaker}: {s.text}" for s in t.segments) + "\n",
                            encoding="utf-8")

    rv = [f"# Review list: {stem}", "", f"{len(review)} segment(s) need a human check.", ""]
    for r in review:
        rv.append(f"- **[{r['time']}] {r['speaker']}** (segment {r['id']}): {r['text']}")
        for reason in r["reasons"]:
            rv.append(f"  - {reason}")
        if r["raw_text"] and r["raw_text"] != r["text"]:
            rv.append(f"  - raw STT: _{r['raw_text']}_")
    paths["review"].write_text("\n".join(rv) + "\n", encoding="utf-8")

    if analysis is not None:
        from .analysis import to_markdown

        paths["analysis"] = out_dir / f"{stem}.analysis.json"
        paths["analysis"].write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        paths["analysis_md"] = out_dir / f"{stem}.analysis.md"
        paths["analysis_md"].write_text(to_markdown(analysis, stem), encoding="utf-8")
    return paths


_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def normalize_for_wer(text: str) -> str:
    """Lowercase, drop 'Name:' speaker prefixes, [unclear], timestamps and punctuation."""
    text = re.sub(r"^\s*(\[[\d:]+\]\s*)?[^:\n]{1,40}:\s*", "", text, flags=re.MULTILINE)
    text = text.replace(UNCLEAR, " ")
    text = _PUNCT.sub(" ", text.lower())
    return " ".join(text.split())


def wer(reference: str, hypothesis: str) -> float:
    import jiwer

    return jiwer.wer(normalize_for_wer(reference), normalize_for_wer(hypothesis))
