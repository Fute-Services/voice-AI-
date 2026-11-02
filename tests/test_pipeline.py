"""Unit tests run offline. The integration test needs GEMINI_API_KEY and ffmpeg:

    pytest                 # everything (integration skipped without a key)
    pytest -m "not integration"
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from transcriber.cleanup import apply_edits, build_prompt, plan_chunks
from transcriber.config import Settings
from transcriber.cost import CostTracker
from transcriber.models import Segment, Transcript, Word
from transcriber.output import normalize_for_wer, relabel_speakers, review_items, wer, write_outputs
from transcriber.providers.assemblyai import parse_utterances
from transcriber.providers.base import words_to_segments

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = FIXTURES / "sample_call.mp3"
REFERENCE = FIXTURES / "sample_call.reference.txt"


def fake_segments(duration: float, every: float = 7.0) -> list[Segment]:
    segs, t, i = [], 0.0, 0
    while t + 5 <= duration:
        segs.append(Segment(id=i, speaker="AB"[i % 2], start=t, end=t + 5, text=f"line {i} haan theek hai"))
        t += every
        i += 1
    return segs


# --- chunking / merge -------------------------------------------------------

def test_short_file_is_one_chunk():
    segs = fake_segments(600)
    chunks = plan_chunks(segs, 600, threshold=1200, length=600, overlap=15)
    assert len(chunks) == 1 and chunks[0].owned == segs and not chunks[0].context


def test_long_file_every_segment_owned_exactly_once():
    duration = 60 * 60
    segs = fake_segments(duration)
    chunks = plan_chunks(segs, duration, threshold=1200, length=600, overlap=15)
    assert len(chunks) == 6
    owned_ids = [s.id for c in chunks for s in c.owned]
    assert sorted(owned_ids) == [s.id for s in segs]  # no duplicates, nothing lost
    for c in chunks:
        assert c.audio_start <= c.owned[0].start and c.audio_end >= c.owned[-1].end
        assert not {s.id for s in c.context} & {s.id for s in c.owned}
    assert all(c.context for c in chunks)  # overlap gives neighbours as context
    assert chunks[1].audio_start == pytest.approx(600 - 15, abs=1)


def test_segment_straddling_boundary_goes_to_one_chunk():
    segs = [Segment(0, "A", 590, 598, "a"), Segment(1, "B", 597, 607, "b"), Segment(2, "A", 608, 615, "c"),
            Segment(3, "B", 1500, 1505, "d")]
    chunks = plan_chunks(segs, 1510, threshold=1200, length=600, overlap=15)
    owners = {s.id: c.index for c in chunks for s in c.owned}
    assert owners[1] == owners[2] and owners[0] != owners[1]  # midpoint 602 -> second chunk
    assert len(owners) == 4


def test_prompt_uses_clip_relative_times():
    segs = fake_segments(2400)
    c = plan_chunks(segs, 2400, 1200, 600, 15)[1]
    p = build_prompt(c)
    assert "CONTEXT" in p and "SEGMENTS TO EDIT" in p
    first = json.loads(p.split("SEGMENTS TO EDIT:\n")[1].splitlines()[0])
    assert 0 <= first["start"] <= 30


# --- verbatim guards --------------------------------------------------------

def test_apply_edits_guards():
    segs = [Segment(0, "A", 0, 5, "मैं राहुल बोल रहा हूं"),
            Segment(1, "B", 5, 9, "haan theek hai bilkul main kal tak bhej dunga"),
            Segment(2, "A", 9, 12, "okay bye")]
    apply_edits(segs, {0: "Main Rahul bol raha hoon.", 1: "Haan, theek hai."})  # 1 = summarised, 2 = missing
    assert segs[0].text == "Main Rahul bol raha hoon." and not segs[0].flags
    assert segs[1].text == segs[1].raw_text and "cleanup_rejected" in segs[1].flags[0]
    assert segs[2].text == "okay bye" and "cleanup_missing" in segs[2].flags[0]


# --- providers --------------------------------------------------------------

def test_assemblyai_parse():
    data = {"utterances": [
        {"speaker": "A", "start": 1000, "end": 2500, "text": "Hello Priya", "confidence": 0.93,
         "words": [{"text": "Hello", "start": 1000, "end": 1400, "confidence": 0.99, "speaker": "A"},
                   {"text": "Priya", "start": 1500, "end": 2500, "confidence": 0.31, "speaker": "A"}]},
        {"speaker": "B", "start": 3000, "end": 4000, "text": "Haan boliye", "confidence": 0.5, "words": []}]}
    segs = parse_utterances(data)
    assert [s.speaker for s in segs] == ["A", "B"]
    assert segs[0].start == 1.0 and segs[0].words[1].end == 2.5


def test_words_to_segments_splits_on_speaker_and_pause():
    w = [Word("haan", 0, 0.3, speaker="S0"), Word("boliye.", 0.3, 0.8, speaker="S0"),
         Word("main", 1.0, 1.2, speaker="S1"), Word("abhi", 4.0, 4.3, speaker="S1")]
    segs = words_to_segments(w)
    assert [(s.speaker, s.text) for s in segs] == [("S0", "haan boliye."), ("S1", "main"), ("S1", "abhi")]


# --- outputs ----------------------------------------------------------------

def test_outputs_and_review(tmp_path):
    segs = [Segment(0, "A", 0, 3, "Hello [unclear] ji", confidence=0.9),
            Segment(1, "B", 3, 6, "Haan", confidence=0.4,
                    words=[Word("Haan", 3, 4, confidence=0.2, speaker="B")]),
            Segment(2, "A", 6, 8, "Okay", confidence=0.95)]
    t = Transcript(segs, duration=8, provider="test", model="m")
    relabel_speakers(t)
    assert [s.speaker for s in segs] == ["Speaker 1", "Speaker 2", "Speaker 1"]
    review = review_items(t, low_conf=0.6)
    assert [r["id"] for r in review] == [0, 1]
    paths = write_outputs(t, tmp_path, "x", review, {"total_usd": 0})
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert {"speaker", "start", "end", "text", "confidence"} <= set(data["segments"][0])
    assert "[00:00:03] Speaker 2" in paths["md"].read_text(encoding="utf-8")


def test_wer_normalisation():
    assert normalize_for_wer("Rahul: Haan, boliye!\n[00:00:05] Speaker 2: OK [unclear]") == "haan boliye ok"
    assert wer("Rahul: haan boliye", "Speaker 1: Haan, boliye.") == 0


# --- call analysis ----------------------------------------------------------

def call_segments() -> list[Segment]:
    return [Segment(0, "Speaker 1", 0, 6, "Namaste sir, main Neha bol rahi hoon Green Valley se, aapne 2 BHK ke liye enquiry ki thi."),
            Segment(1, "Speaker 2", 6, 12, "Haan haan, budget mera 80 lakh tak hai, Noida mein chahiye."),
            Segment(2, "Speaker 1", 12, 16, "Sir site visit ke liye aa sakte hain?"),
            Segment(3, "Speaker 2", 16, 21, "Theek hai, Sunday ko 11 baje aa jaunga, family ke saath.")]


def fake_analysis(**over) -> dict:
    base = {"customer_speaker": "Speaker 2", "agent_speaker": "Speaker 1", "outcome": "site_visit_scheduled",
            "lead_temperature": "hot", "confidence": "high", "reason": "Fixed a visit on Sunday.",
            "summary": "Lead wants a 2 BHK in Noida under 80 lakh and booked a visit.",
            "next_action": "Confirm Sunday 11 am visit on WhatsApp.", "follow_up_when": "Sunday 11 baje",
            "requirements": {"budget": "80 lakh", "location": "Noida", "property_type": "", "bhk": "2 BHK",
                             "purpose": "", "timeline": "", "other": ""},
            "objections": [],
            "evidence": [{"segment_id": 3, "quote": "Sunday ko 11 baje aa jaunga"}]}
    return base | over


def test_analysis_verify_keeps_real_quotes():
    from transcriber.analysis import verify

    a = verify(fake_analysis(evidence=[{"segment_id": 3, "quote": "sunday ko 11 baje aa jaunga,"},   # case/punct ok
                                       {"segment_id": 0, "quote": "budget mera 80 lakh"},           # wrong id -> fixed
                                       {"segment_id": 1, "quote": "main zaroor kharidunga"}]),      # invented
               call_segments())
    assert [e["segment_id"] for e in a["evidence"]] == [3, 1]
    assert a["evidence"][0]["start"] == 16 and a["evidence"][0]["speaker"] == "Speaker 2"
    assert a["confidence"] == "high" and not a["needs_review"]
    assert a["outcome_label"] == "Site visit scheduled"
    assert "1 evidence quote" in a["flags"][0]


def test_analysis_verify_downgrades_unsupported_decision():
    from transcriber.analysis import verify

    a = verify(fake_analysis(evidence=[{"segment_id": 3, "quote": "haan main flat le lunga"}]), call_segments())
    assert a["evidence"] == [] and a["confidence"] == "low" and a["needs_review"]


def test_analyzer_with_fake_client_and_outputs(tmp_path):
    from transcriber.analysis import Analyzer, build_prompt

    class Resp:
        text = json.dumps(fake_analysis())
        usage_metadata = None

    class Models:
        async def generate_content(self, model, contents, config):
            assert '"id": 3' in contents and "Sunday" in contents
            assert "REAL ESTATE" in config.system_instruction and "Green Valley" in config.system_instruction
            return Resp()

    s = Settings()
    s.gemini_api_key, s.analysis_context = "test", "Project: Green Valley"
    an = Analyzer(s, CostTracker())
    an.client = type("C", (), {"aio": type("A", (), {"models": Models()})()})()
    a = asyncio.run(an.analyze(call_segments()))
    assert a["outcome"] == "site_visit_scheduled" and a["evidence"] and not a["needs_review"]
    assert build_prompt(call_segments()).startswith("TRANSCRIPT:")

    t = Transcript(call_segments(), duration=21, provider="test", model="m")
    paths = write_outputs(t, tmp_path, "x", [], {"total_usd": 0}, a)
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["analysis"]["outcome"] == "site_visit_scheduled"
    md = paths["analysis_md"].read_text(encoding="utf-8")
    assert "Site visit scheduled" in md and "Budget: 80 lakh" in md and "[00:00:16] Speaker 2" in md


def test_analyzer_empty_transcript():
    from transcriber.analysis import Analyzer

    s = Settings()
    s.gemini_api_key = "test"
    a = asyncio.run(Analyzer(s, CostTracker()).analyze([]))
    assert a["outcome"] == "unclear" and a["needs_review"]


# --- integration ------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("GEMINI_API_KEY") and not Path(".env").exists(), reason="needs GEMINI_API_KEY")
@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_end_to_end_sample(tmp_path):
    from transcriber.pipeline import run

    s = Settings()
    if not s.gemini_api_key:
        pytest.skip("needs GEMINI_API_KEY")
    res = asyncio.run(run(SAMPLE, tmp_path, s, reference=REFERENCE))
    data = json.loads(res.paths["json"].read_text(encoding="utf-8"))
    segs = data["segments"]
    assert len(segs) >= 6
    assert len({x["speaker"] for x in segs}) >= 2
    assert all(a["start"] <= b["start"] for a, b in zip(segs, segs[1:]))
    assert len({x["id"] for x in segs}) == len(segs)
    # Most residual "errors" are digits vs spelled-out numbers and Hinglish spelling variants.
    assert res.wer["final"] < 0.30, res.wer
