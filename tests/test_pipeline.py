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
            "mindset": {"motive": "self_use", "urgency": "immediate", "decision_maker": "joint_with_family",
                        "financing": "unknown", "primary_concern": "", "competitors": [],
                        "pitch_angle": "Make the Sunday visit family-friendly."},
            "behaviour": {"buyer_type": "serious_buyer",
                          "traits": {"seriousness": "high", "engagement": "high", "openness": "high",
                                     "decisiveness": "high", "trust": "unknown", "price_sensitivity": "medium"},
                          "tone": "positive", "communication_style": "Direct and cooperative",
                          "verdict": "Genuine buyer who shared budget unprompted and fixed a day himself.",
                          "how_to_handle": ["Confirm the visit in writing."],
                          "observations": [{"observation": "Shared budget without being asked", "segment_id": 1,
                                            "quote": "budget mera 80 lakh tak hai"},
                                           {"observation": "Asked for a discount", "segment_id": 1,
                                            "quote": "discount kitna milega"}]},
            "buying_signals": [{"signal": "Bringing family to the visit", "segment_id": 3, "quote": "family ke saath"}],
            "risk_signals": [],
            "discovery": [{"topic": "budget", "status": "answered", "answer": "80 lakh"},
                          {"topic": "next_step", "status": "answered", "answer": "Sunday 11 baje"},
                          {"topic": "financing", "status": "asked_not_answered", "answer": ""}],
            "next_questions": [
                {"topic": "next_step", "hinglish": "Sunday ko family mein kaun kaun aa raha hai?",
                 "english": "Who from the family is coming on Sunday?", "why": "Plan the visit around them."},
                {"topic": "financing", "hinglish": "Loan ke liye hum aapki koi madad kar sakte hain?",
                 "english": "Can we help you with the loan?", "why": "He avoided the loan question."},
                {"topic": "financing", "hinglish": "Loan ke liye hum aapki koi madad kar sakte hain?",   # repeat
                 "english": "", "why": ""},
                {"topic": "made_up", "hinglish": "", "english": "", "why": "empty"}],
            "evidence": [{"segment_id": 3, "quote": "Sunday ko 11 baje aa jaunga"}]}
    return base | over


def test_analysis_mindset_discovery_and_score():
    from transcriber.analysis import DISCOVERY, lead_score, verify

    a = verify(fake_analysis(risk_signals=[{"signal": "Only wants the rate", "segment_id": 1, "quote": "bas rate bata do"}]),
               call_segments())
    # Real quote kept with its timestamp; invented one dropped and flagged.
    assert a["buying_signals"] == [{"signal": "Bringing family to the visit", "segment_id": 3, "speaker": "Speaker 2",
                                    "start": 16, "quote": "family ke saath"}]
    assert a["risk_signals"] == [] and any("2 signal(s)" in f for f in a["flags"])
    # Behaviour: invented observation dropped, label added, talk stats measured from the transcript.
    b = a["behaviour"]
    assert [o["observation"] for o in b["observations"]] == ["Shared budget without being asked"]
    assert b["observations"][0]["start"] == 6 and b["buyer_type_label"] == "Serious buyer"
    assert b["stats"] == {"talk_share_pct": 46, "customer_turns": 2, "avg_words_per_turn": 11, "questions_asked": 0}
    # Discovery: each topic is answered (with what was said), dodged, or not discussed yet.
    d = a["discovery"]
    assert [(c["key"], c["answer"]) for c in d["covered"]] == [("budget", "80 lakh"), ("next_step", "Sunday 11 baje")]
    assert [x["key"] for x in d["dodged"]] == ["financing"]
    assert len(d["missed"]) == len(DISCOVERY) - 3 and d["total"] == len(DISCOVERY)
    assert d["missed"][0]["key"] == "purpose" and "financing" not in [m["key"] for m in d["missed"]]
    # Questions: the model's own, in its order; repeats and empty ones dropped.
    q = a["next_questions"]
    assert [x["key"] for x in q] == ["next_step", "financing"] and all(x["tailored"] for x in q)
    assert q[0]["hinglish"].startswith("Sunday ko family") and q[1]["topic"] == "Loan or own funds"
    # visit 45 + immediate 20 + joint 6 + budget 8 + location 4 + motive 3
    assert a["lead_score"] == 86
    assert lead_score(fake_analysis(outcome="not_interested")) == 15
    assert lead_score(fake_analysis(outcome="wrong_number")) == 0
    wrong = verify(fake_analysis(outcome="wrong_number"), call_segments())
    assert wrong["discovery"]["missed"] == [] and wrong["next_questions"] == []
    # Analyses saved before these fields existed must still verify.
    old = {k: v for k, v in fake_analysis().items()
           if k not in ("mindset", "behaviour", "buying_signals", "risk_signals", "discovery", "next_questions")}
    assert verify(dict(old), call_segments())["lead_score"] == 45 + 8 + 4
    legacy = verify(old | {"discovery_covered": ["budget", "next_step"]}, call_segments())
    assert [c["key"] for c in legacy["discovery"]["covered"]] == ["budget", "next_step"]
    assert "discovery_covered" not in legacy


def test_next_questions_fallback_is_short_and_fits_the_outcome():
    from transcriber.analysis import DISCOVERY, MAX_QUESTIONS, verify

    # Model wrote no questions: a short starter list from the bank, the dodged topic first - never all 11.
    a = verify(fake_analysis(outcome="follow_up", next_questions=[]), call_segments())
    q = a["next_questions"]
    assert [x["key"] for x in q] == ["financing", "purpose", "timeline"] and not any(x["tailored"] for x in q)
    assert q[0]["hinglish"] == DISCOVERY["financing"][2] and q[0]["english"].endswith("?")
    # A thin list from the model is topped up, without repeating a topic it already asked about.
    one = [{"topic": "financing", "hinglish": "Loan mein madad chahiye?", "english": "", "why": ""}]
    q = verify(fake_analysis(outcome="interested", next_questions=one), call_segments())["next_questions"]
    assert [(x["key"], x["tailored"]) for x in q] == [("financing", True), ("purpose", False), ("timeline", False)]
    # ...but not for a visit that is already fixed.
    assert len(verify(fake_analysis(next_questions=one), call_segments())["next_questions"]) == 1
    # Someone who said no is never handed a generic checklist.
    assert verify(fake_analysis(outcome="not_interested", next_questions=[]), call_segments())["next_questions"] == []
    # A model that over-produces is capped.
    many = [{"topic": "other", "hinglish": f"Sawal {i}?", "english": "", "why": ""} for i in range(9)]
    capped = verify(fake_analysis(next_questions=many), call_segments())["next_questions"]
    assert len(capped) == MAX_QUESTIONS and capped[0]["topic"] == "Follow-up"


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
    assert "score: 86/100" in md and "Decision maker: Joint with family" in md and "- Financing: " not in md
    assert "## Buying signals" in md and "## Questions to ask next" in md
    assert "1. **Sunday ko family mein kaun kaun aa raha hai?**" in md and "_Can we help you with the loan?_" in md
    assert "## Discovery (2/11 topics known)" in md and "- Budget: 80 lakh" in md
    assert "- Asked but not answered: Loan or own funds" in md and "- Not discussed yet: Purpose" in md
    assert "**Type: Serious buyer** · tone: positive" in md and "- Seriousness: high" in md
    assert "- Trust in the agent" not in md and "Customer spoke 46% of the call" in md


def test_analyzer_empty_transcript():
    from transcriber.analysis import Analyzer

    s = Settings()
    s.gemini_api_key = "test"
    a = asyncio.run(Analyzer(s, CostTracker()).analyze([]))
    assert a["outcome"] == "unclear" and a["needs_review"]


# --- speed: thinking level, model cooldown, progress events ------------------

def test_generate_caps_thinking_and_falls_back_when_rejected():
    from google.genai import errors, types

    from transcriber import llm

    seen = []

    class Models:
        async def generate_content(self, model, contents, config):
            level = config.thinking_config.thinking_level.value if config.thinking_config else None
            seen.append((model, level))
            if model == "old-model" and level:
                raise errors.APIError(400, {"error": {"message": "thinking_level is not supported", "status": "INVALID_ARGUMENT"}})
            return "ok"

    client = type("C", (), {"aio": type("A", (), {"models": Models()})()})()
    cfg = types.GenerateContentConfig(system_instruction="x")
    llm._no_thinking.clear()
    assert asyncio.run(llm.generate(client, "new-model", "hi", cfg, "low")) == "ok"
    assert asyncio.run(llm.generate(client, "new-model", "hi", cfg, "default")) == "ok"   # model's own setting
    assert asyncio.run(llm.generate(client, "old-model", "hi", cfg, "low")) == "ok"       # retried without it
    assert asyncio.run(llm.generate(client, "old-model", "hi", cfg, "low")) == "ok"       # remembered
    assert seen == [("new-model", "LOW"), ("new-model", None), ("old-model", "LOW"), ("old-model", None),
                    ("old-model", None)]
    assert cfg.thinking_config is None  # the shared config is never mutated


def test_fallback_remembers_an_overloaded_model():
    from transcriber import retry

    calls = []

    async def fn(model):
        calls.append(model)
        if model == "busy":
            raise retry.APIError("503 high demand")
        return model.upper()

    retry._cooling.clear()
    assert asyncio.run(retry.with_fallback("x", fn, ["busy", "spare"])) == ("SPARE", "spare")
    calls.clear()
    # Next call (another chunk, or the analysis) goes straight to the healthy model.
    assert asyncio.run(retry.with_fallback("x", fn, ["busy", "spare"])) == ("SPARE", "spare")
    assert calls == ["spare"]
    retry._cooling.clear()

    # A model that timed out already cost the whole deadline: move on instead of waiting it out again.
    async def slow(model):
        calls.append(model)
        if model == "hung":
            raise TimeoutError("deadline")
        return "done"

    calls.clear()
    assert asyncio.run(retry.with_fallback("x", slow, ["hung", "spare"])) == ("done", "spare")
    assert calls == ["hung", "spare"]
    retry._cooling.clear()


def test_cleanup_deadline_scales_with_the_clip():
    from transcriber.cleanup import MAX_TIMEOUT_S, request_timeout

    assert request_timeout(72) < 60 < request_timeout(600) < request_timeout(3600) == MAX_TIMEOUT_S


def test_pipeline_reports_stages_and_transcript_before_analysis(tmp_path, monkeypatch):
    from transcriber import pipeline

    class Provider:
        async def transcribe(self, audio, duration):
            assert audio.suffix == ".flac"
            return Transcript(call_segments(), duration=duration, provider="test", model="m")

    class FakeAnalyzer:
        def __init__(self, settings, cost):
            pass

        async def analyze(self, segments):
            from transcriber.analysis import verify
            return verify(fake_analysis(), segments)

    async def fake_preprocess(src, work, denoise=True):
        return work / "x.16k.flac"

    async def fake_duration(src):
        return 21.0

    monkeypatch.setattr(pipeline, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "probe_duration", fake_duration)
    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    monkeypatch.setattr(pipeline, "get_provider", lambda s, c: Provider())
    monkeypatch.setattr(pipeline, "Analyzer", FakeAnalyzer)
    src = tmp_path / "call.mp3"
    src.write_bytes(b"x")
    s = Settings()
    s.cleanup = False
    events = []
    res = asyncio.run(pipeline.run(src, tmp_path, s, on_event=lambda name, data: events.append((name, data))))
    assert [(n, d.get("stage")) for n, d in events] == [("stage", "preprocess"), ("stage", "stt"),
                                                         ("transcript", None), ("stage", "analysis")]
    assert len(events[2][1]["segments"]) == 4 and events[2][1]["review"] == res.review
    assert res.analysis["outcome"] == "site_visit_scheduled" and res.analysis["next_questions"]


# --- advanced analytics: metrics, agent coaching, moments, Q&A, dashboard ---------

def test_call_metrics_talk_time_gaps_and_interruptions():
    from transcriber.metrics import call_metrics

    segs = [Segment(0, "Speaker 1", 0, 40, "Namaste sir, project ke baare mein bata doon? " + "bla " * 80),
            Segment(1, "Speaker 2", 39, 45, "Haan boliye. Price kya hai?"),        # starts 1s early = interruption
            Segment(2, "Speaker 1", 50, 70, "Sir 80 lakh se start hai."),          # 5s dead air before
            Segment(3, "Speaker 2", 71, 80, "Possession kab hai? Loan milega?")]
    m = call_metrics(segs, 80, agent="Speaker 1", customer="Speaker 2")
    a, c = m["speakers"]["Speaker 1"], m["speakers"]["Speaker 2"]
    assert (a["talk_seconds"], c["talk_seconds"]) == (60, 15) and (a["talk_pct"], c["talk_pct"]) == (80, 20)
    assert m["talk_ratio"] == "80:20" and m["agent_talk_pct"] == 80
    assert c["interruptions"] == 1 and a["interruptions"] == 0
    assert m["dead_air"]["count"] == 1 and m["dead_air"]["longest"][0] == {"start": 45, "seconds": 5.0}
    assert c["questions"] == 3 and a["questions"] == 1 and a["turns"] == 2
    assert m["longest_monologue"]["speaker"] == "Speaker 1" and m["longest_monologue"]["seconds"] == 40
    assert a["avg_reply_seconds"] == 5.0 and c["avg_reply_seconds"] == 1.0
    hints = [h["text"] for h in m["hints"]]
    assert any("Agent talked 80%" in h for h in hints) and any("Customer asked 3" in h for h in hints)
    assert any("asked only 1 question" in h for h in hints)
    # Without roles it still measures per speaker, but makes no agent/customer claims.
    assert "hints" not in call_metrics(segs, 80) and call_metrics([], 10) == {}


def agent_perf() -> dict:
    return {"skills": [{"skill": "opening", "score": 5, "comment": "Clear intro.", "segment_id": 0,
                        "quote": "main Neha bol rahi hoon Green Valley se"},
                       {"skill": "closing", "score": 4, "comment": "Fixed a visit.", "segment_id": 2,
                        "quote": "Sir site visit ke liye aa sakte hain?"},
                       {"skill": "objection_handling", "score": 0, "comment": "No objections.", "segment_id": -1,
                        "quote": ""},
                       {"skill": "discovery", "score": 9, "comment": "Asked budget.", "segment_id": 1,  # customer line
                        "quote": "budget mera 80 lakh tak hai"},
                       {"skill": "opening", "score": 1, "comment": "dup", "segment_id": -1, "quote": ""},
                       {"skill": "made_up", "score": 3, "comment": "", "segment_id": -1, "quote": ""}],
            "strengths": ["Fixed a concrete visit time", " "],
            "improvements": [{"tip": "Ask about loan needs", "segment_id": 2, "quote": "site visit ke liye",
                              "better_line": "Sir, loan ke liye bhi help chahiye kya?"},
                             {"tip": "", "segment_id": -1, "quote": "", "better_line": ""}],
            "compliance": []}


def test_agent_scorecard_moments_and_mood():
    from transcriber.analysis import to_markdown, verify

    a = verify(fake_analysis(agent_performance=agent_perf(),
                             key_moments=[{"type": "commitment", "segment_id": 3, "label": "Fixed Sunday visit"},
                                          {"type": "buying_signal", "segment_id": 1, "label": "Shared budget"},
                                          {"type": "objection", "segment_id": 99, "label": "Invented"},
                                          {"type": "nonsense", "segment_id": 1, "label": "Bad type"}],
                             sentiment_arc={"start": "neutral", "middle": "positive", "end": "weird"},
                             sentiment_trend="Warmed up once the budget fit."), call_segments())
    ap = a["agent_performance"]
    assert [(x["key"], x["score"]) for x in ap["skills"]] == [("opening", 5), ("discovery", 5),
                                                            ("objection_handling", 0), ("closing", 4)]
    assert ap["skills"][0]["start"] == 0 and "quote" not in ap["skills"][1]   # a customer line is not agent proof
    assert ap["score"] == 93 and ap["grade"] == "A"                           # (5+5+4)/3*20, n/a not counted
    assert ap["strengths"] == ["Fixed a concrete visit time"] and len(ap["improvements"]) == 1
    assert ap["improvements"][0]["start"] == 12
    assert any("agent-coaching quote" in f for f in a["flags"])
    assert [(m["type"], m["start"]) for m in a["key_moments"]] == [("buying_signal", 6), ("commitment", 16)]
    assert a["sentiment_arc"] == {"start": "neutral", "middle": "positive", "end": "unknown"}
    # Re-verifying a saved analysis keeps the cleaned scorecard as it is.
    assert verify(dict(a), call_segments())["agent_performance"] == ap
    md = to_markdown(a, "x")
    assert "## Agent scorecard: 93/100 (grade A)" in md and "- Objection handling: n/a" in md
    assert "Better: _Sir, loan ke liye bhi help chahiye kya?_" in md
    assert "start: neutral -> middle: positive -> end: unknown" in md and "[00:00:16] Commitment" in md
    # Older analyses without these fields still verify.
    assert verify(fake_analysis(), call_segments())["agent_performance"] == {}
    # A wrong number is not a sales call: no agent score to drag the team average down.
    wrong = verify(fake_analysis(outcome="wrong_number", agent_performance=agent_perf()), call_segments())
    assert wrong["agent_performance"]["score"] is None and wrong["agent_performance"]["grade"] is None
    assert "## Agent scorecard\n" in to_markdown(wrong, "x")


def test_pipeline_saves_metrics_with_roles(tmp_path, monkeypatch):
    from transcriber import pipeline

    class Provider:
        async def transcribe(self, audio, duration):
            return Transcript(call_segments(), duration=duration, provider="test", model="m")

    class FakeAnalyzer:
        def __init__(self, settings, cost):
            pass

        async def analyze(self, segments):
            from transcriber.analysis import verify
            return verify(fake_analysis(), segments)

    async def fake_preprocess(src, work, denoise=True):
        return work / "x.16k.flac"

    async def fake_duration(src):
        return 21.0

    monkeypatch.setattr(pipeline, "check_ffmpeg", lambda: None)
    monkeypatch.setattr(pipeline, "probe_duration", fake_duration)
    monkeypatch.setattr(pipeline, "preprocess", fake_preprocess)
    monkeypatch.setattr(pipeline, "get_provider", lambda s, c: Provider())
    monkeypatch.setattr(pipeline, "Analyzer", FakeAnalyzer)
    src = tmp_path / "call.mp3"
    src.write_bytes(b"x")
    s = Settings()
    s.cleanup = False
    res = asyncio.run(pipeline.run(src, tmp_path, s))
    m = json.loads(res.paths["json"].read_text(encoding="utf-8"))["meta"]["metrics"]
    assert m["roles"] == {"agent": "Speaker 1", "customer": "Speaker 2"} and m["talk_ratio"] == "48:52"


def test_call_qa_checks_citations():
    from transcriber.qa import CallQA

    class Resp:
        text = json.dumps({"answer": "Budget is 80 lakh, in Noida.",
                           "citations": [{"segment_id": 1, "quote": "budget mera 80 lakh tak hai"},
                                         {"segment_id": 1, "quote": "2 crore bhi chalega"}]})
        usage_metadata = None

    class Models:
        async def generate_content(self, model, contents, config):
            assert contents.endswith("QUESTION: Budget kya hai?") and "Q: pehla sawal" in contents
            return Resp()

    s = Settings()
    s.gemini_api_key = "test"
    qa = CallQA(s)
    qa.client = type("C", (), {"aio": type("A", (), {"models": Models()})()})()
    out = asyncio.run(qa.ask(call_segments(), "Budget kya hai?", [{"q": "pehla sawal", "a": "jawab"}]))
    assert out["answer"].startswith("Budget is 80 lakh")
    assert out["citations"] == [{"segment_id": 1, "speaker": "Speaker 2", "start": 6,
                                 "quote": "budget mera 80 lakh tak hai"}]
    with pytest.raises(ValueError):
        asyncio.run(qa.ask(call_segments(), "   "))


def test_dashboard_endpoints(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app as webapp
    from transcriber.analysis import verify

    monkeypatch.setattr(webapp, "RUNS", tmp_path)
    monkeypatch.setattr(webapp, "APP_PASSWORD", "")
    run = tmp_path / "abcdef0123"
    run.mkdir()
    (run / "call.mp3").write_bytes(b"x")
    t = Transcript(call_segments(), duration=21, provider="test", model="m")
    a = verify(fake_analysis(agent_performance=agent_perf()), call_segments())
    write_outputs(t, run, "call", [], {"total_usd": 0.01}, a)
    (tmp_path / "0000000000").mkdir()   # a failed run with no transcript is skipped

    c = TestClient(webapp.app)
    runs = c.get("/api/runs").json()["runs"]
    assert len(runs) == 1 and runs[0]["run_id"] == "abcdef0123" and runs[0]["file"] == "call.mp3"
    assert runs[0]["outcome"] == "site_visit_scheduled" and runs[0]["lead_score"] == 86
    assert runs[0]["agent_grade"] == "A"
    full = c.get("/api/runs/abcdef0123").json()
    assert full["analysis"]["outcome"] == "site_visit_scheduled" and "words" not in full["segments"][0]
    assert set(full["downloads"]) >= {"json", "analysis_md"} and full["audio_url"].endswith("/audio")
    assert c.get("/api/runs/0000000000").status_code == 404 and c.get("/api/runs/zzz").status_code == 404
    csv_text = c.get("/api/leads.csv").text
    assert csv_text.startswith("﻿created,file,outcome_label") and "Site visit scheduled" in csv_text

    class FakeQA:
        def __init__(self, settings):
            pass

        async def ask(self, segs, question, history):
            assert len(segs) == 4 and question == "Budget?"
            return {"answer": "80 lakh", "citations": []}

    monkeypatch.setattr(webapp, "CallQA", FakeQA)
    assert c.post("/api/runs/abcdef0123/ask", json={"question": "Budget?"}).json()["answer"] == "80 lakh"


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
