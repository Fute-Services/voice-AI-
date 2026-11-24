"""Call outcome analysis for real-estate sales calls.

Reads the final transcript and decides what the customer actually said:
interested / site visit booked / follow up / not interested / ..., plus their
requirements (budget, location, BHK...) and the next step for the sales team.

It also profiles the customer's mindset (motive, urgency, decision maker, financing, concerns,
buying / risk signals), tracks each discovery topic (answered / asked but dodged / never raised), writes
the few questions worth asking THIS customer next, and computes a 0-100 lead score.

Every decision must be backed by quotes from the transcript. Quotes are checked
against the segment text; if none of them can be found, the decision is marked
low-confidence and sent for human review instead of being trusted blindly.
"""
from __future__ import annotations

import json
import logging
import re

from google import genai
from google.genai import types

from .config import ConfigError, Settings
from .cost import CostTracker
from .llm import generate
from .models import Segment
from .retry import APIError, Transient, with_fallback

log = logging.getLogger(__name__)

OUTCOMES = {
    "site_visit_scheduled": "Customer agreed to a site visit / meeting with a date or day fixed.",
    "interested": "Customer clearly wants to go ahead or explore (asks for details, brochure, price, wants a visit) "
                  "but nothing is fixed yet.",
    "follow_up": "Customer neither said yes nor no: asked to call back later, will discuss with family, busy now, "
                 "'sochke batata hoon'.",
    "not_interested": "Customer clearly declined: not interested, already bought, budget does not match, "
                      "don't call again.",
    "wrong_number": "Wrong person / not a real lead / never enquired / call is not a sales conversation.",
    "unclear": "Call too short, cut off, or the customer's intent cannot be judged from what was said.",
}
TEMPERATURES = ("hot", "warm", "cold")
CONFIDENCE = ("high", "medium", "low")

# Discovery topics: what a sales agent should find out about a lead.
# key -> (topic, fallback question in English, the same in Hinglish)
# The model writes the questions for each call (see next_questions); this bank is only the fallback when it
# returns none, so keep the wording short, polite and natural enough to say on a live call.
DISCOVERY = {
    "purpose": ("Purpose",
                "Are you looking at this for your own use or as an investment?",
                "Aap yeh khud rehne ke liye dekh rahe hain ya investment ke liye?"),
    "trigger": ("Reason for buying now",
                "What made you start looking for a property now?",
                "Abhi property lene ka plan kaise bana?"),
    "timeline": ("Timeline",
                 "By when are you hoping to finalise?",
                 "Aap kab tak finalise karna chah rahe hain?"),
    "budget": ("Budget",
               "What budget range are you looking in?",
               "Aap kis budget range mein dekh rahe hain?"),
    "financing": ("Loan or own funds",
                  "Will you go for a home loan or use your own funds?",
                  "Payment ke liye loan lenge ya apne funds se karenge?"),
    "decision_maker": ("Who decides",
                       "Who else will be part of the final decision?",
                       "Final decision mein aapke saath aur kaun rahega?"),
    "priorities": ("Must-haves",
                   "What matters most to you in the property?",
                   "Property mein aapke liye sabse zaroori kya hai?"),
    "alternatives": ("Other projects",
                     "Which other projects have you seen so far?",
                     "Ab tak aapne aur kaun se projects dekhe hain?"),
    "concerns": ("Concerns",
                 "Is there anything you are not fully sure about yet?",
                 "Koi aisi baat hai jisko lekar aap abhi sure nahi hain?"),
    "commitment": ("Ready to book",
                   "If you like the site, by when would you think of booking?",
                   "Agar site pasand aa jaye, toh booking kab tak soch rahe hain?"),
    "next_step": ("Site visit",
                  "Which day works for you for a site visit?",
                  "Site visit ke liye aapko kaun sa din theek rahega?"),
}
# What counts as each topic being known - shown to the model so it does not confuse look-alikes
# (the project's possession date is not the customer's timeline; the agent's quoted price is not their budget).
DISCOVERY_MEANING = {
    "purpose": "own use or investment",
    "trigger": "why they are looking for a property now",
    "timeline": "when the CUSTOMER wants to buy or move - not the project's possession date",
    "budget": "the customer's own budget figure or range - not a price the agent quoted",
    "financing": "home loan or own funds",
    "decision_maker": "who else is involved in the decision",
    "priorities": "what they must have: location, size, amenities, school, commute...",
    "alternatives": "other projects, builders or areas they are considering",
    "concerns": "what is holding them back",
    "commitment": "how ready they are to book, or what would make them book",
    "next_step": "a site visit or meeting: answered only if they agreed to one or refused; a vague 'will see' "
                 "to the agent's invitation is asked_not_answered",
}
# Fallback order: what is worth asking first when the model wrote no questions. Earlier = more useful early on.
ASK_ORDER = ("purpose", "budget", "timeline", "priorities", "decision_maker", "next_step", "financing",
             "concerns", "alternatives", "trigger", "commitment")
MAX_QUESTIONS = 5
# A lead still being worked gets at least this many questions: the bank tops up a thin list from the model.
MIN_QUESTIONS = 3
TOP_UP_OUTCOMES = ("interested", "follow_up")
# Outcomes where there is no next sales conversation to prepare questions for.
NO_QUESTIONS = ("wrong_number", "unclear")

MOTIVES = ("self_use", "investment", "upgrade", "rental_income", "unknown")
URGENCY = ("immediate", "within_3_months", "within_6_months", "exploring", "unknown")
DECISION_MAKERS = ("self", "joint_with_family", "someone_else", "unknown")
FINANCING = ("home_loan", "self_funded", "loan_and_own_funds", "unknown")
# Behaviour profile: how the customer behaved on the call, so a salesperson can judge them.
BUYER_TYPES = {
    "serious_buyer": ("Serious buyer", "Clear need, shares details freely, asks concrete questions, moves towards "
                                       "a visit or booking."),
    "careful_evaluator": ("Careful evaluator", "Interested but analytical: compares options, asks detailed "
                                               "questions, wants proof before deciding."),
    "price_focused": ("Price-focused negotiator", "Talks mainly about rate / discount / offers and shows little "
                                                  "interest in anything else."),
    "investor": ("Investor", "Thinks in returns: appreciation, rental yield, resale, payment schedule."),
    "family_dependent": ("Family-dependent buyer", "Interested but will not decide alone; defers to spouse, "
                                                   "parents or someone else."),
    "casual_browser": ("Casual browser", "Collecting information only: no timeline, vague or generic answers."),
    "avoider": ("Avoiding / disengaged", "Busy, very short answers, deflects questions, wants to end the call."),
    "not_a_buyer": ("Not a buyer", "No real need or intent to buy."),
    "unknown": ("Not enough to judge", "The customer said too little to judge their behaviour."),
}
LEVELS = ("low", "medium", "high", "unknown")
TONES = ("positive", "neutral", "hesitant", "irritated", "rushed", "unknown")
TRAITS = {
    "seriousness": ("Seriousness", "how real the intent to buy looks from their own words and actions"),
    "engagement": ("Engagement", "how actively they take part: asks questions, gives full answers, stays on the call"),
    "openness": ("Openness", "how freely they share budget, needs and personal context when asked"),
    "decisiveness": ("Decisiveness", "how readily they commit to a clear answer, day or next step"),
    "trust": ("Trust in the agent", "how much they accept what the agent says versus doubting or challenging it"),
    "price_sensitivity": ("Price sensitivity", "how much of their attention is on price, discount and cost"),
}

# Agent scorecard: how well the SALES AGENT handled the call. key -> (label, what a 5 looks like)
AGENT_SKILLS = {
    "opening": ("Opening", "greets, introduces self and company, states why they are calling, checks it is a good "
                           "time"),
    "rapport": ("Rapport & listening", "warm and respectful, uses the customer's name, listens, acknowledges what "
                                       "the customer says instead of talking over it"),
    "discovery": ("Need discovery", "asks open questions to learn purpose, budget, timeline, location and who decides"),
    "pitch": ("Relevant pitch", "presents the project in terms of what THIS customer said they need; no generic "
                                "brochure reading"),
    "objection_handling": ("Objection handling", "acknowledges each concern, answers it with facts, checks the "
                                                 "customer is satisfied"),
    "closing": ("Closing / next step", "asks for a concrete next step (visit, call-back time, WhatsApp details) and "
                                       "confirms it"),
    "professionalism": ("Professionalism", "polite, clear, no pressure tactics, no false promises or made-up "
                                           "offers"),
}
AGENT_GRADES = ((85, "A"), (70, "B"), (55, "C"), (0, "D"))
MOMENT_TYPES = {
    "buying_signal": "Buying signal",
    "objection": "Objection",
    "price_talk": "Price discussion",
    "commitment": "Commitment",
    "concern": "Concern",
    "turning_point": "Turning point",
    "rapport": "Rapport moment",
}
MAX_MOMENTS = 6
MOODS = ("positive", "neutral", "negative", "unknown")

MINDSET_LABELS = {"motive": "Buying motive", "urgency": "Urgency", "decision_maker": "Decision maker",
                  "financing": "Financing", "primary_concern": "Main concern"}

SYSTEM = """You analyse recorded phone calls between a REAL ESTATE sales agent and a customer (lead) in India.
The calls are usually in Hinglish (Hindi + English mixed). You get a transcript with segment ids and speakers.

Decide what the CUSTOMER actually decided or said - not what the agent hoped for.

OUTCOME - pick exactly one:
{outcomes}

RULES
1. Base everything ONLY on what is said in the transcript. Never assume. A polite "haan haan" or "okay" while the
   agent is talking is NOT agreement; look for a real commitment from the customer.
2. "Sochke batata hoon", "family se baat karke", "baad mein call karo" = follow_up, not interested.
3. If the customer agrees to a visit but no day/date is fixed, use "interested", not "site_visit_scheduled".
4. evidence: 1-3 quotes copied EXACTLY (character for character) from the CUSTOMER's segments that prove the outcome,
   with their segment id. Short quotes (a few words up to one sentence) are best.
5. lead_temperature: hot = ready to visit/buy soon, warm = interested but needs time/info, cold = not interested
   or no real intent.
6. confidence: how sure you are of the outcome from the transcript. Use "low" if the audio seems cut off, the
   customer barely spoke, or the transcript has many [unclear] parts.
7. requirements: fill only what was actually mentioned (budget, location, property type, BHK, purpose, timeline,
   possession). Use "" for anything not mentioned. Keep amounts as spoken (e.g. "80 lakh - 1 crore").
8. follow_up_when: the day/date/time the customer gave for a visit or callback, as spoken (e.g. "Sunday 11 baje",
   "next week"). "" if none.
9. summary, reason, next_action and objections in simple English for a sales manager.
10. mindset: profile the customer's thinking ONLY from what they said. Use "unknown" / "" / an empty list when
    the call does not reveal it - never guess.
    - motive: self_use (to live in), investment, upgrade (moving from a current home), rental_income.
    - urgency: immediate (within about a month), within_3_months, within_6_months, exploring (no timeline,
      just looking).
    - decision_maker: self, joint_with_family (spouse / parents involved), someone_else.
    - financing: home_loan, self_funded, loan_and_own_funds.
    - primary_concern: the single biggest thing holding the customer back, in a few words. "" if none.
    - competitors: other projects / builders / areas the customer said they are considering.
    - pitch_angle: 1-2 sentences telling the agent how to position the next conversation for THIS customer,
      based on their motive and concern. "" if the call reveals nothing to base it on.
11. buying_signals / risk_signals: things the CUSTOMER said that show real intent (asks about payment plan,
    possession date, loan, wants to bring family) or weak intent (only wants the rate, vague timeline, avoids
    committing). Each needs a short description in English and a quote copied EXACTLY from a customer segment
    with its segment id. Leave the list empty if there are none.
12. discovery: for each topic below that came up in the call, one entry. Leave out topics that never came up.
    - status "answered": the CUSTOMER gave the information (asked or volunteered). Put what they said in
      `answer`, in a few words (e.g. "80 lakh - 1 crore", "wife will decide with him").
    - status "asked_not_answered": the agent asked but the customer dodged, deflected or gave no real answer.
      `answer` = "".
    The agent merely mentioning a topic (e.g. quoting a price) does NOT make it answered.
    Topics:
{discovery}
13. behaviour: judge HOW the customer behaved on the call - their manner, not just their words. Judge only from
    this transcript. If the customer spoke very little, use buyer_type "unknown" and trait level "unknown"
    rather than guessing; one polite sentence is not enough to call someone serious or trusting.
    - buyer_type - pick exactly one:
{buyer_types}
    - traits - rate each low / medium / high / unknown:
{traits}
    - tone: the customer's overall emotional tone: positive, neutral, hesitant, irritated, rushed.
    - communication_style: a few words on how they talk (e.g. "Direct and brief", "Polite but evasive",
      "Detailed and questioning").
    - verdict: 2-3 sentences for a sales manager: what kind of customer this is, how genuine they seem, and what
      that judgement is based on.
    - how_to_handle: 2-4 short, practical tips for dealing with THIS customer's behaviour on the next call.
    - observations: 1-4 specific behaviours you noticed (e.g. "Avoided giving a budget", "Asked about possession
      twice"), each with a quote copied EXACTLY from a customer segment and its segment id.
14. next_questions: the questions the agent should ask THIS customer in the NEXT conversation, most important
    first. At most {max_questions}; fewer is better than padding. These are read out on a live call, so:
    - Never ask for something the customer already answered. Build on it instead (budget given -> ask about
      stretch or payment plan, not "what is your budget").
    - Tie each question to this call: use what the customer said, their concern, their stage. A generic
      checklist is useless.
    - Match the stage. Barely-engaged or busy customer: start with one easy opener and the basics (purpose,
      budget, timeline) - no booking or commitment questions. Interested: uncover what blocks a visit.
      Visit fixed: only what helps the visit go well (who is coming, what they want to see, loan help).
      not_interested: at most 1-2 respectful questions to learn the real reason or whether to reconnect
      later; none if they said not to call again. wrong_number: none.
    - A topic marked asked_not_answered needs a softer, different wording - not the same question again.
    - `hinglish`: exactly how a polite Indian sales agent would say it on the phone - Roman script, "aap",
      one short sentence, one question only, no English jargon the customer did not use.
    - `english`: the same question in plain English.
    - `why`: under 15 words for the agent: what this unlocks or why it matters for this customer.
    - `topic`: the discovery topic it belongs to, or "other".
15. agent_performance: coach the SALES AGENT like a strict but fair sales manager, from this transcript only.
    - skills: one entry per skill below. score 1-5 (1 = poor, 3 = acceptable, 5 = excellent), or 0 when the skill
      did not apply to this call (e.g. objection_handling when the customer raised no objection; EVERY skill when
      the call is not a sales conversation at all). Never give 1 for "not applicable". `comment` = one
      short sentence explaining the score. `segment_id` + `quote` = a line copied EXACTLY from an AGENT segment
      that shows it, or segment_id -1 and quote "" when there is nothing to quote.
{skills}
    - strengths: 1-3 specific things the agent did well.
    - improvements: 1-3 specific coaching points. Each: `tip` (what to do differently), the `segment_id` + `quote`
      of the AGENT line where it went wrong (copied EXACTLY, or -1 and ""), and `better_line`: what the agent
      could have said instead, in natural Hinglish (Roman script) as spoken on a call.
    - compliance: problems a manager must know about - pressure tactics, rude remarks, false promises, made-up
      discounts, guaranteed returns. Empty list if none.
16. key_moments: up to {max_moments} moments that mattered most in the call, in time order: where the customer
    showed intent, objected, talked price, committed, raised a concern, or where the call turned. Each: `type`
    (one of {moment_types}), `segment_id` of the segment where it happens, and `label` (under 10 words, English).
17. sentiment_arc: the CUSTOMER's mood at the start, middle and end of the call: positive / neutral / negative,
    or unknown if they said too little in that part. `sentiment_trend` = one short sentence on how it changed.
{context}Return JSON only."""

_STR = {"type": "string"}
_SIGNALS = {"type": "array",
            "items": {"type": "object",
                      "properties": {"signal": _STR, "segment_id": {"type": "integer"}, "quote": _STR},
                      "required": ["signal", "segment_id", "quote"]}}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "customer_speaker": {"type": "string", "description": "Speaker label of the customer, e.g. 'Speaker 2'"},
        "agent_speaker": _STR,
        "outcome": {"type": "string", "enum": list(OUTCOMES)},
        "lead_temperature": {"type": "string", "enum": list(TEMPERATURES)},
        "confidence": {"type": "string", "enum": list(CONFIDENCE)},
        "reason": {"type": "string", "description": "One sentence: why this outcome"},
        "summary": {"type": "string", "description": "2-3 sentence summary of the call"},
        "next_action": {"type": "string", "description": "What the sales agent should do next"},
        "follow_up_when": _STR,
        "requirements": {
            "type": "object",
            "properties": {k: _STR for k in
                           ("budget", "location", "property_type", "bhk", "purpose", "timeline", "other")},
            "required": ["budget", "location", "property_type", "bhk", "purpose", "timeline", "other"],
        },
        "objections": {"type": "array", "items": _STR},
        "mindset": {
            "type": "object",
            "properties": {
                "motive": {"type": "string", "enum": list(MOTIVES)},
                "urgency": {"type": "string", "enum": list(URGENCY)},
                "decision_maker": {"type": "string", "enum": list(DECISION_MAKERS)},
                "financing": {"type": "string", "enum": list(FINANCING)},
                "primary_concern": _STR,
                "competitors": {"type": "array", "items": _STR},
                "pitch_angle": _STR,
            },
            "required": ["motive", "urgency", "decision_maker", "financing", "primary_concern", "competitors",
                         "pitch_angle"],
        },
        "behaviour": {
            "type": "object",
            "properties": {
                "buyer_type": {"type": "string", "enum": list(BUYER_TYPES)},
                "traits": {"type": "object",
                           "properties": {k: {"type": "string", "enum": list(LEVELS)} for k in TRAITS},
                           "required": list(TRAITS)},
                "tone": {"type": "string", "enum": list(TONES)},
                "communication_style": _STR,
                "verdict": _STR,
                "how_to_handle": {"type": "array", "items": _STR},
                "observations": {
                    "type": "array",
                    "items": {"type": "object",
                              "properties": {"observation": _STR, "segment_id": {"type": "integer"}, "quote": _STR},
                              "required": ["observation", "segment_id", "quote"]}},
            },
            "required": ["buyer_type", "traits", "tone", "communication_style", "verdict", "how_to_handle",
                         "observations"],
        },
        "buying_signals": _SIGNALS,
        "risk_signals": _SIGNALS,
        "discovery": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"topic": {"type": "string", "enum": list(DISCOVERY)},
                                     "status": {"type": "string", "enum": ["answered", "asked_not_answered"]},
                                     "answer": _STR},
                      "required": ["topic", "status", "answer"]}},
        "next_questions": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"topic": {"type": "string", "enum": [*DISCOVERY, "other"]},
                                     "hinglish": _STR, "english": _STR, "why": _STR},
                      "required": ["topic", "hinglish", "english", "why"]}},
        "evidence": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"segment_id": {"type": "integer"}, "quote": _STR},
                      "required": ["segment_id", "quote"]},
        },
        "agent_performance": {
            "type": "object",
            "properties": {
                "skills": {"type": "array",
                           "items": {"type": "object",
                                     "properties": {"skill": {"type": "string", "enum": list(AGENT_SKILLS)},
                                                    "score": {"type": "integer"}, "comment": _STR,
                                                    "segment_id": {"type": "integer"}, "quote": _STR},
                                     "required": ["skill", "score", "comment", "segment_id", "quote"]}},
                "strengths": {"type": "array", "items": _STR},
                "improvements": {"type": "array",
                                 "items": {"type": "object",
                                           "properties": {"tip": _STR, "segment_id": {"type": "integer"},
                                                          "quote": _STR, "better_line": _STR},
                                           "required": ["tip", "segment_id", "quote", "better_line"]}},
                "compliance": {"type": "array", "items": _STR},
            },
            "required": ["skills", "strengths", "improvements", "compliance"],
        },
        "key_moments": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"type": {"type": "string", "enum": list(MOMENT_TYPES)},
                                     "segment_id": {"type": "integer"}, "label": _STR},
                      "required": ["type", "segment_id", "label"]}},
        "sentiment_arc": {
            "type": "object",
            "properties": {k: {"type": "string", "enum": list(MOODS)} for k in ("start", "middle", "end")},
            "required": ["start", "middle", "end"],
        },
        "sentiment_trend": _STR,
    },
    "required": ["customer_speaker", "agent_speaker", "outcome", "lead_temperature", "confidence", "reason",
                 "summary", "next_action", "follow_up_when", "requirements", "objections", "mindset",
                 "behaviour", "buying_signals", "risk_signals", "discovery", "next_questions", "evidence",
                 "agent_performance", "key_moments", "sentiment_arc", "sentiment_trend"],
}

OUTCOME_LABELS = {
    "site_visit_scheduled": "Site visit scheduled",
    "interested": "Interested",
    "follow_up": "Follow up",
    "not_interested": "Not interested",
    "wrong_number": "Wrong number / not a lead",
    "unclear": "Unclear",
}


REQ_LABELS = {"budget": "Budget", "location": "Location", "property_type": "Property type", "bhk": "BHK",
              "purpose": "Purpose", "timeline": "Timeline", "other": "Other"}


def build_prompt(segments: list[Segment]) -> str:
    lines = [json.dumps({"id": s.id, "speaker": s.speaker, "text": s.text}, ensure_ascii=False) for s in segments]
    return "TRANSCRIPT:\n" + "\n".join(lines)


def _norm(t: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", t.lower(), flags=re.UNICODE).split())


def _locate(quote: str, segment_id, segments: list[Segment]) -> Segment | None:
    """The segment a quote really occurs in: the cited one, or (wrong id) any other. None if invented."""
    quote = _norm(quote or "")
    if not quote:
        return None
    cited = next((s for s in segments if s.id == segment_id), None)
    if cited and quote in _norm(cited.text):
        return cited
    return next((s for s in segments if quote in _norm(s.text)), None)


_OUTCOME_PTS = {"site_visit_scheduled": 45, "interested": 30, "follow_up": 15, "unclear": 5}
_URGENCY_PTS = {"immediate": 20, "within_3_months": 14, "within_6_months": 7, "exploring": 2}
_DECISION_PTS = {"self": 10, "joint_with_family": 6, "someone_else": 3}


def conversation_stats(segments: list[Segment], customer: str) -> dict:
    """Measured (not judged) behaviour: how much the customer talked and how many questions they asked."""
    mine = [s for s in segments if s.speaker == customer]
    if not mine:
        return {}
    words = sum(len(s.text.split()) for s in mine)
    total = sum(len(s.text.split()) for s in segments)
    return {"talk_share_pct": round(100 * words / total) if total else 0,
            "customer_turns": len(mine),
            "avg_words_per_turn": round(words / len(mine)),
            "questions_asked": sum(s.text.count("?") for s in mine)}


def lead_score(result: dict) -> int:
    """0-100, computed from the outcome and how much real buying intent the call revealed."""
    outcome = result.get("outcome")
    if outcome == "wrong_number":
        return 0
    m, req = result.get("mindset") or {}, result.get("requirements") or {}
    score = (_OUTCOME_PTS.get(outcome, 0) + _URGENCY_PTS.get(m.get("urgency"), 0)
             + _DECISION_PTS.get(m.get("decision_maker"), 0)
             + (10 if m.get("financing") not in (None, "", "unknown") else 0)
             + (8 if req.get("budget") else 0) + (4 if req.get("location") else 0)
             + (3 if m.get("motive") not in (None, "", "unknown") else 0))
    return min(score, 15) if outcome == "not_interested" else score


def discovery_report(entries, outcome: str | None) -> dict:
    """Per-topic discovery status: answered (with what the customer said), dodged, or never raised.

    `entries` is the model's list of {topic, status, answer}; a plain list of topic keys (analyses saved
    before per-topic status existed) is read as "answered".
    """
    if isinstance(entries, dict):  # an already-built report (re-verifying a saved analysis)
        entries = [{"topic": c["key"], "status": "answered", "answer": c.get("answer", "")}
                   for c in entries.get("covered") or []] + [
                  {"topic": c["key"], "status": "asked_not_answered"} for c in entries.get("dodged") or []]
    status: dict[str, tuple[str, str]] = {}
    for e in entries or []:
        key, st, answer = (e, "answered", "") if isinstance(e, str) else (
            e.get("topic"), e.get("status"), (e.get("answer") or "").strip())
        if key in DISCOVERY and key not in status:
            status[key] = ("dodged" if st == "asked_not_answered" else "answered", answer)
    covered = [{"key": k, "topic": DISCOVERY[k][0], "answer": status[k][1]}
               for k in DISCOVERY if status.get(k, ("",))[0] == "answered"]
    dodged = [{"key": k, "topic": DISCOVERY[k][0]} for k in DISCOVERY if status.get(k, ("",))[0] == "dodged"]
    # No point listing gaps for a wrong number or a call that could not be judged.
    missed = [] if outcome in NO_QUESTIONS else [
        {"key": k, "topic": topic, "question": q, "hinglish": h}
        for k, (topic, q, h) in DISCOVERY.items() if k not in status]
    return {"covered": covered, "dodged": dodged, "missed": missed, "total": len(DISCOVERY)}


def next_questions(raw, discovery: dict, outcome: str | None) -> list[dict]:
    """The model's tailored questions, cleaned up; the static bank fills in only when it wrote none."""
    if outcome in NO_QUESTIONS:
        return []
    out, seen = [], set()
    for q in raw or []:
        hinglish, english = (q.get("hinglish") or "").strip(), (q.get("english") or "").strip()
        if not (hinglish or english) or _norm(hinglish or english) in seen:
            continue
        seen.add(_norm(hinglish or english))
        key = q.get("topic") if q.get("topic") in DISCOVERY else "other"
        out.append({"key": key, "topic": DISCOVERY[key][0] if key in DISCOVERY else "Follow-up",
                    "hinglish": hinglish or english, "english": english, "why": (q.get("why") or "").strip(),
                    "tailored": True})
    out = out[:MAX_QUESTIONS]
    # Never push a generic checklist at someone who said no, or pad the prep for a visit that is already fixed.
    if outcome not in TOP_UP_OUTCOMES and (out or outcome == "not_interested"):
        return out
    # Top up from the bank: topics the agent raised but got no answer on first, then the most basic unknowns.
    missed, asked = {m["key"] for m in discovery["missed"]}, {q["key"] for q in out}
    open_keys = [k for k in [d["key"] for d in discovery["dodged"]] + [k for k in ASK_ORDER if k in missed]
                 if k not in asked]
    return out + [{"key": k, "topic": DISCOVERY[k][0], "hinglish": DISCOVERY[k][2], "english": DISCOVERY[k][1],
                   "why": "", "tailored": False} for k in open_keys[:max(0, MIN_QUESTIONS - len(out))]]


def agent_report(raw: dict | None, segments: list[Segment], agent: str) -> tuple[dict, int]:
    """Clean the agent scorecard: clamp scores, keep only real agent quotes, compute the 0-100 score and grade.

    Returns (report, number of quotes removed). A skill scored 0 did not apply and does not count."""
    if not raw or not isinstance(raw, dict):
        return {}, 0
    dropped = 0

    def quote_of(item: dict) -> dict:
        nonlocal dropped
        q = (item.get("quote") or "").strip()
        if not q:
            return {}
        hit = _locate(q, item.get("segment_id"), segments)
        if hit and (not agent or hit.speaker == agent):
            return {"segment_id": hit.id, "start": hit.start, "quote": q}
        dropped += 1
        return {}

    skills, seen = [], set()
    for sk in raw.get("skills") or []:
        key = sk.get("skill")
        if key not in AGENT_SKILLS or key in seen:
            continue
        seen.add(key)
        try:
            score = max(0, min(5, int(sk.get("score") or 0)))
        except (TypeError, ValueError):
            score = 0
        skills.append({"key": key, "label": AGENT_SKILLS[key][0], "score": score,
                       "comment": (sk.get("comment") or "").strip(), **quote_of(sk)})
    skills.sort(key=lambda x: list(AGENT_SKILLS).index(x["key"]))
    rated = [x["score"] for x in skills if x["score"]]
    overall = round(sum(rated) / len(rated) * 20) if rated else None
    improvements = []
    for imp in raw.get("improvements") or []:
        tip = (imp.get("tip") or "").strip()
        if tip:
            improvements.append({"tip": tip, "better_line": (imp.get("better_line") or "").strip(), **quote_of(imp)})
    return {"skills": skills, "score": overall,
            "grade": next(g for cut, g in AGENT_GRADES if overall >= cut) if overall is not None else None,
            "strengths": [x.strip() for x in raw.get("strengths") or [] if x and x.strip()][:3],
            "improvements": improvements[:3],
            "compliance": [x.strip() for x in raw.get("compliance") or [] if x and x.strip()]}, dropped


def moments_report(raw, segments: list[Segment]) -> list[dict]:
    """Key moments pinned to real segments (unknown ids dropped), in time order, with their timestamps."""
    by_id = {s.id: s for s in segments}
    out, seen = [], set()
    for m in raw or []:
        seg, kind = by_id.get(m.get("segment_id")), m.get("type")
        label = (m.get("label") or "").strip()
        if not seg or kind not in MOMENT_TYPES or not label or (seg.id, kind) in seen:
            continue
        seen.add((seg.id, kind))
        out.append({"type": kind, "type_label": MOMENT_TYPES[kind], "label": label, "segment_id": seg.id,
                    "speaker": seg.speaker, "start": seg.start})
    return sorted(out, key=lambda x: x["start"])[:MAX_MOMENTS]


def verify(result: dict, segments: list[Segment]) -> dict:
    """Keep only evidence quotes that really occur in the transcript; downgrade unsupported decisions."""
    good, dropped = [], 0
    for e in result.get("evidence") or []:
        quote = (e.get("quote") or "").strip()
        hit = _locate(quote, e.get("segment_id"), segments)
        if hit:
            good.append({"segment_id": hit.id, "speaker": hit.speaker, "start": hit.start, "quote": quote})
        else:
            dropped += 1
    result["evidence"] = good

    flags = []
    if dropped:
        flags.append(f"{dropped} evidence quote(s) not found in transcript and removed")
    unbacked = 0
    for key in ("buying_signals", "risk_signals"):
        kept = []
        for sig in result.get(key) or []:
            quote, text = (sig.get("quote") or "").strip(), (sig.get("signal") or "").strip()
            hit = _locate(quote, sig.get("segment_id"), segments)
            if hit and text:
                kept.append({"signal": text, "segment_id": hit.id, "speaker": hit.speaker, "start": hit.start,
                             "quote": quote})
            else:
                unbacked += 1
        result[key] = kept
    b = result.get("behaviour") or {}
    if b:
        seen = []
        for o in b.get("observations") or []:
            quote, text = (o.get("quote") or "").strip(), (o.get("observation") or "").strip()
            hit = _locate(quote, o.get("segment_id"), segments)
            if hit and text:
                seen.append({"observation": text, "segment_id": hit.id, "speaker": hit.speaker, "start": hit.start,
                             "quote": quote})
            else:
                unbacked += 1
        b["observations"] = seen
        b["buyer_type_label"] = BUYER_TYPES.get(b.get("buyer_type"), BUYER_TYPES["unknown"])[0]
        b["stats"] = conversation_stats(segments, result.get("customer_speaker") or "")
    result["behaviour"] = b
    if unbacked:
        flags.append(f"{unbacked} signal(s) / observation(s) had no matching quote in the transcript and were removed")
    result["mindset"] = result.get("mindset") or {}
    # "discovery_covered" = the field before per-topic status existed (still read when re-verifying old analyses).
    result["discovery"] = discovery_report(result.pop("discovery", None) or result.pop("discovery_covered", None),
                                           result.get("outcome"))
    result.pop("discovery_covered", None)
    result["next_questions"] = next_questions(result.get("next_questions"), result["discovery"],
                                              result.get("outcome"))
    result["lead_score"] = lead_score(result)
    # A saved analysis being re-verified already holds the cleaned report: keep it as it is.
    ap = result.get("agent_performance")
    if not (isinstance(ap, dict) and "grade" in ap):
        ap, bad_agent = agent_report(ap, segments, result.get("agent_speaker") or "")
        if bad_agent:
            flags.append(f"{bad_agent} agent-coaching quote(s) not found in the agent's lines and removed")
        # Not a sales conversation: sales skills cannot be judged, so no score that would drag the agent's average.
        if ap and result.get("outcome") == "wrong_number":
            for sk in ap["skills"]:
                sk["score"] = 0
            ap.update(score=None, grade=None, improvements=[])
    result["agent_performance"] = ap or {}
    result["key_moments"] = moments_report(result.get("key_moments"), segments)
    arc = result.get("sentiment_arc") or {}
    result["sentiment_arc"] = {k: arc.get(k) if arc.get(k) in MOODS else "unknown" for k in ("start", "middle", "end")}
    result["sentiment_trend"] = (result.get("sentiment_trend") or "").strip()
    if not good and result.get("outcome") != "unclear":
        flags.append("no verifiable evidence for the outcome")
        result["confidence"] = "low"
    n_unclear = sum(s.text.count("[unclear]") for s in segments)
    if n_unclear >= max(3, len(segments) // 5):
        flags.append(f"transcript has {n_unclear} [unclear] parts")
        if result.get("confidence") == "high":
            result["confidence"] = "medium"
    result["outcome_label"] = OUTCOME_LABELS.get(result.get("outcome"), result.get("outcome"))
    result["needs_review"] = result.get("confidence") == "low" or result.get("outcome") == "unclear"
    result["flags"] = flags
    return result


def empty_result(reason: str) -> dict:
    return {"outcome": "unclear", "outcome_label": OUTCOME_LABELS["unclear"], "lead_temperature": "cold",
            "confidence": "low", "reason": reason, "summary": "", "next_action": "Listen to the recording manually.",
            "follow_up_when": "", "requirements": {}, "objections": [], "evidence": [], "customer_speaker": "",
            "mindset": {}, "behaviour": {}, "buying_signals": [], "risk_signals": [], "lead_score": 0,
            "discovery": {"covered": [], "dodged": [], "missed": [], "total": len(DISCOVERY)}, "next_questions": [],
            "agent_speaker": "", "agent_performance": {}, "key_moments": [],
            "sentiment_arc": {"start": "unknown", "middle": "unknown", "end": "unknown"}, "sentiment_trend": "",
            "needs_review": True, "flags": [reason]}


class Analyzer:
    def __init__(self, settings: Settings, cost: CostTracker):
        if not settings.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set (.env) - needed for call analysis (or use --no-analysis)")
        self.s = settings
        self.cost = cost
        self.client = genai.Client(api_key=settings.gemini_api_key,
                                   http_options=types.HttpOptions(timeout=60_000))
        outcomes = "\n".join(f"- {k}: {v}" for k, v in OUTCOMES.items())
        ctx = f"\nBUSINESS CONTEXT (projects, locations, names):\n{settings.analysis_context}\n\n" \
            if settings.analysis_context else "\n"
        topics = "\n".join(f"    - {k}: {DISCOVERY_MEANING[k]}" for k in DISCOVERY)
        buyer_types = "\n".join(f"      * {k}: {desc}" for k, (_, desc) in BUYER_TYPES.items())
        traits = "\n".join(f"      * {k}: {desc}" for k, (_, desc) in TRAITS.items())
        skills = "\n".join(f"      * {k}: {desc}" for k, (_, desc) in AGENT_SKILLS.items())
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM.format(outcomes=outcomes, context=ctx, discovery=topics,
                                             buyer_types=buyer_types, traits=traits, max_questions=MAX_QUESTIONS,
                                             skills=skills, max_moments=MAX_MOMENTS,
                                             moment_types=", ".join(MOMENT_TYPES)),
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            max_output_tokens=12288,
        )

    async def analyze(self, segments: list[Segment]) -> dict:
        if not segments:
            return empty_result("empty transcript")
        prompt = build_prompt(segments)

        async def call(model: str) -> dict:
            resp = await generate(self.client, model, prompt, self.config, self.s.thinking_level)
            self.cost.add_gemini("analysis", model, resp.usage_metadata)
            try:
                data = json.loads(resp.text)
                if data.get("outcome") not in OUTCOMES:
                    raise ValueError(f"bad outcome {data.get('outcome')!r}")
                return data
            except (TypeError, ValueError, KeyError) as exc:
                raise Transient(f"unparseable analysis output: {exc}") from exc

        models = [self.s.analysis_model or self.s.cleanup_model, *self.s.fallback_models]
        try:
            data, model = await with_fallback("Analysis", call, models)
        except APIError:
            log.error("call analysis failed on %s", models)
            return empty_result("analysis failed (AI service busy or unavailable) - try again in a few minutes")
        data["model"] = model
        return verify(data, segments)


def _pretty(value: str) -> str:
    """'joint_with_family' -> 'Joint with family'; free text is left alone."""
    return value.replace("_", " ").capitalize() if re.fullmatch(r"[a-z0-9_]+", value) else value


def to_markdown(a: dict, stem: str) -> str:
    req = {k: v for k, v in (a.get("requirements") or {}).items() if v}
    out = [f"# Call analysis: {stem}", "",
           f"**Outcome: {a['outcome_label']}**  ",
           f"Lead: {a['lead_temperature']} · score: {a.get('lead_score', 0)}/100 · confidence: {a['confidence']}"
           + (" · **needs human review**" if a.get("needs_review") else ""), "",
           f"**Why:** {a.get('reason', '')}", "",
           f"**Summary:** {a.get('summary', '')}", "",
           f"**Next action:** {a.get('next_action', '')}"]
    if a.get("follow_up_when"):
        out.append(f"  \n**When:** {a['follow_up_when']}")
    if req:
        out += ["", "## Requirements", *[f"- {REQ_LABELS.get(k, k)}: {v}" for k, v in req.items()]]
    if a.get("objections"):
        out += ["", "## Objections", *[f"- {o}" for o in a["objections"]]]
    from .output import ts
    b = a.get("behaviour") or {}
    if b:
        out += ["", "## Customer behaviour", f"**Type: {b.get('buyer_type_label', '')}**"
                + (f" · tone: {b['tone']}" if b.get("tone") not in (None, "", "unknown") else "")
                + (f" · style: {b['communication_style']}" if b.get("communication_style") else "")]
        if b.get("verdict"):
            out += ["", b["verdict"]]
        rated = [f"- {label}: {(b.get('traits') or {}).get(k)}" for k, (label, _) in TRAITS.items()
                 if (b.get("traits") or {}).get(k) not in (None, "", "unknown")]
        if rated:
            out += ["", *rated]
        st = b.get("stats") or {}
        if st:
            out += ["", f"Customer spoke {st['talk_share_pct']}% of the call · {st['customer_turns']} turns · "
                        f"{st['avg_words_per_turn']} words per turn · asked {st['questions_asked']} question(s)"]
        if b.get("how_to_handle"):
            out += ["", "**How to handle:**", *[f"- {t}" for t in b["how_to_handle"]]]
        if b.get("observations"):
            out += ["", "**Behaviour observed:**",
                    *[f"- {o['observation']} - [{ts(o['start'])}] \"{o['quote']}\"" for o in b["observations"]]]
    m = a.get("mindset") or {}
    mind = [f"- {label}: {_pretty(m[k])}" for k, label in MINDSET_LABELS.items()
            if m.get(k) not in (None, "", "unknown")]
    if m.get("competitors"):
        mind.append(f"- Also considering: {', '.join(m['competitors'])}")
    if mind:
        out += ["", "## Customer mindset", *mind]
    if m.get("pitch_angle"):
        out += ["", f"**How to pitch next:** {m['pitch_angle']}"]
    for key, title in (("buying_signals", "Buying signals"), ("risk_signals", "Risk signals")):
        if a.get(key):
            out += ["", f"## {title}", *[f"- {x['signal']} - [{ts(x['start'])}] \"{x['quote']}\"" for x in a[key]]]
    d = a.get("discovery") or {}
    qs = a.get("next_questions") or []
    if qs:
        out += ["", "## Questions to ask next"]
        for i, q in enumerate(qs, 1):
            note = " - ".join(x for x in (f"_{q['english']}_" if q.get("english") else "", q.get("why")) if x)
            out += [f"{i}. **{q['hinglish']}**" + (f"  \n   {note}" if note else "")]
    if d.get("covered") or d.get("dodged") or d.get("missed"):
        out += ["", f"## Discovery ({len(d.get('covered') or [])}/{d.get('total', len(DISCOVERY))} topics known)"]
        out += [f"- {c['topic']}: {c.get('answer') or 'covered'}" for c in d.get("covered") or []]
        if d.get("dodged"):
            out.append("- Asked but not answered: " + ", ".join(x["topic"] for x in d["dodged"]))
        if d.get("missed"):
            out.append("- Not discussed yet: " + ", ".join(x["topic"] for x in d["missed"]))
    ap = a.get("agent_performance") or {}
    if ap.get("skills"):
        out += ["", f"## Agent scorecard: {ap['score']}/100 (grade {ap['grade']})" if ap.get("score") is not None
                else "## Agent scorecard"]
        out += [f"- {x['label']}: " + (f"{x['score']}/5" if x["score"] else "n/a")
                + (f" - {x['comment']}" if x.get("comment") else "") for x in ap["skills"]]
        if ap.get("strengths"):
            out += ["", "**Did well:**", *[f"- {x}" for x in ap["strengths"]]]
        if ap.get("improvements"):
            out += ["", "**Coaching:**"]
            for x in ap["improvements"]:
                out.append(f"- {x['tip']}" + (f" - [{ts(x['start'])}] \"{x['quote']}\"" if x.get("quote") else ""))
                if x.get("better_line"):
                    out.append(f"  - Better: _{x['better_line']}_")
        if ap.get("compliance"):
            out += ["", "**Compliance issues:**", *[f"- {x}" for x in ap["compliance"]]]
    arc = a.get("sentiment_arc") or {}
    if any(v not in (None, "", "unknown") for v in arc.values()):
        out += ["", "## Customer mood",
                " -> ".join(f"{k}: {arc.get(k, 'unknown')}" for k in ("start", "middle", "end"))
                + (f"  \n{a['sentiment_trend']}" if a.get("sentiment_trend") else "")]
    if a.get("key_moments"):
        out += ["", "## Key moments",
                *[f"- [{ts(x['start'])}] {x['type_label']}: {x['label']}" for x in a["key_moments"]]]
    if a.get("evidence"):
        out += ["", "## Evidence", *[f"- [{ts(e['start'])}] {e['speaker']}: \"{e['quote']}\"" for e in a["evidence"]]]
    if a.get("flags"):
        out += ["", "## Flags", *[f"- {f}" for f in a["flags"]]]
    return "\n".join(out) + "\n"
