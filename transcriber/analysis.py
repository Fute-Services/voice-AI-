"""Call outcome analysis for real-estate sales calls.

Reads the final transcript and decides what the customer actually said:
interested / site visit booked / follow up / not interested / ..., plus their
requirements (budget, location, BHK...) and the next step for the sales team.

It also profiles the customer's mindset (motive, urgency, decision maker, financing, concerns,
buying / risk signals), checks which discovery questions the agent covered, lists the ones still
to ask, and computes a 0-100 lead score.

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

# Discovery question bank: what a sales agent should find out on a call.
# key -> (topic, the question in English, the same in Hinglish)
# Keep these short and natural: an agent must be able to say them on a live call without sounding scripted.
DISCOVERY = {
    "purpose": ("Purpose",
                "Is this for your own use or for investment?",
                "Aap apne rehne ke liye le rahe hain ya investment ke liye?"),
    "trigger": ("Reason for buying now",
                "Why are you looking for a property now?",
                "Abhi property lene ka plan kaise bana?"),
    "timeline": ("Timeline",
                 "When do you plan to buy?",
                 "Aap kab tak lena chahte hain?"),
    "budget": ("Budget",
               "What is your budget?",
               "Aapka budget kitna hai?"),
    "financing": ("Loan or own funds",
                  "Will you take a home loan or pay yourself?",
                  "Loan lenge ya khud payment karenge?"),
    "decision_maker": ("Who decides",
                       "Who else will decide with you?",
                       "Aapke saath aur kaun decide karega?"),
    "priorities": ("Must-haves",
                   "What is most important for you in the property?",
                   "Property mein aapke liye sabse zaroori kya hai?"),
    "alternatives": ("Other projects",
                     "Have you seen any other projects?",
                     "Aapne aur koi project dekha hai?"),
    "concerns": ("Concerns",
                 "Is anything stopping you right now?",
                 "Koi cheez hai jo aapko rok rahi hai?"),
    "commitment": ("Ready to book",
                   "If you like it, can you book this month?",
                   "Agar pasand aaya, toh is mahine booking kar sakte hain?"),
    "next_step": ("Site visit",
                  "When can you come for a site visit?",
                  "Site visit ke liye kab aa sakte hain?"),
}

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
12. discovery_covered: the discovery topics that were covered in the call - either the agent asked about it or
    the customer gave the information on their own. Topics:
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
        "discovery_covered": {"type": "array", "items": {"type": "string", "enum": list(DISCOVERY)}},
        "evidence": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"segment_id": {"type": "integer"}, "quote": _STR},
                      "required": ["segment_id", "quote"]},
        },
    },
    "required": ["customer_speaker", "agent_speaker", "outcome", "lead_temperature", "confidence", "reason",
                 "summary", "next_action", "follow_up_when", "requirements", "objections", "mindset",
                 "behaviour", "buying_signals", "risk_signals", "discovery_covered", "evidence"],
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


def discovery_report(covered: list[str] | None, outcome: str | None) -> dict:
    """Which discovery topics the call covered, and the questions the agent should still ask."""
    seen = set(covered or [])
    # No point coaching on questions for a wrong number or a call that could not be judged.
    missed = [] if outcome in ("wrong_number", "unclear") else [
        {"key": k, "topic": topic, "question": q, "hinglish": h}
        for k, (topic, q, h) in DISCOVERY.items() if k not in seen]
    return {"covered": [{"key": k, "topic": DISCOVERY[k][0]} for k in DISCOVERY if k in seen], "missed": missed,
            "total": len(DISCOVERY)}


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
    result["discovery"] = discovery_report(result.pop("discovery_covered", None), result.get("outcome"))
    result["lead_score"] = lead_score(result)
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
            "discovery": {"covered": [], "missed": [], "total": len(DISCOVERY)},
            "agent_speaker": "", "needs_review": True, "flags": [reason]}


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
        topics = "\n".join(f"    - {k}: {topic}" for k, (topic, _, _) in DISCOVERY.items())
        buyer_types = "\n".join(f"      * {k}: {desc}" for k, (_, desc) in BUYER_TYPES.items())
        traits = "\n".join(f"      * {k}: {desc}" for k, (_, desc) in TRAITS.items())
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM.format(outcomes=outcomes, context=ctx, discovery=topics,
                                             buyer_types=buyer_types, traits=traits),
            response_mime_type="application/json",
            response_json_schema=RESPONSE_SCHEMA,
            max_output_tokens=8192,
        )

    async def analyze(self, segments: list[Segment]) -> dict:
        if not segments:
            return empty_result("empty transcript")
        prompt = build_prompt(segments)

        async def call(model: str) -> dict:
            resp = await self.client.aio.models.generate_content(model=model, contents=prompt, config=self.config)
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
    if d.get("missed"):
        out += ["", f"## Questions to ask next ({len(d.get('covered') or [])}/{d.get('total', len(DISCOVERY))} "
                    "topics covered)"]
        for q in d["missed"]:
            out += [f"- **{q['topic']}:** {q['question']}  ", f"  _{q['hinglish']}_"]
    if a.get("evidence"):
        out += ["", "## Evidence", *[f"- [{ts(e['start'])}] {e['speaker']}: \"{e['quote']}\"" for e in a["evidence"]]]
    if a.get("flags"):
        out += ["", "## Flags", *[f"- {f}" for f in a["flags"]]]
    return "\n".join(out) + "\n"
