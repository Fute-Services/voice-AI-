"""Call outcome analysis for real-estate sales calls.

Reads the final transcript and decides what the customer actually said:
interested / site visit booked / follow up / not interested / ..., plus their
requirements (budget, location, BHK...) and the next step for the sales team.

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
from .retry import APIError, Transient, with_retry

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
{context}Return JSON only."""

_STR = {"type": "string"}
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
        "evidence": {
            "type": "array",
            "items": {"type": "object",
                      "properties": {"segment_id": {"type": "integer"}, "quote": _STR},
                      "required": ["segment_id", "quote"]},
        },
    },
    "required": ["customer_speaker", "agent_speaker", "outcome", "lead_temperature", "confidence", "reason",
                 "summary", "next_action", "follow_up_when", "requirements", "objections", "evidence"],
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


def verify(result: dict, segments: list[Segment]) -> dict:
    """Keep only evidence quotes that really occur in the transcript; downgrade unsupported decisions."""
    by_id = {s.id: s for s in segments}
    good, dropped = [], 0
    for e in result.get("evidence") or []:
        quote = (e.get("quote") or "").strip()
        seg = by_id.get(e.get("segment_id"))
        # Accept the quote if it's in the cited segment, or (wrong id) in any segment.
        if quote and seg and _norm(quote) in _norm(seg.text):
            good.append({"segment_id": seg.id, "speaker": seg.speaker, "start": seg.start, "quote": quote})
            continue
        hit = next((s for s in segments if quote and _norm(quote) in _norm(s.text)), None)
        if hit:
            good.append({"segment_id": hit.id, "speaker": hit.speaker, "start": hit.start, "quote": quote})
        else:
            dropped += 1
    result["evidence"] = good

    flags = []
    if dropped:
        flags.append(f"{dropped} evidence quote(s) not found in transcript and removed")
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
            "agent_speaker": "", "needs_review": True, "flags": [reason]}


class Analyzer:
    def __init__(self, settings: Settings, cost: CostTracker):
        if not settings.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set (.env) - needed for call analysis (or use --no-analysis)")
        self.s = settings
        self.cost = cost
        self.client = genai.Client(api_key=settings.gemini_api_key,
                                   http_options=types.HttpOptions(timeout=300_000))
        outcomes = "\n".join(f"- {k}: {v}" for k, v in OUTCOMES.items())
        ctx = f"\nBUSINESS CONTEXT (projects, locations, names):\n{settings.analysis_context}\n\n" \
            if settings.analysis_context else "\n"
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM.format(outcomes=outcomes, context=ctx),
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

        models = [m for m in (self.s.analysis_model or self.s.cleanup_model, self.s.cleanup_fallback_model) if m]
        for model in dict.fromkeys(models):
            try:
                data = await with_retry(f"Analysis ({model})", call, model, attempts=4)
                data["model"] = model
                return verify(data, segments)
            except APIError as exc:
                log.warning("%s", exc)
        log.error("call analysis failed on %s", models)
        return empty_result("analysis failed (API error) - check the logs")


def to_markdown(a: dict, stem: str) -> str:
    req = {k: v for k, v in (a.get("requirements") or {}).items() if v}
    out = [f"# Call analysis: {stem}", "",
           f"**Outcome: {a['outcome_label']}**  ",
           f"Lead: {a['lead_temperature']} · confidence: {a['confidence']}"
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
    if a.get("evidence"):
        from .output import ts
        out += ["", "## Evidence", *[f"- [{ts(e['start'])}] {e['speaker']}: \"{e['quote']}\"" for e in a["evidence"]]]
    if a.get("flags"):
        out += ["", "## Flags", *[f"- {f}" for f in a["flags"]]]
    return "\n".join(out) + "\n"
