"""Ask the call: answer a free-form question about one transcript ("Customer ka budget kya tha?").

The answer must come from the transcript. The model cites the segments it used; each cited quote is checked
against the transcript and dropped if it is not really there, so the UI only links to lines that exist.
"""
from __future__ import annotations

import json
import logging

from google import genai
from google.genai import types

from .analysis import _locate, build_prompt
from .config import ConfigError, Settings
from .cost import CostTracker
from .llm import generate
from .models import Segment
from .retry import APIError, Transient, with_fallback

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 500
MAX_HISTORY = 6  # earlier question/answer pairs sent along, so follow-ups like "aur uska timeline?" work

SYSTEM = """You answer questions about ONE recorded phone call between a real-estate sales agent and a customer in
India. The call is usually in Hinglish. You get the transcript with segment ids and speakers.

RULES
1. Answer ONLY from the transcript. If the call does not contain the answer, say so plainly ("The call does not
   mention the customer's budget.") - never guess or fill in from general knowledge.
2. Keep the answer short: 1-4 sentences, or a few bullet lines starting with "- " for lists.
3. Answer in the language of the question: English question -> simple English; Hinglish question -> Hinglish in
   Roman script.
4. citations: 1-3 quotes copied EXACTLY (character for character) from the transcript that support the answer,
   with their segment id. Empty list only if the answer is that the call does not say.
5. Refer to people as "the agent" and "the customer" (or by name if the call gives one).
{context}Return JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array",
                      "items": {"type": "object",
                                "properties": {"segment_id": {"type": "integer"}, "quote": {"type": "string"}},
                                "required": ["segment_id", "quote"]}},
    },
    "required": ["answer", "citations"],
}

SUGGESTED = [
    "What is the customer's budget and preferred location?",
    "Customer ne kya objection diya?",
    "What did the agent promise to do after the call?",
    "Did the customer mention any other project or builder?",
    "Agent ne kahan galti ki?",
]


def check_citations(raw, segments: list[Segment]) -> list[dict]:
    out, seen = [], set()
    for c in raw or []:
        quote = (c.get("quote") or "").strip()
        hit = _locate(quote, c.get("segment_id"), segments)
        if hit and (hit.id, quote) not in seen:
            seen.add((hit.id, quote))
            out.append({"segment_id": hit.id, "speaker": hit.speaker, "start": hit.start, "quote": quote})
    return out[:3]


class CallQA:
    def __init__(self, settings: Settings, cost: CostTracker | None = None):
        if not settings.gemini_api_key:
            raise ConfigError("GEMINI_API_KEY is not set (.env) - needed to ask questions about a call")
        self.s = settings
        self.cost = cost or CostTracker()
        self.client = genai.Client(api_key=settings.gemini_api_key, http_options=types.HttpOptions(timeout=45_000))
        ctx = f"\nBUSINESS CONTEXT:\n{settings.analysis_context}\n\n" if settings.analysis_context else "\n"
        self.config = types.GenerateContentConfig(system_instruction=SYSTEM.format(context=ctx),
                                                  response_mime_type="application/json",
                                                  response_json_schema=RESPONSE_SCHEMA, max_output_tokens=2048)

    async def ask(self, segments: list[Segment], question: str, history: list[dict] | None = None) -> dict:
        question = (question or "").strip()[:MAX_QUESTION_CHARS]
        if not question:
            raise ValueError("Ask a question")
        if not segments:
            return {"answer": "This call has no transcript to answer from.", "citations": [], "model": None}
        past = [h for h in (history or []) if h.get("q") and h.get("a")][-MAX_HISTORY:]
        prompt = build_prompt(segments)
        if past:
            prompt += "\n\nEARLIER QUESTIONS IN THIS CHAT:\n" + "\n".join(
                f"Q: {h['q'][:MAX_QUESTION_CHARS]}\nA: {h['a'][:1000]}" for h in past)
        prompt += f"\n\nQUESTION: {question}"

        async def call(model: str) -> dict:
            resp = await generate(self.client, model, prompt, self.config, self.s.thinking_level)
            self.cost.add_gemini("qa", model, resp.usage_metadata)
            try:
                data = json.loads(resp.text)
                if not (data.get("answer") or "").strip():
                    raise ValueError("empty answer")
                return data
            except (TypeError, ValueError) as exc:
                raise Transient(f"unparseable answer: {exc}") from exc

        models = [self.s.analysis_model or self.s.cleanup_model, *self.s.fallback_models]
        try:
            data, model = await with_fallback("Ask", call, models)
        except APIError as exc:
            log.error("call Q&A failed on %s", models)
            raise APIError("The AI service is busy - try the question again in a minute.") from exc
        return {"answer": data["answer"].strip(), "citations": check_citations(data.get("citations"), segments),
                "model": model, "usd": self.cost.total_usd}
