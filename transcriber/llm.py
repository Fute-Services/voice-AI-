"""One place for Gemini text calls, so every stage gets the same latency settings."""
from __future__ import annotations

import logging

from google.genai import errors as genai_errors
from google.genai import types

log = logging.getLogger(__name__)

THINKING_LEVELS = ("minimal", "low", "medium", "high")
# Models that rejected a thinking level once: don't pay for the failed round-trip again.
_no_thinking: set[str] = set()


async def generate(client, model: str, contents, config: types.GenerateContentConfig, thinking_level: str = "",
                   timeout_s: float | None = None):
    """generate_content with a capped thinking level and, optionally, a deadline sized to this request.

    By default these models "think" before answering, which roughly doubles the latency and the output-token
    bill of the cleanup and analysis calls without changing their result. A model that does not accept the
    setting is simply called again without it.
    """
    if timeout_s:
        config = config.model_copy(update={"http_options": types.HttpOptions(timeout=int(timeout_s * 1000))})
    level = (thinking_level or "").strip().lower()
    if level in THINKING_LEVELS and model not in _no_thinking:
        capped = config.model_copy(update={"thinking_config": types.ThinkingConfig(thinking_level=level.upper())})
        try:
            return await client.aio.models.generate_content(model=model, contents=contents, config=capped)
        except genai_errors.APIError as exc:
            if exc.code != 400 or "think" not in str(exc).lower():
                raise
            _no_thinking.add(model)
            log.info("%s does not accept thinking_level=%s; using its default", model, level)
    return await client.aio.models.generate_content(model=model, contents=contents, config=config)
