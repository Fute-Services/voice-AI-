"""Retry with exponential backoff for transient API failures."""
from __future__ import annotations

import logging
import re

import httpx
from google.genai import errors as genai_errors
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential_jitter

log = logging.getLogger(__name__)

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
# A 429 asking us to wait longer than this is a quota wall, not a blip: fail fast.
MAX_SERVER_RETRY_DELAY_S = 120


class APIError(RuntimeError):
    """Non-retryable API failure with a human-readable message."""


class Transient(RuntimeError):
    """Raise for failures worth retrying that aren't HTTP errors (e.g. malformed LLM JSON)."""


def _server_retry_delay(exc: BaseException) -> float | None:
    m = re.search(r"retry in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", str(exc))
    if not m:
        return None
    h, mi, s = m.groups()
    return int(h or 0) * 3600 + int(mi or 0) * 60 + float(s)


def is_transient(exc: BaseException) -> bool:
    if isinstance(exc, (httpx.TransportError, TimeoutError, Transient)):
        return True
    status = None
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    elif isinstance(exc, genai_errors.APIError):
        status = exc.code
    if status not in RETRY_STATUS:
        return False
    delay = _server_retry_delay(exc)
    return delay is None or delay <= MAX_SERVER_RETRY_DELAY_S


def _log_retry(state) -> None:
    exc = state.outcome.exception()
    log.warning("attempt %d failed (%s: %s); retrying", state.attempt_number, type(exc).__name__, str(exc)[:200])


async def with_retry(what: str, fn, *args, attempts: int = 5, **kwargs):
    """Await fn(*args, **kwargs), retrying transient errors. Wraps final failure in APIError."""
    try:
        async for attempt in AsyncRetrying(
            retry=retry_if_exception(is_transient),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential_jitter(initial=3, max=60),
            before_sleep=_log_retry,
            reraise=True,
        ):
            with attempt:
                return await fn(*args, **kwargs)
    except APIError:
        raise
    except Exception as exc:  # noqa: BLE001 - convert to a clear message
        raise APIError(f"{what} failed: {describe(exc)}") from exc


def describe(exc: BaseException) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        body = exc.response.text[:300]
        hint = {401: " (check the API key)", 403: " (key lacks access)", 429: " (rate limit / quota)"}.get(code, "")
        return f"HTTP {code}{hint}: {body}"
    if isinstance(exc, genai_errors.APIError):
        hint = ""
        if exc.code == 429:
            hint = " (quota exceeded - free-tier keys have low limits; enable billing or pick another model)"
        elif exc.code in (400, 403) and "API key" in str(exc):
            hint = " (check GEMINI_API_KEY)"
        return f"Gemini {exc.code}{hint}: {str(exc)[:400]}"
    return f"{type(exc).__name__}: {exc}"
