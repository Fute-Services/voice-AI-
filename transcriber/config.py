"""Settings, loaded from environment / .env and overridable from the CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path.cwd() / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name) or default


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    return default if v is None or v == "" else v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    # Keys
    assemblyai_api_key: str = field(default_factory=lambda: os.getenv("ASSEMBLYAI_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # STT
    stt_provider: str = field(default_factory=lambda: _env("STT_PROVIDER", "auto"))  # auto|assemblyai|gemini
    # AssemblyAI: Universal-3.5 Pro handles en+hi code-switching; Universal-2 as fallback.
    assemblyai_models: list[str] = field(
        default_factory=lambda: _env("ASSEMBLYAI_MODELS", "universal-3-5-pro,universal-2").split(",")
    )
    language_codes: list[str] = field(default_factory=lambda: _env("LANGUAGE_CODES", "en,hi").split(","))
    speakers_expected: int | None = field(
        default_factory=lambda: int(os.environ["SPEAKERS_EXPECTED"]) if os.getenv("SPEAKERS_EXPECTED") else None
    )
    gemini_stt_model: str = field(default_factory=lambda: _env("GEMINI_STT_MODEL", "gemini-3.5-transcribe"))
    # Names / jargon to bias recognition (comma separated), e.g. "Sharma Logistics,IFSC,UTR"
    keyterms: list[str] = field(
        default_factory=lambda: [t.strip() for t in os.getenv("KEYTERMS", "").split(",") if t.strip()]
    )

    # Pre-processing
    denoise: bool = field(default_factory=lambda: _bool("DENOISE", True))

    # LLM cleanup
    cleanup: bool = field(default_factory=lambda: _bool("CLEANUP", True))
    # gemini-4-argon is the most accurate but needs a paid (billing-enabled) key.
    cleanup_model: str = field(default_factory=lambda: _env("CLEANUP_MODEL", "gemini-3.6-flash"))
    # Used for a chunk when the primary model keeps failing (e.g. 503 "high demand"). Empty = none.
    cleanup_fallback_model: str = field(default_factory=lambda: _env("CLEANUP_FALLBACK_MODEL", "gemini-3.5-flash"))
    hindi_script: str = field(default_factory=lambda: _env("HINDI_SCRIPT", "roman"))  # roman|devanagari
    chunk_threshold_s: float = field(default_factory=lambda: float(_env("CHUNK_THRESHOLD_S", "1200")))
    chunk_len_s: float = field(default_factory=lambda: float(_env("CHUNK_LEN_S", "600")))
    chunk_overlap_s: float = field(default_factory=lambda: float(_env("CHUNK_OVERLAP_S", "15")))
    max_concurrency: int = field(default_factory=lambda: int(_env("MAX_CONCURRENCY", "2")))

    # Call outcome analysis (real-estate: interested / site visit / follow up / not interested ...)
    analysis: bool = field(default_factory=lambda: _bool("ANALYSIS", True))
    # Empty = same model as cleanup (falls back to CLEANUP_FALLBACK_MODEL on failure)
    analysis_model: str = field(default_factory=lambda: os.getenv("ANALYSIS_MODEL", ""))
    # Optional business context, e.g. "Projects: Green Valley (Sector 150, Noida), Skyline Towers"
    analysis_context: str = field(default_factory=lambda: os.getenv("ANALYSIS_CONTEXT", ""))

    # Review
    low_confidence: float = field(default_factory=lambda: float(_env("LOW_CONFIDENCE", "0.6")))

    def resolve_provider(self) -> str:
        if self.stt_provider != "auto":
            return self.stt_provider
        if self.assemblyai_api_key:
            return "assemblyai"
        if self.gemini_api_key:
            return "gemini"
        raise ConfigError("No STT key found. Set ASSEMBLYAI_API_KEY (recommended) or GEMINI_API_KEY in .env")


class ConfigError(RuntimeError):
    pass
