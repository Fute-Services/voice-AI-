from ..config import ConfigError, Settings
from ..cost import CostTracker
from .assemblyai import AssemblyAIProvider
from .base import STTProvider
from .gemini import GeminiProvider

PROVIDERS: dict[str, type[STTProvider]] = {
    AssemblyAIProvider.name: AssemblyAIProvider,
    GeminiProvider.name: GeminiProvider,
}


def get_provider(settings: Settings, cost: CostTracker) -> STTProvider:
    name = settings.resolve_provider()
    try:
        return PROVIDERS[name](settings, cost)
    except KeyError:
        raise ConfigError(f"Unknown STT provider {name!r}; choose from {sorted(PROVIDERS)}") from None
