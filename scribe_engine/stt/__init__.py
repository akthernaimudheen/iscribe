"""STT provider abstraction.

ScribeEngine depends on this interface, never on a concrete Whisper
implementation. Providers are registered by id and resolved via
`get_provider(provider_id)` so the experimental Malayalam pipeline can be
benchmarked against (and later selected instead of) the current provider
without touching engine code.
"""

from .base import STTProvider, Transcript, Segment, SpeechLanguageUnavailable

# ids of built-in providers
CURRENT_PROVIDER_ID = "current_faster_whisper"
DEEPGRAM_PROVIDER_ID = "deepgram"
MALAYALAM_PROVIDER_ID = "malayalam_indicconformer"

# Language -> provider preference. Malayalam is NEVER in PRODUCTION_PRIORITY:
# it is reachable only when a consultation explicitly asks for it, because an
# ml-locked model returns confident nonsense for English audio.
LANGUAGE_PROVIDERS = {"ml": (MALAYALAM_PROVIDER_ID,)}

# Production preference order for the English path. Deepgram first when it is
# configured; faster-whisper is the always-available local fallback, so local
# development never needs a Deepgram key.
PRODUCTION_PRIORITY = (DEEPGRAM_PROVIDER_ID, CURRENT_PROVIDER_ID)

_PROVIDERS: dict = {}


def register(provider_id: str, factory):
    _PROVIDERS[provider_id] = factory


def get_provider(provider_id: str | None = None, **kwargs) -> STTProvider:
    """Build an STTProvider by id. Defaults to the current provider."""
    pid = provider_id or CURRENT_PROVIDER_ID
    if pid not in _PROVIDERS:
        raise KeyError(
            f"Unknown STT provider '{pid}'. Available: {sorted(_PROVIDERS)}"
        )
    return _PROVIDERS[pid](**kwargs)


def available_providers() -> list:
    return sorted(_PROVIDERS)


from .current_provider import CurrentSTTProvider  # noqa: E402
from .deepgram_provider import DeepgramSTTProvider  # noqa: E402
from .malayalam_provider import MalayalamSidecarProvider  # noqa: E402

register(CURRENT_PROVIDER_ID, CurrentSTTProvider)
register(DEEPGRAM_PROVIDER_ID, DeepgramSTTProvider)
register(MALAYALAM_PROVIDER_ID, MalayalamSidecarProvider)


def resolve_production_provider(preferred: str | None = None,
                                language: str | None = None, **kwargs):
    """Pick the STT provider for the production path.

    Order: the language map first (a non-English consultation must get a
    provider that speaks it), then an explicitly preferred provider, then
    Deepgram when a key is configured, then faster-whisper. Returns
    ``(provider, reason)`` so the service can log and display which engine
    actually ran and why.

    A non-English consultation overrides any preferred provider: language is
    a requirement, not a preference. Measured on real consultation audio, a
    preferred English engine handed Malayalam audio returns confident fluent
    nonsense (benchmarks/malayalam/real_audio_asr/), so the language map is
    consulted first even when ISCRIBE_STT_PROVIDER names an engine explicitly.
    """
    lang = (language or "en").split("-")[0].lower()
    lang_pids = LANGUAGE_PROVIDERS.get(lang, ())
    if lang_pids:
        for pid in lang_pids:
            provider = get_provider(pid, **kwargs)
            if getattr(provider, "is_configured", True):
                reason = f"language provider for {lang}"
                if preferred and pid != preferred:
                    reason += f" (overrides preferred '{preferred}')"
                return provider, reason
        raise SpeechLanguageUnavailable(
            lang,
            f"Consultation language is '{lang}' but its speech engine "
            f"({', '.join(lang_pids)}) is not available. Start the Malayalam "
            "sidecar, or set the consultation language to English.")

    if preferred:
        return get_provider(preferred, **kwargs), f"explicitly selected: {preferred}"

    for pid in PRODUCTION_PRIORITY:
        provider = get_provider(pid, **kwargs)
        # Hosted providers advertise whether they are usable; local ones always are.
        if getattr(provider, "is_configured", True):
            reason = ("preferred provider is configured" if pid == PRODUCTION_PRIORITY[0]
                      else "fallback: no hosted provider configured")
            return provider, reason

    return get_provider(CURRENT_PROVIDER_ID, **kwargs), "last-resort fallback"


__all__ = [
    "STTProvider",
    "Transcript",
    "Segment",
    "CurrentSTTProvider",
    "DeepgramSTTProvider",
    "MalayalamSidecarProvider",
    "CURRENT_PROVIDER_ID",
    "DEEPGRAM_PROVIDER_ID",
    "MALAYALAM_PROVIDER_ID",
    "LANGUAGE_PROVIDERS",
    "PRODUCTION_PRIORITY",
    "get_provider",
    "register",
    "available_providers",
    "resolve_production_provider",
]
