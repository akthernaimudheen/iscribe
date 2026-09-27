"""STT provider interface.

Transcript is language-aware by design: `language` plus `source_language`
preserve which language the audio actually was, and `translated_transcript`
carries the English rendering when a pipeline performs translation. The
original (non-English) transcript is never discarded — clinical traceability
requires both.
"""

from dataclasses import dataclass, field
from typing import Optional


class SpeechLanguageUnavailable(RuntimeError):
    """A consultation's language was requested but no engine that speaks it ran.

    Raised instead of silently degrading to an English engine, which produces
    confident nonsense on non-English audio (measured on real consultation
    audio). The message is written for the clinician; technical detail travels
    in `detail` for server logs only.
    """

    _LANGUAGE_NAMES = {"ml": "Malayalam", "en": "English", "hi": "Hindi",
                       "ta": "Tamil", "te": "Telugu", "kn": "Kannada"}

    def __init__(self, requested_language: str, detail: str = ""):
        self.requested_language = requested_language
        self.detail = detail
        code = (requested_language or "").split("-")[0].lower()
        lang = self._LANGUAGE_NAMES.get(code,
                                        code.upper() if code else "the consultation language")
        super().__init__(
            f"Speech recognition for {lang} is not available right now. "
            "No transcript was created, so no clinical note was generated. "
            "Please try again shortly; if it keeps failing, select English "
            "for this consultation or contact the study team."
        )


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    text: str
    segments: list = field(default_factory=list)
    language: Optional[str] = None              # detected/declared language of the audio
    language_probability: Optional[float] = None
    duration: Optional[float] = None
    # Language-pair fields (used by translation pipelines; None for English-native STT):
    source_language: Optional[str] = None       # e.g. "ml" when audio was Malayalam
    source_transcript: Optional[str] = None     # transcript in the original language
    target_language: Optional[str] = None       # e.g. "en"
    translated_transcript: Optional[str] = None # English rendering, when produced
    provider_id: Optional[str] = None
    provider_meta: dict = field(default_factory=dict)


class STTProvider:
    """Interface every transcription provider implements.

    `transcribe` must return a Transcript. Providers that translate must ALSO
    populate source_transcript in the original language.

    Retention/security capabilities (default answers are the honest minimum):
    the retention layer must never assume a provider deleted data, and must
    never hard-code one vendor's behaviour. Concrete providers override what
    they can truthfully claim; anything unknown stays "unknown".
    """

    provider_id: str = "base"

    def transcribe(self, audio_path, language: Optional[str] = None) -> Transcript:
        raise NotImplementedError

    # -- retention / security capability surface -----------------------------
    def provider_name(self) -> str:
        """Human-safe provider name for audit and UI display."""
        return self.provider_id

    def supports_delete_after_processing(self) -> bool:
        """True ONLY when this integration can verifiably request provider-side
        deletion. Default: False (do not claim deletions we cannot prove)."""
        return False

    def retention_configuration(self) -> dict:
        """What we truthfully know about provider-side retention.

        `retention_known: False` means the application holds no verified
        contractual/technical knowledge — a documentation claim, never a
        deletion guarantee. `audio_sent_off_host` distinguishes our storage
        retention from third-party processor retention.
        """
        return {
            "retention_known": False,
            "audio_sent_off_host": False,
            "deletion_supported": False,
            "notes": "provider does not declare retention behaviour",
        }

    def health_check(self) -> dict:
        """Presence-only health (never leaks keys or URLs with credentials)."""
        return {"provider": self.provider_id, "ok": True}

    def data_location(self) -> str:
        """Geography where the provider processes the audio, only when that is
        truthfully known from provider documentation. Default: unknown."""
        return "unknown (provider does not declare processing location)"
