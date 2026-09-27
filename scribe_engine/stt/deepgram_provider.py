"""DeepgramSTTProvider — hosted English STT for the production path.

Implements the existing STTProvider interface, so the engine selects it exactly
like any other provider and nothing downstream changes.

Transport: Deepgram's documented pre-recorded REST endpoint
(``POST https://api.deepgram.com/v1/listen``) called with httpx, which is
already a dependency. The ``deepgram-sdk`` package is not installed in this
environment, and adding a heavy SDK for one HTTP POST would be a larger change
than the repair warrants. No API parameter here is invented — each maps to a
documented query parameter of that endpoint.

The API key is read from the environment only. It is never logged, never placed
in a URL, and never returned to a caller.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from .base import Segment, STTProvider, Transcript

logger = logging.getLogger(__name__)

# Deepgram's medical-tuned English model. Overridable because model availability
# depends on the account/plan; an unknown name is surfaced as a clear error
# rather than silently falling back to a general model.
DEFAULT_MODEL = "nova-3-medical"

# Regional, data-residency endpoints (Deepgram-documented): EU, Australia,
# India. Same keys and API; only the base URL changes. The global endpoint
# carries no residency guarantee.
REGIONAL_ENDPOINTS = {
    "global": "api.deepgram.com",
    "eu": "api.eu.deepgram.com",
    "au": "api.au.deepgram.com",
    "in": "api.in.deepgram.com",
}

_EXT_CONTENT_TYPES = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}


class DeepgramError(RuntimeError):
    """Deepgram could not produce a transcript."""


class DeepgramSTTProvider(STTProvider):
    provider_id = "deepgram"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        timeout: float = 300.0,
        diarize: bool = True,
        data_endpoint: str | None = None,
        mip_opt_out: bool = True,
        **_ignored,
    ):
        # Environment only. Never accept a key from request data or source.
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY", "")
        self.model = model
        self.language = language
        self.timeout = timeout
        self.diarize = diarize
        # Regional endpoint (data residency boundary, Deepgram-documented:
        # api.eu / api.au / api.in.deepgram.com). Same keys; the base URL is
        # the only change. Default is the global endpoint.
        self.data_endpoint = (data_endpoint
                              or os.environ.get("ISCRIBE_DEEPGRAM_ENDPOINT", "")
                              or "api.deepgram.com").replace("https://", "").strip("/")
        # Model Improvement Program opt-out. Deepgram documentation: requests
        # participate in MIP by default and are RETAINED for training; with
        # mip_opt_out=true the request audio and transcript are "retained only
        # for the duration needed to process the request" (zero data
        # retention). Clinical audio defaults to opted-out.
        self.mip_opt_out = mip_opt_out

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # -- retention / security capability surface -----------------------------
    def provider_name(self) -> str:
        return "Deepgram (nova-3-medical, pre-recorded REST)"

    def supports_delete_after_processing(self) -> bool:
        # There is no per-request deletion API to call. With mip_opt_out=true
        # the documented behaviour is zero retention, so there is nothing to
        # delete; with MIP enabled, deletion is a contractual/account matter,
        # not an API capability.
        return False

    def data_location(self) -> str:
        # Regional endpoints (api.eu/api.au/api.in.deepgram.com) are the
        # documented residency boundary; the global endpoint gives none.
        return {
            "api.eu.deepgram.com": "EU",
            "api.au.deepgram.com": "Australia",
            "api.in.deepgram.com": "India",
        }.get(self.data_endpoint, "global (no residency guarantee)")

    def retention_configuration(self) -> dict:
        """Evidence-based account of third-party processor retention,
        per Deepgram's published documentation (developers.deepgram.com,
        'Your Data at Deepgram' and the Model Improvement Program page).

        This describes THIRD-PARTY PROCESSOR RETENTION. It says nothing about
        OUR storage retention, which the service lifecycle layer owns.
        Contractual commitments (DPA/BAA) are NOT verified by this codebase.
        """
        return {
            "retention_known": True,
            "audio_sent_off_host": True,
            "deletion_supported": False,
            "endpoint": f"https://{self.data_endpoint}/v1/listen",
            "data_location": self.data_location(),
            "model": self.model,
            "diarize": self.diarize,
            "mip_opt_out": self.mip_opt_out,
            "retention_behaviour": (
                "mip_opt_out=true: zero data retention - audio and transcript "
                "retained only for the duration needed to process the request "
                "(provider-documented). Default (MIP): audio and transcript "
                "retained to improve Deepgram models. Request metadata and "
                "usage logs are retained 90 days either way and contain no "
                "audio or transcripts."
            ),
            "metadata_sent": [
                "audio bytes", "model", "language", "punctuate",
                "smart_format", "numerals", "diarize", "utterances",
                "profanity_filter", "mip_opt_out",
            ],
            "notes": (
                "Behaviour described here is from Deepgram's published "
                "documentation, not from a contract this codebase can verify. "
                "Obtain the DPA (and BAA where applicable) and regional "
                "commitments for the account before production use, or run "
                "ISCRIBE_STT_PROVIDER=current_faster_whisper for fully local "
                "processing."
            ),
        }

    def health_check(self) -> dict:
        # Presence-only: never echoes the key or full request state.
        return {
            "provider": self.provider_id,
            "ok": self.is_configured,
            "configured": self.is_configured,
        }

    def _params(self, language: str | None) -> dict:
        """Documented /v1/listen query parameters, tuned for clinical English."""
        return {
            "model": self.model,
            # Explicit language. Never 'detect' on the production English path.
            "language": language or self.language,
            "punctuate": "true",
            "smart_format": "true",   # dates, times, common clinical formatting
            "numerals": "true",       # "fifty milligrams" -> "50 mg"
            "diarize": "true" if self.diarize else "false",
            "utterances": "true",     # speaker-attributed utterances
            "profanity_filter": "false",  # never alter clinical speech
            # Zero data retention: excluded from Deepgram's Model Improvement
            # Program, so audio + transcript are retained only to process the
            # request. Documented behaviour; verify the DPA for the account.
            "mip_opt_out": "true" if self.mip_opt_out else "false",
        }

    def transcribe(self, audio_path, language: str | None = None) -> Transcript:
        if not self.is_configured:
            raise DeepgramError(
                "Deepgram is implemented but not enabled because DEEPGRAM_API_KEY "
                "is not configured."
            )

        import httpx  # already a dependency (via the service stack)

        path = Path(audio_path)
        content_type = _EXT_CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        requested_language = language or self.language

        t0 = time.time()
        try:
            with path.open("rb") as fh:
                response = httpx.post(
                    f"https://{self.data_endpoint}/v1/listen",
                    params=self._params(requested_language),
                    headers={
                        "Authorization": f"Token {self.api_key}",
                        "Content-Type": content_type,
                    },
                    content=fh.read(),
                    timeout=self.timeout,
                )
        except httpx.HTTPError as exc:
            # Message only — never the headers, which carry the key.
            raise DeepgramError(f"Deepgram request failed: {type(exc).__name__}") from exc

        if response.status_code != 200:
            detail = ""
            try:
                body = response.json()
                detail = body.get("err_msg") or body.get("error") or ""
            except Exception:
                detail = response.text[:200]
            raise DeepgramError(
                f"Deepgram returned HTTP {response.status_code}"
                + (f": {detail}" if detail else "")
            )

        payload = response.json()
        elapsed = time.time() - t0
        transcript = self._to_transcript(payload, requested_language, elapsed)
        # Provider boundary metadata for the audit trail: where the audio
        # went and under which retention flag. Never includes the key.
        transcript.provider_meta["data_endpoint"] = self.data_endpoint
        transcript.provider_meta["mip_opt_out"] = self.mip_opt_out
        return transcript

    # -- response mapping ---------------------------------------------------
    def _to_transcript(self, payload: dict, requested_language: str,
                       elapsed: float) -> Transcript:
        results = payload.get("results") or {}
        channels = results.get("channels") or []
        if not channels:
            raise DeepgramError("Deepgram returned no channels.")
        alternatives = channels[0].get("alternatives") or []
        if not alternatives:
            raise DeepgramError("Deepgram returned no transcription alternatives.")

        best = alternatives[0]
        text = (best.get("transcript") or "").strip()

        # Prefer utterances: they carry speaker attribution and clean boundaries.
        segments: list[Segment] = []
        speaker_turns: list[dict] = []
        for utt in results.get("utterances") or []:
            utt_text = (utt.get("transcript") or "").strip()
            if not utt_text:
                continue
            start = float(utt.get("start", 0.0))
            end = float(utt.get("end", start))
            segments.append(Segment(start=start, end=end, text=utt_text))
            if utt.get("speaker") is not None:
                speaker_turns.append({
                    "speaker": f"Speaker {utt['speaker']}",
                    "start": start, "end": end, "text": utt_text,
                })

        if not segments:
            # No utterances (diarization disabled or unsupported) — fall back to
            # paragraph/word timings so downstream stage reporting still works.
            for para in ((best.get("paragraphs") or {}).get("paragraphs") or []):
                for sent in para.get("sentences") or []:
                    sent_text = (sent.get("text") or "").strip()
                    if sent_text:
                        segments.append(Segment(
                            start=float(sent.get("start", 0.0)),
                            end=float(sent.get("end", 0.0)),
                            text=sent_text,
                        ))
        if not segments and text:
            segments = [Segment(start=0.0, end=0.0, text=text)]

        metadata = payload.get("metadata") or {}
        # Deepgram echoes the language it actually used; keep it so the
        # validation layer can catch a mismatch instead of trusting the request.
        detected = (channels[0].get("detected_language")
                    or metadata.get("language")
                    or requested_language)

        return Transcript(
            text=text,
            segments=segments,
            language=detected,
            duration=metadata.get("duration"),
            source_language=detected,
            provider_id=self.provider_id,
            provider_meta={
                "model": self.model,
                "requested_language": requested_language,
                "request_seconds": round(elapsed, 2),
                "model_uuid": (metadata.get("model_info") or {}),
                "speaker_turns": speaker_turns,
                "diarized": bool(speaker_turns),
                "request_id": metadata.get("request_id"),
            },
        )
