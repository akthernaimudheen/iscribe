"""MalayalamSidecarProvider — Malayalam STT via the local sidecar.

Implements the standard STTProvider interface, so the engine treats Malayalam
audio exactly like any other provider and nothing downstream changes.

Why a sidecar rather than an in-process model: IndicConformer requires Python
3.12 with the AI4Bharat NeMo fork and numpy 1.26, while the iScribe service runs
Python 3.14 with faster-whisper and numpy 2.x. The two cannot coexist in one
interpreter, so the model runs in its own process and is reached over loopback
HTTP. Nothing leaves the machine.

This provider is registered but NEVER selected automatically. It is used only
when a consultation explicitly requests Malayalam, because an ml-locked model
returns confident nonsense for English audio.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .base import Segment, STTProvider, Transcript

logger = logging.getLogger(__name__)

DEFAULT_URL = "http://127.0.0.1:8131"


class MalayalamSidecarError(RuntimeError):
    """The Malayalam sidecar could not produce a transcript."""


class MalayalamSidecarProvider(STTProvider):
    provider_id = "malayalam_indicconformer"

    def __init__(self, url: str | None = None, timeout: float = 600.0,
                 translate: bool = True, **_ignored):
        self.url = (url or os.environ.get("MALAYALAM_SIDECAR_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout
        # The English rendering is a convenience; the Malayalam transcript is
        # the record. Translation failing must not fail the transcription.
        self.translate = translate

    @property
    def is_configured(self) -> bool:
        """True when the sidecar is reachable AND has its model loaded.

        Checked live rather than from config: the sidecar is a separate process
        that can be down, and claiming Malayalam is available when it is not
        would strand a clinician mid-consultation.
        """
        try:
            import httpx

            r = httpx.get(f"{self.url}/ready", timeout=3.0)
            return r.status_code == 200 and bool(r.json().get("asr_ready"))
        except Exception:
            return False

    def transcribe(self, audio_path, language: str | None = None) -> Transcript:
        import httpx

        path = Path(audio_path)
        try:
            with path.open("rb") as fh:
                response = httpx.post(
                    f"{self.url}/transcribe",
                    files={"file": (path.name, fh, "application/octet-stream")},
                    data={"translate": "true" if self.translate else "false"},
                    timeout=self.timeout,
                )
        except httpx.HTTPError as exc:
            raise MalayalamSidecarError(
                f"Malayalam sidecar unreachable at {self.url} "
                f"({type(exc).__name__}). Start it with "
                "scripts/start-malayalam-sidecar.ps1"
            ) from exc

        if response.status_code != 200:
            detail = ""
            try:
                detail = response.json().get("detail", "")
            except Exception:
                detail = response.text[:200]
            raise MalayalamSidecarError(
                f"Malayalam sidecar returned HTTP {response.status_code}"
                + (f": {detail}" if detail else "")
            )

        payload = response.json()
        malayalam = (payload.get("malayalam_text") or "").strip()
        corrected = (payload.get("corrected_text") or "").strip() or malayalam
        corrections = payload.get("asr_corrections") or []
        english = (payload.get("english_text") or "").strip() or None

        if not malayalam:
            raise MalayalamSidecarError("Malayalam sidecar returned an empty transcript")

        if payload.get("translation_error"):
            logger.warning("Malayalam translation unavailable: %s",
                           payload["translation_error"])

        # `text` stays Malayalam: it is what was actually said, and the
        # validation layer checks it against the requested language. The English
        # rendering is carried separately so the UI can show both and never
        # present the translation as the transcript.
        return Transcript(
            text=malayalam,
            segments=[Segment(start=0.0, end=0.0, text=malayalam)],
            language="ml",
            source_language="ml",
            source_transcript=malayalam,
            target_language="en" if english else None,
            translated_transcript=english,
            provider_id=self.provider_id,
            provider_meta={
                "asr_model": payload.get("asr_model"),
                "asr_revision": payload.get("asr_revision"),
                "decoder": payload.get("decoder"),
                "device": payload.get("device"),
                "asr_seconds": payload.get("asr_seconds"),
                "translation_seconds": payload.get("translation_seconds"),
                "translation_error": payload.get("translation_error"),
                # Layer-1 verified ASR corrections: provenance + the corrected
                # copy. The raw Malayalam text above is never modified.
                "asr_corrections": corrections,
                "corrected_text": corrected,
                "sidecar_url": self.url,
                "experimental": True,
            },
        )
