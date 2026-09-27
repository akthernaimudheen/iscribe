"""MalayalamIndicProvider — experimental STTProvider (Pipeline B).

Implements the same interface as CurrentSTTProvider so the benchmark can run
both pipelines interchangeably. Language contract (per task spec):

    audio → Malayalam transcript (IndicConformer) → English (IndicTrans2)

The Transcript preserves BOTH languages:
    source_transcript       = Malayalam text (never discarded)
    translated_transcript   = English rendering
    text                    = English (the working text for downstream NLP)

NOT registered in the provider registry — integration into the demo flow is a
separate, explicit decision after benchmark review.
"""

from ....stt.base import STTProvider, Transcript
from .asr import MODEL_ID as ASR_MODEL_ID, transcribe_malayalam
from .translation import MODEL_ID as MT_MODEL_ID, translate_ml_to_en


class MalayalamIndicProvider(STTProvider):
    provider_id = "experimental_indic_malayalam"

    def __init__(self, device: str = "cpu"):
        self.device = device

    def transcribe(self, audio_path, language=None) -> Transcript:
        # NOTE: language parameter intentionally ignored — this pipeline is
        # Malayalam-specific by design. Mixed-language behavior is measured in
        # the benchmark, not assumed.
        asr = transcribe_malayalam(audio_path, device=self.device)
        mt = translate_ml_to_en(asr["malayalam_text"] or " ", device=self.device)
        return Transcript(
            text=mt["english_text"],
            segments=[],
            language="ml",
            source_language="ml",
            source_transcript=asr["malayalam_text"],
            target_language="en",
            translated_transcript=mt["english_text"],
            provider_id=self.provider_id,
            provider_meta={
                "asr_model": asr["model"],
                "mt_model": mt["model"],
                "asr_time_seconds": asr["asr_time_seconds"],
                "translation_time_seconds": mt["translation_time_seconds"],
                "device": self.device,
            },
        )


__all__ = ["MalayalamIndicProvider", "ASR_MODEL_ID", "MT_MODEL_ID"]
