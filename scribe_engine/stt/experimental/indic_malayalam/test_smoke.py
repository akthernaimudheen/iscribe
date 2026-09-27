"""Deterministic smoke test — one short non-PHI fixture, no exact-text assertions.

Verifies: audio → Malayalam ASR → transcript exists → translation exists →
English non-empty. Detects catastrophic failure only; it does NOT measure
clinical accuracy and does not assert exact wording (model/runtime differences
can legitimately change wording).

Run from the isolated experiment environment:
    python -m pytest scribe_engine/stt/experimental/indic_malayalam/test_smoke.py -q
"""

import os
import pathlib

import pytest

FIXTURE = pathlib.Path(__file__).parent / "test_audio" / "ml_fever_cough.wav"


def _experiment_env() -> bool:
    """True when the isolated experiment venv (NeMo + pytorch_lightning) is present.

    This smoke test loads real ASR/translation models and is documented to run
    from the isolated experiment environment, not the lightweight demo venv.
    Skipping there keeps the main suite deterministic.
    """
    try:
        import pytorch_lightning  # noqa: F401
    except ImportError:
        return False
    try:
        import nemo.collections.asr  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _experiment_env(),
    reason="requires the isolated experiment venv (NeMo + pytorch_lightning); demo venv lacks the ASR stack",
)


def test_smoke_malayalam_end_to_end():
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    from scribe_engine.stt.experimental.indic_malayalam.asr import transcribe_malayalam

    asr = transcribe_malayalam(FIXTURE, device="cpu")
    assert "errors" not in asr
    ml = (asr.get("malayalam_text") or "").strip()
    assert ml, "Malayalam transcript is empty — ASR failed"

    from scribe_engine.stt.experimental.indic_malayalam.translation import translate_ml_to_en

    mt = translate_ml_to_en(ml, device="cpu")
    en = (mt.get("english_text") or "").strip()
    assert en, "English translation is empty — translation failed"

    # Both languages preserved for traceability
    print("ML :", ml[:80])
    print("EN :", en[:80])
