"""Tests for the production English path.

Covers the guarantees the client demo depends on: English is forced rather than
detected, a non-English or hallucinated transcript is caught before it can
become a clinical note, provider selection prefers Deepgram only when it is
actually configured, and the clinical layer never invents a finding.
"""

from __future__ import annotations

import importlib
import sys

import pytest

from scribe_engine import clinical, prescription
from scribe_engine.stt import (
    CURRENT_PROVIDER_ID,
    DEEPGRAM_PROVIDER_ID,
    DeepgramSTTProvider,
    available_providers,
    resolve_production_provider,
)
from scribe_engine.validation import validate_transcript

import scribe_engine.stt as stt  # noqa: E402  (language-provider routing)

ENGLISH = (
    "Doctor: Good morning, what brings you in today? "
    "Patient: I have had a fever and a cough for three days. "
    "Doctor: Any vomiting? Patient: Yes, yesterday. "
    "Doctor: Take paracetamol five hundred milligrams twice a day."
)


# -- language forcing -------------------------------------------------------
def test_engine_defaults_to_english():
    from scribe_engine import ScribeEngine

    assert ScribeEngine().language == "en"


def test_whisper_provider_receives_forced_language(monkeypatch):
    """The engine must pass language through, never leave it None (auto-detect)."""
    from scribe_engine.stt.current_provider import CurrentSTTProvider

    seen = {}

    def fake_transcribe(audio_path, model_size="base", device="cpu",
                        compute_type="int8", language=None, cpu_threads=0):
        seen["language"] = language
        return {"segments": [], "language": "en", "duration": 1.0}

    monkeypatch.setattr(
        "scribe_engine.stt.current_provider._fw_transcribe", fake_transcribe)
    CurrentSTTProvider().transcribe("x.wav", language="en")
    assert seen["language"] == "en", "language must be forced, not auto-detected"


# -- validation: the guardrail ----------------------------------------------
def test_valid_english_passes():
    v = validate_transcript(ENGLISH, expected_language="en", reported_language="en")
    assert v.ok, v.issues


def test_malayalam_script_is_rejected():
    v = validate_transcript(
        "എനിക്ക് മൂന്ന് ദിവസമായി പനിയും ചുമയും ഉണ്ട് തലവേദയും ഉണ്ട്",
        expected_language="en", reported_language="ml")
    assert v.is_invalid
    assert "wrong_script" in v.codes or "unexpected_language" in v.codes


def test_devanagari_script_is_rejected():
    v = validate_transcript("मुझे तीन दिन से बुखार और खांसी है",
                            expected_language="en", reported_language="hi")
    assert v.is_invalid


def test_tamil_script_is_rejected():
    v = validate_transcript("எனக்கு மூன்று நாட்களாக காய்ச்சல் மற்றும் இருமல் உள்ளது",
                            expected_language="en", reported_language="ta")
    assert v.is_invalid


def test_provider_reporting_non_english_is_invalid_even_with_latin_text():
    """Whisper mislabelling English as Sinhala must not pass silently."""
    v = validate_transcript(ENGLISH, expected_language="en", reported_language="si")
    assert v.is_invalid
    assert "unexpected_language" in v.codes


def test_mixed_script_is_flagged_as_warning():
    """A mostly-English transcript with a stray foreign word must be flagged.

    Varied wording on purpose: repeating one sentence would trip the
    repetition detector instead and test the wrong thing.
    """
    english = (
        "Doctor asked what brings you in today and the patient described "
        "a persistent fever with a dry cough starting about three days ago, "
        "along with a headache in the evenings and one episode of vomiting "
        "yesterday afternoon after meals, no chest pain reported so far. "
    )
    v = validate_transcript(english + " പനി", expected_language="en",
                            reported_language="en")
    assert v.severity in ("warning", "invalid"), v.metrics
    assert "mixed_script" in v.codes or "wrong_script" in v.codes, v.codes
    assert "Malayalam" in v.metrics["scripts_detected"]


def test_empty_transcript_is_invalid():
    assert validate_transcript("", expected_language="en").is_invalid
    assert "empty_transcript" in validate_transcript("", expected_language="en").codes


def test_pathological_repetition_is_invalid():
    """The classic whisper decoder loop."""
    v = validate_transcript("thank you for watching " * 40,
                            expected_language="en", reported_language="en")
    assert v.is_invalid
    assert "pathological_repetition" in v.codes


def test_repeated_identical_segments_is_invalid():
    segs = [{"start": i, "end": i + 1, "text": "please subscribe"} for i in range(6)]
    v = validate_transcript("please subscribe " * 6, expected_language="en",
                            reported_language="en", segments=segs)
    assert v.is_invalid


def test_validation_message_is_human_readable():
    v = validate_transcript("", expected_language="en")
    assert "review" in v.message.lower()
    # Must not leak internals to a clinician.
    assert "Traceback" not in v.message and "\\" not in v.message


# -- clinical safety: nothing invented --------------------------------------
def test_note_never_invents_findings():
    structured = clinical.extract_medical_info(ENGLISH)
    note = clinical.build_clinical_note(structured, ["Take paracetamol"], ["I have a fever"])
    blob = note["text"].lower()
    for invented in ("no acute distress", "lungs", "blood test", "allergy screening",
                     "mild improvement", "adequate hydration", "diabetes"):
        assert invented not in blob, f"fabricated content: {invented}"


def test_note_impression_absent_when_no_diagnosis_stated():
    """The old fallback invented 'Possible upper respiratory condition'."""
    structured = clinical.extract_medical_info("Patient: I feel unwell today.")
    note = clinical.build_clinical_note(structured, [], ["I feel unwell today"])
    assert note["fields"]["impression"] == clinical.NOT_MENTIONED
    assert "upper respiratory" not in note["text"].lower()


def test_prescription_never_invents_dosage_or_followup():
    rx = prescription.build_prescription(["We will see how it goes"])
    assert rx["fields"]["medications"] == []
    for key in ("dosage_instructions", "duration_of_treatment", "follow_up",
                "lifestyle_advice", "dietary_guidance", "precautions"):
        assert rx["fields"][key] == prescription.NOT_MENTIONED


def test_empty_note_is_generated_for_invalid_transcript():
    note = clinical.empty_clinical_note("bad audio")
    assert note["fields"]["not_generated"] is True
    assert "no clinical note was generated" in note["text"].lower()
    for invented in ("distress", "blood test", "hydration"):
        assert invented not in note["text"].lower()


def test_empty_prescription_for_invalid_transcript():
    rx = prescription.empty_prescription("bad audio")
    assert rx["fields"]["not_generated"] is True
    assert rx["fields"]["medications"] == []


# -- provider selection ------------------------------------------------------
def test_both_providers_registered():
    assert CURRENT_PROVIDER_ID in available_providers()
    assert DEEPGRAM_PROVIDER_ID in available_providers()


def test_falls_back_to_whisper_without_deepgram_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    provider, reason = resolve_production_provider(None, model_size="base", device="cpu")
    assert provider.provider_id == CURRENT_PROVIDER_ID
    assert "fallback" in reason.lower()


def test_prefers_deepgram_when_key_is_configured(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-a-real-credential")
    provider, reason = resolve_production_provider(None, model_size="base", device="cpu")
    assert provider.provider_id == DEEPGRAM_PROVIDER_ID


def test_language_overrides_preferred_provider(monkeypatch):
    """language=ml must win over the server's preferred English provider.

    Production default ISCRIBE_STT_PROVIDER=current_faster_whisper previously
    captured ml consultations before the language map was consulted, and the
    English engine returned fluent hallucinated English for real Malayalam
    audio (benchmarks/malayalam/real_audio_asr/). Language is a requirement,
    not a preference.
    """
    monkeypatch.setenv("MALAYALAM_SIDECAR_URL", "http://127.0.0.1:1")  # unreachable
    provider = stt.get_provider(stt.MALAYALAM_PROVIDER_ID)
    monkeypatch.setattr(
        type(provider), "is_configured",
        property(lambda self: True))  # sidecar assumed up; routing is what we pin
    got, reason = resolve_production_provider(
        CURRENT_PROVIDER_ID, language="ml", model_size="base", device="cpu")
    assert got.provider_id == stt.MALAYALAM_PROVIDER_ID
    assert "language provider" in reason


def test_language_provider_down_fails_loudly_not_whisper(monkeypatch):
    """Sidecar down + language=ml => SpeechLanguageUnavailable, never whisper."""
    from scribe_engine.stt.base import SpeechLanguageUnavailable

    monkeypatch.setenv("MALAYALAM_SIDECAR_URL", "http://127.0.0.1:1")  # unreachable
    with pytest.raises(SpeechLanguageUnavailable) as exc:
        resolve_production_provider(
            CURRENT_PROVIDER_ID, language="ml", model_size="base", device="cpu")
    # Clinician-safe message: names the language and the English alternative,
    # never internal engine/sidecar terminology.
    assert "malayalam" in str(exc.value).lower()
    assert "english" in str(exc.value).lower()


def test_explicit_provider_selection_wins(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-a-real-credential")
    provider, reason = resolve_production_provider(
        CURRENT_PROVIDER_ID, model_size="base", device="cpu")
    assert provider.provider_id == CURRENT_PROVIDER_ID
    assert "explicit" in reason.lower()


def test_deepgram_without_key_raises_clear_message(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    provider = DeepgramSTTProvider()
    assert not provider.is_configured
    with pytest.raises(Exception) as exc:
        provider.transcribe("x.wav", language="en")
    assert "DEEPGRAM_API_KEY is not configured" in str(exc.value)


def test_deepgram_forces_english_and_medical_settings(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-a-real-credential")
    params = DeepgramSTTProvider()._params("en")
    assert params["language"] == "en"
    assert "detect" not in str(params).lower()
    assert params["punctuate"] == "true"
    assert params["smart_format"] == "true"
    assert params["numerals"] == "true"
    assert params["diarize"] == "true"
    assert params["profanity_filter"] == "false"
    assert "medical" in params["model"]


def test_deepgram_key_never_appears_in_provider_meta(monkeypatch):
    key = "super-secret-key-value-abc123"
    monkeypatch.setenv("DEEPGRAM_API_KEY", key)
    provider = DeepgramSTTProvider()
    payload = {
        "results": {"channels": [{"alternatives": [{"transcript": "hello there doctor"}]}]},
        "metadata": {"duration": 1.0, "request_id": "abc"},
    }
    transcript = provider._to_transcript(payload, "en", 0.5)
    assert key not in str(transcript.provider_meta)
    assert key not in str(transcript.__dict__)


# -- speaker roles ------------------------------------------------------------
def test_provider_diarization_keeps_neutral_labels():
    from scribe_engine import diarization

    turns = [
        {"speaker": "Speaker 0", "start": 0.0, "end": 1.0, "text": "what brings you in"},
        {"speaker": "Speaker 1", "start": 1.0, "end": 2.0, "text": "i have a fever"},
    ]
    out = diarization.build_labeled_transcript([], provider_turns=turns)
    assert out["method"] == "provider_diarization"
    assert out["roles_known"] is False
    # Identities must NOT be manufactured from content.
    assert {t["speaker"] for t in out["turns"]} == {"Speaker 0", "Speaker 1"}


def test_alternating_labels_are_marked_untrusted():
    from scribe_engine import diarization

    segs = [{"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "i have a fever"}]
    out = diarization.build_labeled_transcript(segs)
    assert out["roles_known"] is False
    assert out["confidence"] == "low"


def test_explicit_roles_are_trusted():
    from scribe_engine import diarization

    out = diarization.build_labeled_transcript(
        [], raw_transcript_text="Doctor: hello\nPatient: i have a fever")
    assert out["roles_known"] is True
    assert out["confidence"] == "high"
