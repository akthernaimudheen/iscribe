"""Unit tests for scribe_engine (no model downloads required).

Upstream-derived behavior is asserted as-is where it is quirky on purpose:
those assertions document upstream defects we intentionally carry over
(see ANALYSIS.md section 4) until the engine-improvement phase.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scribe_engine import audio as audio_mod
from scribe_engine import clinical, diarization, prescription

FIXTURE_CONVO = """
Doctor: Hello, good morning! How can I help you today?
Patient: Good morning, doctor. I'm Ramesh. I've been feeling some chest pain and shortness of breath.
Doctor: When did it start?
Patient: I think it started around 5 days ago.
Doctor: Have you experienced any other symptoms?
Patient: I've also had fatigue and a bit of nausea.
Doctor: Based on what you're telling me, it looks like angina. How old are you?
Patient: I am 45 years old.
Doctor: Got it. We'll run a few more tests to confirm.
"""


def test_extract_medical_info_matches_upstream_sample():
    """Upstream nlp.py __main__ block fixture — same input, must give same key fields."""
    r = clinical.extract_medical_info(FIXTURE_CONVO)
    assert r["patient_name"] == "Ramesh"
    assert r["age"] == 45
    assert "chest pain" in r["symptoms"]
    assert "shortness of breath" in r["symptoms"]
    assert r["diagnoses"] and r["diagnoses"][0] == "angina"
    # Documents upstream defect ANALYSIS.md §4.1: the duration regex matches
    # "around 5" from "started around 5 days ago" — upstream nlp.py behavior,
    # intentionally preserved until the engine-improvement phase.
    assert r["duration"] == "around 5"


def test_extract_medical_info_empty_input():
    r = clinical.extract_medical_info("")
    assert r["symptoms"] == []
    assert "Invalid or empty input" in r["warnings"]


def test_keyword_and_medication_scans():
    patient = "I have asthma and a runny nose"
    doctor = "You can take allegra and use a nasal spray twice a day"
    assert "asthma" in clinical.scan_symptom_keywords(patient)
    assert "allegra" in clinical.scan_medications(doctor)


def test_note_generation_flags_templates():
    structured = clinical.extract_medical_info(FIXTURE_CONVO)
    doctor_lines = ["It looks like angina, we will run tests"]
    patient_lines = ["I have chest pain"]
    note = clinical.build_clinical_note(structured, doctor_lines, patient_lines)
    assert "angina" in note["fields"]["impression"].lower()
    assert "chest pain" in note["fields"]["symptoms_reported"]

    # CLINICAL SAFETY: fields nobody spoke about must say so, not carry
    # plausible boilerplate. This previously asserted the presence of
    # "Doctor noted patient condition as stable and no acute distress.
    # [template]" — a fabricated examination finding that reached exports.
    for absent in ("physical_findings", "investigations",
                   "treatment_response", "additional_remarks"):
        assert note["fields"][absent] == clinical.NOT_MENTIONED, absent

    blob = note["text"].lower()
    for invented in ("acute distress", "blood test", "allergy screening",
                     "mild improvement", "adequate hydration", "[template]"):
        assert invented not in blob, f"note fabricated: {invented}"


def test_prescription_generation():
    doctor_lines = [
        "I will prescribe allegra for you",
        "Take one tablet daily after food",
    ]
    rx = prescription.build_prescription(doctor_lines)
    names = [m["name"] for m in rx["fields"]["medications"]]
    assert "allegra" in names
    assert any("allegra" in i.lower() for m in rx["fields"]["medications"] for i in m["instructions"])

    # CLINICAL SAFETY: a dosage, duration or follow-up interval nobody stated
    # must never be supplied. This previously asserted the presence of
    # "Review patient response in 1-2 weeks ... [template]" — a follow-up
    # interval no clinician prescribed.
    for absent in ("follow_up", "duration_of_treatment", "dosage_instructions",
                   "lifestyle_advice", "dietary_guidance", "precautions"):
        assert rx["fields"][absent] == prescription.NOT_MENTIONED, absent

    blob = rx["text"].lower()
    for invented in ("5-7 days", "1-2 weeks", "maintain hydration",
                     "warm fluids", "[template]"):
        assert invented not in blob, f"prescription fabricated: {invented}"


def test_explicit_roles_highest_priority():
    labeled = diarization.build_labeled_transcript([], raw_transcript_text=FIXTURE_CONVO)
    assert labeled["method"] == "explicit_roles"
    assert labeled["confidence"] == "high"
    assert labeled["turns"][0]["speaker"] == "Doctor"
    assert labeled["turns"][1]["speaker"] == "Patient"


def test_alternating_flips_when_patient_opens():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "I have been having really bad allergies lately"},
        {"start": 2.0, "end": 4.0, "text": "I see, how long have you had these symptoms?"},
    ]
    labeled = diarization.build_labeled_transcript(segs)
    assert labeled["method"] == "alternating"
    assert labeled["turns"][0]["speaker"] == "Patient"
    assert labeled["turns"][1]["speaker"] == "Doctor"


def test_alternating_keeps_doctor_open():
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Good morning, I am Dr. Rao, how can I help you today?"},
        {"start": 2.0, "end": 4.0, "text": "I have a sore throat and fever"},
    ]
    labeled = diarization.build_labeled_transcript(segs)
    assert labeled["turns"][0]["speaker"] == "Doctor"
    assert labeled["turns"][1]["speaker"] == "Patient"


def test_audio_validation():
    try:
        audio_mod.validate_audio("does_not_exist.mp3")
        assert False, "should have raised"
    except audio_mod.AudioError:
        pass
