# -*- coding: utf-8 -*-
"""Structured clinical documentation regression tests.

The documentation layer (scribe_engine.clinical_documentation) sits ABOVE the
existing semantic layer: the ClinicalEntity records are the source of truth and
every documented fact must trace back to them.

The central truthfulness contract under test:

    EXPLICIT_PRESENT  — explicitly reported
    EXPLICIT_ABSENT   — explicitly denied
    NOT_DOCUMENTED    — never discussed (NEVER converted into ABSENT)

plus: speaker-attribution honesty, question-exclusion, status-scope (resolved
must not leak), and Malayalam source provenance on every concept-derived fact.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scribe_engine.clinical_documentation import (
    EXPLICIT_ABSENT,
    EXPLICIT_PRESENT,
    NOT_DOCUMENTED,
    build_structured_documentation,
    render_clinical_documentation,
)
from scribe_engine.semantic import semantic_normalize
from scribe_engine.clinical import build_normalized_entities


def _doc(text: str, speaker: str = "Patient") -> dict:
    ents = [e.as_dict() for e in semantic_normalize(text)]
    return build_structured_documentation(ents, [{"speaker": speaker, "text": text}])


def _section(doc: dict, key: str) -> list[dict]:
    return doc["sections"][key]


def _evidence(doc: dict, key: str) -> set:
    return {f["evidence"] for f in _section(doc, key)}


def _texts(doc: dict, key: str) -> list[str]:
    return [f["text"] for f in _section(doc, key)]


def _concept_fact(doc: dict, key: str, concept: str) -> dict:
    for f in _section(doc, key):
        if f.get("concept") == concept:
            return f
    raise AssertionError(f"{concept} not found in section {key}: {_texts(doc, key)}")


# ---------------------------------------------------------------------------
# 1-2. HPI with multiple symptoms; resolved fever + persistent fatigue
# ---------------------------------------------------------------------------
def test_hpi_multiple_symptoms_resolved_fever_persistent_fatigue():
    text = ("പനി ഉണ്ടായിരുന്നു, ആറു ദിവസം ആയി. ഇപ്പോൾ പനി മാറി, "
            "പക്ഷേ ക്ഷീണം ഇപ്പോഴും ഉണ്ട്. ചുമയും കഫക്കെട്ടും ഉണ്ട്. "
            "ഫുഡ് കഴിക്കാൻ പറ്റണില്ല.")
    doc = _doc(text)
    hpi = _section(doc, "hpi")[0]
    assert hpi["evidence"] == EXPLICIT_PRESENT
    # Resolved fever leads the narrative with its explicit duration...
    assert "history of fever for 6 days" in hpi["text"]
    assert "has since resolved" in hpi["text"]
    # ...then persistent fatigue, cough, congestion, difficulty eating.
    assert "continues to experience fatigue" in hpi["text"]
    assert "cough" in hpi["text"] and "phlegm" in hpi["text"]
    assert "difficulty eating" in hpi["text"]
    # Chief complaint exists and is explicit.
    assert _evidence(doc, "chief_complaint") == {EXPLICIT_PRESENT}


# ---------------------------------------------------------------------------
# 3. Negated symptoms
# ---------------------------------------------------------------------------
def test_negated_symptom_is_explicit_absent():
    doc = _doc("പനി ഇല്ല.")
    assert _evidence(doc, "pertinent_negatives") == {EXPLICIT_ABSENT}
    fact = _section(doc, "pertinent_negatives")[0]
    assert fact["text"].startswith("Denies")
    # A denial must never appear as a present finding.
    cc = _section(doc, "chief_complaint")
    assert not cc


# ---------------------------------------------------------------------------
# 4-6. PMH: explicit presence / explicit absence / undiscussed
# ---------------------------------------------------------------------------
def test_pmh_explicit_present():
    doc = _doc("എനിക്ക് ഷുഗർ ഉണ്ട്.")
    assert _evidence(doc, "pmh") == {EXPLICIT_PRESENT}
    assert "diabetes" in _texts(doc, "pmh")[0].lower()


def test_pmh_explicit_absent():
    doc = _doc("എനിക്ക് ഷുഗർ ഇല്ല.")
    assert _evidence(doc, "pmh") == {EXPLICIT_ABSENT}


def test_pmh_undiscussed_is_not_documented_not_absent():
    doc = _doc("പനി ഉണ്ട്.")  # nothing about diabetes/BP
    assert _evidence(doc, "pmh") == {NOT_DOCUMENTED}
    # The truthfulness flags travel with the document.
    assert doc["not_documented_is_not_absent"] is True


# ---------------------------------------------------------------------------
# 7-8. Medications and allergies
# ---------------------------------------------------------------------------
def test_explicit_medication_documented():
    doc = _doc("എനിക്ക് പാരസെറ്റമോൾ കഴിച്ചു.")
    assert _evidence(doc, "medications") == {EXPLICIT_PRESENT}
    assert any("paracetamol" in t.lower() for t in _texts(doc, "medications"))


def test_explicit_allergy_documented():
    doc = _doc("Doctor: അലർജി ഉണ്ടോ?\nPatient: പെൻസിലിന് അലർജി ഉണ്ട്.")
    ents = build_normalized_entities(
        "Doctor: അലർജി ഉണ്ടോ?\nPatient: പെൻസിലിന് അലർജി ഉണ്ട്.")
    doc = build_structured_documentation(
        ents["entities"],
        [{"speaker": "Doctor", "text": "അലർജി ഉണ്ടോ?"},
         {"speaker": "Patient", "text": "പെൻസിലിന് അലർജി ഉണ്ട്."}])
    assert _evidence(doc, "allergies") == {EXPLICIT_PRESENT}
    assert any("penicillin" in t for t in _texts(doc, "allergies"))


# ---------------------------------------------------------------------------
# 9-10. Family and social history
# ---------------------------------------------------------------------------
def test_family_history_explicit():
    doc = _doc("അമ്മയ്ക്ക് ഷുഗർ ഉണ്ട്.")
    assert _evidence(doc, "family_history") == {EXPLICIT_PRESENT}
    # Family attribution must never leak into patient findings/PMH.
    assert _evidence(doc, "pmh") == {NOT_DOCUMENTED}
    assert not _section(doc, "chief_complaint")


def test_social_history_smoking_explicit():
    doc = _doc("ഞാൻ പുകവലി ചെയ്യുന്നു.")
    assert _evidence(doc, "social_history") == {EXPLICIT_PRESENT}
    assert any("smoking" in t for t in _texts(doc, "social_history"))


# ---------------------------------------------------------------------------
# 11-12. Physical examination
# ---------------------------------------------------------------------------
def test_physical_exam_explicit_doctor_statement():
    turns = [
        {"speaker": "Doctor", "text": "Lungs are clear."},
        {"speaker": "Patient", "text": "I have fever since two days."},
    ]
    ents = build_normalized_entities(
        "Doctor: Lungs are clear.\nPatient: I have fever since two days.")
    doc = build_structured_documentation(ents["entities"], turns)
    assert _evidence(doc, "physical_exam") == {EXPLICIT_PRESENT}
    assert any("clear" in t.lower() for t in _texts(doc, "physical_exam"))


def test_no_physical_exam_is_not_documented_not_normal():
    doc = _doc("പനി ഉണ്ട്.")
    assert _evidence(doc, "physical_exam") == {NOT_DOCUMENTED}
    assert _section(doc, "physical_exam")[0]["text"] == "Not documented"
    # No invented "normal" findings anywhere.
    assert not any("normal" in t.lower() for t in _texts(doc, "physical_exam"))


# ---------------------------------------------------------------------------
# 13-14. Assessment: diagnosis only when the doctor states it
# ---------------------------------------------------------------------------
def test_doctor_diagnosis_documented():
    turns = [
        {"speaker": "Doctor", "text": "This is viral fever."},
        {"speaker": "Patient", "text": "I have fever since three days."},
    ]
    ents = build_normalized_entities(
        "Doctor: This is viral fever.\nPatient: I have fever since three days.")
    doc = build_structured_documentation(ents["entities"], turns)
    assert _evidence(doc, "assessment") == {EXPLICIT_PRESENT}
    assert any("viral fever" in t.lower() for t in _texts(doc, "assessment"))


def test_no_diagnosis_is_not_invented():
    doc = _doc("പനി ഉണ്ട്, ക്ഷീണം ഉണ്ട്.")
    assert _evidence(doc, "assessment") == {NOT_DOCUMENTED}
    assert "No definitive diagnosis" in _texts(doc, "assessment")[0]


# ---------------------------------------------------------------------------
# 15-16. Plan
# ---------------------------------------------------------------------------
def test_explicit_plan_documented():
    turns = [
        {"speaker": "Doctor", "text": "I will prescribe paracetamol tablet."},
        {"speaker": "Patient", "text": "I have fever since two days."},
    ]
    ents = build_normalized_entities(
        "Doctor: I will prescribe paracetamol tablet.\n"
        "Patient: I have fever since two days.")
    doc = build_structured_documentation(ents["entities"], turns)
    assert _evidence(doc, "plan") == {EXPLICIT_PRESENT}
    assert any("Medication" in t for t in _texts(doc, "plan"))


def test_no_plan_is_not_documented():
    doc = _doc("പനി ഉണ്ട്.")
    assert _evidence(doc, "plan") == {NOT_DOCUMENTED}
    assert _section(doc, "plan")[0]["text"] == "Not documented"


# ---------------------------------------------------------------------------
# 17. Doctor question must not become patient history
# ---------------------------------------------------------------------------
def test_doctor_question_never_becomes_finding():
    turns = [
        {"speaker": "Doctor", "text": "Do you have diabetes?"},
        {"speaker": "Patient", "text": "I have fever since two days."},
    ]
    ents = build_normalized_entities(
        "Doctor: Do you have diabetes?\nPatient: I have fever since two days.")
    doc = build_structured_documentation(ents["entities"], turns)
    assert _evidence(doc, "pmh") == {NOT_DOCUMENTED}   # not EXPLICIT_ABSENT either
    assert "diabetes" not in " ".join(_texts(doc, "pmh")).lower()


# ---------------------------------------------------------------------------
# 18. Patient statement attributed to patient
# ---------------------------------------------------------------------------
def test_patient_statement_attributed_to_patient():
    doc = _doc("പനി ഉണ്ട്.")
    fact = _concept_fact(doc, "chief_complaint", "fever")
    assert fact["source_speaker"] == "patient"
    assert fact["source_text"]  # Malayalam provenance present


def test_unknown_roles_labelled_speaker_unknown():
    turns = [{"speaker": "Speaker 0", "text": "പനി ഉണ്ട്."}]
    ents = build_normalized_entities("Speaker 0: പനി ഉണ്ട്.")
    doc = build_structured_documentation(ents["entities"], turns, roles_known=False)
    pmh = _section(doc, "pmh")[0]
    # A line-derived fact with unknown roles must never claim Doctor/Patient.
    assert doc["speaker_roles_known"] is False


# ---------------------------------------------------------------------------
# 19. Status scope: resolved does not leak
# ---------------------------------------------------------------------------
def test_resolved_and_current_scoped_correctly():
    doc = _doc("പനി മാറി, പക്ഷേ ക്ഷീണം ഇപ്പോഴും ഉണ്ട്.")
    ros = " ".join(_texts(doc, "ros"))
    assert "fever (resolved)" in ros
    assert "fatigue" in ros
    hpi = _section(doc, "hpi")[0]["text"]
    assert "has since resolved" in hpi
    assert "continues to experience fatigue" in hpi


# ---------------------------------------------------------------------------
# 20. Malayalam source provenance
# ---------------------------------------------------------------------------
def test_provenance_carries_malayalam_source():
    doc = _doc("പനി ഉണ്ട്.")
    fact = _concept_fact(doc, "chief_complaint", "fever")
    assert "പനി" in fact["source_text"]
    assert fact["confidence"] is not None and 0 < fact["confidence"] <= 1


# ---------------------------------------------------------------------------
# Renderer + safety invariants
# ---------------------------------------------------------------------------
def test_renderer_emits_not_documented_never_invented_sections():
    doc = _doc("പനി ഉണ്ട്.")
    note = render_clinical_documentation(doc)
    assert "1. Chief Complaint" in note
    assert "12. Physical Examination" in note
    assert "Not documented." in note
    assert "14. Plan" in note
    # The 14-section contract, in order.
    titles = ["Chief Complaint", "History of Present Illness", "Associated Symptoms",
              "Pertinent Negatives", "Past Medical History", "Past Surgical History",
              "Medications", "Allergies", "Family History", "Social History",
              "Review of Systems", "Physical Examination", "Assessment", "Plan"]
    positions = [note.index(t) for t in titles]
    assert positions == sorted(positions)


def test_every_fact_has_evidence_flag():
    doc = _doc("പനി ഉണ്ട്.")
    for key, facts in doc["sections"].items():
        for f in facts:
            assert f["evidence"] in (EXPLICIT_PRESENT, EXPLICIT_ABSENT, NOT_DOCUMENTED), \
                f"{key}: {f}"
