"""Regression: structured review fields are a PROJECTION of the fact graph.

Background (the production bug): the canonical Clinical Note was rendered from
the typed fact graph (clinical_facts V2), but the editable Review-page fields
were produced by a SEPARATE legacy keyword/regex extractor. The two diverged:
the fact graph held the full clinical picture while the fields showed
"symptoms: pain", "duration: buying tight", "physical findings: Not mentioned",
"diagnoses: []" and a false "No diagnosis detected" warning.

Architecture now (single source): transcript -> fact graph -> canonical note
AND deterministic field projection (field_projection.project_structured_fields).
These tests pin that contract on the referral-letter scenario that failed in
production. Synthetic data only — no real PHI in the repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scribe_engine.clinical import render_clinical_note_text
from scribe_engine.clinical_facts import build_clinical_facts
from scribe_engine.field_projection import (NOT_DOCUMENTED, NOT_MENTIONED,
                                            project_structured_fields,
                                            warnings_from_facts)
from scribe_engine.note_v2 import render_clinical_note
from scribe_engine.pipeline import ScribeEngine

REPO_ROOT = Path(__file__).resolve().parents[1]

# Synthetic referral dictation with the same clinical shape as the production
# case (values are fictitious).
REFERRAL_TEXT = """Referral letter. Re: Mr A Sample.
Thank you for referring this gentleman regarding pain in the left medial foot arch.
The pain in the left medial foot arch began after buying tight footwear.
The pain has been fluctuating since March.
No benefit was obtained from physiotherapy.
The pain worsens after resuming exercise.
On examination there is no bony tenderness.
There is tenderness over the left medial arch.
I suspect plantar fasciitis and the clinician considered this.
A steroid injection would benefit from consideration.
I would be grateful if you could refer him to a consultant orthopaedic surgeon for a specialist opinion.
Solicitors reference: ZZ / QQ. Date of birth: 01/23/1945. Address: 12, Sample Lane, Testtown TT10 1AA. Telephone: 07700900099.
Yours sincerely, Dr B Reference."""


@pytest.fixture(scope="module")
def projection() -> dict:
    doc = build_clinical_facts(REFERRAL_TEXT)
    return project_structured_fields(doc)


# ---------------------------------------------------------------------------
# 1. The required field content (the exact failures seen in production)
# ---------------------------------------------------------------------------

def test_symptom_carries_anatomical_location(projection):
    symptoms = projection["clinical_note_fields"]["symptoms_reported"].lower()
    assert "pain" in symptoms
    assert "left medial foot arch" in symptoms


def test_trigger_is_not_in_duration(projection):
    fields = projection["clinical_note_fields"]
    assert "tight footwear" in fields["onset_trigger"].lower()
    assert "footwear" not in fields["duration"].lower()
    # The old extractor's failure, pinned: duration must never hold a trigger.
    assert fields["duration"].lower() != "buying tight"
    assert NOT_MENTIONED not in fields["duration"]  # real value or Not documented


def test_temporal_course_fluctuating_since_march(projection):
    course = projection["clinical_note_fields"]["temporal_course"].lower()
    assert "fluctuating" in course and "march" in course


def test_failed_physiotherapy_in_treatment_response(projection):
    response = projection["clinical_note_fields"]["treatment_response"].lower()
    assert "no benefit" in response and "physiotherapy" in response


def test_exercise_as_aggravating_factor(projection):
    aggravating = projection["clinical_note_fields"]["aggravating_factors"].lower()
    assert "worsens" in aggravating and "exercise" in aggravating


def test_physical_findings_from_exam_facts(projection):
    findings = projection["clinical_note_fields"]["physical_findings"].lower()
    assert "no bony tenderness" in findings
    assert "left medial arch" in findings and "tenderness" in findings
    # The exam absence must NOT leak into the patient-reported denies field.
    denies = projection["clinical_note_fields"]["denies"].lower()
    assert "bony" not in denies


def test_suspected_diagnosis_present_and_unconfirmed(projection):
    fields = projection["clinical_note_fields"]
    diagnoses = fields["diagnoses"]
    assert diagnoses, "diagnoses must not be an empty array"
    names = [d["name"].lower() for d in diagnoses]
    assert any("plantar fasciitis" in n for n in names)
    suspected = [d for d in diagnoses
                 if "plantar fasciitis" in d["name"].lower()][0]
    assert suspected["status"] == "suspected"
    assert suspected["confirmed"] is False
    # The impression keeps the uncertainty wording.
    assert "suspected" in fields["impression"].lower()


def test_plan_considered_not_performed_and_referral(projection):
    plan = projection["clinical_note_fields"]["plan"].lower()
    assert "steroid injection - considered, not performed" in plan
    assert "specialist opinion - recommended" in plan


def test_referral_recipient_role(projection):
    referral = projection["clinical_note_fields"]["referral_context"].lower()
    assert "orthopaedic surgeon" in referral


def test_other_documentation_metadata(projection):
    other = projection["clinical_note_fields"]["other_documentation"].lower()
    assert "solicitors reference" in other
    assert "01/23/1945" in other
    assert "testtown" in other
    assert "07700900099" in other


# ---------------------------------------------------------------------------
# 2. Warnings and confidence semantics
# ---------------------------------------------------------------------------

def test_warning_suspected_satisfies_diagnosis_present(projection):
    warnings = projection["warnings"]
    assert not any("no diagnosis" in w.lower() for w in warnings)
    assert any("suspected" in w.lower() for w in warnings)


def test_no_diagnosis_warning_when_truly_absent():
    doc = build_clinical_facts(
        "Doctor: Hello.\nPatient: I have a cough.\nDoctor: Any fever? "
        "No? Okay, drink water.")
    warnings = warnings_from_facts(doc)
    assert any("no diagnosis" in w.lower() for w in warnings)


def test_confidence_is_not_fabricated(projection):
    assert projection["confidence"] is None
    assert "unavailable" in projection["confidence_note"].lower()
    # Per-fact confidence with real provenance still lives in the graph.
    doc = build_clinical_facts(REFERRAL_TEXT)
    confidences = [f.get("confidence") for f in doc["facts"]]
    assert any(c is not None and 0 < c <= 1 for c in confidences)


# ---------------------------------------------------------------------------
# 3. Canonical note and fields must not diverge
# ---------------------------------------------------------------------------

def test_fields_match_canonical_note_sections(projection):
    doc = build_clinical_facts(REFERRAL_TEXT)
    rendered = render_clinical_note(doc)
    fields = projection["clinical_note_fields"]
    # Every canonical section's key content appears in the fields projection.
    assert "No bony tenderness." in rendered["note"]
    assert "Left medial arch: tenderness." in rendered["note"]
    assert "Suspected (not confirmed): plantar fasciitis" in rendered["note"]
    assert "considered, not performed" in rendered["note"]
    for required in (fields["physical_findings"], fields["impression"],
                     fields["plan"]):
        assert required.split(";")[0].strip() in rendered["note"] or \
            required in rendered["note"]
    # And the rendered document text (what the export carries after review)
    # round-trips the fields identically.
    text = render_clinical_note_text(fields)
    assert "Suspected (not confirmed): plantar fasciitis" in text
    assert "considered, not performed" in text
    assert "orthopaedic surgeon" in text


def test_diagnosis_name_does_not_carry_the_certainty_cue():
    doc = build_clinical_facts("Assessment: I suspect plantar fasciitis.")
    diagnoses = [f for f in doc["facts"] if f["fact_type"] == "DIAGNOSIS"]
    assert diagnoses and diagnoses[0]["english"] == "plantar fasciitis"
    assert diagnoses[0]["certainty"] == "SUSPECTED"


def test_metadata_signature_not_stitched():
    doc = build_clinical_facts(REFERRAL_TEXT)
    telephone = doc["metadata"]["telephone"]["value"]
    assert "sincerely" not in telephone.lower()
    assert "Dr" not in telephone


# ---------------------------------------------------------------------------
# 4. End-to-end through the pipeline (both flows project from the fact graph)
# ---------------------------------------------------------------------------

def test_pipeline_text_flow_projects_fields():
    engine = ScribeEngine.__new__(ScribeEngine)
    engine.language = "en"
    engine.stt_provider_id = None
    engine.stt_policy = None
    engine.stt_fail_closed = False
    result = engine.process_text(REFERRAL_TEXT, language="en")
    note = result["clinical_note"]
    assert note.get("fields_source") == "clinical_facts_v2"
    fields = note["fields"]
    assert "left medial foot arch" in fields["symptoms_reported"].lower()
    assert "tight footwear" in fields["onset_trigger"].lower()
    assert fields["duration"].lower() != "buying tight"
    assert fields["diagnoses"]
    assert any(d["status"] == "suspected" for d in fields["diagnoses"])
    assert not any("no diagnosis" in w.lower() for w in note["warnings"])
    assert note["confidence"] is None


def test_projection_docstring_honesty():
    """The module must keep its no-guessing contract documented."""
    import scribe_engine.field_projection as fp
    doc = fp.__doc__ or ""
    assert "not documented" in doc.lower()
