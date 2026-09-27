# -*- coding: utf-8 -*-
"""Clinical Intelligence V2 regression tests — the real documents.

Fixtures are real transcripts captured by the production pipeline
(Deepgram nova-3-medical, English forced) plus the verified Malayalam
consultation, frozen under tests/fixtures/:

    english_real_cases/referral_letter_orthopaedic.txt  — orthopaedic referral
        ("Mr Jones ... left medial foot arch pain ... I wonder if this is
          unresolved plantar fasciitis and whether he would benefit from a
          steroid injection.")
    english_real_cases/medicolegal_jones.txt            — medico-legal report
    english_real_cases/medicolegal_finton.txt           — medico-legal report
    malayalam_real_consultation.txt                     — real Malayalam visit
        (fever RESOLVED; fatigue/headache/cough/phlegm/difficulty eating)

What these tests lock down (each one failed on the pre-V2 pipeline):

    * no hallucinated medication ("paracetamol" out of "New para.")
    * treatment consideration is not a symptom or current medication
    * suspected diagnosis never becomes confirmed
    * greeting / administrative text never becomes an assessment
    * examination findings reach Physical Examination
    * an explicit "no past medical history of note" is documented PMH
    * plan statements (considered treatment, expert-opinion request, explicit
      absence of further recommendations) reach Plan
    * a resolved episode and a current residual problem stay separate facts
    * HPI keeps duration, trajectory and anatomical location
    * every clinical fact carries verbatim, in-transcript evidence
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scribe_engine import asr_correction                                    # noqa: E402
from scribe_engine import clinical_facts as CF                               # noqa: E402
from scribe_engine.clinical_facts import (CERTAINTIES, FACT_TYPES, SECTIONS,  # noqa: E402
                                          STATUSES, TEMPORAL_CONTEXTS,
                                          build_clinical_facts,
                                          validate_clinical_facts)
from scribe_engine.note_v2 import render_clinical_note, validate_note_v2     # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
REAL = FIXTURES / "english_real_cases"

# The frozen fixtures are REAL consultation content (Malayalam visit transcript,
# medico-legal dictations with names/DOB/phone). They are deliberately NOT in
# version control (see .gitignore). Tests that need them skip cleanly on a
# repository clone; the recording owner's machine runs the full suite.

def _read(path: Path) -> str:
    if not path.exists():
        pytest.skip(f"real fixture not present outside git: {path.name}")
    return path.read_text(encoding="utf-8")


def _facts(doc, *, fact_type=None, status=None, contains=None):
    out = []
    for f in doc["facts"]:
        if fact_type and f["fact_type"] != fact_type:
            continue
        if status and f["status"] != status:
            continue
        if contains and contains.lower() not in (f["english"] or "").lower():
            continue
        out.append(f)
    return out


def _section(note: str, heading: str) -> str:
    """The text of one section of the rendered note, matched by title (the
    number prefix shifts between document types, so it is not hardcoded)."""
    lines = note.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(rf"^\s*\d+\.\s+{re.escape(heading)}\s*$", line.strip(), re.I):
            start = i + 1
            break
    if start is None:
        return ""
    out = []
    for line in lines[start:]:
        if re.match(r"^\s*\d+\.\s+\S", line):
            break
        out.append(line)
    return "\n".join(out)


@pytest.fixture(scope="module")
def referral_text() -> str:
    return _read(REAL / "referral_letter_orthopaedic.txt")


@pytest.fixture(scope="module")
def referral(referral_text):
    return build_clinical_facts(referral_text)


@pytest.fixture(scope="module")
def referral_note(referral):
    return render_clinical_note(referral)


@pytest.fixture(scope="module")
def jones():
    return build_clinical_facts(_read(REAL / "medicolegal_jones.txt"))


@pytest.fixture(scope="module")
def jones_note(jones):
    return render_clinical_note(jones)


@pytest.fixture(scope="module")
def finton():
    return build_clinical_facts(_read(REAL / "medicolegal_finton.txt"))


@pytest.fixture(scope="module")
def finton_note(finton):
    return render_clinical_note(finton)


@pytest.fixture(scope="module")
def malayalam():
    text = _read(FIXTURES / "malayalam_real_consultation.txt").strip()
    corrected = asr_correction.apply_corrections(text).corrected_text
    return text, build_clinical_facts(text, [], roles_known=False,
                                      corrected_text=corrected)


# ---------------------------------------------------------------------------
# 1. Evidence grounding: no fact without a verbatim source, no hallucination
# ---------------------------------------------------------------------------
class TestEvidenceGrounding:
    def test_every_fact_has_verbatim_evidence(self, referral, jones, finton):
        for doc in (referral, jones, finton):
            evidence = doc["evidence_text"]
            assert doc["facts"], "no facts extracted"
            for f in doc["facts"]:
                assert f["source_span"] and f["source_span"][0] is not None, f
                start, end = f["source_span"]
                assert f["source_text"] in evidence[start:end]
                assert f["evidence_text"]
                assert f["speaker"]
                assert f["confidence"] is not None
                assert f["certainty"] in CERTAINTIES
                assert f["status"] in STATUSES
                assert f["temporal_context"] in TEMPORAL_CONTEXTS

    def test_fact_types_and_sections_are_a_closed_vocabulary(self, referral, jones,
                                                             finton, malayalam):
        for doc in (referral, jones, finton, malayalam[1]):
            for f in doc["facts"]:
                assert f["fact_type"] in FACT_TYPES, f
                assert f["section"] in SECTIONS, f

    def test_dictated_markup_is_not_clinical_evidence(self, referral):
        for f in referral["facts"]:
            assert "New para" not in f["source_text"]
            assert "full stop" not in (f["english"] or "")

    def test_markup_command_never_yields_a_medication_even_in_a_compound_clause(
            self, referral, referral_note):
        """"New para." matched the Malayalam shorthand "para" through the shared
        concept alias table; in real dictated audio the command rides in a clause
        that also carries clinical content ("… on his left foot, full stop.
        New para."). The alias inside the markup span must be dropped while the
        rest of the clause still yields facts."""
        text = ("On examination there is no bony tenderness. He is tender over "
                "the midpoint of his medial arch on his left foot, full stop. "
                "New para. I wonder if this is unresolved plantar fasciitis.")
        doc = build_clinical_facts(text)
        meds = _facts(doc, fact_type="MEDICATION")
        assert not meds
        assert not _facts(doc, contains="paracetamol")
        assert _facts(doc, fact_type="EXAM_FINDING"), "exam findings lost"
        assert _facts(doc, fact_type="DIAGNOSIS"), "suspected diagnosis lost"
        assert "paracetamol" not in render_clinical_note(doc)["note"].lower()

    def test_speaker_labels_are_not_metadata_content(self):
        """Anonymous-upload turns carry "Speaker 0:" prefixes; a metadata value
        joined across turns must not keep them."""
        text = ("Address, 1, The House, 2, The Road, Speaker 0: Liverpool L 13. "
                "Date of birth, Speaker 0: 01/23/1945. Telephone number "
                "Speaker 0: 01632 960123.")
        doc = build_clinical_facts(text)
        assert doc["metadata"]["address"]["value"] == \
            "1, The House, 2, The Road, Liverpool L 13"
        assert doc["metadata"]["date_of_birth"]["value"] == "01/23/1945"
        assert doc["metadata"]["telephone"]["value"] == "01632 960123"
        for v in doc["metadata"].values():
            assert "speaker" not in str(v["value"]).lower()

    def test_absent_medication_is_never_created(self, referral, referral_note):
        """"New para." matched the Malayalam shorthand alias "para" before V2."""
        assert not _facts(referral, contains="paracetamol")
        assert "paracetamol" not in referral_note["note"].lower()

    def test_validator_rejects_a_fact_without_evidence(self, referral):
        forged = dict(referral["facts"][0])
        forged.update({"fact_type": "MEDICATION", "english": "paracetamol",
                       "status": "RESOLVED", "source_text": "",
                       "evidence_text": "para", "source_span": None})
        result = validate_clinical_facts(referral["evidence_text"],
                                        [forged])
        assert not result["valid"]
        assert any("paracetamol" in v for v in result["violations"])

    def test_validator_rejects_a_span_that_does_not_contain_the_claim(self, referral):
        forged = dict(referral["facts"][0])
        forged.update({"english": "paracetamol", "fact_type": "MEDICATION",
                       "source_text": "paracetamol 500 mg twice daily"})
        result = validate_clinical_facts(referral["evidence_text"], [forged])
        assert not result["valid"]

    def test_raw_transcript_is_never_modified(self, malayalam):
        raw, doc = malayalam
        assert doc["raw_text"].strip() == raw
        assert doc["asr_corrected"] is True
        assert doc["evidence_text"] != raw       # the corrected copy is separate


# ---------------------------------------------------------------------------
# 2. Fact types: treatment is never a symptom, certainty is real
# ---------------------------------------------------------------------------
class TestFactTypesAndCertainty:
    def test_injection_is_a_considered_procedure_not_a_symptom(self, referral):
        procedures = [f for f in _facts(referral, fact_type="PROCEDURE")
                      if "injection" in f["english"].lower()]
        assert procedures, "steroid injection was not extracted"
        p = procedures[0]
        assert p["status"] == "CONSIDERED"
        assert p["certainty"] == "CONSIDERED"
        assert p["section"] == "plan"
        assert not _facts(referral, fact_type="SYMPTOM", contains="injection")
        assert not _facts(referral, fact_type="MEDICATION", contains="injection")

    def test_treatment_consideration_reaches_the_plan(self, referral_note):
        plan = _section(referral_note["note"], "Plan")
        assert "injection" in plan.lower()
        assert "considered" in plan.lower()

    def test_physiotherapy_is_treatment_history_not_a_symptom(self, referral):
        treatments = _facts(referral, fact_type="TREATMENT")
        assert any("physiotherapy" in f["english"].lower() for f in treatments)
        assert not _facts(referral, fact_type="SYMPTOM", contains="physiotherapy")
        no_benefit = [f for f in treatments if
                      (f["attributes"] or {}).get("benefit") == "NONE"]
        assert no_benefit, "no-benefit response was not recorded against the treatment"

    def test_suspected_diagnosis_stays_suspected(self, referral, referral_note):
        dx = _facts(referral, fact_type="DIAGNOSIS")
        assert dx and all(f["certainty"] == "SUSPECTED" for f in dx)
        assert any("plantar fasciitis" in f["english"].lower() for f in dx)
        assessment = _section(referral_note["note"], "Assessment")
        assert "plantar fasciitis" in assessment.lower()
        # the renderer states the uncertainty explicitly rather than implying it
        assert "suspected" in assessment.lower()
        assert "not confirmed" in assessment.lower()
        assert "diagnosis:" not in assessment.lower()

    def test_confirmed_diagnosis_requires_a_clinician_statement(self, jones):
        dx = _facts(jones, fact_type="DIAGNOSIS")
        names = " ".join(f["english"].lower() for f in dx)
        assert "whiplash" in names
        for f in dx:
            if f["certainty"] == "CONFIRMED":
                assert f["certainty_evidence"], f

    def test_greeting_can_never_become_an_assessment(self, referral, referral_note):
        assert "whitfield" not in referral_note["note"].lower()
        assert "hi. this is" not in referral_note["note"].lower()
        for f in _facts(referral, fact_type="DIAGNOSIS"):
            assert "doctor" not in f["source_text"].lower()

    def test_administrative_reference_can_never_become_an_assessment(self, finton,
                                                                    finton_note):
        for f in _facts(finton, fact_type="DIAGNOSIS"):
            assert "slash" not in f["source_text"].lower()
            assert "solicitors ref" not in f["source_text"].lower()
        assessment = _section(finton_note["note"], "Assessment")
        assert "whiplash" in assessment.lower()
        assert "f gh" not in assessment.lower()
        assert "solicitors" not in assessment.lower()
        # the reference is kept only as document metadata, never as clinical text
        other = _section(finton_note["note"], "Other Documentation")
        assert "solicitors reference" in other.lower()
        assert finton["metadata"]["solicitors_reference"]["source_span"]


# ---------------------------------------------------------------------------
# 3. Section-aware routing
# ---------------------------------------------------------------------------
class TestSectionRouting:
    def test_examination_findings_reach_physical_examination(self, referral,
                                                             referral_note):
        exam = _facts(referral, fact_type="EXAM_FINDING")
        assert any(f["english"] == "no bony tenderness" and f["status"] == "ABSENT"
                   for f in exam)
        tender = [f for f in exam if "tenderness" in f["english"] and
                  f["status"] == "PRESENT"]
        assert tender and "medial arch" in (tender[0]["attributes"]["location"] or "")
        block = _section(referral_note["note"], "Physical Examination")
        assert "no bony tenderness" in block.lower()
        assert "tenderness" in block.lower()
        assert all(f["section"] == "physical_exam" for f in exam)

    def test_occupation_content_is_evidenced(self, finton, jones, jones_note):
        """"He is a welder and had 2 weeks off work" must produce an evidenced
        OCCUPATIONAL_HISTORY fact rendered in Occupational History — never
        fabricated from metadata and never polluted by the 'Occupation'
        metadata label or 'driver's license'. The frozen jones fixture carries
        only the work-absence statement; the live welder statement is verified
        against the real Finton audio run separately."""
        occ = _facts(jones, fact_type="OCCUPATIONAL_HISTORY")
        assert occ and all(f["section"] == "occupational_history" for f in occ)
        assert any((f["attributes"] or {}).get("work_absence") for f in occ)
        for f in occ:
            assert f["source_text"].lower() in jones["evidence_text"].lower()
        # no license/label leakage into the rendered section
        block = _section(jones_note["note"], "Occupational History").lower()
        assert "driver's license" not in block and "driver’s license" not in block
        assert "occupation." not in block

    def test_welder_identity_statement_yields_occupation_fact(self):
        doc = build_clinical_facts(
            "He is a welder and had 2 weeks off work following his accident.")
        occ = _facts(doc, fact_type="OCCUPATIONAL_HISTORY")
        assert any("welder" in f["english"].lower() for f in occ)
        assert any((f["attributes"] or {}).get("work_absence") for f in occ)

    def test_metadata_occupation_never_carries_turn_labels(self):
        doc = build_clinical_facts(
            "Review of notes, none. Identification, driver's license. "
            "Occupation, 1st heading, accident, emboldened in line at 12:30PM.")
        occ_meta = doc["metadata"].get("occupation")
        assert occ_meta is None or "speaker" not in str(
            occ_meta.get("value")).lower()

    def test_medico_legal_examination_is_captured(self, jones):
        exam = _facts(jones, fact_type="EXAM_FINDING")
        assert len(exam) >= 8
        names = " ".join(f["english"].lower() for f in exam)
        assert "no spinal tenderness" in names
        assert "straight leg raising" in names
        assert "no inappropriate responses" in names

    def test_explicit_no_past_medical_history_is_documented(self, jones, jones_note):
        pmh = _facts(jones, fact_type="HISTORY", status="ABSENT")
        assert any("past medical history" in f["english"].lower() for f in pmh)
        assert all(f["section"] == "pmh" for f in pmh)
        block = _section(jones_note["note"], "Past Medical History")
        assert "explicitly none reported" in block.lower()
        assert "not documented" not in block.lower()

    def test_pmh_undiscussed_stays_not_documented(self, referral_note):
        block = _section(referral_note["note"], "Past Medical History")
        assert "not documented" in block.lower()

    def test_plan_captures_considered_treatment_and_recommendations(self, referral_note):
        plan = _section(referral_note["note"], "Plan")
        assert "steroid injection" in plan.lower()
        assert "not performed" in plan.lower()
        assert "specialist opinion" in plan.lower()

    def test_plan_captures_explicit_absence_of_recommendations(self, jones, jones_note):
        closure = [f for f in jones["facts"]
                   if f["fact_type"] == "PLAN" and f["status"] == "ABSENT"]
        assert closure, "the 'no further recommendations' statement was dropped"
        block = _section(jones_note["note"], "Plan")
        assert "no further recommendations" in block.lower()

    def test_administrative_metadata_is_documented_separately(self, referral,
                                                             referral_note):
        meta = referral["metadata"]
        dob = meta["date_of_birth"]["value"]
        assert "date_of_birth" in meta and dob
        assert "address" in meta
        # ... and never as clinical content
        for f in referral["facts"]:
            assert dob not in f["source_text"]
            assert f["section"] != "other_documentation"
        assert dob in referral_note["note"]

    def test_document_types_are_classified(self, referral, jones, finton, malayalam):
        assert referral["document"]["document_type"] == "REFERRAL_LETTER"
        assert jones["document"]["document_type"] == "MEDICOLEGAL_REPORT"
        assert finton["document"]["document_type"] == "MEDICOLEGAL_REPORT"
        assert malayalam[1]["document"]["document_type"] == \
            "STANDARD_CLINICAL_CONSULTATION"

    def test_referral_context_is_kept_for_referral_documents(self, referral_note):
        note = referral_note["note"]
        assert "Referral Context" in note
        assert "orthopedic surgeon" in note.lower()


# ---------------------------------------------------------------------------
# 4. Temporal reasoning: episodes are never collapsed
# ---------------------------------------------------------------------------
class TestTemporalReasoning:
    def test_resolved_episode_and_current_residual_are_separate(self, jones):
        pain = _facts(jones, fact_type="SYMPTOM")
        historical = [f for f in pain if (f["attributes"] or {}).get("duration")]
        current = [f for f in pain if f["status"] == "PRESENT"
                   and f["temporal_context"] in ("INTERMITTENT", "ONGOING",
                                                 "CURRENT")
                   and not (f["attributes"] or {}).get("duration")]
        assert historical, "the four-week episode was lost"
        assert historical[0]["attributes"]["duration"] == "4 weeks"
        assert any("back" in (f["english"] or "").lower() for f in current), \
            "the residual current backache was lost"
        # history and the residual problem are different facts, not one status
        assert len({f["fact_id"] for f in historical + current}) >= 2

    def test_improving_history_and_residual_pain_both_survive(self, finton, finton_note):
        hpi = _section(finton_note["note"], "History of Present Illness")
        assert "1 week" in hpi or "one week" in hpi
        assert "improved" in hpi.lower()
        assert "left arm" in hpi.lower()
        assert "right arm" in hpi.lower()
        # the current problem must not be presented as resolved
        assert hpi.index("left arm") > hpi.index("improved")

    def test_multiple_sites_are_separate_facts(self, jones):
        locations = {(f["attributes"] or {}).get("location")
                     for f in _facts(jones, fact_type="SYMPTOM")}
        assert "head" in locations
        assert "neck and back" in locations

    def test_negation_and_temporality_structure(self):
        """The A-F contract: local clause structure decides polarity and
        temporality — a distant 'not' must never deny through an intervening
        presence predicate, and separate episodes must stay separate."""
        # A. "No pain." -> pain ABSENT
        doc = build_clinical_facts("No pain.")
        assert [(f["english"], f["status"]) for f in doc["facts"]]
        assert doc["facts"][0]["status"] == "ABSENT"

        # B. "No injection was given." -> procedure ABSENT
        doc = build_clinical_facts("No injection was given.")
        procs = _facts(doc, fact_type="PROCEDURE")
        assert procs and procs[0]["status"] == "ABSENT"

        # C. "He has no history of depression." -> explicit negative PMH
        doc = build_clinical_facts("He has no history of depression.")
        hist = _facts(doc, fact_type="HISTORY")
        assert hist and hist[0]["status"] == "ABSENT"
        assert hist[0]["section"] == "pmh"
        assert hist[0]["evidence"] == "EXPLICIT_ABSENT"

        # D. the Finton residual-pain sentence: distant 'not' + intervening
        #    'still has' must leave the pain PRESENT / intermittent / right arm
        doc = build_clinical_facts(
            "He is not yet symptomatic in that he still has residual "
            "intermittent pain in his right arm.")
        pain = [f for f in doc["facts"] if f["fact_type"] == "SYMPTOM"]
        assert pain and all(f["status"] == "PRESENT" for f in pain)
        assert pain[0]["temporal_context"] in ("INTERMITTENT", "CURRENT")
        assert (pain[0]["attributes"] or {}).get("location") == "right arm"

        # E. improving episode and current intermittent episode stay separate
        doc = build_clinical_facts(
            "The pain has improved but he still has intermittent pain in his arm.")
        temps = sorted(f["temporal_context"] for f in doc["facts"])
        assert "IMPROVING" in temps and "INTERMITTENT" in temps
        assert all(f["status"] != "RESOLVED" for f in doc["facts"])

        # F. resolved episode and residual current episode: two facts, not one
        doc = build_clinical_facts("Pain resolved, but residual arm pain remains.")
        pains = [f for f in doc["facts"] if f["fact_type"] == "SYMPTOM"]
        assert len(pains) >= 2
        assert any(f["status"] == "RESOLVED" for f in pains)
        assert any(f["status"] == "PRESENT" and f["temporal_context"] != "RESOLVED"
                   for f in pains)

    def test_questioned_facts_are_not_findings(self):
        from scribe_engine import ScribeEngine
        result = ScribeEngine().process_text(
            "Patient: I have had a fever for three days.\n"
            "Doctor: Any chest pain?\nPatient: No chest pain.", language="en")
        types = [(f["fact_type"], f["english"], f["status"])
                 for f in result["clinical_facts_v2"]["facts"]]
        assert ("SYMPTOM", "chest pain", "QUESTIONED") in types
        assert "denies chest pain" in result["clinical_note_v2"]["note"].lower()

    def test_english_fronted_negation_denies_the_procedure(self):
        """"No injection was given." must be an ABSENT procedure: the fronted
        negation outranks the past-tense verb that follows the concept.
        (Regression: the polarity heuristic read the following 'was' as a
        presence marker and produced PROCEDURE injection = PRESENT.)"""
        doc = build_clinical_facts("No injection was given.")
        procs = _facts(doc, fact_type="PROCEDURE")
        assert procs and procs[0]["status"] == "ABSENT"

    def test_english_fronted_negation_keeps_present_mentions(self):
        """The negation fix must not deny genuinely present findings."""
        for text in ("I have chest pain.", "The patient has fever."):
            doc = build_clinical_facts(text)
            assert any(f["status"] == "PRESENT" for f in doc["facts"]), text

    def test_negation_defeated_by_intervening_presence(self):
        """A real medico-legal sentence: 'He is not yet symptomatic in that he
        still HAS some residual intermittent pain in his right arm.' The 'not'
        is too far away and a presence marker sits between it and the concept —
        the residual pain must stay PRESENT. (Regression: the fronted-negation
        rule denied this pain outright on the live Deepgram transcript.)"""
        doc = build_clinical_facts(
            "He is not yet symptomatic in that he still has some residual "
            "intermittent pain in his right arm.")
        pain = [f for f in doc["facts"] if f["fact_type"] == "SYMPTOM"]
        assert pain and all(f["status"] == "PRESENT" for f in pain)

    def test_owned_history_is_pmh_not_diagnosis(self):
        """"He has a history of pneumonia" documents a PAST condition in PMH —
        never a current finding and never an assessment diagnosis."""
        doc = build_clinical_facts("He has a history of pneumonia.")
        assert _facts(doc, fact_type="HISTORY", contains="pneumonia")
        assert not _facts(doc, fact_type="DIAGNOSIS")
        assert not _facts(doc, fact_type="SYMPTOM", contains="pneumonia")
        assert all(f["section"] == "pmh"
                   for f in _facts(doc, contains="pneumonia"))

    def test_history_negation_is_not_pmh_content(self):
        """"No history of pneumonia" / family history stay out of the owned-"""
        doc = build_clinical_facts("There is no history of pneumonia.")
        assert not _facts(doc, fact_type="HISTORY", contains="pneumonia",
                          status=None) or \
            all(f["status"] == "ABSENT" for f in _facts(doc, contains="pneumonia"))

    def test_condition_noun_diagnosis_with_explicit_cues(self):
        """Suffix-free condition nouns ("pneumonia") are diagnoses only with an
        explicit diagnostic cue — and 'rule out' can never confirm."""
        doc = build_clinical_facts("Pneumonia was diagnosed.")
        dx = _facts(doc, fact_type="DIAGNOSIS")
        assert dx and dx[0]["certainty"] == "CONFIRMED"
        doc = build_clinical_facts("We should rule out pneumonia.")
        dx = _facts(doc, fact_type="DIAGNOSIS")
        assert dx and dx[0]["certainty"] == "SUSPECTED"
        assert "rule out" not in dx[0]["english"]

    def test_ongoing_trajectory_for_continue(self):
        """"The pain continues" is ONGOING, not a plain CURRENT present."""
        doc = build_clinical_facts("The pain continues.")
        assert doc["facts"][0]["temporal_context"] == "ONGOING"


# ---------------------------------------------------------------------------
# 5. HPI quality
# ---------------------------------------------------------------------------
class TestHpiComposition:
    def test_hpi_keeps_onset_mechanism_and_location(self, referral_note):
        hpi = _section(referral_note["note"], "History of Present Illness")
        assert "left medial foot arch" in hpi.lower()
        assert "began after buying tight footwear" in hpi.lower()

    def test_hpi_keeps_trajectory_and_aggravating_factor(self, referral_note):
        hpi = _section(referral_note["note"], "History of Present Illness")
        assert "fluctuating since March" in hpi
        assert "gets worse after" in hpi.lower()

    def test_hpi_keeps_treatment_response(self, referral_note):
        hpi = _section(referral_note["note"], "History of Present Illness")
        assert "no benefit was obtained from physiotherapy" in hpi.lower()

    def test_hpi_never_mentions_a_medication_that_was_not_taken(self, referral_note,
                                                                finton_note):
        for out in (referral_note, finton_note):
            hpi = _section(out["note"], "History of Present Illness").lower()
            for invented in ("paracetamol", "cetirizine", "ibuprofen", "injection"):
                assert invented not in hpi

    def test_hpi_is_narrative_not_a_keyword_list(self, referral_note):
        hpi = _section(referral_note["note"], "History of Present Illness")
        bullets = [l for l in hpi.splitlines() if l.strip().startswith("-")]
        assert len(bullets) >= 4
        assert all(l.strip().endswith(".") for l in bullets)


# ---------------------------------------------------------------------------
# 6. Note validator is fail-closed
# ---------------------------------------------------------------------------
class TestNoteValidatorFailClosed:
    def _tamper(self, out, extra: str):
        return out["note"] + f"\n{extra}\n"

    def test_deterministic_notes_validate(self, referral_note, jones_note, finton_note,
                                         malayalam):
        for out in (referral_note, jones_note, finton_note,
                    render_clinical_note(malayalam[1])):
            assert out["validation"]["valid"], out["validation"]["violations"]

    def test_injected_medication_is_rejected(self, referral, referral_note):
        v = validate_note_v2(self._tamper(referral_note,
                                          "7. Medications\n- paracetamol 500 mg"),
                             referral)
        assert not v["valid"]

    def test_injected_diagnosis_is_rejected(self, referral, referral_note):
        v = validate_note_v2(self._tamper(referral_note,
                                          "13. Assessment\n- Pneumonia."),
                             referral)
        assert not v["valid"]

    def test_injected_greeting_is_rejected(self, referral, referral_note):
        v = validate_note_v2(self._tamper(referral_note,
                                          "Hi. This is Doctor. Sarah Whitfield."),
                             referral)
        assert not v["valid"]

    def test_injected_duration_is_rejected(self, referral, referral_note):
        v = validate_note_v2(self._tamper(referral_note,
                                          "The pain has been present for six months."),
                             referral)
        assert not v["valid"]

    def test_injected_denial_is_rejected(self, referral, referral_note):
        v = validate_note_v2(self._tamper(referral_note,
                                          "The patient denies wheezing."),
                             referral)
        assert not v["valid"]

    def test_note_validator_also_rejects_an_invalid_fact_graph(self, referral):
        broken = build_clinical_facts(_read(REAL / "referral_letter_orthopaedic.txt"))
        broken["facts"][0]["source_span"] = None
        out = render_clinical_note(broken)
        assert not out["validation"]["valid"]


# ---------------------------------------------------------------------------
# 7. Malayalam path parity (corrections are the evidence text)
# ---------------------------------------------------------------------------
class TestMalayalamParity:
    def test_the_verified_concepts_survive(self, malayalam):
        _, doc = malayalam
        got = {f["english"]: f["status"] for f in doc["facts"]
               if f["fact_type"] == "SYMPTOM"}
        assert got.get("fever") == "RESOLVED"
        for concept in ("fatigue", "headache", "cough", "phlegm",
                        "difficulty eating"):
            assert got.get(concept) == "PRESENT", got

    def test_uncorrected_transcript_alone_is_not_enough(self):
        """The corrections are load-bearing: without them three concepts vanish."""
        raw = _read(FIXTURES / "malayalam_real_consultation.txt").strip()
        without = build_clinical_facts(raw, [], roles_known=False)
        assert len(without["facts"]) < 6

    def test_note_renders_without_speaker_roles(self, malayalam):
        out = render_clinical_note(malayalam[1])
        assert out["validation"]["valid"]
        assert "Doctor" not in out["note"] and "Patient" not in out["note"]
        assert "fever" in out["note"].lower()


# ---------------------------------------------------------------------------
# 8. Pipeline wiring (additive, still gated on validation)
# ---------------------------------------------------------------------------
class TestPipelineWiring:
    def test_pipeline_exposes_typed_facts_and_a_validated_note(self):
        from scribe_engine import ScribeEngine

        result = ScribeEngine().process_text(
            "Patient: I have had a fever and a cough for three days.\n"
            "Doctor: Take paracetamol twice a day.\n"
            "Patient: I have no other complaints.", language="en")
        v2 = result["clinical_note_v2"]
        facts = result["clinical_facts_v2"]
        assert v2["validation"]["valid"], v2["validation"]["violations"]
        assert v2["document_type"] == "STANDARD_CLINICAL_CONSULTATION"
        assert facts["facts"] and facts["validation"]["valid"]
        assert any(f["fact_type"] == "SYMPTOM" for f in facts["facts"])
        assert "fever" in v2["note"].lower()
        assert "raw_text" not in facts          # the transcript is not duplicated

    def test_async_completion_gate_reads_the_v2_validation(self):
        """service/app.py gates job completion on this exact field."""
        from scribe_engine import ScribeEngine

        result = ScribeEngine().process_text(
            "Patient: I have a headache.\nDoctor: Since when?\n"
            "Patient: For two days.", language="en")
        assert result["clinical_note_v2"]["validation"]["valid"] is True


# ---------------------------------------------------------------------------
# 9. UI/rendering refinements on the real cases (regression battery)
# ---------------------------------------------------------------------------
class TestOccupationRoleVsOccupation:
    def test_driver_in_accident_context_is_not_occupation(self):
        """TASK 3A: a road-user role in a collision narration ('He was a
        driver' between vehicle/accident clauses) is not the patient's
        occupation. The welder statement in the same document IS."""
        text = ("Fintan was traveling in a Nissan fitted with seat belts and "
                "headrests, full stop. He was a driver, full stop. No warning. "
                "Whilst turning right, another vehicle collided with his car. "
                "He is a welder and had 2 weeks off work following his accident.")
        doc = build_clinical_facts(text)
        occ = _facts(doc, fact_type="OCCUPATIONAL_HISTORY")
        names = [f["english"] for f in occ]
        assert names == ["welder"], names
        assert not _facts(doc, contains="driver")

    def test_event_anchored_and_vehicle_narration_roles_are_not_occupations(self):
        text = ("He was a driver when the collision occurred. He is a nurse. "
                "He was a passenger in his father's car when the accident happened.")
        doc = build_clinical_facts(text)
        names = [f["english"] for f in _facts(doc, fact_type="OCCUPATIONAL_HISTORY")]
        assert names == ["nurse"], names

    def test_present_occupation_and_qualified_titles_stay_occupations(self):
        doc = build_clinical_facts(
            "He works as a taxi driver in the city. She works as a bus driver.")
        names = [f["english"] for f in _facts(doc, fact_type="OCCUPATIONAL_HISTORY")]
        assert any("taxi" in n for n in names), names
        assert any("bus" in n for n in names), names

    def test_finton_driver_role_never_renders_in_occupational_history(
            self, finton, finton_note):
        assert not _facts(finton, fact_type="OCCUPATIONAL_HISTORY",
                          contains="driver")
        block = _section(finton_note["note"], "Occupational History").lower()
        assert "driver" not in block


class TestDuplicateRendering:
    def test_duplicate_occupation_statement_is_not_rendered_twice(self, finton,
                                                                  finton_note):
        """TASK 3B: the welder-with-absence statement and the later
        accident-caused-absence statement are the same underlying fact — both
        stay in the fact graph with their own provenance, but the note renders
        the work absence exactly once."""
        occ = _facts(finton, fact_type="OCCUPATIONAL_HISTORY")
        assert any((f.get("attributes") or {}).get("work_absence")
                   for f in occ), occ
        block = _section(finton_note["note"], "Occupational History")
        assert block.lower().count("2 weeks off work") == 1
        assert "He is a welder" in block

    def test_duplicate_exam_findings_are_not_rendered_twice(self, finton,
                                                            finton_note):
        """TASK 3C: 'full range of movement' / 'full spinal movement' over the
        same cervical spine is one finding; the broader statement survives."""
        block = _section(finton_note["note"], "Physical Examination").lower()
        movement_lines = [l for l in block.splitlines()
                          if "movement" in l and "full" in l]
        assert len(movement_lines) <= 1, movement_lines
        # distinct findings are preserved
        assert "no muscular tenderness" in block
        assert "no other marks, scars or bruises" in block

    def test_equivalent_exam_findings_from_worded_differently_dedupe(self):
        from scribe_engine.note_v2 import _exam_lines
        facts = [
            {"fact_type": "EXAM_FINDING", "english": "full range of movement",
             "status": "PRESENT", "attributes": {"location": "cervical spine"}},
            {"fact_type": "EXAM_FINDING", "english": "full spinal movement",
             "status": "PRESENT", "attributes": {"location": "cervical spine"}},
        ]
        lines = _exam_lines(facts)
        assert len(lines) == 1, lines
        # distinct sites stay distinct
        facts2 = [
            {"fact_type": "EXAM_FINDING", "english": "full range of movement",
             "status": "PRESENT", "attributes": {"location": "cervical spine"}},
            {"fact_type": "EXAM_FINDING", "english": "full range of movement",
             "status": "PRESENT", "attributes": {"location": "lumbosacral spine"}},
        ]
        assert len(_exam_lines(facts2)) == 2


class TestNegativeStatements:
    def test_no_immediate_symptoms_is_not_a_current_hpi_symptom(self, finton,
                                                               finton_note):
        """TASK 3E: kept as an explicit historical negative in Pertinent
        Negatives; never a current HPI symptom line."""
        hpi = _section(finton_note["note"], "History of Present Illness").lower()
        assert "no immediate symptoms" not in hpi
        assert not _facts(finton, contains="immediate symptoms",
                          status="PRESENT")
        pn = _section(finton_note["note"], "Pertinent Negatives").lower()
        assert "no immediate symptoms" in pn

    def test_no_consequence_does_not_appear_as_an_unexplained_hpi_statement(
            self, finton, finton_note, jones, jones_note):
        """TASK 3D: an event-attributed 'no consequence' is rendered once with
        its event reference (Prognosis); a bare 'no consequence' is omitted."""
        hpi = _section(finton_note["note"], "History of Present Illness")
        assert "No consequence." not in hpi
        assert "consequence of the accident" in \
            _section(finton_note["note"], "Prognosis").lower()
        assert "no consequence" not in \
            _section(finton_note["note"], "Pertinent Negatives").lower()
        # a bare 'no consequence' with no event reference produces no fact
        doc = build_clinical_facts("There has been no consequence, full stop.")
        assert not [f for f in doc["facts"] if "consequence" in f["english"]]

    def test_psychological_negatives_render_once(self, finton, finton_note,
                                                 jones, jones_note):
        """TASK 3F: the psych negative is documented once (Pertinent
        Negatives), never duplicated in HPI; the fact itself is kept."""
        for doc, note in ((finton, finton_note), (jones, jones_note)):
            psych = _facts(doc, contains="psychological")
            assert psych, "psych negative fact was deleted"
            assert all(f["status"] == "ABSENT" for f in psych)
            hpi = _section(note["note"], "History of Present Illness").lower()
            assert "psychological" not in hpi, hpi
            pn = _section(note["note"], "Pertinent Negatives").lower()
            assert pn.count("psychological") == 1, pn


class TestSpeakerLabelHygiene:
    def test_anonymous_speaker_labels_never_reach_rendered_v2_content(self):
        """TASK 2: 'Speaker 0:' is presentation structure. Stored/pasted
        transcripts can carry it mid-clause; it must not appear in any
        rendered clinical content."""
        text = ("Speaker 0: He is a welder and had 2 weeks off work following "
                "his accident. Speaker 1: There have been no serious "
                "psychological symptoms associated with the accident.")
        doc = build_clinical_facts(text)
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        assert "Speaker" not in out["note"]
        for f in doc["facts"]:
            assert "speaker" not in (f["source_text"] or "").lower(), f
        for e in doc["elements"]:
            assert "speaker" not in (e["source_text"] or "").lower(), e

    def test_audio_pipeline_strips_anonymous_labels_from_turn_join(self):
        """The live pipeline (_line in pipeline.py) never prefixes anonymous
        labels onto transcript lines."""
        from scribe_engine import pipeline as P
        import inspect
        src = inspect.getsource(P.ScribeEngine.process_audio)
        assert "Speaker\\s*\\d+" in src


class TestSingleAuthoritativeNote:
    def test_legacy_clinical_note_never_overrides_v2(self):
        """TASK 6.8: the pipeline result carries the legacy extraction for
        backward compatibility, but the V2 note is what the UI/export treat as
        the clinical note."""
        from scribe_engine import ScribeEngine
        result = ScribeEngine().process_text(
            "Patient: I have had a fever for three days.\n"
            "Doctor: Take paracetamol twice a day.", language="en")
        v2 = result["clinical_note_v2"]
        legacy = result["clinical_note"]
        assert v2["validation"]["valid"]
        assert isinstance(legacy, dict) and "fields" in legacy  # still present
        assert v2["note"] != legacy.get("text", "")
        assert "fever" in v2["note"].lower()

    def test_export_uses_v2_note_and_labels_legacy(self):
        """The export document shows the validated V2 note as THE clinical
        note; the legacy template text is appended only as labelled,
        superseded debug content."""
        import service.app as S
        from scribe_engine import ScribeEngine
        result = ScribeEngine().process_text(
            "Patient: I have had a fever for three days.\n"
            "Doctor: Take paracetamol twice a day.", language="en")
        record = {"id": "test1234", "patient_id": "TRIAL-001",
                  "doctor": "Dr. Demo", "department": "General Medicine",
                  "consultation_type": "General Consultation",
                  "reviewed": False, "completed_at": "2026-01-01 00:00:00",
                  "created_at": "2026-01-01 00:00:00", "result": result}
        text = S._export_text(record)
        assert "CLINICAL NOTE" in text
        assert "LEGACY TEMPLATE NOTE" in text
        v2_header = result["clinical_note_v2"]["note"].strip().splitlines()[0]
        assert text.index(v2_header) < text.index("LEGACY TEMPLATE NOTE")

    def test_legacy_prescription_does_not_override_v2_plan(self, referral,
                                                           referral_note):
        """TASK 6.9: the V2 Plan is rendered from PLAN/PROCEDURE facts; the
        legacy prescription extractor is a separate structure that never
        touches it."""
        plan = _section(referral_note["note"], "Plan").lower()
        assert "steroid injection" in plan
        assert "considered, not performed" in plan
        assert "specialist opinion" in plan
        # and the rendered note carries no legacy 'Additional advice' wording
        assert "additional advice" not in referral_note["note"].lower()

    def test_ui_shows_one_authoritative_note(self):
        """TASK 6.8 (frontend contract): the Results page renders the V2 note
        panel as THE note; the legacy extraction lives in a labelled debug
        panel and can no longer surface as a competing 'Clinical Note'."""
        html = (Path(__file__).resolve().parent.parent / "service" /
                "static" / "index.html").read_text(encoding="utf-8")
        results = html.split('id="view-results"')[1]
        assert 'id="note-v2-panel"' in results
        assert 'id="legacy-extraction-panel"' in results
        assert "Clinical Note</h2>" not in results, \
            "a second, competing Clinical Note card is still present"
        js = (Path(__file__).resolve().parent.parent / "service" /
              "static" / "app.js").read_text(encoding="utf-8")
        assert "Fact graph validated" in js   # indicator retained
        assert 'legacyPanel.classList.remove("hidden")' in js


class TestReferralAndValidatorRegressions:
    def test_referral_case_remains_clinically_unchanged(self, referral_note):
        """TASK 4 / 10: suspected stays suspected; injection stays a
        considered procedure; the note shape is the verified one."""
        note = referral_note["note"]
        assessment = _section(note, "Assessment").lower()
        assert "suspected" in assessment and "plantar fasciitis" in assessment
        assert "not confirmed" in assessment
        plan = _section(note, "Plan").lower()
        assert "steroid injection" in plan and "not performed" in plan
        hpi = _section(note, "History of Present Illness").lower()
        assert "began after buying tight footwear" in hpi
        assert "fluctuating since march" in hpi
        assert "no benefit was obtained from physiotherapy" in hpi

    def test_finton_case_remains_evidence_valid(self, finton, finton_note):
        """TASK 6.11: every rendered Finton line still traces to the fact
        graph and the graph to the transcript."""
        assert finton["validation"]["valid"]
        assert finton_note["validation"]["valid"], finton_note["validation"]
        for f in finton["facts"]:
            assert f["source_text"] in finton["evidence_text"], f

    def test_occupational_lines_must_be_verbatim(self, referral, referral_note):
        """TASK 6.12 extension: injected occupational content is rejected by
        the note validator (the renderer emits source_text verbatim)."""
        from scribe_engine.note_v2 import validate_note_v2
        doctored = dict(referral)
        appended = " He is a deep sea diver."
        doctored["evidence_text"] = doctored["evidence_text"] + appended
        start = len(doctored["evidence_text"]) - len(appended) + 1
        doctored["facts"] = list(referral["facts"]) + [dict(
            referral["facts"][0],
            fact_id="FZ", fact_type="OCCUPATIONAL_HISTORY",
            english="deep sea diver", status="PRESENT", certainty="CONFIRMED",
            evidence="EXPLICIT_PRESENT", section="occupational_history",
            source_text="He is a deep sea diver",
            source_span=[start, start + len("He is a deep sea diver")],
            mention_span=[start, start + len("deep sea diver")],
            evidence_text="deep sea diver",
            attributes={"location": None})]
        out = render_clinical_note(doctored)
        assert out["validation"]["valid"], out["validation"]["violations"]
        # tamper with the rendered line -> must fail closed
        tampered = out["note"].replace(
            "He is a deep sea diver", "He is a deep sea diver with benefits")
        v = validate_note_v2(tampered, doctored)
        assert not v["valid"], v
       
