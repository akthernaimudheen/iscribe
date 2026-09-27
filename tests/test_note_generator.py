# -*- coding: utf-8 -*-
"""Note generator regression tests: fact graph -> 14-section note.

Covers the briefing contract:
  - HPI chronology (resolved-with-duration first, ongoing, then current)
  - every truth-model state (PRESENT / ABSENT / RESOLVED / UNCERTAIN /
    QUESTIONED / NOT_DOCUMENTED) and NOT_DOCUMENTED never becoming ABSENT
  - PMH / PSH / medications / allergies / family / social only from evidence
  - ROS only from evidence; exam / diagnosis / plan never invented
  - provenance trace for every rendered fact
  - fail-closed validator: hallucinated diagnosis / medication / exam / plan /
    negative / duration / severity / causality are all rejected
  - LLM renderer disabled by default; fallback note equals deterministic note

Fact-graph construction here uses the exact shapes produced by
scribe_engine.fact_graph.build_fact_graph from semantic-layer entities.
"""

import io
import sys

import pytest

sys.path.insert(0, __import__("os").path.dirname(
    __import__("os").path.dirname(__import__("os").path.abspath(__file__))))

from scribe_engine.fact_graph import build_fact_graph, validate_fact_graph  # noqa: E402
from scribe_engine.note_generator import (  # noqa: E402
    LLM_SYSTEM_PROMPT,
    render_deterministic_note,
    render_llm_note,
    validate_note_against_fact_graph,
)


def _fact(concept, english, status, *, surface="surf", clause="clause",
          speaker="patient", confidence=0.9, duration=None, frequency=None,
          temporality=None, attrs=None, uncertainty=None, ongoing=False):
    f = {
        "concept": concept, "english": english, "status": status,
        "certainty": "CONFIRMED" if status in ("PRESENT", "ABSENT", "RESOLVED")
        else "UNCERTAIN",
        "temporality": temporality or ("PAST_RESOLVED" if status == "RESOLVED"
                                       else "CURRENT"),
        "duration": duration, "onset": None, "frequency": frequency,
        "surface_text": surface, "source_clause": clause,
        "speaker": speaker, "confidence": confidence,
        "attributes": attrs or {}, "uncertainty_reason": uncertainty,
        "raw_text": "RAW",
    }
    if ongoing:
        f["ongoing"] = True
    return f


def _graph(facts, turns=None, roles_known=False):
    return build_fact_graph(facts, turns or [], raw_text="RAW ASR TEXT",
                            roles_known=roles_known)


# ---------------------------------------------------------------------------
# Truth model & section content
# ---------------------------------------------------------------------------

class TestTruthModel:
    def test_present(self):
        g = _graph([_fact("cough", "cough", "PRESENT")])
        note = render_deterministic_note(g)["note"]
        assert "reports cough" in note

    def test_absent_pertinent_negative(self):
        g = _graph([_fact("fever", "fever", "ABSENT")])
        note = render_deterministic_note(g)["note"]
        assert "Denies fever" in note

    def test_resolved_temporal_language(self):
        g = _graph([_fact("fever", "fever", "RESOLVED")])
        note = render_deterministic_note(g)["note"]
        assert "previously experienced fever" in note
        assert "has since resolved" in note

    def test_ongoing_temporal_language(self):
        g = _graph([_fact("fatigue", "fatigue", "PRESENT", ongoing=True)])
        note = render_deterministic_note(g)["note"]
        assert "continues to experience fatigue" in note
        # Ongoing fatigue must not also be claimed as merely "reported" in the
        # grouped PRESENT sentence — persistence wording is exclusive.
        assert "reports fatigue" not in note

    def test_uncertain_reports_possible(self):
        g = _graph([_fact("fever", "fever", "UNCERTAIN",
                          uncertainty="patient unsure")])
        note = render_deterministic_note(g)["note"]
        assert "possible fever" in note
        assert "uncertain" in note

    def test_questioned_never_a_finding(self):
        g = _graph([_fact("fever", "fever", "QUESTIONED", confidence=0.0)])
        note = render_deterministic_note(g)["note"]
        assert "reports fever" not in note
        assert "Denies fever" not in note

    def test_not_documented_never_becomes_absent(self):
        # No diabetes fact at all -> PMH not documented, never "denies diabetes"
        g = _graph([_fact("cough", "cough", "PRESENT")])
        out = render_deterministic_note(g)
        assert "Not documented." in out["note"]
        assert "diabetes" not in out["note"].lower()

    def test_explicit_pmh_absent_is_explicit(self):
        g = _graph([_fact("diabetes_history", "diabetes", "ABSENT")])
        note = render_deterministic_note(g)["note"]
        assert "Denies diabetes" in note


# ---------------------------------------------------------------------------
# HPI chronology & attributes
# ---------------------------------------------------------------------------

class TestHPI:
    def test_resolved_first_with_duration_then_current(self):
        facts = [
            _fact("fatigue", "fatigue", "PRESENT", ongoing=True),
            _fact("fever", "fever", "RESOLVED", duration="6 days"),
        ]
        note = render_deterministic_note(_graph(facts))["note"]
        hpi = note.split("2. History of Present Illness")[1].split("3.")[0]
        assert "history of fever for approximately six days, which has since resolved" in hpi
        assert "continues to experience fatigue" in hpi
        assert hpi.index("fever") < hpi.index("fatigue")

    def test_present_grouped_sentence(self):
        facts = [
            _fact("cough", "cough", "PRESENT"),
            _fact("headache", "headache", "PRESENT"),
            _fact("difficulty_eating", "difficulty eating", "PRESENT"),
        ]
        hpi = render_deterministic_note(_graph(facts))["note"] \
            .split("2. History of Present Illness")[1].split("3.")[0]
        assert "reports cough, headache, and difficulty eating" in hpi

    def test_no_hallucinated_severity_or_cause(self):
        g = _graph([_fact("fever", "fever", "RESOLVED", duration="6 days")])
        note = render_deterministic_note(g)["note"]
        assert "severe" not in note.lower()
        assert "due to" not in note.lower()
        assert "viral" not in note.lower()

    def test_duration_only_from_fact_graph(self):
        g = _graph([_fact("fever", "fever", "PRESENT", duration="6 days")])
        note = render_deterministic_note(g)["note"]
        assert "for approximately six days" in note
        # A different fact with no duration must not inherit one
        g2 = _graph([_fact("cough", "cough", "PRESENT")])
        assert "six days" not in render_deterministic_note(g2)["note"]

    # --- English note-quality grammar (deterministic, evidence-gated) ---

    def test_bare_resolved_uses_previously_experienced(self):
        """'history of' without a documented episode is forbidden (spec: it
        implies historical documentation, not a recently resolved symptom)."""
        g = _graph([_fact("fever", "fever", "RESOLVED")])
        note = render_deterministic_note(g)["note"]
        assert "The patient previously experienced fever, which has since resolved." in note
        assert "history of" not in note

    def test_resolved_with_duration_keeps_history_of(self):
        g = _graph([_fact("fever", "fever", "RESOLVED", duration="6 days")])
        note = render_deterministic_note(g)["note"]
        assert "a history of fever for approximately six days, which has since resolved." in note

    def test_currently_reports_only_after_resolved_fact(self):
        """The temporal pivot appears iff a resolved fact precedes."""
        with_resolved = _graph([
            _fact("fever", "fever", "RESOLVED", duration="6 days"),
            _fact("cough", "cough", "PRESENT")])
        note = render_deterministic_note(with_resolved)["note"]
        assert "The patient currently reports cough." in note

        without = _graph([_fact("cough", "cough", "PRESENT")])
        note2 = render_deterministic_note(without)["note"]
        assert "The patient reports cough." in note2
        assert "currently" not in note2.lower()

    def test_grouped_denial_in_hpi(self):
        g = _graph([_fact("chest_pain", "chest pain", "ABSENT"),
                    _fact("shortness_of_breath", "shortness of breath", "ABSENT")])
        hpi = render_deterministic_note(g)["note"] \
            .split("2. History of Present Illness")[1].split("3.")[0]
        assert "The patient denies chest pain and shortness of breath." in hpi

    def test_mixed_present_absent_hpi(self):
        """Example H: explicit negatives appear in the HPI narrative."""
        g = _graph([_fact("fatigue", "fatigue", "PRESENT"),
                    _fact("shortness_of_breath", "shortness of breath", "ABSENT")])
        hpi = render_deterministic_note(g)["note"] \
            .split("2. History of Present Illness")[1].split("3.")[0]
        assert "The patient reports fatigue." in hpi
        assert "The patient denies shortness of breath." in hpi

    def test_approximation_hedge_preserved_end_to_end(self):
        g = _graph([_fact("fatigue", "fatigue", "PRESENT", duration="6 days")])
        hpi = render_deterministic_note(g)["note"] \
            .split("2. History of Present Illness")[1].split("3.")[0]
        assert "fatigue for approximately six days" in hpi
        assert "fatigue for six days" not in hpi

    def test_real_consultation_hpi_shape(self):
        """The verified 6-fact consultation renders the polished narrative."""
        facts = [
            _fact("fever", "fever", "RESOLVED", duration="6 days"),
            _fact("fatigue", "fatigue", "PRESENT", ongoing=True),
            _fact("headache", "headache", "PRESENT"),
            _fact("cough", "cough", "PRESENT"),
            _fact("phlegm", "congestion", "PRESENT"),
            _fact("difficulty_eating", "difficulty eating", "PRESENT"),
        ]
        hpi = render_deterministic_note(_graph(facts))["note"] \
            .split("2. History of Present Illness")[1].split("3.")[0]
        assert "history of fever for approximately six days, which has since resolved." in hpi
        assert "The patient continues to experience fatigue." in hpi
        assert "reports headache, cough, congestion, and difficulty eating." in hpi
        # Ongoing is never merged into the grouped sentence.
        assert "continues to experience headache" not in hpi

    def test_pmh_absent_renders_explicit_denial_form(self):
        g = _graph([_fact("diabetes_history", "diabetes", "ABSENT")])
        pmh = render_deterministic_note(g)["note"] \
            .split("5. Past Medical History")[1].split("6.")[0]
        assert "Diabetes: Explicitly denied." in pmh


# ---------------------------------------------------------------------------
# Sections only from evidence
# ---------------------------------------------------------------------------

class TestEvidenceOnlySections:
    def test_pmh_only_from_pmh_concepts(self):
        g = _graph([_fact("hypertension_history", "hypertension", "PRESENT")])
        note = render_deterministic_note(g)["note"]
        assert "Hypertension: Present." in note

    def test_pmh_silence_not_documented(self):
        g = _graph([_fact("cough", "cough", "PRESENT")])
        pmh = render_deterministic_note(g)["note"] \
            .split("5. Past Medical History")[1].split("6.")[0]
        assert "Not documented" in pmh

    def test_medications_only_from_medication_facts(self):
        g = _graph([_fact("paracetamol", "paracetamol", "PRESENT",
                          attrs={"frequency": "twice daily"})])
        med = render_deterministic_note(g)["note"] \
            .split("7. Medications")[1].split("8.")[0]
        assert "paracetamol" in med
        assert "frequency twice daily" in med

    def test_no_medication_inferred_from_symptoms(self):
        g = _graph([_fact("fever", "fever", "PRESENT"),
                    _fact("cough", "cough", "PRESENT")])
        med = render_deterministic_note(g)["note"] \
            .split("7. Medications")[1].split("8.")[0]
        assert "Not documented" in med

    def test_allergies_explicit_absent(self):
        g = _graph([_fact("allergy", "allergy", "ABSENT")])
        alg = render_deterministic_note(g)["note"] \
            .split("8. Allergies")[1].split("9.")[0]
        assert "explicitly denied" in alg

    def test_family_history_from_family_speaker(self):
        g = _graph([_fact("diabetes_history", "diabetes", "PRESENT",
                          speaker="mother")])
        fam = render_deterministic_note(g)["note"] \
            .split("9. Family History")[1].split("10.")[0]
        assert "diabetes in mother" in fam

    def test_family_symptom_not_a_patient_finding(self):
        g = _graph([_fact("cough", "cough", "PRESENT", speaker="mother")])
        out = render_deterministic_note(g)
        hpi = out["note"].split("2. History of Present Illness")[1].split("3.")[0]
        assert "cough" not in hpi

    def test_exam_only_from_doctor_lines(self):
        g = _graph([_fact("cough", "cough", "PRESENT")],
                   turns=[{"speaker": "Doctor", "text": "Lungs are clear."}])
        exam = render_deterministic_note(g)["note"] \
            .split("12. Physical Examination")[1].split("13.")[0]
        assert "Lungs clear to auscultation" in exam

    def test_no_exam_evidence_not_documented_no_normals(self):
        g = _graph([_fact("cough", "cough", "PRESENT")])
        exam = render_deterministic_note(g)["note"] \
            .split("12. Physical Examination")[1].split("13.")[0]
        assert "Not documented" in exam
        for banned in ("vitals", "alert", "well appearing", "no distress"):
            assert banned not in exam.lower()

    def test_no_diagnosis_no_invention(self):
        g = _graph([_fact("fever", "fever", "PRESENT"),
                    _fact("cough", "cough", "PRESENT"),
                    _fact("phlegm", "congestion", "PRESENT")])
        note = render_deterministic_note(g)["note"]
        assert "No definitive diagnosis documented in this encounter" in note
        assert "viral URI" not in note
        assert "pneumonia" not in note.lower()

    def test_clinician_diagnosis_documented_verbatim(self):
        g = _graph([_fact("fever", "fever", "PRESENT")],
                   turns=[{"speaker": "Doctor",
                           "text": "This is a viral fever."}])
        note = render_deterministic_note(g)["note"]
        assert "Clinician-stated diagnosis (verbatim)" in note
        assert "viral fever" in note

    def test_plan_only_from_doctor_actions(self):
        g = _graph([_fact("cough", "cough", "PRESENT")],
                   turns=[{"speaker": "Doctor", "text": "I will advise rest."}])
        plan = render_deterministic_note(g)["note"] \
            .split("14. Plan")[1].strip()
        assert "General advice given" in plan

    def test_no_plan_not_documented(self):
        g = _graph([_fact("cough", "cough", "PRESENT")])
        plan = render_deterministic_note(g)["note"] \
            .split("14. Plan")[1].strip()
        assert "Not documented" in plan

    def test_doctor_question_not_plan_or_diagnosis(self):
        g = _graph([_fact("cough", "cough", "PRESENT")],
                   turns=[{"speaker": "Doctor",
                           "text": "Do you have any chest pain?"}])
        out = render_deterministic_note(g)["note"]
        assert "chest pain" not in out.split("14. Plan")[1].lower()
        assert "No definitive diagnosis" in out

    def test_pipeline_text_flow_exposes_clinical_note_v2(self):
        """Wiring: the fact-graph note is additive on the engine result and
        self-validates. Deterministic renderer only; no LLM anywhere."""
        from scribe_engine import ScribeEngine

        result = ScribeEngine().process_text(
            "Patient: I have had a fever for three days.\n"
            "Doctor: Any chest pain?\nPatient: No chest pain.", language="en")
        v2 = result.get("clinical_note_v2")
        assert v2 and v2["note"] and v2["validation"]["valid"], v2
        # Existing outputs untouched (additive wiring).
        assert "clinical_note" in result and "clinical_documentation" in result
        assert "fever" in v2["note"].lower()

    def test_roles_unknown_no_doctor_patient_labels(self):
        g = _graph([_fact("cough", "cough", "PRESENT")],
                   turns=[{"speaker": "Speaker 0", "text": "I will advise rest."}],
                   roles_known=False)
        out = render_deterministic_note(g)
        assert "Doctor" not in out["note"]
        assert "Patient" not in out["note"]


# ---------------------------------------------------------------------------
# ROS / chief complaint / associated symptoms
# ---------------------------------------------------------------------------

class TestRosAndGrouping:
    def test_ros_only_evidence_systems(self):
        g = _graph([_fact("fatigue", "fatigue", "PRESENT"),
                    _fact("cough", "cough", "PRESENT"),
                    _fact("phlegm", "congestion", "PRESENT")])
        ros = render_deterministic_note(g)["note"] \
            .split("11. Review of Systems")[1].split("12.")[0]
        assert "Constitutional: fatigue" in ros
        assert "Respiratory: cough; congestion" in ros
        assert "Cardiovascular" not in ros      # no evidence, no system
        assert "Genitourinary" not in ros

    def test_chief_complaint_first_present(self):
        g = _graph([_fact("cough", "cough", "PRESENT"),
                    _fact("headache", "headache", "PRESENT")])
        cc = render_deterministic_note(g)["note"] \
            .split("1. Chief Complaint")[1].split("2.")[0]
        assert "Cough" in cc          # capitalized, no stray period
        assert "cough." not in cc

    def test_associated_groups_not_causal(self):
        g = _graph([_fact("cough", "cough", "PRESENT"),
                    _fact("phlegm", "congestion", "PRESENT"),
                    _fact("fatigue", "fatigue", "PRESENT")])
        assoc = render_deterministic_note(g)["note"] \
            .split("3. Associated Symptoms")[1].split("4.")[0]
        assert "Respiratory: cough, congestion" in assoc
        assert "Constitutional: fatigue" in assoc
        assert "caused" not in assoc.lower()


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_every_fact_sentence_traceable(self):
        g = _graph([
            _fact("fever", "fever", "RESOLVED", duration="6 days",
                  surface="പനി മാറിട്ടും", clause="പനി മാറിട്ടും"),
            _fact("cough", "cough", "PRESENT", surface="ചൊമ്മയുണ്ട്",
                  clause="ചൊമ്മയുണ്ട്"),
        ])
        out = render_deterministic_note(g)
        assert out["provenance"], "provenance trace must not be empty"
        for p in out["provenance"]:
            assert p["concept"] and p["source_text"] and p["source_clause"]
            assert p["speaker"] and p["confidence"] is not None

    def test_raw_transcript_not_in_clinician_note(self):
        g = _graph([_fact("cough", "cough", "PRESENT")])
        out = render_deterministic_note(g)
        assert "RAW ASR TEXT" not in out["note"]     # debug only
        assert g.raw_text == "RAW ASR TEXT"          # but preserved in the graph


# ---------------------------------------------------------------------------
# Fail-closed validator
# ---------------------------------------------------------------------------

class TestFailClosedValidator:
    def setup_method(self):
        self.facts = [
            _fact("fever", "fever", "RESOLVED", duration="6 days",
                  surface="പനി മാറിട്ടും", clause="പനി മാറിട്ടും"),
            _fact("cough", "cough", "PRESENT", surface="ചൊമ്മയുണ്ട്",
                  clause="ചൊമ്മയുണ്ട്"),
        ]
        self.out = render_deterministic_note(_graph(self.facts))
        assert self.out["validation"]["valid"]

    def _validate(self, note):
        return validate_note_against_fact_graph(note, _graph(self.facts))

    def test_hallucinated_diagnosis_rejected(self):
        v = self._validate(self.out["note"].replace(
            "No definitive diagnosis documented in this encounter.",
            "Assessment: This is a viral URI."))
        assert not v["valid"]

    def test_hallucinated_medication_rejected(self):
        v = self._validate(self.out["note"] + "\n   - paracetamol 500mg.")
        assert not v["valid"]
        assert any("paracetamol" in x for x in v["violations"])

    def test_hallucinated_exam_rejected(self):
        v = self._validate(self.out["note"].replace(
            "12. Physical Examination\n   Not documented.",
            "12. Physical Examination\n   - Lungs clear to auscultation."))
        assert not v["valid"]

    def test_hallucinated_plan_rejected(self):
        v = self._validate(self.out["note"].replace(
            "14. Plan\n   Not documented.",
            "14. Plan\n   - Investigation advised."))
        assert not v["valid"]

    def test_unsupported_negative_rejected(self):
        v = self._validate(self.out["note"] + "\n   Denies chest pain.")
        assert not v["valid"]

    def test_unsupported_duration_rejected(self):
        v = self._validate(self.out["note"].replace(
            "reports cough", "reports cough for three weeks"))
        assert not v["valid"]
        assert any("duration" in x for x in v["violations"])

    def test_unsupported_severity_rejected(self):
        v = self._validate(self.out["note"].replace(
            "reports cough", "reports severe cough"))
        assert not v["valid"]
        assert any("severity" in x for x in v["violations"])

    def test_causal_claim_rejected(self):
        v = self._validate(self.out["note"].replace(
            "1. Chief Complaint", "1. Chief Complaint (cough due to fever)\n   - cough."))
        assert not v["valid"]
        assert any("causal" in x for x in v["violations"])

    def test_unsupported_persistence_rejected(self):
        """§17: an unsupported 'continues to experience X' must fail closed."""
        v = self._validate(self.out["note"].replace(
            "reports cough", "continues to experience cough"))
        assert not v["valid"]
        assert any("persistence" in x for x in v["violations"])

    def test_supported_persistence_accepted(self):
        g = _graph([_fact("fatigue", "fatigue", "PRESENT", ongoing=True)])
        note = render_deterministic_note(g)["note"]
        v = validate_note_against_fact_graph(note, g)
        assert v["valid"], v["violations"]
        assert "continues to experience fatigue" in note

    def test_unprovenanced_fact_graph_rejected(self):
        bad = build_fact_graph([
            _fact("cough", "cough", "PRESENT", surface="", clause="")],
            raw_text="R")
        v = validate_fact_graph(bad)
        assert not v["valid"]
        out = render_deterministic_note(bad)
        assert not out["validation"]["valid"]   # fail closed end-to-end

    def test_deterministic_note_always_passes(self):
        # The baseline renderer must never emit an unsupported fact of its own.
        g = _graph(self.facts + [
            _fact("fatigue", "fatigue", "UNCERTAIN", uncertainty="unsure"),
            _fact("fever2", "headache", "ABSENT"),
            _fact("diabetes_history", "diabetes", "PRESENT"),
            _fact("hypertension_history", "hypertension", "ABSENT"),
            _fact("paracetamol", "paracetamol", "PRESENT",
                  attrs={"frequency": "twice daily"}),
            _fact("allergy", "allergy", "ABSENT"),
        ], turns=[{"speaker": "Doctor", "text": "Lungs are clear. This is a viral fever. I will advise rest."},
                  {"speaker": "Patient", "text": "okay doctor"}])
        out = render_deterministic_note(g)
        assert out["validation"]["valid"], out["validation"]["violations"]


# ---------------------------------------------------------------------------
# LLM renderer interface (disabled by default)
# ---------------------------------------------------------------------------

class TestLLMRendererDisabled:
    def test_disabled_by_default_falls_back(self):
        g = _graph([_fact("cough", "cough", "PRESENT")])
        det = render_deterministic_note(g)
        llm = render_llm_note(g)
        assert llm["llm_used"] is False
        assert llm["note"] == det["note"]

    def test_mandated_prompt_contract(self):
        for required in ("complete source of truth", "Do not add clinical facts",
                         "Do not infer diagnoses", "Do not infer missing history",
                         "Do not convert NOT_DOCUMENTED into ABSENT",
                         "Do not invent examination findings or treatment plans"):
            assert required in LLM_SYSTEM_PROMPT

    def test_llm_output_must_pass_validator(self):
        g = _graph([_fact("cough", "cough", "PRESENT")])

        def bad_llm(system, payload):
            return "Assessment: This is pneumonia. Plan: start amoxicillin."

        llm = render_llm_note(g, llm_call=bad_llm)
        assert llm["llm_used"] is False
        assert llm["llm_rejected"]      # violations recorded, deterministic note returned

        def good_llm(system, payload):
            return render_deterministic_note(g)["note"]

        llm2 = render_llm_note(g, llm_call=good_llm)
        assert llm2["llm_used"] is True


# ---------------------------------------------------------------------------
# Real consultation fact set (the verified six)
# ---------------------------------------------------------------------------

class TestRealConsultationFacts:
    def _real_graph(self):
        return _graph([
            _fact("fever", "fever", "RESOLVED", duration="6 days",
                  surface="പനി മാറിട്ടും",
                  clause="പനി മാറിട്ടും അങ്ങോട്ട് റെഡി ആയിട്ടില്ല",
                  confidence=0.8),
            _fact("fatigue", "fatigue", "PRESENT", surface="ഭയങ്കര ക്ഷീണവും",
                  clause="ഭയങ്കര ക്ഷീണവും", confidence=0.5, ongoing=True),
            _fact("headache", "headache", "PRESENT", surface="തലവനൊക്കെയുണ്ട്",
                  clause="തലവനൊക്കെയുണ്ട്", confidence=0.5),
            _fact("cough", "cough", "PRESENT", surface="ചൊമ്മയുണ്ട്",
                  clause="ചൊമ്മയുണ്ട്", confidence=0.5),
            _fact("phlegm", "congestion", "PRESENT", surface="കബക്കെട്ട്",
                  clause="കബക്കെട്ട്", confidence=0.5),
            _fact("difficulty_eating", "difficulty eating", "PRESENT",
                  surface="ഫുഡ് കഴിക്കാൻ പറ്റണില്ലതൊക്കെയാണ്",
                  clause="ഫുഡ് കഴിക്കാൻ പറ്റണില്ലതൊക്കെയാണ് പ്രശ്നങ്ങൾ",
                  confidence=0.85),
        ])

    def test_note_valid_and_complete(self):
        out = render_deterministic_note(self._real_graph())
        assert out["validation"]["valid"], out["validation"]["violations"]
        note = out["note"]
        assert "history of fever for approximately six days, which has since resolved" in note
        assert "continues to experience fatigue" in note
        assert "headache, cough, congestion, and difficulty eating" in note
        for section in ("Past Medical History", "Past Surgical History",
                        "Medications", "Allergies", "Family History",
                        "Social History", "Physical Examination", "Plan"):
            block = note.split(section)[1].split("\n\n")[0]
            assert "Not documented" in block, section

    def test_chronology_is_narrative_not_list(self):
        hpi = render_deterministic_note(self._real_graph())["note"] \
            .split("2. History of Present Illness")[1].split("3.")[0]
        assert hpi.index("fever") < hpi.index("fatigue")
        assert "which has since resolved" in hpi

    def test_provenance_covers_all_six(self):
        out = render_deterministic_note(self._real_graph())
        concepts = {p["concept"] for p in out["provenance"]}
        assert {"fever", "fatigue", "headache", "cough", "phlegm",
                "difficulty_eating"} <= concepts
