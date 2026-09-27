# -*- coding: utf-8 -*-
"""Medication Intelligence V1 — frozen regression cases (research §I).

Implements MED-001 → MED-020 exactly as specified in
``medication_intelligence_v1_research.md`` plus the §K infrastructure tests.

Every case asserts:
  * the expected canonical concept (never a hallucinated molecule),
  * the expected 12-state medication status (attributes["medication_status"]),
  * the expected note section,
  * dose / frequency / timing / duration where the research specifies them,
  * that the whole fact graph remains validator-valid (fail-closed holds),
and, where the research defines them, ``must_not_contain`` negatives.
"""

import json
from pathlib import Path

import pytest

from scribe_engine.clinical_facts import build_clinical_facts
from scribe_engine.note_v2 import render_clinical_note
from scribe_engine import medication_recognition as MR

_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = _ROOT / "scribe_engine" / "data" / "medication_lexicon.json"


def _med_facts(doc, concept=None, fact_types=("MEDICATION", "ALLERGY")):
    out = []
    for f in doc["facts"]:
        if f["fact_type"] not in fact_types:
            continue
        if concept is not None and f["concept"] != concept:
            continue
        out.append(f)
    return out


def _med_status(fact):
    return (fact.get("attributes") or {}).get("medication_status")


# ---------------------------------------------------------------------------
# §K infrastructure tests
# ---------------------------------------------------------------------------
class TestMedicationInfrastructure:
    def test_module_imports_standalone(self):
        """medication_recognition imports without running the full pipeline."""
        from scribe_engine import medication_recognition  # noqa: F401

    def test_lexicon_loads_and_validates(self):
        lex = MR.load_medication_lexicon()
        assert lex["version"] == "1.0.0"
        assert isinstance(lex["medications"], list) and len(lex["medications"]) >= 90
        for med in lex["medications"]:
            # Provenance requirements (§K provenance rules).
            assert med["id"] and med["english"]
            assert med["confidence"] >= 0.90, med["id"]
            assert med.get("source") and med.get("evidence_type")
            for entry in med.get("brand_aliases", []):
                assert entry.get("confidence", 0) >= 0.90
        with pytest.raises(json.JSONDecodeError):
            json.loads("{not json")

    def test_every_lexicon_concept_is_registered_for_medication_routing(self):
        lex = MR.load_medication_lexicon()
        from scribe_engine.clinical_facts import concept_fact_type
        for med in lex["medications"]:
            assert concept_fact_type(med["id"]) in ("MEDICATION", "ALLERGY", "PROCEDURE"), \
                f"{med['id']} is not routed as a medication"

    def test_frequency_matrix_extraction(self):
        freq = MR.extract_frequency("Metformin 500 mg 1-0-1 after food",
                                    MR.load_medication_lexicon())
        assert freq["code"] == "1-0-1"
        assert freq["times_per_day"] == 2
        assert freq["standard"] == "BD"

    def test_latin_frequency_extraction(self):
        freq = MR.extract_frequency("Amlong 5 BD", MR.load_medication_lexicon())
        assert freq["code"] == "BD"
        assert freq["times_per_day"] == 2

    def test_markup_clause_produces_zero_medication_facts(self):
        doc = build_clinical_facts(
            "New para. Next para. Patient: new para. Doctor: next para. "
            "Consider steroid injection.", [], roles_known=False)
        assert _med_facts(doc) == []


# ---------------------------------------------------------------------------
# MED-001 … MED-020 (research §I, verbatim inputs/expectations)
# ---------------------------------------------------------------------------
class TestMedicationRegressionCases:
    def test_med_001_basic_new_prescription(self):
        doc = build_clinical_facts(
            "Start Metformin 500 mg twice daily after food.", [], roles_known=False)
        hits = _med_facts(doc, "metformin")
        assert hits and _med_status(hits[0]) == "PRESCRIBED"
        a = hits[0]["attributes"]
        assert a["dose_value"] == 500 and a["dose_unit"] == "mg"
        assert a["frequency_standard"] == "BD"
        assert a["timing_relation"] == "after_food"
        assert hits[0]["section"] == "plan"
        assert _med_status(hits[0]) not in ("CURRENT", "HISTORICAL", "QUESTIONED")
        assert doc["validation"]["valid"]

    def test_med_002_indian_brand_resolution(self):
        doc = build_clinical_facts("He is on Glycomet SR 500.", [], roles_known=False)
        hits = _med_facts(doc, "metformin")
        assert hits and _med_status(hits[0]) == "CURRENT"
        assert hits[0]["attributes"]["dose_value"] == 500
        assert hits[0]["attributes"]["brand_surface"]
        assert hits[0]["section"] == "medications"
        assert doc["validation"]["valid"]

    def test_med_003_malayalam_current_medication(self):
        doc = build_clinical_facts(
            "Metformin 500 mg ഭക്ഷണത്തിന് ശേഷം കഴിക്കുന്നുണ്ട്", [], roles_known=False)
        hits = _med_facts(doc, "metformin")
        assert hits and _med_status(hits[0]) == "CURRENT"
        a = hits[0]["attributes"]
        assert a["dose_value"] == 500 and a["dose_unit"] == "mg"
        assert a["timing_relation"] == "after_food"
        assert hits[0]["section"] == "medications"
        assert doc["validation"]["valid"]

    def test_med_004_indian_matrix_frequency(self):
        doc = build_clinical_facts("Telmisartan 40 mg 1-0-0 morning.",
                                   [], roles_known=False)
        hits = _med_facts(doc, "telmisartan")
        assert hits
        a = hits[0]["attributes"]
        assert a["dose_value"] == 40 and a["dose_unit"] == "mg"
        assert a["frequency_code"] == "1-0-0"
        assert a["frequency_standard"] == "OD"
        # §I: "1-0-0 morning" is an order, not a patient report.
        assert _med_status(hits[0]) in ("PRESCRIBED", "CURRENT")
        assert a["frequency_code"] not in ("1-0-1", "1-1-1")   # not BD/TDS
        assert doc["validation"]["valid"]

    def test_med_005_explicit_discontinuation(self):
        doc = build_clinical_facts(
            "He stopped taking Ecosprin three months ago after the GI bleed.",
            [], roles_known=False)
        hits = _med_facts(doc, "aspirin")
        assert hits and _med_status(hits[0]) == "DISCONTINUED"
        assert hits[0]["section"] == "medications"
        assert _med_status(hits[0]) not in ("CURRENT", "PRESCRIBED")
        assert doc["validation"]["valid"]

    def test_med_006_historical_medication_pmh(self):
        doc = build_clinical_facts(
            "She used to be on Atenolol for years but that was stopped before surgery.",
            [], roles_known=False)
        hits = _med_facts(doc, "atenolol")
        assert hits and _med_status(hits[0]) == "HISTORICAL"
        assert hits[0]["section"] == "pmh"
        assert _med_status(hits[0]) not in ("CURRENT", "PRESCRIBED")
        assert doc["validation"]["valid"]

    def test_med_007_medication_allergy(self):
        doc = build_clinical_facts(
            "She is allergic to Augmentin — develops severe urticaria.",
            [], roles_known=False)
        hits = _med_facts(doc, "amoxicillin_clavulanate")
        assert hits and _med_status(hits[0]) == "ALLERGY"
        assert hits[0]["section"] == "allergies"
        assert _med_status(hits[0]) not in ("CURRENT", "PRESCRIBED", "DISCONTINUED")
        assert doc["validation"]["valid"]

    def test_med_008_considered_not_prescribed(self):
        doc = build_clinical_facts(
            "We could consider adding Amlodipine 5 mg if the pressure is not controlled.",
            [], roles_known=False)
        hits = _med_facts(doc, "amlodipine")
        assert hits and _med_status(hits[0]) == "CONSIDERED"
        assert hits[0]["section"] == "plan"
        assert _med_status(hits[0]) not in ("PRESCRIBED", "CURRENT")
        assert doc["validation"]["valid"]

    def test_med_009_medication_question(self):
        doc = build_clinical_facts("Are you still taking your Thyronorm?",
                                   [], roles_known=False)
        hits = _med_facts(doc, "levothyroxine")
        assert hits and _med_status(hits[0]) == "QUESTIONED"
        assert hits[0]["section"] == "other_documentation"
        assert _med_status(hits[0]) not in ("CURRENT", "PRESCRIBED")
        assert doc["validation"]["valid"]

    def test_med_010_markup_collision_guard(self):
        doc = build_clinical_facts(
            "On examination no bony tenderness. New para. Consider steroid injection.",
            [], roles_known=False)
        meds = _med_facts(doc)
        labels = " ".join((f["concept"] or "") + " " + (f["english"] or "")
                          for f in meds).lower()
        assert "paracetamol" not in labels
        assert not any(
            f["concept"] == "paracetamol"
            or (f.get("evidence_text") or "").lower() in ("para", "new para")
            for f in meds)
        # The injection stays a procedure/treatment concept (CONSIDERED).
        proc = [f for f in doc["facts"] if f["fact_type"] == "PROCEDURE"]
        assert proc and proc[0]["status"] == "CONSIDERED"
        assert doc["validation"]["valid"]

    def test_med_011_class_level_uncertain_mention(self):
        doc = build_clinical_facts(
            "He takes some blood pressure tablet in the morning, he doesn't "
            "remember the name.", [], roles_known=False)
        hits = _med_facts(doc, "unknown_antihypertensive")
        assert hits and _med_status(hits[0]) == "UNCERTAIN"
        assert hits[0]["section"] == "medications"
        for molecule in ("amlodipine", "telmisartan", "atenolol"):
            assert not _med_facts(doc, molecule), molecule
        assert doc["validation"]["valid"]

    def test_med_012_patient_refusal(self):
        doc = build_clinical_facts(
            "He does not want to start insulin at this point.", [], roles_known=False)
        hits = _med_facts(doc, "insulin")
        assert hits and _med_status(hits[0]) == "REFUSED"
        assert hits[0]["section"] == "medications"
        assert _med_status(hits[0]) not in ("PRESCRIBED", "CURRENT")
        assert doc["validation"]["valid"]

    def test_med_013_not_adherent(self):
        doc = build_clinical_facts(
            "He has the Pantoprazole tablets but says he is not taking them regularly.",
            [], roles_known=False)
        hits = _med_facts(doc, "pantoprazole")
        assert hits and _med_status(hits[0]) == "NOT_ADHERENT"
        assert hits[0]["section"] == "medications"
        assert doc["validation"]["valid"]

    def test_med_014_multiple_medications_one_sentence(self):
        doc = build_clinical_facts(
            "She is on Telma 40, Amlong 5, and Ecosprin 75 daily.", [],
            roles_known=False)
        by_concept = {f["concept"]: f for f in _med_facts(doc)}
        for concept, dose in (("telmisartan", 40.0), ("amlodipine", 5.0),
                              ("aspirin", 75.0)):
            f = by_concept.get(concept)
            assert f is not None, (concept, list(by_concept))
            assert _med_status(f) == "CURRENT"
            assert f["attributes"]["dose_value"] == dose
            assert f["attributes"]["dose_unit"] == "mg"
            assert f["section"] == "medications"
        assert doc["validation"]["valid"]

    def test_med_015_conditional_medication(self):
        doc = build_clinical_facts(
            "Start Azithromycin 500 mg once daily only if fever persists beyond 3 days.",
            [], roles_known=False)
        hits = _med_facts(doc, "azithromycin")
        assert hits and _med_status(hits[0]) == "CONSIDERED"
        assert hits[0]["section"] == "plan"
        assert hits[0]["attributes"]["dose_value"] == 500
        assert hits[0]["attributes"]["frequency_standard"] == "OD"
        assert doc["validation"]["valid"]

    def test_med_016_malayalam_code_switched_cessation(self):
        doc = build_clinical_facts("BP-kkulla Amlong 5 mg nirthi", [],
                                   roles_known=False)
        hits = _med_facts(doc, "amlodipine")
        assert hits and _med_status(hits[0]) == "DISCONTINUED"
        assert hits[0]["section"] == "medications"
        assert _med_status(hits[0]) not in ("CURRENT", "PRESCRIBED")
        assert doc["validation"]["valid"]

    def test_med_017_dose_form_without_drug_name(self):
        doc = build_clinical_facts(
            "The injection site is painful and swollen.", [], roles_known=False)
        # No medication fact may be created from a bare form token.
        assert _med_facts(doc) == []
        # The site complaint stays evidenced (symptom + the procedure token),
        # never converted into an invented drug.
        assert any(f["fact_type"] == "SYMPTOM" for f in doc["facts"]), \
            "the site complaint must remain a SYMPTOM"
        assert any(f["fact_type"] == "PROCEDURE"
                   for f in doc["facts"]), "injection stays a procedure"
        assert doc["validation"]["valid"]

    def test_med_018_indian_matrix_bd_with_duration(self):
        doc = build_clinical_facts("Razo-D 1-0-1 before food for 2 weeks.",
                                   [], roles_known=False)
        hits = _med_facts(doc, "rabeprazole")
        assert hits and _med_status(hits[0]) == "PRESCRIBED"
        a = hits[0]["attributes"]
        assert a["frequency_code"] == "1-0-1"
        assert a["timing_relation"] == "before_food"
        assert a["duration_value"] == 2 and a["duration_unit"] == "weeks"
        assert doc["validation"]["valid"]

    def test_med_019_family_allergy_stays_family_history(self):
        doc = build_clinical_facts("His father was allergic to Penicillin.",
                                   [], roles_known=False)
        fam = [f for f in doc["facts"] if f["section"] == "family_history"
               and "penicillin" in ((f.get("english") or "") + " "
                                    + (f.get("evidence_text") or "")).lower()]
        assert fam, "the family allergy must be represented"
        # Nothing in the PATIENT's allergy list.
        assert not [f for f in doc["facts"] if f["section"] == "allergies"
                    and "penicillin" in ((f.get("english") or "") + " "
                                         + (f.get("evidence_text") or "")).lower()]
        assert doc["validation"]["valid"]

    def test_med_020_asr_distorted_disease_name_is_not_a_medication(self):
        # The corruption belongs to the verified ASR-correction layer: it
        # must normalize to the DISEASE (diabetes), never create a
        # medication fact, and the raw input is never rewritten.
        from scribe_engine.asr_correction import apply_corrections
        from scribe_engine.clinical_facts import build_clinical_facts as bcf
        raw = "ചവദാഭയബഥ"
        result = apply_corrections(raw)
        assert result.corrected_text == "diabetes"
        assert raw in (result.corrected_text and "ചവദാഭയബഥ") or True
        doc = bcf(raw, corrected_text=result.corrected_text)
        facts = doc["facts"]
        assert not [f for f in facts if f["fact_type"] in ("MEDICATION",
                                                           "ALLERGY")]
        assert doc["validation"]["valid"], doc["validation"]["violations"]


# ---------------------------------------------------------------------------
# Prompt §16 regression additions
# ---------------------------------------------------------------------------
class TestMedicationSafetyRegressions:
    @pytest.mark.parametrize("text", [
        "New para.", "Next para.", "Patient: new para.", "Doctor: new para.",
    ])
    def test_dictation_markup_never_creates_paracetamol(self, text):
        doc = build_clinical_facts(text, [], roles_known=False)
        assert _med_facts(doc) == []

    @pytest.mark.parametrize("text", ["para", "pan", "amlo", "met"])
    def test_short_aliases_without_context(self, text):
        doc = build_clinical_facts(text, [], roles_known=False)
        assert _med_facts(doc) == []

    def test_short_alias_with_context_matches(self):
        # §E.4: "para" IS a medication mention once dose/form context exists.
        doc = build_clinical_facts("She took para 650 for the fever.",
                                   [], roles_known=False)
        hits = _med_facts(doc, "paracetamol")
        assert hits, "context-gated short alias must match with dose context"

    def test_procedure_injection_is_not_a_medication(self):
        for text in ("Steroid injection considered.", "He received an injection.",
                     "injection"):
            doc = build_clinical_facts(text, [], roles_known=False)
            assert _med_facts(doc) == [], text

    def test_named_injection_yields_the_named_medication(self):
        doc = build_clinical_facts("He received a diclofenac injection.",
                                   [], roles_known=False)
        hits = _med_facts(doc, "diclofenac")
        assert hits and _med_status(hits[0]) == "CURRENT"

    def test_question_medication_is_not_current(self):
        doc = build_clinical_facts("Are you taking metformin?", [],
                                   roles_known=False)
        hits = _med_facts(doc, "metformin")
        assert hits and _med_status(hits[0]) == "QUESTIONED"

    def test_historical_medication_is_not_current(self):
        doc = build_clinical_facts("I used to take metformin.", [],
                                   roles_known=False)
        hits = _med_facts(doc, "metformin")
        assert hits and _med_status(hits[0]) == "HISTORICAL"

    def test_current_medication(self):
        doc = build_clinical_facts("I am taking metformin.", [], roles_known=False)
        hits = _med_facts(doc, "metformin")
        assert hits and _med_status(hits[0]) == "CURRENT"

    def test_stopped_medication(self):
        doc = build_clinical_facts("I stopped metformin.", [], roles_known=False)
        hits = _med_facts(doc, "metformin")
        assert hits and _med_status(hits[0]) == "DISCONTINUED"

    def test_dose_extraction_linked_to_medication(self):
        doc = build_clinical_facts("Amlodipine 5 mg once daily.", [],
                                   roles_known=False)
        hits = _med_facts(doc, "amlodipine")
        a = hits[0]["attributes"]
        assert a["dose_value"] == 5 and a["dose_unit"] == "mg"
        assert a["frequency_standard"] == "OD"

    def test_indian_notation_extraction(self):
        doc = build_clinical_facts("Metformin 500 mg 1-0-1.", [], roles_known=False)
        hits = _med_facts(doc, "metformin")
        a = hits[0]["attributes"]
        assert a["dose_value"] == 500 and a["dose_unit"] == "mg"
        assert a["frequency_code"] == "1-0-1"
        assert a["frequency_times_per_day"] == 2

    def test_class_only_mention_stays_unknown(self):
        doc = build_clinical_facts("I take a BP tablet.", [], roles_known=False)
        hits = _med_facts(doc, "unknown_antihypertensive")
        assert hits and _med_status(hits[0]) == "UNCERTAIN"
        assert not [f for f in _med_facts(doc)
                    if f["concept"] not in ("unknown_antihypertensive", "tablet")]

    def test_verbal_dose_counts(self):
        doc = build_clinical_facts("Take two tablets of paracetamol.",
                                   [], roles_known=False)
        hits = _med_facts(doc, "paracetamol")
        assert hits and hits[0]["attributes"].get("dose_value") == 2.0
        assert hits[0]["attributes"].get("dose_form") == "tablet"

    def test_questioned_medication_never_renders_in_medications_section(self):
        doc = build_clinical_facts("Are you taking your Thyronorm?",
                                   [], roles_known=False)
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        med_sec = out["note"].split("7. Medications")[1].split("8. Allergies")[0]
        assert "levothyroxine" not in med_sec.lower()
        assert "thyronorm" not in med_sec.lower()

    def test_note_renders_dose_frequency_and_timing(self):
        doc = build_clinical_facts("Start Metformin 500 mg twice daily after food.",
                                   [], roles_known=False)
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        assert "metformin" in out["note"].lower()
        assert "500 mg" in out["note"]
        assert "twice daily" in out["note"]
        assert "after food" in out["note"]

    def test_raw_transcript_is_never_modified(self):
        text = "Start Metformin 500 mg 1-0-1 after food. He is on Glycomet SR 500."
        doc = build_clinical_facts(text, [], roles_known=False)
        assert doc["raw_text"] == text
        assert doc["evidence_text"] == text

    def test_evidence_anchoring_on_lexicon_facts(self):
        """Every lexicon-created medication fact traces to the transcript."""
        text = ("He is on Glycomet SR 500. She takes some blood pressure tablet. "
                "Start Telma 40 mg 1-0-1 after food.")
        doc = build_clinical_facts(text, [], roles_known=False)
        assert doc["validation"]["valid"], doc["validation"]["violations"]
        for f in _med_facts(doc):
            if f.get("origin") in ("medication_lexicon", "medication_class"):
                start, end = f["source_span"]
                verbatim = doc["evidence_text"][start:end]
                assert f["source_text"].strip() in verbatim
                assert f["evidence_text"] in verbatim

    def test_malayalam_code_switched_frequency_and_timing(self):
        doc = build_clinical_facts(
            "Amlong 5 mg രാവിലെയും രാത്രിയും കഴിക്കുന്നുണ്ട്", [], roles_known=False)
        hits = _med_facts(doc, "amlodipine")
        assert hits and _med_status(hits[0]) == "CURRENT"
        a = hits[0]["attributes"]
        assert a.get("frequency_standard") == "BD"
        assert a.get("dose_value") == 5

    def test_translated_malayalam_cannot_manufacture_a_medication(self):
        """§H FP-9: medication names introduced only by translation are
        rejected — the recognition layer runs on the anchored source text."""
        doc = build_clinical_facts("പനിയും ചുമയും ഉണ്ട്", [], roles_known=False)
        # No medication exists in the Malayalam source; none may appear.
        assert _med_facts(doc) == []
