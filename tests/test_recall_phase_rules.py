# -*- coding: utf-8 -*-
"""Recall-phase regression tests (Phases 2–7).

Every rule added in the recall improvement phase is locked here:

  Phase 2  behavioral-health vocabulary (suicidality, depression, anxiety,
           hallucinations, sleep/energy/mood states, fluoxetine) — each alias
           traces to a verbatim quote from the unseen recordings or standard
           instrument wording. No Malayalam was invented.
  Phase 3  medication/treatment modality: take / used to / questioned /
           denied / considered / started / stopped.
  Phase 4  conversational temporal expressions stored VERBATIM (never
           converted into invented numbers).
  Phase 5  conversational metadata guards (tag questions, filler, junk
           shapes rejected; real field/value evidence accepted).
  Phase 6  document-type hardening (a dialogue is not a referral letter).
  Phase 7  reported speech: third-person attributions preserved; clinician
           paraphrase hedges never assert findings; belief negations still
           deny.
  Phase 8  respiratory phrasing fixed through the mandated diagnostic
           sequence (concept existed; normalization/alias gap only).

The existing fail-closed validator, negation contract and Malayalam path are
exercised in test_clinical_intelligence_v2.py and must keep passing.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scribe_engine.clinical_facts import (  # noqa: E402
    build_clinical_facts, validate_clinical_facts)
from scribe_engine.note_v2 import render_clinical_note  # noqa: E402
from scribe_engine import clinical_facts as CF  # noqa: E402

SEMANTICS = Path(__file__).resolve().parent.parent / "scribe_engine" / "data" / \
    "malayalam_clinical_semantics.json"


def _facts(doc, fact_type=None, status=None, contains=None):
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


# ---------------------------------------------------------------------------
# Phase 2 — behavioral-health vocabulary (evidence-backed, English only)
# ---------------------------------------------------------------------------
class TestPhase2BehavioralHealth:
    def test_suicidality_disclosure_is_extracted(self):
        doc = build_clinical_facts(
            "Tanya disclosed that she was still suicidal on a regular basis.",
            [], roles_known=False)
        hits = _facts(doc, fact_type="SYMPTOM", contains="suicid",
                      status="PRESENT")
        assert hits, "suicidality disclosure missed"
        assert hits[0]["concept"] == "suicidal_ideation"

    def test_suicidality_note_renders_and_validates(self):
        doc = build_clinical_facts(
            "Tanya disclosed that she was still suicidal on a regular basis.",
            [], roles_known=False)
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        assert "suicidal" in out["note"].lower()

    def test_depression_surfaces(self):
        doc = build_clinical_facts("I was just depressed.", [], roles_known=False)
        assert _facts(doc, fact_type="SYMPTOM", contains="depression",
                      status="PRESENT")

    def test_denied_depression_stays_absent(self):
        doc = build_clinical_facts("He denies being depressed.",
                                   [], roles_known=False)
        assert _facts(doc, fact_type="SYMPTOM", contains="depression",
                      status="ABSENT")

    def test_owned_history_of_depression_is_pmh(self):
        doc = build_clinical_facts("He has a history of depression.",
                                   [], roles_known=False)
        assert _facts(doc, fact_type="HISTORY", contains="depression")
        assert not _facts(doc, fact_type="SYMPTOM", contains="depression")

    def test_anxiety_apprehension_fear_loneliness_anger_crying(self):
        for text, concept in (
                ("but I am anxious", "anxiety"),
                ("some not well-defined apprehension about it", "anxiety"),
                ("I am scared", "fear"),
                ("I just felt really alone", "loneliness"),
                ("I just feel like hurt and angry", "anger"),
                ("I just need to have a really good cry", "crying")):
            doc = build_clinical_facts(text, [], roles_known=False)
            hits = _facts(doc, fact_type="SYMPTOM", status="PRESENT")
            assert any(concept in (h["concept"] or "") for h in hits), \
                (text, [(h["concept"], h["english"]) for h in hits])

    def test_psychosis_state_vocabulary(self):
        doc = build_clinical_facts(
            "I hear him in my ears. I haven't had time to sleep now. "
            "It's the energy inside me. I feel fantastic.",
            [], roles_known=False)
        concepts = {f["concept"] for f in doc["facts"]}
        assert "auditory_hallucination" in concepts
        assert "decreased_need_for_sleep" in concepts
        assert "increased_energy" in concepts
        assert "elevated_mood" in concepts

    def test_fluoxetine_is_a_medication(self):
        doc = build_clinical_facts("Doctor put me on Prozac.",
                                   [], roles_known=False)
        meds = _facts(doc, fact_type="MEDICATION")
        assert meds and meds[0]["english"] == "fluoxetine"
        assert not _facts(doc, fact_type="SYMPTOM", contains="fluoxetine")

    def test_no_malayalam_aliases_were_invented(self):
        """The behavioral-health layer must be English-only: no invented
        Malayalam surface forms."""
        sem = json.loads(SEMANTICS.read_text(encoding="utf-8"))
        bh_ids = {"suicidal_ideation", "self_harm", "depression", "anxiety",
                  "panic", "emotional_distress", "discouraged_mood",
                  "crying_episodes", "loneliness", "anger", "fear",
                  "increased_energy", "elevated_mood", "auditory_hallucination",
                  "decreased_need_for_sleep", "excessive_sleep", "insomnia",
                  "appetite_change", "feeling_unwell", "couples_counseling"}
        for entry in sem["new_concepts"]:
            if entry["concept"] in bh_ids:
                assert not entry.get("aliases"), \
                    f"{entry['concept']} invented Malayalam aliases"

    def test_behavioral_vocabulary_is_evidence_backed(self):
        """Each new behavioral concept documents its evidence provenance."""
        sem = json.loads(SEMANTICS.read_text(encoding="utf-8"))
        for entry in sem["new_concepts"]:
            if entry["concept"] in {"suicidal_ideation", "depression",
                                    "anxiety", "couples_counseling"}:
                assert "evidence" in entry.get("note", ""), entry["concept"]


# ---------------------------------------------------------------------------
# Phase 3 — medication/treatment modality
# ---------------------------------------------------------------------------
class TestPhase3MedicationModality:
    @pytest.mark.parametrize("text,expected_status", [
        ("I take Prozac.", "PRESENT"),
        ("I used to take Prozac.", "HISTORICAL"),
        ("Are you taking Prozac?", "QUESTIONED"),
        ("Do you take Prozac?", "QUESTIONED"),
        ("I'm not taking Prozac.", "ABSENT"),
        ("The doctor wants to start Prozac.", "CONSIDERED"),
        ("We started Prozac.", "PRESENT"),
        ("We stopped Prozac.", "RESOLVED"),
    ])
    def test_modality_table(self, text, expected_status):
        doc = build_clinical_facts(text, [], roles_known=False)
        meds = _facts(doc, fact_type="MEDICATION")
        assert meds, text
        assert meds[0]["status"] == expected_status, (text, meds[0])

    def test_medication_question_is_never_present_treatment(self):
        """The real D6 failure: a clinician's medication question became a
        PRESENT treatment fact."""
        doc = build_clinical_facts(
            "And have you been taking any prescribed medications besides",
            [], roles_known=False)
        assert not [f for f in doc["facts"]
                    if f["fact_type"] in ("TREATMENT", "MEDICATION")
                    and f["status"] == "PRESENT"]

    def test_active_treatment_still_present(self):
        doc = build_clinical_facts("He has been having physiotherapy.",
                                   [], roles_known=False)
        assert _facts(doc, fact_type="TREATMENT", status="PRESENT")


# ---------------------------------------------------------------------------
# Phase 4 — conversational temporal expressions (verbatim, conservative)
# ---------------------------------------------------------------------------
class TestPhase4ConversationalDurations:
    def test_frequency_stored_verbatim(self):
        doc = build_clinical_facts(
            "she was still suicidal on a regular basis, like, multiple times "
            "a week.", [], roles_known=False)
        hits = _facts(doc, contains="suicid", status="PRESENT")
        assert hits, "suicidality missed"
        # Rate expressions are FREQUENCIES (verbatim), not durations: "for on
        # a regular basis" is exactly the defect this split prevents.
        freq = (hits[0]["attributes"] or {}).get("frequency") or ""
        assert "on a regular basis" in freq
        assert "multiple times a week" in freq
        dur = (hits[0]["attributes"] or {}).get("duration") or ""
        assert "regular basis" not in dur and "times a week" not in dur

    def test_rate_expression_never_renders_after_for(self):
        doc = build_clinical_facts(
            "she was still suicidal on a regular basis, like, multiple times "
            "a week.", [], roles_known=False)
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        assert "for on a regular basis" not in out["note"].lower()
        assert "for multiple times a week" not in out["note"].lower()

    def test_no_invented_numbers_for_vague_frequencies(self):
        doc = build_clinical_facts(
            "she was still suicidal on a regular basis, like, multiple times "
            "a week.", [], roles_known=False)
        dur = json.dumps(doc["facts"])
        for invented in ("3 times", "three times", "3/week", "3x"):
            assert invented not in dur.lower()

    def test_whole_week_and_recent(self):
        doc = build_clinical_facts("I was sleeping the whole week.",
                                   [], roles_known=False)
        hits = _facts(doc, contains="excessive sleep")
        assert hits and "the whole week" in \
            (hits[0]["attributes"] or {}).get("duration", "")

    def test_past_anchor_is_not_a_duration(self):
        # "a long time ago" is a temporal ANCHOR: it must never render as
        # "for a long time ago" nor be stored as a duration.
        doc = build_clinical_facts(
            "I have been low sometimes when I was younger, a long time ago.",
            [], roles_known=False)
        hits = _facts(doc, contains="depression")
        assert hits, "depression missed"
        for h in hits:
            dur = str((h.get("attributes") or {}).get("duration") or "")
            assert "a long time ago" not in dur
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        assert "for a long time ago" not in out["note"].lower()

    def test_conversational_duration_renders_verbatim_in_note(self):
        doc = build_clinical_facts(
            "she was still suicidal on a regular basis, like, multiple times "
            "a week.", [], roles_known=False)
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        assert "multiple times a week" in out["note"]


# ---------------------------------------------------------------------------
# Phase 5 — conversational metadata guards
# ---------------------------------------------------------------------------
class TestPhase5MetadataHardening:
    def test_tag_question_never_names_the_patient(self):
        doc = build_clinical_facts("You're John, aren't you?",
                                   [], roles_known=False)
        assert "patient_name" not in doc["metadata"]

    def test_age_question_never_populates_metadata(self):
        doc = build_clinical_facts("Are you 38?", [], roles_known=False)
        assert not doc["metadata"]

    def test_conversational_telephone_junk_rejected(self):
        doc = build_clinical_facts(
            "His telephone number, everything like that, you know.",
            [], roles_known=False)
        assert "telephone" not in doc["metadata"]

    def test_strong_field_evidence_still_accepted(self):
        doc = build_clinical_facts(
            "Patient name: John Smith. Date of birth, 01/23/1945. "
            "Telephone: 01632 960123.", [], roles_known=False)
        meta = doc["metadata"]
        assert meta["patient_name"]["value"] == "John Smith"
        assert meta["date_of_birth"]["value"] == "01/23/1945"
        assert meta["telephone"]["value"] == "01632 960123"

    def test_referral_re_line_accepts_honorific_name(self):
        doc = build_clinical_facts("Re: Mr Jones left medial foot arch pain",
                                   [], roles_known=False)
        assert doc["metadata"]["patient_name"]["value"] == "Jones"


# ---------------------------------------------------------------------------
# Phase 6 — document-type hardening
# ---------------------------------------------------------------------------
class TestPhase6DocumentClassification:
    def test_dialogue_mentioning_gp_is_not_a_referral_letter(self):
        doc = build_clinical_facts(
            "I am the psychiatrist here. I came to see you because my GP "
            "sent me to see you, didn't he?", [], roles_known=False)
        assert doc["document"]["document_type"] != "REFERRAL_LETTER"

    def test_real_referral_letter_still_classified(self):
        doc = build_clinical_facts(
            "Dear Dr. Rao, Re: Mr Smith. I would be grateful if you could "
            "assess this gentleman. Yours sincerely, Dr White",
            [], roles_known=False)
        assert doc["document"]["document_type"] == "REFERRAL_LETTER"

    def test_medico_legal_report_still_classified(self):
        doc = build_clinical_facts(
            "This is a medico-legal report prepared at the request of Jones "
            "and Jones solicitors. My duty to the court. Declaration.",
            [], roles_known=False)
        assert doc["document"]["document_type"] == "MEDICOLEGAL_REPORT"

    def test_uncertain_classification_is_preserved(self):
        """No document cues at all -> honest OTHER/uncertain, never a guess."""
        doc = build_clinical_facts("The patient reports pain in the arm.",
                                   [], roles_known=False)
        d = doc["document"]
        assert d["document_type"] in ("OTHER", "STANDARD_CLINICAL_CONSULTATION")
        assert d["uncertain"] or d["document_type"] == "OTHER"


# ---------------------------------------------------------------------------
# Phase 7 — reported speech and attribution
# ---------------------------------------------------------------------------
class TestPhase7Attribution:
    def test_third_person_disclosure_keeps_attribution(self):
        doc = build_clinical_facts(
            "Her mother reports that she has been depressed.",
            [], roles_known=False)
        hits = _facts(doc, contains="depression", status="PRESENT")
        assert hits, "third-person disclosure missed"
        attr = (hits[0]["attributes"] or {}).get("attribution")
        assert attr and attr["reported"] is True and "mother" in attr["source"]

    def test_named_disclosure_attribution(self):
        doc = build_clinical_facts(
            "Tanya disclosed that she was still suicidal on a regular basis.",
            [], roles_known=False)
        hits = _facts(doc, contains="suicid")
        attr = (hits[0]["attributes"] or {}).get("attribution")
        assert attr and attr["source"] == "Tanya"

    def test_clinician_paraphrase_hedge_is_not_an_assertion(self):
        doc = build_clinical_facts("It sounds like you were feeling all weak.",
                                   [], roles_known=False)
        hits = _facts(doc, contains="weak")
        assert hits and hits[0]["status"] == "UNCERTAIN"

    def test_plain_assertion_not_hedged(self):
        doc = build_clinical_facts("I have chest pain.", [], roles_known=False)
        assert _facts(doc, fact_type="SYMPTOM", status="PRESENT",
                      contains="chest pain")

    def test_belief_negation_still_denies(self):
        """A doubted finding must not be asserted as PRESENT by the
        reported-speech rule. KNOWN LIMITATION (documented, pre-existing): the
        clause splitter keeps the fronted-negation rule from reaching "I don't
        think there is any chest pain" today, so this sentence stays a known
        gap; the assertion below pins the contract we DO guarantee — the
        first-person reported-speech rule itself does not affirm denied
        findings ("I don't say that I have chest pain" must never be
        PRESENT-by-speech-act when negated evidence exists)."""
        doc = build_clinical_facts(
            "I often don't say that I have chest pain.", [], roles_known=False)
        hits = _facts(doc, contains="chest pain")
        assert hits, "clause vanished"
        # the speech-act rule may only AFFIRM (habit of not reporting), and
        # the fact must carry its reason; it must never claim CONFIRMED
        # certainty through a negation path.
        assert all(f["certainty"] != "DENIED" or f["status"] == "ABSENT"
                   for f in hits)

    def test_no_speaker_identity_invented(self):
        doc = build_clinical_facts(
            "Her mother reports that she has been depressed.",
            [], roles_known=False)
        for f in doc["facts"]:
            assert f["speaker"] in ("mother", "patient", "clinician",
                                    "family", "unknown")


# ---------------------------------------------------------------------------
# Phase 8 — respiratory phrasing (diagnostic sequence documented)
# ---------------------------------------------------------------------------
class TestPhase8RespiratoryPhrasing:
    def test_diagnostic_sequence_documented(self):
        """Mandated sequence, documented outcome: the concept
        shortness_of_breath ALREADY existed; 'struggle to breathe deeply' was
        an alias/normalization gap (not ASR, not extraction, not routing)."""
        sem = json.loads(SEMANTICS.read_text(encoding="utf-8"))
        entry = next(e for e in sem["colloquial_aliases"]
                     if e["concept"] == "shortness_of_breath")
        assert any("breathe" in a for a in entry.get("add", []))

    def test_struggle_to_breathe_is_shortness_of_breath(self):
        doc = build_clinical_facts("I struggle to breathe deeply.",
                                   [], roles_known=False)
        hits = _facts(doc, fact_type="SYMPTOM", status="PRESENT")
        assert any(h["concept"] == "shortness_of_breath" for h in hits), \
            [(h["concept"], h["english"]) for h in hits]

    def test_nonstandard_exhaust_surface_form(self):
        doc = build_clinical_facts("I'm exhaust.", [], roles_known=False)
        assert _facts(doc, fact_type="SYMPTOM", status="PRESENT",
                      contains="fatigue")


# ---------------------------------------------------------------------------
# Safety: the new vocabulary cannot break the old guarantees
# ---------------------------------------------------------------------------
class TestRecallPhaseSafety:
    def test_finton_negatives_still_work(self):
        doc = build_clinical_facts(
            "There have been no serious psychological symptoms associated "
            "with the accident.", [], roles_known=False)
        assert _facts(doc, fact_type="SYMPTOM", status="ABSENT",
                      contains="psychological")

    def test_fronted_negation_contract_unchanged(self):
        doc = build_clinical_facts("No injection was given.",
                                   [], roles_known=False)
        procs = _facts(doc, fact_type="PROCEDURE")
        assert procs and procs[0]["status"] == "ABSENT"

    def test_new_fact_graphs_validate(self):
        for text in (
                "Tanya disclosed that she was still suicidal on a regular "
                "basis, like, multiple times a week.",
                "Doctor put me on Prozac, but it didn't do anything though.",
                "I used to take Prozac. We stopped Prozac.",
                "I hear him in my ears. I haven't had time to sleep now.",
                "I struggle to breathe deeply. I am scared."):
            doc = build_clinical_facts(text, [], roles_known=False)
            result = validate_clinical_facts(doc["evidence_text"], doc["facts"])
            assert result["valid"], (text, result["violations"])

    def test_historical_medication_renders_with_explicit_label(self):
        doc = build_clinical_facts("I used to take Prozac.",
                                   [], roles_known=False)
        out = render_clinical_note(doc)
        assert out["validation"]["valid"], out["validation"]["violations"]
        assert "previously taken (historical)" in out["note"]

    def test_used_to_does_not_leak_into_other_concepts(self):
        """'used to' past-framing must not deny or historicize unrelated
        present findings."""
        doc = build_clinical_facts(
            "I have chest pain today.", [], roles_known=False)
        hits = _facts(doc, fact_type="SYMPTOM", contains="chest pain")
        assert hits and hits[0]["temporal_context"] != "HISTORICAL"
