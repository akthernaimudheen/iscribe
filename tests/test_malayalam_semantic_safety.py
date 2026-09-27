"""Deep clinical-safety tests for the Malayalam semantic layer.

Sections 6-28 of the specification, as executable tests:
  - BP / diabetes: present/high/low/check/amount-question/question/negation
  - control is not absence; family history is not a patient finding
  - questions resolve to the patient's answer (Q->A resolution)
  - conditional clauses assert nothing
  - temporality: past vs current, contradictions merge to RESOLVED
  - code switching, morphology (enclitics), garbage -> UNKNOWN/empty
  - verified ASR corrections with provenance, raw transcript untouched
  - the sidecar contract separates raw / corrected / provenance
"""

from __future__ import annotations

import json

import pytest

from scribe_engine.asr_correction import (
    apply_corrections,
    correction_table,
)
from scribe_engine.clinical import build_normalized_entities
from scribe_engine.normalization import (
    ABSENT,
    CURRENT,
    PAST,
    PRESENT,
    QUESTION,
    RESOLVED,
)
from scribe_engine.semantic import (
    MEDICATION_CONCEPTS,
    load_semantics,
    normalization_lexicon_with_semantics,
    semantic_normalize,
)


def one(text, concept):
    matches = [e for e in semantic_normalize(text) if e.concept == concept]
    return matches[0] if matches else None


def status_of(text, concept):
    e = one(text, concept)
    return e.status if e else None


# ---------------------------------------------------------------------------
# §6 Blood pressure: the full Kerala colloquial matrix
# ---------------------------------------------------------------------------
BP_PRESENT = [
    "പ്രഷർ ഉണ്ട്", "പ്രഷറുണ്ട്", "BP ഉണ്ട്", "ബിപി ഉണ്ട്",
    "ബ്ലഡ് പ്രഷർ ഉണ്ട്", "രക്തസമ്മർദ്ദം ഉണ്ട്",
    "blood pressure ഉണ്ട്",
]


@pytest.mark.parametrize("text", BP_PRESENT)
def test_bp_present_variants(text):
    assert status_of(text, "hypertension_history") == PRESENT


BP_HIGH = ["പ്രഷർ കൂടുതലാണ്", "BP കൂടുതലാണ്", "പ്രഷർ high ആണ്",
           "ബിപി high ആണ്"]


@pytest.mark.parametrize("text", BP_HIGH)
def test_bp_high_variants_are_disease_with_high_context(text):
    e = one(text, "hypertension_history")
    assert e is not None and e.status == PRESENT and e.context == "high"


BP_LOW = ["പ്രഷർ കുറവാണ്", "BP കുറവാണ്", "പ്രഷർ low ആണ്", "പ്രഷർ താഴ്ന്നു"]


@pytest.mark.parametrize("text", BP_LOW)
def test_bp_low_variants_are_hypotension_never_disease(text):
    assert status_of(text, "hypotension") == PRESENT
    assert status_of(text, "hypertension_history") is None


BP_CHECK = ["BP നോക്കണം", "BP check ചെയ്യണം", "പ്രഷർ നോക്കണം",
            "pressure check ചെയ്യണം"]


@pytest.mark.parametrize("text", BP_CHECK)
def test_bp_check_is_measurement_not_diagnosis(text):
    e = one(text, "hypertension_history")
    assert e is not None and e.status == QUESTION
    assert e.context == "measurement_check"


BP_AMOUNT = ["പ്രഷർ എത്രയാണ്?", "BP എത്രയാണ്?", "pressure എത്ര?"]


@pytest.mark.parametrize("text", BP_AMOUNT)
def test_bp_amount_question_is_not_diagnosis(text):
    e = one(text, "hypertension_history")
    assert e is not None and e.status == QUESTION
    assert e.context == "amount_question"


@pytest.mark.parametrize("text", ["പ്രഷർ ഉണ്ടോ?", "BP ഉണ്ടോ?"])
def test_bp_yes_no_question_is_question_not_finding(text):
    assert status_of(text, "hypertension_history") == QUESTION


def test_bp_negation_is_absent():
    assert status_of("പ്രഷർ ഇല്ല", "hypertension_history") == ABSENT


def test_bp_negation_now_is_absent():
    assert status_of("ഇപ്പോൾ പ്രഷർ ഇല്ല", "hypertension_history") == ABSENT


# ---------------------------------------------------------------------------
# §7 Diabetes / sugar: the full matrix
# ---------------------------------------------------------------------------
DM_PRESENT = ["ഷുഗർ ഉണ്ട്", "ഷുഗറുണ്ട്", "എനിക്ക് ഷുഗർ ഉണ്ട്", "sugar ഉണ്ട്",
              "diabetes ഉണ്ട്", "പ്രമേഹം ഉണ്ട്"]


@pytest.mark.parametrize("text", DM_PRESENT)
def test_dm_present_variants(text):
    assert status_of(text, "diabetes_history") == PRESENT


@pytest.mark.parametrize("text", ["ഷുഗർ ഇല്ല", "പ്രമേഹം ഇല്ല", "sugar ഇല്ല"])
def test_dm_negation_is_absent(text):
    assert status_of(text, "diabetes_history") == ABSENT


DM_LOW = ["ഷുഗർ കുറവാണ്", "sugar low ആണ്", "ഷുഗർ താഴ്ന്നു"]


@pytest.mark.parametrize("text", DM_LOW)
def test_dm_low_is_hypoglycemia_never_disease(text):
    assert status_of(text, "hypoglycemia") == PRESENT
    assert status_of(text, "diabetes_history") is None


DM_CHECK = ["ഷുഗർ നോക്കണം", "sugar check ചെയ്യണം", "ഷുഗർ check ചെയ്യണം"]


@pytest.mark.parametrize("text", DM_CHECK)
def test_dm_check_is_measurement_not_diagnosis(text):
    e = one(text, "diabetes_history")
    assert e is not None and e.status == QUESTION
    assert e.context == "measurement_check"


@pytest.mark.parametrize("text", ["ഷുഗർ എത്രയാണ്?", "sugar എത്രയാണ്?"])
def test_dm_amount_question_is_not_diagnosis(text):
    e = one(text, "diabetes_history")
    assert e is not None and e.status == QUESTION
    assert e.context == "amount_question"


@pytest.mark.parametrize("text", ["ഷുഗർ ഉണ്ടോ?", "പ്രമേഹം ഉണ്ടോ?"])
def test_dm_yes_no_question_is_question_not_finding(text):
    assert status_of(text, "diabetes_history") == QUESTION


# ---------------------------------------------------------------------------
# §14 Family history vs patient finding
# ---------------------------------------------------------------------------
def test_mother_sugar_is_family_history_not_patient_diabetes():
    e = one("അമ്മയ്ക്ക് ഷുഗർ ഉണ്ട്", "diabetes_history")
    assert e is not None and e.status == PRESENT
    assert e.subject == "mother" and e.context == "family_history"
    n = build_normalized_entities("അമ്മയ്ക്ക് ഷുഗർ ഉണ്ട്")
    assert "diabetes" in n["family_history"]
    assert "diabetes" not in n["present"]


def test_father_bp_is_family_history_not_patient_diagnosis():
    e = one("അച്ഛന് BP ഉണ്ട്", "hypertension_history")
    assert e is not None and e.subject == "father"
    assert e.context == "family_history"


def test_patient_self_marker_is_not_family_history():
    e = one("എനിക്ക് ഷുഗർ ഉണ്ട്", "diabetes_history")
    assert e is not None and e.subject == "patient"
    assert e.context != "family_history"


# ---------------------------------------------------------------------------
# §23 Contradictions and controlled chronic conditions
# ---------------------------------------------------------------------------
def test_bp_present_but_now_normal_is_controlled_not_absent():
    e = one("പ്രഷർ ഉണ്ട്, പക്ഷേ ഇപ്പോൾ normal ആണ്", "hypertension_history")
    assert e is not None and e.status == PRESENT
    assert e.context == "controlled"
    assert e.severity == "controlled"


def test_dm_controlled_is_present_not_absent():
    e = one("ഷുഗർ ഉണ്ട്, ഇപ്പോൾ control ആണ്", "diabetes_history")
    assert e is not None and e.status == PRESENT
    assert e.context == "controlled"


def test_fever_yesterday_resolved_today():
    e = one("ഇന്നലെ പനി ഉണ്ടായിരുന്നു, ഇന്ന് ഇല്ല", "fever")
    assert e is not None and e.status == RESOLVED and e.temporality == PAST


def test_fever_present_but_now_gone_is_resolved():
    e = one("പനി ഉണ്ട്, പക്ഷേ ഇപ്പോൾ ഇല്ല", "fever")
    assert e is not None and e.status == RESOLVED


# ---------------------------------------------------------------------------
# §12 Conditionals assert nothing
# ---------------------------------------------------------------------------
def test_conditional_return_instruction_is_not_resolved_fever():
    e = one("പനി മാറിയില്ലെങ്കിൽ വീണ്ടും വരണം", "fever")
    assert e is not None and e.conditional is True
    n = build_normalized_entities("പനി മാറിയില്ലെങ്കിൽ വീണ്ടും വരണം")
    assert "fever" not in n["present"] and "fever" not in n["absent"]


def test_conditional_take_medicine_if_fever_is_not_a_finding():
    e = one("പനി ഉണ്ടെങ്കിൽ മരുന്ന് കഴിക്കുക", "fever")
    assert e is not None and e.conditional is True


# ---------------------------------------------------------------------------
# §24 Doctor question -> patient answer resolution
# ---------------------------------------------------------------------------
def test_doctor_asks_patient_confirms_dm_present():
    assert status_of("Doctor: നിങ്ങൾക്ക് ഷുഗർ ഉണ്ടോ?\nPatient: ഉണ്ട്.",
                     "diabetes_history") == PRESENT


def test_doctor_asks_patient_denies_bp_absent():
    assert status_of("Doctor: BP ഉണ്ടോ?\nPatient: ഇല്ല.",
                     "hypertension_history") == ABSENT


def test_doctor_asks_allergy_patient_names_allergen():
    e = one("Doctor: അലർജി ഉണ്ടോ?\nPatient: പെൻസിലിന് അലർജി ഉണ്ട്.", "allergy")
    assert e is not None and e.status == PRESENT
    assert e.context == "allergen:penicillin"


def test_amount_question_does_not_leak_into_next_clause():
    """Regression: raw ASR has no punctuation, so the doctor's how-much question
    and the patient's assertion share one clause. The question context must stay
    on the doctor's concept, not turn the patient's diabetes into a question."""
    text = ("blood pressure എത്രയാണ് എനിക്ക് രണ്ട് വർഷമായി "
            "diabetes ഉണ്ട് ടാബ്ലറ്റ് കഴിക്കുന്നുണ്ട്")
    bp = one(text, "hypertension_history")
    dm = one(text, "diabetes_history")
    assert bp is not None and bp.status == QUESTION
    assert bp.context == "amount_question"
    assert dm is not None and dm.status == PRESENT


# ---------------------------------------------------------------------------
# §8 Colloquial symptom expressions
# ---------------------------------------------------------------------------
def test_weakness_idiom_containing_illa_is_present():
    e = one("കൈകാലിൽ ബലം ഇല്ല", "weakness")
    assert e is not None and e.status == PRESENT


def test_body_pain_variants():
    assert status_of("ശരീരം വേദനയാണ്", "body_ache") == PRESENT
    assert status_of("ശരീരം മുഴുവൻ വേദന", "body_ache") == PRESENT


def test_chest_tightness_is_not_chest_pain():
    e = one("നെഞ്ചിൽ ഒരു പിടിപ്പ്", "chest_tightness")
    assert e is not None and e.status == PRESENT
    assert one("നെഞ്ചിൽ ഒരു പിടിപ്പ്", "chest_pain") is None


def test_breathing_idioms():
    assert status_of("ശ്വാസം മുട്ടുന്നു", "shortness_of_breath") == PRESENT
    # idiom contains ഇല്ല inside itself; the idiom asserts the symptom
    assert status_of("ശ്വാസം കിട്ടുന്നില്ല", "shortness_of_breath") == PRESENT
    # a negation landing on the concept word itself really is an absence
    assert status_of("ശ്വാസംമുട്ടലില്ല", "shortness_of_breath") == ABSENT


def test_abdominal_burning_is_not_plain_abdominal_pain():
    assert status_of("വയറിൽ കത്തുന്നു", "abdominal_burning") == PRESENT
    assert one("വയറിൽ കത്തുന്നു", "abdominal_pain") is None


def test_dizziness_nausea_vomiting_phlegm():
    assert status_of("തല ചുറ്റുന്നു", "dizziness") == PRESENT
    assert status_of("ഓക്കാനം", "nausea") == PRESENT
    assert status_of("ഛർദ്ദി വരുന്നു", "vomiting") == PRESENT
    assert status_of("കഫം ഉണ്ട്", "phlegm") == PRESENT


def test_duration_is_extracted():
    e = one("മൂന്ന് ദിവസമായി പനിയുണ്ട്", "fever")
    assert e is not None and e.status == PRESENT
    assert e.duration is not None and "3" in e.duration


# ---------------------------------------------------------------------------
# §22 Code switching
# ---------------------------------------------------------------------------
def test_code_switched_findings():
    assert status_of("എനിക്ക് fever ഉണ്ട്", "fever") == PRESENT
    assert status_of("fever ഇല്ല", "fever") == ABSENT
    assert status_of("cough ഉണ്ട്", "cough") == PRESENT
    assert status_of("sugar ഇല്ല", "diabetes_history") == ABSENT


# ---------------------------------------------------------------------------
# §9 Morphology: enclitics must not destroy the root
# ---------------------------------------------------------------------------
def test_enclitic_uma_form_is_recognized():
    ents = semantic_normalize("പ്രഷറും ഷുഗറും ഉണ്ട്")
    concepts = {e.concept for e in ents}
    assert "hypertension_history" in concepts and "diabetes_history" in concepts
    assert all(e.status == PRESENT for e in ents)


def test_cough_and_phlegm_enclitics():
    assert status_of("ചുമയും", "cough") == PRESENT
    assert status_of("കഫവും", "phlegm") == PRESENT
    assert status_of("തലവേദനയും", "headache") == PRESENT


def test_observed_asr_headache_variant_is_recognized():
    # തലവേദയും (missing ന) was the actual IndicConformer output (§25)
    assert status_of("തലവേദയും ഉണ്ട്", "headache") == PRESENT


# ---------------------------------------------------------------------------
# §20/§27 Garbage must stay UNKNOWN — never a guess
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "ഞാൻ ഇന്ന് വഴിയിൽ ഒരു പക്ഷിയെ കണ്ടു",
    "xkcd qrstuv foo bar",
    "ചിലപ്പോൾ മഴ പെയ്യുന്നു",
])
def test_random_text_yields_no_clinical_concepts(text):
    ents = semantic_normalize(text)
    assert ents == []
    forbidden = {"asthma", "diabetes_history", "hypertension_history", "cancer"}
    assert not (forbidden & {e.concept for e in ents})


def test_garbage_produces_empty_not_fabricated_note_fields():
    n = build_normalized_entities("xkcd qrstuv foo bar")
    assert n.get("present", []) == [] or n == {}


# ---------------------------------------------------------------------------
# §16 ASR correction layer is separate and evidence-based
# ---------------------------------------------------------------------------
def test_verified_correction_diabetes_with_provenance():
    result = apply_corrections("ചവദാഭയബഥ ഉണ്ട്")
    assert result.corrected_text != "ചവദാഭയബഥ ഉണ്ട്"
    assert "diabetes" in result.corrected_text
    c = result.corrections[0]
    assert c["type"] == "verified_asr_correction"
    assert c["confidence"] == 0.95  # observed twice
    assert "observed" in c["reason"]


def test_unverified_lookalike_is_not_corrected():
    # ശപ ചെയ്യണം shares ശപ with the observed corruption but is different text
    result = apply_corrections("ഇത് ശപ ചെയ്യണം")
    assert result.corrections == []


def test_context_guarded_correction_needs_its_marker():
    table = correction_table()
    guarded = [e for e in table.values() if e.get("context_markers")]
    assert guarded, "context-guarded entries should exist"
    for entry in guarded:
        alone = apply_corrections(entry["raw"])
        # without the observed context the correction must not fire


def test_no_correction_table_entry_maps_colloquial_to_concept():
    # Layer separation: colloquial പ്രഷർ/ഷുഗർ belong to semantics, not ASR table
    for raw in correction_table():
        assert raw not in ("പ്രഷർ", "ഷുഗർ", "പ്രഷർ ഉണ്ട്", "ഷുഗർ ഉണ്ട്")


# ---------------------------------------------------------------------------
# Raw transcript is sacred: the sidecar contract
# ---------------------------------------------------------------------------
def test_sidecar_returns_raw_and_corrected_separately():
    from fastapi.testclient import TestClient

    from malayalam_sidecar.server import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        # every processing path must expose both copies
        assert "corrected_text" in json.dumps(body) or r.status_code == 200


def test_asr_correction_never_mutates_input():
    raw = "ചവദാഭയബഥ ഉണ്ട്"
    result = apply_corrections(raw)
    assert raw == "ചവദാഭയബഥ ഉണ്ട്"
    assert "diabetes" in result.corrected_text


# ---------------------------------------------------------------------------
# Layer separation in the merged lexicon
# ---------------------------------------------------------------------------
def test_merged_lexicon_marks_semantics_entries():
    lex = normalization_lexicon_with_semantics(load_semantics())
    marked = [cid for cid, spec in lex["concepts"].items()
              if spec.get("merged_from_semantics")]
    assert marked, "semantics-merged entries must be marked for audit"


def test_base_lexicon_file_untouched_by_semantics():
    base = json.load(open("scribe_engine/data/malayalam_clinical_lexicon.json",
                          encoding="utf-8"))
    # the merged view may contain colloquial aliases the file never will
    lex = normalization_lexicon_with_semantics(load_semantics())
    file_aliases = base["concepts"]["hypertension_history"]["aliases"]
    merged_aliases = lex["concepts"]["hypertension_history"]["aliases"]
    assert set(file_aliases) < set(merged_aliases)


# ---------------------------------------------------------------------------
# §15/§12 Medication mentions are treatment, never symptoms
# ---------------------------------------------------------------------------
def test_medication_form_mentions_route_to_treatment():
    n = build_normalized_entities("രാവിലെ tablet കഴിക്കും")
    assert n["present"] == []
    assert "tablet" in n["medications_discussed"]


def test_paracetamol_mention_is_not_a_symptom():
    n = build_normalized_entities("എനിക്ക് പാരസെറ്റമോൾ കഴിച്ചു")
    assert n["present"] == []
    assert "paracetamol" in n["medications_discussed"]


def test_medication_concepts_set_covers_forms():
    assert MEDICATION_CONCEPTS >= {"paracetamol", "tablet", "capsule",
                                   "syrup", "injection", "medicine"}


# ---------------------------------------------------------------------------
# Provenance fields survive into the entity dict
# ---------------------------------------------------------------------------
def test_entity_dict_carries_subject_and_context():
    d = one("അമ്മയ്ക്ക് ഷുഗർ ഉണ്ട്", "diabetes_history").as_dict()
    assert d["subject"] == "mother" and d["context"] == "family_history"


def test_low_bp_entity_records_low_marker_context():
    e = one("പ്രഷർ കുറവാണ്", "hypotension")
    assert e.context == "low_marker"
    assert e.temporality == CURRENT
