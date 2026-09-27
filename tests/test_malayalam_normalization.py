"""Malayalam/English clinical normalization tests.

Every case in the specification's mandatory list, plus the safety properties:
questions are not findings, negation does not leak across clauses, and nothing
is ever upgraded from a symptom to a diagnosis.
"""

from __future__ import annotations

import pytest

from scribe_engine.normalization import (
    ABSENT,
    CURRENT,
    PAST,
    POSSIBLE,
    PRESENT,
    QUESTION,
    RECURRENT,
    RESOLVED,
    UNKNOWN,
    entities_to_english_summary,
    load_lexicon,
    normalize_clinical_text,
)


def find(entities, concept):
    matches = [e for e in entities if e.concept == concept]
    return matches[0] if matches else None


def status_of(text, concept):
    e = find(normalize_clinical_text(text), concept)
    return e.status if e else None


# -- §1 core examples (mandatory) -------------------------------------------
@pytest.mark.parametrize("text,expected_status,expected_temporality", [
    ("എനിക്ക് പനി ഉണ്ട്", PRESENT, CURRENT),
    ("എനിക്ക് പനി ഇല്ല", ABSENT, CURRENT),
    ("ഇപ്പോൾ പനി ഉണ്ട്", PRESENT, CURRENT),
    ("ഇപ്പോൾ പനി ഇല്ല", ABSENT, CURRENT),
    ("ഇന്നലെ പനി ഉണ്ടായിരുന്നു", PRESENT, PAST),
    ("പനി വരാറുണ്ട്", PRESENT, RECURRENT),
    ("പനി വന്നിരുന്നു", PRESENT, PAST),
])
def test_core_fever_cases(text, expected_status, expected_temporality):
    e = find(normalize_clinical_text(text), "fever")
    assert e is not None, f"fever not detected in {text!r}"
    assert e.status == expected_status, f"{text!r} -> {e.status}"
    assert e.temporality == expected_temporality, f"{text!r} -> {e.temporality}"


def test_resolved_across_clauses():
    """ഇന്നലെ പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ ഇല്ല -> RESOLVED, one entity not two."""
    entities = normalize_clinical_text("ഇന്നലെ പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ ഇല്ല")
    fevers = [e for e in entities if e.concept == "fever"]
    assert len(fevers) == 1, f"expected one merged fever entity, got {len(fevers)}"
    assert fevers[0].status == RESOLVED


def test_duration_three_days():
    e = find(normalize_clinical_text("മൂന്ന് ദിവസമായി പനി ഉണ്ട്"), "fever")
    assert e.status == PRESENT
    assert e.duration == "3 days", e.duration
    assert e.duration_value == 3 and e.duration_unit == "days"


def test_duration_since_onset_phrasing():
    e = find(normalize_clinical_text("പനി വന്നിട്ട് മൂന്ന് ദിവസമായി"), "fever")
    assert e.status == PRESENT
    assert e.duration_value == 3


def test_question_is_not_a_finding():
    e = find(normalize_clinical_text("പനി ഉണ്ടോ?"), "fever")
    assert e.status == QUESTION, "a question must never be recorded as PRESENT"


def test_confirmation_question_is_not_present():
    e = find(normalize_clinical_text("പനി ഇല്ലല്ലോ?"), "fever")
    assert e.status == QUESTION


def test_unknown():
    assert status_of("പനി ഉണ്ടോ എന്ന് അറിയില്ല", "fever") == UNKNOWN


@pytest.mark.parametrize("text", [
    "പനി ഉണ്ടാകാം",
    "പനി ഉണ്ടാകാൻ സാധ്യതയുണ്ട്",
])
def test_possible(text):
    assert status_of(text, "fever") == POSSIBLE


def test_future_negation():
    assert status_of("പനി വരില്ല", "fever") == ABSENT


# -- §3 negation -------------------------------------------------------------
@pytest.mark.parametrize("text,concept", [
    ("പനി ഇല്ല", "fever"),
    ("പനിയില്ല", "fever"),
    ("ചുമ ഇല്ല", "cough"),
    ("ചുമയില്ല", "cough"),
    ("നെഞ്ചുവേദന ഇല്ല", "chest_pain"),
    ("ശ്വാസം മുട്ടൽ ഇല്ല", "shortness_of_breath"),
    ("കഫം ഇല്ല", "phlegm"),
])
def test_negation_forms(text, concept):
    assert status_of(text, concept) == ABSENT, text


# -- §4 negation scope (the critical safety property) ------------------------
def test_shared_negation_applies_to_both_concepts():
    entities = normalize_clinical_text("പനിയും ചുമയും ഇല്ല")
    assert find(entities, "fever").status == ABSENT
    assert find(entities, "cough").status == ABSENT


def test_negation_does_not_leak_past_connector():
    entities = normalize_clinical_text("പനി ഇല്ല, പക്ഷേ ചുമ ഉണ്ട്")
    assert find(entities, "fever").status == ABSENT
    assert find(entities, "cough").status == PRESENT, "negation leaked across പക്ഷേ"


def test_negation_scope_with_engkilum():
    entities = normalize_clinical_text("പനി ഇല്ലെങ്കിലും ചുമ ഉണ്ട്")
    assert find(entities, "fever").status == ABSENT
    assert find(entities, "cough").status == PRESENT


def test_negation_scope_two_concepts_opposite_polarity():
    entities = normalize_clinical_text("നെഞ്ചുവേദന ഇല്ല, ശ്വാസം മുട്ടൽ ഉണ്ട്")
    assert find(entities, "chest_pain").status == ABSENT
    assert find(entities, "shortness_of_breath").status == PRESENT


def test_negation_within_one_clause_binds_nearest_marker():
    entities = normalize_clinical_text("ചുമ ഉണ്ട് കഫം ഇല്ല")
    assert find(entities, "cough").status == PRESENT
    assert find(entities, "phlegm").status == ABSENT


def test_mixed_polarity_with_past_and_severity():
    entities = normalize_clinical_text(
        "വേദന ഇല്ല, പക്ഷേ ഇന്നലെ ചെറിയ തലവേദന ഉണ്ടായിരുന്നു")
    headache = find(entities, "headache")
    assert headache.status == PRESENT
    assert headache.temporality == PAST
    assert headache.severity == "mild"


# -- §6 lexicon coverage -----------------------------------------------------
@pytest.mark.parametrize("text,concept", [
    ("ചൂടുണ്ട്", "fever"),
    ("തലവേദന ഉണ്ട്", "headache"),
    ("തല ചുറ്റുന്നു", "dizziness"),
    ("ക്ഷീണം ഉണ്ട്", "fatigue"),
    ("തൊണ്ടവേദന ഉണ്ട്", "sore_throat"),
    ("ചുമ ഉണ്ട്", "cough"),
    ("കഫം ഉണ്ട്", "phlegm"),
    ("ശ്വാസം മുട്ടുന്നു", "shortness_of_breath"),
    ("നെഞ്ചുവേദന ഉണ്ട്", "chest_pain"),
    ("ഹൃദയമിടിപ്പ് കൂടുന്നു", "palpitations"),
    ("മൂക്കൊലിപ്പ് ഉണ്ട്", "runny_nose"),
    ("തുമ്മൽ ഉണ്ട്", "sneezing"),
    ("ഓക്കാനം ഉണ്ട്", "nausea"),
    ("ഛർദ്ദി ഉണ്ട്", "vomiting"),
    ("വയറിളക്കം ഉണ്ട്", "diarrhea"),
    ("മലബന്ധം ഉണ്ട്", "constipation"),
    ("വയറുവേദന ഉണ്ട്", "abdominal_pain"),
    ("നെഞ്ചെരിച്ചിൽ ഉണ്ട്", "heartburn"),
    ("ചൊറിച്ചിൽ ഉണ്ട്", "itching"),
    ("വീക്കം ഉണ്ട്", "swelling"),
    ("മരവിപ്പ് ഉണ്ട്", "numbness"),
    ("വിറയൽ ഉണ്ട്", "tremor"),
    ("കാഴ്ച മങ്ങുന്നു", "blurred_vision"),
])
def test_lexicon_recognises_concept(text, concept):
    assert find(normalize_clinical_text(text), concept) is not None, text


def test_specific_concept_beats_generic_pain():
    entities = normalize_clinical_text("നെഞ്ചുവേദന ഉണ്ട്")
    assert find(entities, "chest_pain") is not None
    assert find(entities, "pain") is None, "generic pain should not double-fire"


# -- §10 severity / §11 frequency -------------------------------------------
def test_severity_mild():
    e = find(normalize_clinical_text("ചെറിയ തലവേദന ഉണ്ട്"), "headache")
    assert e.severity == "mild"


def test_severity_severe():
    e = find(normalize_clinical_text("വളരെ കഠിനമായ വയറുവേദന ഉണ്ട്"), "abdominal_pain")
    assert e.severity == "severe"


def test_frequency_recorded():
    e = find(normalize_clinical_text("രാത്രിയിൽ ചുമ കൂടുതലാണ്"), "cough")
    assert e.frequency is not None


# -- §17 English and mixed input --------------------------------------------
@pytest.mark.parametrize("text,concept,expected", [
    ("I have fever.", "fever", PRESENT),
    ("I don't have fever.", "fever", ABSENT),
    ("I had fever yesterday.", "fever", PRESENT),
    ("I don't have any cough.", "cough", ABSENT),
    ("No chest pain.", "chest_pain", ABSENT),
    ("There's no shortness of breath.", "shortness_of_breath", ABSENT),
    ("I sometimes get fever.", "fever", PRESENT),
])
def test_english_input(text, concept, expected):
    assert status_of(text, concept) == expected, text


def test_english_past_tense_temporality():
    e = find(normalize_clinical_text("I had fever yesterday."), "fever")
    assert e.temporality == PAST


def test_english_recurrence():
    e = find(normalize_clinical_text("I sometimes get fever."), "fever")
    assert e.temporality == RECURRENT


def test_english_resolution():
    assert status_of("Fever is gone now.", "fever") == RESOLVED


def test_english_duration():
    e = find(normalize_clinical_text("I've been having cough for three days."), "cough")
    assert e.duration_value == 3 and e.duration_unit == "days"


def test_code_switched_sentence():
    """പനി ഉണ്ട് but cough ഇല്ല -> fever PRESENT, cough ABSENT."""
    entities = normalize_clinical_text("പനി ഉണ്ട് but cough ഇല്ല")
    assert find(entities, "fever").status == PRESENT
    assert find(entities, "cough").status == ABSENT


@pytest.mark.parametrize("text,concept,expected", [
    ("fever ഉണ്ട്", "fever", PRESENT),
    ("fever ഇല്ല", "fever", ABSENT),
])
def test_english_concept_malayalam_marker(text, concept, expected):
    assert status_of(text, concept) == expected, text


# -- §18 morphological variation ---------------------------------------------
@pytest.mark.parametrize("text", [
    "പനി ഉണ്ട്", "പനിയുണ്ട്", "പനി ഉണ്ടായിരുന്നു",
    "പനിയായിരുന്നു", "പനി ഉണ്ടായിട്ടുണ്ട്",
])
def test_fever_inflections_detected(text):
    e = find(normalize_clinical_text(text), "fever")
    assert e is not None and e.status in (PRESENT, RESOLVED), text


# -- §14 must NOT over-normalize (no diagnoses) ------------------------------
@pytest.mark.parametrize("text,forbidden", [
    ("പനി ഉണ്ട്", ["infection", "viral", "influenza"]),
    ("ചുമ ഉണ്ട്", ["bronchitis", "pneumonia"]),
    ("നെഞ്ചുവേദന ഉണ്ട്", ["heart attack", "angina", "myocardial"]),
    ("ശ്വാസം മുട്ടൽ ഉണ്ട്", ["asthma", "copd"]),
    ("വയറുവേദന ഉണ്ട്", ["gastritis", "ulcer", "appendicitis"]),
])
def test_never_upgrades_symptom_to_diagnosis(text, forbidden):
    entities = normalize_clinical_text(text)
    blob = " ".join(f"{e.concept} {e.english}" for e in entities).lower()
    for term in forbidden:
        assert term not in blob, f"{text!r} was over-normalized into {term!r}"


def test_lexicon_contains_no_diagnoses():
    """A lexicon entry that maps a symptom to a disease would be a silent trap."""
    lex = load_lexicon()
    banned = ["asthma", "bronchitis", "pneumonia", "heart attack", "myocardial",
              "gastritis", "appendicitis", "influenza", "covid", "tuberculosis",
              "diabetes mellitus", "angina"]
    for cid, spec in lex["concepts"].items():
        blob = (cid + " " + spec["english"]).lower()
        for term in banned:
            assert term not in blob, f"concept {cid} encodes a diagnosis"


# -- §16 raw transcript preservation ------------------------------------------
def test_raw_text_is_never_modified():
    raw = "എനിക്ക് മൂന്ന് ദിവസമായി പനി ഉണ്ട്, ചുമയും ഉണ്ട് പക്ഷേ ശ്വാസം മുട്ടൽ ഇല്ല."
    before = raw
    normalize_clinical_text(raw)
    assert raw == before, "normalization must not mutate its input"


# -- §23 final acceptance fixture ---------------------------------------------
CONSULTATION = ("ഡോക്ടറെ, എനിക്ക് മൂന്ന് ദിവസമായി പനി ഉണ്ട്. ചുമയും കഫവും ഉണ്ട്. "
                "തലവേദനയും ഉണ്ട്. പക്ഷേ ശ്വാസം മുട്ടൽ ഇല്ല.")


def test_full_consultation_acceptance():
    entities = normalize_clinical_text(CONSULTATION)

    fever = find(entities, "fever")
    assert fever.status == PRESENT
    assert fever.temporality == CURRENT
    assert fever.duration == "3 days"

    assert find(entities, "cough").status == PRESENT
    assert find(entities, "phlegm").status == PRESENT
    assert find(entities, "headache").status == PRESENT
    assert find(entities, "shortness_of_breath").status == ABSENT


def test_full_consultation_english_summary():
    summary = entities_to_english_summary(normalize_clinical_text(CONSULTATION))
    low = summary.lower()
    assert "fever" in low and "3 days" in low
    assert "cough" in low and "phlegm" in low and "headache" in low
    assert "denies" in low and "shortness of breath" in low
    # A normalized representation, never presented as the verbatim transcript.
    assert "ഡോക്ടറെ" not in summary


def test_entities_carry_traceability():
    for e in normalize_clinical_text(CONSULTATION):
        assert e.surface_text, "every entity must cite the text that produced it"
        assert e.source_clause
        assert 0.0 < e.confidence <= 1.0


# -- conditional / hypothetical clauses are not findings ---------------------
def test_conditional_clause_is_not_a_finding():
    """'come back if the fever does not settle' must not deny a present fever."""
    entities = normalize_clinical_text(
        "I have a fever and cough. Come back in one week if the fever does not settle.")
    fever = find(entities, "fever")
    assert fever.status == PRESENT, "a conditional overrode the actual complaint"
    assert fever.conditional is False


def test_conditional_flag_is_recorded():
    entities = normalize_clinical_text("Come back if the fever does not settle.")
    fever = find(entities, "fever")
    assert fever is not None and fever.conditional is True


def test_malayalam_conditional():
    entities = normalize_clinical_text("പനി ഉണ്ട്. പനി മാറിയില്ലെങ്കിൽ വരണം.")
    assert find(entities, "fever").status in (PRESENT, RESOLVED)


def test_denied_history_is_absent_not_present():
    entities = normalize_clinical_text("Do you have any allergies? No allergies doctor.")
    allergy = find(entities, "allergy")
    assert allergy is not None and allergy.status in (ABSENT, QUESTION)


def test_doctor_question_in_multi_sentence_text_is_not_a_finding():
    """The '?' must survive clause splitting, or questions become findings."""
    entities = normalize_clinical_text(
        "I have a fever. Any history of diabetes or high blood pressure? "
        "My blood pressure was high last year.")
    diabetes = find(entities, "diabetes_history")
    assert diabetes is not None
    assert diabetes.status == QUESTION, (
        f"a doctor's question became {diabetes.status}")
    assert find(entities, "fever").status == PRESENT


def test_question_mark_preserved_by_clause_split():
    from scribe_engine.normalization import load_lexicon, split_clauses

    clauses = split_clauses("I have fever. Do you have cough? No cough.", load_lexicon())
    assert any("?" in c for c in clauses), clauses


# -- pasted transcript with no speaker prefixes -------------------------------
def test_unlabelled_pasted_text_is_not_discarded():
    """Text without Doctor:/Patient: prefixes previously produced zero turns."""
    from scribe_engine import diarization

    out = diarization.build_labeled_transcript(
        [], raw_transcript_text="എനിക്ക് മൂന്ന് ദിവസമായി പനി ഉണ്ട്.")
    assert out["turns"], "the transcript was silently dropped"
    assert out["method"] == "unlabelled_text"
    assert out["roles_known"] is False
    assert "പനി" in out["turns"][0]["text"]


def test_malayalam_text_through_process_text():
    """End-to-end: Malayalam in, structured concepts out, transcript preserved."""
    from scribe_engine import ScribeEngine

    ml = ("ഡോക്ടറെ, എനിക്ക് മൂന്ന് ദിവസമായി പനി ഉണ്ട്. ചുമയും കഫവും ഉണ്ട്. "
          "പക്ഷേ ശ്വാസം മുട്ടൽ ഇല്ല.")
    result = ScribeEngine().process_text(ml, language="ml")

    assert result["validation"]["severity"] == "ok", result["validation"]
    assert "പനി" in result["transcript"]["text"], "raw Malayalam must be preserved"

    n = result["normalized_clinical_entities"]
    assert "fever" in [e["concept"] for e in n["entities"]]
    assert "shortness of breath" in n["absent"]
    assert "fever" in n["present"]


def test_english_text_without_prefixes_also_works():
    from scribe_engine import ScribeEngine

    result = ScribeEngine().process_text(
        "I have had a fever and cough for three days. No chest pain.", language="en")
    assert result["validation"]["severity"] == "ok"
    n = result["normalized_clinical_entities"]
    assert "fever" in n["present"]
    assert "chest pain" in n["absent"]


# -- ASR output has no punctuation -------------------------------------------
ASR_RUN_ON = ("എന്ത് പ്രശ്നമാണ് ഉള്ളത് എനിക്ക് മൂന്ന് ദിവസമായി പനിയും ചുമയും "
              "ഉണ്ട്ഛർദ്ദിയോ ഉണ്ടോ തലവേദയും ഉണ്ട് ഇന്നലെ ഛർദ്ദിയും ഉണ്ടായിരുന്നു "
              "പാരസിറ്റമോൾ ടാബ്ലറ്റ് രാവിലെ വൈകുന്നേരം കഴിക്കുക")


def test_unpunctuated_asr_output_splits_into_clauses():
    """Speech output has no punctuation; one clause would ruin every scope."""
    from scribe_engine.normalization import load_lexicon, split_clauses

    clauses = split_clauses(ASR_RUN_ON, load_lexicon())
    assert len(clauses) >= 4, f"run-on not segmented: {clauses}"


def test_doctor_question_in_asr_run_on_does_not_infect_other_symptoms():
    """A single 'ഉണ്ടോ' previously turned every symptom into a QUESTION."""
    entities = normalize_clinical_text(ASR_RUN_ON)
    fever = find(entities, "fever")
    cough = find(entities, "cough")
    assert fever.status == PRESENT, f"fever became {fever.status}"
    assert cough.status == PRESENT, f"cough became {cough.status}"
    assert fever.duration == "3 days"


def test_asr_run_on_keeps_past_tense_separate():
    entities = normalize_clinical_text(ASR_RUN_ON)
    vomiting = find(entities, "vomiting")
    assert vomiting.status == PRESENT
    assert vomiting.temporality == PAST


def test_verb_split_does_not_cut_inside_a_word():
    """'ഉണ്ട' is a presence marker and a prefix of 'ഉണ്ടാകാം'."""
    from scribe_engine.normalization import load_lexicon, split_clauses

    assert split_clauses("പനി ഉണ്ടാകാം", load_lexicon()) == ["പനി ഉണ്ടാകാം"]
    assert status_of("പനി ഉണ്ടാകാം", "fever") == POSSIBLE


def test_verb_split_keeps_subordinated_clause_together():
    from scribe_engine.normalization import load_lexicon, split_clauses

    assert split_clauses("പനി ഉണ്ടോ എന്ന് അറിയില്ല", load_lexicon()) == [
        "പനി ഉണ്ടോ എന്ന് അറിയില്ല"]
    assert status_of("പനി ഉണ്ടോ എന്ന് അറിയില്ല", "fever") == UNKNOWN


def test_observed_asr_spelling_variant_is_recognised():
    """ASR writes തലവേദയും (dropped ന); the concept is still unambiguous."""
    assert find(normalize_clinical_text("തലവേദയും ഉണ്ട്"), "headache") is not None


def test_garbled_loanwords_are_never_mapped_to_a_concept():
    """The ASR mangles 'blood pressure'. Guessing the term back is forbidden."""
    for garbled in ("ചവദാഭയബഥ", "ചവതാഭയപട", "ലാൻഡിലാണ്"):
        entities = normalize_clinical_text(f"{garbled} ഉണ്ട്")
        assert not entities, f"{garbled!r} was guessed into {[e.concept for e in entities]}"
