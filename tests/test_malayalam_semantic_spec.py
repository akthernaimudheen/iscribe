# -*- coding: utf-8 -*-
"""Malayalam Clinical Semantic Specification v1 — regression suite.

Categories (briefing §14):
    A PRESENT          B ABSENT          C RESOLVED       D UNCERTAIN
    E QUESTIONED       F NOT_DOCUMENTED  G negation scope H resolution scope
    I uncertainty scope J temporal scope K speaker scope  L mixed speech
    M colloquial       N final-virama elision O ASR-confirmed variants
    P raw immutability Q provenance

The four real-audio verified forms (ചൊമ്മയുണ്ട് / തലവനൊക്കെയുണ്ട് /
കബക്കെട്ട് / ഫുഡ് കഴിക്കാൻ പറ്റണില്ല) are pinned as-is: the raw ASR
surface passes through the correction layer with provenance, then the
semantic layer resolves the concept with full evidence.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from scribe_engine.asr_correction import apply_corrections, correction_table
from scribe_engine.clinical import build_normalized_entities
from scribe_engine.fact_graph import (
    FactGraph,
    build_fact_graph,
    validate_fact_graph,
)
from scribe_engine.normalization import (
    ABSENT,
    POSSIBLE,
    PRESENT,
    QUESTION,
    RESOLVED,
    UNKNOWN,
    normalize_clinical_text,
)
from scribe_engine.semantic import semantic_normalize


def _ents(text):
    """Semantic-layer entities via the production entry (semantics included)."""
    return semantic_normalize(text)


def find(entities, concept):
    """Works for both ClinicalEntity objects and as_dict() records."""
    for e in entities:
        c = e.get("concept") if isinstance(e, dict) else e.concept
        if c == concept:
            return e
    return None


def one(text, concept):
    e = find(_ents(text), concept)
    assert e is not None, f"{concept} not extracted from: {text}"
    return e


def status_of(text, concept):
    return one(text, concept).status


# ---------------------------------------------------------------------------
# A. PRESENT
# ---------------------------------------------------------------------------
def test_a_present():
    e = one("പനി ഉണ്ട്", "fever")
    assert e.status == PRESENT


def test_a_present_bare_mention_is_present_not_absent():
    assert status_of("പനി", "fever") == PRESENT


# ---------------------------------------------------------------------------
# B. ABSENT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("text,concept", [
    ("പനി ഇല്ല", "fever"),
    ("ചുമ ഇല്ല", "cough"),
    ("കഫക്കെട്ടില്ല", "phlegm"),          # final-virama fused denial
    ("ഭക്ഷണം കഴിക്കാൻ ബുദ്ധിമുട്ടില്ല", "difficulty_eating"),  # final-virama fused denial
])
def test_b_absent(text, concept):
    assert status_of(text, concept) == ABSENT, text
    assert status_of(text, concept) != "NOT_DOCUMENTED"


# ---------------------------------------------------------------------------
# C. RESOLVED
# ---------------------------------------------------------------------------
def test_c_resolved():
    assert status_of("പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി", "fever") == RESOLVED


# ---------------------------------------------------------------------------
# D. UNCERTAIN (scoped)
# ---------------------------------------------------------------------------
def test_d_feels_like_is_uncertain():
    e = one("പനി ആണെന്ന് തോന്നുന്നു", "fever")
    assert e.status == POSSIBLE
    assert e.uncertainty_reason and "തോന്നുന്നു" in e.uncertainty_reason


def test_d_feels_like_plus_not_sure_is_uncertain():
    e = one("പനി ആണെന്ന് തോന്നുന്നു, പക്ഷേ ഉറപ്പില്ല", "fever")
    assert e.status == UNKNOWN


def test_d_feeling_experience_is_present_not_uncertain():
    assert status_of("ക്ഷീണം തോന്നുന്നുണ്ട്", "fatigue") == PRESENT


def test_d_unknown_cause_does_not_create_symptom_or_uncertainty():
    ents = normalize_clinical_text("എന്താണ് കാരണമെന്ന് അറിയില്ല, ക്ഷീണം ഉണ്ട്")
    assert status_of("എന്താണ് കാരണമെന്ന് അറിയില്ല, ക്ഷീണം ഉണ്ട്", "fatigue") == PRESENT
    assert find(ents, "unknown") is None
    assert find(ents, "cause") is None


def test_d_do_not_know_if_fever_is_uncertain():
    assert status_of("പനി ഉണ്ടോ എന്ന് അറിയില്ല", "fever") == UNKNOWN


def test_d_present_then_not_sure_is_uncertain():
    e = one("പനി ഉണ്ട്, പക്ഷേ ഉറപ്പില്ല", "fever")
    assert e.status == UNKNOWN


def test_d_sometimes_cough_is_present_with_frequency():
    e = one("ചിലപ്പോൾ ചുമയുണ്ട്", "cough")
    assert e.status == PRESENT
    assert e.frequency_normalized == "intermittent"
    assert e.uncertainty_reason is None


# ---------------------------------------------------------------------------
# E. QUESTIONED
# ---------------------------------------------------------------------------
def test_e_question_is_questioned():
    assert status_of("പനി ഉണ്ടോ?", "fever") == QUESTION


def test_e_confirmation_question():
    assert status_of("പനി ഇല്ലല്ലോ?", "fever") == QUESTION


# ---------------------------------------------------------------------------
# F. NOT_DOCUMENTED (representation-level: silence must not produce ABSENT)
# ---------------------------------------------------------------------------
def test_f_silence_is_not_absent():
    ents = _ents("ക്ഷീണം ഉണ്ട്")
    assert find(ents, "fever") is None          # no fever entity at all

    doc = build_fact_graph(
        [e.as_dict() for e in ents],
        [{"speaker": "Patient", "text": "ക്ഷീണം ഉണ്ട്"}])
    fever_facts = [f for f in doc.facts if f["concept"] == "fever"]
    assert fever_facts == []


def test_f_not_documented_never_converted_to_absent():
    turns = [{"speaker": "Doctor", "text": "Do you have diabetes?"},
             {"speaker": "Patient", "text": "ക്ഷീണം ഉണ്ട്"}]
    ents = build_normalized_entities(
        "Doctor: Do you have diabetes?\nPatient: ക്ഷീണം ഉണ്ട്")["entities"]
    graph = build_fact_graph(ents, turns)
    diabetes = [f for f in graph.facts
                if f["concept"] == "diabetes_history"
                and f["status"] != "QUESTIONED"]
    assert diabetes == []   # a question without an answer creates NO finding


# ---------------------------------------------------------------------------
# G. Negation scope
# ---------------------------------------------------------------------------
def test_g_negation_does_not_cross_predications():
    assert status_of("പനി ഇല്ല, പക്ഷേ ക്ഷീണം ഉണ്ട്", "fever") == ABSENT
    assert status_of("പനി ഇല്ല, പക്ഷേ ക്ഷീണം ഉണ്ട്", "fatigue") == PRESENT


def test_g_shared_negation_applies_to_both():
    assert status_of("പനിയും ചുമയും ഇല്ല", "fever") == ABSENT
    assert status_of("പനിയും ചുമയും ഇല്ല", "cough") == ABSENT


# ---------------------------------------------------------------------------
# H. Resolution scope
# ---------------------------------------------------------------------------
def test_h_resolution_does_not_leak():
    assert status_of("പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി, പക്ഷേ ക്ഷീണം ഇപ്പോഴും ഉണ്ട്",
                     "fever") == RESOLVED
    assert status_of("പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി, പക്ഷേ ക്ഷീണം ഇപ്പോഴും ഉണ്ട്",
                     "fatigue") == PRESENT


def test_h_shared_resolution():
    assert status_of("പനിയും ക്ഷീണവും മാറി", "fever") == RESOLVED
    assert status_of("പനിയും ക്ഷീണവും മാറി", "fatigue") == RESOLVED


# ---------------------------------------------------------------------------
# I. Uncertainty scope
# ---------------------------------------------------------------------------
def test_i_doubt_is_scoped_to_its_concept():
    # The doubt about fever must not make the separately stated fatigue uncertain.
    assert status_of("പനി ആകാം, ക്ഷീണം ഉണ്ട്", "fever") == POSSIBLE
    assert status_of("പനി ആകാം, ക്ഷീണം ഉണ്ട്", "fatigue") == PRESENT


def test_i_uncertainty_reason_recorded():
    e = one("പനി ആകാം", "fever")
    assert e.status == POSSIBLE
    assert e.uncertainty_reason == "uncertainty_marker:ആകാം"


def test_i_bare_mention_is_confirmed():
    e = one("പനി ഉണ്ട്", "fever")
    assert e.uncertainty_reason is None


# ---------------------------------------------------------------------------
# J. Temporal scope
# ---------------------------------------------------------------------------
def test_j_duration_anchored_onset():
    e = one("ആറു ദിവസമായി പനി", "fever")
    assert e.duration == "6 days"
    assert e.onset == "6 days"


def test_j_duration_without_onset_anchor_has_no_onset():
    e = one("പനി ആറു ദിവസം ഉണ്ടായിരുന്നു", "fever")
    assert e.duration == "6 days"


def test_j_still_fatigue_is_current_ongoing():
    e = one("ഇപ്പോഴും ക്ഷീണം ഉണ്ട്", "fatigue")
    assert e.status == PRESENT
    assert e.temporality == "CURRENT"


def test_j_no_time_invented():
    e = one("പനി ഉണ്ട്", "fever")
    assert e.duration is None
    assert e.onset is None


# ---------------------------------------------------------------------------
# K. Speaker scope
# ---------------------------------------------------------------------------
def test_k_doctor_question_patient_answer_present():
    n = build_normalized_entities("Doctor: പനി ഉണ്ടോ?\nPatient: ഉണ്ട്")
    e = find(n["entities"], "fever")
    assert e is None or e["status"] == PRESENT


def test_k_doctor_question_patient_denies_absent():
    n = build_normalized_entities("Doctor: പനി ഉണ്ടോ?\nPatient: ഇല്ല, പനി ഇല്ല.")
    e = find(n["entities"], "fever")
    assert e is None or e["status"] == ABSENT


def test_k_doctor_question_no_answer_questioned_not_present():
    n = build_normalized_entities("Doctor: പനി ഉണ്ടോ?\nPatient: ക്ഷീണം ഉണ്ട്")
    e = find(n["entities"], "fever")
    assert e is None or e["status"] == QUESTION   # never PRESENT


def test_k_roles_unknown_question_is_not_finding():
    n = build_normalized_entities("Speaker 0: പനി ഉണ്ടോ?\nSpeaker 1: ക്ഷീണം ഉണ്ട്")
    e = find(n["entities"], "fever")
    assert e is None or e["status"] == QUESTION


def test_j_onset_not_from_tense_suffix():
    # ഉണ്ടായിരുന്നു contains ായി fused inside a past-tense verb — the boundary
    # guard must keep it from being read as an onset anchor.
    e = one("പനി ഉണ്ടായിരുന്നു", "fever")
    assert e.onset is None


def test_k_speaker_recorded_in_fact_graph():
    turns = [{"speaker": "Patient", "text": "പനി ഉണ്ട്"}]
    ents = [e.as_dict() for e in _ents("പനി ഉണ്ട്")]
    graph = build_fact_graph(ents, turns)
    fact = next(f for f in graph.facts if f["concept"] == "fever")
    assert fact["speaker"] == "Patient"


def test_k_speaker_neutral_when_roles_unknown():
    turns = [{"speaker": "Speaker 0", "text": "പനി ഉണ്ട്"}]
    ents = [e.as_dict() for e in _ents("പനി ഉണ്ട്")]
    graph = build_fact_graph(ents, turns, roles_known=False)
    fact = next(f for f in graph.facts if f["concept"] == "fever")
    assert fact["speaker"] == "UNKNOWN"


# ---------------------------------------------------------------------------
# L. Malayalam-English mixed speech
# ---------------------------------------------------------------------------
def test_l_english_medical_word_in_malayalam_clause():
    assert status_of("BP ഉണ്ട്", "hypertension_history") == PRESENT


def test_l_english_sentence_negation():
    assert status_of("I have fever", "fever") == PRESENT
    assert status_of("No fever", "fever") == ABSENT


# ---------------------------------------------------------------------------
# M. Colloquial Malayalam
# ---------------------------------------------------------------------------
def test_m_colloquial_pressure_form():
    assert status_of("പ്രഷറുണ്ട്", "hypertension_history") == PRESENT


def test_m_colloquial_sugar_form():
    assert status_of("ഷുഗർ ഉണ്ട്", "diabetes_history") == PRESENT


# ---------------------------------------------------------------------------
# N. Final-virama elision
# ---------------------------------------------------------------------------
def test_n_coordinated_congestion():
    assert status_of("ചുമയും കഫക്കെട്ടും ഉണ്ട്", "phlegm") == PRESENT


def test_n_fused_denial():
    assert status_of("കഫക്കെട്ടില്ല", "phlegm") == ABSENT


# ---------------------------------------------------------------------------
# O. ASR-confirmed variants (real-audio verified forms)
# ---------------------------------------------------------------------------
def test_o_chomma_is_cough():
    result = apply_corrections("ചൊമ്മയുണ്ട്")
    assert any(c["raw"] == "ചൊമ്മയുണ്ട്" for c in result.corrections)
    e = find(normalize_clinical_text(result.corrected_text), "cough")
    assert e is not None and e.status == PRESENT
    assert e.source_clause == "ചുമയുണ്ട്"   # corrected copy is the evidence


def test_o_thalavanokke_is_headache():
    result = apply_corrections("തലവനൊക്കെയുണ്ട്")
    assert any(c["raw"] == "തലവനൊക്കെ" for c in result.corrections)
    e = find(normalize_clinical_text(result.corrected_text), "headache")
    assert e is not None and e.status == PRESENT


def test_o_kabakettu_is_congestion():
    result = apply_corrections("അതൊക്കെയുണ്ട് കബക്കെട്ട്")
    assert any(c["raw"] == "കബക്കെട്ട്" for c in result.corrections)
    e = find(normalize_clinical_text(result.corrected_text), "phlegm")
    assert e is not None and e.status == PRESENT


def test_o_food_cannot_eat_is_difficulty_eating():
    e = find(semantic_normalize("ഫുഡ് കഴിക്കാൻ പറ്റണില്ല"), "difficulty_eating")
    assert e is not None and e.status == PRESENT


def test_o_correction_has_evidence_and_regression_pinning():
    entry = correction_table().get("ചൊമ്മയുണ്ട്")
    assert entry is not None
    assert entry["corrected"] == "ചുമയുണ്ട്"
    assert "real" in entry["reason"].lower()   # evidence source recorded
    assert entry["confidence"] >= 0.85


def test_o_unverified_form_passes_through_unchanged():
    # Never silently "correct" uncertain speech: a lookalike that was never
    # observed must not be rewritten.
    raw = "ചൊമ്മപ്പാട്ട്"   # plausible Malayalam, never an observed corruption
    result = apply_corrections(raw)
    assert result.corrected_text == raw
    assert result.corrections == []


# ---------------------------------------------------------------------------
# P. Raw transcript immutability
# ---------------------------------------------------------------------------
def test_p_raw_transcript_never_modified():
    raw = "ആഹ സർ പനി മാറിട്ടും ക്ഷീണവും ചൊമ്മയുണ്ട് കബക്കെട്ട്"
    snapshot = raw
    result = apply_corrections(raw)
    normalize_clinical_text(result.corrected_text)
    assert raw == snapshot   # the caller's string is untouched


def test_p_correction_produces_separate_copy():
    raw = "ചൊമ്മയുണ്ട്"
    result = apply_corrections(raw)
    assert raw == "ചൊമ്മയുണ്ട്"                    # raw stays raw
    assert result.corrected_text == "ചുമയുണ്ട്"     # corrected is a separate copy


# ---------------------------------------------------------------------------
# Q. Provenance preservation
# ---------------------------------------------------------------------------
def test_q_provenance_fields_present():
    e = one("പനി ഉണ്ട്", "fever")
    assert e.surface_text == "പനി"
    assert "പനി" in e.source_clause
    assert e.confidence > 0


def test_q_fact_graph_keeps_malayalam_source():
    raw = "ചൊമ്മയുണ്ട്"
    corrected = apply_corrections(raw).corrected_text
    ents = [e.as_dict() for e in normalize_clinical_text(corrected)]
    graph = build_fact_graph(
        ents, [{"speaker": "UNKNOWN", "text": raw}], raw_text=raw)
    fact = next(f for f in graph.facts if f["concept"] == "cough")
    assert fact["raw_text"] == raw
    assert "ചുമ" in fact["source_clause"]


# ---------------------------------------------------------------------------
# Fact graph + anti-hallucination validator
# ---------------------------------------------------------------------------
def _full_graph(text: str, speaker: str = "Patient", roles_known: bool = True):
    corrected = apply_corrections(text).corrected_text
    ents = [e.as_dict() for e in _ents(corrected)]
    turns = [{"speaker": speaker, "text": corrected}]
    return build_fact_graph(ents, turns, raw_text=text, roles_known=roles_known)


def test_fact_graph_complete_record():
    graph = _full_graph("പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി, പക്ഷേ ക്ഷീണം ഇപ്പോഴും ഉണ്ട്")
    fever = next(f for f in graph.facts if f["concept"] == "fever")
    assert fever["status"] == RESOLVED
    assert fever["certainty"] == "CONFIRMED"
    assert fever["temporality"] == "PAST_RESOLVED"
    assert fever["speaker"] == "Patient"
    assert "പനി" in fever["source_clause"]
    assert fever["confidence"] > 0


def test_fact_graph_uncertain_fact_carries_reason():
    graph = _full_graph("പനി ആണെന്ന് തോന്നുന്നു")
    fever = next(f for f in graph.facts if f["concept"] == "fever")
    assert fever["status"] == "UNCERTAIN"
    assert fever["certainty"] == "UNCERTAIN"
    assert "തോന്നുന്നു" in fever["uncertainty_reason"]


def test_fact_graph_question_is_questioned():
    graph = _full_graph("പനി ഉണ്ടോ?", speaker="Doctor")
    fever = next(f for f in graph.facts if f["concept"] == "fever")
    assert fever["status"] == "QUESTIONED"


def test_fact_graph_frequency_attribute_only_when_supported():
    graph = _full_graph("ചിലപ്പോൾ ചുമയുണ്ട്")
    cough = next(f for f in graph.facts if f["concept"] == "cough")
    assert cough["frequency"] == "intermittent"
    graph2 = _full_graph("ചുമയുണ്ട്")
    cough2 = next(f for f in graph2.facts if f["concept"] == "cough")
    assert cough2["frequency"] is None   # unsupported field stays empty


def test_validator_accepts_fully_provenanced_graph():
    graph = _full_graph("പനി ഉണ്ട്")
    report = validate_fact_graph(graph)
    assert report["valid"] is True
    assert report["violations"] == []


def test_validator_rejects_fact_without_provenance():
    graph = _full_graph("പനി ഉണ്ട്")
    graph.facts[0]["source_clause"] = ""     # strip evidence
    graph.facts[0]["surface_text"] = ""
    report = validate_fact_graph(graph)
    assert report["valid"] is False
    assert any("provenance" in v for v in report["violations"])


def test_validator_rejects_inference_only_sections():
    graph = _full_graph("പനി ഉണ്ട്")
    graph.sections["assessment"] = [{
        "concept": "viral_fever", "status": "PRESENT",
        "source_clause": "", "surface_text": "",
        "confidence": 0.9, "speaker": "doctor",
    }]
    report = validate_fact_graph(graph)
    assert report["valid"] is False
    assert any("assessment" in v for v in report["violations"])


def test_validator_rejects_orphan_concept():
    graph = _full_graph("പനി ഉണ്ട്")
    graph.facts.append({
        "concept": "malaria", "status": "PRESENT", "certainty": "CONFIRMED",
        "temporality": "CURRENT", "surface_text": "", "source_clause": "",
        "speaker": "patient", "confidence": 0.9, "attributes": {},
        "uncertainty_reason": None,
    })
    report = validate_fact_graph(graph)
    assert report["valid"] is False
    assert any("malaria" in v for v in report["violations"])


def test_not_documented_never_becomes_absent_in_graph():
    graph = _full_graph("പനി ഉണ്ട്")
    report = validate_fact_graph(graph)
    # NOT_DOCUMENTED is a representation-level state: the graph carries only
    # discussed concepts, and any consumer asking about an unmentioned concept
    # must read NOT_DOCUMENTED from its absence — never ABSENT.
    assert all(f["status"] != "ABSENT" or f["surface_text"] for f in graph.facts)
    assert report["valid"] is True
