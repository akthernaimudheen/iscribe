"""Status-scope regression tests.

A resolution/polarity marker must attach to the concept it governs — never
leak across the clause onto an unrelated symptom. Root cause fixed in
normalization.py: the clause-wide ``if resolution: status = RESOLVED`` flip
resolved every concept sharing a clause with the resolved one (real-audio
evidence: fever's "മാറിട്ടും" resolved the fatigue named beside it).

The general mechanism under test:
  - Malayalam is verb-final: a resolution verb governs the mentions BEFORE it
    (its subject), never a new predication started after it.
  - A clause whose final predicate is a resolution verb and which names no
    symptom resolves the symptom just discussed (ellipsis).
  - Coordination with one shared verb ("പനിയും ക്ഷീണവും മാറി") still resolves
    both concepts.
  - Bare mentions keep their polarity-derived status.
"""
import pytest

from scribe_engine.normalization import (
    ABSENT,
    PRESENT,
    QUESTION,
    RESOLVED,
    load_lexicon,
    normalize_clinical_text,
)
from scribe_engine.semantic import semantic_normalize


def find(entities, concept):
    for e in entities:
        if e.concept == concept:
            return e
    return None


def status_of(text, concept):
    e = find(semantic_normalize(text), concept)
    assert e is not None, f"{concept} not extracted from: {text}"
    return e.status


# ---------------------------------------------------------------------------
# Required regression cases
# ---------------------------------------------------------------------------

def test_main_case_fever_resolved_fatigue_present():
    """The reported real-world failure: fever's resolution must not resolve fatigue."""
    entities = semantic_normalize(
        "പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി. പക്ഷേ ക്ഷീണം ഇപ്പോഴും ഉണ്ട്.")
    fever = find(entities, "fever")
    fatigue = find(entities, "fatigue")
    assert fever is not None and fever.status == RESOLVED
    assert fatigue is not None and fatigue.status == PRESENT


def test_resolution_ellipsis_clause():
    """'ഇപ്പോൾ മാറി.' names no symptom but resolves the one just discussed."""
    assert status_of("പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ മാറി.", "fever") == RESOLVED


def test_fever_denied():
    assert status_of("പനി ഇല്ല.", "fever") == ABSENT


def test_fever_past_form_recorded_present():
    """ഉണ്ടായിരുന്നു alone asserts past presence; without a current-absent
    half it is not automatically resolved (temporality is separate)."""
    assert status_of("പനി ഉണ്ടായിരുന്നു.", "fever") == PRESENT


def test_fever_now_absent():
    assert status_of("ഇപ്പോൾ പനി ഇല്ല.", "fever") == ABSENT


def test_fever_resolved():
    assert status_of("പനി മാറി.", "fever") == RESOLVED


def test_fatigue_still_present():
    assert status_of("ക്ഷീണം ഇപ്പോഴും ഉണ്ട്.", "fatigue") == PRESENT


def test_cough_resolved_phlegm_present():
    """Cough resolves; congestion (കഫക്കെട്ട്) named in its own clause stays present."""
    entities = semantic_normalize(
        "ചുമയും കഫക്കെട്ടും ഉണ്ടായിരുന്നു, ഇപ്പോൾ ചുമ മാറി പക്ഷേ കഫക്കെട്ട് ഉണ്ട്.")
    cough = find(entities, "cough")
    phlegm = find(entities, "phlegm")
    assert cough is not None and cough.status == RESOLVED
    assert phlegm is not None and phlegm.status == PRESENT


def test_fever_denied_but_fatigue_present():
    assert status_of("പനി ഇല്ല, പക്ഷേ ക്ഷീണം ഉണ്ട്.", "fatigue") == PRESENT
    assert status_of("പനി ഇല്ല, പക്ഷേ ക്ഷീണം ഉണ്ട്.", "fever") == ABSENT


# ---------------------------------------------------------------------------
# Coordination: shared predicate vs separate predications
# ---------------------------------------------------------------------------

def test_coordinated_concepts_share_resolution():
    """One verb, both subjects preceding it: both resolve."""
    assert status_of("പനിയും ക്ഷീണവും മാറി", "fever") == RESOLVED
    assert status_of("പനിയും ക്ഷീണവും മാറി", "fatigue") == RESOLVED


def test_coordinated_concepts_share_presence():
    assert status_of("പനിയും ക്ഷീണവും ഉണ്ട്", "fatigue") == PRESENT


def test_resolution_before_new_predication_does_not_leak():
    """മാറിട്ടും ends its predication; the next concept keeps its own status."""
    assert status_of(
        "പനി മാറിട്ടും ഭയങ്കര ക്ഷീണവും കാര്യങ്ങളാണ്", "fatigue") == PRESENT


# ---------------------------------------------------------------------------
# Constructed multi-clause consultation (synthetic): exercises the same rules
# the real frozen consultation did — resolution before a new predication must
# not leak, coordinated/interleaved mentions keep their own status, negation
# and difficulty constructs survive — without committing any patient content.
# The real transcript itself stays OUT of version control (tests/fixtures/).
# ---------------------------------------------------------------------------

SYNTHETIC_CONSULTATION = (
    "ഗുഡ് മോണിം ഡോക്ടർ. പനി മാറിട്ടും ഇപ്പോഴും ഭയങ്കര ക്ഷീണവും ഉണ്ട്. "
    "ഇടക്ക് തലവേദനയും ഉണ്ട്. കഫക്കെട്ട് ഇല്ല. "
    "പക്ഷേ ഫുഡ് കഴിക്കാൻ ബുദ്ധിമുട്ട് ഉണ്ട്.")


def test_multi_clause_consultation_no_resolution_leak():
    entities = semantic_normalize(SYNTHETIC_CONSULTATION)
    fever = find(entities, "fever")
    fatigue = find(entities, "fatigue")
    headache = find(entities, "headache")
    assert fever is not None and fever.status == RESOLVED
    assert fatigue is not None and fatigue.status == PRESENT
    assert headache is not None and headache.status == PRESENT
    # Provenance intact: every entity keeps its evidence clause.
    for e in entities:
        assert e.source_clause, e.concept


def test_real_transcript_raw_source_preserved():
    """The semantic layer never rewrites the input."""
    text = "പനി മാറിട്ടും ക്ഷീണവും ഉണ്ട്"
    semantic_normalize(text)
    assert text == "പനി മാറിട്ടും ക്ഷീണവും ഉണ്ട്"


# ---------------------------------------------------------------------------
# English path must keep its own (unchanged) resolution semantics
# ---------------------------------------------------------------------------

def test_english_resolution_scopes_to_its_concept():
    entities = normalize_clinical_text(
        "Fever is gone but fatigue is still present.")
    fever = find(entities, "fever")
    fatigue = find(entities, "fatigue")
    assert fever is not None and fever.status == RESOLVED
    assert fatigue is not None and fatigue.status == PRESENT


def test_question_marker_does_not_resolve():
    assert status_of("പനി ഉണ്ടോ?", "fever") == QUESTION


# ---------------------------------------------------------------------------
# Lexicon: കഫക്കെട്ട് (congestion) is the standard colloquial word for phlegm
# ---------------------------------------------------------------------------

def test_congestion_word_extracted():
    assert status_of("കഫക്കെട്ട് ഉണ്ട്", "phlegm") == PRESENT


def test_congestion_word_negated():
    assert status_of("കഫക്കെട്ട് ഇല്ല", "phlegm") == ABSENT
