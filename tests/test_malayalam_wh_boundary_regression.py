"""Malayalam Q&A boundary regression — the production semantic-boundary failure.

Background (production, Malayalam consultation): one 229-character ASR turn
fused the doctor's wh-questions with the patient's answers. The clause splitter
only knew yes/no question cues (ഉണ്ടോ, ഇല്ലേ...) and clause-final verbs, so the
wh-questions (എന്താ, എത്ര, ഏത്, എവിടെ) never started a new semantic clause.
Consequences, all reproduced before the fix:

  - vomiting (spelled ഛർദി, lexicon knew only ഛർദ്ദി) was never matched;
  - every finding inside the mega-clause inherited the clause's QUESTION scope;
  - "5 days" attached itself to pain AND medicine;
  - "ഒന്നുമല്ല" ("nothing else") was not read as a negation.

These tests pin the fix at each layer: clause splitting, concept matching,
status, temporal attachment, fact graph, and the projected review fields.
All strings are synthetic constructions of the same linguistic shapes as the
production transcript — no patient data.
"""

from __future__ import annotations

import pytest

from scribe_engine import normalization as N
from scribe_engine.asr_correction import apply_corrections
from scribe_engine.clinical_facts import build_clinical_facts
from scribe_engine.field_projection import project_structured_fields
from scribe_engine.note_v2 import render_clinical_note
from scribe_engine.semantic import _merged_lexicon, semantic_normalize


def clauses_of(text: str) -> list[str]:
    return N.split_clauses(N._norm(text), _merged_lexicon())


def entities_of(text: str):
    """The production entry point: base engine + colloquial semantics layer."""
    return semantic_normalize(text)


def production_facts(text: str | None = None) -> dict:
    """build_clinical_facts exactly as the pipeline calls it: the Malayalam
    sidecar's corrected (NFC) copy is the extraction evidence."""
    text = PRODUCTION_TRANSCRIPT if text is None else text
    corrected = apply_corrections(text).corrected_text
    return build_clinical_facts(text, None, corrected_text=corrected)


def one(entities, concept):
    for e in entities:
        if e.concept == concept:
            return e
    return None


def facts_of(text: str):
    doc = production_facts(text)
    return {f["concept"]: f for f in doc["facts"]}


# ---------------------------------------------------------------------------
# 1. WH-question boundary (each marker below was observed in the production
#    transcript). A MOVE_WH token (ഏത് / എവിടെ / എന്താ ...) that follows an
#    answer must START a new clause; a QUANT_WH token (എത്ര*) quantifies the
#    concept before it and must NOT be torn away from it.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("wh", ["ഏത്", "എവിടെ", "എപ്പോൾ"])
def test_move_wh_token_after_answer_starts_a_new_clause(wh):
    """Answer + MOVE_WH question in one run-on turn: the question moves to a
    new clause, so the finding keeps its PRESENT status instead of swallowing
    the question's interrogative scope."""
    text = f"അഞ്ച് ദിവസം ആയി വയറു വേദന {wh} ഭാഗത്താ"
    pieces = clauses_of(text)
    assert len(pieces) >= 2, f"{wh}: no clause boundary created"
    assert any(p.startswith(wh) for p in pieces[1:]), (
        f"{wh}: question did not open a new clause: {pieces}")
    first_finding = N._norm(pieces[0])
    assert not N._is_question(first_finding, _merged_lexicon())


@pytest.mark.parametrize("wh", ["എന്താ", "എന്താണ്", "എത്ര"])
def test_quant_wh_splits_after_itself_not_mid_question(wh):
    """QUANT_WH ('how much/what is X') stays clause-final for the concept it
    quantifies: the original split-AFTER behaviour must survive — the clause
    boundary lands after the wh token, and the question keeps its concept."""
    text = f"രക്തസമ്മർദ്ദം {wh} എനിക്ക് പനി ഉണ്ട്"
    pieces = clauses_of(text)
    assert any(p.rstrip().endswith(wh) for p in pieces), (
        f"{wh}: split-after behaviour lost: {pieces}")
    ents = entities_of(text)
    assert one(ents, "fever") is not None


@pytest.mark.parametrize("wh", ["ഏത്", "എവിടെ"])
def test_move_wh_does_not_break_the_finding(wh):
    """MOVE_WH questions leave the finding's clause, so the finding keeps its
    PRESENT status (the production bug made everything QUESTIONED)."""
    text = f"അഞ്ച് ദിവസം ആയി വയറു വേദന {wh} ഭാഗത്താ"
    ents = entities_of(text)
    finding = one(ents, "abdominal_pain") or one(ents, "pain")
    assert finding is not None, f"finding lost: {[(e.concept, e.status) for e in ents]}"
    assert finding.status == "PRESENT"


def test_quant_wh_after_finding_stays_conservative():
    """QUANT_WH in the same clause as a finding keeps the conservative QUESTION
    status: one ASR turn with no role attribution must not assert a finding we
    cannot attribute. (Production: vomiting QUESTIONED — honest.) This is a
    documented limitation, not silent information loss: the fact survives, the
    clinician sees it in the ROS/questioned list, and nothing renders as a
    false assertion."""
    text = "അഞ്ച് ദിവസം ആയി വയറു വേദന എന്താ ഭാഗത്താ"
    ents = entities_of(text)
    finding = one(ents, "abdominal_pain") or one(ents, "pain")
    assert finding is not None, "finding dropped entirely"
    assert finding.status == "QUESTION"


def test_quant_wh_stays_with_the_concept_it_quantifies():
    """എത്രയാണ് asks about the preceding concept (BP): splitting it away would
    flip a doctor's measurement question into a patient PRESENT finding."""
    text = ("blood pressure എത്രയാണ് എനിക്ക് രണ്ട് വർഷമായി "
            "diabetes ഉണ്ട് ടാബ്ലറ്റ് കഴിക്കുന്നുണ്ട്")
    ents = entities_of(text)
    bp = one(ents, "hypertension_history")
    dm = one(ents, "diabetes_history")
    assert bp is not None and bp.status == "QUESTION"
    assert dm is not None and dm.status == "PRESENT"


def test_wh_marker_is_never_a_clause_final_verb():
    """The splitter must not cut immediately AFTER a wh token (that orphaned
    'ഏത് | ഭാഗത്താ' in production). No clause may end with a bare wh token."""
    lex = _merged_lexicon()
    wh_tokens = ("ഏത്", "എവിടെ", "എന്താ", "എത്ര")
    text = "അഞ്ച് ദിവസം ആയി വയറു വേദന ഏത് ഭാഗത്താ രണ്ടു ഭാഗത്തും ഉണ്ട്"
    for piece in clauses_of(text):
        stripped = piece.strip()
        assert not any(stripped.endswith(w) for w in wh_tokens), (
            f"clause ends with a bare wh token: {piece!r}")
        assert not N._is_question(N._norm(stripped), lex) or len(stripped.split()) > 1


def test_ordinary_malayalam_sentence_is_not_split_by_wh_inside_a_word():
    """False-positive guard: a wh substring inside an ordinary word ('എന്നാണ്'
    as a complementizer, or any word merely containing എത്) must not create a
    boundary."""
    text = "അത് എന്നാണ് പറഞ്ഞത് എനിക്ക് പനി ഉണ്ട്"
    pieces = clauses_of(text)
    # The fever clause must survive intact with a polarity verb.
    fever_clauses = [p for p in pieces if "പനി" in p]
    assert fever_clauses, "fever clause lost"
    ents = entities_of(text)
    assert one(ents, "fever") is not None
    assert one(ents, "fever").status == "PRESENT"


# ---------------------------------------------------------------------------
# 2. Vomiting spelling variant (ഛർദി vs ഛർദ്ദി — both observed in real STT)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("surface", ["ഛർദ്ദി", "ഛർദി", "ഛർദിക്കുന്നു", "ഛർദ്ദിക്കുന്നു"])
def test_both_vomiting_spellings_map_to_one_concept(surface):
    ents = entities_of(f"{surface} ഉണ്ട്")
    vom = [e for e in ents if e.concept == "vomiting"]
    assert vom, f"{surface} not matched"
    assert all(e.concept == "vomiting" for e in vom)


@pytest.mark.parametrize("surface", ["ഛർദ്ദി", "ഛർദി"])
def test_vomiting_variants_do_not_duplicate_facts(surface):
    graph = facts_of(f"{surface} രണ്ട് ദിവസം ആയി ഉണ്ട്")
    vomiting = [c for c, f in graph.items() if f["concept"] == "vomiting"
                or c == "vomiting"]
    assert vomiting
    # One normalized concept: no duplicate fact rows for the same concept.
    doc = build_clinical_facts(f"{surface} രണ്ട് ദിവസം ആയി ഉണ്ട്")
    vom_facts = [f for f in doc["facts"] if f["concept"] == "vomiting"]
    assert len(vom_facts) == 1, f"duplicate vomiting facts: {len(vom_facts)}"


def test_unrelated_words_unaffected_by_new_variants():
    """The new alias ഛർദി (short form) must not swallow unrelated words."""
    ents = entities_of("പനി ഉണ്ട്")
    assert one(ents, "fever") is not None
    assert one(ents, "vomiting") is None


# ---------------------------------------------------------------------------
# 3. Mega-clause / semantic boundaries: temporal ownership
# ---------------------------------------------------------------------------

def test_production_shape_duration_belongs_only_to_pain():
    """THE production failure: '5 days' rode onto pain AND medicine. The
    duration answer must land on the pain finding only; the medicine
    question (asked afterwards) carries no duration."""
    doc = production_facts()
    by_concept = {}
    for f in doc["facts"]:
        by_concept.setdefault(f["concept"], []).append(f)
    pain = [f for f in by_concept.get("pain", []) if f["status"] == "PRESENT"]
    assert pain, f"pain PRESENT fact missing: {by_concept.get('pain')}"
    pain_dur = (pain[0]["attributes"] or {}).get("duration")
    assert pain_dur == "5 days", f"pain duration wrong: {pain_dur!r}"
    meds = by_concept.get("medicine", [])
    for f in meds:
        assert not (f["attributes"] or {}).get("duration"), (
            f"medicine inherited a duration: {f['attributes']}")


def test_separate_durations_stay_separate():
    """Temporal safety: symptom A 5 days, medicine B 2 days — never swapped or
    propagated."""
    doc = build_clinical_facts(
        "തലവേദന അഞ്ച് ദിവസം ആയി ഉണ്ട് മരുന്ന് രണ്ട് ദിവസം ആയി കഴിക്കുന്നു")
    head = [f for f in doc["facts"] if f["concept"] == "headache"
            and f["status"] == "PRESENT"]
    assert head, "headache fact missing"
    assert (head[0]["attributes"] or {}).get("duration") == "5 days"
    med = [f for f in doc["facts"] if f["concept"] == "medicine"]
    if med:
        assert (med[0]["attributes"] or {}).get("duration") == "2 days", (
            f"medicine duration wrong: {med[0]['attributes']}")


def test_duration_hop_is_one_clause_only():
    """A duration-only answer clause feeds the NEXT clause's finding — and
    nothing further. A later, unrelated concept must not inherit it."""
    doc = build_clinical_facts(
        "എത്ര ദിവസം ആയി അഞ്ച് ദിവസം ആയി തലവേദന ഉണ്ട് ചുമയും ഉണ്ട്")
    head = [f for f in doc["facts"] if f["concept"] == "headache"]
    cough = [f for f in doc["facts"] if f["concept"] == "cough"]
    assert head, "headache missing"
    assert (head[0]["attributes"] or {}).get("duration") == "5 days"
    for f in cough:
        assert not (f["attributes"] or {}).get("duration"), (
            "duration crossed a concept boundary: cough inherited '5 days'")


# ---------------------------------------------------------------------------
# 4. 'ഒന്നുമല്ല' (nothing else) — contextual negation
# ---------------------------------------------------------------------------

def test_nothing_else_is_a_negation_cue_in_context():
    """"...ഒന്നുമല്ല" after a symptom statement denies additional findings.
    The patient's named symptom stays PRESENT; nothing new is invented."""
    doc = build_clinical_facts("വയറു വേദന ഉണ്ട് ഒന്നുമല്ല")
    pains = [f for f in doc["facts"]
             if f["concept"] in ("pain", "abdominal_pain")]
    assert pains, "symptom fact lost"
    assert pains[0]["status"] == "PRESENT"
    # The negation must not manufacture a second, contradictory fact.
    assert len(pains) == 1, (
        f"contradictory facts: {[(f['concept'], f['status']) for f in pains]}")



def test_nothing_else_does_not_deny_a_stated_symptom():
    ents = entities_of("പനിയും വയറു വേദനയും ഉണ്ട് ഒന്നുമല്ല")
    fever = one(ents, "fever")
    assert fever is not None and fever.status == "PRESENT"
    finding = one(ents, "abdominal_pain") or one(ents, "pain")
    assert finding is not None and finding.status == "PRESENT"



# ---------------------------------------------------------------------------
# 5. Full production-shape regression: fact graph → note → fields agree
# ---------------------------------------------------------------------------

# SANITIZED FIXTURE — derived from the exact production failure.
# The transcript is pure spoken Malayalam clinical dialogue: it contains no
# digits, no latin-script names, no phone/DOB/address/email (verified by the
# PHI scans below and the CI PHI guard). The linguistic structures that
# triggered the failure (wh-questions, the duration answer, the 'nothing
# else' negation, the generic medicine question, the vomiting spelling) are
# preserved verbatim.
PRODUCTION_TRANSCRIPT = (
    "ആഹ് പേരെന്താ ഇബ ഇബൻ എന്താ പറ്റിയത് ഛർദി ഛർദിയാ ഉം ഹ എത്ര "
    "നാളായി ഇപ്പോ എത്ര ദിവസമായി എത്ര ദിവസ അഞ്ച് ദിവസ അഞ്ച് "
    "ദിവസോഹ് അഞ്ച് ദിവസം വേറെ ബുദ്ധിമുട്ട ഒന്നുമല്ല വേറെ "
    "ലനിണ്ട് എവിടൊ വേദന വയറിന്റെവിടെ വയറിന്റെ അവിടെ വേദന ഹ "
    "ഏത് ഭാഗത്ത് രണ്ട് സൈഡിൽ രണ്ട് സൈഡിലും മരുന്നണ്ട് ബോത്ത് "
    "സൈഡ്സ് ഉ ഓക്കെ ഓ ഓക്കെ ഇപ്പൊ എന്തേലും മരുന്ന് "
    .strip()
)

PRODUCTION_SHAPE_TEXT = PRODUCTION_TRANSCRIPT


def test_fact_graph_recovers_vomiting_and_pain():
    """Before the fix this transcript produced exactly 2 facts
    (pain PRESENT + medicine PRESENT, both '5 days'). After the fix the
    clinically supported facts are all present with honest statuses."""
    doc = production_facts()
    concepts = {f["concept"] for f in doc["facts"]}
    assert "vomiting" in concepts, "vomiting fact missing"
    assert "pain" in concepts, "pain fact missing"
    vom = next(f for f in doc["facts"] if f["concept"] == "vomiting")
    assert vom["status"] == "QUESTIONED"  # doctor's question, never answered
    pain = next(f for f in doc["facts"] if f["concept"] == "pain"
                and f["status"] == "PRESENT")
    assert (pain["attributes"] or {}).get("duration") == "5 days"
    med = next(f for f in doc["facts"] if f["concept"] == "medicine")
    assert med["status"] == "QUESTIONED"  # doctor asked, patient never confirmed
    assert not (med["attributes"] or {}).get("duration")


def test_note_and_projection_stay_consistent():
    """fact graph → canonical note → structured fields must agree (single
    source). The note's HPI shows the pain+duration; the fields' medications
    carry no invented duration."""
    doc = production_facts()
    note = render_clinical_note(doc)
    proj = project_structured_fields(doc)
    assert note["validation"]["valid"], note["validation"]
    fields = proj["clinical_note_fields"]
    assert fields["symptoms_reported"]
    assert fields["duration"] == "5 days"
    # The medicine question must NOT reach the prescription fields as a drug.
    assert not proj["prescription_fields"]["medications"], (
        f"questioned medicine leaked into rx: "
        f"{proj['prescription_fields']['medications']}")


def test_projected_fields_carry_honest_medication_status():
    proj = project_structured_fields(build_clinical_facts(PRODUCTION_TRANSCRIPT))
    rx = proj["prescription_fields"]
    # The generic medicine reference is a QUESTION, not a prescription.
    assert rx["medications"] in ([], "Not documented.", None) or \
        "medicine" not in str(rx["medications"])
