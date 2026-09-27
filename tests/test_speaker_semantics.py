"""Speaker-semantics regression tests: a doctor's question is never a finding.

The clinical contract under test (three shapes the briefing requires):

    Doctor asks + patient denies  -> ABSENT (never PRESENT)
    Doctor asks + patient affirms -> PRESENT
    Doctor asks, no answer        -> QUESTION (never PRESENT)

These run through the production entry point (build_normalized_entities) with
speaker-prefixed lines exactly as pipeline.py feeds them, including the
roles-unknown shape (every line scanned for both sides). The transcript text
mirrors what real speech produces: the question and the answer are separate
clauses, and the verb-final scope engine must bind the polarity marker to the
mention inside its own clause — a patient's ഇല്ല must not be leaked backwards
onto the doctor's question clause, and the doctor's ഉണ്ടോ must not make the
patient's affirmative answer a question.
"""
from scribe_engine.clinical import build_normalized_entities


def find(norm: dict, concept: str):
    """The concept's entity dict from build_normalized_entities output."""
    for e in norm.get("entities", []):
        if e["concept"] == concept:
            return e
    return None


def one(text, concept):
    e = find(build_normalized_entities(text), concept)
    assert e is not None, f"{concept} not extracted from: {text}"
    return e


def status_of(text, concept):
    return one(text, concept)["status"]


# ---------------------------------------------------------------------------
# Doctor asks + patient answers
# ---------------------------------------------------------------------------

def test_doctor_asks_patient_denies_is_absent():
    n = build_normalized_entities(
        "Doctor: പനി ഉണ്ടോ?\nPatient: ഇല്ല, പനി ഇല്ല.")
    fever = one("Doctor: പനി ഉണ്ടോ?\nPatient: ഇല്ല, പനി ഇല്ല.", "fever")
    assert fever["status"] == "ABSENT"
    assert "fever" in n["absent"]
    assert "fever" not in n["present"]


def test_doctor_asks_patient_affirms_is_present():
    text = "Doctor: പനി ഉണ്ടോ?\nPatient: ഉണ്ട്, മൂന്ന് ദിവസമായി പനി ഉണ്ട്."
    assert one(text, "fever")["status"] == "PRESENT"


def test_doctor_asks_without_answer_is_questioned_not_present():
    text = "Doctor: പനി ഉണ്ടോ?"
    assert one(text, "fever")["status"] == "QUESTION"
    n = build_normalized_entities(text)
    assert "fever" in n["questioned"]
    assert "fever" not in n["present"] and "fever" not in n["absent"]


def test_roles_unknown_question_still_not_a_finding():
    """roles_known=False: every line is scanned; a question line must still
    yield QUESTION, not PRESENT, for the concept it asks about."""
    text = "Speaker 0: ചുമ ഉണ്ടോ?\nSpeaker 1: ഉണ്ട്."
    assert one(text, "cough")["status"] == "PRESENT"  # answered affirmatively in its own clause
    text_unanswered = "Speaker 0: ചുമ ഉണ്ടോ?\nSpeaker 1: രണ്ട് ദിവസമായി."
    assert one(text_unanswered, "cough")["status"] == "QUESTION"


# ---------------------------------------------------------------------------
# The question/answer boundary must hold in both clause orders
# ---------------------------------------------------------------------------

def test_patient_denial_after_question_not_contaminated():
    """Patient's denial clause comes after the question clause; the question
    marker must not convert the denial into a question."""
    text = "Doctor: ചുമയുണ്ടോ?\nPatient: ചുമയില്ല."
    assert one(text, "cough")["status"] == "ABSENT"


def test_patient_affirmation_after_question_not_contaminated():
    text = "Doctor: ചുമയുണ്ടോ?\nPatient: ചുമയുണ്ട്, രാത്രി കൂടുതലാണ്."
    assert one(text, "cough")["status"] == "PRESENT"


# ---------------------------------------------------------------------------
# difficulty_eating: the real-consultation functional symptom
# ---------------------------------------------------------------------------

def test_difficulty_eating_observed_phrase():
    """The exact phrase from the real consultation audio."""
    e = one("ഫുഡ് കഴിക്കാൻ പറ്റണില്ല", "difficulty_eating")
    assert e["status"] == "PRESENT"


def test_difficulty_eating_real_transcript_tail():
    """The real transcript's run-on tail (പ്രശ്നങ്ങൾ etc. after the phrase)."""
    e = one("പിന്നെ ഫുഡ് കഴിക്കാൻ പറ്റണില്ലതൊക്കെയാണ് പ്രശ്നങ്ങൾ",
            "difficulty_eating")
    assert e["status"] == "PRESENT"


def test_difficulty_eating_grammatical_variants():
    for text in (
        "ഭക്ഷണം കഴിക്കാൻ പറ്റുന്നില്ല",
        "ഭക്ഷണം കഴിക്കാൻ പറ്റണില്ല",
        "ഭക്ഷണം കഴിക്കാൻ ബുദ്ധിമുട്ടുണ്ട്",
        "ഫുഡ് കഴിക്കാൻ ബുദ്ധിമുട്ട്",
        "തിന്നാൻ പറ്റുന്നില്ല",
        "ഒന്നും കഴിക്കാൻ പറ്റുന്നില്ല",
    ):
        e = one(text, "difficulty_eating")
        assert e["status"] == "PRESENT", text


def test_difficulty_eating_english_variant():
    e = one("I cannot eat anything since yesterday.", "difficulty_eating")
    assert e["status"] == "PRESENT"


def test_true_denial_of_eating_difficulty_is_absent():
    """A genuine negation of the difficulty must record ABSENT, never flip to
    PRESENT via the negation-idiom list — and an assertion of ability
    ('കഴിക്കാൻ പറ്റുന്നുണ്ട്' = CAN eat) extracts nothing at all, which is the
    safe silence: no finding without evidence."""
    n = build_normalized_entities("ഭക്ഷണം കഴിക്കാൻ ബുദ്ധിമുട്ടില്ല.")
    e = find(n, "difficulty_eating")
    assert e is not None and e["status"] == "ABSENT"
    assert "difficulty eating" in n["absent"] and "difficulty eating" not in n["present"]
    # Ability asserted: no alias matches, so no entity — safe silence.
    n2 = build_normalized_entities("ഇപ്പോൾ ഭക്ഷണം കഴിക്കാൻ പറ്റുന്നുണ്ട്.")
    assert find(n2, "difficulty_eating") is None
    assert not n2.get("present") and not n2.get("absent")


def test_difficulty_eating_not_mapped_to_any_diagnosis():
    """Functional symptom only: no nausea/anorexia/dysphagia inference."""
    n = build_normalized_entities("ഫുഡ് കഴിക്കാൻ പറ്റണില്ല")
    assert n["present"] == ["difficulty eating"]
    assert "nausea" not in n["present"] and "anorexia" not in n["present"]


def test_note_consumes_difficulty_eating():
    """The clinical note adds the normalized finding to symptoms_reported."""
    from scribe_engine.clinical import build_clinical_note, extract_medical_info
    n = build_normalized_entities("ഫുഡ് കഴിക്കാൻ പറ്റണില്ല")
    note = build_clinical_note(extract_medical_info("ഫുഡ് കഴിക്കാൻ പറ്റണില്ല"),
                               [], ["ഫുഡ് കഴിക്കാൻ പറ്റണില്ല"], normalized=n)
    assert "difficulty eating" in note["fields"]["symptoms_reported"]
