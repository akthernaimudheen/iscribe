"""Clinical information extraction and note generation.

Adapted from upstream AI-MEDICAL-SCRIBE (commit faf97d5):
- `extract_medical_info` — from nlp.py (the best-guarded copy; final1.py and
  end.py carry degraded duplicates of the same function).
- Note generation — from the front.py / new.py keyword+template scheme,
  converted from inline Streamlit UI code into pure functions returning
  structured data.

Known upstream defects carried over intentionally (documented in ANALYSIS.md
section 4): the duration regex is fragile on real transcripts and the fuzzy
matcher is aggressive. These are NOT silently fixed here — engine improvement
is a later phase.
"""

import logging
import re

import spacy
from rapidfuzz import fuzz, process
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upstream data (verbatim): symptom phrase list and vague durations from nlp.py
# ---------------------------------------------------------------------------
SYMPTOM_PHRASES = [
    "chest pain", "shortness of breath", "headache", "fever", "nausea", "fatigue",
    "dizziness", "vomiting", "cough", "sore throat", "chills", "pain", "rash",
    "abdominal pain", "back pain", "joint pain", "muscle pain", "wheezing",
    "swelling", "diarrhea", "constipation",
]
VAGUE_DURATIONS = ["a while", "some time", "awhile", "forever", "recently", "a bit"]

# Upstream data (verbatim): keyword scan list from front.py / new.py
SYMPTOM_KEYWORDS = [
    "asthma", "allergies", "fever", "headache", "fatigue", "nausea", "cough", "cold",
    "sneezing", "sore throat", "runny nose", "body ache", "chest pain", "shortness of breath",
    "vomiting", "dizziness", "loss of appetite", "congestion", "itching", "rashes",
    "throat pain", "stomach pain", "back pain", "joint pain", "ear pain", "eye irritation",
    "blurred vision", "diarrhea", "constipation", "weight loss", "weight gain", "anxiety",
    "depression", "insomnia", "palpitations", "swelling", "bleeding", "bruising", "hair loss",
    "dry skin", "acne", "frequent urination", "thirst", "chills", "night sweats",
    "muscle cramps", "tremors", "memory loss", "confusion", "irritability", "mood swings",
    "menstrual irregularities", "hot flashes", "cold intolerance", "heat intolerance",
    "frequent infections", "slow healing wounds", "numbness", "tingling", "weakness",
    "balance issues", "coordination problems", "type 2 diabetes", "hypertension",
    "hyperlipidemia", "palpitations",
]

# Upstream data (verbatim): medication regex from front.py / new.py
MEDICATION_PATTERN = (
    r"\b(allegra|claritin|zyrtec|zytech|spray|anti-histamine|inhaler|paracetamol|"
    r"cetirizine|ibuprofen|antibiotic|nasal drop|metformin|vitamin-b complex|amlodipine)\b"
)

# Upstream regexes (verbatim) for diagnosis and durations (nlp.py)
DIAGNOSIS_PRIMARY_RE = re.compile(r"it looks like ([\w\s\-]+)")
DIAGNOSIS_SECONDARY_RE = re.compile(r"diagnosed with ([\w\s\-]+)")
DURATION_PATTERNS = [
    re.compile(r"for (?:the last|past)? ?(\d+ \w+(?: to \d+ \w+)?)"),
    re.compile(r"(?:going on|started|since) (\w+ \w+|\d+ \w+(?: to \d+ \w+)?)"),
]
# DEVIATION from upstream (documented in ANALYSIS.md section 4): upstream's name
# regex is case-sensitive ("i'm" never matches "I'm"), so it fails on upstream's
# own demo fixture. IGNORECASE restores the evident intent; the captured name is
# still required to be a capitalized word.
NAME_RE = re.compile(r"(?:my name is|i am|i'm)\s+([A-Z][a-z]+)", re.IGNORECASE)
AGE_RE = re.compile(r"(?:i am|i'm)\s+(\d{1,2})\s*(?:years old|yo)?")

_nlp = None
_phrase_matcher = None


def _get_nlp():
    """Lazy-load spaCy + PhraseMatcher so importing the module stays cheap."""
    global _nlp, _phrase_matcher
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed. Run: "
                "python -m pip install https://github.com/explosion/spacy-models/"
                "releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
            ) from e
        _phrase_matcher = PhraseMatcher(_nlp.vocab, attr="LOWER")
        _phrase_matcher.add("SYMPTOM", [_nlp.make_doc(t) for t in SYMPTOM_PHRASES])
    return _nlp


def fuzzy_match_symptom(text, symptom_list, threshold=85):
    """Upstream nlp.py function, verbatim logic."""
    match = process.extractOne(text, symptom_list, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= threshold:
        return match[0]
    return None


def extract_medical_info(conversation):
    """Upstream nlp.py extract_medical_info, same output schema."""
    if not conversation or not isinstance(conversation, str):
        logger.warning("Invalid or empty conversation input")
        return {
            "patient_name": None,
            "age": None,
            "symptoms": [],
            "duration": None,
            "diagnoses": [],
            "warnings": ["Invalid or empty input"],
            "confidence": {"symptoms": 0.0, "duration": 0.0, "diagnoses": 0.0},
        }

    nlp = _get_nlp()
    doc = nlp(conversation)
    symptoms = set()
    durations = []
    diagnoses = []
    warnings = []
    patient_name = None
    age = None

    for sent in doc.sents:
        text = sent.text.lower()
        sent_doc = nlp(sent.text)

        for _, start, end in _phrase_matcher(sent_doc):
            symptoms.add(sent_doc[start:end].text.lower())

        for token in sent_doc:
            if token.pos_ in ["NOUN", "ADJ"]:
                matched = fuzzy_match_symptom(token.text.lower(), SYMPTOM_PHRASES)
                if matched:
                    symptoms.add(matched)

        m = DIAGNOSIS_PRIMARY_RE.search(text)
        if m:
            diagnoses.insert(0, m.group(1).strip().rstrip("."))
        else:
            m = DIAGNOSIS_SECONDARY_RE.search(text)
            if m:
                diagnoses.append(m.group(1).strip().rstrip("."))

        for pattern in DURATION_PATTERNS:
            m = pattern.search(text)
            if m:
                duration = m.group(1).strip()
                durations.append(duration)
                if any(v in duration for v in VAGUE_DURATIONS):
                    warnings.append(f"Vague duration detected: '{duration}'")
                break

        nm = NAME_RE.search(sent.text)
        if nm and not patient_name:
            patient_name = nm.group(1)
        am = AGE_RE.search(text)
        if am and not age:
            age = int(am.group(1))

    confidence = {
        "symptoms": 1.0 if symptoms else 0.5,
        "duration": 0.8 if durations and not any(v in d for d in durations for v in VAGUE_DURATIONS) else 0.4,
        "diagnoses": 1.0 if diagnoses else 0.5,
    }
    if not symptoms:
        warnings.append("No symptoms detected")
    if not durations:
        warnings.append("No duration detected")
    if not diagnoses:
        warnings.append("No diagnosis detected")

    return {
        "patient_name": patient_name,
        "age": age,
        "symptoms": sorted(symptoms),
        "duration": durations[0] if durations else None,
        "diagnoses": diagnoses,
        "warnings": warnings,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Note generation (upstream front.py / new.py scheme, as structured data)
# ---------------------------------------------------------------------------

def scan_symptom_keywords(patient_text: str) -> list:
    """Upstream front.py behavior: keyword scan over patient lines only."""
    text = " ".join(patient_text).lower() if isinstance(patient_text, list) else patient_text.lower()
    return [w for w in SYMPTOM_KEYWORDS if w in text]


def scan_medications(doctor_text: str) -> list:
    """Upstream front.py behavior: lowercase text, then regex scan over doctor lines only."""
    text = " ".join(doctor_text) if isinstance(doctor_text, list) else doctor_text
    return sorted(set(re.findall(MEDICATION_PATTERN, text.lower(), re.IGNORECASE)))


# Shown wherever nothing was actually said. Never replaced by plausible-sounding
# clinical text: an empty field is a true statement, boilerplate is a false one.
NOT_MENTIONED = "Not mentioned"


def build_normalized_entities(transcript_text: str,
                              corrected_text: str | None = None) -> dict:
    """Language-aware clinical normalization of the transcript.

    Two layers, both additive and both leaving the raw transcript untouched:
      1. ASR correction (verified corruptions only, with provenance) when the
         caller supplies ``corrected_text`` from the Malayalam sidecar.
      2. Semantic normalization (scribe_engine.semantic): colloquial Kerala
         expressions, high/low/check/question context, family-history subject
         attribution - on top of the base scope engine.

    Returns {} when nothing is recognised, so callers can ignore it entirely.
    """
    try:
        from .normalization import entities_to_english_summary
        from .semantic import MEDICATION_CONCEPTS, semantic_normalize
    except Exception:  # pragma: no cover - normalization must never break a note
        logger.warning("clinical normalization unavailable")
        return {}

    try:
        entities = semantic_normalize(corrected_text or transcript_text)
    except Exception:  # pragma: no cover
        logger.exception("clinical normalization failed")
        return {}

    if not entities:
        return {}

    # Conditional mentions ("come back if the fever does not settle") describe a
    # hypothetical, not the patient's state, and are excluded from every finding
    # list. They stay in `entities` so the reasoning remains auditable.
    asserted = [e for e in entities if not e.conditional]

    # Family history ("അമ്മയ്ക്ക് ഷുഗർ ഉണ്ട്") is never a patient finding. An
    # explicit patient marker or no marker at all keeps the conversational
    # default: the patient is the one describing themselves.
    family_subjects = {"mother", "father", "family"}
    family_history = [e for e in asserted
                      if e.subject in family_subjects and e.status == "PRESENT"]
    patient_asserted = [e for e in asserted if e.subject not in family_subjects]

    # Medication mentions are treatment, not symptoms; the prescription layer
    # owns them. They stay in `entities` for audit but never reach symptom lists.
    meds = [e for e in patient_asserted if e.concept in MEDICATION_CONCEPTS]
    findings = [e for e in patient_asserted if e.concept not in MEDICATION_CONCEPTS]

    corrections: list[dict] = []
    if corrected_text:
        try:
            from .asr_correction import apply_corrections

            corrections = apply_corrections(transcript_text).corrections
        except Exception:  # pragma: no cover
            logger.exception("asr-correction provenance failed")

    return {
        "entities": [e.as_dict() for e in entities],
        # A structured restatement, NOT a transcript. Labelled so the UI and
        # any exporter can never present it as the patient's own words.
        "normalized_summary": entities_to_english_summary(findings),
        "summary_is_not_a_transcript": True,
        "present": sorted({e.english for e in findings if e.status == "PRESENT"}),
        "absent": sorted({e.english for e in findings if e.status == "ABSENT"}),
        "resolved": sorted({e.english for e in findings if e.status == "RESOLVED"}),
        "questioned": sorted({e.english for e in findings if e.status == "QUESTION"}),
        "uncertain": sorted({e.english for e in findings
                             if e.status in ("POSSIBLE", "UNKNOWN")}),
        "conditional": sorted({e.english for e in entities if e.conditional}),
        "family_history": sorted({e.english for e in family_history}),
        "medications_discussed": sorted({e.english for e in meds}),
        "asr_corrections_applied": corrections,
        "normalized_from": "asr_corrected" if corrected_text else "raw",
    }


def build_clinical_note(structured: dict, doctor_lines: list, patient_lines: list,
                        normalized: dict | None = None) -> dict:
    """Produce the clinical note from the transcript ONLY.

    CLINICAL SAFETY RULE: every field here is either extracted from the
    transcript or reads "Not mentioned". Upstream front.py filled physical
    findings, investigations, treatment response and remarks with hard-coded
    sentences ("no acute distress", "Basic blood test and allergy screening",
    "Patient showing mild improvement"), and fell back to a fabricated
    impression ("Possible upper respiratory condition") when nothing was
    detected. Those are examination findings, investigation orders and a
    diagnosis that no clinician said. They are removed, not merely flagged —
    a [template] marker still ends up in an exported record.
    """
    detected = scan_symptom_keywords(" ".join(patient_lines))
    medications = scan_medications(" ".join(doctor_lines))
    diagnoses = structured.get("diagnoses") or []

    # Normalized entities only ADD symptoms that were actually stated as present.
    # Anything the normalizer marked ABSENT, QUESTION, UNKNOWN or POSSIBLE is
    # deliberately excluded: a denied symptom and a doctor's question are not
    # findings, and keyword scanning cannot tell them apart.
    normalized = normalized or {}
    denied = normalized.get("absent", [])
    questioned = normalized.get("questioned", [])

    # Remove keyword-scanner false positives. The scanner matches "allergies"
    # in "No allergies doctor" and records a denied history as a present
    # symptom; it also cannot tell a doctor's question from a patient's report.
    # The normalizer knows the polarity, so it corrects the list.
    suppressed = {d.lower() for d in denied} | {q.lower() for q in questioned}
    detected = [s for s in detected
                if not any(sup in s.lower() or s.lower() in sup for sup in suppressed)]

    for english in normalized.get("present", []):
        if english.lower() not in " ".join(detected).lower():
            detected = detected + [english]

    fields = {
        "symptoms_reported": ", ".join(detected) if detected else NOT_MENTIONED,
        # Explicit negatives are clinically valuable and were previously lost.
        "denies": ", ".join(denied) if denied else NOT_MENTIONED,
        "current_treatment_discussed": ", ".join(medications) if medications else NOT_MENTIONED,
        # Not derivable from audio at all — there is no examination in a transcript.
        "physical_findings": NOT_MENTIONED,
        # Only ever the diagnosis the doctor actually stated. No fallback.
        "impression": (
            f"Possible {', '.join(diagnoses)}." if diagnoses else NOT_MENTIONED
        ),
        "investigations": NOT_MENTIONED,
        "treatment_response": NOT_MENTIONED,
        "additional_remarks": NOT_MENTIONED,
        "patient_name": structured.get("patient_name"),
        "age": structured.get("age"),
        "duration": structured.get("duration") or NOT_MENTIONED,
        "diagnoses": diagnoses,
        "warnings": structured.get("warnings", []),
        "confidence": structured.get("confidence", {}),
    }

    return {"fields": fields, "text": render_clinical_note_text(fields)}


def empty_clinical_note(reason: str) -> dict:
    """A note for a transcript that failed validation.

    Deliberately carries no clinical content: when the transcript is not
    trustworthy, the correct output is nothing plus the reason, not a draft the
    clinician might partially accept.
    """
    fields = {key: NOT_MENTIONED for key, _ in NOTE_LAYOUT}
    fields.update({
        "patient_name": None,
        "age": None,
        "duration": NOT_MENTIONED,
        "diagnoses": [],
        "warnings": [reason],
        "confidence": {},
        "not_generated": True,
        "not_generated_reason": reason,
    })
    text = ("Clinical Notes:\n"
            "No clinical note was generated because the transcript did not pass "
            f"quality validation.\nReason: {reason}\n")
    return {"fields": fields, "text": text}


# Field key -> numbered heading, in the order upstream front.py printed them.
NOTE_LAYOUT = (
    ("symptoms_reported", "Symptoms reported"),
    ("denies", "Explicitly denied"),
    ("current_treatment_discussed", "Current treatment discussed"),
    ("physical_findings", "Physical findings"),
    ("impression", "Impression / provisional diagnosis"),
    ("investigations", "Investigations suggested"),
    ("treatment_response", "Treatment response"),
    ("additional_remarks", "Additional remarks"),
)


def render_clinical_note_text(fields: dict) -> str:
    """Render the note text block from its fields.

    Fields are the single source of truth: the doctor edits fields during
    review, and the exported document must reflect those edits rather than a
    text block frozen at generation time.
    """
    lines = ["Clinical Notes:"]
    for index, (key, heading) in enumerate(NOTE_LAYOUT, start=1):
        value = fields.get(key)
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value) if value else "Not specified"
        lines.append(f"{index}. {heading}: {value if value not in (None, '') else 'Not specified'}")
    return "\n".join(lines) + "\n"
