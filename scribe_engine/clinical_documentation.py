# -*- coding: utf-8 -*-
"""Structured clinical documentation layer.

Sits ABOVE the existing semantic layer (scribe_engine.semantic /
clinical.build_normalized_entities) — it never replaces or bypasses it. The
semantic layer's ClinicalEntity records remain the single source of truth for
symptoms; this module organizes them plus speaker-segmented transcript lines
into a clinical document structure and renders a clinically readable note.

TRUTHFULNESS CONTRACT (mandatory)
---------------------------------
Every section fact is exactly one of:

    EXPLICIT_PRESENT   the patient/clinician explicitly reported it
    EXPLICIT_ABSENT    the patient/clinician explicitly denied it
    NOT_DOCUMENTED     the consultation never discussed it

NOT_DOCUMENTED is NEVER converted into ABSENT: silence is not a denial. Only
the semantic layer's explicit ABSENT/DENIED statuses may produce
EXPLICIT_ABSENT facts.

An LLM may later be used for controlled prose generation AFTER this structured
representation is built; the structured representation is the source of truth
and no fact may originate from free-form generation.
"""

from __future__ import annotations

import re

# Concepts classified as chronic/history conditions (render in PMH only).
_PMH_CONCEPTS = frozenset({
    "diabetes_history", "hypertension_history", "asthma", "heart_disease",
    "thyroid_disease", "kidney_disease", "tb_history",
})
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Truthfulness vocabulary
# ---------------------------------------------------------------------------
EXPLICIT_PRESENT = "EXPLICIT_PRESENT"
EXPLICIT_ABSENT = "EXPLICIT_ABSENT"
NOT_DOCUMENTED = "NOT_DOCUMENTED"

# Semantic-layer statuses that mean the patient/doctor explicitly said "no".
# QUESTION/UNKNOWN/POSSIBLE are explicitly NOT here: an unanswered question or
# an unsure patient is documented as such, never as an absence and never as a
# presence.
_ABSENT_STATUSES = {"ABSENT", "DENIED"}
# Statuses that assert nothing about the patient and must never become facts.
_NON_FINDING_STATUSES = {"QUESTION", "UNKNOWN", "POSSIBLE"}

_MEDICATION_CONCEPTS = {"paracetamol", "tablet", "capsule", "syrup", "injection",
                        "medicine"}


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------
@dataclass
class ClinicalFact:
    """One auditable clinical fact in the documentation layer.

    provenance links the fact back to: source speaker, the verbatim source
    clause (original Malayalam), the normalized concept and its status, and
    the semantic layer's confidence. The raw transcript itself is never
    modified anywhere in this module.
    """

    section: str
    text: str
    evidence: str                                    # EXPLICIT_PRESENT | EXPLICIT_ABSENT | NOT_DOCUMENTED
    concept: Optional[str] = None                    # normalized concept id, when concept-derived
    status: Optional[str] = None                     # semantic-layer status (PRESENT/RESOLVED/...)
    source_speaker: Optional[str] = None             # "patient" | "doctor" | "SPEAKER_UNKNOWN" | ...
    source_text: str = ""                            # verbatim source clause / line (original language)
    confidence: Optional[float] = None               # semantic-layer confidence
    attributes: dict = field(default_factory=dict)   # duration/severity/course/...

    def as_dict(self) -> dict:
        return {
            "section": self.section,
            "text": self.text,
            "evidence": self.evidence,
            "concept": self.concept,
            "status": self.status,
            "source_speaker": self.source_speaker,
            "source_text": self.source_text,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


# ---------------------------------------------------------------------------
# Line-level evidence scanning (speaker-segmented lines only)
# ---------------------------------------------------------------------------
# Past medical/surgical history: only lines containing an explicit ownership or
# denial marker qualify as evidence. A bare "ഷുഗർ" inside a doctor's question
# with no patient answer must NOT create a PMH fact — question exclusion is
# handled by _strip_questions before any line evidence is read.
_PMH_OWNER_ML = ("എനിക്ക്", "എന്റെ", "എനിക്കാണ്")
_PMH_OWNER_EN = ("i have", "i had", "i am a", "i was diagnosed", "known case of",
                 "history of", "diagnosed with", "suffering from")
_PMH_DENY_ML = ("ഇല്ല", "ഇല്ലെന്ന്", "ഉണ്ടായിരുന്നില്ല")
_PMH_DENY_EN = ("no history of", "never had", "do not have", "don't have",
                "denies", "not a known", "no known")

# Surgeries: explicit operation words + ownership/denial markers.
_SURGERY_PAT_ML = ("ഓപ്പറേഷൻ", "ശസ്ത്രക്രിയ", "സർജറി")
_SURGERY_PAT_EN = ("surgery", "operation", "operated", "hysterectomy", "appendectomy",
                   "cesarean", "c-section", "bypass", "stent")
_SURGERY_NEG_EN = ("no surgeries", "no operations", "never had surgery",
                   "never been operated", "no surgical history")

# Allergies.
_ALLERGY_PAT_ML = ("അലർജി",)
_ALLERGY_PAT_EN = ("allerg",)  # allergy/allergic/allergies
_ALLERGY_NEG_EN = ("no allergies", "no known allergies", "no drug allergies",
                   "not allergic", "no allergy")

# Social history: capture only explicitly discussed items.
_SOCIAL_PATTERNS_EN = (
    (r"\b(?:smok(?:e|es|ing|ed)|cigarette|beedi|beedis|tobacco)\b", "smoking", "smoking"),
    (r"\b(?:drinks? alcohol|drinking alcohol|alcohol(?:ic)? (?:drink|use|consumption)|beer|wine|whisk(?:e)?y|arrack)\b",
     "alcohol use", "alcohol"),
    (r"\b(?:recreational drugs|cannabis|marijuana|ganja)\b", "recreational drug use",
     "recreational_drugs"),
    (r"\b(?:i work(?:s|ed)? (?:as|at)|my job|occupation)\b", "occupation", "occupation"),
    (r"\b(?:vegetarian|non-?vegetarian|diet (?:is|consists))\b", "diet", "diet"),
)
_SOCIAL_ML = (
    ("പുകവലി", "smoking", "smoking"),
    ("മദ്യം", "alcohol use", "alcohol"),
    ("ലഹരി", "recreational drug use", "recreational_drugs"),
    ("ജോലി ചെയ്യുന്നു", "occupation", "occupation"),
)

_FAMILY_EN = ("my mother", "my father", "family history", "runs in my family",
              "in the family", "my sister", "my brother")
_FAMILY_ML = ("അമ്മയ്ക്ക്", "അച്ഛന്", "വീട്ടിൽ", "കുടുംബത്തിൽ")

# Physical examination: doctor lines with explicit exam verbs/findings.
_EXAM_PAT_EN = (
    (r"lungs? (?:are|is|sound|sounds)?\s*clear", "Lungs clear to auscultation"),
    (r"chest (?:is|sounds) clear", "Chest clear to auscultation"),
    (r"no (?:signs? of )?(?:wheez|rales|crackles|rhonchi)", "No adventitious lung sounds"),
    (r"no (?:peripheral )?oedema|no edema", "No peripheral oedema"),
    (r"\bbp\b (?:is )?\d{2,3}\s*/\s*\d{2,3}", "Blood pressure recorded"),
    (r"\btemp(?:erature)?\b (?:is )?\d{2}(?:\.\d)?", "Temperature recorded"),
    (r"\bpulse\b (?:is )?\d{2,3}", "Pulse recorded"),
    (r"\bspo2\b (?:is )?\d{2,3}|saturation (?:is )?\d{2,3}", "SpO2 recorded"),
    (r"abdomen (?:is )?(?:soft|tender|distended|normal)", "Abdominal examination documented"),
    (r"throat (?:is )?(?:clear|congested|red|injected)", "Throat examination documented"),
    (r"heart sounds? (?:are|is) normal", "Normal heart sounds"),
)
_EXAM_ML = (
    ("നെഞ്ച് ക്ലിയർ", "Chest clear on examination"),
    ("നെഞ്ച് വ്യക്തം", "Chest clear on examination"),
    ("ശ്വാസം ക്ലിയർ", "Lungs clear to auscultation"),
    ("ഓസ്കൾട്ടേഷൻ", "Auscultation performed"),
    ("ബിപി എടുത്തു", "Blood pressure recorded"),
)

# Plan: doctor lines with explicit actions.
_PLAN_PATTERNS_EN = (
    (r"\bi (?:will|am going to|have) (?:prescrib|start|add|give|order|advise)|"
     r"\b(?:start|take|continue)\b.{0,40}\b(?:tab(?:let)?s?|cap(?:sule)?s?|syrup|mg|ml)\b",
     "Medication advised/prescribed"),
    (r"\b(?:get|do|order|advise|send)\b.{0,40}\b(?:blood test|cbc|x-?ray|scan|ultrasound|ecg|echo|"
     r"blood sugar|sugar test|lipid|culture|ct scan|mri)\b", "Investigation advised"),
    (r"\b(?:refer|referral|see a|consult)\b.{0,40}\b(?:specialist|cardiolog|neurolog|ent|"
     r"dermatolog|ortho|surgeon|physician)\b", "Referral advised"),
    (r"\b(?:come back|follow.?up|review)\b.{0,40}\b(?:\d+\s*(?:day|week|month)|if|tomorrow)\b",
     "Follow-up advised"),
    (r"\b(?:rest|fluids|plenty of water|steam|gargle|avoid)\b", "General advice given"),
)
_PLAN_ML = (
    ("വിശ്രമം", "Rest advised"),
    ("വെള്ളം കുടി", "Fluids advised"),
    ("സ്റ്റീം", "Steam inhalation advised"),
    ("ടെസ്റ്റ്", "Investigation advised"),
    ("സ്കാൻ", "Investigation advised"),
    ("എക്സ്-റേ", "Investigation advised"),
    ("രക്തപരിശോധന", "Investigation advised"),
)

# Doctor-stated diagnosis lines.
_DIAGNOSIS_PAT_EN = (
    r"\byou (?:have|had|are having|'ve got)\b",
    r"\bthis (?:is|looks like)\b",
    r"\bit(?:'s| is) (?:a|an)\b",
    r"\bdiagnos(?:is|ed)\b",
)
_DIAGNOSIS_ML = ("വൈറൽ പനി", "ന്യുമോണിയ", "ബ്രോങ്കൈറ്റിസ്", "ടൈഫോയ്ഡ്",
                 "ഡെംഗിവ", "അലർജി പനി", "ഗാസ്ട്രൈറ്റിസ്")

# Questions to strip from doctor evidence (a question is never a finding).
_QUESTION_PAT_EN = re.compile(
    # English heuristic is deliberately conservative: an interrogative counts
    # only when the line STARTS with an auxiliary/wh-word (subject-auxiliary
    # inversion — the actual English question signal) or contains '?'.
    # Matching auxiliaries anywhere would misclassify statements like
    # "Lungs are clear" as questions.
    r"^(?:do|does|did|are|is|was|were|have|has|had|can|could|would|will|"
    r"any|how|what|when|where|why)\b", re.I)
_QUESTION_ML_MARKERS = ("ഉണ്ടോ", "ഇല്ലേ", "ആണോ", "എത്ര", "എപ്പോൾ")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ci(text: str, needles: tuple) -> bool:
    """Case-insensitive containment for Latin-script needles."""
    low = (text or "").lower()
    return any(n.lower() in low for n in needles)


def _attr_suffix(e: dict) -> str:
    """Explicit-attribute suffix for HPI sentences ("for 6 days", "severe").

    Only attributes the semantic layer extracted appear — nothing is inferred.
    """
    parts = []
    if e.get("duration"):
        parts.append(f"for {e['duration']}")
    if e.get("severity"):
        parts.append(f"described as {e['severity']}")
    if e.get("frequency"):
        parts.append(f"occurring {e['frequency']}")
    if e.get("temporality") == "recurrent":
        parts.append("(recurrent)")
    return " " + " ".join(parts) if parts else ""


def _is_question(line: str) -> bool:
    if "?" in line:
        return True
    if _ci(line, _QUESTION_ML_MARKERS):
        return True
    return bool(_QUESTION_PAT_EN.search(line or ""))


def _fact_speaker(entity_dict: dict, roles_known: bool = True) -> str:
    """Honest source_speaker for a concept-derived fact.

    The conversational default (per clinical.py): a non-question mention with no
    explicit subject marker is the patient describing themselves. Family
    markers stay attributed to the family member. Unknown roles can never
    upgrade to Doctor/Patient, but a PRESENT finding is by definition the
    patient's report, so the default remains "patient".
    """
    subject = entity_dict.get("subject") or "unknown"
    if subject in ("mother", "father", "family"):
        return subject
    return "patient"


def _strip_questions(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln and not _is_question(ln)]


def _fact_speaker(entity_dict: dict) -> str:
    """Honest source_speaker for a concept-derived fact.

    The conversational default (same rule clinical.py applies): a non-question
    mention with no explicit subject marker is the patient describing
    themselves. Family markers stay attributed to the family member.
    """
    subject = entity_dict.get("subject") or "unknown"
    if subject in ("mother", "father", "family"):
        return subject
    return "patient"


# ---------------------------------------------------------------------------
# Section registry
# ---------------------------------------------------------------------------
SECTION_ORDER = (
    "chief_complaint", "hpi", "associated_symptoms", "pertinent_negatives",
    "pmh", "psh", "medications", "allergies", "family_history", "social_history",
    "ros", "physical_exam", "assessment", "plan",
)

SECTION_TITLES = {
    "chief_complaint": "Chief Complaint",
    "hpi": "History of Present Illness",
    "associated_symptoms": "Associated Symptoms",
    "pertinent_negatives": "Pertinent Negatives",
    "pmh": "Past Medical History",
    "psh": "Past Surgical History",
    "medications": "Medications",
    "allergies": "Allergies",
    "family_history": "Family History",
    "social_history": "Social History",
    "ros": "Review of Systems",
    "physical_exam": "Physical Examination",
    "assessment": "Assessment",
    "plan": "Plan",
}

# Rough ROS grouping: concept -> organ system. Only systems with documented
# evidence are emitted (never auto-populate every body system).
_ROS_GROUPS = {
    "constitutional": ("fever", "fatigue", "weakness", "body_ache", "chills", "sweating",
                       "night_sweats", "loss_of_appetite"),
    "heent": ("headache", "sore_throat", "runny_nose", "nasal_congestion", "sneezing",
              "blurred_vision", "double_vision", "hearing_loss", "tinnitus",
              "ear_blockage", "difficulty_swallowing", "painful_swallowing",
              "hoarseness", "voice_change", "loss_of_smell", "loss_of_taste",
              "toothache", "ear_pain"),
    "respiratory": ("cough", "dry_cough", "productive_cough", "phlegm",
                    "shortness_of_breath", "chest_tightness", "chest_heaviness",
                    "burning_chest_pain"),
    "cardiovascular": ("chest_pain", "palpitations"),
    "gastrointestinal": ("nausea", "vomiting", "diarrhea", "constipation",
                         "abdominal_pain", "bloating", "abdominal_gas", "heartburn",
                         "acid_reflux", "abdominal_burning"),
    "genitourinary": ("painful_urination", "urinary_frequency", "reduced_urine_output",
                      "blood_in_urine"),
    "musculoskeletal": ("muscle_pain", "back_pain"),
    "neurological": ("dizziness", "numbness", "tingling", "tremor", "insomnia",
                     "excessive_sleepiness"),
    "skin": ("itching", "rash", "skin_lesion", "swelling"),
}
# Symptoms that are not ROS-eligible (functional/problem statements).
_ROS_EXCLUDED = {"difficulty_eating"}

_ROS_LABELS = {
    "constitutional": "Constitutional", "heent": "HEENT",
    "respiratory": "Respiratory", "cardiovascular": "Cardiovascular",
    "gastrointestinal": "Gastrointestinal", "genitourinary": "Genitourinary",
    "musculoskeletal": "Musculoskeletal", "neurological": "Neurological",
    "skin": "Skin",
}

# Chronic-condition concepts the semantics layer already classifies with
# subject attribution; they populate PMH when owned by the patient.
_PMH_CONCEPTS = {"hypertension_history", "diabetes_history", "asthma",
                 "thyroid_disorder", "high_cholesterol", "ulcer", "heart_disease",
                 "kidney_disease", "copd"}


def _ros_group_for(concept: str) -> Optional[str]:
    for group, concepts in _ROS_GROUPS.items():
        if concept in concepts:
            return group
    return None


def _provenance(concept: Optional[str], speaker, text: str,
                confidence: Optional[float]) -> dict:
    return {
        "concept": concept,
        "source_speaker": speaker,
        "source_text": text,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------
def build_structured_documentation(
    entities_dicts: list[dict],
    turns: list[dict],
    roles_known: bool = True,
) -> dict:
    """Build the structured clinical representation from semantic-layer output.

    Args:
        entities_dicts: ClinicalEntity.as_dict() records from the semantic layer
            (clinical.build_normalized_entities(...)["entities"]).
        turns: speaker turns from diarization.build_labeled_transcript(...):
            [{"speaker": "Doctor"|"Patient"|"Speaker 0"|..., "text": ...}, ...].
        roles_known: False when speaker roles could not be attributed; then
            line-based sections scan all lines but label the speaker honestly
            as SPEAKER_UNKNOWN — never "Doctor"/"Patient" by guesswork.
    """
    sections: dict[str, list[ClinicalFact]] = {k: [] for k in SECTION_ORDER}
    entities = [e for e in (entities_dicts or []) if isinstance(e, dict)]
    turns = turns or []

    doctor_lines = [t.get("text", "") for t in turns if t.get("speaker") == "Doctor"]
    patient_lines = [t.get("text", "") for t in turns if t.get("speaker") == "Patient"]
    if not roles_known:
        doctor_lines = patient_lines = [t.get("text", "") for t in turns]

    def speaker(pool: str) -> str:
        """Honest speaker label for line-derived facts."""
        return pool if roles_known else "SPEAKER_UNKNOWN"

    # -- Semantic-layer symptom partition (source of truth) ------------------
    symptom_e = [e for e in entities if e.get("concept") not in _MEDICATION_CONCEPTS]
    present = [e for e in symptom_e
               if e.get("status") == "PRESENT"
               and e.get("subject") not in ("mother", "father", "family")]
    resolved = [e for e in symptom_e
                if e.get("status") == "RESOLVED"
                and e.get("subject") not in ("mother", "father", "family")]
    denied = [e for e in symptom_e
              if e.get("status") in _ABSENT_STATUSES
              and e.get("subject") not in ("mother", "father", "family")]
    family = [e for e in symptom_e
              if e.get("subject") in ("mother", "father", "family")
              and e.get("status") not in _NON_FINDING_STATUSES]
    allergy = [e for e in symptom_e if e.get("concept") == "allergy"]

    # --- 1. Chief complaint: first PRESENT symptom -------------------------
    if present:
        cc = present[0]
        sections["chief_complaint"].append(ClinicalFact(
            "chief_complaint", cc.get("english") or cc.get("concept"), EXPLICIT_PRESENT,
            concept=cc.get("concept"), status=cc.get("status"),
            source_speaker=_fact_speaker(cc),
            source_text=cc.get("source_clause") or cc.get("surface_text") or "",
            confidence=cc.get("confidence")))

    # --- 2. HPI: narrative reconstructed from structured facts only --------
    # Order mirrors the consultation: resolved problems first ("had fever, it
    # resolved"), then ongoing ones — matching how the patient narrated them.
    hpi_sources = resolved + present
    if hpi_sources:
        sentences = []
        for e in resolved:
            sentences.append(
                f"The patient reports a history of {(e.get('english') or 'the symptom')}"
                f"{_attr_suffix(e)}, which has since resolved.")
        for e in present:
            if (e.get("source_clause") or "") and "ഇപ്പോഴും" in (e.get("source_clause") or ""):
                sentences.append(
                    f"The patient continues to experience {e.get('english')}"
                    f"{_attr_suffix(e)}.")
            else:
                sentences.append(
                    f"The patient reports {e.get('english')}{_attr_suffix(e)}.")
        hpi_text = " ".join(sentences)
        evidence = EXPLICIT_PRESENT
    else:
        hpi_text = "No symptom history documented in the available consultation."
        evidence = NOT_DOCUMENTED
    sections["hpi"].append(ClinicalFact(
        "hpi", hpi_text, evidence, source_speaker="patient" if hpi_sources else None,
        source_text=" | ".join((e.get("source_clause") or "")
                               for e in hpi_sources if e.get("source_clause")),
        confidence=(min(e.get("confidence") or 0 for e in hpi_sources)
                    if hpi_sources else None)))

    # --- 3. Associated symptoms: PRESENT beyond the chief complaint --------
    for e in present[1:]:
        sections["associated_symptoms"].append(ClinicalFact(
            "associated_symptoms", e.get("english") or e.get("concept"),
            EXPLICIT_PRESENT, concept=e.get("concept"), status=e.get("status"),
            source_speaker=_fact_speaker(e),
            source_text=e.get("source_clause") or e.get("surface_text") or "",
            confidence=e.get("confidence"),
            attributes={k: e[k] for k in ("duration", "severity", "frequency",
                                          "body_location", "pain_quality") if e.get(k)}))

    # --- 4. Pertinent negatives: explicit denials only ---------------------
    for e in denied:
        sections["pertinent_negatives"].append(ClinicalFact(
            "pertinent_negatives", f"Denies {e.get('english')}", EXPLICIT_ABSENT,
            concept=e.get("concept"), status=e.get("status"),
            source_speaker=_fact_speaker(e),
            source_text=e.get("source_clause") or e.get("surface_text") or "",
            confidence=e.get("confidence")))

    # --- 5. PMH: semantic-layer chronic conditions owned by the patient ----
    pmh_found = False
    for e in entities:
        c = e.get("concept") or ""
        if c not in _PMH_CONCEPTS:
            continue
        if e.get("subject") in ("mother", "father", "family"):
            continue  # family history, handled below
        if e.get("status") in _NON_FINDING_STATUSES:
            continue  # an unanswered question about diabetes is not PMH
        ev = EXPLICIT_ABSENT if e.get("status") in _ABSENT_STATUSES else EXPLICIT_PRESENT
        sections["pmh"].append(ClinicalFact(
            "pmh", e.get("english") or c, ev, concept=c, status=e.get("status"),
            source_speaker=_fact_speaker(e),
            source_text=e.get("source_clause") or e.get("surface_text") or "",
            confidence=e.get("confidence"),
            attributes={"controlled": True} if e.get("severity") == "controlled" else {}))
        pmh_found = True
    if not pmh_found:
        sections["pmh"].append(ClinicalFact(
            "pmh", "No past medical history documented", NOT_DOCUMENTED,
            source_speaker=None, source_text=""))

    # --- 6. PSH: explicit line evidence only -------------------------------
    psh_found = False
    for ln in _strip_questions(patient_lines):
        if _ci(ln, _SURGERY_NEG_EN):
            sections["psh"].append(ClinicalFact(
                "psh", "No prior surgeries reported", EXPLICIT_ABSENT,
                source_speaker=speaker("patient"), source_text=ln))
            psh_found = True
            break
        if (_ci(ln, _SURGERY_PAT_EN) and _ci(ln, _PMH_OWNER_EN)) or \
           (any(m in ln for m in _SURGERY_PAT_ML) and any(m in ln for m in _PMH_OWNER_ML)):
            sections["psh"].append(ClinicalFact(
                "psh", f"Surgical history: {ln.strip()}", EXPLICIT_PRESENT,
                source_speaker=speaker("patient"), source_text=ln))
            psh_found = True
            break
    if not psh_found:
        sections["psh"].append(ClinicalFact(
            "psh", "No surgical history documented", NOT_DOCUMENTED,
            source_speaker=None, source_text=""))

    # --- 7. Medications: semantic-layer medication entities only -----------
    med_entities = [e for e in entities
                    if e.get("concept") in _MEDICATION_CONCEPTS
                    and e.get("status") in ("PRESENT", "RESOLVED")]
    if med_entities:
        for e in med_entities:
            sections["medications"].append(ClinicalFact(
                "medications", f"{e.get('english')} (mentioned)", EXPLICIT_PRESENT,
                concept=e.get("concept"), status=e.get("status"),
                source_speaker=_fact_speaker(e),
                source_text=e.get("source_clause") or e.get("surface_text") or "",
                confidence=e.get("confidence"),
                attributes={k: e[k] for k in ("duration", "frequency") if e.get(k)}))
    else:
        sections["medications"].append(ClinicalFact(
            "medications", "No medications documented", NOT_DOCUMENTED,
            source_speaker=None, source_text=""))

    # --- 8. Allergies: semantic-layer first, then explicit line evidence ---
    if allergy:
        for e in allergy:
            ctx = e.get("context") or ""
            allergen = ctx.split(":", 1)[1] if ctx.startswith("allergen:") else None
            text = f"Allergy to {allergen}" if allergen else "Allergy reported"
            ev = (EXPLICIT_ABSENT if e.get("status") in _ABSENT_STATUSES
                  else EXPLICIT_PRESENT)
            sections["allergies"].append(ClinicalFact(
                "allergies", text, ev, concept="allergy", status=e.get("status"),
                source_speaker=_fact_speaker(e),
                source_text=e.get("source_clause") or e.get("surface_text") or "",
                confidence=e.get("confidence")))
    else:
        for ln in _strip_questions(patient_lines + doctor_lines):
            if _ci(ln, _ALLERGY_NEG_EN):
                sections["allergies"].append(ClinicalFact(
                    "allergies", "No known allergies", EXPLICIT_ABSENT,
                    source_speaker=speaker("patient"), source_text=ln))
                break
            if _ci(ln, _ALLERGY_PAT_EN) or any(m in ln for m in _ALLERGY_PAT_ML):
                sections["allergies"].append(ClinicalFact(
                    "allergies", f"Allergy reported: {ln.strip()}", EXPLICIT_PRESENT,
                    source_speaker=speaker("patient"), source_text=ln))
                break
        else:
            sections["allergies"].append(ClinicalFact(
                "allergies", "No allergies documented", NOT_DOCUMENTED,
                source_speaker=None, source_text=""))

    # --- 9. Family history: semantic-layer family subjects are ----------
    #     authoritative; explicit line evidence is the fallback.
    if family:
        for e in family:
            sections["family_history"].append(ClinicalFact(
                "family_history", f"{e.get('english')} (family history)", EXPLICIT_PRESENT,
                concept=e.get("concept"), status=e.get("status"),
                source_speaker=e.get("subject") or "family",
                source_text=e.get("source_clause") or e.get("surface_text") or "",
                confidence=e.get("confidence")))
    else:
        for ln in _strip_questions(patient_lines + doctor_lines):
            if _ci(ln, _FAMILY_EN) or any(m in ln for m in _FAMILY_ML):
                sections["family_history"].append(ClinicalFact(
                    "family_history", f"Family history: {ln.strip()}", EXPLICIT_PRESENT,
                    source_speaker=speaker("patient"), source_text=ln))
                break
        else:
            sections["family_history"].append(ClinicalFact(
                "family_history", "No family history documented", NOT_DOCUMENTED,
                source_speaker=None, source_text=""))

    # --- 10. Social history: only explicitly discussed items ---------------
    for ln in _strip_questions(patient_lines + doctor_lines):
        for pat, label, concept in _SOCIAL_PATTERNS_EN:
            if re.search(pat, ln, re.I):
                sections["social_history"].append(ClinicalFact(
                    "social_history", label, EXPLICIT_PRESENT, concept=concept,
                    source_speaker=speaker("patient"), source_text=ln))
        for marker, label, concept in _SOCIAL_ML:
            if marker in ln:
                sections["social_history"].append(ClinicalFact(
                    "social_history", label, EXPLICIT_PRESENT, concept=concept,
                    source_speaker=speaker("patient"), source_text=ln))
    if not sections["social_history"]:
        sections["social_history"].append(ClinicalFact(
            "social_history", "No social history documented", NOT_DOCUMENTED,
            source_speaker=None, source_text=""))

    # --- 11. ROS: group only systems with documented evidence --------------
    ros_groups: dict[str, list[str]] = {}
    for e in present + resolved + denied:
        concept = e.get("concept") or ""
        if concept in _ROS_EXCLUDED:
            continue
        group = _ros_group_for(concept)
        if not group:
            continue
        if e.get("status") in _ABSENT_STATUSES:
            ros_groups.setdefault(group, []).append(f"Denies {e.get('english')}")
        elif e.get("status") == "RESOLVED":
            ros_groups.setdefault(group, []).append(f"{e.get('english')} (resolved)")
        else:
            ros_groups.setdefault(group, []).append(e.get("english"))
    for group in sorted(ros_groups, key=lambda g: list(_ROS_LABELS).index(g)):
        sections["ros"].append(ClinicalFact(
            "ros", f"{_ROS_LABELS[group]}: " + "; ".join(ros_groups[group]),
            EXPLICIT_PRESENT, source_speaker="patient"))
    if not sections["ros"]:
        sections["ros"].append(ClinicalFact(
            "ros", "No review of systems documented", NOT_DOCUMENTED,
            source_speaker=None, source_text=""))

    # --- 12. Physical examination: doctor's explicit statements only -------
    for ln in _strip_questions(doctor_lines):
        for pat, finding in _EXAM_PAT_EN:
            if re.search(pat, ln, re.I):
                sections["physical_exam"].append(ClinicalFact(
                    "physical_exam", finding, EXPLICIT_PRESENT,
                    source_speaker=speaker("doctor"), source_text=ln))
        for marker, finding in _EXAM_ML:
            if marker in ln:
                sections["physical_exam"].append(ClinicalFact(
                    "physical_exam", finding, EXPLICIT_PRESENT,
                    source_speaker=speaker("doctor"), source_text=ln))
    if not sections["physical_exam"]:
        sections["physical_exam"].append(ClinicalFact(
            "physical_exam", "Not documented", NOT_DOCUMENTED,
            source_speaker=None, source_text=""))

    # --- 13. Assessment: ONLY a doctor-stated diagnosis --------------------
    for ln in _strip_questions(doctor_lines):
        if _ci(ln, _DIAGNOSIS_ML) or re.search("|".join(_DIAGNOSIS_PAT_EN), ln, re.I):
            sections["assessment"].append(ClinicalFact(
                "assessment", f"Clinician-stated: {ln.strip()}", EXPLICIT_PRESENT,
                source_speaker=speaker("doctor"), source_text=ln))
    if not sections["assessment"]:
        sections["assessment"].append(ClinicalFact(
            "assessment",
            "No definitive diagnosis documented in the available consultation.",
            NOT_DOCUMENTED, source_speaker=None, source_text=""))

    # --- 14. Plan: doctor's explicit actions -------------------------------
    for ln in _strip_questions(doctor_lines):
        for pat, label in _PLAN_PATTERNS_EN:
            if re.search(pat, ln, re.I):
                sections["plan"].append(ClinicalFact(
                    "plan", label, EXPLICIT_PRESENT,
                    source_speaker=speaker("doctor"), source_text=ln))
        for marker, label in _PLAN_ML:
            if marker in ln:
                sections["plan"].append(ClinicalFact(
                    "plan", label, EXPLICIT_PRESENT,
                    source_speaker=speaker("doctor"), source_text=ln))
    if not sections["plan"]:
        sections["plan"].append(ClinicalFact(
            "plan", "Not documented", NOT_DOCUMENTED,
            source_speaker=None, source_text=""))

    # --- Provenance ledger: every fact, auditable --------------------------
    provenance = [f.as_dict() for k in SECTION_ORDER for f in sections[k]]

    return {
        "sections": {k: [f.as_dict() for f in sections[k]] for k in SECTION_ORDER},
        "provenance": provenance,
        "truthfulness_legend": {
            "EXPLICIT_PRESENT": "explicitly reported in the consultation",
            "EXPLICIT_ABSENT": "explicitly denied in the consultation",
            "NOT_DOCUMENTED": "not discussed in the consultation",
        },
        "not_documented_is_not_absent": True,
        "structured_representation_is_source_of_truth": True,
        "speaker_roles_known": bool(roles_known),
    }


# ---------------------------------------------------------------------------
# Note rendering
# ---------------------------------------------------------------------------
def render_clinical_documentation(doc: dict) -> str:
    """Render the 14-section note text from the structured representation.

    NOT_DOCUMENTED sections print "Not documented." — never a silent skip and
    never a fabricated finding. The structured sections dict is the source of
    truth; the text is derived from it and from nothing else.
    """
    lines: list[str] = ["CLINICAL DOCUMENTATION", ""]
    for i, key in enumerate(SECTION_ORDER, 1):
        facts = doc["sections"].get(key, [])
        lines.append(f"{i}. {SECTION_TITLES[key]}")
        if not facts or all(f.get("evidence") == NOT_DOCUMENTED for f in facts):
            lines.append("   Not documented.")
            lines.append("")
            continue
        for f in facts:
            if f["evidence"] == NOT_DOCUMENTED:
                lines.append(f"   {f['text']}")
            else:
                lines.append(f"   - {f['text']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
