"""Kerala colloquial clinical semantics.

Layer 2 on top of the raw transcript:
    raw -> asr_correction -> THIS LAYER -> existing scope engine

Maps what a patient/doctor MEANT to canonical clinical concepts without
inventing what they did not say. NOT ASR correction (asr_correction.py) and
NOT a symptom alias list (the base lexicon's job).

Context rules (all test-backed):
    low marker      -> derived low concept, NEVER the disease
    measurement     -> request for a reading, never a diagnosis
    amount question -> question about a reading, never a diagnosis
    control         -> chronic condition PRESENT (controlled), never ABSENT
    family subject  -> family history, never a patient finding
    unknown/garbage -> passes through untouched (UNKNOWN beats GUESS)
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from .normalization import (
    ABSENT,
    CURRENT,
    PAST,
    PRESENT,
    QUESTION,
    ClinicalEntity,
    load_lexicon,
)

SEMANTICS_PATH = Path(__file__).parent / "data" / "malayalam_clinical_semantics.json"

_semantics_cache: dict | None = None


def load_semantics(path: Path | str | None = None) -> dict:
    """Load (and cache) the semantics data file."""
    global _semantics_cache
    if _semantics_cache is None or path is not None:
        target = Path(path) if path else SEMANTICS_PATH
        data = json.loads(Path(target).read_text(encoding="utf-8"))
        if path is None:
            _semantics_cache = data
        return data
    return _semantics_cache


# ---------------------------------------------------------------------------
# Lexicon merge: colloquial aliases + new concepts flow into the base lexicon
# so the existing scope engine (negation scope, questions, temporality,
# conditionals, ellipsis) applies to them unchanged.
# ---------------------------------------------------------------------------
def normalization_lexicon_with_semantics(semantics: dict | None = None) -> dict:
    """Base lexicon merged with colloquial aliases from the semantics file.

    Merged entries carry ``merged_from_semantics`` so tests can prove the two
    layers stay separate: the base lexicon file is never touched.
    """
    sem = semantics or load_semantics()
    lex = json.loads(json.dumps(load_lexicon()))  # deep copy

    for entry in sem.get("colloquial_aliases", []):
        cid = entry["concept"]
        if cid not in lex["concepts"]:
            continue
        spec = lex["concepts"][cid]
        for alias in entry.get("add", []):
            if alias not in spec["aliases"]:
                spec["aliases"].append(alias)
        if entry.get("inherently_negative"):
            spec["inherently_negative"] = True
        spec["merged_from_semantics"] = True

    # Marker-list additions (e.g. fused conditional verb forms like െങ്കിൾ)
    # flow in here so the base lexicon file is never edited by the semantics layer.
    for key, additions in sem.get("lexicon_additions", {}).items():
        if key in lex and isinstance(lex[key], list):
            for item in additions:
                if item not in lex[key]:
                    lex[key].append(item)

    # Negative idioms: aliases whose literal wording embeds a negation ("ശ്വാസം
    # കിട്ടുന്നില്ല" = breath not coming). The idiom asserts the symptom; only a
    # negation landing on the concept word itself ("ശ്വാസംമുട്ടലില്ല") denies it.
    for entry in sem.get("negative_idioms", []):
        cid = entry["concept"]
        if cid in lex["concepts"]:
            idioms = lex["concepts"][cid].setdefault("negative_idioms", [])
            for alias in entry.get("aliases", []):
                if alias not in idioms:
                    idioms.append(alias)
            lex["concepts"][cid]["merged_from_semantics"] = True

    # Colloquial idioms for new concepts ("tablet കഴിക്കും" = takes the tablet):
    # add aliases to an existing concept, or define a brand-new one.
    for entry in sem.get("colloquial_idioms", []):
        cid = entry["concept"]
        adds = entry.get("add", [])
        if cid in lex["concepts"]:
            aliases = lex["concepts"][cid].setdefault("aliases", [])
            for alias in adds:
                if alias not in aliases:
                    aliases.append(alias)
            lex["concepts"][cid]["merged_from_semantics"] = True
        elif adds:
            lex["concepts"][cid] = {
                "english": entry.get("english", cid),
                "aliases": list(adds),
                "english_aliases": [entry.get("english", cid)],
                "medication": cid in sem.get("medications", {}),
                "merged_from_semantics": True,
            }

    for entry in sem.get("new_concepts", []):
        cid = entry["concept"]
        if cid in lex["concepts"]:
            continue
        lex["concepts"][cid] = {
            "english": entry["english"],
            "aliases": entry.get("aliases", []),
            "english_aliases": entry.get("english_aliases", []),
            "merged_from_semantics": True,
        }

    # Medications become first-class concepts so the scope engine sees them,
    # but they are flagged and routed separately (treatment, not symptom).
    for name, mspec in sem.get("medications", {}).items():
        if name in lex["concepts"]:
            continue
        lex["concepts"][name] = {
            "english": mspec["english"],
            "aliases": mspec.get("aliases", []),
            "english_aliases": [name],
            "medication": True,
            "merged_from_semantics": True,
        }

    # Controlled targets (BP/sugar): colloquial aliases must be recognizable by
    # the base matcher, including the English abbreviations real speech uses.
    for target in sem.get("controlled_targets", {}).values():
        cid = target["concept"]
        if cid not in lex["concepts"]:
            continue
        spec = lex["concepts"][cid]
        for alias in target.get("aliases_malayalam", []):
            if alias not in spec["aliases"]:
                spec["aliases"].append(alias)
        for alias in target.get("aliases_english", []):
            low = alias.lower()
            if low not in {a.lower() for a in spec["english_aliases"]}:
                spec["english_aliases"].append(low)
        spec["merged_from_semantics"] = True
    return lex


# Concept ids whose colloquial names are context-dependent measurements.
_CONTROLLED = {"hypertension_history": "hypertension", "diabetes_history": "diabetes"}

# Concepts that are treatment mentions, not findings. Routed separately so a
# medication never appears in the symptom list.
MEDICATION_CONCEPTS = {
    # existing (do not remove)
    "paracetamol", "tablet", "capsule", "syrup", "injection", "medicine",
    # Medication Intelligence V1 — P0 Indian primary-care generics
    # (medication_intelligence_v1_research.md §C / §K); every id has a
    # vocabulary entry in data/medication_lexicon.json.
    "ibuprofen", "aceclofenac", "diclofenac", "aspirin", "pantoprazole",
    "omeprazole", "domperidone", "ondansetron", "metformin", "glimepiride",
    "teneligliptin", "voglibose", "amlodipine", "cilnidipine", "telmisartan",
    "atorvastatin", "rosuvastatin", "clopidogrel", "amoxicillin",
    "amoxicillin_clavulanate", "azithromycin", "cefixime", "ciprofloxacin",
    "metronidazole", "salbutamol", "levosalbutamol", "budesonide",
    "montelukast", "cetirizine", "levocetirizine", "levothyroxine",
    "cholecalciferol", "folic_acid", "ferrous_sulfate", "methylcobalamin",
    "escitalopram", "clonazepam", "olanzapine", "prednisolone", "insulin",
    # Class-level (Layer 4) unknown concepts — never resolved to a molecule.
    "unknown_antihypertensive", "unknown_antidiabetic", "unknown_analgesic",
    "unknown_antibiotic", "unknown_other",
    # P1/P2 additions (research §C; vocabulary lives in
    # data/medication_lexicon.json): GI, diabetes, cardiology, respiratory,
    # antibiotics, neuro/psych, endocrine, supplements, dermatology.
    "mefenamic_acid", "etoricoxib", "naproxen", "tramadol", "rabeprazole",
    "esomeprazole", "lansoprazole", "metoclopramide", "sucralfate",
    "lactulose", "ranitidine", "gliclazide", "glipizide", "vildagliptin",
    "sitagliptin", "linagliptin", "dapagliflozin", "empagliflozin",
    "canagliflozin", "pioglitazone", "insulin_glargine", "insulin_degludec",
    "human_mixtard", "losartan", "olmesartan", "valsartan", "ramipril",
    "enalapril", "metoprolol", "atenolol", "bisoprolol", "carvedilol",
    "nebivolol", "fenofibrate", "chlorthalidone", "hydrochlorothiazide",
    "furosemide", "spironolactone", "isosorbide_mononitrate", "formoterol",
    "fluticasone", "tiotropium", "ipratropium", "fexofenadine", "bilastine",
    "doxofylline", "theophylline", "clarithromycin", "doxycycline",
    "cefuroxime", "cefpodoxime", "levofloxacin", "ornidazole",
    "nitrofurantoin", "co_trimoxazole", "sertraline", "paroxetine",
    "duloxetine", "venlafaxine", "amitriptyline", "alprazolam", "lorazepam",
    "diazepam", "quetiapine", "risperidone", "aripiprazole",
    "sodium_valproate", "valproic_acid", "levetiracetam", "carbamazepine",
    "lamotrigine", "phenytoin", "pregabalin", "gabapentin", "carbimazole",
    "propylthiouracil", "calcium_carbonate", "zinc", "pyridoxine",
    "dexamethasone", "methylprednisolone", "hydrocortisone", "clotrimazole",
    "terbinafine", "mupirocin", "permethrin", "penicillin",
}

_merged_lexicon_cache: dict | None = None


def _merged_lexicon() -> dict:
    global _merged_lexicon_cache
    if _merged_lexicon_cache is None:
        _merged_lexicon_cache = normalization_lexicon_with_semantics()
    return _merged_lexicon_cache


def _subject_of(clause: str, sem: dict) -> str:
    """patient | mother | father | family | unknown.

    Family markers win over patient markers: "അമ്മയ്ക്ക് ഷുഗർ ഉണ്ട്" mentions
    the dative marker ക്ക് twice, but the family word is the one that refers to
    the person the condition belongs to.
    """
    for subject, markers in sem.get("family_subject_markers", {}).items():
        if _contains_any(clause, markers):
            return subject
    if _contains_any(clause, sem.get("patient_self_markers", [])):
        return "patient"
    return "unknown"


# ---------------------------------------------------------------------------
# Post-pass: context resolution over the base engine's output
# ---------------------------------------------------------------------------
def _post_pass(entities: list[ClinicalEntity], sem: dict) -> list[ClinicalEntity]:
    """Apply context rules to the base engine's entities, in place.

    The base scope engine still decides polarity (negation, question,
    temporality); this pass only re-resolves what the base engine cannot know:
    that a "low" marker changes the concept, that a check request or a reading
    question is not a diagnosis, that "controlled" is not "absent", and who
    the mention belongs to.
    """
    targets = sem.get("controlled_targets", {})
    for e in entities:
        clause = _norm(e.source_clause)

        # Medication mention: treatment context, never a symptom finding.
        if e.concept in MEDICATION_CONCEPTS:
            e.context = "medication"
            continue

        # Allergen identification: "പെൻസിലിൻ അലർജി ഉണ്ട്".
        if e.concept == "allergy":
            for surface, english in sem.get("allergens", {}).items():
                if _norm(surface) in clause:
                    e.context = f"allergen:{english}"
                    break

        key = _CONTROLLED.get(e.concept)
        if not key:
            continue
        spec = targets.get(key)
        if not spec:
            continue

        e.subject = _subject_of(clause, sem)

        # Amount question ("ഷുഗർ എത്രയാണ്?"): the marker itself signals a
        # reading question, even without a '?' - "എത്രയാണ്" contains "ആണ്",
        # which the base engine would otherwise read as a positive marker.
        if (_contains_any(clause, spec.get("amount_question_markers_malayalam", []))
                or _contains_any_ci(clause, spec.get("amount_question_markers_english", []))):
            e.status = QUESTION
            e.confidence = max(e.confidence, 0.85)
            e.context = "amount_question"
            continue

        # Measurement/check request: "BP നോക്കണം", "sugar check ചെയ്യണം".
        if (_contains_any(clause, spec.get("measurement_markers_malayalam", []))
                or _contains_any_ci(clause, spec.get("measurement_markers_english", []))):
            e.status = QUESTION
            e.confidence = max(e.confidence, 0.85)
            e.context = "measurement_check"
            continue

        # Low marker: the derived low concept, never the disease. Polarity from
        # the base engine is preserved (a negated low reading stays ABSENT).
        if (_contains_any(clause, spec.get("low_markers_malayalam", []))
                or _contains_any_ci(clause, spec.get("low_markers_english", []))):
            e.concept = spec["low_derived_concept"]
            e.english = spec["low_derived_english"]
            e.context = "low_marker"
            continue

        # Control marker: chronic condition present and managed - never ABSENT.
        if (_contains_any(clause, spec.get("control_markers_malayalam", []))
                or _contains_any_ci(clause, spec.get("control_markers_english", []))):
            e.severity = "controlled"
            e.context = "controlled"
            if e.status == ABSENT:
                e.status = PRESENT
            continue

        # High marker: the disease with a high reading.
        if (_contains_any(clause, spec.get("high_markers_malayalam", []))
                or _contains_any_ci(clause, spec.get("high_markers_english", []))):
            e.context = "high"

        # Family subject: family history, never conflated with the patient.
        if e.subject in sem.get("family_subject_markers", {}):
            e.context = "family_history"

    return entities


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def semantic_normalize(text: str) -> list[ClinicalEntity]:
    """Normalize colloquial Malayalam/English clinical text to concepts.

    Same contract as normalize_clinical_text, plus the semantics layer: the
    input is never modified, unknown surface forms pass through untouched.
    """
    if not text or not isinstance(text, str):
        return []
    from .normalization import normalize_clinical_text

    lex = _merged_lexicon()
    entities = normalize_clinical_text(text, lex)
    return _post_pass(entities, load_semantics())



# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


def _contains_any(clause: str, markers: list[str]) -> bool:
    return any(_norm(m) in clause for m in markers)


def _contains_any_ci(clause: str, markers: list[str]) -> bool:
    low = clause.lower()
    return any(_norm(m).lower() in low for m in markers)
