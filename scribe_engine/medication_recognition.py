"""Medication Intelligence V1 — recognition, structure, and status.

Implements the research specification (medication_intelligence_v1_research.md,
§K) as a standalone module called from ``clinical_facts.build_clinical_facts``
after the base fact graph is assembled:

  * token-boundary, exact-match medication scanning (no fuzzy matching),
  * brand → generic normalization (the transcript is never rewritten),
  * context-gated short aliases ("para", "pan", "telma", "rosu", "iron"),
  * structured dose / dose_form / frequency (incl. the Indian 1-0-1 matrix) /
    timing / route / duration extraction — null when not spoken,
  * the deterministic 12-state medication-status ontology with the research
    priority chain (§B.2) and the English (§F.1) + Malayalam (§F.2) triggers,
  * class-level mentions ("BP tablet") as UNCERTAIN unknown_[class] facts —
    never resolved to a molecule (FP-4).

The module contains zero clinical decision logic beyond the researched
recognition rules: it never invents a medication name, never promotes a
question into an active medication, and never modifies the raw transcript.
Every medication fact remains traceable to its source span in the evidence
text (the fail-closed validator keeps enforcing that).
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Optional

_DATA_PATH = Path(__file__).parent / "data" / "medication_lexicon.json"
_lexicon_cache: dict | None = None

# ---------------------------------------------------------------------------
# Status ontology (research §B.2) and its mapping onto the engine's closed
# status vocabulary. The full 12-state precision lives on
# attributes["medication_status"]; the engine status keeps the fail-closed
# validator and the deterministic renderer working unchanged.
# ---------------------------------------------------------------------------
STATUS_PRIORITY = (
    "ALLERGY", "REFUSED", "DISCONTINUED", "PRESCRIBED", "CONTINUED",
    "CURRENT", "HISTORICAL", "RECOMMENDED", "CONSIDERED",
    "QUESTIONED", "NOT_ADHERENT", "UNCERTAIN",
)

# medication_status -> (engine status, engine certainty)
STATUS_TO_ENGINE = {
    "PRESCRIBED": ("RECOMMENDED", "RECOMMENDED"),
    "CONTINUED": ("PRESENT", "CONFIRMED"),
    "CURRENT": ("PRESENT", "CONFIRMED"),
    "HISTORICAL": ("HISTORICAL", "CONFIRMED"),
    "DISCONTINUED": ("RESOLVED", "CONFIRMED"),
    "CONSIDERED": ("CONSIDERED", "CONSIDERED"),
    "RECOMMENDED": ("RECOMMENDED", "RECOMMENDED"),
    "QUESTIONED": ("QUESTIONED", "UNCERTAIN"),
    "REFUSED": ("ABSENT", "DENIED"),
    "NOT_ADHERENT": ("UNCERTAIN", "UNCERTAIN"),
    "ALLERGY": ("PRESENT", "CONFIRMED"),
    "UNCERTAIN": ("UNCERTAIN", "UNCERTAIN"),
}

# medication_status -> note section (all values exist in the engine SECTIONS)
STATUS_TO_SECTION = {
    "PRESCRIBED": "plan",
    "CONTINUED": "medications",
    "CURRENT": "medications",
    "HISTORICAL": "pmh",
    "DISCONTINUED": "medications",
    "CONSIDERED": "plan",
    "RECOMMENDED": "plan",
    "QUESTIONED": "other_documentation",
    "REFUSED": "medications",
    "NOT_ADHERENT": "medications",
    "ALLERGY": "allergies",
    "UNCERTAIN": "medications",
}

# Fallback when the clause carries no explicit medication trigger: the base
# layer's status is preserved (mapped), so existing behaviour never regresses.
_EXISTING_TO_MED_STATUS = {
    "PRESENT": "CURRENT",
    "QUESTIONED": "QUESTIONED",
    "HISTORICAL": "HISTORICAL",
    "RESOLVED": "HISTORICAL",
    "CONSIDERED": "CONSIDERED",
    "RECOMMENDED": "RECOMMENDED",
    "UNCERTAIN": "UNCERTAIN",
    "ABSENT": "REFUSED",
}

# English status triggers (research §F.1). Ordered checks; each entry is
# (medication_status, pattern). A clause may match several — the priority
# chain resolves the winner.
_ENGLISH_TRIGGERS: tuple[tuple[str, str], ...] = (
    ("ALLERGY",
     r"\ballergic to\b|\ballergy to\b|\breacts to\b|\bdevelops? (?:a )?"
     r"(?:rash|urticaria|hives|angioedema|anaphyla\w*) with\b|\banaphyla\w* to\b"
     r"|\bcannot tolerate\b|\bside effect(?:s)? (?:with|from|to)\b"),
    ("REFUSED",
     r"\b(?:does not|doesn'?t|did not|didn'?t|will not|won'?t|refus(?:e|ed|es)"
     r"|declin(?:e|ed|es))\s+(?:want|wants|to take|taking|start|started)?\s*"
     r"(?:to )?(?:take|taking|start|starting)?\b|\bnot willing to take\b"
     r"|\brefused\b|\bdeclined\b"),
    ("DISCONTINUED",
     r"\bstopped (?:taking )?\b|\bdiscontinued\b|\bcame off\b|\bno longer on\b"
     r"|\bwas stopped\b|\boff \b|\bput a stop to\b|\bnirthi\b|\bnirthiyittu\b"),
    ("PRESCRIBED",
     r"\bstart(?:ed|ing)? (?:you on |him on |her on |them on |the )?\b"
     r"|\bprescri(?:be|bed|bing)\b|\bi(?:'| a)?ll give you\b|\badding\b"
     r"|\bhere is a prescription for\b|\bകഴിക്കണം\b|\bകഴിക്കൂ\b|\bതുടങ്ങാം\b"
     r"|\bwants to start\b|\bwant to start\b"),
    ("CONSIDERED",
     r"\bwants? to start\b"),
    ("CONTINUED",
     r"\bcontinue(?:s|d)? (?:with |taking |the )?\b|\bcarry on with\b"
     r"|\bkeep taking\b|\bmaintain(?:ing)?\b|\bsame .* as before\b"
     r"|\bcontinue ചെയ്യൂ\b"),
    ("CURRENT",
     r"\b(?:am|is|are) taking\b|\b(?:i|he|she|we|they) take(?:s)?\b"
     r"|\bcurrently on\b|\bon \b|\bprescribed .* by\b|\bകഴിക്കുന്നുണ്ട്\b"
     r"|\bകഴിക്കുന്നു\b"),
    ("HISTORICAL",
     r"\bused to (?:take|be on|use)\b|\bwas on\b|\bwas taking\b"
     r"|\bpreviously (?:on|taking|took)\b|\bin the past\b|\byears ago\b"
     r"|\bകഴിച്ചിരുന്നു\b"),
    ("RECOMMENDED",
     r"\b(?:you|he|she) should take\b|\badvised (?:starting |to start |taking )\b"
     r"|\brecommend(?:ed|ing)?\b|\bsuggest(?:ed|ing)?\b"),
    ("CONSIDERED",
     r"\bconsider (?:starting |adding )?\b|\bmight benefit from\b"
     r"|\bthinking of starting\b|\bwould consider\b|\boption would be\b"
     r"|\bwe could try\b|\bif .* (?:persists|worsens|not controlled)\b"),
    ("NOT_ADHERENT",
     r"\bhas the .* but (?:is )?not taking\b|\bnot taking .* regularly\b"
     r"|\bnot compliant\b|\bskipping\b|\bmisses doses\b|\bകഴിക്കുന്നില്ല\b"),
)

# Clause fragments that DISCHARGE a refusal reading: "he doesn't remember the
# name" is a memory statement about the NAME, not a refusal of the tablet.
_REFUSAL_DISCHARGER_RE = re.compile(
    r"\b(?:don'?t|doesn'?t|do not|does not|didn'?t|did not) (?:remember|know)\b"
    r"|\b(?:doesn'?t|doesn't) remember the name\b|\bnot sure (?:of|about) the name\b"
    r"|\bname .{0,12}(?:unknown|not known|forgotten)\b", re.I)

# Malayalam triggers (research §F.2) — checked against the raw clause.
_MALAYALAM_TRIGGERS: tuple[tuple[str, str], ...] = (
    ("ALLERGY", r"അലർജി ഉണ്ട്"),
    ("DISCONTINUED", r"നിർത്തി(യിരിക്കുന്നു)?"),
    ("PRESCRIBED", r"കഴിക്കണം|കഴിക്കൂ|തുടങ്ങാം"),
    ("CONTINUED", r"continue ചെയ്യൂ"),
    ("CURRENT", r"കഴിക്കുന്നുണ്ട്|കഴിക്കുന്നു"),
    ("HISTORICAL", r"കഴിച്ചിരുന്നു"),
    ("QUESTIONED", r"കഴിക്കണോ\?|കഴിക്കുന്നുണ്ടോ\?"),
    ("REFUSED", r"കഴിക്കുന്നില്ല"),
    ("UNCERTAIN", r"ഏതോ .*ഗുളിക|tablet name അറിയില്ല|പേര് അറിയില്ല"),
)

# Dictation-markup guard (FP-2). Kept aligned with clinical_facts._MARKUP_RE;
# duplicated here so this module is safely testable standalone (a medication
# alias must never match inside a markup command span).
_MARKUP_RE = re.compile(
    r"\b(?:new|next)\s+(?:para(?:graph)?|line|section|heading|page)\b"
    r"|\bparagraph\b|\bfull stop\b|\bcomma\b|\bopen brackets?\b|\bclose brackets?\b"
    r"|\bnew line\b|\bbold\b|\bunderline[ds]?\b|\bnew para\b|\bnext para\b"
    r"|\bcolon\b|\bsemicolon\b|\bperiod\b|\bhyphen\b|\bcapital\b"
    r"|\ball (?:in )?caps?\b|\bnumerical\b|\bbullet(?:ed| point)?\b|\bdash\b"
    r"|\binverted commas?\b|\bquotes?\b|\bbrace(?:s|ts)?\b|\btab\b(?!\w)"
    r"|\bforward slash\b|\bback slash\b|\bdot\b|\bfull point\b",
    re.I)

_QUESTION_RE = re.compile(
    r"\?\s*$|^\s*(?:are|is|do|does|did|have|has|was|were|can|could|would|will|"
    r"shall|should)\b|\b(?:are you|have you|did you|is he|is she|are they)\b"
    r"|കഴിക്കുന്നുണ്ടോ|കഴിക്കണോ", re.I)

_AFFIRMATION_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|correct|right|sure|i am|i do|i have)\b"
    r"|,\s*(?:yes|yeah)\b|അതെ|ഉവ്വ", re.I)

_FAMILY_SUBJECT_RE = re.compile(
    r"\b(?:my|his|her|their)\s+(?:mother|father|mom|dad|brother|sister|"
    r"husband|wife|son|daughter|uncle|aunt|grandmother|grandfather|grandma|"
    r"grandpa|family|parents)\b|\b(?:mother|father|family)\b", re.I)

# Explicit medication-context markers (research §H FP-1): a taking/stop/start
# verb in the clause is context evidence for a short alias ("പാര കഴിച്ചு",
# "she took para").
_MEDICATION_VERB_RE = re.compile(
    r"\b(?:tak(?:e|es|ing|en)|took|us(?:e|es|ing|ed)|prescri(?:be|bed|bing)|"
    r"start(?:ed|ing)?|stop(?:ped|ping)?|continue[sd]?|given?|കഴിച്ചു|"
    r"കഴിക്കുന്നു|കഴിക്കൂ|കഴിക്കണം|നിർത്തി)\b", re.I)


def _brand_strength_dose(clause: str, mention: dict) -> Optional[dict]:
    """Brand-strength speech: "Amlong 5" → 5 mg, "Telma 40" → 40 mg (§I
    MED-014, "Ecosprin 75"). Fires only when the number directly follows the
    brand surface and is not already claimed by a unit dose."""
    if mention.get("match_type") not in ("brand", "short_alias"):
        return None
    pattern = re.escape(mention["surface_text"]) + r"\s*[- ]?(\d+(?:\.\d+)?)"
    m = re.search(pattern, clause, re.I)
    if not m:
        return None
    # A number that continues into an Indian frequency matrix ("Razo-D
    # 1-0-1") is dosing notation, not a brand strength.
    if re.match(r"\s*[-–]\s*\d", clause[m.end():m.end() + 6]):
        return None
    return {"value": float(m.group(1)), "unit": "mg",
            "raw": m.group(0)}


def _clause_around(text: str, pos: int) -> tuple[str, int, int]:
    """(clause, start, end) of the sentence containing ``pos`` — the span the
    class-level fact anchors to (the validator requires the cited source text
    to sit inside the span, so the full sentence is the honest anchor)."""
    start = max(text.rfind(".", 0, pos), text.rfind("!", 0, pos),
                text.rfind("?", 0, pos), text.rfind("\n", 0, pos)) + 1
    ends = [text.find(ch, pos) for ch in ".!?\n"]
    ends = [e for e in ends if e != -1]
    end = min(ends) + 1 if ends else len(text)
    return text[start:end].strip(), start, end

_FORM_TOKEN_RE = re.compile(
    r"\b(?:tablet|tablets|tab|tabs|capsule|capsules|cap|syrup|puff|puffs|"
    r"inhaler|drop|drops|cream|ointment|sachet|injection)\b"
    r"|ടാബ്ലറ്റ്|ഗുളിക|കാപ്സ്യൂൾ|സിറപ്പ്|കുത്തിവെപ്പ്|ഇൻഹേലർ", re.I)

_FREQUENCY_TOKEN_RE = re.compile(
    r"\b(?:od|bd|tds|qid|hs|sos|prn|stat)\b|\b\d(?:/\d)?\s*-\s*\d(?:/\d)?\s*"
    r"-\s*\d(?:/\d)?\b|\b(?:once|twice|three times|four times)\s+(?:a\s+|per\s+)?"
    r"(?:day|daily)\b|\b(?:daily|every day|weekly)\b"
    r"|ദിവസം .*നേരം|നേരം", re.I)

_DOSE_TOKEN_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:mg|g|mcg|ml|units?|iu|µg|ug)\b"
    r"|\b(?:one|two|three|half|quarter|a)\s+(?:tablet|tablets|tab|capsule|"
    r"capsules|cap|puff|puffs|drop|drops)\b", re.I)

_DoseForward = 60      # research §K.4: dose searched 60 chars ahead of the mention
_ContextWindow = 40    # research §K.3: context window for gated aliases


# ---------------------------------------------------------------------------
# 1. Lexicon loader (cached singleton, mirrors normalization.load_lexicon)
# ---------------------------------------------------------------------------
def load_medication_lexicon(path: str | Path | None = None) -> dict:
    """Load and cache medication_lexicon.json (§K.1)."""
    global _lexicon_cache
    if path is not None:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    if _lexicon_cache is None:
        _lexicon_cache = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return _lexicon_cache


def load_semantics_medications() -> dict:
    """The Malayalam semantics ``medications`` block (alias provenance for
    the FP-1 gate: short code-switched forms like പാര/para live there)."""
    path = Path(__file__).parent / "data" / "malayalam_clinical_semantics.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(
            "medications", {})
    except Exception:                     # pragma: no cover
        return {}


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")


# ---------------------------------------------------------------------------
# 2. Medication name scanner
# ---------------------------------------------------------------------------
def _alias_pattern(surface: str) -> re.Pattern:
    """Token-boundary regex for an alias: whitespace/hyphen flexible between
    words, exact boundaries at both ends (no substring matches, no fuzzy)."""
    words = [_nfc(surface).strip().lower().split()]
    parts = [re.escape(w) for chunk in words for w in chunk]
    body = r"[\s\-]*".join(parts)
    return re.compile(rf"(?<![a-z0-9ം-ൿ]){body}(?![a-z0-9ം-ൿ])", re.I)


def _compile_alias_index(lexicon: dict) -> list[dict]:
    """Flat alias index: (pattern, concept_id, match_type, context_required,
    alias_entry, confidence)."""
    index: list[dict] = []
    for med in lexicon.get("medications", []):
        cid = med["id"]
        # The generic English name itself: never context-gated.
        index.append({"pattern": _alias_pattern(med["english"]), "concept_id": cid,
                      "match_type": "generic", "context_required": False,
                      "alias": None, "confidence": med.get("confidence", 0.9)})
        for alias in med.get("english_aliases", []):
            index.append({"pattern": _alias_pattern(alias), "concept_id": cid,
                          "match_type": "english_alias", "context_required": False,
                          "alias": None, "confidence": med.get("confidence", 0.9)})
        for entry in med.get("brand_aliases", []):
            index.append({"pattern": _alias_pattern(entry["surface"]),
                          "concept_id": cid, "match_type": "brand",
                          "context_required": bool(entry.get("context_required")),
                          "alias": entry,
                          "confidence": entry.get("confidence", 0.9)})
        for entry in med.get("phonetic_aliases", []):
            index.append({"pattern": re.compile(re.escape(_nfc(entry["surface"]))) if
                          " " not in entry["surface"] else _alias_pattern(entry["surface"]),
                          "concept_id": cid, "match_type": "phonetic",
                          "context_required": False, "alias": entry,
                          "confidence": entry.get("confidence", 0.9)})
        for entry in med.get("context_required_aliases", []):
            index.append({"pattern": _alias_pattern(entry["surface"]),
                          "concept_id": cid, "match_type": "short_alias",
                          "context_required": True, "alias": entry,
                          "confidence": entry.get("confidence", 0.9)})
    # Longest/most specific surface first so "telma 40" beats "telma".
    index.sort(key=lambda e: -len(e["pattern"].pattern))
    return index


def find_medication_mentions(text: str, lexicon: dict) -> list[dict]:
    """Scan text for medication mentions (§K.2). Exact, token-boundary only.

    Returns dicts {concept_id, surface_text, start, end, match_type,
    brand_surface, confidence, alias, combo, implies_dose}. Markup-command
    spans never yield mentions (FP-2). Context-gated aliases are returned only
    when _context_is_valid passes (FP-1).
    """
    text_n = _nfc(text)
    mentions: list[dict] = []
    taken: list[tuple[int, int]] = []
    markup_spans = [m.span() for m in _MARKUP_RE.finditer(text_n)]

    for entry in _compile_alias_index(lexicon):
        for m in entry["pattern"].finditer(text_n):
            start, end = m.span()
            if any(start < b and a < end for a, b in taken):
                continue
            if any(start < mb and ma < end for ma, mb in markup_spans):
                continue
            if entry["context_required"] and not _context_is_valid(
                    text_n, start, end):
                continue
            alias = entry["alias"] or {}
            mentions.append({
                "concept_id": entry["concept_id"],
                "surface_text": text_n[start:end],
                "start": start, "end": end,
                "match_type": entry["match_type"],
                "brand_surface": (text_n[start:end]
                                  if entry["match_type"] in ("brand", "short_alias")
                                  else None),
                "confidence": entry["confidence"],
                "alias": alias,
                "combo": alias.get("combo"),
                "implies_dose": alias.get("implies_dose"),
            })
            taken.append((start, end))
    return sorted(mentions, key=lambda x: x["start"])


# ---------------------------------------------------------------------------
# 3. Context validator (§K.3)
# ---------------------------------------------------------------------------
def _context_is_valid(text: str, match_start: int, match_end: int) -> bool:
    """True only when a dose, form, frequency, or medication-verb token sits
    within the ±40-character window of the gated alias (research §E.4 / §H
    FP-1: "dose/quantity token, form token, frequency token, or explicit
    medication context marker in the same clause")."""
    window = text[max(0, match_start - _ContextWindow):
                  min(len(text), match_end + _ContextWindow)]
    return bool(_DOSE_TOKEN_RE.search(window) or _FORM_TOKEN_RE.search(window)
                or _FREQUENCY_TOKEN_RE.search(window)
                or _MEDICATION_VERB_RE.search(window))


# ---------------------------------------------------------------------------
# 4–8. Structured field extractors (§K.4–§K.8)
# ---------------------------------------------------------------------------
_DOSE_UNIT_RE = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(mg|milligrams?|g|grams?|mcg|micrograms?|ml|"
    r"units?|iu|µg|ug)\b", re.I)
_UNIT_NORMALIZE = {"milligram": "mg", "milligrams": "mg", "g": "g", "grams": "g",
                   "gram": "g", "mcg": "mcg", "microgram": "mcg",
                   "micrograms": "mcg", "ml": "ml", "unit": "units",
                   "units": "units", "iu": "IU", "µg": "mcg", "ug": "mcg",
                   "mg": "mg"}


def extract_dose(clause: str, med_end: int) -> Optional[dict]:
    """Dose {value, unit, form, raw} within 60 chars after the mention (§K.4).
    Numeric dose wins; a verbal count ("two tablets", "half tablet") fills
    dose_form + dose_value (count). Returns None when nothing was spoken."""
    window = clause[med_end:med_end + _DoseForward]
    m = _DOSE_UNIT_RE.search(window)
    if m:
        value = float(m.group(1))
        unit = _UNIT_NORMALIZE.get(m.group(2).lower(), m.group(2).lower())
        return {"value": value, "unit": unit, "form": None, "raw": m.group(0)}
    lex = load_medication_lexicon()
    counts = lex.get("dose_verbal_counts", {})
    forms = lex.get("dose_forms", {})
    # Verbal counts on either side of the drug ("two tablets of
    # paracetamol", "paracetamol two tablets"): scan EVERY word pair, not
    # just the first, and accept the first count+form combination.
    for span_text in (clause[max(0, med_end - _ContextWindow):med_end], window):
        words = re.findall(r"[A-Za-zം-ൿ]+", span_text)
        for w1, w2 in zip(words, words[1:]):
            word, form_word = w1.lower(), w2.lower()
            form_key = form_word[:-1] if form_word.endswith("s") else form_word
            if word in counts and any(
                    form_key in variants or form_word in variants
                    for variants in forms.values()):
                return {"value": float(counts[word]), "unit": None,
                        "form": form_key, "raw": f"{w1} {w2}"}
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(tablet|tablets|tab|capsule|capsules|cap|"
                  r"puff|puffs|drop|drops|ml)\b", window, re.I)
    if m:
        return {"value": float(m.group(1)), "unit": None,
                "form": m.group(2).lower().rstrip("s"), "raw": m.group(0)}
    return None


def extract_frequency(clause: str, lexicon: dict) -> Optional[dict]:
    """Frequency {raw, code, standard, times_per_day, as_needed, timing} (§K.5).
    Indian matrix first, then Latin abbreviations, then verbal/English/
    Malayalam phrases. Returns None when the transcript establishes nothing."""
    # System A — Indian numeric matrix (3- or 4-part).
    m = re.search(
        r"(?<![0-9])(\d(?:/\d)?|\d)\s*[-–]\s*(\d(?:/\d)?)\s*[-–]\s*(\d(?:/\d)?)"
        r"(?:\s*[-–]\s*(\d(?:/\d)?))?(?![0-9])", clause)
    if m:
        parts = [m.group(1), m.group(2), m.group(3)]
        if m.group(4):
            parts.append(m.group(4))
        code = "-".join(parts)
        entry = lexicon.get("frequency_matrix", {}).get(code)
        if entry:
            out = {"raw": m.group(0), "code": code,
                   "standard": entry["standard"],
                   "times_per_day": entry["times_per_day"],
                   "as_needed": False,
                   "timing": ("bedtime" if code == "0-0-1"
                              else "morning" if code == "1-0-0" else None)}
            out.update({k: entry[k] for k in
                        ("morning", "afternoon", "night") if k in entry})
            out["label"] = entry.get("label")
            return out
        return {"raw": m.group(0), "code": code, "standard": None,
                "times_per_day": None, "as_needed": False,
                "timing": None, "label": None}
    # System B — Latin abbreviations (uppercase, word-boundary).
    for code, entry in lexicon.get("latin_frequencies", {}).items():
        m = re.search(rf"(?<![A-Za-z]){code}(?![A-Za-z])", clause)
        if m:
            return {"raw": m.group(0), "code": code,
                    "standard": entry["standard"],
                    "times_per_day": entry.get("times_per_day"),
                    "as_needed": bool(entry.get("as_needed")),
                    "timing": entry.get("timing"), "label": entry.get("label")}
    # System C — verbal English + Malayalam phrases (longest first).
    for phrase in sorted(lexicon.get("verbal_frequencies", {}),
                         key=len, reverse=True):
        m = re.search(rf"(?<![A-Za-zം-ൿ]){re.escape(phrase)}(?![A-Za-zം-ൿ])",
                      clause, re.I)
        if m:
            entry = lexicon["verbal_frequencies"][phrase]
            return {"raw": m.group(0), "code": entry.get("code", phrase),
                    "standard": entry["standard"],
                    "times_per_day": entry.get("times_per_day"),
                    "as_needed": bool(entry.get("as_needed")),
                    "timing": entry.get("timing"), "label": None}
    return None


def extract_timing(clause: str, lexicon: dict) -> Optional[str]:
    """Meal-relation / time-of-day timing key (§K.6)."""
    for phrase in sorted(lexicon.get("timing_relations", {}),
                         key=len, reverse=True):
        if re.search(rf"(?<![A-Za-zം-ൿ]){re.escape(phrase)}(?![A-Za-zം-ൿ])",
                     clause, re.I):
            return lexicon["timing_relations"][phrase]
    return None


def extract_duration(clause: str) -> Optional[dict]:
    """Duration {value, unit, raw} (§K.7). days | weeks | months only."""
    m = re.search(r"\b(\d+)\s*(day|days|week|weeks|month|months)\b", clause, re.I)
    if m:
        unit = m.group(2).lower().rstrip("s") + "s"
        return {"value": int(m.group(1)), "unit": unit, "raw": m.group(0)}
    lex = load_medication_lexicon()
    words = {"one": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7}
    m = re.search(rf"\b(?:for\s+)?({'|'.join(words)})\s*"
                  r"(day|days|week|weeks|month|months)\b", clause, re.I)
    # "a day"/"a week" inside a frequency phrase ("twice a day", "three
    # times a week") is the rate's article, NOT a duration: the token right
    # before must not be a rate word.
    if m:
        before_words = clause[:m.start()].lower().split()
        # "twice"/"thrice" and "N time(s)" are rate constructions; their
        # trailing "a day/a week" is the rate's article, not a duration.
        if before_words and (before_words[-1] in ("twice", "thrice")
                             or before_words[-1].rstrip("s") in ("time", "x")):
            m = None
    if m:
        unit = m.group(2).lower().rstrip("s") + "s"
        return {"value": words[m.group(1).lower()], "unit": unit,
                "raw": m.group(0)}
    m = re.search(r"\b(\d+)\s*(ദിവസം|ദിവസത്തേക്ക്|ആഴ്ച|ആഴ്ചത്തേക്ക്|മാസം|"
                  r"മാസത്തേക്ക്)", clause)
    if m:
        unit = {"ദിവസം": "days", "ദിവസത്തേക്ക്": "days", "ആഴ്ച": "weeks",
                "ആഴ്ചത്തേക്ക്": "weeks", "മാസം": "months",
                "മാസത്തേക്ക്": "months"}[m.group(2)]
        return {"value": int(m.group(1)), "unit": unit, "raw": m.group(0)}
    lexicon = load_medication_lexicon()
    for phrase, entry in lexicon.get("duration_patterns", {}).get(
            "malayalam_words", {}).items():
        if phrase in clause:
            return {"value": entry["value"], "unit": entry["unit"],
                    "raw": phrase}
    return None


def extract_route(clause: str) -> Optional[str]:
    """Route key (§K.8) — route tokens only; never inferred from class."""
    lex = load_medication_lexicon()
    for route, tokens in lex.get("route_tokens", {}).items():
        for token in tokens:
            if re.search(rf"(?<![A-Za-zം-ൿ]){re.escape(token)}(?![A-Za-zം-ൿ])",
                         clause, re.I):
                return route
    return None


# ---------------------------------------------------------------------------
# 9. Deterministic status classifier (§K.9)
# ---------------------------------------------------------------------------
def classify_medication_status(
    clause: str,
    question: bool,
    speaker: str,
    existing_status: str,
    existing_conditional: bool,
) -> tuple[str, str]:
    """Classify (medication_status, section) from the clause (§K.9).

    Deterministic: every assignment traces to a trigger pattern from §F.1
    (English) or §F.2 (Malayalam). Priority chain §B.2 resolves multiple
    matches; ALLERGY is never overwritten. A question never becomes an
    active status (§6) unless the patient affirms in the same clause.
    """
    triggered: set[str] = set()
    for status, pattern in _ENGLISH_TRIGGERS:
        if re.search(pattern, clause, re.I):
            triggered.add(status)
    for status, pattern in _MALAYALAM_TRIGGERS:
        if re.search(pattern, clause, re.I):
            triggered.add(status)

    # Intention vs. execution vs. institution (§F.1 vs the modality table):
    # "The doctor wants to start X" is an intention (CONSIDERED), and a
    # completed narrative start ("We started X") means the medication IS in
    # the regimen now (CURRENT). Only imperative/institutional starts
    # ("Start Metformin 500 mg") are PRESCRIBED.
    if re.search(r"\bwants? to start\b|\bwanted to start\b", clause, re.I):
        triggered.discard("PRESCRIBED")
        triggered.add("CONSIDERED")
    elif re.search(r"\b(?:we|i|he|she|they) (?:have |just )?started\b",
                   clause, re.I) and not re.search(r"\bstart(?:ing)? you on\b",
                                                   clause, re.I):
        triggered.discard("PRESCRIBED")
        triggered.add("CURRENT")
    is_question = bool(question or _QUESTION_RE.search(clause))
    if _REFUSAL_DISCHARGER_RE.search(clause):
        # "...he doesn't remember the name" discharges the refusal reading.
        triggered -= {"REFUSED"}
    if is_question and not _AFFIRMATION_RE.search(clause):
        # A question captures the medication as QUESTIONED (never CURRENT),
        # unless a strictly higher-priority clinical fact is in the clause
        # (e.g. an allergy statement inside a question).
        non_question = triggered - {"CURRENT", "CONTINUED", "PRESCRIBED"}
        if non_question:
            triggered = non_question
        else:
            triggered = {"QUESTIONED"}

    # MED-011 / §B.2: "He takes some blood pressure tablet ... he doesn't
    # remember the name" — name-uncertainty dominates the taking-verb.
    if _REFUSAL_DISCHARGER_RE.search(clause):
        status, section = "UNCERTAIN", STATUS_TO_SECTION["UNCERTAIN"]
        return status, section
    if triggered:
        status = next(s for s in STATUS_PRIORITY if s in triggered)
        # Historical-frame dominance (§F.1 HISTORICAL): "She used to be on
        # Atenolol ... but that was stopped before surgery" — the "was
        # stopped" describes the HISTORICAL regimen's END, it does not make
        # the medication a recently discontinued current one.
        if status == "DISCONTINUED" and "HISTORICAL" in triggered:
            status = "HISTORICAL"
    elif existing_conditional:
        # FP-8: a conditional medication is never PRESCRIBED.
        status = "CONSIDERED"
    else:
        # Bare prescription dictation ("Razo-D 1-0-1 before food for 2
        # weeks.", §I MED-018): structured dose/frequency/timing with no
        # patient self-reference and no taking-verb is a clinician dictating
        # an order — PRESCRIBED, not a patient-reported CURRENT.
        if (_DOSE_TOKEN_RE.search(clause) or _FREQUENCY_TOKEN_RE.search(clause)) \
                and not re.search(r"\b(?:i|we|you)\b", clause, re.I) \
                and not re.search(r"\btak(?:e|es|ing|en)\b", clause, re.I):
            return "PRESCRIBED", STATUS_TO_SECTION["PRESCRIBED"]
        status = _EXISTING_TO_MED_STATUS.get(existing_status, "UNCERTAIN")

    if existing_conditional and status in ("PRESCRIBED",):
        status = "CONSIDERED"
    return status, STATUS_TO_SECTION[status]


# ---------------------------------------------------------------------------
# 10. Main entry point (§K.10)
# ---------------------------------------------------------------------------
def _clause_facts(mentions: list[dict]) -> list[dict]:
    return mentions


def enrich_medication_facts(
    facts: list[dict],
    text: str,
    lexicon: dict | None = None,
    markup_spans: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """Enrich MEDICATION facts and add lexicon medications the base layer
    could not see (§K.10). Additive: non-MEDICATION facts are never touched,
    existing facts are never removed, and the evidence text is never modified.
    """
    lexicon = lexicon or load_medication_lexicon()
    text_n = _nfc(text)
    lex_markup = [m.span() for m in _MARKUP_RE.finditer(text_n)]
    all_markup = list(markup_spans or []) + lex_markup

    # FP-1 (research §E.4/§H): short aliases are context-gated. The base
    # semantic layer has no gate, so a bare "para"/"pan"/"telma" fact it
    # produced WITHOUT dose/form/frequency/medication-verb context is
    # dropped here — the alias alone is not evidence of a medication.
    gated_surfaces = {
        entry["surface"].strip().lower()
        for med in lexicon.get("medications", [])
        for entry in med.get("context_required_aliases", [])}
    gated_surfaces |= {
        alias.strip().lower()
        for spec in load_semantics_medications().values()
        for alias in spec.get("aliases", [])
        if len(alias.strip()) <= 5}

    def _is_ungated_alias(f: dict) -> bool:
        ev = _nfc((f.get("evidence_text") or "").strip()).lower()
        if ev not in gated_surfaces:
            return False
        clause = f.get("source_text") or ""
        pos = clause.lower().find(ev)
        return not _context_is_valid(clause, max(0, pos),
                                     pos + len(ev) if pos >= 0 else len(clause))

    facts = [f for f in facts if not (
        f.get("fact_type") in ("MEDICATION", "ALLERGY")
        and _is_ungated_alias(f))]

    existing_spans: list[tuple[str, tuple[int, int]]] = []
    named_med_spans: list[tuple[str, tuple[int, int]]] = []
    # Clauses the base extractor already owns for a concept: a lexicon match
    # of the SAME concept inside one of these clauses is the same fact seen
    # twice ("I am allergic to penicillin" — the statement path produced the
    # ALLERGY fact; the lexicon path re-saw "penicillin" in the same clause
    # with a one-character-different span, and both used to render).
    existing_concept_clauses: list[tuple[str, tuple[int, int]]] = []
    for f in facts:
        if f.get("fact_type") == "MEDICATION" and f.get("mention_span"):
            existing_spans.append((f.get("concept") or "",
                                   tuple(f["mention_span"])))
            # Base-layer generic form tokens ("tablet") are not names: they
            # must not suppress a class-level fact.
            if f.get("concept") not in ("tablet", "capsule", "syrup",
                                        "medicine", "injection"):
                named_med_spans.append((f.get("concept") or "",
                                        tuple(f["mention_span"])))
        if (f.get("fact_type") in ("MEDICATION", "ALLERGY")
                and f.get("concept") and f.get("source_span")):
            existing_concept_clauses.append((f["concept"],
                                             tuple(f["source_span"])))

    for f in facts:
        if f.get("fact_type") not in ("MEDICATION", "ALLERGY"):
            continue
        clause = f.get("source_text") or ""
        attrs = f.setdefault("attributes", {})
        # FP-7: a family member's allergy/medication is family history.
        # The fact is still enriched (the drug name stays evidenced) but is
        # never rendered in the patient's medications/allergies list.
        family_subject = bool(_FAMILY_SUBJECT_RE.search(clause))
        if family_subject:
            attrs["family_subject"] = True
            f["section"] = "family_history"
        mentions = find_medication_mentions(clause, lexicon)
        mention = next((m for m in mentions if m["concept_id"] == f.get("concept")),
                       mentions[0] if mentions else None)
        if mention is None:
            continue
        med_end = mention["end"]
        dose = extract_dose(clause, med_end)
        if dose:
            attrs["dose_value"] = dose["value"]
            attrs["dose_unit"] = dose["unit"]
            attrs["dose_form"] = dose["form"] or attrs.get("dose_form")
            attrs["dose_raw"] = dose["raw"]
        elif mention.get("implies_dose"):
            dm = re.match(r"(\d+)\s*(\w+)", mention["implies_dose"])
            if dm:
                attrs["dose_value"] = float(dm.group(1))
                attrs["dose_unit"] = _UNIT_NORMALIZE.get(
                    dm.group(2).lower(), dm.group(2).lower())
                attrs["dose_raw"] = mention["implies_dose"]
                attrs["dose_provenance"] = "brand-implied (spoken brand carries the dose)"
        else:
            # Brand-strength speech: "Amlong 5", "Telma 40", "Rosu 20" —
            # the number after a brand surface is the milligram strength
            # (research §I MED-014). Gated to brand surfaces + dose-unit
            # absence so "Ecosprin 75 daily" style strengths land correctly
            # while prose numbers never become doses.
            brand_dose = _brand_strength_dose(clause, mention)
            if brand_dose:
                attrs["dose_value"] = brand_dose["value"]
                attrs["dose_unit"] = brand_dose["unit"]
                attrs["dose_raw"] = brand_dose["raw"]
                attrs["dose_provenance"] = (
                    "brand-strength number (spoken brand + strength)")
        freq = extract_frequency(clause, lexicon)
        if freq:
            attrs["frequency_raw"] = freq["raw"]
            attrs["frequency_code"] = freq["code"]
            attrs["frequency_standard"] = freq["standard"]
            attrs["frequency_times_per_day"] = freq["times_per_day"]
            attrs["frequency_as_needed"] = freq["as_needed"]
        timing = extract_timing(clause, lexicon)
        if timing:
            attrs["timing_relation"] = timing
        duration = extract_duration(clause)
        if duration:
            attrs["duration_value"] = duration["value"]
            attrs["duration_unit"] = duration["unit"]
            attrs["duration_raw"] = duration["raw"]
        route = extract_route(clause)
        if route:
            attrs["route"] = route
        if mention.get("brand_surface"):
            attrs["brand_surface"] = mention["brand_surface"]
        if mention.get("combo"):
            attrs["combo"] = mention["combo"]
        if mention.get("match_type"):
            attrs["medication_match_type"] = mention["match_type"]

        status, section = classify_medication_status(
            clause,
            question=bool(f.get("evidence") == "QUESTIONED")
            or bool(_QUESTION_RE.search(clause)),
            speaker=f.get("speaker") or "unknown",
            existing_status=f.get("status") or "PRESENT",
            existing_conditional=f.get("certainty") == "CONDITIONAL",
        )
        attrs["medication_status"] = status
        attrs["provenance_notes"] = (
            f"12-state status {status} from clause context; "
            f"match_type={mention['match_type']}")
        # The canonical drug name is the honest English label when the base
        # layer only captured the generic concept ("allergy"), while the
        # clause names the drug ("allergic to Penicillin").
        if f.get("concept") in ("allergy", "medicine", "tablet", "capsule",
                                "syrup", "injection"):
            f["english"] = mention["concept_id"].replace("_", " ")
        if f.get("fact_type") == "ALLERGY" and f.get("concept") == "allergy":
            f["evidence_text"] = mention["surface_text"]
            f["concept"] = mention["concept_id"]
        engine_status, engine_certainty = STATUS_TO_ENGINE[status]
        f["status"] = engine_status
        f["certainty"] = engine_certainty
        if status == "ALLERGY":
            f["fact_type"] = "ALLERGY"
        if status == "CONTINUED":
            attrs["also_in_plan"] = True
        if f.get("section") not in ("allergies",) or status != "ALLERGY":
            f["section"] = section if status != "HISTORICAL" else (
                "pmh" if f.get("section") != "pmh" else f["section"])

        # FP-7 (re-assert after status routing): family facts stay in
        # family_history whatever the medication status resolved to.
        if family_subject and f.get("section") in ("medications", "allergies"):
            f["section"] = "family_history"

    # --- Class-level mentions (Layer 4, FP-4): "BP tablet" becomes an
    # UNCERTAIN unknown_[class] MEDICATION fact — never a named molecule.
    class_spans: list[tuple[int, int]] = []
    for cm in find_class_mentions(text_n, lexicon):
        cstart, cend = cm["start"], cm["end"]
        if any(s < cend and cstart < e for s, e in class_spans):
            continue
        if any(cstart < me and ms < cend for ms, me in all_markup):
            continue
        # A class mention overlapping a NAMED medication fact is redundant
        # ("Glycomet, the sugar tablet") — the named fact wins. Generic form
        # tokens (the base layer's "tablet") never block a class fact.
        if any(cstart < e and s < cend for _, s in named_med_spans):
            continue
        class_spans.append((cstart, cend))
        clause, cl_start, cl_end = _clause_around(text_n, cstart)
        status, section = classify_medication_status(
            clause, question=bool(_QUESTION_RE.search(clause)),
            speaker="unknown", existing_status="PRESENT",
            existing_conditional=bool(re.search(r"\b(?:if|unless)\b", clause, re.I)))
        # §F.1 UNCERTAIN: "BP tablet" (without name) is an uncertain class
        # mention even when the taking-verb is present; temporal/negative
        # class facts (stopped/refused/historical/questioned) remain.
        if status in ("CURRENT", "CONTINUED", "PRESCRIBED", "RECOMMENDED",
                      "CONSIDERED", "NOT_ADHERENT"):
            status = "UNCERTAIN"
        engine_status, engine_certainty = STATUS_TO_ENGINE[status]
        cspan = (cm["start"], cm["end"])
        facts.append({
            "fact_id": "",
            "concept": cm["concept_id"],
            "english": _UNKNOWN_LABELS.get(cm["concept_id"],
                                           cm["concept_id"].replace("_", " ")),
            "fact_type": "MEDICATION",
            "status": engine_status,
            "certainty": engine_certainty,
            "temporal_context": "HISTORICAL" if status == "HISTORICAL" else "CURRENT",
            "source_text": clause.strip(),
            "source_span": [cl_start, cl_end],
            "mention_span": list(cspan),
            "section": section,
            "speaker": "unknown",
            "confidence": cm["confidence"],
            "evidence": "UNCERTAIN" if engine_status == "UNCERTAIN" else "EXPLICIT_PRESENT",
            "attributes": {"medication_status": status,
                           "medication_match_type": "class_level",
                           "provenance_notes": (
                               "class-level mention kept at class precision; "
                               "no molecule invented")},
            "evidence_text": cm["surface_text"],
            "certainty_evidence": None,
            "section_evidence": None,
            "origin": "medication_class",
        })

    # --- New facts: lexicon medications the base extractor could not see ----
    clauses_with_spans: list[tuple[str, tuple[int, int]]] = []
    for m in re.finditer(r"[^.!?;\n]+", text_n):
        clauses_with_spans.append((m.group(0), (m.start(), m.end())))

    for clause, (cstart, cend) in clauses_with_spans:
        if any(ms < cend and cstart < me for ms, me in all_markup):
            continue
        # Family-subject guard (FP-7): "His father was allergic to
        # Penicillin" is family history, never a patient medication fact.
        if _FAMILY_SUBJECT_RE.search(clause):
            continue
        mentions = find_medication_mentions(clause, lexicon)
        for mention in mentions:
            mspan = (cstart + mention["start"], cstart + mention["end"])
            if any(cid == mention["concept_id"]
                   and not (mspan[1] <= s[0] or s[1] <= mspan[0])
                   for cid, s in existing_spans):
                continue
            # Same concept already documented by the base extractor inside
            # this clause (clause spans may differ by a trailing punctuation
            # character): the lexicon path adds nothing new here.
            if any(cid == mention["concept_id"]
                   and not (cend <= s[0] or s[1] <= cstart)
                   for cid, s in existing_concept_clauses):
                continue
            # An ALLERGY clause the base extractor already owns ("I am
            # allergic to penicillin"): the lexicon re-saw the drug name with
            # a different concept id (generic 'allergy' vs 'penicillin') and
            # used to render a second, identical allergy fact.
            if (classify_medication_status(
                    clause, question=bool(_QUESTION_RE.search(clause)),
                    speaker="unknown", existing_status="PRESENT",
                    existing_conditional=False)[0] == "ALLERGY"
                    and any(cid == "allergy"
                            and not (cend <= s[0] or s[1] <= cstart)
                            for cid, s in existing_concept_clauses)):
                continue
            status, section = classify_medication_status(
                clause, question=bool(_QUESTION_RE.search(clause)),
                speaker="unknown", existing_status="PRESENT",
                existing_conditional=bool(re.search(
                    r"\b(?:if|only if|unless)\b", clause, re.I)))
            engine_status, engine_certainty = STATUS_TO_ENGINE[status]
            new_fact = {
                "fact_id": "",
                "concept": mention["concept_id"],
                "english": mention["concept_id"].replace("_", " ")
                if not mention["concept_id"].startswith("unknown_") else
                _UNKNOWN_LABELS.get(mention["concept_id"],
                                    mention["concept_id"].replace("_", " ")),
                "fact_type": "ALLERGY" if status == "ALLERGY" else "MEDICATION",
                "status": engine_status,
                "certainty": engine_certainty,
                "temporal_context": ("HISTORICAL" if status == "HISTORICAL"
                                     else "CURRENT"),
                "source_text": clause.strip(),
                "source_span": [cstart, cend],
                "mention_span": list(mspan),
                "section": section,
                "speaker": "unknown",
                "confidence": round(mention["confidence"], 2),
                "evidence": "EXPLICIT_PRESENT" if engine_status not in (
                    "ABSENT", "UNCERTAIN") else (
                    "EXPLICIT_ABSENT" if engine_status == "ABSENT"
                    else "UNCERTAIN"),
                "attributes": {
                    "medication_status": status,
                    "medication_match_type": mention["match_type"],
                    "provenance_notes": (
                        f"medication_lexicon V1; match_type="
                        f"{mention['match_type']}"),
                },
                "evidence_text": mention["surface_text"],
                "certainty_evidence": None,
                "section_evidence": None,
                "origin": "medication_lexicon",
            }
            # Structured fields on new facts (same extractors, one clause).
            dose = extract_dose(clause, mention["end"])
            if dose:
                new_fact["attributes"].update({
                    "dose_value": dose["value"], "dose_unit": dose["unit"],
                    "dose_form": dose["form"], "dose_raw": dose["raw"]})
            elif mention.get("implies_dose"):
                dm = re.match(r"(\d+)\s*(\w+)", mention["implies_dose"])
                if dm:
                    new_fact["attributes"].update({
                        "dose_value": float(dm.group(1)),
                        "dose_unit": _UNIT_NORMALIZE.get(
                            dm.group(2).lower(), dm.group(2).lower()),
                        "dose_raw": mention["implies_dose"],
                        "dose_provenance": "brand-implied (spoken brand carries the dose)"})
            else:
                # Brand-strength speech ("Amlong 5"), same rule as the
                # enrich path (§I MED-014).
                brand_dose = _brand_strength_dose(clause, mention)
                if brand_dose:
                    new_fact["attributes"].update({
                        "dose_value": brand_dose["value"],
                        "dose_unit": brand_dose["unit"],
                        "dose_raw": brand_dose["raw"],
                        "dose_provenance": (
                            "brand-strength number (spoken brand + strength)")})
            freq = extract_frequency(clause, lexicon)
            if freq:
                new_fact["attributes"].update({
                    "frequency_raw": freq["raw"],
                    "frequency_code": freq["code"],
                    "frequency_standard": freq["standard"],
                    "frequency_times_per_day": freq["times_per_day"],
                    "frequency_as_needed": freq["as_needed"]})
            timing = extract_timing(clause, lexicon)
            if timing:
                new_fact["attributes"]["timing_relation"] = timing
            duration = extract_duration(clause)
            if duration:
                new_fact["attributes"].update({
                    "duration_value": duration["value"],
                    "duration_unit": duration["unit"],
                    "duration_raw": duration["raw"]})
            route = extract_route(clause)
            if route:
                new_fact["attributes"]["route"] = route
            if mention.get("brand_surface"):
                new_fact["attributes"]["brand_surface"] = mention["brand_surface"]
            if mention.get("combo"):
                new_fact["attributes"]["combo"] = mention["combo"]
            facts.append(new_fact)
            existing_spans.append((mention["concept_id"], mspan))
    return facts


# Class-level concept labels (research §I MED-011 / §E Layer 4).
_UNKNOWN_LABELS = {
    "unknown_antihypertensive": "blood pressure tablet (unspecified)",
    "unknown_antidiabetic": "diabetes tablet (unspecified)",
    "unknown_analgesic": "painkiller (unspecified)",
    "unknown_antibiotic": "antibiotic (unspecified)",
    "unknown_other": "unspecified medication",
}


def find_class_mentions(text: str, lexicon: dict | None = None) -> list[dict]:
    """Layer-4 class-level mentions ("BP tablet") as UNCERTAIN unknown_[class]
    candidates — never resolved to a molecule (FP-4)."""
    lexicon = lexicon or load_medication_lexicon()
    out: list[dict] = []
    for spec in lexicon.get("class_mentions", {}).get("patterns", []):
        for m in re.finditer(spec["pattern"], text, re.I):
            if _MARKUP_RE.search(text[max(0, m.start() - 20):m.end() + 20]):
                continue
            out.append({"concept_id": spec["concept_id"],
                        "surface_text": m.group(0),
                        "start": m.start(), "end": m.end(),
                        "confidence": spec["confidence"]})
    return out
