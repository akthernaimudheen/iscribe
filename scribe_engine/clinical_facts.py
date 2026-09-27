# -*- coding: utf-8 -*-
"""Clinical Intelligence V2 — document context, section-aware extraction, and
evidence-grounded typed clinical facts.

Pipeline position (additive; the semantic layer and the v1 fact graph are
untouched and stay the source of what was *mentioned*):

    ASR transcript (immutable)
      -> semantic layer mentions        (normalization/semantic — unchanged)
      -> V2: document context           (what KIND of document is this?)
      -> V2: section segmentation       (WHICH part of the document is this?)
      -> V2: contextual facts           (typed, certainty-scoped, temporal)
      -> V2 typed fact graph            (this module)  -> note_v2 renderer

Why this layer exists
---------------------
Two real English documents (an orthopaedic referral letter and a medico-legal
report) exposed failures that no amount of keyword tuning fixes:

    "New para."                     -> paracetamol (dictation markup matched the
                                       Malayalam medication shorthand "para")
    "steroid injection" (considered) -> medication PRESENT
    "Hi. This is Doctor. Sarah Whitfield." / "It is a b slash 12" -> assessment
    "there's no bony tenderness"     -> Physical Examination: Not documented
    "There is no past medical history of note" -> PMH: Not documented
    "She had severe pain ... for 4 weeks, which then improved" AND
    "she still has some intermittent low backache" -> one fact: pain RESOLVED

Root causes, respectively: dictation/admin text treated as clinical text; no
consideration modality; diagnosis detection by bare keyword match on whole
lines; exam findings not represented at all; PMH modelled as a concept lookup
instead of a section; and one concept collapsed into a single fact regardless
of how many distinct episodes it had.

The contract every V2 fact must satisfy (fail closed):

    * fact_type     — one of the clinical types below, never "generic entity"
    * status        — PRESENT / ABSENT / RESOLVED / CONSIDERED / RECOMMENDED / ...
    * certainty     — CONFIRMED / SUSPECTED / POSSIBLE / ... (modality is real)
    * temporal_context — CURRENT / HISTORICAL / RESOLVED / IMPROVING / ...
    * section       — the document section the evidence sits in
    * source_text + source_span — VERBATIM evidence in the raw transcript
    * speaker, confidence

A fact whose source_span cannot be located in the raw transcript is REJECTED by
validate_clinical_facts(). Nothing is inferred into the graph: no fact without
evidence, no consideration rendered as a finding, no greeting rendered as a
diagnosis.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from . import normalization as N
from .semantic import MEDICATION_CONCEPTS, _merged_lexicon, _post_pass, load_semantics

# ---------------------------------------------------------------------------
# Vocabulary (closed sets — the note renderer and validator both rely on them)
# ---------------------------------------------------------------------------
SECTIONS = (
    "chief_complaint", "hpi", "associated_symptoms", "pertinent_negatives",
    "pmh", "psh", "medications", "allergies", "family_history", "social_history",
    "occupational_history", "ros", "physical_exam", "assessment", "plan",
    "prognosis", "referral_context", "other_documentation",
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
    "occupational_history": "Occupational History",
    "ros": "Review of Systems",
    "physical_exam": "Physical Examination",
    "assessment": "Assessment",
    "plan": "Plan",
    "prognosis": "Prognosis",
    "referral_context": "Referral Context",
    "other_documentation": "Other Documentation",
}

# Sections rendered only when they carry facts (document-specific extras).
OPTIONAL_SECTIONS = ("occupational_history", "prognosis", "referral_context",
                     "other_documentation")

DOCUMENT_TYPES = ("STANDARD_CLINICAL_CONSULTATION", "REFERRAL_LETTER",
                  "MEDICAL_REPORT", "MEDICOLEGAL_REPORT", "FOLLOW_UP", "OTHER")

FACT_TYPES = (
    "SYMPTOM", "DIAGNOSIS", "MEDICATION", "PROCEDURE", "TREATMENT",
    "EXAM_FINDING", "INVESTIGATION", "HISTORY", "SOCIAL_HISTORY",
    "OCCUPATIONAL_HISTORY", "ALLERGY", "PLAN", "RECOMMENDATION", "PROGNOSIS",
)

STATUSES = ("PRESENT", "ABSENT", "RESOLVED", "CONSIDERED", "RECOMMENDED",
            "QUESTIONED", "UNCERTAIN", "HISTORICAL")

CERTAINTIES = ("CONFIRMED", "SUSPECTED", "POSSIBLE", "UNCERTAIN", "RULED_OUT",
               "DENIED", "CONSIDERED", "RECOMMENDED", "CONDITIONAL")

TEMPORAL_CONTEXTS = ("CURRENT", "HISTORICAL", "RESOLVED", "ONGOING",
                     "IMPROVING", "WORSENING", "INTERMITTENT", "RECURRENT",
                     "UNCERTAIN")

# ---------------------------------------------------------------------------
# Concept classification (from the project's own lexicon, not a new word list)
# ---------------------------------------------------------------------------
_CHRONIC_HISTORY_CONCEPTS = {
    "diabetes_history", "hypertension_history", "asthma", "thyroid_disorder",
    "high_cholesterol", "heart_disease", "kidney_disease", "tb_history",
}
_MEDICATION_CONCEPTS = set(MEDICATION_CONCEPTS) | {
    "tablet", "capsule", "syrup", "medicine", "paracetamol", "fluoxetine"}
# Therapy/intervention concepts: a treatment mention is never a symptom.
_TREATMENT_CONCEPTS = {
    "couples_counseling", "counseling", "cognitive_behavioral_therapy",
    "psychotherapy", "cbt",
}
_PROCEDURE_CONCEPTS = {"injection"}
# A concept's semantic entry may override the type map (the "injection" entry
# carries fact_type_hint=procedure: a clinician naming an injection is almost
# always naming an intervention, and the MEDICATION misread is exactly the
# treatment-as-symptom failure class).
_FACT_TYPE_HINTS = {"injection": "PROCEDURE"}
_SOCIAL_CONCEPTS = {"smoking", "alcohol", "recreational_drugs", "diet"}
_OCCUPATIONAL_CONCEPTS = {"occupation"}


def fact_type_from_concept(concept: str) -> str:
    """Public alias used by the medication-modality layer."""
    return concept_fact_type(concept)


def concept_fact_type(concept: str) -> str:
    if concept in _FACT_TYPE_HINTS:
        return _FACT_TYPE_HINTS[concept]
    if concept in _TREATMENT_CONCEPTS:
        return "TREATMENT"
    if concept in _MEDICATION_CONCEPTS:
        return "MEDICATION"
    if concept == "allergy":
        return "ALLERGY"
    if concept in _PROCEDURE_CONCEPTS:
        return "PROCEDURE"
    if concept in _CHRONIC_HISTORY_CONCEPTS:
        return "HISTORY"
    if concept in _SOCIAL_CONCEPTS:
        return "SOCIAL_HISTORY"
    if concept in _OCCUPATIONAL_CONCEPTS:
        return "OCCUPATIONAL_HISTORY"
    return "SYMPTOM"


# ---------------------------------------------------------------------------
# Dictation markup — the layer's first gate
# ---------------------------------------------------------------------------
# These documents were dictated: the speaker says "full stop", "new para",
# "next heading", "bold", "underlined", spells surnames ("C H E A D L E") and
# reads out reference numbers. None of it is clinical content, and one of these
# commands ("New para.") matched a medication alias in the semantic layer.
_MARKUP_PHRASES = (
    r"\bfull stop\b", r"\bcomma\b", r"\bopen brackets?\b", r"\bclose brackets?\b",
    r"\bnew (?:line|para|paragraph|sentence|word|heading|page)\b",
    r"\bnext (?:line|para|paragraph|sentence|word|heading|page)\b",
    r"\b(?:1st|2nd|3rd|first|second|third)\s+heading\b", r"\bheading\b",
    r"\bbold(?:ed|er|ened)?\b", r"\bunderlin(?:e|ed|ing)\b", r"\bemboldened\b",
    r"\bin line\b", r"\bon the next page\b", r"\bthe next page\b",
    r"\bend of (?:his |her |the )?report\b", r"\bplease (?:put|write|note)\b[^.]*",
    r"\bi'?m dictating\b", r"\bdictating\b", r"\bnew word\b",
    r"\bput (?:that|this)\b[^.]*", r"\bin bold\b", r"\bat the top of the report\b",
    r"\bmark it\b[^.]*", r"\byou could date it\b[^.]*", r"\btoday'?s date\b",
    r"\bnew paragraph\b", r"\bwell debt\b", r"\bnon-?occupation\b",
)
_MARKUP_RE = re.compile("|".join(_MARKUP_PHRASES), re.I)
# A clause made only of single letters / digits ("c h e a d l e", "uk", "16")
_SPELLED_RE = re.compile(r"^(?:[a-z]|\d{1,2}|[/\\\-.,;:()'\"]|\s)+$", re.I)
_HEADING_MARKER_RE = re.compile(
    r"\b(?:next (?:heading|page)|(?:1st|2nd|3rd|first|second|third)\s+heading|"
    r"\bheading\b|bold|underline[d]?|emboldened|new line|next line)\b", re.I)

# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------
_DOC_CUES = (
    ("MEDICOLEGAL_REPORT", (
        r"\bmedico-?legal\b", r"\bsolicitors?\b", r"\binstructions? from\b",
        r"\b(?:my|the expert'?s|expert) duty\b", r"\bduty to the court\b",
        r"\brule 3[45]\b", r"\bdeclaration\b", r"\bstatement of truth\b",
        r"\bmedical report\b", r"\bdate of (?:the )?accident\b",
        r"\bexpert evidence\b", r"\bclient'?s? name\b")),
    ("REFERRAL_LETTER", (
        r"\breferral letter\b", r"\bdear (?:dr|mr|mrs|ms|miss|sir|madam)\b",
        r"\byours (?:sincerely|faithfully)\b", r"\bexpert opinion\b",
        r"\bi would be grateful if you could see\b", r"\bconsultant\b",
        r"\bre\s+[A-Z][a-z]+\b", r"\bi would be grateful\b")),
    ("FOLLOW_UP", (
        r"\bfollow[- ]?up\b", r"\breview in \d", r"\bcame back\b",
        r"\bsince last (?:visit|time)\b")),
)

_CONSULTATION_TURN_RE = re.compile(r"^\s*(?:doctor|dr\.?|patient|parent|speaker \d+)\s*:", re.I)


def detect_document_type(raw_text: str, turns: list[dict] | None = None) -> dict:
    """Classify the document, with evidence and honest uncertainty.

    A referral letter and a medico-legal report are NOT a standard outpatient
    consultation; forcing either into the 14-section consultation template is
    what made administrative text land in Assessment. When the evidence is
    mixed, the classification says so instead of guessing.
    """
    text = raw_text or ""
    low = text.lower()
    evidence: list[dict] = []
    scores: dict[str, int] = {t: 0 for t, _ in _DOC_CUES}

    for dtype, patterns in _DOC_CUES:
        for pat in patterns:
            m = re.search(pat, text, re.I)
            if m:
                scores[dtype] += 1
                evidence.append({"document_type": dtype, "cue": m.group(0),
                                 "span": [m.start(), m.end()]})

    # A consultation is evidenced by role-labelled turns with a clinician/patient
    # exchange, not by the mere presence of clinical words.
    consultation_score = 0
    for t in turns or []:
        if _CONSULTATION_TURN_RE.search(t.get("text") or "") or t.get("speaker") in (
                "Doctor", "Patient", "Parent"):
            consultation_score += 1
    # A single run-on consultation still reads like an exchange: questions asked
    # and symptoms answered. Without this, an unlabelled consultation was
    # classified OTHER even when it plainly was one.
    lex = _merged_lexicon()
    clauses = N.split_clauses(text, lex)
    asked = sum(1 for c in clauses
                if N._is_question(N._norm(c), lex)
                # The fused interrogative ("മാറുന്നില്ലേ") is a question in
                # speech even though it is not one of the marker spellings.
                or "ില്ലേ" in c or "ുണ്ടോ" in c)
    symptoms = len({m for c in clauses
                    for _s, _e, m, _x in N._concept_matches(N._norm(c), lex)})
    if asked >= 1:
        consultation_score += 1
    if symptoms >= 3:
        # Three or more distinct clinical concepts, an exchange or not, is a
        # consultation in shape; a referral letter or report is identified by
        # its own cues above and is never outscored by this.
        consultation_score += 2
    if consultation_score >= 2:
        scores["STANDARD_CLINICAL_CONSULTATION"] = consultation_score
        evidence.append({"document_type": "STANDARD_CLINICAL_CONSULTATION",
                         "cue": f"{consultation_score} role-labelled turns",
                         "span": None})

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    # PHASE 6 — a document-shaped SPECIAL type needs more than one weak cue
    # when the material is conversational (a dialogue): "my GP sent me" or a
    # single "consultant" mention must not turn an interview into a referral
    # letter. Strong multi-cue documents (real referral letters, medico-legal
    # reports) score 2+ and are unaffected.
    if best in ("REFERRAL_LETTER", "MEDICOLEGAL_REPORT") and best_score < 2 \
            and consultation_score >= 2:
        best, best_score = "STANDARD_CLINICAL_CONSULTATION", consultation_score
    if best_score == 0:
        if consultation_score >= 2:
            document_type, confidence, uncertain = (
                "STANDARD_CLINICAL_CONSULTATION", 0.5, consultation_score < 3)
        else:
            document_type, confidence, uncertain = "OTHER", 0.3, True
    else:
        document_type = best
        confidence = round(min(0.95, 0.5 + 0.1 * best_score), 2)
        # Competing evidence of a different document shape means the template is
        # not certain — recorded, never silently resolved.
        uncertain = runner_up > 0 and (best_score - runner_up) <= 1
    return {
        "document_type": document_type,
        "confidence": confidence,
        "uncertain": bool(uncertain),
        "evidence": evidence,
        "scores": {k: v for k, v in scores.items() if v},
    }


# ---------------------------------------------------------------------------
# Section segmentation
# ---------------------------------------------------------------------------
_HEADING_MAP = (
    (r"^current symptoms and treatment\b", "hpi"),
    (r"^chronological progression\b", "hpi"),
    (r"^immediate symptoms\b", "hpi"),
    (r"^movement of (?:the )?client\b", "hpi"),
    (r"^history of present illness\b", "hpi"),
    (r"^presenting complaint\b", "hpi"),
    (r"^accident\b", "hpi"),
    (r"^past medical history\b|^previous medical history\b|^medical history\b", "pmh"),
    (r"^past surgical history\b|^surgical history\b|^operations\b", "psh"),
    (r"^allerg", "allergies"),
    (r"^family history\b", "family_history"),
    (r"^social history\b|^personal history\b|^loss consequential\b", "social_history"),
    (r"^occupational history\b|^occupation\b", "occupational_history"),
    (r"^examination\b|^on examination\b|^physical examination\b|^examination findings\b",
     "physical_exam"),
    (r"^summary and conclusions\b|^opinion\b|^impression\b|^assessment\b|"
     r"^diagnosis\b|^clinical assessment\b", "assessment"),
    (r"^prognosis\b", "prognosis"),
    (r"^recommendation|^plan\b|^management\b|^treatment plan\b", "plan"),
    (r"^review of (?:medical )?records\b|^review of notes\b|^declaration\b|"
     r"^instructions?\b|^identification\b|^statement of truth\b", "other_documentation"),
    (r"^referral\b|^referring\b", "referral_context"),
    (r"^medications?\b|^current medications?\b|^treatment\b", "medications"),
)

# Content cues: they decide the section of the clause they appear in (and of the
# facts it yields) even without a heading — that is what "section-aware" means
# here, as opposed to matching isolated keywords anywhere in the transcript.
_EXAM_CUES = re.compile(
    r"\b(?:on examination|on inspection|examination (?:of|shows|reveals|is)|"
    r"no bony tenderness|tender(?:ness)? over|no muscular tenderness|"
    r"no spinal tenderness|range of movement|straight leg raising|"
    r"no (?:other )?marks|inappropriate responses|normal (?:build|gait)|"
    r"no peripheral oedema|lungs? (?:are|is) clear|chest (?:is|sounds) clear|"
    r"heart sounds?|abdomen (?:is|was)|blood pressure|pulse|temperature|spo2)\b", re.I)
_ASSESSMENT_CUES = re.compile(
    r"\b(?:i wonder if|i think (?:this|it) (?:is|might|could)|my impression|"
    r"in my opinion|summary and conclusions?|diagnos(?:is|ed)|sustained an?|"
    r"suffered (?:a|an)|query|possible|suspect(?:ed)?|consistent with|"
    r"compatible with|unresolved)\b", re.I)
_PLAN_CUES = re.compile(
    r"\b(?:would benefit from|benefit from a|consider(?:ed|ing)?|recommend|"
    r"refer(?:ral|red)? to|i would be grateful|further investigations?|"
    r"specialist opinions?|follow[- ]?up in|come back in|start (?:you |him |her )?on|"
    r"i (?:will|would) (?:prescribe|start|order|advise)|"
    r"no further recommendations?)\b", re.I)
_PROGNOSIS_CUES = re.compile(
    r"\b(?:prognosis|recovery can be anticipated|balance of probability|"
    r"long[- ]?term (?:effects|implications)|no long term)\b", re.I)
_PROGNOSIS_STATEMENT_RE = re.compile(
    r"\b(?:recovery (?:can|is|will) (?:be )?(?:anticipated|expected|complete)|"
    r"(?:full|complete) (?:physical )?recovery|"
    r"anticipated (?:within|at a time)|"
    r"no long[- ]?term (?:effects|implications|effect)|"
    r"will have no long[- ]?term|"
    r"prognosis is\b|"
    r"recovery to date has been (?:satisfactory|good))\b", re.I)
_PROGNOSIS_LEAD_RE = re.compile(
    r"^(?:in my opinion(?: and on the balance of probability)?,?|"
    r"on the balance of probability,?|i base my opinion on (?:the facts that)?|"
    r"my opinion on the balance of probability (?:for|that)?)\s*", re.I)


def _clean_prognosis(text: str) -> str:
    cleaned = _PROGNOSIS_LEAD_RE.sub("", text.strip()).strip()
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned
_ADMIN_CUES = re.compile(
    r"\b(?:solicitors? (?:ref|reference)|our reference|instructions? from|"
    r"client'?s? name|date of birth|date of (?:the )?accident|date of (?:the )?report|"
    r"address|telephone|identification|review of notes|"
    r"private and confidential|dear (?:dr|mr|mrs|ms|sir|madam)|"
    r"yours (?:sincerely|faithfully)|review of medical records)\b", re.I)
_PMH_STATEMENT_CUES = re.compile(
    r"\b(?:no|nil|denies|gives no|without)\s+(?:past|previous|significant|"
    r"relevant|any)?\s*(?:medical )?history\b|\bpast medical history\b|"
    r"\bhistory of [a-z][a-z '\-]{0,40}\b", re.I)
_OCCUPATION_CUES = re.compile(
    r"\b(?:off work|time off work|unable to work|returned? to work|back to work|"
    r"occupation|welder|teacher|driver|engineer|nurse|carpenter|fitter|"
    r"works? as a|worked as a)\b", re.I)
_GREETING_CUES = re.compile(
    r"\b(?:hi|hello|good morning|good afternoon|good evening)\b|"
    r"^\s*this is (?:doctor|dr)\b", re.I)
_SIGNOFF_CUES = re.compile(
    r"\byours (?:sincerely|faithfully)\b|\bthank you very much for seeing\b|"
    r"\bthank you,? doctor\b", re.I)


@dataclass
class SectionAssignment:
    section: str
    evidence: str
    heading: str | None = None


# Turn-label prefixes the diarizer prepends ("Doctor: ...", "Patient: ...").
# They are not clinical content and must not keep a markup-only clause alive:
# "Patient: new para." stripped of markup leaves "patient:", whose 8-letter
# token would otherwise defeat the short-token check below and let the
# dictation command reach the medication alias "para".
_SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(?:doctor|patient|parent|nurse|clinician|mother|father)\s*:\s*", re.I)
# Diarizer labels can also appear mid-value ("Address: …, Speaker 0: Liverpool")
# when an upload produced anonymous turns; metadata values must not carry them.
_SPEAKER_LABEL_RE = re.compile(
    r"\b(?:doctor|patient|parent|nurse|clinician|mother|father|speaker \d+)\s*:\s*",
    re.I)


def _is_markup(clause: str) -> bool:
    """True when a clause carries no clinical content (dictation/format only).

    Two forms: nothing but dictation commands ("Next heading. Bold."), or a name
    spelled out letter by letter ("C H E A D L E"). Numbers and dates are NOT
    markup — "01/23/1945" is a metadata value.
    """
    stripped = _SPEAKER_PREFIX_RE.sub("", (clause or "").strip())
    if not stripped:
        return True
    remainder = _MARKUP_RE.sub(" ", stripped)
    remainder = re.sub(r"[\s.,;:()\-/'\"]+", " ", remainder).strip()
    if not remainder:
        return True
    tokens = remainder.split()
    if tokens and all(len(t) <= 2 for t in tokens):
        # Spelled-out names ("C H E A D L E") and bare numbers/dates: characters
        # only, no words — no clinical content to evidence a fact with.
        return True
    return False


def _is_markup_command(clause: str) -> bool:
    """True for a pure dictation command ("Next line", "Bold", "New para")."""
    stripped = clause.strip().rstrip(".!,").strip().lower()
    if not stripped:
        return False
    return bool(re.fullmatch(
        r"(?:next|new)\s+(?:line|para|paragraph|sentence|word|heading|page)"
        r"|(?:1st|2nd|3rd|first|second|third)\s+heading|heading|bold|bolder|"
        r"emboldened|underline[d]?|full stop|comma|in line", stripped))


def _segment(clauses: list[str], document_type: str) -> list[SectionAssignment]:
    """Assign every clause to a document section, in order.

    Order of authority: an explicit heading (dictation or spoken) that starts
    the clause > a content cue in the clause > the section the document is
    currently in. The pointer moves only on headings, so a stray keyword in the
    middle of a paragraph cannot relocate the document.
    """
    out: list[SectionAssignment] = []
    current = "hpi"
    pending_heading = False

    for clause in clauses:
        text = clause.strip()
        stripped = _MARKUP_RE.sub(" ", text)
        stripped = re.sub(r"\s+", " ", stripped).strip(" .,;:")
        markup = _is_markup(text)

        # The heading marker itself carries no content: a heading is expected next.
        if _HEADING_MARKER_RE.search(text):
            if markup:
                pending_heading = True
                out.append(SectionAssignment(current, "markup", None))
                continue
            pending_heading = True

        # Explicit heading at the start of the clause (works with or without a
        # preceding "next heading" marker).
        heading_section = None
        heading_title = None
        for pat, section in _HEADING_MAP:
            m = re.match(pat, stripped, re.I)
            if m:
                heading_section, heading_title = section, m.group(0)
                break
        if heading_section and (pending_heading or _HEADING_MARKER_RE.search(text)
                                or len(stripped) <= 60):
            current = heading_section
            pending_heading = False
            out.append(SectionAssignment(current, f"heading:{heading_title}", heading_title))
            continue
        if pending_heading and not markup and len(stripped) <= 60:
            # A short clause right after a heading marker IS the heading text;
            # unknown titles keep the document where it was (never guessed).
            pending_heading = False

        if markup:
            out.append(SectionAssignment(current, "markup", None))
            continue

        # Content cues override for the clause they appear in (never for the
        # rest of the document — that is the difference between section-aware
        # extraction and matching keywords anywhere in the transcript).
        if _GREETING_CUES.search(text) or _SIGNOFF_CUES.search(text):
            out.append(SectionAssignment("other_documentation", "greeting/sign-off"))
            continue
        if _OCCUPATION_CUES.search(text):
            out.append(SectionAssignment("occupational_history", "occupational cue"))
            continue
        if _EXAM_CUES.search(text) or _exam_rule_match(text):
            out.append(SectionAssignment("physical_exam", "exam cue"))
            continue
        if _PROGNOSIS_CUES.search(text):
            out.append(SectionAssignment("prognosis", "prognosis cue"))
            continue
        if _ADMIN_CUES.search(text):
            out.append(SectionAssignment("other_documentation", "administrative"))
            continue
        if _PMH_STATEMENT_CUES.search(text):
            # A standalone PMH statement ("There is no past medical history of
            # note") names its own section — it must not stay in the HPI bin,
            # where the PMH statement rules below are never reached.
            out.append(SectionAssignment("pmh", "pmh statement cue"))
            continue
        if _PLAN_CUES.search(text) and _ASSESSMENT_CUES.search(text) is None:
            out.append(SectionAssignment("plan", "plan cue"))
            continue
        if _ASSESSMENT_CUES.search(text):
            out.append(SectionAssignment("assessment", "assessment cue"))
            continue
        out.append(SectionAssignment(current, "document position", None))
    return out


def _exam_rule_match(clause: str) -> bool:
    """True when a clause contains an explicit examination finding phrase."""
    return any(re.search(pat, clause, re.I) for pat, _e, _s in _EXAM_RULES)


# ---------------------------------------------------------------------------
# Evidence-grounded mention extraction (per clause, per mention — no merging)
# ---------------------------------------------------------------------------
def _clause_spans(text: str, clauses: list[str]) -> list[tuple[int, int] | None]:
    """Character spans of each clause inside the raw transcript."""
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for clause in clauses:
        idx = text.find(clause, cursor)
        if idx < 0:
            idx = text.find(clause)
        if idx < 0:
            spans.append(None)
            continue
        spans.append((idx, idx + len(clause)))
        cursor = idx + len(clause)
    return spans


def extract_mentions(text: str, lexicon: dict | None = None) -> list[dict]:
    """Per-mention extraction: one record per (clause, concept) occurrence.

    Same scope engine as the semantic layer (reused, not duplicated), but WITHOUT
    the cross-clause merge — the merge is exactly what destroyed the temporal
    separation between an episode that resolved and one that persists.
    """
    lex = lexicon or _merged_lexicon()
    sem = load_semantics()
    clauses = N.split_clauses(text, lex)
    spans = _clause_spans(text, clauses)

    raw_records: list[dict] = []
    entities: list[N.ClinicalEntity] = []
    for idx, clause in enumerate(clauses):
        cn = N._norm(clause)
        question = N._is_question(cn, lex)
        conditional = N._is_conditional(cn, lex)
        temporality, _recurrent = N._temporality(cn, lex)
        duration, dval, dunit = N._extract_duration(cn, lex)
        frequency_raw, frequency_norm = N._frequency(cn, lex)
        # Verbatim English rate expressions ("on a regular basis") are
        # FREQUENCIES, not durations — captured separately so a duration
        # never renders as "for on a regular basis".
        frequency_verbatim = N._frequency_verbatim(cn, lex)
        matches = N._concept_matches(cn, lex)
        # A concept alias matching INSIDE a dictation command ("para" within
        # "new para") is format residue, not a medication the clinician named —
        # even when the clause also carries real content and so passes the
        # markup-only gate. Drop the mention, keep the clause.
        markup_spans = [m.span() for m in _MARKUP_RE.finditer(cn)]
        if markup_spans:
            matches = [(s, e, cid, surf) for s, e, cid, surf in matches
                       if not any(s < me and ms < e for ms, me in markup_spans)]
        resolution_scope = N._resolution_scope(cn, matches, lex)

        for start, end, cid, surface in matches:
            spec = lex["concepts"][cid]
            status, confidence, reason = N._status_for(cn, start, end, lex, question)
            # PHASE 3 — medication/treatment modality. A medication mention's
            # clause states HOW it is taken; the modality outranks the default
            # status. Every pattern is general clause evidence:
            #   "I take Prozac."          -> PRESENT
            #   "I used to take Prozac."  -> HISTORICAL (past episode, not current)
            #   "Are you taking Prozac?"  -> QUESTIONED (asked, never asserted)
            #   "I'm not taking Prozac."  -> ABSENT (denied)
            #   "wants to start Prozac"   -> CONSIDERED (proposed, not taken)
            #   "We started Prozac."      -> PRESENT (initiated)
            #   "We stopped Prozac."      -> RESOLVED-equivalent DISCONTINUED
            modality = None
            if spec.get("medication") or fact_type_from_concept(cid) in (
                    "MEDICATION", "TREATMENT", "PROCEDURE"):
                low_c = cn
                if re.search(r"\b(?:are|am|is)\s+(?:you|he|she|they|i)\s+\w*ing\b"
                             r"|\bdo\s+(?:you|they)\s+(?:take|use|drink)\b"
                             r"|\bdid\s+(?:you|they)\s+(?:take|use)\b"
                             r"|\bhave\s+(?:you|they)\s+been\s+(?:taking|using)\b",
                             low_c) or (question and re.search(
                                 r"\b(?:take|taking|taking|using|on)\b", low_c)):
                    modality = "QUESTIONED"
                elif re.search(r"\bused to\s+(?:take|be on|use)\b|"
                               r"\bwas (?:on|taking)\b", low_c):
                    modality = "HISTORICAL"
                elif re.search(r"\b(?:stopped|stopping|came off|went off|"
                               r"discontinued|no longer taking|off the)\b", low_c):
                    modality = "DISCONTINUED"
                elif re.search(r"\bwants? to (?:start|try|put (?:me|him|her) on)|"
                               r"\bthinking (?:of|about) (?:starting|trying)|"
                               r"\bwould benefit from|\bconsider(?:ing|ed)? (?:starting)?",
                               low_c):
                    modality = "CONSIDERED"
                elif re.search(r"\b(?:started|starting)\b", low_c):
                    modality = "STARTED"
                elif re.search(r"\b(?:take|takes|taking|on|using|prescribed|"
                               r"put me on|puts? me on)\b", low_c) and \
                        status not in (N.ABSENT, N.QUESTION):
                    modality = "TAKEN"
                if modality in ("QUESTIONED",):
                    status, confidence, reason = N.QUESTION, 0.95, "medication_question"
                elif modality == "HISTORICAL":
                    status, confidence, reason = N.PAST_STATUS, 0.85, "used_to_take"
                elif modality == "DISCONTINUED":
                    status, confidence, reason = N.RESOLVED, 0.9, "medication_stopped"
                elif modality == "CONSIDERED":
                    status, confidence, reason = N.CONSIDERED_STATUS, 0.85, "proposed_not_taken"

            negative_idioms = {N._norm(x) for x in spec.get("negative_idioms", [])}
            if status == N.ABSENT and (
                    (spec.get("inherently_negative")
                     and N._suffix_negated(cn, start, end, lex))
                    or surface in negative_idioms):
                status, confidence = N.PRESENT, 0.85
            if ((start, end) in resolution_scope and status in (N.PRESENT, N.ABSENT)
                    and not conditional):
                status, confidence = N.RESOLVED, max(confidence, 0.8)

            entity = N.ClinicalEntity(
                concept=cid, english=spec["english"], surface_text=surface,
                status=status,
                temporality=(N.RECURRENT if _recurrent else (temporality or N.CURRENT)),
                duration=duration, duration_value=dval, duration_unit=dunit,
                severity=N._nearest_attribute(
                    cn, start, {**lex.get("severity_markers", {}),
                                **lex.get("english_severity_markers", {})}),
                frequency=frequency_raw, frequency_normalized=frequency_norm,
                body_location=N._nearest_attribute(
                    cn, start, lex.get("body_locations", {}), window=20),
                pain_quality=N._nearest_attribute(
                    cn, start, lex.get("pain_quality", {}), window=30),
                onset=(duration if (duration and N._ONSET_RE.search(cn)) else None),
                confidence=round(confidence, 2), source_clause=clause.strip(),
                conditional=conditional, uncertainty_reason=reason,
            )
            raw_records.append({
                "clause_index": idx, "clause": clause.strip(), "clause_span": spans[idx],
                "concept": cid, "surface": surface, "status": status,
                "modality": modality,
                "confidence": entity.confidence, "question": question,
                "conditional": conditional, "uncertainty_reason": reason,
                "duration": duration, "severity": entity.severity,
                "body_location": entity.body_location,
                "temporality": entity.temporality,
                "frequency": frequency_norm,
                "frequency_verbatim": frequency_verbatim,
                "mention_span": (
                    [spans[idx][0] + start, spans[idx][0] + end]
                    if spans[idx] else None),
            })
            entities.append(entity)

    # The colloquial-semantics post-pass (low/controlled/family/medication
    # context) still applies: V2 re-uses the layer, it does not replace it.
    entities = _post_pass(entities, sem)
    for record, entity in zip(raw_records, entities):
        record["concept"] = entity.concept
        record["concept_english"] = entity.english
        record["status"] = entity.status
        record["confidence"] = entity.confidence
        record["context"] = entity.context
        record["subject"] = entity.subject
        record["severity"] = entity.severity or record["severity"]
        record["temporality"] = entity.temporality or record["temporality"]
    return raw_records


# ---------------------------------------------------------------------------
# Contextual (non-concept) clinical statements
# ---------------------------------------------------------------------------
_BODY_SITE_RE = re.compile(
    r"\b(left|right|bilateral)?\s*"
    r"(medial (?:foot )?arch|arch|heel|foot|ankle|knee|thigh|hip|leg|toe|toes|"
    r"upper limb|lower limb|arm|forearm|elbow|wrist|hand|finger|fingers|"
    r"neck|shoulders?|low back|lower back|back|lumbosacral spine|cervical spine|"
    r"thoracic spine|spine|head|chest|abdomen)\b", re.I)

_ENGLISH_SEVERITY_RE = re.compile(
    r"\b(mild|moderate|severe|excruciating|slight|somewhat)\b", re.I)
_ONSET_DATE_RE = re.compile(
    r"\bsince\s+(?:the\s+)?((?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|"
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december)|[a-z]+ (?:day|week|month|year)s? ago)\b", re.I)
_TRAJECTORY_MAP = (
    (r"\bfluctuat\w*\b", "INTERMITTENT"), (r"\bintermittent\b", "INTERMITTENT"),
    (r"\bon and off\b", "INTERMITTENT"), (r"\boccasional\w*\b", "INTERMITTENT"),
    (r"\brecurrent\b", "RECURRENT"), (r"\bconstant\w*\b", "ONGOING"),
    (r"\bcontinuous\w*\b", "ONGOING"), (r"\bpersist\w*\b", "ONGOING"),
    (r"\bstill\b", "ONGOING"), (r"\bcontinu(?:e|es|ed|ing)\b", "ONGOING"),
    (r"\bnot yet (?:fully )?(?:resolved|recovered)\b",
                                "IMPROVING"),
    (r"\bimproved\b|\bimproving\b|\bgetting better\b", "IMPROVING"),
    (r"\bworse(?:ning)?\b|\bgetting worse\b|\bdeteriorat\w*\b", "WORSENING"),
)
_AGGRAVATING_RE = re.compile(
    r"\b(?:gets? worse|is worse|worse(?:ns|ened)?|aggravated|increases?)\b"
    r"(?:\s+(?:after|on|when|with|if|during))?", re.I)
_RELIEVING_RE = re.compile(
    r"\b(?:gets? better|is better|relieved|eases?|improves with)\b"
    r"(?:\s+(?:after|on|when|with))?", re.I)
_NO_BENEFIT_RE = re.compile(
    r"\b(?:no|without|little) (?:benefit|improvement|relief|change)\b"
    r"(?:\s+(?:from|with|after))?", re.I)
_BENEFIT_RE = re.compile(
    r"\b(?:proved (?:helpful|effective|beneficial)|been (?:helpful|effective)|"
    r"benefited from|helped)\b", re.I)
# Onset mechanism: what the patient was doing when the symptom started. The
# captured phrase must be a short noun group ("buying tight footwear") that ends
# at a verb or clause boundary — never a whole sentence.
_MECHANISM_RE = re.compile(
    r"\b(?:since|after|following|as a result of|because of|due to)\s+"
    r"((?:(?:buying|wearing|starting|beginning|playing|doing|using|taking|working)\s+)?"
    r"[a-z][a-z'\-]*(?:\s+[a-z][a-z'\-]*){0,3}?)"
    r"(?=\s*(?:,|\.|;|and\b|but\b|which\b|that\b|has\b|have\b|had\b|he\b|she\b|"
    r"they\b|the patient\b|it\b|his\b|her\b|$))", re.I)

# Treatment / procedure nouns that a consideration can introduce.
_TREATMENT_TERMS = (
    "steroid injection", "injection", "physiotherapy", "surgery", "operation",
    "corticosteroid injection", "local anaesthetic injection", "hydrotherapy",
    "acupuncture", "manipulation", "exercise therapy", "analgesia", "analgesics",
    "painkillers", "medication", "tablets", "physio", "insole", "insoles",
    "orthotics", "operation", "arthroscopy", "x-ray", "x ray", "mri", "ct scan",
    "ultrasound", "blood test", "referral", "specialist opinion", "second opinion",
)
_OCCUPATION_TERMS = (
    "welder", "teacher", "driver", "engineer", "nurse", "carpenter", "fitter",
    "off work", "time off work", "unable to work", "returned to work",
    "back to work", "occupation",
)
_INVESTIGATION_TERMS = (
    "x-ray", "x ray", "x rays", "mri", "ct scan", "ultrasound", "blood test",
    "blood tests", "electrical studies", "nerve conduction", "scan",
)

# Diagnosis terms: the project's Malayalam diagnosis list plus the English
# morphology of a diagnosis. A term is admitted only when a diagnosis cue in the
# same clause introduces it — keyword presence alone is never a diagnosis.
_DIAGNOSIS_ML = ("വൈറൽ പനി", "ന്യുമോണിയ", "ബ്രോങ്കൈറ്റിസ്", "ടൈഫോയിഡ്",
                 "ഡെങ്കിവ", "അലർജി പനി", "ഗാസ്ട്രൈറ്റിസ്")
# Diagnosis morphology. Two shapes, because a real diagnosis is either a word
# built from a medical suffix (plantar fasciitis, tendinopathy) or a qualified
# injury noun (whiplash injury, lower back strain). The greedy leading group
# takes the qualifier with the term ("whiplash injury", not "whiplash").
_DIAGNOSIS_SUFFIX_RE = re.compile(
    r"\b((?:[a-z][a-z'\-]*\s+){0,3}"
    r"[a-z][a-z'\-]*(?:itis|algia|opathy|osis|oedema|edema|neuralgia|"
    r"tendinopathy|degeneration))\b", re.I)
_DIAGNOSIS_WORD_RE = re.compile(
    r"\b((?:[a-z][a-z'\-]*\s+){0,2}"
    r"(?:injury|injuries|strain|sprain|fracture|syndrome|spasm|"
    r"nerve entrapment|impingement))\b", re.I)
# Common diagnoses whose name carries no diagnostic suffix ("pneumonia" is not
# "-itis"/"-osis"/an injury word) — without this, "pneumonia was diagnosed"
# produced no fact at all while "meningitis was diagnosed" did. Kept to a
# short list of frequent, unambiguous condition nouns.
_DIAGNOSIS_CONDITION_WORDS = (
    "pneumonia", "asthma", "diabetes", "hypertension", "migraine",
    "tuberculosis", "malaria", "dengue", "typhoid", "gout", "sciatica",
    "eczema", "anaemia", "anemia", "seizure", "stroke",
)
_DIAGNOSIS_CONDITION_RE = re.compile(
    r"\b((?:[a-z][a-z'\-]*\s+){0,2}(?:"
    + "|".join(_DIAGNOSIS_CONDITION_WORDS) + r"))\b", re.I)
_DIAGNOSIS_CUE_CONFIRMED = re.compile(
    r"\b(?:was|were|is|are|has been|have been)\s+diagnosed\b|\bdiagnosis of\b|"
    r"\bdiagnosed as\b|\bsustained\b|\bsuffered\b|\bconsistent with\b|"
    r"\bcompatible with\b|\brevealed\b|\bshows?\b|in my opinion\b|\bassessment\b", re.I)
_DIAGNOSIS_CUE_SUSPECTED = re.compile(
    r"\b(?:wonder if|query|possible|possibly|suspect(?:ed)?|likely|think (?:this|it)|"
    r"might be|could be|may be|rule out|unresolved)\b", re.I)

# PMH statements (explicit ownership / explicit absence), English + Malayalam.
_PMH_NEGATIVE_RE = re.compile(
    r"\b(?:no|nil|denies|no known|gives no|without)\s+"
    r"(?:past|previous|significant|relevant|any)?\s*"
    r"(?:medical history|history of [a-z][a-z '\-]{0,40}|pmh)\b", re.I)
# "He has a history of pneumonia" / "history of asthma": a documented past
# condition — a PMH HISTORY fact, not a current acute finding and not an
# assessment diagnosis. "Family history of X" stays out (it is a relative's
# condition); "no history of X" is handled by the absence rules above.
_PMH_OWNED_HISTORY_RE = re.compile(
    r"\b(?:has|have|had)\s+a\s+history\s+of\s+"
    r"((?:[a-z][a-z'\-]*\s+){0,2}[a-z][a-z'\-]+)", re.I)
_PMH_ABSENCE_OF_NOTE_RE = re.compile(
    r"\bno (?:past|previous|relevant|significant) medical history of note\b", re.I)
_EVENT_ABSENCE_RE = re.compile(
    r"\bno (?:immediate|significant|serious|previous)?\s*"
    r"(immediate symptoms?|symptoms?|psychological symptoms?|"
    r"history of [a-z][a-z '\-]{0,40}|stress (?:disorder|distress disorder)|"
    r"depression|musculoskeletal problems?|complaints?)\b", re.I)
# "No consequence of the accident" attributes the absence to a NAMED EVENT: it
# states the event's late effects (a prognosis fact), not a current symptom.
# Matched BEFORE the generic rule. An unattributed "no consequence" matches
# neither pattern and is omitted entirely — a bare "consequence" carries no
# object, and inventing one would be fabrication.
_EVENT_CAUSAL_ABSENCE_RE = re.compile(
    r"\bno consequences?\s+(?:of|to|for)\s+(?:the\s+|his\s+|her\s+|their\s+)?"
    r"(accident|injury|injuries|incident|fall|trauma|rta|"
    r"road traffic accident)\b", re.I)


def _clause_body_site(clause: str) -> str | None:
    """The anatomical site a finding belongs to, as stated in the clause.

    Handles the side stated separately ("medial arch on his left foot") and two
    coordinated sites ("neck and back pain"), which the old fact graph lost.
    """
    clause = clause or ""
    matches = list(_BODY_SITE_RE.finditer(clause))
    if not matches:
        return None

    def _label(m) -> str:
        laterality, site = (m.group(1) or "").lower(), m.group(2).lower()
        if site.endswith("s") and site not in ("shoulders", "toes", "fingers"):
            site = site.rstrip("s")
        if not laterality:
            others = clause[:m.start()] + " " + clause[m.end():]
            m2 = re.search(r"\b(left|right|bilateral)\b", others, re.I)
            laterality = m2.group(1).lower() if m2 else ""
        return f"{laterality + ' ' if laterality else ''}{site}".strip()

    first = _label(matches[0])
    if len(matches) > 1:
        between = clause[matches[0].end():matches[1].start()].lower()
        if re.search(r"\band\b|\bacross\b|\btogether with\b|,|/", between):
            return f"{first} and {_label(matches[1])}"
    return first


def _english_severity(clause: str) -> str | None:
    m = _ENGLISH_SEVERITY_RE.search(clause)
    return m.group(1).lower() if m else None


def _first_term(clause: str, terms) -> tuple[str, tuple[int, int]] | None:
    """First matching domain term, tolerating a plain plural ("specialist
    opinions" is the same term as "specialist opinion")."""
    low = clause.lower()
    best = None
    for term in terms:
        pattern = rf"(?<![a-z]){re.escape(term)}s?(?![a-z])"
        for m in re.finditer(pattern, low):
            if best is None or m.start() < best[1][0]:
                best = (m.group(0), (m.start(), m.end()))
    return best


def _span_for(clause_span, match_span) -> list[int] | None:
    if not clause_span or not match_span:
        return None
    return [clause_span[0] + match_span[0], clause_span[0] + match_span[1]]


def _statement_facts(
    clauses: list[str],
    clause_spans: list[tuple[int, int] | None],
    sections: list[SectionAssignment],
    document_type: str,
    speaker_for: "callable",
) -> list[dict]:
    """Typed facts for statements that are not a single concept mention.

    Examination findings, diagnoses, treatments and plan considerations are
    *sentences*, not words: they are extracted here with their own cues, status
    and certainty, and every one of them keeps the verbatim clause as evidence.
    """
    facts: list[dict] = []

    def add(**kw) -> dict:
        fact = {
            "fact_id": "",
            "concept": None, "english": None, "fact_type": "SYMPTOM",
            "status": "PRESENT", "certainty": "CONFIRMED",
            "temporal_context": "CURRENT",
            "source_text": "", "source_span": None, "mention_span": None,
            "section": "hpi", "speaker": "clinician", "confidence": 0.6,
            "evidence": "EXPLICIT_PRESENT",
            "attributes": {}, "evidence_text": "", "certainty_evidence": None,
            "section_evidence": None,            "negation_cue": None,
            "conditional": False, "episode_id": None, "element": None,
            "origin": "statement",
        }
        fact.update(kw)
        facts.append(fact)
        return fact

    for idx, clause in enumerate(clauses):
        text = clause.strip()
        if _is_markup(text):
            continue
        section = sections[idx]
        span = clause_spans[idx]
        low = text.lower()
        next_clause = clauses[idx + 1].strip() if idx + 1 < len(clauses) else ""
        # Administrative text (reference numbers, addresses, dates, greetings,
        # declarations) is documented as metadata and can never become a
        # clinical fact — this is what stopped "It is a b slash 12" and
        # "Hi. This is Doctor. Sarah Whitfield." from becoming an assessment.
        if section.section == "other_documentation":
            continue

        # -- examination findings ------------------------------------------
        exam = _exam_findings(text, span, next_clause)
        if exam:
            for finding in exam:
                add(fact_type="EXAM_FINDING",
                    english=finding["english"], fact_type_evidence=finding["evidence"],
                    status=finding["status"],
                    certainty=("CONFIRMED" if finding["status"] in ("PRESENT", "ABSENT")
                               else "UNCERTAIN"),
                    evidence=("EXPLICIT_ABSENT" if finding["status"] == "ABSENT"
                              else "EXPLICIT_PRESENT"),
                    temporal_context="CURRENT",
                    section="physical_exam", speaker="clinician",
                    confidence=0.75, source_text=text, source_span=span,
                    mention_span=_span_for(span, finding["span"]),
                    evidence_text=finding["evidence"],
                    attributes={"location": finding.get("location")},
                    section_evidence=section.evidence)
            continue

        # -- "(has a) history of <condition>" ---------------------------------
        # A documented past condition is PMH HISTORY, never a current finding
        # and never an assessment diagnosis — the assessment block below would
        # otherwise read "history of pneumonia" as a suspected pneumonia.
        # Runs BEFORE the PMH-section branch: that branch only handles explicit
        # absences and `continue`s past everything else.
        m = _PMH_OWNED_HISTORY_RE.search(text)
        if m and not re.search(r"\b(?:no|never|denies|family)\b", low[:m.start()]):
            cond = _clean_diagnosis_term(m.group(1))
            if _usable_diagnosis_term(cond):
                add(fact_type="HISTORY", english=cond, status="PRESENT",
                    certainty="CONFIRMED", evidence="EXPLICIT_PRESENT",
                    section="pmh", speaker="clinician", confidence=0.75,
                    source_text=text, source_span=span,
                    mention_span=_span_for(span, m.span(1)),
                    evidence_text=m.group(0),
                    certainty_evidence="history of condition stated",
                    attributes={"temporal": "past"},
                    section_evidence=section.evidence)
                continue

        # -- past medical / surgical history statements ---------------------
        # PMH is a SECTION, not a concept lookup: "There is no past medical
        # history of note" is a documented PMH statement even though it names
        # no condition, and the old concept-only model reported it as
        # "Not documented".
        if section.section in ("pmh", "psh"):
            negative = (_PMH_ABSENCE_OF_NOTE_RE.search(text)
                        or _PMH_NEGATIVE_RE.search(text))
            if negative:
                specific = _EVENT_ABSENCE_RE.search(text)
                add(fact_type="HISTORY",
                    english=(specific.group(1).lower() if specific
                             else "past medical history"),
                    status="ABSENT", certainty="CONFIRMED",
                    evidence="EXPLICIT_ABSENT", section=section.section,
                    speaker="clinician", confidence=0.85, source_text=text,
                    source_span=span, evidence_text=(specific.group(0) if specific
                                                     else negative.group(0)),
                    mention_span=_span_for(span, (specific or negative).span()),
                    certainty_evidence="explicit absence stated",
                    section_evidence=section.evidence,
                    attributes={"statement": "explicitly none of note"})
            continue

        # -- event absences ("no immediate symptoms", "no history of X") ----
        causal = _EVENT_CAUSAL_ABSENCE_RE.search(text)
        if causal:
            # The absence is attributed to a named event ("no consequence of
            # the accident"): a documentary statement about that event's late
            # effects, rendered in Prognosis — never the current HPI symptom
            # list and never a pertinent negative — with the event reference
            # kept verbatim so nothing is inferred.
            add(fact_type="PROGNOSIS",
                english="no long-term consequence of the accident",
                status="ABSENT", certainty="CONFIRMED",
                evidence="EXPLICIT_ABSENT", section="prognosis",
                speaker="clinician", confidence=0.8,
                source_text=text, source_span=span,
                mention_span=_span_for(span, causal.span()),
                evidence_text=causal.group(0), section_evidence=section.evidence,
                certainty_evidence="explicit absence of consequence stated",
                attributes={"negative_statement": True,
                            "causal_reference": causal.group(1).lower(),
                            "statement": "explicit absence of consequence"})
        else:
            m = _EVENT_ABSENCE_RE.search(text)
            if m:
                phrase = m.group(0).lower()
                is_history = "history" in phrase
                add(fact_type=("HISTORY" if is_history else "SYMPTOM"),
                    english=_clean_absence(phrase),
                    status="ABSENT", certainty="CONFIRMED", evidence="EXPLICIT_ABSENT",
                    section=("pmh" if is_history else "pertinent_negatives"),
                    speaker="clinician", confidence=0.8,
                    source_text=text, source_span=span,
                    mention_span=_span_for(span, m.span()),
                    evidence_text=m.group(0), section_evidence=section.evidence,
                    attributes={"negative_statement": True})
                # not a `continue`: the clause may also carry a diagnosis
                # ("...where he was examined, no X rays were taken, and whiplash
                # injury was diagnosed").

        # -- denied intent (behavioral-health safety) -------------------------
        # "He has no intention of anything" said during a behavioral-health
        # discussion denies self-harm intent: the object of "intention" comes
        # from the surrounding conversation (anaphora), so the rule fires only
        # when a behavioral-health concept sits in the adjacent clauses. The
        # denial is conservative clinical content (an explicitly documented
        # negative), never an invented positive.
        m_intent = re.search(r"\bno intention of (?:self[- ]?harm|harming\s+"
                             r"(?:myself|himself|herself)|anything)\b", text, re.I)
        if m_intent:
            near_bh = " ".join(clauses[max(0, idx - 4):idx + 5])
            if re.search(r"\b(?:suicid|self[- ]?harm|harm myself|cry|crying|"
                         r"sob|depress|cut myself)\b", near_bh, re.I):
                add(fact_type="SYMPTOM",
                    english="self-harm", status="ABSENT", certainty="DENIED",
                    evidence="EXPLICIT_ABSENT", section="pertinent_negatives",
                    speaker="patient", confidence=0.8,
                    source_text=text, source_span=span,
                    mention_span=_span_for(span, m_intent.span()),
                    evidence_text=m_intent.group(0),
                    attributes={"negative_statement": True,
                                "statement": "denied intent (anaphoric)"},
                    certainty_evidence="no intention stated",
                    section_evidence=section.evidence)

        # -- explicit absence of further recommendations ----------------------
        m = re.search(r"\bno further recommendations?\b[^.]{0,120}", text, re.I)
        if m:
            add(fact_type="PLAN",
                english="no further recommendations regarding investigations, "
                        "specialist opinion, or treatment",
                status="ABSENT", certainty="CONFIRMED", evidence="EXPLICIT_ABSENT",
                section="plan", speaker="clinician", confidence=0.85,
                source_text=text, source_span=span,
                mention_span=_span_for(span, m.span()), evidence_text=m.group(0),
                attributes={"explicit_absence": True},
                certainty_evidence="no further recommendations",
                section_evidence=section.evidence)
            continue

        # -- investigations --------------------------------------------------
        inv = _first_term(text, _INVESTIGATION_TERMS)
        if inv:
            negated = bool(re.search(
                rf"\b(?:no|not|without|no more)\b[^.]{{0,30}}{re.escape(inv[0])}\b",
                low)) or bool(re.search(
                rf"\b{re.escape(inv[0])}s?\b[^.]{{0,20}}\b(?:were|was) not taken\b", low))
            add(fact_type="INVESTIGATION", english=inv[0], status=("ABSENT" if negated
                                                                  else "PRESENT"),
                certainty="CONFIRMED",
                evidence=("EXPLICIT_ABSENT" if negated else "EXPLICIT_PRESENT"),
                section=section.section, speaker="clinician", confidence=0.7,
                source_text=text, source_span=span,
                mention_span=_span_for(span, inv[1]), evidence_text=inv[0],
                section_evidence=section.evidence)

        # -- diagnoses -------------------------------------------------------
        # A diagnosis is an assessment statement wherever it was dictated, so
        # its section is ASSESSMENT (not the narrative section the sentence
        # happens to sit in).
        for dx in _diagnosis_facts(text, span):
            add(**{**dx, "source_text": text, "source_span": span,
                   "section": dx.get("section", "assessment"),
                   "speaker": "clinician",
                   "section_evidence": section.evidence})

        # -- occupation / work history -----------------------------------------
        # "He is a welder" / "works as a driver" / "had 2 weeks off work": an
        # evidenced occupational fact. Reuses the _OCCUPATION_CUES vocabulary
        # (the same words that route the SECTION); a clause is only admitted
        # here when it states occupation content, and every fact carries the
        # verbatim clause as source evidence.
        occ = _first_term(text, _OCCUPATION_TERMS)
        # Only a clause that PREDICATES occupation content counts: an identity
        # statement ("He is a welder", "works as a driver") or a work-status
        # statement ("had 2 weeks off work", "unable to work"). A bare label
        # ("Occupation."), an identification value ("driver's license") or an
        # accident-role mention ("the driver without warning") states nothing
        # about the patient's occupation.
        occ_predicate = bool(
            re.search(r"\b(?:is|was|are|were)\s+(?:a|an)?\s*\w+|"
                      r"\bworks? as|\bworked as|"
                      r"\boff work\b|\bunable to work\b|"
                      r"\breturned to work\b|\bback to work\b", low))
        # A road-user role is not a job. Inside a collision/accident narration,
        # "he was a driver" names the patient's ROLE in the event (who was at
        # the wheel) — never their occupation. This is a GENERAL rule about the
        # predicate (driving a vehicle during an event), not a case exception:
        # "He was a driver in a road traffic accident" and "He is a welder" are
        # then distinguished by their evidence, not by a phrase list. Any
        # work-status clause in the same sentence keeps the fact.
        _VEHICLE_EVENT_RE = re.compile(
            r"\b(accident|collision|crash|impact|another (?:vehicle|car|lorry|truck)|"
            r"road traffic|whilst (?:turning|driving|stationary)|when (?:turning|driving)|"
            r"the (?:other|other) driver|traffic lights|junction|car|van|lorry|truck|"
            r"vehicle|headrests?|seat\s?belts?|seatbelt|airbag|passenger|"
            r"travel(?:ling|led|s)?|driving|drov(?:e|en)|stationary)\b", re.I)
        if occ and occ_predicate and section.section != "other_documentation" \
                and not _looks_like_label(text):
            # Road-user role vs occupation. A bare PAST-tense identity
            # statement ("He was a driver") inside a collision/accident
            # narration names the patient's ROLE in the event — who was at the
            # wheel — not their occupation. A PRESENT-tense identity
            # statement ("He is a welder"), any work-status clause ("off
            # work"), any qualified job statement ("a welder for 20 years"),
            # and a vehicle word that is part of the job title ("taxi driver")
            # all remain occupations. The clause itself is the evidence; there
            # are no phrase exceptions.
            near = " ".join(
                c for c in clauses[max(0, idx - 6):idx + 7]
                if len(c.strip()) > 12 and not _is_markup_command(c.strip()))
            _bare = re.sub(r"\b(?:full stop|comma|new paragraph|new para|period)\b",
                           " ", text, flags=re.I).strip().rstrip(".,")
            bare_past_role = bool(re.fullmatch(
                r"(?:he|she|they)\s+was\s+(?:a|an|the)?\s*[a-z][a-z'\-]+",
                _bare, re.I))
            # "when/during/while + the accident" anchors the role AT the event
            # ("a driver when the collision occurred"); "before/after the
            # accident" anchors a past OCCUPATION in time ("a welder before
            # the accident") and is deliberately not matched.
            in_clause_event = bool(re.search(
                r"\b(?:when|while|during|in)\s+(?:the|a|an|his|her|their)?\s*"
                r"(?:road\s+traffic\s+)?(?:accident|collision|crash|incident)\b",
                low))
            if (in_clause_event
                    or (bare_past_role and _VEHICLE_EVENT_RE.search(near))) \
                    and not re.search(r"\boff work\b|\bunable to work\b|"
                                      r"\breturned? to work\b|\bback to work\b", low):
                # A vehicle word that is part of the job title itself ("taxi
                # driver", "lorry driver") sits immediately before the term;
                # that is an occupation, not an event role.
                prefix = text[max(0, occ[1][0] - 12):occ[1][0]].lower()
                if not re.search(r"(?:car|van|lorry|truck|taxi|bus|cab|vehicle)\s*$",
                                 prefix):
                    occ = None
            if occ:
                attrs = {"location": None}
                off_match = re.search(r"\b(?:\d+|one|two|three|four|five|six|\w+)\s+"
                                      r"(?:weeks?|days?|months?)\s+off work\b", low)
                if off_match:
                    attrs["work_absence"] = off_match.group(0)
                m_is = re.search(r"\b(?:is|was|works? as|worked as)\s+(?:a|an)?\s*"
                                 r"([a-z][a-z'\-]+)\b", text, re.I)
                english = (m_is.group(1).lower() if m_is and m_is.group(1).lower()
                           not in ("not", "no", "still") else occ[0])
                if english == "occupation":
                    english = occ[0]
                add(fact_type="OCCUPATIONAL_HISTORY", english=english,
                    status="PRESENT", certainty="CONFIRMED", evidence="EXPLICIT_PRESENT",
                    section="occupational_history", speaker="clinician", confidence=0.7,
                    source_text=text, source_span=span,
                    mention_span=_span_for(span, occ[1]), evidence_text=occ[0],
                    attributes=attrs, section_evidence=section.evidence)

        # -- treatments: received / considered / recommended -----------------
        # PHASE 3 (statement layer): a QUESTION about treatment or medication
        # ("have you been taking any prescribed medications besides…", "are
        # you on anything for the pain?") states nothing about the patient —
        # it is an asked question, never a PRESENT treatment fact. The mention
        # layer records it QUESTIONED instead.
        _clause_is_question = text.strip().endswith("?") or re.search(
            r"\b(?:are|am|is|do|does|did|have|has|can|could|would)\s+"
            r"(?:you|he|she|they|i|we)\b", text, re.I) is not None
        treatment = None if _clause_is_question else _first_term(text, _TREATMENT_TERMS)
        if treatment:
            term, term_span = treatment
            considered = bool(re.search(
                r"\b(?:would benefit from|benefit from a|considers?|considered|"
                r"whether (?:he|she|they) would|option of|trial of|may need|"
                r"might need|could (?:have|try))\b", low))
            recommended = bool(re.search(
                r"\b(?:recommend(?:ed)?|i would (?:be grateful|suggest|advise)|"
                r"advise[d]? (?:a|an|to)|should (?:have|try)|refer(?:ral)? for)\b", low))
            absence = bool(re.search(
                r"\bno further (?:recommendations?|treatment|investigations?)\b", low))
            received = bool(re.search(
                r"\b(?:has|have|had|received|underwent|been having|attended)\b", low))
            no_benefit = bool(_NO_BENEFIT_RE.search(text))
            benefit = bool(_BENEFIT_RE.search(text))
            if considered or recommended or received or absence or no_benefit:
                if considered:
                    status, certainty = "CONSIDERED", "CONSIDERED"
                    evidence = "EXPLICIT_PLAN_CONSIDERED"
                elif recommended:
                    status, certainty = "RECOMMENDED", "RECOMMENDED"
                    evidence = "EXPLICIT_RECOMMENDATION"
                elif absence:
                    status, certainty = "ABSENT", "CONFIRMED"
                    evidence = "EXPLICIT_ABSENT"
                else:
                    status, certainty = "PRESENT", "CONFIRMED"
                    evidence = "EXPLICIT_PRESENT"
                if absence:
                    add(fact_type="PLAN",
                        english="no further recommendations regarding further "
                                "investigations, specialist opinions, or treatment",
                        status="ABSENT", certainty="CONFIRMED",
                        evidence="EXPLICIT_ABSENT", section="plan",
                        speaker="clinician", confidence=0.85, source_text=text,
                        source_span=span, mention_span=_span_for(span, term_span),
                        evidence_text=text, attributes={"explicit_absence": True},
                        certainty_evidence="no further recommendations",
                        section_evidence=section.evidence)
                else:
                    attrs = {}
                    if no_benefit:
                        attrs["benefit"] = "NONE"
                        attrs["treatment_response"] = "no benefit"
                    elif benefit:
                        attrs["benefit"] = "HELPFUL"
                        attrs["treatment_response"] = "helpful"
                    if status in ("CONSIDERED", "RECOMMENDED"):
                        treatment_section = "plan"
                    elif attrs.get("treatment_response") or status == "PRESENT":
                        treatment_section = "hpi"      # history of treatment
                    else:
                        treatment_section = section.section
                    add(fact_type=("PROCEDURE" if "injection" in term or "surgery" in term
                                   else "TREATMENT"),
                        english=term, status=status, certainty=certainty,
                        evidence=evidence, section=treatment_section,
                        speaker="clinician", confidence=0.75, source_text=text,
                        source_span=span, mention_span=_span_for(span, term_span),
                        evidence_text=term, attributes=attrs,
                        certainty_evidence=(
                            "consideration cue" if considered else
                            "recommendation cue" if recommended else None),
                        section_evidence=section.evidence)

        # -- referral / expert-opinion requests ------------------------------
        if re.search(r"\bi would be grateful\b|\bgrateful for your (?:expert )?opinion\b"
                     r"|\bexpert opinion\b|\bthank you very much for seeing\b", low):
            add(fact_type="RECOMMENDATION",
                english="specialist opinion / expert review requested",
                status="RECOMMENDED", certainty="RECOMMENDED",
                evidence="EXPLICIT_RECOMMENDATION", section="plan",
                speaker="clinician", confidence=0.8, source_text=text,
                source_span=span, mention_span=None, evidence_text=text,
                attributes={"explicit_request": True},
                certainty_evidence="grateful for your expert opinion",
                section_evidence=section.evidence)

        # -- prognosis -------------------------------------------------------
        # A prognosis is a statement about the expected course, not any sentence
        # that happens to sit under a "Prognosis" heading; the heading alone
        # must never become a fact.
        m = _PROGNOSIS_STATEMENT_RE.search(text)
        if m and (section.section == "prognosis" or _PROGNOSIS_CUES.search(text)):
            add(fact_type="PROGNOSIS", english=_clean_prognosis(text),
                status="PRESENT", certainty="SUSPECTED",
                evidence="EXPLICIT_PRESENT", section="prognosis",
                speaker="clinician", confidence=0.7, source_text=text,
                source_span=span, mention_span=_span_for(span, m.span()),
                evidence_text=m.group(0), section_evidence=section.evidence)

    return facts


# -- examination findings ----------------------------------------------------
_EXAM_RULES: tuple[tuple[str, str, str], ...] = (
    (r"\bno bony tenderness\b", "no bony tenderness", "ABSENT"),
    (r"\bno muscular tenderness\b", "no muscular tenderness", "ABSENT"),
    (r"\bno spinal tenderness\b", "no spinal tenderness", "ABSENT"),
    (r"\bno tenderness\b", "no tenderness", "ABSENT"),
    (r"\bno peripheral oedema\b|\bno edema\b", "no peripheral oedema", "ABSENT"),
    (r"\bno (?:other )?marks\b[^.]*", "no other marks, scars or bruises", "ABSENT"),
    (r"\bno inappropriate responses\b", "no inappropriate responses", "ABSENT"),
    (r"\bno (?:signs? of )?(?:wheez|rales|crackles|rhonchi)\w*",
     "no adventitious lung sounds", "ABSENT"),
    (r"\bnormal build\b[^.]*|\bnormal gait\b", "normal build and gait", "PRESENT"),
    (r"\b(?:full range of movement|range of movement (?:is )?full)\b[^.]*",
     "full range of movement", "PRESENT"),
    (r"\b(?:spine|lumbosacral spine|cervical spine) is full\b",
     "full spinal movement", "PRESENT"),
    (r"\bstraight leg raising is full\b", "straight leg raising full", "PRESENT"),
    (r"\bpain free\b", "pain free on examination", "PRESENT"),
    (r"\bexamination of (?:her|his|the) [a-z ]+ (?:is|was|are|were) normal\b",
     "normal examination", "PRESENT"),
    (r"\b(?:upper|lower) limbs?[a-z ]{0,20}(?:is|are|was|were) normal\b",
     "limbs normal on examination", "PRESENT"),
    (r"\b(?:spine|abdomen|chest|gait|build)[a-z ]{0,20}(?:is|was|are|were) normal\b",
     "normal examination", "PRESENT"),
    (r"\bsomewhat tender\b|\btender(?:ness)? over\b|\btender on palpation\b",
     "tenderness", "PRESENT"),
    (r"\blungs? (?:are|is) clear\b", "lungs clear to auscultation", "PRESENT"),
    (r"\bchest (?:is|sounds) clear\b", "chest clear to auscultation", "PRESENT"),
    (r"\bheart sounds? (?:are|is) normal\b", "normal heart sounds", "PRESENT"),
    (r"\babdomen (?:is|was) (?:soft|normal)\b", "soft abdomen", "PRESENT"),
    (r"\bbp\b (?:is )?\d{2,3}\s*/\s*\d{2,3}\b", "blood pressure recorded", "PRESENT"),
    (r"\bpulse\b (?:is )?\d{2,3}\b", "pulse recorded", "PRESENT"),
    (r"\btemperature\b (?:is )?\d{2}(?:\.\d)?\b", "temperature recorded", "PRESENT"),
    (r"\bspo2\b (?:is )?\d{2,3}\b", "oxygen saturation recorded", "PRESENT"),
)


def _exam_findings(clause: str, span, next_clause: str = "") -> list[dict]:
    """Explicit clinician examination findings in one clause.

    The finding phrases themselves are specific ("no bony tenderness", "full
    range of movement", "pain free"); the location is taken from the same
    clause, or from the following clause when the finding was dictated before
    its site ("somewhat tender over the midpoint of his middle... Medial arch
    on his left foot").
    """
    out: list[dict] = []
    for pat, english, status in _EXAM_RULES:
        m = re.search(pat, clause, re.I)
        if not m:
            continue
        location = None
        if status == "PRESENT":
            location = _clause_body_site(clause) or _clause_body_site(next_clause or "")
        out.append({
            "english": english, "status": status, "evidence": m.group(0)[:120],
            "span": m.span(), "location": location,
        })
    return out


# -- diagnoses ---------------------------------------------------------------
def _diagnosis_facts(clause: str, span) -> list[dict]:
    """Diagnosis facts — confirmed only with a confirmation cue, and a real
    diagnosis term. A greeting or an administrative reference cannot produce a
    diagnosis, because it contains no diagnosis term."""
    out: list[dict] = []
    confirmed_cue = _DIAGNOSIS_CUE_CONFIRMED.search(clause)
    suspected_cue = _DIAGNOSIS_CUE_SUSPECTED.search(clause)

    for term, term_span, kind in _diagnosis_terms(clause):
        certainty = None
        if suspected_cue:
            certainty = "SUSPECTED"
        elif confirmed_cue:
            certainty = "CONFIRMED"
        if certainty is None:
            continue
        certainty_evidence = (suspected_cue or confirmed_cue).group(0)
        out.append({
            "concept": None,
            "english": term,
            "fact_type": "DIAGNOSIS",
            "status": "PRESENT",
            "certainty": certainty,
            "temporal_context": "CURRENT",
            "section": "assessment",
            "evidence": "EXPLICIT_PRESENT",
            "confidence": 0.85 if certainty == "CONFIRMED" else 0.7,
            "mention_span": _span_for(span, term_span),
            "evidence_text": term,
            "certainty_evidence": certainty_evidence.strip(),
            "attributes": {"diagnosis_basis": kind},
        })
    return out


_GENERIC_INJURY_WORDS = {"injury", "injuries", "accident", "pain", "pain free",
                        "strain", "sprain", "whiplash", "spasm"}

# Leading filler that a dictation puts in front of the diagnosis itself.
_TERM_LEAD_RE = re.compile(
    r"^(?:a|an|the|her|his|their|my|no|other|and|with|of|is|are|was|were|be|"
    r"been|has|have|had|sustained|suffered|developed|reported|possible|possibly|"
    r"likely|unresolved|type|style|slight|severe|chronic|acute)\s+", re.I)


def _clean_diagnosis_term(term: str) -> str:
    cleaned = term.strip()
    while True:
        stripped = _TERM_LEAD_RE.sub("", cleaned, count=1).strip()
        if stripped == cleaned or not stripped:
            break
        cleaned = stripped
    return cleaned.strip()


# Words that share a diagnosis suffix but are not diagnoses.
_NON_DIAGNOSIS_WORDS = {"prognosis", "diagnosis", "osmosis", "apotheosis",
                        "diagnoses", "prognoses"}


def _usable_diagnosis_term(term: str) -> bool:
    """A diagnosis needs a real term: "whiplash injury" yes, "her injuries" no."""
    cleaned = _clean_diagnosis_term(term).lower()
    words = cleaned.split()
    if not words:
        return False
    if words[0] in ("no", "not", "nil"):
        return False
    if cleaned in _GENERIC_INJURY_WORDS or cleaned in _NON_DIAGNOSIS_WORDS:
        return False
    if words[-1] in ("injury", "injuries") and len(words) < 2:
        return False
    # A mechanism-of-injury description ("forced flexion extension movement",
    # ASR-garbled "double flexion and stroke extension") is what HAPPENED, not
    # a diagnosis: it only ever produces diagnoses via ASR noise, never via a
    # clinician naming a condition.
    if "movement" in words or "flexion" in words or "extension" in words:
        return False
    return True


def _diagnosis_terms(clause: str) -> list[tuple[str, tuple[int, int], str]]:
    out: list[tuple[str, tuple[int, int], str]] = []
    for term in _DIAGNOSIS_ML:
        idx = clause.find(term)
        if idx >= 0:
            out.append((term, (idx, idx + len(term)), "lexicon"))
    for pattern in (_DIAGNOSIS_SUFFIX_RE, _DIAGNOSIS_WORD_RE,
                    _DIAGNOSIS_CONDITION_RE):
        for m in pattern.finditer(clause):
            raw = m.group(1).strip().strip(".,;:")
            # A certainty cue captured with the term ("rule out pneumonia") is
            # the CUE, not part of the condition's name; the cue itself decides
            # certainty in _diagnosis_facts.
            raw = re.sub(r"^(?:rule[ds]?\s+out|r/o\.?|query)\s+", "", raw,
                         flags=re.I)
            term = _clean_diagnosis_term(raw)
            if not _usable_diagnosis_term(term):
                continue
            span = m.span(1)
            offset = clause.lower().find(term.lower(), span[0])
            if 0 <= offset and offset + len(term) <= span[1] + 1:
                span = (offset, offset + len(term))
            kind = "morphology" if pattern is not _DIAGNOSIS_CONDITION_RE \
                else "condition-word"
            out.append((term, span, kind))
    # Deduplicate overlapping terms, longest first.
    out.sort(key=lambda t: (t[1][0], -(t[1][1] - t[1][0])))
    deduped: list[tuple[str, tuple[int, int], str]] = []
    for term, tsp, kind in out:
        if any(not (tsp[1] <= a or tsp[0] >= b) for _t, (a, b), _k in deduped):
            continue
        deduped.append((term, tsp, kind))
    return deduped


# ---------------------------------------------------------------------------
# Document metadata (administrative content stays OUT of clinical sections)
# ---------------------------------------------------------------------------
# (key, label pattern, tail capture) — the tail is the metadata value itself,
# and when the dictation put the value in the NEXT clause the extractor follows
# it ("Date of birth," then "01/23/1945").
_METADATA_RULES = (
    ("solicitors_reference",
     r"\b(?:solicitors?|our|their) (?:ref|reference)\b[\s,]*([^.]*)"),
    ("instructions_from", r"\binstructions? from\b[\s,]*([^.]*)"),
    ("client_name", r"\bclient'?s? names?\b[\s,]*([^.]*)"),
    ("patient_name", r"\bre[:\s]+(?:mr|mrs|ms|miss|dr)\.?\s+"
     r"((?-i:[A-Z][a-z]+(?: [A-Z][a-z]+)?))\b"),
    ("patient_name", r"\bpatient(?:'s)? name\b[:\s]+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?)"),
    ("date_of_birth", r"\bdate of birth\b[\s,]*([^.]*)"),
    ("date_of_accident", r"\bdate of (?:the )?accident\b[\s,]*([^.]*)"),
    ("date_of_report", r"\bdate of (?:the )?report\b[\s,]*([^.]*)"),
    ("address", r"\baddress\b[\s,]*([^.]*)"),
    ("telephone", r"\b(?:telephone|phone)(?: number)?\b[\s,]*([^.]*)"),
    ("identification", r"\bidentification\b[\s,]*([^.]*)"),
    ("review_of_notes", r"\breview of notes\b[\s,]*([^.]*)"),
    ("occupation", r"\boccupation\b[\s,]*([^.]*)"),
    ("referral_addressee",
     r"\bdear\s+((?:dr|mr|mrs|ms|miss|sir|madam)\.?\s*[A-Z][a-z]*)"),
    ("referral_recipient_role",
     r"\bconsultant\s+([a-z][a-z ]{2,40}?)(?=\s*,|\s*in\b|$)"),
    ("accident_time",
     r"\b(?:accident|impact)\b[^.]{0,20}?\b(\d{1,2}[:.]\d{2}\s*(?:am|pm)?)\b"),
)
_CLINICIAN_INTRO_RE = re.compile(
    r"\bthis is (?:doctor|dr)\.?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I)

# PHASE 5 — conversational-speech guard for metadata. A conversational clause
# ("You're John, aren't you?", "Are you 38?", "I don't know, whatever") must
# never populate a metadata field: questions, tag confirmations, guesses and
# conversational filler are not field/value evidence. Dictated documents
# ("Patient name: John Smith") are unaffected — no question marks, no filler,
# short values.
_TAG_QUESTION_RE = re.compile(
    r"\b(?:aren'?t|isn'?t|wasn'?t|weren'?t|don'?t|doesn'?t|didn'?t|haven'?t|"
    r"hasn'?t|hadn'?t|won'?t|wouldn'?t|can'?t|couldn'?t|aren)\s+you\b"
    r"|\b(?:is|are|was|were|do|does|did|have|has|had)\s+(?:he|she|it|they)\b"
    r"[^.]*\?", re.I)
_CONVERSATIONAL_FILLER_RE = re.compile(
    r"\b(?:you know|i mean|i (?:think|guess|suppose|wanna|don'?t know)|"
    r"whatever|something like that|or something)\b", re.I)
_METADATA_MAX_VALUE_LEN = 90


def _metadata_value_rejected(clause: str, value: str, key: str = "") -> str | None:
    """Reason a metadata match is conversational junk (None = accept)."""
    if "?" in (value or "") or "?" in (clause or ""):
        return "interrogative"
    if _TAG_QUESTION_RE.search(clause):
        return "tag question"
    if len((value or "").strip()) > _METADATA_MAX_VALUE_LEN:
        return "value too long for a dictated field"
    if _CONVERSATIONAL_FILLER_RE.search(value or ""):
        return "conversational filler"
    # Field-shape checks: a telephone number contains digits; a person name is
    # title-case words. Conversational fragments ("everything like that",
    # "still filling") fail these shapes and are rejected.
    v = (value or "").strip()
    if key == "telephone" and not any(ch.isdigit() for ch in v):
        return "telephone value without digits"
    if key in ("patient_name", "client_name") and v and not re.fullmatch(
            r"[A-Z][A-Za-z]*(?:[ .,'-]+[A-Z][A-Za-z.]*)*", v):
        return "implausible person name"
    return None
# A name spelled out for the dictation system: at least four single letters in a
# row ("C H E A D L E"), which keeps "a c c" reference spellings out.
# A spelled name starts at a word boundary (rather than mid-word), so
# "That's C H E A D L E" does not capture the possessive "s".
# A spelled name is at least four letters ("C H E A D L E"); three letters would
# match dictated reference initials ("a c c").
_SPELLED_LETTERS_RE = re.compile(r"(?:(?<=\s)|(?<=^))((?:[a-z]\s+){3,}[a-z])(?![a-z])",
                                re.I)

def _clean_absence(phrase: str) -> str:
    """Render an explicit absence statement as clinical text ("no immediate
    symptoms" -> "immediate symptoms")."""
    cleaned = re.sub(r"^\s*(?:no|nil|denies)\s+", "", phrase.strip(), flags=re.I)
    return cleaned.strip()
_METADATA_TITLE_RE = re.compile(r"^(?:dr|mr|mrs|ms|miss|sir|madam)\.?$", re.I)


def _normalize_escaped(text: str) -> str:
    text = _MARKUP_RE.sub(" ", text or "")
    out = re.sub(r"^(?:it is|it's|that is|that's)\s+", "", text.strip(), flags=re.I)
    out = re.sub(r"\bslash\b", "/", out, flags=re.I)
    out = re.sub(r"\bcomma\b", ",", out, flags=re.I)
    out = re.sub(r"\bfull stop\b", ".", out, flags=re.I)
    out = re.sub(r"\s+", " ", out).strip(" .,;:")
    return out


def _looks_like_label(clause: str) -> bool:
    return any(re.search(pat, clause, re.I) for _k, pat in _METADATA_RULES)


# Metadata keys whose dictated value is commonly split across clauses by the
# comma/period the dictation puts inside it (an address, a full name).
_JOINABLE_METADATA = {
    "address", "telephone", "client_name", "patient_name", "date_of_birth",
    "date_of_accident", "date_of_report", "instructions_from", "identification",
    "review_of_notes", "referral_addressee",
}


def _join_value(clauses: list[str], idx: int, value: str, span,
                clause_spans) -> tuple[str, object]:
    """Continue a metadata value across the clauses a comma split it into.

    A dictated address arrives as "Address, 1, The House, 2, The Road,
    Liverpool L 13 B A"; stopping at the first clause would document the
    address as "1". Continuation stops at the next label or a long clause.
    """
    limit = idx + 4
    cursor = idx + 1
    while cursor < len(clauses) and cursor <= limit:
        nxt = clauses[cursor].strip().strip(" .,;:")
        tokens = nxt.split()
        spelled_run = len(tokens) >= 2 and all(len(t) <= 2 for t in tokens)
        if (not re.search(r"[a-z0-9]", nxt, re.I) or _is_markup_command(nxt)
                or _looks_like_label(nxt) or len(nxt) > 40 or spelled_run):
            break
        value = f"{value}, {nxt}" if value else nxt
        span = [span[0] if span else None,
                (clause_spans[cursor] or [None, None])[1]]
        cursor += 1
    return value, span


def extract_metadata(text: str, clause_spans: list[tuple[int, int] | None],
                     clauses: list[str]) -> dict:
    """Administrative document metadata — evidence-bearing, never clinical."""
    meta: dict[str, dict] = {}

    def put(key: str, value: str, span):
        value = _normalize_escaped(value).strip()
        # Diarizer labels ("Speaker 0:", "Patient:") can be stitched into a
        # joined value when an upload produced anonymous turns; they are turn
        # structure, not document content.
        value = _SPEAKER_LABEL_RE.sub("", value)
        value = re.sub(r"\s{2,}", " ", value).strip(" .,;:")
        if not value or key in meta:
            return
        meta[key] = {"value": value, "source_span": list(span) if span else None}

    for idx, clause in enumerate(clauses):
        span = clause_spans[idx]
        nxt = clauses[idx + 1].strip() if idx + 1 < len(clauses) else ""
        if _is_markup_command(nxt):
            nxt = ""
        for key, pat in _METADATA_RULES:
            m = re.search(pat, clause, re.I)
            if not m:
                continue
            value = m.group(1) if m.groups() else m.group(0)
            value_span = _span_for(span, m.span(1 if m.groups() else 0))
            # "Occupation, 1st heading, accident, emboldened…": the dictated
            # tail is formatting commands, not a value. Only stitch the next
            # clause when the tail carries real content — a pure markup/heading
            # tail means the document names no value here (rendered as
            # "Not documented", never inferred from turn labels).
            tail_content = re.sub(r"[\s.,;:]", "",
                                  _MARKUP_RE.sub(" ", value))
            tail_is_heading = bool(re.fullmatch(
                r"[\s,]*(?:1st|2nd|3rd|first|second|third|next)[\s,]*", value, re.I))
            # The dictation often leaves the value in the next clause: either the
            # tail is empty ("Date of birth," / "01/23/1945") or it stops at a
            # title ("Dear Mr." / "Spring").
            tail_is_markup_only = not _MARKUP_RE.sub(" ", value).strip(" .,;:")
            if nxt and (not _normalize_escaped(value) or tail_is_markup_only
                        or tail_is_heading
                        or _METADATA_TITLE_RE.match(value.strip())):
                value = f"{value.strip()} {nxt}".strip()
                value_span = [value_span[0] if value_span else None,
                              (clause_spans[idx + 1] or [None, None])[1]]
                if value_span[0] is None:
                    value_span = clause_spans[idx + 1]
                if key in _JOINABLE_METADATA:
                    value, value_span = _join_value(clauses, idx + 1, value,
                                                    value_span, clause_spans)
            elif key in _JOINABLE_METADATA:
                value, value_span = _join_value(clauses, idx, value,
                                                value_span, clause_spans)
            # Conversational guard: questions, tag confirmations and filler
            # are never metadata. The clause is left unmatched rather than
            # storing a fabricated field value.
            if _metadata_value_rejected(clause, value, key):
                continue
            put(key, value, value_span)
        m = _CLINICIAN_INTRO_RE.search(clause)
        if m:
            put("clinician", m.group(1), _span_for(span, m.span(1)))
        m = _SPELLED_LETTERS_RE.search(clause)
        if m:
            spelled = re.sub(r"\s+", "", m.group(1)).upper()
            entry = meta.setdefault("spelled_names",
                                    {"value": [], "source_span": None})
            if spelled not in entry["value"]:
                entry["value"].append(spelled)
            entry["source_span"] = _span_for(span, m.span(1))
    return meta


# ---------------------------------------------------------------------------
# Episodes: temporal reconciliation WITHOUT collapsing distinct episodes
# ---------------------------------------------------------------------------
def _episode_facts(mentions: list[dict], sections: list[SectionAssignment],
                   clauses: list[str], document_type: str,
                   speaker_of: "callable") -> list[dict]:
    """Turn per-mention records into episode facts.

    Only mentions of the *same concept in the same body site* are reconciled,
    and only in the one direction the semantics justify (a past episode that is
    absent now = resolved). Everything else stays a separate episode, so
    "neck pain for four weeks, which improved" and "still has intermittent arm
    pain" are two facts, not one status.
    """
    facts: list[dict] = []

    for m in mentions:
        clause = m["clause"]
        section = sections[m["clause_index"]]
        if section.section == "other_documentation":
            continue                      # administrative text is not clinical
        if _is_markup(clause):
            continue
        # English idioms whose wording contains a clinical word but whose
        # meaning is not the finding ("I'm not sweating it" = not worried).
        if re.search(r"\bsweating it\b", clause, re.I) and \
                m["concept"] == "sweating":
            continue
        # Impersonal commentary: "Maybe sometimes it's just exhausting" — the
        # subject is the SITUATION, not the patient; a hedge + impersonal "it
        # is just" + state adjective is commentary, never a finding.
        if re.match(r"^\s*(?:maybe|perhaps)\s+(?:sometimes\s+)?it'?s\s+just\b",
                    clause, re.I):
            continue
        if m["concept"] is None:
            continue
        if not m.get("clause_span"):
            continue
        if m.get("mention_span") is None:
            continue

        status = m["status"]
        certainty = "CONFIRMED"
        evidence = "EXPLICIT_PRESENT"
        temporal = "CURRENT"
        negation_cue = None

        if status == N.QUESTION:
            status, certainty, evidence = "QUESTIONED", "UNCERTAIN", "EXPLICIT_QUESTION"
        elif status == N.ABSENT:
            certainty, evidence = "DENIED", "EXPLICIT_ABSENT"
            negation_cue = _negation_cue(clause)
        elif status == N.RESOLVED:
            certainty, evidence = "CONFIRMED", "EXPLICIT_RESOLVED"
            temporal = "RESOLVED"
        elif status == N.PAST_STATUS:
            # "I used to take Prozac": a clearly historical treatment. Kept
            # separate from RESOLVED (an episode that ended) and from PRESENT.
            status, certainty, evidence = "HISTORICAL", "CONFIRMED", "EXPLICIT_PRESENT"
            temporal = "HISTORICAL"
        elif status == N.CONSIDERED_STATUS:
            status, certainty, evidence = "CONSIDERED", "CONSIDERED", "EXPLICIT_PRESENT"
        elif status == N.UNKNOWN or status == N.POSSIBLE:
            status, certainty = "UNCERTAIN", "POSSIBLE"
        elif m.get("conditional"):
            status, certainty, evidence = "UNCERTAIN", "CONDITIONAL", "CONDITIONAL"

        if status in ("PRESENT", "RESOLVED"):
            temporal = _temporal_context(clause, m)
        elif m.get("temporality") in (N.PAST, N.RECURRENT) and status == "PRESENT":
            temporal = "HISTORICAL"
        # PAST-FRAMING anchors (checked after the default chain so they
        # override the CURRENT default): "when I was younger", "back then",
        # "used to" — the finding belongs to the patient's history, not the
        # present state, even though the clause reads in present/past tense.
        # The clause splitter cuts mid-sentence ("...low sometimes / when I
        # was younger"), so the ADJACENT clauses count as the frame too.
        if status == "PRESENT" and re.search(
                r"\b(?:when|back) (?:I|we|he|she|they) (?:was|were) younger\b"
                r"|\bback then\b|\bused to (?:happen|get|feel|be|come|go)\b"
                r"|\bin (?:my|his|her|their) (?:youth|twenties|thirties|"
                r"childhood)\b|\ba long time ago\b|\blong ago\b"
                r"|\bfor ages(?: now)?\b",
                " ".join(clauses[max(0, m["clause_index"] - 1):
                                   m["clause_index"] + 2]), re.I):
            temporal = "HISTORICAL"
        elif status in ("HISTORICAL", "CONSIDERED", "QUESTIONED"):
            temporal = "HISTORICAL" if status == "HISTORICAL" else temporal

        english = m.get("concept_english") or m["concept"].replace("_", " ")
        # Conversational duration continuation: the clause splitter cuts
        # mid-sentence ("...on a regular basis / like, multiple times a week
        # and suggested..."), so a clause that does not END the sentence
        # inherits the verbatim temporal expressions of the clause that
        # immediately follows it — but only substantial clause-continuations
        # (more than 3 words): a bare fragment ("beardie", "yeah") is a
        # split artifact, not a sentence continuation, and must not pull
        # another sentence's expressions onto this fact.
        attrs_duration = m.get("duration") or None
        # A verbatim rate expression is a FREQUENCY, never a duration: the
        # continuation walk below may still find a genuine "for N units" in
        # the following fragment, but rate expressions themselves must not
        # land in attributes.duration (they would render as "for on a
        # regular basis").
        freq_verbatim = m.get("frequency_verbatim") or \
            N._frequency_verbatim(N._norm(clause), _merged_lexicon())
        if attrs_duration:
            # Rate expressions that leaked into a stored duration (older
            # mention records, or "on a regular basis" captured before this
            # split) are re-homed onto frequency.
            parts = [p.strip() for p in attrs_duration.split(";")]
            kept, leaked = [], []
            for p in parts:
                (leaked if N._frequency_verbatim(p, _merged_lexicon()) else kept).append(p)
            if leaked:
                have = freq_verbatim.split("; ") if freq_verbatim else []
                freq_verbatim = "; ".join(dict.fromkeys(have + leaked)) or None
                attrs_duration = "; ".join(kept) or None
        if clause and clause[-1:] not in ".!?":
            # Walk forward across CONTINUATION fragments of the same sentence
            # ("...on a regular basis / like / multiple times a week."). A
            # fragment continues the sentence when it starts lowercase or with
            # a coordinating connective; Deepgram capitalizes genuine new
            # sentences ("That was long ago."), which therefore never leak
            # their temporal expressions onto this fact.
            j = m["clause_index"] + 1
            while j < len(clauses):
                frag = clauses[j].strip()
                if not frag or not (frag[:1].islower() or re.match(
                        r"^(?:and|or|but|like|which|so|because)\b", frag, re.I)):
                    break
                extra, _v, _u = N._extract_duration(N._norm(frag),
                                                    _merged_lexicon())
                if extra:
                    attrs_duration = (f"{attrs_duration}; {extra}"
                                      if attrs_duration else extra)
                extra_freq = N._frequency_verbatim(N._norm(frag),
                                                   _merged_lexicon())
                if extra_freq:
                    have = (freq_verbatim or "").split("; ") if freq_verbatim else []
                    freq_verbatim = "; ".join(
                        dict.fromkeys(have + extra_freq.split("; "))) or None
                if frag[-1:] in ".!?":
                    break
                j += 1
        location = m.get("body_location") or _clause_body_site(clause)
        fact_type = concept_fact_type(m["concept"])
        severity = m.get("severity") or _english_severity(clause)
        # A finding the patient reports belongs where findings are documented,
        # even when the sentence was dictated inside the clinician's opinion
        # section. The opposite — an opinion presented as a patient report —
        # cannot happen: a mention is only ever a mention.
        section_name = section.section
        if fact_type in ("SYMPTOM", "HISTORY", "SOCIAL_HISTORY",
                         "OCCUPATIONAL_HISTORY", "MEDICATION", "ALLERGY") \
                and section_name in ("assessment", "prognosis", "plan",
                                     "referral_context"):
            section_name = "hpi"

        facts.append({
            "fact_id": "",
            "concept": m["concept"],
            "english": english,
            "fact_type": fact_type,
            "status": status,
            "certainty": certainty,
            "temporal_context": temporal,
            "source_text": clause,
            "source_span": list(m["clause_span"]),
            "mention_span": list(m["mention_span"]),
            "section": section_name,
            "speaker": speaker_of(m["clause_index"], m),
            "confidence": m.get("confidence") or 0.0,
            "evidence": evidence,
            "attributes": {
                "duration": attrs_duration,
                "severity": severity,
                "location": location,
                "frequency": "; ".join(dict.fromkeys(
                    ([m.get("frequency")] if m.get("frequency") else [])
                    + (freq_verbatim.split("; ") if freq_verbatim else [])))
                or None,
                "onset_date": _onset_date(clause),
                "trajectory": _trajectory(clause),
                "context": m.get("context"),
                "subject": m.get("subject"),
            },
            "evidence_text": m["surface"],
            "certainty_evidence": m.get("uncertainty_reason"),
            "section_evidence": section.evidence,
            "negation_cue": negation_cue,
            "conditional": bool(m.get("conditional")),
            "episode_id": None,
            "element": None,
            "origin": "mention",
        })
        # PHASE 7 — reported speech: a third-person reporting frame ("Tanya
        # disclosed that she was still suicidal", "her mother reports that…")
        # is preserved as ATTRIBUTION on the fact. The finding itself stays
        # (the disclosure is clinical evidence) but the record shows whose
        # words carried it; nothing is inferred about speakers the transcript
        # does not name.
        m_rep = re.search(
            r"\b([A-Z][a-z]+|[Hh]er (?:mother|husband|sister|brother|partner)|[Hh]is "
            r"(?:mother|husband|wife|sister|brother|partner))\s+"
            r"(?:discloses?|disclosed|reports?|reported|says?|said|mentions?|"
            r"mentioned|told (?:me|us))\b(?:\s+that\b)?", clause)
        if m_rep:
            facts[-1].setdefault("attributes", {})["attribution"] = {
                "reported": True,
                "source": m_rep.group(1),
                "frame": "third-person disclosure",
            }

    return _reconcile_episodes(facts)


def _negation_cue(clause: str) -> str | None:
    m = re.search(r"\b(?:no|not|never|without|denies|nil)\b[^.,;]{0,40}", clause, re.I)
    return m.group(0).strip() if m else None


def _onset_date(clause: str) -> str | None:
    m = _ONSET_DATE_RE.search(clause)
    return m.group(1) if m else None


def _trajectory(clause: str) -> str | None:
    for pat, value in _TRAJECTORY_MAP:
        if re.search(pat, clause, re.I):
            return value
    return None


def _temporal_context(clause: str, mention: dict) -> str:
    trajectory = _trajectory(clause)
    if trajectory in ("IMPROVING", "WORSENING", "INTERMITTENT", "ONGOING",
                      "RECURRENT"):
        return trajectory
    if mention.get("temporality") == N.RECURRENT or mention.get("frequency") == "intermittent":
        return "INTERMITTENT"
    if mention.get("temporality") == N.PAST:
        return "HISTORICAL"
    if mention.get("status") == N.RESOLVED:
        return "RESOLVED"
    return "CURRENT"


def _reconcile_episodes(facts: list[dict]) -> list[dict]:
    """Merge only past-present + current-absent pairs of the SAME concept+site."""
    groups: dict[tuple, list[dict]] = {}
    for f in facts:
        if f["fact_type"] not in ("SYMPTOM", "HISTORY"):
            continue
        key = (f["concept"], (f["attributes"].get("location") or "").lower())
        groups.setdefault(key, []).append(f)

    for key, group in groups.items():
        historical = [f for f in group
                      if f["status"] == "PRESENT"
                      and f["temporal_context"] in ("HISTORICAL", "IMPROVING",
                                                    "WORSENING", "INTERMITTENT")]
        current_absent = [f for f in group
                          if f["status"] == "ABSENT"
                          and f["temporal_context"] == "CURRENT"]
        resolved = [f for f in group if f["status"] == "RESOLVED"]
        keep = resolved or historical
        if not keep:
            continue
        partner = current_absent or ([f for f in group if f["status"] == "ABSENT"])
        if not partner:
            continue
        base = keep[0]
        base["status"] = "RESOLVED"
        base["temporal_context"] = "RESOLVED"
        base["evidence"] = "EXPLICIT_RESOLVED"
        base["certainty"] = "CONFIRMED"
        base["confidence"] = max(base.get("confidence") or 0.0, 0.85)
        base["attributes"]["secondary_span"] = partner[0]["source_span"]
        base["attributes"]["secondary_text"] = partner[0]["source_text"]
        for f in partner:
            if f is not base:
                f["attributes"]["superseded_by"] = "resolved-episode"

    for i, f in enumerate(facts, 1):
        if f["section"] in ("pmh",) and f["fact_type"] == "HISTORY":
            f["episode_id"] = f"E{i}"
        else:
            f["episode_id"] = f"E{i}"
    return facts


# ---------------------------------------------------------------------------
# HPI elements — chronology, trajectory, aggravating and treatment response
# ---------------------------------------------------------------------------
def _hpi_elements(clauses: list[str], clause_spans, sections, mentions: list[dict]
                  ) -> list[dict]:
    """Chronology elements that are not a concept mention.

    "began after buying tight footwear", "fluctuating since March", "worse after
    he resumes his exercise regime", "no benefit from physiotherapy" — these are
    the clinical content of an HPI, and none of them survives a symptom-only
    representation.
    """
    elements: list[dict] = []

    def add(kind, text, mspan, idx, related):
        elements.append({
            "element": kind, "text": text, "source_text": clauses[idx].strip(),
            "source_span": list(clause_spans[idx]) if clause_spans[idx] else None,
            "mention_span": _span_for(clause_spans[idx], mspan),
            "section": sections[idx].section, "related_concept": related,
        })

    for idx, clause in enumerate(clauses):
        text = clause.strip()
        if _is_markup(text) or not clause_spans[idx]:
            continue
        related = None
        for m in mentions:
            if m["clause_index"] == idx and m["status"] in (N.PRESENT, N.RESOLVED):
                related = m["concept"]
                break

        # An onset mechanism is *what the patient was doing* when it started.
        # The same clause's aggravation cue ("gets worse after he resumes") is a
        # different clinical statement and is captured separately.
        m = _MECHANISM_RE.search(text)
        if (m and related and not _AGGRAVATING_RE.search(text)
                and re.search(r"\b(?:began|started|since|after|following)\b",
                              text, re.I)):
            add("ONSET_MECHANISM", m.group(1).strip(), m.span(1), idx, related)

        m = _ONSET_DATE_RE.search(text)
        if m:
            add("ONSET_DATE", m.group(1), m.span(1), idx, related)

        m = _AGGRAVATING_RE.search(text)
        if m:
            add("AGGRAVATING", text, m.span(), idx, related)

        m = _RELIEVING_RE.search(text)
        if m:
            add("RELIEVING", text, m.span(), idx, related)

        m = _NO_BENEFIT_RE.search(text)
        if m:
            add("TREATMENT_NO_BENEFIT", text, m.span(), idx, related)

        m = _BENEFIT_RE.search(text)
        if m:
            add("TREATMENT_BENEFIT", text, m.span(), idx, related)

        for pat, value in _TRAJECTORY_MAP:
            m = re.search(pat, text, re.I)
            if m:
                add(f"TRAJECTORY:{value}", text, m.span(), idx, related)
                break

        if _INVESTIGATION_TERMS and re.search(
                r"\bno (?:x-?rays?|x rays?|imaging|scans?)\b[^.]*", text, re.I):
            m = re.search(r"\bno (?:x-?rays?|x rays?|imaging|scans?)\b[^.]*", text, re.I)
            add("INVESTIGATION_ABSENT", m.group(0), m.span(), idx, related)
    return elements


# ---------------------------------------------------------------------------
# Validation (fail closed)
# ---------------------------------------------------------------------------
def validate_clinical_facts(evidence_text: str, facts: list[dict]) -> dict:
    """Reject any fact without verifiable, in-transcript evidence.

    ``evidence_text`` is the text the facts were extracted from: the raw ASR
    transcript, or its corrected copy when the verified ASR-correction layer
    produced one (the raw transcript itself is never modified).
    """
    violations: list[str] = []
    for f in facts:
        label = f"{f.get('fact_type')}/{f.get('english') or f.get('concept')}"
        if f.get("fact_type") not in FACT_TYPES:
            violations.append(f"{label}: unknown fact_type")
        if f.get("status") not in STATUSES:
            violations.append(f"{label}: unknown status {f.get('status')!r}")
        if f.get("certainty") not in CERTAINTIES:
            violations.append(f"{label}: unknown certainty {f.get('certainty')!r}")
        if f.get("temporal_context") not in TEMPORAL_CONTEXTS:
            violations.append(f"{label}: unknown temporal_context "
                              f"{f.get('temporal_context')!r}")
        if f.get("section") not in SECTIONS:
            violations.append(f"{label}: unknown section {f.get('section')!r}")
        if not f.get("source_text"):
            violations.append(f"{label}: no source_text (unsupported fact rejected)")
        if not f.get("evidence_text"):
            violations.append(f"{label}: no evidence_text (unsupported fact rejected)")
        span = f.get("source_span")
        if not span or len(span) != 2 or span[0] is None:
            violations.append(f"{label}: no source_span")
        else:
            start, end = int(span[0]), int(span[1])
            if not (0 <= start < end <= len(evidence_text)):
                violations.append(f"{label}: source_span outside the transcript")
            else:
                verbatim = evidence_text[start:end].strip()
                if f.get("source_text", "").strip() and f["source_text"].strip() not in verbatim:
                    violations.append(
                        f"{label}: source_span does not contain the cited source text")
        mspan = f.get("mention_span")
        if mspan and (mspan[0] is None
                      or not (0 <= mspan[0] < mspan[1] <= len(evidence_text))):
            violations.append(f"{label}: mention_span outside the transcript")
        if not f.get("speaker"):
            violations.append(f"{label}: no speaker")
        if f.get("confidence") is None:
            violations.append(f"{label}: no confidence")
        if f["fact_type"] == "DIAGNOSIS" and f["certainty"] == "CONFIRMED" and not (
                f.get("certainty_evidence")):
            violations.append(f"{label}: confirmed diagnosis without a confirmation cue")
        if f["fact_type"] in ("TREATMENT", "PROCEDURE") and f["status"] == "PRESENT" \
                and f.get("attributes", {}).get("benefit") == "NONE" and not (
                f.get("evidence_text")):
            violations.append(f"{label}: treatment non-response without evidence")
    return {"valid": not violations, "violations": violations,
            "checked_facts": len(facts)}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_clinical_facts(raw_text: str, turns: list[dict] | None = None,
                         roles_known: bool = True,
                         entities: list[dict] | None = None,
                         corrected_text: str | None = None) -> dict:
    """Build the V2 typed fact graph from a transcript.

    ``entities`` (the semantic layer's output) is accepted for caller
    convenience and audited, but the facts are extracted from the transcript
    itself so that per-episode structure and character spans survive.

    ``corrected_text`` is the verified ASR-correction copy (Malayalam path). The
    facts are read from THAT text — the raw transcript is never modified — and
    every fact keeps its span in the corrected copy plus, when the clause is
    unchanged, a best-effort span back into the raw transcript.
    """
    text = raw_text or ""
    evidence_text = corrected_text or text
    turns = turns or []
    lex = _merged_lexicon()
    clauses = N.split_clauses(evidence_text, lex)
    clause_spans = _clause_spans(evidence_text, clauses)

    document = detect_document_type(evidence_text, turns)
    document_type = document["document_type"]
    sections = _segment(clauses, document_type)

    def speaker_of(idx: int, mention: dict) -> str:
        subject = (mention or {}).get("subject")
        if subject in ("mother", "father", "family"):
            return subject
        if document_type in ("REFERRAL_LETTER", "MEDICOLEGAL_REPORT",
                             "MEDICAL_REPORT"):
            return "clinician"
        clause = clauses[idx]
        for t in turns:
            if clause and clause in (t.get("text") or ""):
                sp = (t.get("speaker") or "unknown").lower()
                if sp in ("doctor", "dr", "clinician"):
                    return "clinician"
                if sp in ("patient", "parent"):
                    return "patient"
                if not roles_known:
                    return "unknown"
        return "unknown" if not roles_known else "patient"

    mentions = extract_mentions(evidence_text, lex)
    mention_facts = _episode_facts(mentions, sections, clauses, document_type,
                                   speaker_of)
    statement_facts = _statement_facts(clauses, clause_spans, sections,
                                       document_type, speaker_of)
    elements = _hpi_elements(clauses, clause_spans, sections, mentions)
    metadata = extract_metadata(evidence_text, clause_spans, clauses)

    # A statement fact and a mention fact may describe the same finding (e.g.
    # "no muscular tenderness" yields an EXAM_FINDING, and the pain-family
    # mention inside it is dropped). Facts are deduplicated on
    # (fact_type, english, source_span) so no finding is reported twice.
    facts: list[dict] = []
    seen: set[tuple] = set()
    # Spans that a richer statement fact already owns: a medication PROCEDURE
    # fact ("steroid injection", CONSIDERED) must not also appear as a current
    # medication mention, and "pain free on examination" must not also appear as
    # a pain symptom — the source clause said one thing, once.
    owned_spans: list[tuple] = []
    for f in statement_facts:
        if f.get("mention_span"):
            owned_spans.append((f["mention_span"], f["fact_type"]))
        if f["fact_type"] == "EXAM_FINDING":
            # An examination sentence that yielded findings is not also a
            # symptom report ("...is full and pain free" is not "pain").
            owned_spans.append((f.get("source_span"), f["fact_type"]))

    def span_owned(span) -> bool:
        if not span:
            return False
        return any(o and not (span[1] <= o[0] or span[0] >= o[1])
                   for o, _t in owned_spans)

    for f in statement_facts + mention_facts:
        key = (f["fact_type"], (f.get("english") or "").lower(),
               tuple(f.get("source_span") or ()))
        if key in seen:
            continue
        if (f.get("origin") == "mention" and f.get("concept")
                and f["fact_type"] in ("MEDICATION", "PROCEDURE", "SYMPTOM")
                and span_owned(f.get("mention_span"))):
            continue
        seen.add(key)
        facts.append(f)

    # Diagnoses and recommendations are documented once, however many times the
    # dictation repeated them ("whiplash injury" appears in the chronology and
    # in the summary; "I would be grateful" twice in the referral letter). The
    # extra occurrences stay attached as secondary evidence.
    deduped: list[dict] = []
    seen_label: dict[str, dict] = {}
    for f in facts:
        if f["fact_type"] in ("DIAGNOSIS", "RECOMMENDATION"):
            label = f"{f['fact_type']}:{(f.get('english') or '').lower()}"
            first = seen_label.get(label)
            if first is not None:
                first.setdefault("attributes", {}).setdefault(
                    "also_stated_at", []).append(f.get("source_span"))
                continue
            seen_label[label] = f
        deduped.append(f)
    facts = deduped

    # Medication Intelligence V1 (research §K): enrich MEDICATION facts with
    # brand/dose/frequency/timing/route/duration and the 12-state status, and
    # add lexicon medications + class-level mentions the base extractor could
    # not see. Additive only — non-medication fact paths are untouched, and
    # every new fact is span-anchored so the validator below keeps enforcing
    # transcript evidence for it.
    from .medication_recognition import enrich_medication_facts, \
        load_medication_lexicon
    try:
        facts = enrich_medication_facts(facts, text,
                                        lexicon=load_medication_lexicon())
    except Exception as _med_err:            # pragma: no cover - safety valve
        # A medication-layer failure must never corrupt the clinical fact
        # graph; the failure is surfaced in the document metadata.
        facts.append({
            "fact_id": "", "concept": "medication_layer_error",
            "english": "medication layer error", "fact_type": "PLAN",
            "status": "UNCERTAIN", "certainty": "UNCERTAIN",
            "temporal_context": "CURRENT",
            "source_text": text[:0] or "medication layer error",
            "source_span": [0, 0], "mention_span": None,
            "section": "other_documentation", "speaker": "system",
            "confidence": 0.0, "evidence": "UNCERTAIN",
            "attributes": {"error": str(_med_err)},
            "evidence_text": "medication layer error",
            "certainty_evidence": None, "section_evidence": None,
            "origin": "medication_layer_error",
        })

    # The document order is evidence order: stable ids for the renderer.
    facts.sort(key=lambda f: (f.get("source_span") or [0])[0])
    for i, f in enumerate(facts, 1):
        f["fact_id"] = f"F{i}"

    # Provenance back to the raw transcript: verbatim whenever the verified ASR
    # correction layer did not touch the clause.
    for f in facts:
        clause = f.get("source_text") or ""
        idx = text.find(clause) if clause else -1
        f["raw_span"] = [idx, idx + len(clause)] if idx >= 0 else None
    for e in elements:
        clause = e.get("source_text") or ""
        idx = text.find(clause) if clause else -1
        e["raw_span"] = [idx, idx + len(clause)] if idx >= 0 else None

    # Anonymous turn labels are presentation structure, not clinical content.
    # The audio pipeline strips "Speaker N:" from the turn joins, but a stored
    # or pasted transcript can still carry them ("Speaker: Speaker 0: He is a
    # welder..."). Strip them from the STORED clinical text only — spans and
    # evidence anchoring are untouched, and the cleaned text remains a
    # substring of the evidenced span, so the fact validator is unaffected.
    for f in facts:
        st = f.get("source_text") or ""
        if _SPEAKER_LABEL_RE.search(st):
            st = _SPEAKER_LABEL_RE.sub("", st)
            st = re.sub(r"^\s*speaker\s*\d*\s*:\s*", "", st, flags=re.I).strip()
            f["source_text"] = st
    for e in elements:
        st = e.get("source_text") or ""
        if _SPEAKER_LABEL_RE.search(st):
            st = _SPEAKER_LABEL_RE.sub("", st)
            st = re.sub(r"^\s*speaker\s*\d*\s*:\s*", "", st, flags=re.I).strip()
            e["source_text"] = st

    section_index: dict[str, list[str]] = {}
    for f in facts:
        section_index.setdefault(f["section"], []).append(f["fact_id"])

    validation = validate_clinical_facts(evidence_text, facts)
    return {
        "document": document,
        "facts": facts,
        "elements": elements,
        "metadata": metadata,
        "sections": section_index,
        "validation": validation,
        "raw_text": text,
        "evidence_text": evidence_text,
        "asr_corrected": bool(corrected_text),
        "counts": {
            "facts": len(facts),
            "by_type": {t: sum(1 for f in facts if f["fact_type"] == t)
                        for t in FACT_TYPES},
            "by_section": {s: len(v) for s, v in section_index.items()},
            "mentions": len(mentions),
        },
    }
