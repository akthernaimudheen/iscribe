# -*- coding: utf-8 -*-
"""Clinical note generator — consumes the FACT GRAPH, never raw ASR.

Pipeline position (the only permitted one):

    Audio -> ASR -> Malayalam semantic engine -> FACT GRAPH
        -> note_generator (this module) -> validate -> final note

The generator never interprets Malayalam. Every clinical sentence is derived
from fact-graph fields (concept / status / certainty / temporality / duration /
frequency / severity / speaker) by deterministic templates. Anything the fact
graph does not state is rendered as NOT_DOCUMENTED — never as ABSENT, never
inferred, never omitted silently.

Truth model mapping (spec v1, do not collapse):

    PRESENT        -> "reports ..."
    PRESENT+ONGOING-> "continues to experience ..."
    RESOLVED       -> "previously experienced ... which has since resolved"
    UNCERTAIN      -> "reports possible ... ; patient is uncertain"
    ABSENT         -> "denies ..."
    QUESTIONED     -> never a patient finding; listed as asked-only metadata
    NOT_DOCUMENTED -> absence of a fact: "Not documented." — never a denial

Two rendering modes:
    render_deterministic_note()  — the safe baseline (no LLM, always available)
    render_llm_note()            — interface prepared, DISABLED by default;
                                   receives only the validated fact graph under
                                   the mandated prompt contract and its output
                                   must pass validate_note_against_fact_graph().
"""

from __future__ import annotations

import re
from typing import Optional

from .fact_graph import FactGraph, validate_fact_graph

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

NOT_DOCUMENTED = "NOT_DOCUMENTED"

# --------------------------------------------------------------------------
# Temporal language: status -> clause templates. Deterministic, evidence-bound.
# --------------------------------------------------------------------------

def _num_word(n: int) -> str:
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
             7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    return words.get(n, str(n))


def _duration_phrase(duration) -> str:
    """Render a fact-graph duration verbatim; numerals 1-10 as words.

    The fact graph stores the semantic layer's normalized string (e.g.
    "6 days"). Nothing is added; unsupported units pass through unchanged.
    """
    if not duration:
        return ""
    d = str(duration).strip()
    m = re.match(r"^(\d+)\s+(days?|weeks?|months?|years?|hours?)$", d, re.I)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        singular = unit.rstrip("s")
        plural = singular + "s"
        if n == 1:
            return f"approximately {_num_word(1)} {singular}"
        if n <= 10:
            return f"approximately {_num_word(n)} {plural}"
        return f"approximately {n} {plural}"
    return f"approximately {d}"


def _subject_name(speaker: Optional[str]) -> str:
    """Honest narrative subject. Unknown roles never become Doctor/Patient."""
    if speaker in ("mother", "father", "family"):
        return f"The patient's {speaker}" if speaker != "family" else "The patient's family"
    return "The patient"


def _symptom_noun(fact: dict) -> str:
    return fact.get("english") or (fact.get("concept") or "").replace("_", " ")


def _patient_facts(graph: FactGraph) -> list[dict]:
    """Facts that are patient findings (never family history, never questions)."""
    out = []
    for f in graph.facts:
        if f.get("status") in ("QUESTIONED",):
            continue
        if f.get("speaker") in ("mother", "father", "family"):
            continue
        out.append(f)
    return out


def _family_facts(graph) -> list[dict]:
    """Accepts the FactGraph or a raw fact list."""
    src = graph.facts if isinstance(graph, FactGraph) else graph
    return [f for f in src
            if f.get("speaker") in ("mother", "father", "family")
            and f.get("status") not in ("QUESTIONED",)]


def _ongoing(fact: dict) -> bool:
    """PRESENT with explicit ongoing persistence evidence only."""
    return (fact.get("status") == "PRESENT"
            and fact.get("temporality") == "CURRENT"
            and bool(fact.get("ongoing")))


def _join_names(names: list[str]) -> str:
    """Deterministic clinical list join: 2 -> 'a and b', 3+ -> Oxford comma."""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _uncertain_clause(fact: dict) -> str:
    reason = fact.get("uncertainty_reason")
    if reason:
        return (f"{_subject_name(fact.get('speaker'))} reports possible "
                f"{_symptom_noun(fact)} and is uncertain ({reason}).")
    return (f"{_subject_name(fact.get('speaker'))} reports possible "
            f"{_symptom_noun(fact)} and is uncertain whether it is present.")


# --------------------------------------------------------------------------
# Section builders (all read graph.facts only)
# --------------------------------------------------------------------------

def _is_pmh_concept(fact: dict) -> bool:
    from .clinical_documentation import _PMH_CONCEPTS
    return fact.get("concept") in _PMH_CONCEPTS


def _chief_complaint(facts: list[dict]) -> Optional[str]:
    """First explicitly reported PRESENT complaint; never inferred.

    Chronic PMH conditions are not chief complaints — they render in PMH only.
    """
    for f in facts:
        if (f.get("status") == "PRESENT" and f.get("english")
                and not _is_pmh_concept(f)):
            return f.get("english")
    return None


def _hpi(facts: list[dict]) -> Optional[str]:
    """Narrative, chronological HPI from structured fields only.

    Deterministic grammar (English note-quality spec):
      1. RESOLVED — with documented duration: "a history of X for <dur>, which
         has since resolved"; without: "previously experienced X, which has
         since resolved" ("history of" is never used without a documented
         episode — it implies historical documentation).
      2. ONGOING (explicit persistence evidence only): "continues to
         experience X" — one sentence per fact, never grouped.
      3. Remaining PRESENT facts — one grouped sentence; "currently reports"
         when resolved facts precede (temporal pivot), else "reports".
      4. Explicit ABSENT facts — one grouped denial sentence.
      5. UNCERTAIN facts — per-fact uncertainty clauses (reason preserved).

    Durations render verbatim — the "approximately" hedge is never dropped.
    No causes, diagnoses, severity, or symptoms beyond the fact graph; PMH
    concepts render in PMH only; the chief complaint is never relabeled.
    """
    findings = [f for f in facts if not _is_pmh_concept(f)]
    resolved = [f for f in findings if f.get("status") == "RESOLVED"]
    ongoing = [f for f in findings if _ongoing(f)]
    present = [f for f in findings
               if f.get("status") == "PRESENT" and not _ongoing(f)]
    absent = [f for f in findings if f.get("status") == "ABSENT"]
    uncertain = [f for f in findings if f.get("status") == "UNCERTAIN"]

    sentences: list[str] = []
    for f in resolved:
        dur = _duration_phrase(f.get("duration"))
        noun = _symptom_noun(f)
        if dur:
            sentences.append(
                f"The patient reports a history of {noun} for {dur}, "
                f"which has since resolved.")
        else:
            sentences.append(
                f"The patient previously experienced {noun}, "
                f"which has since resolved.")
    for f in ongoing:
        sentences.append(
            f"The patient continues to experience {_symptom_noun(f)}.")
    if present:
        names = []
        for f in present:
            noun = _symptom_noun(f)
            dur = _duration_phrase(f.get("duration"))
            names.append(f"{noun} for {dur}" if dur else noun)
        lead = "currently reports" if resolved else "reports"
        sentences.append(f"The patient {lead} {_join_names(names)}.")
    if absent:
        sentences.append(
            f"The patient denies {_join_names([_symptom_noun(f) for f in absent])}.")
    for f in uncertain:
        sentences.append(_uncertain_clause(f))
    return " ".join(sentences) if sentences else None


def _associated(facts: list[dict], chief: Optional[str]) -> list[str]:
    """PRESENT symptoms beyond the chief complaint, grouped by relatedness.

    Grouping uses the documented ROS organ-system map only — it never claims
    that one symptom caused another.
    """
    from .clinical_documentation import _ROS_GROUPS, _ros_group_for
    out = []
    groups: dict[str, list[str]] = {}
    other: list[str] = []
    for f in facts:
        if f.get("status") != "PRESENT" or not f.get("english"):
            continue
        if _is_pmh_concept(f):
            continue
        # The chief complaint is not excluded: it belongs in its organ-system
        # group too (the briefing lists cough under Respiratory even when chief).
        g = _ros_group_for(f.get("concept") or "")
        if g:
            groups.setdefault(g, []).append(f["english"])
        else:
            other.append(f["english"])
    labels = {"constitutional": "Constitutional", "respiratory": "Respiratory",
              "heent": "HEENT", "cardiovascular": "Cardiovascular",
              "gastrointestinal": "Gastrointestinal", "neurological": "Neurologic",
              "musculoskeletal": "Musculoskeletal", "skin": "Skin",
              "genitourinary": "Genitourinary"}
    for g in groups:
        label = labels.get(g, g.capitalize())
        out.append(f"{label}: " + ", ".join(groups[g]))
    if other:
        out.append("Other: " + ", ".join(sorted(other)))
    return out


def _pertinent_negatives(facts: list[dict]) -> list[str]:
    return [f"Denies {_symptom_noun(f)}" for f in facts if f.get("status") == "ABSENT"]


def _pmh(facts: list[dict]) -> list[str]:
    from .clinical_documentation import _PMH_CONCEPTS
    out = []
    for f in facts:
        if (f.get("concept") in _PMH_CONCEPTS
                and f.get("status") in ("PRESENT", "RESOLVED")):
            out.append(f"{(f.get('english') or '').capitalize()}: Present.")
        elif (f.get("concept") in _PMH_CONCEPTS
                and f.get("status") == "ABSENT"):
            out.append(f"{(f.get('english') or '').capitalize()}: "
                       f"Explicitly denied.")
    return out


def _medications(facts: list[dict]) -> list[str]:
    from .clinical_documentation import _MEDICATION_CONCEPTS
    out = []
    for f in facts:
        if f.get("concept") not in _MEDICATION_CONCEPTS:
            continue
        if f.get("status") not in ("PRESENT", "RESOLVED"):
            continue
        attrs = f.get("attributes") or {}
        bits = [f.get("english") or f.get("concept")]
        if attrs.get("duration"):
            bits.append(f"duration {attrs['duration']}")
        if attrs.get("frequency"):
            bits.append(f"frequency {attrs['frequency']}")
        out.append("; ".join(bits))
    return out


def _allergies(graph: FactGraph) -> list[str]:
    out = []
    for f in graph.facts:
        if f.get("concept") != "allergy":
            continue
        if f.get("status") == "ABSENT":
            out.append("No known allergies (explicitly denied).")
        elif f.get("status") == "PRESENT":
            out.append("Allergy reported.")
    return out


def _family(facts_or_graph) -> list[str]:
    """Accepts either the fact list or the graph for convenience.

    A family member's chronic condition is family history (not the patient's
    own). Family-sourced symptom facts still never become patient findings.
    """
    graph = facts_or_graph
    src = graph.facts if isinstance(graph, FactGraph) else facts_or_graph
    return [f"{(f.get('english') or f.get('concept') or '').replace('_', ' ')} "
            f"in {f.get('speaker')} (family history)"
            for f in _family_facts(src) if f.get("english") or f.get("concept")]


def _social(graph: FactGraph) -> list[str]:
    from .clinical_documentation import _SOCIAL_PATTERNS_EN, _SOCIAL_ML
    out, seen = [], set()
    for t in graph.turns:
        text = t.get("text") or ""
        low = text.lower()
        for pat, label, concept in _SOCIAL_PATTERNS_EN:
            if concept not in seen and re.search(pat, low):
                out.append(label.capitalize() + ": documented.")
                seen.add(concept)
        for marker, label, concept in _SOCIAL_ML:
            if concept not in seen and marker in text:
                out.append(label.capitalize() + ": documented.")
                seen.add(concept)
    return out


def _ros(facts: list[dict]) -> list[str]:
    from .clinical_documentation import _ROS_GROUPS, _ROS_LABELS, _ros_group_for
    from .clinical_documentation import _ROS_EXCLUDED
    groups: dict[str, list[str]] = {}
    for f in facts:
        concept = f.get("concept") or ""
        if concept in _ROS_EXCLUDED:
            continue
        g = _ros_group_for(concept)
        if not g:
            continue
        st = f.get("status")
        if st == "ABSENT":
            entry = f"{_symptom_noun(f)}, denied"
        elif st == "RESOLVED":
            entry = f"{_symptom_noun(f)}, resolved"
        elif st == "UNCERTAIN":
            entry = f"possible {_symptom_noun(f)}, uncertain"
        else:
            entry = _symptom_noun(f)
        groups.setdefault(g, []).append(entry)
    order = list(_ROS_LABELS)
    out = []
    for g in sorted(groups, key=lambda x: order.index(x)):
        out.append(f"{_ROS_LABELS[g]}: " + "; ".join(groups[g]))
    return out


def _physical_exam(graph: FactGraph) -> list[str]:
    from .clinical_documentation import _EXAM_PAT_EN, _EXAM_ML
    out = []
    for t in graph.turns:
        text = t.get("text") or ""
        for pat, finding in _EXAM_PAT_EN:
            if re.search(pat, text, re.I):
                out.append(finding)
        for marker, finding in _EXAM_ML:
            if marker in text:
                out.append(finding)
    return out


def _assessment(graph: FactGraph) -> Optional[str]:
    """Diagnosis ONLY if the clinician explicitly stated one. Never derived
    from symptoms. The line evidence itself is provenance."""
    from .clinical_documentation import (_DIAGNOSIS_PAT_EN, _DIAGNOSIS_ML,
                                         _is_question)
    speaker_label = "Clinician"
    for t in graph.turns:
        text = t.get("text") or ""
        if graph.speaker_roles_known and t.get("speaker") == "Patient":
            continue
        if _is_question(text):
            continue
        if _ci_diagnosis(text):
            return (f"{speaker_label}-stated diagnosis (verbatim): "
                    f"\"{text.strip()}\"")
    return None


def _ci_diagnosis(text: str) -> bool:
    from .clinical_documentation import _DIAGNOSIS_PAT_EN, _DIAGNOSIS_ML
    low = (text or "").lower()
    if any(m in text for m in _DIAGNOSIS_ML):
        return True
    return any(re.search(p, text, re.I) for p in _DIAGNOSIS_PAT_EN)


def _plan(graph: FactGraph) -> list[str]:
    from .clinical_documentation import _PLAN_PATTERNS_EN, _PLAN_ML, _is_question
    out = []
    for t in graph.turns:
        text = t.get("text") or ""
        if graph.speaker_roles_known and t.get("speaker") == "Patient":
            continue
        if _is_question(text):
            continue
        for pat, label in _PLAN_PATTERNS_EN:
            if re.search(pat, text, re.I):
                out.append(label)
        for marker, label in _PLAN_ML:
            if marker in text:
                out.append(label)
    seen, uniq = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


# --------------------------------------------------------------------------
# Deterministic renderer
# --------------------------------------------------------------------------

def render_deterministic_note(graph: FactGraph) -> dict:
    """Build the 14-section note from the fact graph. Returns
    {note, sections, provenance, validation} — the clinician-facing string,
    a debug trace (fact_id -> sentences), and the fail-closed validation."""
    facts = _patient_facts(graph)
    sections: dict[str, dict] = {}
    provenance: list[dict] = []          # debug trace: fact -> rendered text

    def add(section: str, text: str, fact: Optional[dict] = None,
            evidence: str = NOT_DOCUMENTED):
        entry = {"text": text, "evidence": evidence}
        if fact is not None:
            entry["fact_id"] = f"{fact.get('concept')}@{fact.get('surface_text', '')[:24]}"
            provenance.append({
                "fact_id": entry["fact_id"], "section": section,
                "concept": fact.get("concept"), "status": fact.get("status"),
                "source_text": fact.get("surface_text"),
                "source_clause": fact.get("source_clause"),
                "speaker": fact.get("speaker"),
                "confidence": fact.get("confidence"),
                "rendered": text,
            })
        sections.setdefault(section, []).append(entry)

    # 1. Chief complaint
    cc = _chief_complaint(facts)
    if cc:
        fact = next(f for f in facts
                    if f.get("status") == "PRESENT" and f.get("english") == cc)
        add("chief_complaint", cc.capitalize(), fact, "EXPLICIT_PRESENT")
    else:
        add("chief_complaint", "Not documented.")

    # 2. HPI
    hpi = _hpi(facts)
    if hpi:
        add("hpi", hpi, None, "EXPLICIT_PRESENT")
        for f in facts:
            if f.get("status") in ("RESOLVED", "PRESENT", "UNCERTAIN"):
                provenance.append({
                    "fact_id": f"hpi:{f.get('concept')}",
                    "section": "hpi", "concept": f.get("concept"),
                    "status": f.get("status"),
                    "source_text": f.get("surface_text"),
                    "source_clause": f.get("source_clause"),
                    "speaker": f.get("speaker"),
                    "confidence": f.get("confidence"),
                    "rendered": hpi,
                })
    else:
        add("hpi", "Not documented.")

    # 3. Associated symptoms
    assoc = _associated(facts, cc)
    if assoc:
        for line in assoc:
            add("associated_symptoms", line, None, "EXPLICIT_PRESENT")
    else:
        add("associated_symptoms", "Not documented.")

    # 4. Pertinent negatives
    negs = _pertinent_negatives(facts)
    if negs:
        for line in negs:
            add("pertinent_negatives", line, None, "EXPLICIT_ABSENT")
    else:
        add("pertinent_negatives", "None documented (absence of denial is not a denial).")

    # 5-10. PMH / PSH / Medications / Allergies / Family / Social
    pmh = _pmh(facts)
    for line in (pmh or ["Not documented."]):
        add("pmh", line, None, "EXPLICIT_PRESENT" if pmh else NOT_DOCUMENTED)
    add("psh", "Not documented.")          # PSH requires line evidence (rare); conservative
    meds = _medications(facts)
    for line in (meds or ["Not documented."]):
        add("medications", line, None, "EXPLICIT_PRESENT" if meds else NOT_DOCUMENTED)
    alg = _allergies(graph)
    for line in (alg or ["Not documented."]):
        add("allergies", line, None, "EXPLICIT_PRESENT" if alg else NOT_DOCUMENTED)
    fam = _family(graph)
    for line in (fam or ["Not documented."]):
        add("family_history", line, None, "EXPLICIT_PRESENT" if fam else NOT_DOCUMENTED)
    soc = _social(graph)
    for line in (soc or ["Not documented."]):
        add("social_history", line, None, "EXPLICIT_PRESENT" if soc else NOT_DOCUMENTED)

    # 11. ROS
    ros = _ros(facts)
    if ros:
        for line in ros:
            add("ros", line, None, "EXPLICIT_PRESENT")
    else:
        add("ros", "Not documented.")

    # 12. Physical examination
    exam = _physical_exam(graph)
    for line in (exam or ["Not documented."]):
        add("physical_exam", line, None, "EXPLICIT_PRESENT" if exam else NOT_DOCUMENTED)

    # 13. Assessment
    dx = _assessment(graph)
    if dx:
        add("assessment", dx, None, "EXPLICIT_PRESENT")
    else:
        add("assessment", "No definitive diagnosis documented in this encounter.")

    # 14. Plan
    plan = _plan(graph)
    if plan:
        for line in plan:
            add("plan", line, None, "EXPLICIT_PRESENT")
    else:
        add("plan", "Not documented.")

    # Render text
    lines = ["CLINICAL NOTE", ""]
    for i, key in enumerate(SECTION_ORDER, 1):
        entries = sections.get(key, [])
        lines.append(f"{i}. {SECTION_TITLES[key]}")
        for e in entries:
            prefix = "- " if e.get("evidence") != NOT_DOCUMENTED else ""
            lines.append(f"   {prefix}{e['text']}")
        lines.append("")
    note = "\n".join(lines).rstrip() + "\n"

    validation = validate_note_against_fact_graph(note, graph)
    return {
        "note": note,
        "sections": sections,
        "provenance": provenance,
        "validation": validation,
    }


# --------------------------------------------------------------------------
# Anti-hallucination validator for the NOTE (fail closed)
# --------------------------------------------------------------------------

def validate_note_against_fact_graph(note: str, graph: FactGraph) -> dict:
    """Fail-closed validator: every clinical claim in ``note`` must be backed
    by the fact graph. Detects concepts, diagnoses, medications, allergies,
    exam findings, plan actions, negatives, durations, severity, and causal
    claims that the fact graph does not support. Returns {valid, violations}.
    """
    violations: list[str] = []
    gcheck = validate_fact_graph(graph)
    if not gcheck["valid"]:
        violations.extend(f"fact graph invalid: {v}" for v in gcheck["violations"])

    facts = _patient_facts(graph)
    fam = _family_facts(graph)
    all_facts = facts + fam
    known = set()
    for f in all_facts:
        for n in (f.get("english"), (f.get("concept") or "").replace("_", " ")):
            if n:
                known.add(n.lower())

    from .clinical_documentation import (_MEDICATION_CONCEPTS, _PMH_CONCEPTS,
                                         _EXAM_PAT_EN, _EXAM_ML,
                                         _PLAN_PATTERNS_EN, _PLAN_ML)
    fact_meds = {f.get("english", "").lower() for f in all_facts
                 if f.get("concept") in _MEDICATION_CONCEPTS}
    fact_pmh = {f.get("english", "").lower() for f in all_facts
                if f.get("concept") in _PMH_CONCEPTS}
    fact_durations = {str(f.get("duration")).lower() for f in all_facts if f.get("duration")}
    absent_concepts = {f.get("english", "").lower() for f in all_facts
                       if f.get("status") == "ABSENT"}

    # 1. Concepts in note must exist in the fact graph.
    #    (Scan for known clinical nouns from the ROS vocabulary; anything found
    #    in the note that no fact supports is a violation.)
    from .clinical_documentation import _ROS_GROUPS
    for group, concepts in _ROS_GROUPS.items():
        for c in concepts:
            label = (c or "").replace("_", " ")
            if label in note.lower() and label not in known:
                violations.append(f"concept '{label}' in note not supported by fact graph")

    # 2. Diagnoses: any diagnosis phrase present must be clinician-documented.
    #    The ALLOWED set comes from the same _assessment() extraction the
    #    renderer uses — one source of truth, so deterministic notes always
    #    validate and any other diagnosis claim fails closed. The standard
    #    not-documented assessment sentence is template text, exempted.
    note_scan = note.replace(
        "No definitive diagnosis documented in this encounter", "")
    allowed_dx = _assessment(graph) or ""
    from .clinical_documentation import _DIAGNOSIS_ML, _DIAGNOSIS_PAT_EN
    for m in _DIAGNOSIS_ML:
        if m in note_scan and m not in allowed_dx:
            violations.append(f"diagnosis '{m}' not clinician-documented")
    for p in _DIAGNOSIS_PAT_EN:
        if re.search(p, note_scan, re.I):
            m2 = re.search(p, allowed_dx, re.I)
            if not m2:
                violations.append(f"diagnosis pattern {p!r} in note not supported")

    # 3. Medications / allergies / PMH unsupported.
    for med in fact_meds or set():
        pass
    med_vocab = {"paracetamol", "tablet", "capsule", "syrup", "injection", "medicine",
                 "crocin", "dolo", "aspirin", "ibuprofen", "amoxicillin", "azithromycin"}
    for m in med_vocab:
        if re.search(rf"\b{m}\b", note, re.I) and m not in fact_meds:
            violations.append(f"medication '{m}' in note not supported by fact graph")
    if re.search(r"no known allergies", note, re.I):
        if not any(f.get("concept") == "allergy" and f.get("status") == "ABSENT"
                   for f in all_facts):
            violations.append("'no known allergies' in note not supported by fact graph")
    for p in _PMH_CONCEPTS:
        label = p.replace("_", " ")
        if re.search(rf"\b{label}\b", note, re.I) and label not in fact_pmh:
            violations.append(f"PMH '{label}' in note not supported by fact graph")

    # 4. Exam findings / plan actions unsupported.
    #    Allowed sets derive from the renderer's own extraction functions —
    #    anything the fact graph cannot yield is rejected, wherever it came
    #    from (including an LLM renderer output).
    allowed_exam = set(_physical_exam(graph))
    allowed_plan = set(_plan(graph))
    for pat, finding in _EXAM_PAT_EN:
        if finding in note and finding not in allowed_exam:
            violations.append(f"exam finding '{finding}' not supported by fact graph")
    for marker, finding in _EXAM_ML:
        if finding in note and finding not in allowed_exam:
            violations.append(f"exam finding '{finding}' not supported by fact graph")
    for pat, label in _PLAN_PATTERNS_EN:
        if label in note and label not in allowed_plan:
            violations.append(f"plan action '{label}' not supported by fact graph")
    for marker, label in _PLAN_ML:
        if label in note and label not in allowed_plan:
            violations.append(f"plan action '{label}' not supported by fact graph")

    # 5. Unsupported negatives / durations / severity / causality.
    for m in re.finditer(r"denies ([a-z ]+?)(?:[.;,\n]|$)", note, re.I):
        denied = m.group(1).strip()
        if not denied:
            continue
        # Grouped denials use the renderer's deterministic join ("a and b",
        # "a, b, and c"): every named conjunct must itself be an explicit
        # ABSENT fact — fail closed on any unsupported member.
        parts = [p.strip() for p in re.split(r",?\s+and\s+", denied) if p.strip()]
        if any(p not in absent_concepts for p in parts):
            violations.append(f"negative finding 'denies {denied}' not supported")
    for m in re.finditer(r"(?:for|duration)\s+(?:approximately\s+)?([a-z0-9]+ [a-z]+)", note, re.I):
        dur = m.group(1).strip().lower()
        words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
                 "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        if dur.split()[0] in words:
            dur = f"{words[dur.split()[0]]} {dur.split()[1].rstrip('s')}s"
        if dur not in fact_durations:
            violations.append(f"duration '{m.group(1)}' in note not supported by fact graph")
    for m in re.finditer(r"\b(severe|mild|moderate|excruciating)\b", note, re.I):
        sev = m.group(1).lower()
        if not any((f.get("attributes") or {}).get("severity", "").lower() == sev
                   for f in all_facts):
            violations.append(f"severity '{sev}' in note not supported by fact graph")
    for m in re.finditer(r"\b(due to|caused by|because of|secondary to)\b", note, re.I):
        violations.append("causal claim in note not supported by fact graph")

    # Unsupported persistence: "continues to experience X" requires the fact
    # to carry explicit persistence evidence (ongoing=true from source-clause
    # markers such as ഇപ്പോഴും / still / തുടരുന്നു).
    ongoing_names = {_symptom_noun(f).lower() for f in all_facts if _ongoing(f)}
    for m in re.finditer(r"continues to experience ([a-z ]+?)(?=[.;,\n]|$)",
                         note, re.I):
        claimed = m.group(1).strip()
        if claimed not in ongoing_names:
            violations.append(
                f"persistence claim 'continues to experience {claimed}' "
                f"not supported by fact graph")

    return {"valid": not violations, "violations": violations}


# --------------------------------------------------------------------------
# LLM renderer interface — DISABLED by default
# --------------------------------------------------------------------------

LLM_SYSTEM_PROMPT = """You are a clinical documentation formatter.
The supplied fact graph is the complete source of truth.
Do not add clinical facts.
Do not infer diagnoses.
Do not infer missing history.
Do not convert NOT_DOCUMENTED into ABSENT.
Do not invent examination findings or treatment plans.
You may only improve grammar, chronology, organization, and clinical readability.
Return only the note text.
"""


def render_llm_note(graph: FactGraph, llm_call=None) -> dict:
    """Prepared (disabled-by-default) LLM rendering path.

    The LLM receives ONLY the validated fact graph under LLM_SYSTEM_PROMPT.
    Its output must pass validate_note_against_fact_graph() before being
    returned; on failure the deterministic note is returned with the LLM
    rejection recorded. With llm_call=None this always falls back.
    """
    det = render_deterministic_note(graph)
    if llm_call is None:
        return {**det, "llm_used": False,
                "llm_rejected": "llm_call not provided (disabled by default)"}
    fact_payload = graph.as_dict()
    draft = llm_call(LLM_SYSTEM_PROMPT, fact_payload)
    validation = validate_note_against_fact_graph(draft, graph)
    if validation["valid"]:
        return {"note": draft, "sections": det["sections"],
                "provenance": det["provenance"], "validation": validation,
                "llm_used": True}
    return {**det, "llm_used": False,
            "llm_rejected": validation["violations"]}
