# -*- coding: utf-8 -*-
"""Clinical fact graph — the ONLY source of truth for future note generation.

Pipeline position:

    ASR transcript (immutable)
      -> speaker segments
      -> clauses                    (normalization.split_clauses)
      -> clinical mentions          (normalization._concept_matches)
      -> scoped modifiers           (uncertainty / negation / resolution / time)
      -> clinical facts             (normalization.ClinicalEntity)
      -> FACT GRAPH                 (this module)
      -> (later) HPI / note generator consumes THIS, never raw ASR

Status model (spec v1 — do not collapse):

    PRESENT        explicitly reported
    ABSENT         explicitly denied
    RESOLVED       reported, then resolved
    UNCERTAIN      speaker unsure / feels-like (uncertainty_reason mandatory)
    QUESTIONED     asked, no finding (a question is never a finding)
    NOT_DOCUMENTED representation-level: concept never discussed. It is carried
                   by the ABSENCE of a fact, never by a fact with ABSENT.

Every fact carries provenance: surface form, source clause, raw text, speaker,
confidence, and (when applicable) uncertainty_reason. The raw transcript is
never modified anywhere in this module; corrections live in a separate copy
whose provenance points back to the raw string.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_STATUS_MAP = {
    "PRESENT": "PRESENT",
    "ABSENT": "ABSENT",
    "RESOLVED": "RESOLVED",
    "QUESTION": "QUESTIONED",
    "UNKNOWN": "UNCERTAIN",
    "POSSIBLE": "UNCERTAIN",
}

_TEMPORALITY_MAP = {
    "CURRENT": "CURRENT",
    "PAST": "PAST",
    "FUTURE": "FUTURE",
    "RECURRENT": "RECURRENT",
}

# Facts from these sections are only ever accepted with full provenance —
# these are the categories an LLM note generator is most likely to fabricate.
_CRITICAL_SECTIONS = (
    "assessment", "medications", "allergies", "pmh", "family_history",
    "social_history", "physical_exam", "plan",
)


@dataclass
class FactGraph:
    """Intermediate representation consumed by future note generation."""

    facts: list[dict] = field(default_factory=list)
    sections: dict[str, list[dict]] = field(default_factory=dict)
    raw_text: str = ""
    turns: list[dict] = field(default_factory=list)
    speaker_roles_known: bool = True
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "facts": self.facts,
            "sections": self.sections,
            "raw_text": self.raw_text,
            "turns": self.turns,
            "speaker_roles_known": self.speaker_roles_known,
            "meta": self.meta,
        }


def _fact_from_entity(e, turns: list[dict], roles_known: bool) -> dict | None:
    if isinstance(e, dict):
        get = e.get
    else:
        get = lambda k, d=None: getattr(e, k, d)   # ClinicalEntity objects
    concept = get("concept")
    if not concept:
        return None
    raw_status = get("status") or ""
    status = _STATUS_MAP.get(raw_status, raw_status)
    if status is None:
        return None

    # Speaker attribution: match the clause to its turn when possible.
    # Dict inputs may carry either the entity-contract key ("subject") or an
    # already-resolved "speaker"; object inputs expose attributes.
    speaker = get("subject") or (get("speaker") if isinstance(e, dict) else None) or "UNKNOWN"
    clause = get("source_clause") or ""
    if speaker not in ("mother", "father", "family"):
        speaker = "UNKNOWN"
        for t in turns:
            ttext = t.get("text") or ""
            # Match on the clause or any of its '|' segments (merged entities
            # join contributing clauses); segments allow line-level matching
            # even when the full merged clause is not a substring of one turn.
            candidates = [s.strip() for s in clause.split("|")] if clause else []
            if any(seg and seg in ttext for seg in candidates):
                sp = t.get("speaker") or "UNKNOWN"
                speaker = sp if roles_known else "UNKNOWN"
                break

    temporality = get("temporality") or "CURRENT"
    if status == "RESOLVED":
        temporality = "PAST_RESOLVED"

    attributes = {}
    for key in ("duration", "severity", "body_location", "pain_quality"):
        if get(key):
            attributes[key] = get(key)
    if get("frequency_normalized"):
        attributes["frequency"] = get("frequency_normalized")
    # Dict inputs may pass a pre-built attributes map (test/adapter shapes);
    # explicit top-level keys win so the entity contract stays authoritative.
    if isinstance(e, dict) and e.get("attributes"):
        for k, v_ in e["attributes"].items():
            attributes.setdefault(k, v_)
    onset = get("onset")
    duration = get("duration")

    fact = {
        "concept": concept,
        "english": get("english"),
        "status": status,
        "certainty": "CONFIRMED" if status in ("PRESENT", "ABSENT", "RESOLVED")
                     else "UNCERTAIN",
        "temporality": temporality,
        "duration": duration,
        "onset": onset,
        "frequency": get("frequency_normalized"),
        "surface_text": get("surface_text") or "",
        "source_clause": clause,
        "speaker": speaker,
        "confidence": get("confidence") or 0.0,
        "attributes": attributes,
        "uncertainty_reason": get("uncertainty_reason"),
        "raw_text": None,     # filled from graph.raw_text at build time
    }
    # Explicit persistence evidence in the source clause ("ഇപ്പോഴും ഉണ്ട്",
    # "തുടരുന്നു", English "still ...") — never inferred from bare PRESENT.
    if any(m in clause for m in ("ഇപ്പോഴും", "തുടരുന്നു")) or " still " in f" {clause} ":
        fact["ongoing"] = True
    elif get("ongoing"):
        fact["ongoing"] = True
    if get("conditional"):
        fact["conditional"] = True
    return fact


def build_fact_graph(entities_dicts: list[dict], turns: list[dict] | None = None,
                     raw_text: str = "", roles_known: bool = True) -> FactGraph:
    """Build the fact graph from semantic-layer entities + speaker turns.

    ``entities_dicts``: ClinicalEntity.as_dict() records (the semantic layer's
    output — unchanged). ``turns``: diarization turns. ``raw_text``: the raw
    ASR transcript, stored verbatim for provenance (never modified).
    """
    facts: list[dict] = []
    raw_text = raw_text or ""
    for e in entities_dicts or []:
        fact = _fact_from_entity(e, turns or [], roles_known)
        if fact:
            fact["raw_text"] = raw_text    # verbatim provenance, never modified
            facts.append(fact)

    return FactGraph(
        facts=facts,
        sections={},                      # populated by documentation layer
        raw_text=raw_text,
        turns=list(turns or []),
        speaker_roles_known=bool(roles_known),
        meta={"fact_count": len(facts)},
    )


def validate_fact_graph(graph: FactGraph) -> dict:
    """Anti-hallucination validator.

    Checks:
      1. every fact has provenance (surface_text AND source_clause AND
         confidence AND speaker);
      2. no orphan concepts — every fact's concept must originate from the
         semantic layer's surface evidence (surface_text non-empty), i.e. no
         concept introduced purely by inference;
      3. section facts in hallucination-prone categories (assessment,
         medications, allergies, PMH, family/social history, exam, plan) are
         rejected unless fully provenanced.
    """
    violations: list[str] = []

    for i, f in enumerate(graph.facts):
        concept = f.get("concept") or f"<fact[{i}]>"
        if not f.get("source_clause") or not f.get("surface_text"):
            violations.append(
                f"{concept}: missing provenance (surface_text/source_clause empty)")
        if f.get("confidence") in (None, 0) and f.get("status") != "QUESTIONED":
            violations.append(f"{concept}: missing confidence")
        if not f.get("speaker"):
            violations.append(f"{concept}: missing speaker attribution")

    for section, facts in (graph.sections or {}).items():
        if section not in _CRITICAL_SECTIONS:
            continue
        for f in facts:
            concept = f.get("concept") or section
            if not f.get("source_clause") or not f.get("surface_text"):
                violations.append(
                    f"{section}/{concept}: inference-only fact without provenance "
                    f"rejected")

    return {
        "valid": not violations,
        "violations": violations,
        "checked_facts": len(graph.facts),
    }
