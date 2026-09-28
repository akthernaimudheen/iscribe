"""Deterministic projection: typed fact graph -> reviewable structured fields.

THE data-flow fix for the review-fields divergence. There is exactly one
clinical extractor (the fact graph in ``clinical_facts``) and this module
projects ITS verified facts into the flat field dictionaries the Review UI
edits. Nothing here re-reads the transcript, re-scans keywords, or invents
values: every populated field traces to fact-graph evidence, and everything
the fact graph did not document reads "Not documented." — never a guess.

Semantics preserved, never collapsed (see note_v2's truth model):

    ABSENT        -> a documented negative ("Denies ...", "No bony tenderness")
    CONSIDERED    -> "considered, not performed"
    RECOMMENDED   -> "- recommended"
    SUSPECTED     -> "Suspected (not confirmed): ..." (uncertainty kept)
    confirmed != suspected; considered != performed

Symptoms keep their anatomical location ("Pain - left medial foot arch"),
exam findings stay examination findings (they are NOT patient-reported
negatives), and temporal structure (onset / course / aggravation / treatment
response) stays separated from duration. The SAME functions the canonical
note renderer uses are reused where a rendering exists, so the fields and the
note cannot silently diverge.
"""

from __future__ import annotations

from .note_v2 import (_assessment_lines, _cap, _exam_lines, _metadata_lines,
                      _norm_label, _plan_lines, _pmh_lines, _sentence,
                      _symptom_noun)

NOT_MENTIONED = "Not mentioned"
NOT_DOCUMENTED = "Not documented."

_LOCATION_PREP_RE = None  # reserved; locations render via _subject_clause


def _fact_location(fact: dict) -> str | None:
    return (fact.get("attributes") or {}).get("location")


def _symptom_label(fact: dict) -> str:
    """A symptom with its documented anatomical site, when one was stated.

    "pain" + left medial foot arch -> "Pain - left medial foot arch" (the
    site is the differentiator the clinician documented; collapsing to the
    bare noun is the information loss this module exists to prevent).
    """
    noun = _symptom_noun(fact)
    location = _fact_location(fact)
    if not location:
        return _cap(noun)
    low_noun, low_loc = noun.lower(), location.lower()
    if low_loc in low_noun or low_noun in low_loc:
        return _cap(noun)
    return f"{_cap(noun)} - {location}"


def _clinical_note_fields(document: dict) -> dict:
    """Project the fact graph into the editable clinical-note fields."""
    facts = document.get("facts") or []
    elements = document.get("elements") or []
    metadata = document.get("metadata") or {}

    symptoms = [f for f in facts if f["fact_type"] == "SYMPTOM"
                and f["status"] in ("PRESENT", "ONGOING", "RESOLVED",
                                    "INTERMITTENT", "UNCERTAIN")]

    # 1. Symptoms reported — every documented symptom with its site. A
    # bare-noun mention of a symptom that also appears WITH a documented site
    # is the same clinical fact mentioned generically (the graph carries one
    # mention per clause): the located statement is the complete one.
    by_noun: dict[str, list[str]] = {}
    noun_order: list[str] = []
    for f in symptoms:
        noun = _symptom_noun(f).lower()
        if noun not in by_noun:
            by_noun[noun] = []
            noun_order.append(noun)
        by_noun[noun].append(_symptom_label(f))
    symptom_labels: list[str] = []
    seen_labels: set[str] = set()
    for noun in noun_order:
        labels = by_noun[noun]
        located = [l for l in labels if " - " in l]
        for label in (located or labels[:1]):
            if _norm_label(label) not in seen_labels:
                seen_labels.add(_norm_label(label))
                symptom_labels.append(label)
    fields_symptoms = "; ".join(symptom_labels) if symptom_labels else NOT_DOCUMENTED

    # 2. Denies / pertinent negatives — ONLY patient-or-exam documented
    # negatives. An examination absence ("no bony tenderness") belongs to the
    # exam section below; it is not a patient-reported denial.
    denies = sorted({_sentence(_symptom_noun(f)).rstrip(".")
                     for f in facts
                     if f["status"] == "ABSENT" and f["fact_type"] == "SYMPTOM"
                     and f.get("concept")
                     and not (f.get("attributes") or {}).get("negative_statement")})
    fields_denies = ", ".join(denies) if denies else NOT_DOCUMENTED

    # 3. Current treatment discussed — treatments actually received/tried.
    received = sorted({f["english"] for f in facts
                       if f["fact_type"] in ("TREATMENT", "PROCEDURE")
                       and f["status"] == "PRESENT"})
    fields_treatment = ", ".join(received) if received else NOT_DOCUMENTED

    # 4. Physical findings — the exam section verbatim from the canonical
    # renderer. "Not mentioned" while the note carries findings is the bug.
    fields_exam = "; ".join(_exam_lines(facts)) if _exam_lines(facts) else NOT_DOCUMENTED

    # 5. Impression — certainty preserved per diagnosis. Suspected stays
    # suspected; CONFIRMED says clinician-documented; never a bare "Possible"
    # list that flattens the graph's certainty model.
    impression_parts = list(_assessment_lines(facts))
    fields_impression = (" ".join(impression_parts)
                         if impression_parts else NOT_DOCUMENTED)

    # Diagnoses as structured rows: name + status, uncertainty kept.
    diagnoses = [{
        "name": f["english"],
        "status": (f.get("certainty") or "UNCERTAIN").lower(),
        "confirmed": f.get("certainty") == "CONFIRMED",
    } for f in facts if f["fact_type"] == "DIAGNOSIS"]

    # Investigations suggested (plan/investigation facts).
    investigations = [f["english"] for f in facts
                      if f["fact_type"] == "INVESTIGATION"
                      and f["section"] == "plan" and f["status"] == "PRESENT"]
    fields_investigations = ", ".join(investigations) if investigations else NOT_DOCUMENTED

    # 6. HPI temporal structure — the verbatim element sentences, grouped by
    # their clinical concept. A trigger is never a duration.
    def element_texts(kind_prefixes: tuple[str, ...]) -> list[str]:
        out: list[str] = []
        seen_ci: set[str] = set()
        for e in elements:
            if e["element"].startswith(kind_prefixes):
                text = (e.get("text") or e.get("source_text") or "").strip()
                key = text.lower().rstrip(".")
                if text and key not in seen_ci:
                    seen_ci.add(key)
                    out.append(text)
        return out

    onset = element_texts(("ONSET_MECHANISM", "ONSET_DATE"))
    course = element_texts(("TRAJECTORY:",))
    aggravating = [e.get("source_text") or e.get("text", "")
                   for e in elements if e["element"] == "AGGRAVATING"]
    relieving = [e.get("source_text") or e.get("text", "")
                 for e in elements if e["element"] == "RELIEVING"]
    no_benefit = [e.get("source_text") or e.get("text", "")
                  for e in elements if e["element"] == "TREATMENT_NO_BENEFIT"]
    benefit = [e.get("source_text") or e.get("text", "")
               for e in elements if e["element"] == "TREATMENT_BENEFIT"]

    fields_onset = "; ".join(onset) if onset else NOT_DOCUMENTED
    fields_course = "; ".join(course) if course else NOT_DOCUMENTED
    fields_aggravating = "; ".join(aggravating) if aggravating else NOT_DOCUMENTED
    fields_relieving = "; ".join(relieving) if relieving else NOT_DOCUMENTED
    response_parts = no_benefit + benefit
    fields_response = "; ".join(response_parts) if response_parts else NOT_DOCUMENTED

    # Documented durations from symptom attributes only (evidence-bearing).
    durations = sorted({(f.get("attributes") or {}).get("duration")
                        for f in symptoms
                        if (f.get("attributes") or {}).get("duration")})
    fields_duration = ", ".join(str(d) for d in durations) if durations else NOT_DOCUMENTED

    # 7-10. History sections — the canonical renderers.
    fields_pmh = "; ".join(_pmh_lines(facts)) if _pmh_lines(facts) else NOT_DOCUMENTED
    psh = [f["english"] for f in facts
           if f["fact_type"] == "HISTORY" and f["section"] == "psh"]
    fields_psh = ", ".join(psh) if psh else NOT_DOCUMENTED
    meds = sorted({f["english"] for f in facts
                   if f["fact_type"] == "MEDICATION"
                   and f["status"] in ("PRESENT", "RESOLVED", "HISTORICAL")})
    fields_medications = ", ".join(meds) if meds else []
    allergies = []
    for f in facts:
        if f["fact_type"] == "ALLERGY":
            allergies.append(f["english"])
    fields_allergies = ", ".join(allergies) if allergies else NOT_DOCUMENTED
    family = sorted({f["english"] for f in facts
                     if f.get("speaker") in ("mother", "father", "family")})
    fields_family = ", ".join(family) if family else NOT_DOCUMENTED
    social = sorted({f["english"] for f in facts
                     if f["fact_type"] == "SOCIAL_HISTORY"})
    fields_social = ", ".join(social) if social else NOT_DOCUMENTED
    ros_lines = []
    for f in facts:
        if f["fact_type"] == "SYMPTOM" and f["status"] == "QUESTIONED":
            ros_lines.append(_symptom_label(f))
    fields_ros = "; ".join(ros_lines) if ros_lines else NOT_DOCUMENTED

    # 14. Plan — considered stays considered, recommended stays recommended.
    fields_plan = "; ".join(_plan_lines(facts)) if _plan_lines(facts) else NOT_DOCUMENTED

    # 15. Referral context — recipient role from document metadata.
    recipient = (metadata.get("referral_recipient_role") or {}).get("value")
    addressee = (metadata.get("referral_addressee") or {}).get("value")
    referral_parts = []
    if recipient:
        referral_parts.append(f"Recipient role: {recipient}")
    if addressee:
        referral_parts.append(f"Addressed to: {addressee}")
    fields_referral = "; ".join(referral_parts) if referral_parts else NOT_DOCUMENTED

    # 16. Other documentation — administrative metadata verbatim (PHI stays in
    # the record; it is never logged by the caller).
    other = []
    for line in _metadata_lines(document):
        if not line.lower().startswith(("referral", "patient name", "clinician")):
            other.append(line)
    fields_other = "; ".join(other) if other else NOT_DOCUMENTED

    return {
        "symptoms_reported": fields_symptoms,
        "denies": fields_denies,
        "current_treatment_discussed": fields_treatment,
        "physical_findings": fields_exam,
        "impression": fields_impression,
        "diagnoses": diagnoses,
        "investigations": fields_investigations,
        "onset_trigger": fields_onset,
        "temporal_course": fields_course,
        "aggravating_factors": fields_aggravating,
        "relieving_factors": fields_relieving,
        "treatment_response": fields_response,
        "duration": fields_duration,
        "pmh": fields_pmh,
        "psh": fields_psh,
        "medications": fields_medications,
        "allergies": fields_allergies,
        "family_history": fields_family,
        "social_history": fields_social,
        "ros": fields_ros,
        "plan": fields_plan,
        "referral_context": fields_referral,
        "other_documentation": fields_other,
    }


def _prescription_fields(document: dict) -> dict:
    """Project plan/medication facts into the prescription-side fields.

    Medications appear only when the fact graph carries them. A considered
    procedure is not a medication and never appears here.
    """
    facts = document.get("facts") or []
    meds = []
    for f in facts:
        if f["fact_type"] != "MEDICATION":
            continue
        if f["status"] not in ("PRESENT", "RECOMMENDED", "RESOLVED",
                               "HISTORICAL", "CONSIDERED"):
            continue
        label = _symptom_noun(f)
        if _norm_label(label) not in {_norm_label(m) for m in meds}:
            meds.append(label)
    return {
        "medications": meds,
        "dosage_instructions": NOT_DOCUMENTED,
        "duration_of_treatment": NOT_DOCUMENTED,
        "follow_up": NOT_DOCUMENTED,
        "lifestyle_advice": NOT_DOCUMENTED,
    }


def warnings_from_facts(document: dict) -> list[str]:
    """Operational warnings derived from the fact graph, never from the
    legacy extractor's blind spots.

    A suspected diagnosis SATISFIES "diagnosis information present" while
    keeping its uncertainty explicit — "No diagnosis detected" for a note
    whose Assessment reads "Suspected: plantar fasciitis" was false and
    dangerous. Uncertainty is surfaced, never suppressed.
    """
    facts = document.get("facts") or []
    warnings: list[str] = []
    diagnoses = [f for f in facts if f["fact_type"] == "DIAGNOSIS"]
    if not diagnoses:
        warnings.append("No diagnosis documented in this consultation")
    elif not any(f.get("certainty") == "CONFIRMED" for f in diagnoses):
        warnings.append("Diagnosis documented as suspected only - "
                        "not confirmed in this consultation")
    if not any(f["fact_type"] == "SYMPTOM" for f in facts):
        warnings.append("No symptoms documented")
    return warnings


def project_structured_fields(document: dict) -> dict:
    """The single authoritative projection of the fact graph.

    Returns the two field dictionaries the Review UI edits and the export
    renders. Confidence is deliberately NOT a fabricated number: the fact
    graph carries per-fact confidence with real provenance, so field-level
    confidence is reported as unavailable rather than misleading.
    """
    note_fields = _clinical_note_fields(document)
    rx_fields = _prescription_fields(document)
    return {
        "clinical_note_fields": note_fields,
        "prescription_fields": rx_fields,
        "warnings": warnings_from_facts(document),
        # Per-fact confidence exists in the graph (evidence-derived); a single
        # per-field number would be invented. Explicitly unavailable.
        "confidence": None,
        "confidence_note": ("Field-level confidence unavailable; per-fact "
                            "confidence with evidence spans is in "
                            "clinical_facts_v2.facts."),
        "projected_from": "clinical_facts_v2",
    }
