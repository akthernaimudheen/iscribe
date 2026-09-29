# -*- coding: utf-8 -*-
"""Clinical note renderer V2 — renders the typed, evidence-grounded fact graph.

Consumes scribe_engine.clinical_facts.build_clinical_facts() output and emits
{note, sections, provenance, validation, document}. Nothing here reads the raw
transcript for clinical content: every sentence is either a deterministic
template over a fact or the VERBATIM source clause of an HPI element (an onset
mechanism, a trajectory, an aggravating factor), which is itself evidence.

Truth model (unchanged from v1, and never collapsed):

    PRESENT        -> "The patient reports ..."
    ONGOING        -> "The patient continues to have ..."
    RESOLVED       -> "... for <duration>, which has since resolved."
    IMPROVING      -> "... for <duration>, which then improved." (kept separate
                      from any residual current problem)
    INTERMITTENT   -> "... has intermittent ..."
    ABSENT         -> "The patient denies ..." / "No <finding>."
    UNCERTAIN      -> "... reports possible ... and is uncertain ..."
    CONSIDERED     -> plan item, explicitly not performed
    RECOMMENDED    -> plan/recommendation item
    NOT_DOCUMENTED -> absence of a fact ("Not documented.") — never a denial

Document types keep their own shape: a referral letter and a medico-legal
report add Prognosis / Referral Context / Occupational History and document
metadata instead of being forced into the 14-section outpatient template.
"""

from __future__ import annotations

import re

from .clinical_facts import (FACT_TYPES, OPTIONAL_SECTIONS, SECTION_TITLES,
                             SECTIONS, TEMPORAL_CONTEXTS)

NOT_DOCUMENTED = "NOT_DOCUMENTED"

# Renderer-owned template wording (not clinical content).
_TEMPLATE_PHRASES = (
    "clinician-documented diagnosis", "clinician considered this",
    "suspected (not confirmed)", "explicitly none reported",
    "considered, not performed", "- recommended", "explicitly documented",
    "explicitly stated", "document type:", "not documented.",
)

# Section order: the standard consultation first, document-specific sections
# after it, in the order a clinical report is read.
SECTION_ORDER = (
    "chief_complaint", "hpi", "associated_symptoms", "pertinent_negatives",
    "pmh", "psh", "medications", "allergies", "family_history", "social_history",
    "occupational_history", "ros", "physical_exam", "assessment", "plan",
    "prognosis", "referral_context", "other_documentation",
)

_HEADING_LIKE_SECTIONS = {"other_documentation", "referral_context"}


def _cap(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    return text[:1].upper() + text[1:]


def _sentence(text: str) -> str:
    text = (text or "").strip().rstrip(" ,;:")
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def _norm_label(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _duration_phrase(duration) -> str:
    """Render a stored duration verbatim (numerals 1-10 as words).

    Only unit-bearing durations ("3 weeks") become "for …" prose; a value
    without a duration unit ("a long time ago") renders as a bare
    parenthetical — the temporal anchor is preserved verbatim but never
    attached with "for", which is ungrammatical for non-duration anchors.
    """
    if not duration:
        return ""
    d = str(duration).strip()
    m = re.match(r"^(\d+)\s*(day|days|week|weeks|month|months|year|years|hour|hours)$",
                 d, re.I)
    if m:
        n, unit = int(m.group(1)), m.group(2).lower().rstrip("s")
        words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
                 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
        label = f"{words[n] if n in words else n} {unit}{'' if n == 1 else 's'}"
        return f"for {label}"
    if re.search(r"\b(day|days|week|weeks|month|months|year|years|hour|hours)\b",
                 d, re.I):
        return f"for {d}"
    if re.match(r"^(?:for|since)\b", d, re.I):
        return d
    if d.lower() == "a while":
        return "for a while"
    return d
# Generic symptom words that carry no anatomical information by themselves.
_GENERIC_WORDS = {"pain", "ache", "aches", "swelling", "soreness", "discomfort",
                  "numbness", "tingling", "tenderness"}


def _split_words(text: str | None) -> set[str]:
    return {w for w in re.split(r"[^a-z]+", (text or "").lower()) if w}


def _overlapping_site(noun: str, location: str | None) -> bool:
    """True when the concept already names the site ("chest pain" + "chest")."""
    if not location:
        return False
    return bool((_split_words(location) - _GENERIC_WORDS) & (_split_words(noun)))


def _subject_clause(location: str | None, noun: str = "") -> str:
    if not location:
        return ""
    if _split_words(noun) <= _split_words(location):
        return ""
    return f" in the {location}"


def _symptom_noun(fact: dict) -> str:
    return fact.get("english") or (fact.get("concept") or "").replace("_", " ")


def _join(names: list[str]) -> str:
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def _group_episodes(facts: list[dict]) -> list[dict]:
    """Group symptom/history facts into episodes by concept, in document order.

    A fact with no anatomical site joins the group of the same concept that has
    one, so "left medial foot arch pain" and a later "the pain has been
    fluctuating" are one episode. Facts of the same concept with *different*
    sites stay separate episodes: that is how resolved neck pain and residual
    arm pain remain two facts.
    """
    episodes: list[dict] = []
    by_concept: dict[str, list[dict]] = {}
    for f in facts:
        if f["fact_type"] not in ("SYMPTOM",):
            continue
        if f["status"] == "QUESTIONED":
            continue
        concept = f.get("concept") or _norm_label(f.get("english") or "")
        site = (f.get("attributes") or {}).get("location") or ""
        group = None
        for ep in by_concept.get(concept, []):
            if not site or not ep["location"] or _norm_label(site) == _norm_label(ep["location"]):
                group = ep
                break
        if group is None:
            group = {"concept": concept, "english": _symptom_noun(f),
                     "location": site, "facts": []}
            episodes.append(group)
            by_concept.setdefault(concept, []).append(group)
        if not group["location"] and site:
            group["location"] = site
        group["facts"].append(f)
    return episodes


# ---------------------------------------------------------------------------
# HPI
# ---------------------------------------------------------------------------
# Narrative lead-ins the dictation puts in front of a statement. Removing them
# is not paraphrase: the clinical content that follows is verbatim.
_LEAD_IN_RE = re.compile(
    r"^(?:and|but|so|then|also|however)\s+|"
    r"^(?:he|she|they|the patient|mr\.? \w+|mrs\.? \w+|ms\.? \w+)\s+"
    r"(?:finds?|reports?|says?|states?|tells me|describes?|notes?)\s+"
    r"(?:that\s+)?", re.I)


def _element_sentences(elements: list[dict], facts: list[dict] | None = None) -> list[dict]:
    """Verbatim HPI element sentences (chronology, trajectory, aggravators).

    The clause itself is the sentence: paraphrasing it would be interpretation.
    Treatment-response statements are excluded — they are rendered from the
    TREATMENT fact so the response is attributed to a named treatment.

    Stored conversational durations ("multiple times a week") captured by a
    fact whose clause the splitter cut mid-sentence are appended in
    parentheses, keeping the speaker's own temporal words on the rendered
    line — never converted into invented numbers.
    """
    out = []
    for e in elements:
        kind = e["element"]
        if kind in ("ONSET_MECHANISM", "INVESTIGATION_ABSENT",
                    "TREATMENT_NO_BENEFIT", "TREATMENT_BENEFIT"):
            continue
        text = e["source_text"]
        while True:
            stripped = _LEAD_IN_RE.sub("", text, count=1).strip()
            if stripped == text:
                break
            text = stripped
        text = _sentence(_cap(text))
        if not text:
            continue
        entry = {
            "text": text, "kind": kind,
            "source_span": e.get("source_span"),
            "source_text": e.get("source_text"),
            "verbatim": True,
        }
        if facts and entry["source_span"]:
            span = entry["source_span"]
            extra: list[str] = []
            for f in facts:
                fattrs = f.get("attributes") or {}
                # Both temporal attribute kinds render verbatim: durations
                # ("several months") and rate expressions ("multiple times a
                # week") are the speaker's own words; a rate must never land
                # on the duration slot, but wherever the value lives it is
                # preserved on the verbatim line.
                temporal_vals = [fattrs.get("duration"), fattrs.get("frequency")]
                fspan = f.get("source_span")
                if not any(temporal_vals) or not fspan:
                    continue
                if fspan[0] >= span[0] and fspan[1] <= span[1]:
                    for val in temporal_vals:
                        if not val:
                            continue
                        for part in str(val).split(";"):
                            part = part.strip()
                            if part and part.lower() not in entry["text"].lower() \
                                    and part not in extra:
                                extra.append(part)
            if extra:
                entry["text"] = entry["text"].rstrip(".") + " (" + "; ".join(extra) + ")."
        out.append(entry)
    return out


def primary_status(episode: dict) -> str:
    order = {"RESOLVED": 0, "ABSENT": 1, "PRESENT": 2, "UNCERTAIN": 3}
    return sorted(episode["facts"],
                  key=lambda f: order.get(f["status"], 4))[0]["status"]


def _episode_sentence(episode: dict, mechanism: str | None) -> str | None:
    """One narrative sentence for an episode, from its most informative fact."""
    facts = episode["facts"]
    if not facts:
        return None
    noun = episode["english"]
    location = episode.get("location")
    # "back pain" reported as "pain in the neck and back": the location is the
    # more complete statement, so the generic word carries the sentence.
    # For a denial, the noun alone already names the site.
    if _overlapping_site(noun, location):
        generic = next((w for w in _split_words(noun) if w in _GENERIC_WORDS), None)
        if primary_status(episode) in ("ABSENT", "RESOLVED") or not generic:
            location = None
        else:
            noun = generic
    site = _subject_clause(location, noun)
    # The most informative fact: RESOLVED > the one with a duration/trajectory.
    order = {"RESOLVED": 0, "ABSENT": 1, "PRESENT": 2, "UNCERTAIN": 3}
    primary = sorted(facts, key=lambda f: (
        order.get(f["status"], 4),
        -len((f.get("attributes") or {})) if f["status"] != "ABSENT" else 0,
    ))[0]
    attrs = primary.get("attributes") or {}
    duration = _duration_phrase(attrs.get("duration")) if attrs.get("duration") else ""
    # Verbatim rate expressions ("on a regular basis; multiple times a week")
    # render exactly where a duration would: the speaker's own temporal words,
    # never converted into invented numbers, and never prefixed with "for"
    # (a rate is not a duration).
    freq = _norm_label(attrs.get("frequency")) if attrs.get("frequency") else ""
    if freq and freq not in duration:
        duration = f"{duration}; {freq}" if duration else freq
    severity = attrs.get("severity")
    temporal = primary.get("temporal_context")

    if primary["status"] == "RESOLVED":
        if duration:
            return (f"The patient reports a history of {noun}{site} {duration}, "
                    f"which has since resolved.")
        return f"The patient previously experienced {noun}{site}, which has since resolved."
    if primary["status"] == "ABSENT":
        if primary.get("attributes", {}).get("negative_statement"):
            return None                    # rendered as an explicit negative
        return f"The patient denies {noun}{site}."
    if primary["status"] == "UNCERTAIN":
        return (f"The patient reports possible {noun}{site} and is uncertain "
                f"whether it is present.")
    if temporal == "ONGOING":
        # A stored conversational duration ("on a regular basis; multiple
        # times a week") is rendered verbatim in parentheses — the speaker's
        # own temporal words, never converted into invented numbers.
        if duration:
            return (f"The patient continues to have {noun}{site} "
                    f"({duration}).")
        return f"The patient continues to have {noun}{site}."
    if temporal in ("HISTORICAL", "IMPROVING") and duration:
        return (f"The patient reports {noun}{site} {duration}, which then improved.")
    if temporal == "INTERMITTENT":
        return (f"The patient reports intermittent {noun}{site}"
                + (f" {_duration_phrase(attrs['duration'])}" if duration else "") + ".")
    text = f"The patient reports {severity + ' ' if severity else ''}{noun}{site}"
    if duration:
        text += f" {duration}"
    if mechanism:
        text += f", which began after {mechanism}"
    return text + "."


def _negative_statement_text(fact: dict) -> str:
    """An explicit absence as a sentence ("no immediate symptoms")."""
    phrase = _symptom_noun(fact)
    if not phrase.lower().startswith(("no ", "nil ", "none")):
        phrase = f"no {phrase}"
    return _sentence(_cap(phrase))


def _treatment_sentences(facts: list[dict], elements: list[dict]) -> list[dict]:
    """Treatment history and response — never a symptom, never a medication."""
    out = []
    for f in facts:
        if f["fact_type"] not in ("TREATMENT", "PROCEDURE"):
            continue
        attrs = f.get("attributes") or {}
        response = attrs.get("treatment_response")
        if f["status"] == "PRESENT" and response == "no benefit":
            out.append({"text": f"No benefit was obtained from {f['english']}.",
                        "kind": "TREATMENT_NO_BENEFIT",
                        "source_span": f.get("source_span"),
                        "verbatim": False})
        elif f["status"] == "PRESENT" and response == "helpful":
            out.append({"text": f"{_cap(f['english'])} was reported to be helpful.",
                        "kind": "TREATMENT_BENEFIT",
                        "source_span": f.get("source_span"),
                        "verbatim": False})
        elif f["status"] == "PRESENT":
            out.append({"text": f"Previous treatment: {f['english']}.",
                        "kind": "TREATMENT_HISTORY",
                        "source_span": f.get("source_span"),
                        "verbatim": False})
    return out


def _hpi_lines(document: dict) -> list[dict]:
    facts = document["facts"]
    elements = document.get("elements") or []
    mechanism_by_concept = {}
    for e in elements:
        if e["element"] == "ONSET_MECHANISM" and e.get("related_concept"):
            mechanism_by_concept.setdefault(e["related_concept"], e["text"])

    lines: list[dict] = []
    # Facts whose clause is already rendered as a verbatim element sentence are
    # not repeated as templates.
    verbatim_clauses = {e.get("source_text") for e in elements
                        if e["element"] in ("AGGRAVATING", "RELIEVING", "ONSET_DATE")
                        or e["element"].startswith("TRAJECTORY:")}
    for episode in _group_episodes(facts):
        if episode["facts"] and all(
                f.get("source_text") in verbatim_clauses for f in episode["facts"]):
            continue
        sentence = _episode_sentence(episode,
                                     mechanism_by_concept.get(episode["concept"]))
        if sentence:
            primary = sorted(episode["facts"], key=lambda f: (
                {"RESOLVED": 0, "ABSENT": 1, "PRESENT": 2, "UNCERTAIN": 3}
                .get(f["status"], 4)))[0]
            attrs = primary.get("attributes") or {}
            # A plain "reports <symptom>" statement with nothing else attached
            # may be grouped with its neighbours (clinical prose, not a list).
            plain = (primary["status"] == "PRESENT"
                     and not episode.get("location")
                     and not attrs.get("duration") and not attrs.get("severity")
                     and not mechanism_by_concept.get(episode["concept"])
                     and primary.get("temporal_context") == "CURRENT"
                     and sentence == f"The patient reports {episode['english']}.")
            lines.append({"text": sentence, "kind": "EPISODE",
                          "source_span": episode["facts"][0].get("source_span"),
                          "verbatim": False,
                          "plain_noun": episode["english"] if plain else None})
    lines.extend(_element_sentences(elements, facts))
    lines.extend(_treatment_sentences(facts, elements))
    for f in facts:
        if f.get("attributes", {}).get("negative_statement"):
            # "No immediate symptoms" style negatives are explicit negative
            # STATEMENTS; they are rendered once, in Pertinent Negatives —
            # never duplicated into the current HPI list, and never treated as
            # a current symptom (their evidence is the dictation's own words).
            continue
        elif f["fact_type"] == "INVESTIGATION" and f["status"] == "ABSENT":
            lines.append({"text": _sentence(
                f"No {f['english']} performed (explicitly documented)"),
                "kind": "INVESTIGATION", "source_span": f.get("source_span"),
                "verbatim": False})
    # Chronological order: the document order of the underlying evidence.
    lines.sort(key=lambda x: (x.get("source_span") or [10 ** 9])[0])
    # A dictation fragment that continues the previous statement ("..., which
    # then improved.") belongs to that sentence, not to its own line.
    merged: list[dict] = []
    for line in lines:
        if merged and line.get("verbatim") and re.match(
                r"^(?:which|that|and|as|where|when|whose|but)\b", line["text"], re.I):
            prev_end = (merged[-1].get("source_span") or [None, None])[1]
            start = (line.get("source_span") or [None])[0]
            if prev_end is not None and start is not None and 0 <= start - prev_end < 60:
                merged[-1]["text"] = merged[-1]["text"].rstrip(".") + " " + \
                    line["text"][:1].lower() + line["text"][1:]
                merged[-1]["source_span"] = [merged[-1]["source_span"][0],
                                             line["source_span"][1]]
                continue
            if re.match(r"^(?:which|that|whose)\b", line["text"], re.I):
                # An orphan relative clause repeats a statement already made by
                # the sentence it belonged to; it is never a clinical statement
                # on its own.
                continue
        merged.append(line)
    # Consecutive plain present-reports read as one clinical sentence.
    grouped: list[dict] = []
    for line in merged:
        prev = grouped[-1] if grouped else None
        if (prev and prev.get("plain_noun") and line.get("plain_noun")):
            prev["nouns"] = (prev.get("nouns") or [prev["plain_noun"]]) + \
                [line["plain_noun"]]
            prev["text"] = f"The patient reports {_join(prev['nouns'])}."
            prev["plain_noun"] = None
            continue
        grouped.append(dict(line))
    for line in grouped:
        line.pop("nouns", None)
        line.pop("plain_noun", None)
    seen, unique = set(), []
    for line in grouped:
        key = _norm_label(line["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)
    return unique


# ---------------------------------------------------------------------------
# Section rendering
# ---------------------------------------------------------------------------
def _ros_line(facts: list[dict]) -> list[str]:
    from .clinical_documentation import _ROS_LABELS, _ros_group_for
    groups: dict[str, list[str]] = {}
    for f in facts:
        if f["fact_type"] not in ("SYMPTOM", "HISTORY"):
            continue
        if f["status"] == "QUESTIONED":
            continue                       # a question is not a finding
        g = _ros_group_for(f.get("concept") or "")
        if not g:
            continue
        noun = _symptom_noun(f)
        if f["status"] == "ABSENT":
            entry = f"{noun}, denied"
        elif f["status"] == "RESOLVED":
            entry = f"{noun}, resolved"
        elif f["status"] == "UNCERTAIN":
            entry = f"possible {noun}, uncertain"
        else:
            entry = noun
        groups.setdefault(g, []).append(entry)
    order = list(_ROS_LABELS)
    out = []
    for g in sorted(groups, key=lambda x: order.index(x)):
        out.append(f"{_ROS_LABELS[g]}: " + "; ".join(dict.fromkeys(groups[g])))
    return out


def _exam_lines(facts: list[dict]) -> list[str]:
    # Provenance equivalence first (on structured records), then the wording
    # squash pass on the surviving lines.
    records: list[dict] = []
    for f in facts:
        if f["fact_type"] != "EXAM_FINDING":
            continue
        location = (f.get("attributes") or {}).get("location")
        name = f["english"]
        if f["status"] == "ABSENT":
            line = _sentence(_cap(name))
        elif location and location.lower() not in name.lower():
            line = _sentence(f"{location.capitalize()}: {name}")
        else:
            line = _sentence(_cap(name))
        core = _exam_core(name)
        status = f["status"]
        quals = _exam_quals(name, location, core)
        site = _norm_label(location or "")
        covered = False
        if core:
            for i, r in enumerate(records):
                if r["core"] != core or r["status"] != status:
                    continue
                rs, ss = r["site"], site
                same_scope = rs == ss or \
                    (rs and ss and (rs in ss or ss in rs)) or \
                    (r["quals"] <= quals or quals <= r["quals"])
                if same_scope:
                    # The BROADER statement survives: it covers the narrower
                    # one, so nothing the transcript documented is lost.
                    if rs and ss and rs in ss:
                        records[i] = {"line": line, "core": core,
                                      "status": status, "site": site,
                                      "quals": quals}
                    covered = True
                    break
        if covered:
            continue
        records.append({"line": line, "core": core, "status": status,
                        "site": site, "quals": quals})
    out: list[str] = []
    for r in records:
        line = r["line"]
        # "Normal examination." and "Limbs normal on examination." describe one
        # finding: keep the more specific line only.
        squashed = _squash(line)
        covered = False
        for existing in list(out):
            existing_squashed = _squash(existing)
            if squashed and squashed in existing_squashed:
                covered = True
                break
            if existing_squashed and existing_squashed in squashed:
                out.remove(existing)
        if not covered:
            out.append(line)
    return out


def _section_bullets(note: str, title: str) -> list[str]:
    """Bullet lines under every occurrence of one numbered note heading.

    Every occurrence, not just the first: a note that repeats a heading (as
    tampering does) must not be able to hide content from the validator.
    """
    heading = re.compile(rf"^\s*\d+\.\s+{re.escape(title)}\s*$", re.I)
    any_heading = re.compile(r"^\s*\d+\.\s+\S")
    out: list[str] = []
    inside = False
    for line in note.splitlines():
        if heading.match(line):
            inside = True
            continue
        if any_heading.match(line):
            inside = False
            continue
        if inside and line.strip().startswith("-"):
            out.append(line.strip()[1:].strip())
    return out


def _squash(text: str) -> str:
    """Content words of a sentence, for near-duplicate comparison."""
    words = [w for w in re.split(r"[^a-z0-9]+", (text or "").lower()) if w]
    stop = {"on", "of", "the", "is", "are", "was", "were", "and", "a", "an",
            "in", "to", "with", "at", "for"}
    return " ".join(w for w in words if w not in stop)


# ---------------------------------------------------------------------------
# Examination-finding equivalence (provenance, not string matching)
# ---------------------------------------------------------------------------
# The same clinical finding is often dictated twice in different words
# ("a full range of movement" / "movement ... is full"). Two exam facts are
# the SAME finding when their normalized finding CORE matches and one
# statement's anatomical scope contains the other's. Distinct findings —
# different core, or disjoint sites/qualifiers — always render.
_EXAM_CORE_RES = (
    # finding core -> the words the core consumes from the name/location
    (re.compile(r"\branges?\s+of\s+movement", re.I), "movement",
     {"range", "movement", "spinal", "full", "complete", "restricted",
      "reduced", "limited", "lost"}),
    (re.compile(r"\bmovement", re.I), "movement",
     {"range", "movement", "spinal", "full", "complete", "restricted",
      "reduced", "limited", "lost"}),
    (re.compile(r"\btenderness|\btender\b", re.I), "tenderness",
     {"tenderness", "tender"}),
    (re.compile(r"\bmarks?\b|\bscars?\b|\bbruises\b", re.I), "marks",
     {"marks", "scars", "bruises"}),
)


def _exam_core(name: str) -> str | None:
    """Normalized finding core of an examination finding (None = no core)."""
    low = (name or "").lower()
    for rx, label, _consumed in _EXAM_CORE_RES:
        if rx.search(low):
            if label == "movement":
                if re.search(r"\bfull\b|\bcomplete\b", low):
                    return "movement:full"
                if re.search(r"\brestricted\b|\breduced\b|\blimited\b|\blost\b", low):
                    return "movement:restricted"
            return label
    return None


def _exam_quals(name: str, location: str | None, core: str | None) -> set[str]:
    """Qualifying words of an exam finding (site + modifiers not consumed by
    the core): 'spinal' vs 'muscular' tenderness stay distinct."""
    words = _split_words(name) | _split_words(location)
    consumed = set()
    for rx, _label, used in _EXAM_CORE_RES:
        if rx.search(name or ""):
            consumed |= used
            break
    return words - consumed - {"no", "not", "without", "of", "the", "a", "an",
                               "and", "or", "is", "are", "was", "were", "on",
                               "in", "there", "his", "her", "their", "all",
                               "some", "other"}


def _assessment_lines(facts: list[dict]) -> list[str]:
    out = []
    for f in facts:
        if f["fact_type"] != "DIAGNOSIS":
            continue
        name = f["english"]
        if f["certainty"] == "CONFIRMED":
            out.append(_sentence(f"{_cap(name)} (clinician-documented diagnosis)"))
        else:
            out.append(_sentence(
                f"Suspected (not confirmed): {name} - clinician considered this"))
    return out


def _plan_lines(facts: list[dict]) -> list[str]:
    out = []
    for f in facts:
        attrs = f.get("attributes") or {}
        if f["fact_type"] == "PLAN" and f["status"] == "ABSENT":
            out.append(_sentence(f"{_cap(f['english'])} (explicitly stated)"))
        elif f["fact_type"] == "RECOMMENDATION" and \
                f["status"] == "RECOMMENDED" and \
                any(o is not f and o["fact_type"] in ("TREATMENT", "PROCEDURE")
                    and o["status"] == "RECOMMENDED"
                    and o["source_span"] and f["source_span"]
                    and not (o["source_span"][1] <= f["source_span"][0]
                             or f["source_span"][1] <= o["source_span"][0])
                    for o in facts):
            # The same clause already yielded a specific recommended item
            # ("I would be grateful if you could refer him to a consultant
            # orthopaedic surgeon") — the generic restatement ("specialist
            # opinion / expert review requested") adds nothing to the Plan.
            continue
        elif f["fact_type"] in ("TREATMENT", "PROCEDURE", "RECOMMENDATION"):
            if f["status"] == "CONSIDERED":
                out.append(_sentence(
                    f"{_cap(f['english'])} - considered, not performed"))
            elif f["status"] == "RECOMMENDED":
                out.append(_sentence(f"{_cap(f['english'])} - recommended"))
        elif f["fact_type"] == "INVESTIGATION" and f["status"] == "PRESENT" \
                and f["section"] == "plan":
            out.append(_sentence(f"Investigation: {f['english']}"))
        elif f["fact_type"] == "MEDICATION" and f["status"] == "RECOMMENDED":
            mattrs = f.get("attributes") or {}
            mparts = [f["english"]]
            if mattrs.get("dose_value") is not None:
                mparts.append(f"{mattrs['dose_value']:g} "
                              f"{mattrs.get('dose_unit') or ''}".strip())
            if mattrs.get("frequency_code"):
                mparts.append(str(mattrs["frequency_code"]))
            elif mattrs.get("frequency_standard"):
                mparts.append(str(mattrs["frequency_standard"]))
            elif mattrs.get("frequency_raw"):
                mparts.append(str(mattrs["frequency_raw"]))
            if mattrs.get("timing_relation"):
                mparts.append(str(mattrs["timing_relation"]).replace("_", " "))
            out.append(_sentence(
                f"{_cap(' - '.join(mparts))} - prescribed (medication)")
                if mattrs.get("medication_status") == "PRESCRIBED"
                else _sentence(f"{_cap(' - '.join(mparts))} - recommended"))
    return out


def _pmh_lines(facts: list[dict]) -> list[str]:
    out = []
    for f in facts:
        if f["fact_type"] != "HISTORY":
            continue
        name = f["english"]
        if f["status"] == "ABSENT":
            out.append(_sentence(f"{_cap(name)} - explicitly none reported"))
        elif f["status"] == "RESOLVED":
            out.append(_sentence(f"{_cap(name)}: previous history, resolved"))
        else:
            out.append(_sentence(f"{_cap(name)}: present"))
    return out


def _prognosis_lines(facts: list[dict]) -> list[str]:
    out = []
    for f in facts:
        if f["fact_type"] == "PROGNOSIS":
            out.append(_sentence(_cap(f["english"])))
    return out


def _metadata_lines(document: dict) -> list[str]:
    out = []
    for key, entry in (document.get("metadata") or {}).items():
        value = entry.get("value")
        if isinstance(value, list):
            value = ", ".join(value)
        if not value:
            continue
        label = {"spelled_names": "Names as spelled"}.get(
            key, key.replace("_", " ").capitalize())
        out.append(f"{label}: {value}")
    return out


def build_sections(document: dict) -> dict[str, list[dict]]:
    facts = document["facts"]
    doc_type = document["document"]["document_type"]
    sections: dict[str, list[dict]] = {}

    def add(section: str, text: str, evidence: str = "EXPLICIT_PRESENT",
            fact: dict | None = None, source_span=None, verbatim: bool = False):
        if not text:
            return
        sections.setdefault(section, []).append({
            "text": text, "evidence": evidence,
            "fact_id": (fact or {}).get("fact_id"),
            "source_span": source_span or (fact or {}).get("source_span"),
            "verbatim": verbatim,
        })

    # 1. Chief complaint — the first reported symptom, never an inference.
    primary = [f for f in facts if f["fact_type"] == "SYMPTOM"
               and f["status"] in ("PRESENT", "RESOLVED")
               and f.get("concept")]
    if primary:
        cc = _symptom_noun(primary[0])
        location = (primary[0].get("attributes") or {}).get("location")
        body = _subject_clause(location, cc)
        add("chief_complaint",
            _cap(_sentence(f"{cc}{body}").rstrip(".")) if body else _cap(cc),
            "EXPLICIT_PRESENT", primary[0])

    # 2. HPI — chronology, trajectory, aggravators, treatment response.
    for line in _hpi_lines(document):
        add("hpi", line["text"],
            "EXPLICIT_ABSENT" if line["kind"] == "NEGATIVE" else "EXPLICIT_PRESENT",
            None, line.get("source_span"), line.get("verbatim", False))

    # 3. Associated symptoms — grouped by organ system (never causal), and
    #    never a restatement of the chief complaint itself.
    chief_concept = primary[0].get("concept") if primary else None
    present = [f for f in facts if f["fact_type"] == "SYMPTOM"
               and f["status"] == "PRESENT" and f.get("concept")
               and f.get("concept") != chief_concept]
    if present:
        from .clinical_documentation import _ros_group_for
        groups: dict[str, list[str]] = {}
        other: list[str] = []
        for f in present:
            g = _ros_group_for(f.get("concept") or "")
            if g:
                groups.setdefault(g, []).append(_symptom_noun(f))
            else:
                other.append(_symptom_noun(f))
        labels = {"constitutional": "Constitutional", "respiratory": "Respiratory",
                  "heent": "HEENT", "cardiovascular": "Cardiovascular",
                  "gastrointestinal": "Gastrointestinal",
                  "neurological": "Neurologic", "musculoskeletal": "Musculoskeletal",
                  "skin": "Skin", "genitourinary": "Genitourinary"}
        for g, names in groups.items():
            add("associated_symptoms", f"{labels.get(g, g.capitalize())}: "
                + ", ".join(dict.fromkeys(names)))
        if other:
            add("associated_symptoms", "Other: " + ", ".join(sorted(set(other))))

    # 4. Pertinent negatives — every explicit negative statement is rendered
    #    exactly once (a dictation that repeats a denial gets one line).
    seen_negatives: set[str] = set()
    for f in facts:
        if f["status"] != "ABSENT":
            continue
        if f["fact_type"] == "SYMPTOM" and f.get("concept") \
                and not f.get("attributes", {}).get("negative_statement"):
            add("pertinent_negatives", _sentence(f"Denies {_symptom_noun(f)}"),
                "EXPLICIT_ABSENT", f)
        elif f.get("attributes", {}).get("negative_statement"):
            # An absence attributed to a named event ("no long-term consequence
            # of the accident") is PROGNOSIS content, not a pertinent negative.
            if f["fact_type"] == "PROGNOSIS" or f["section"] == "prognosis":
                continue
            text = _negative_statement_text(f)
            key = _norm_label(text)
            if key in seen_negatives:
                continue
            seen_negatives.add(key)
            add("pertinent_negatives", text, "EXPLICIT_ABSENT", f)

    # 5-10. History sections
    seen_work_absence: set[str] = set()
    for line in _pmh_lines(facts):
        evidence = "EXPLICIT_ABSENT" if "explicitly none" in line else "EXPLICIT_PRESENT"
        add("pmh", line, evidence)
    for f in facts:
        if f["fact_type"] == "HISTORY" and f["section"] == "psh":
            add("psh", _sentence(_cap(f["english"])), "EXPLICIT_ABSENT", f)
    for f in facts:
        if f["fact_type"] == "MEDICATION" and f["status"] in ("PRESENT", "RESOLVED",
                                                              "HISTORICAL"):
            attrs = f.get("attributes") or {}
            bits = [f["english"]]
            if attrs.get("medication_status") == "DISCONTINUED":
                bits.append("stopped (discontinued)")
            elif f["status"] == "HISTORICAL":
                bits.append("previously taken (historical)")
            if attrs.get("dose_value") is not None:
                dose_bit = (f"{attrs['dose_value']:g} "
                            f"{attrs.get('dose_unit') or ''}".strip())
                if attrs.get("dose_form"):
                    dose_bit += f" {attrs['dose_form']}"
                bits.append(dose_bit)
            if attrs.get("frequency_code"):
                bits.append(str(attrs["frequency_code"]))
            elif attrs.get("frequency_standard"):
                bits.append(str(attrs["frequency_standard"]))
            elif attrs.get("frequency_raw"):
                bits.append(str(attrs["frequency_raw"]))
            if attrs.get("timing_relation"):
                bits.append(str(attrs["timing_relation"]).replace("_", " "))
            if attrs.get("duration_value") is not None:
                bits.append(f"for {attrs['duration_value']:g} "
                            f"{attrs.get('duration_unit') or ''}".strip())
            elif attrs.get("duration"):
                bits.append(f"duration {attrs['duration']}")
            add("medications", "; ".join(bits), "EXPLICIT_PRESENT", f)
        elif f["fact_type"] == "MEDICATION" and f["status"] == "QUESTIONED":
            # A medication mentioned in a question is never an active med:
            # it renders where the question was asked (research §B.2).
            add("other_documentation",
                f"medication question: {f['english']} (asked, not asserted)",
                "QUESTIONED", f)
        elif f["fact_type"] == "MEDICATION" and f["status"] == "UNCERTAIN":
            add("medications",
                f"{f['english']} (unverified: name/details uncertain)",
                "UNCERTAIN", f)
        elif f["fact_type"] == "ALLERGY":
            # FP-7: a family member's allergy is family history, never the
            # patient's allergy list.
            if (f.get("attributes") or {}).get("family_subject") \
                    or f.get("section") == "family_history":
                add("family_history",
                    _sentence(f"Allergy to {f['english']} in a family member "
                              f"(family history)"), "EXPLICIT_PRESENT", f)
            elif f["status"] == "QUESTIONED":
                # Asked, never answered: the question renders where it was
                # asked, and the allergy list stays untouched (the projected
                # fields exclude it for the same reason).
                add("other_documentation",
                    f"allergy question: {f['english']} (asked, not asserted)",
                    "QUESTIONED", f)
            elif f["status"] == "ABSENT":
                add("allergies", "No known allergies (explicitly denied).",
                    "EXPLICIT_ABSENT", f)
            elif f["status"] in ("PRESENT", "RESOLVED", "HISTORICAL",
                                 "UNCERTAIN"):
                # Name the allergen: "Allergy reported." next to a documented
                # allergen hid the one thing the line exists to say.
                add("allergies", _sentence(f"Allergy to {f['english']} reported"),
                    "EXPLICIT_PRESENT", f)
        elif f["fact_type"] == "SYMPTOM" and f["section"] == "family_history":
            add("family_history", _sentence(
                f"{_cap(_symptom_noun(f))} in a family member (family history)"),
                "EXPLICIT_PRESENT", f)
    for f in facts:
        if f["speaker"] in ("mother", "father", "family"):
            add("family_history", _sentence(
                f"{_cap(_symptom_noun(f))} in {f['speaker']} (family history)"),
                "EXPLICIT_PRESENT", f)
        elif f["fact_type"] == "SOCIAL_HISTORY":
            add("social_history", _sentence(
                f"{_cap(f['english'])}: documented"), "EXPLICIT_PRESENT", f)
        elif f["fact_type"] == "OCCUPATIONAL_HISTORY":
            # Dictation often restates the same work fact ("He is a welder and
            # had 2 weeks off work following his accident" / "The accident
            # caused him to have 2 weeks off work"). Facts carrying the same
            # work-absence value are ONE evidenced fact rendered once — the
            # first statement (which usually carries the occupation itself).
            occ_key = ((f.get("attributes") or {}).get("work_absence") or "")
            occ_key = _norm_label(occ_key)
            if occ_key and occ_key in seen_work_absence:
                continue
            if occ_key:
                seen_work_absence.add(occ_key)
            add("occupational_history", _sentence(_cap(f["source_text"])),
                "EXPLICIT_PRESENT", f)

    # 11. ROS
    for line in _ros_line(facts):
        add("ros", line)

    # 12. Physical examination
    for line in _exam_lines(facts):
        add("physical_exam", line)

    # 13. Assessment
    for line in _assessment_lines(facts):
        add("assessment", line, "EXPLICIT_PRESENT")

    # 14. Plan
    for line in _plan_lines(facts):
        add("plan", line, "EXPLICIT_PRESENT")

    # 15-18. Document-specific sections
    for line in _prognosis_lines(facts):
        add("prognosis", line, "EXPLICIT_PRESENT")
    if doc_type in ("REFERRAL_LETTER", "MEDICOLEGAL_REPORT", "MEDICAL_REPORT"):
        meta = _metadata_lines(document)
        for line in meta:
            if line.lower().startswith(("referral", "patient name", "clinician")):
                add("referral_context", line)
            else:
                add("other_documentation", line)
    return sections


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render_clinical_note(document: dict) -> dict:
    """Render the note text, provenance trace and fail-closed validation."""
    facts = document.get("facts") or []
    doc_type = (document.get("document") or {}).get("document_type", "OTHER")
    sections = build_sections(document)

    header = ["CLINICAL NOTE"]
    document_label = doc_type.replace("_", " ").title()
    header.append(f"Document type: {document_label}"
                  + (" (classification uncertain)"
                     if (document.get("document") or {}).get("uncertain") else ""))
    header.append("")

    lines = list(header)
    index = 0
    for key in SECTION_ORDER:
        entries = sections.get(key) or []
        if not entries and key in OPTIONAL_SECTIONS:
            continue
        index += 1
        lines.append(f"{index}. {SECTION_TITLES[key]}")
        if entries:
            for e in entries:
                prefix = "- " if e["evidence"] != NOT_DOCUMENTED else ""
                lines.append(f"   {prefix}{e['text']}")
        else:
            lines.append("   Not documented.")
        lines.append("")
    note = "\n".join(lines).rstrip() + "\n"

    provenance = []
    for section, entries in sections.items():
        for e in entries:
            provenance.append({
                "section": section,
                "fact_id": e.get("fact_id"),
                "rendered": e["text"],
                "evidence": e["evidence"],
                "source_span": e.get("source_span"),
                "verbatim": bool(e.get("verbatim")),
            })

    validation = validate_note_v2(note, document)
    return {
        "note": note,
        "sections": sections,
        "provenance": provenance,
        "validation": validation,
        "document_type": doc_type,
        "renderer": "deterministic-v2",
    }


# ---------------------------------------------------------------------------
# Note validator (fail closed)
# ---------------------------------------------------------------------------
def validate_note_v2(note: str, document: dict) -> dict:
    """Every clinical claim in the note must be backed by a fact.

    Detects: unsupported medication, diagnosis, treatment, investigation, PMH,
    exam finding, duration, severity, denial, greeting or administrative text.
    """
    from .clinical_facts import (_DIAGNOSIS_SUFFIX_RE, _DIAGNOSIS_WORD_RE,
                                 _clean_diagnosis_term, _usable_diagnosis_term,
                                 validate_clinical_facts)

    violations: list[str] = []
    facts = document.get("facts") or []
    # Facts are evidenced against the text they were read from: the raw ASR
    # transcript, or its verified-corrected copy when one exists.
    raw_text = (document.get("evidence_text") or document.get("raw_text") or "")
    # Template wording the renderer itself contributes (never clinical content);
    # removing it keeps the clinical-claim scan precise.
    scan = note
    for phrase in _TEMPLATE_PHRASES:
        scan = re.sub(re.escape(phrase), "", scan, flags=re.I)

    fact_check = validate_clinical_facts(raw_text, facts)
    if not fact_check["valid"]:
        violations.extend(f"fact graph invalid: {v}" for v in fact_check["violations"])

    known_labels = set()
    for f in facts:
        for label in (f.get("english"), f.get("concept"), f.get("evidence_text")):
            if label:
                known_labels.add(_norm_label(label))
    diagnoses = {_norm_label(f["english"]) for f in facts
                 if f["fact_type"] == "DIAGNOSIS"}
    medications = {_norm_label(f["english"]) for f in facts
                   if f["fact_type"] == "MEDICATION"}
    durations = {_norm_label(str((f.get("attributes") or {}).get("duration")))
                 for f in facts if (f.get("attributes") or {}).get("duration")}
    severities = {_norm_label(str((f.get("attributes") or {}).get("severity")))
                  for f in facts if (f.get("attributes") or {}).get("severity")}
    denied = {_norm_label(f.get("english") or "") for f in facts
              if f["status"] == "ABSENT"}
    verbatim_text = "\n".join(sorted({
        _norm_label(e["source_text"]) for e in (document.get("elements") or [])
        if e.get("source_text")} | {
        _norm_label(f["source_text"]) for f in facts if f.get("source_text")}))

    # 1. Diagnoses in the note must come from a diagnosis fact.
    for pattern in (_DIAGNOSIS_SUFFIX_RE, _DIAGNOSIS_WORD_RE):
        for m in pattern.finditer(scan):
            if not _usable_diagnosis_term(m.group(1)):
                continue
            phrase = _norm_label(_clean_diagnosis_term(m.group(1)))
            if not phrase:
                continue
            if any(phrase in d or d in phrase for d in diagnoses):
                continue
            violations.append(f"diagnosis '{m.group(1)}' in note not supported by "
                              f"the fact graph")

    # 1b. Every Assessment line must be a diagnosis the graph actually holds.
    #     Rule 1 knows diagnosis *shapes*; a bare clinical noun ("Pneumonia.",
    #     which matches no morphology) is only admissible through a fact.
    diagnosis_words = set()
    for f in facts:
        if f["fact_type"] == "DIAGNOSIS":
            for label in (f.get("english"), f.get("concept")):
                diagnosis_words |= set(_squash(str(label or "")).split())
    for line in _section_bullets(note, SECTION_TITLES["assessment"]):
        body = re.sub(r"\([^)]*\)", " ", line)
        for phrase in _TEMPLATE_PHRASES:
            body = re.sub(re.escape(phrase), "", body, flags=re.I)
        words = set(_squash(body).split())
        if words and not words & diagnosis_words:
            violations.append(f"assessment line '{line}' is not supported by a "
                              f"diagnosis fact")

    # 1c. Occupational history lines are verbatim transcript statements: the
    # renderer emits f["source_text"], so any content word the transcript did
    # not contain is an injected claim.
    verbatim_words = set(_squash(verbatim_text).split())
    for line in _section_bullets(note, SECTION_TITLES["occupational_history"]):
        words = set(_squash(line).split())
        if words and not words <= verbatim_words:
            violations.append(f"occupational history line '{line}' is not supported "
                              f"by the fact graph")

    # 2. Medications / treatments mentioned must be in the fact graph.
    for f in facts:
        if f["fact_type"] in ("MEDICATION", "TREATMENT", "PROCEDURE"):
            continue
    for m in re.finditer(r"\b(paracetamol|cetirizine|crocin|dolo|ibuprofen|"
                         r"amoxicillin|azithromycin|metformin|amlodipine)\b",
                         scan, re.I):
        if _norm_label(m.group(1)) not in medications:
            violations.append(f"medication '{m.group(1)}' in note not supported by "
                              f"the fact graph")

    # 3. Durations and severities must be stored on a fact. Verbatim
    # conversational durations are multi-word ("the whole week", "several
    # months"), so the capture runs up to five words and trailing connective
    # words are trimmed before the membership check.
    for m in re.finditer(
            r"\bfor (?:approximately )?((?:[a-z0-9]+ ){0,4}[a-z0-9]+)",
            scan, re.I):
        claimed = _norm_label(m.group(1))
        # A trajectory clause the renderer itself appends ("for four weeks
        # which then improved") is not part of the duration: cut at the
        # relative/connective boundary before trimming stray conjunctions.
        claimed = re.split(
            r"\b(?:which|that|who|but|because|while|when|before|after|"
            r"following|so|and|or|until|since)\b", claimed)[0].strip()
        while claimed and claimed.split()[-1] in {
                "and", "but", "or", "so", "because", "which", "that", "who",
                "when", "while", "with", "after", "before", "since", "then",
                "also", "following", "then"}:
            claimed = " ".join(claimed.split()[:-1])
        if not claimed:
            continue
        # A "for" phrase whose head is a document/prose noun ("for the
        # patient", "for the appointment") is not a temporal claim.
        if claimed.split()[-1] in {
                "patient", "note", "record", "visit", "consultation",
                "appointment", "clinic", "letter", "report", "review",
                "assessment", "opinion", "plan", "medication", "prescription",
                "injection", "treatment", "symptom", "finding", "result",
                "investigation", "specialist", "issue", "information"}:
            continue
        if claimed.split()[0].isdigit():
            claimed = f"{claimed.split()[0]} {claimed.split()[1].rstrip('s')}s"
        swapped = claimed.replace("one", "1").replace("two", "2").replace(
            "three", "3").replace("four", "4").replace("five", "5").replace(
            "six", "6").replace("seven", "7").replace("eight", "8").replace(
            "nine", "9").replace("ten", "10")
        if claimed not in durations and swapped not in durations and not any(
                d.endswith(claimed) or claimed.endswith(d)
                for d in durations if d):
            violations.append(f"duration '{m.group(1)}' in note not supported by the "
                              f"fact graph")
    for m in re.finditer(r"\b(severe|mild|moderate|excruciating)\b", scan, re.I):
        if _norm_label(m.group(1)) not in severities:
            violations.append(f"severity '{m.group(1)}' in note not supported by the "
                              f"fact graph")

    # 3b. Rate expressions ("on a regular basis", "multiple times a week")
    # are frequencies: they must be rendered verbatim but NEVER attached
    # with "for" ("for on a regular basis" is a duration/prose defect, and
    # a stored duration built from a rate phrase proves the fact graph
    # conflated the two).
    for f in facts:
        attrs = f.get("attributes") or {}
        for attr_name in ("frequency", "duration"):
            val = attrs.get(attr_name)
            if not val:
                continue
            for part in str(val).split(";"):
                part = part.strip().lower()
                if not part:
                    continue
                is_rate = any(re.search(rf"(?<![a-z]){p}(?![a-z])", part)
                              for p in ("times a week", "times a month",
                                        "times a day", "regular basis",
                                        "most days", "every couple"))
                if is_rate and attr_name == "duration":
                    violations.append(
                        f"rate expression '{part}' stored as duration on "
                        f"fact '{f.get('concept')}' — frequencies are not "
                        f"durations")
                if is_rate and f"for {part}" in scan.lower():
                    violations.append(
                        f"rate expression '{part}' rendered after 'for' in "
                        f"the note")

    # 4. Denials must be backed by an ABSENT fact.
    for m in re.finditer(r"denies ([a-z][a-z \-']+?)(?:[.;,\n]|$)", scan, re.I):
        claimed = _norm_label(m.group(1))
        parts = [p.strip() for p in re.split(r",? and ", claimed) if p.strip()]
        if any(p not in denied for p in parts):
            violations.append(f"denial 'denies {m.group(1)}' not supported by the "
                              f"fact graph")

    # 5. No causal claim that the transcript does not make verbatim.
    for m in re.finditer(r"\b(due to|caused by|because of|secondary to|"
                         r"as a consequence of)\b", scan, re.I):
        sentence = scan[max(0, m.start() - 120):m.end() + 120]
        if _norm_label(sentence) not in verbatim_text and not any(
                _norm_label(m.group(0)) in v for v in verbatim_text.split("\n")):
            violations.append(f"causal claim '{m.group(0)}' in note not supported "
                              f"by the fact graph")

    # 6. Greetings, spelled names and administrative references never appear as
    #    clinical content.
    for pattern, label in (
            (r"\b(?:hi|hello)\b\s*[,.]?\s*this is", "greeting"),
            (r"\bsolicitors? ref\b|\bslash \d{2,3}\b", "administrative reference"),
            (r"\byours (?:sincerely|faithfully)\b", "letter sign-off"),
            (r"\bdate of birth\b", "administrative metadata")):
        if re.search(pattern, scan, re.I) and label != "administrative metadata":
            violations.append(f"{label} text in note is not clinical documentation")

    # 7. Persistence claims require explicit ongoing evidence.
    ongoing = {_norm_label(f.get("english") or "") for f in facts
               if f.get("temporal_context") == "ONGOING"}
    for m in re.finditer(r"continues to have ([a-z][a-z \-']+)", scan, re.I):
        claimed = _norm_label(m.group(1))
        if not any(claimed.startswith(o) or claimed in o for o in ongoing):
            violations.append(f"persistence claim 'continues to have {m.group(1)}' "
                              f"not supported by the fact graph")

    return {"valid": not violations, "violations": violations,
            "checked_facts": len(facts),
            "renderer": "deterministic-v2"}
