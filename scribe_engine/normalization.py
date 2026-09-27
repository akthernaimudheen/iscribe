"""Malayalam/English clinical language normalization.

Turns what was *said* into structured clinical concepts. It is not a translator,
not an STT provider, and emphatically not a diagnoser: "നെഞ്ചുവേദന" normalizes to
`chest_pain`, never to a cardiac diagnosis.

The raw transcript is never modified. This layer only produces a parallel
structure alongside it.

Design notes
------------
*Scope-aware negation.* A naive "if ഇല്ല appears anywhere, negate everything"
rule turns "പനി ഇല്ല, പക്ഷേ ചുമ ഉണ്ട്" into two false negatives. Instead the text
is split into clauses at connectors and punctuation, and within a clause each
concept binds to its **nearest following** polarity marker. Malayalam is
verb-final, so the marker that governs a noun normally follows it. That single
rule handles the hard cases:

    "പനി ഉണ്ട് കഫം ഇല്ല"   -> fever PRESENT, phlegm ABSENT   (different markers)
    "പനിയും ചുമയും ഇല്ല"    -> fever ABSENT, cough ABSENT      (shared marker)
    "പനി ഇല്ല, പക്ഷേ ചുമ ഉണ്ട്" -> split into two clauses

*Questions are not findings.* "പനി ഉണ്ടോ?" asks about fever; it does not record
fever. It yields status QUESTION so the note layer can ignore it.

*Confidence is recognition confidence*, never a way to manufacture certainty.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

LEXICON_PATH = Path(__file__).parent / "data" / "malayalam_clinical_lexicon.json"

# Status values
PRESENT = "PRESENT"
ABSENT = "ABSENT"
QUESTION = "QUESTION"
UNKNOWN = "UNKNOWN"
POSSIBLE = "POSSIBLE"
RESOLVED = "RESOLVED"

# Temporality values
CURRENT = "CURRENT"
PAST = "PAST"
FUTURE = "FUTURE"
RECURRENT = "RECURRENT"

_lexicon: dict | None = None


def load_lexicon(path: Path | str | None = None) -> dict:
    """Load (and cache) the lexicon data file."""
    global _lexicon
    if _lexicon is None or path is not None:
        target = Path(path) if path else LEXICON_PATH
        data = json.loads(target.read_text(encoding="utf-8"))
        if path is None:
            _lexicon = data
        return data
    return _lexicon


@dataclass
class ClinicalEntity:
    """One normalized clinical concept, traceable to the text that produced it."""

    concept: str
    english: str
    surface_text: str
    status: str = PRESENT
    temporality: Optional[str] = CURRENT
    duration: Optional[str] = None
    duration_value: Optional[int] = None
    duration_unit: Optional[str] = None
    severity: Optional[str] = None
    frequency: Optional[str] = None
    body_location: Optional[str] = None
    pain_quality: Optional[str] = None
    confidence: float = 0.0
    source_clause: str = ""
    # True when the mention sits inside a conditional/hypothetical clause
    # ("come back if the fever does not settle"). Such a clause asserts nothing
    # about the patient's current state, so it must never become a finding.
    conditional: bool = False
    # Who the mention belongs to: patient | mother | father | family | unknown.
    # Family history must never be conflated with a patient finding.
    subject: str = "unknown"
    # Why the semantic layer resolved this the way it did (low_marker,
    # measurement_check, amount_question, controlled, high, family_history,
    # allergen:<name>). Provenance for clinical audit.
    context: Optional[str] = None
    # Optional, populated ONLY with evidence: raw frequency marker and its
    # normalized label (ചിലപ്പോൾ → intermittent). Frequency is a presence
    # qualifier, never an uncertainty signal.
    frequency_normalized: Optional[str] = None
    # Why the fact is uncertain (uncertainty_marker:<phrase> /
    # interrogative_tail:<phrase>). Empty/None = CONFIRMED at the semantic
    # layer. Mandatory provenance for UNCERTAIN facts.
    uncertainty_reason: Optional[str] = None
    onset: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Text handling
# --------------------------------------------------------------------------
def _norm(text: str) -> str:
    """NFC-normalise so visually identical Malayalam compares equal."""
    return unicodedata.normalize("NFC", text or "")


def _split_on_verbs(chunk: str, lex: dict) -> list[str]:
    """Split Malayalam text after each clause-final polarity verb.

    Speech recognition output carries no punctuation, so an entire consultation
    arrives as one run-on string. Treating that as a single clause is
    catastrophic for scope: one "ഉണ്ടോ" from the doctor turns every symptom in
    the transcript into a QUESTION, and one "ഇല്ല" would deny all of them.

    Malayalam is verb-final, so the presence/negation/question verb *ends* its
    clause. Splitting immediately after each one reconstructs clause boundaries
    that punctuation would otherwise have provided:

        "...പനിയും ചുമയും ഉണ്ട്ഛർദ്ദിയോ ഉണ്ടോ തലവേദയും ഉണ്ട്..."
          -> "...പനിയും ചുമയും ഉണ്ട്"   (fever, cough PRESENT)
             "ഛർദ്ദിയോ ഉണ്ടോ"            (vomiting QUESTION)
             "തലവേദയും ഉണ്ട്"            (headache PRESENT)

    Longest verbs first so "ഉണ്ടായിരുന്നു" is not cut short by "ഉണ്ട്".
    """
    verbs = sorted(
        set(lex.get("presence_markers", []))
        | set(lex.get("negation_markers", []))
        | {q for q in lex.get("question_markers", []) if q != "?"},
        key=len, reverse=True,
    )
    if not chunk.strip():
        return []

    out: list[str] = []
    cursor = 0
    i = 0
    while i < len(chunk):
        for verb in verbs:
            v = _norm(verb)
            if not v or not chunk.startswith(v, i):
                continue
            end = i + len(v)
            if not _is_clause_end(chunk, end, lex):
                continue
            out.append(chunk[cursor:end])
            cursor = end
            i = end
            break
        else:
            i += 1
    if cursor < len(chunk):
        out.append(chunk[cursor:])
    return [p for p in out if p.strip()] or [chunk]


# Malayalam combining marks: vowel signs, virama, anusvara, visarga. None of
# these can begin a word, so a match followed by one was a match on a prefix of
# a longer word (ഉണ്ട inside ഉണ്ടാകാം) and is not a clause boundary.
_COMBINING = tuple(chr(c) for c in
                   list(range(0x0D3E, 0x0D50)) + [0x0D02, 0x0D03, 0x0D57, 0x0D62, 0x0D63])

# Words that subordinate the clause before them: "പനി ഉണ്ടോ എന്ന് അറിയില്ല" is
# one statement of not knowing, not a question followed by something else.
# എങ്കിലും ("even if") belongs to the conditional verb it follows, so it must
# stay inside that clause — ഉണ്ടെങ്കിലും is one verb form, not verb + new clause.
_SUBORDINATORS = ("എന്ന്", "എന്നു", "എന്ന", "എന്ത്", "എങ്കിലും")


def _is_clause_end(chunk: str, end: int, lex: dict) -> bool:
    """True when position ``end`` is a genuine clause boundary."""
    if end >= len(chunk):
        return True
    if chunk[end] in _COMBINING:
        return False
    rest = chunk[end:].lstrip()
    if any(rest.startswith(s) for s in _SUBORDINATORS):
        return False
    # A trailing uncertainty/unknown marker also belongs to the clause before it.
    for phrase in lex.get("uncertainty_markers", []) + lex.get("unknown_markers", []):
        if rest.startswith(_norm(phrase)):
            return False
    return True


def split_clauses(text: str, lex: dict) -> list[str]:
    """Split on punctuation and contrastive connectors.

    Clause boundaries are where negation scope ends. Splitting on "പക്ഷേ"/"but"
    is what keeps "പനി ഇല്ല, പക്ഷേ ചുമ ഉണ്ട്" from negating the cough.
    """
    text = _norm(text)
    # Keep the terminal punctuation attached to its clause. Splitting it away
    # discards the "?" that marks a doctor's question, and a question that loses
    # its mark is then recorded as a reported symptom.
    parts = re.split(r"(?<=[.!?;])\s+|\n+|(?<=[ഀ-ൿ\w])\s*,\s*", text)
    parts = [p for chunk in parts for p in _split_on_verbs(chunk, lex)]
    out: list[str] = []
    splitters = sorted(lex.get("clause_splitters", []), key=len, reverse=True)
    for part in parts:
        if not part or not part.strip():
            continue
        chunk = [part]
        for sp in splitters:
            nxt: list[str] = []
            for piece in chunk:
                # Split on the connector as a standalone token, keeping the rest.
                pieces = re.split(rf"(?:(?<=\s)|^){re.escape(sp)}(?:(?=\s)|$)", piece)
                nxt.extend(pieces)
            chunk = nxt
        out.extend(c.strip() for c in chunk if c and c.strip())
    return out


def _find_markers(clause: str, markers: list[str]) -> list[tuple[int, str]]:
    """Positions of marker phrases in a clause, longest-first to avoid overlap."""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for marker in sorted(markers, key=len, reverse=True):
        start = 0
        m = _norm(marker)
        while True:
            idx = clause.find(m, start)
            if idx < 0:
                break
            if not any(a <= idx < b for a, b in taken):
                found.append((idx, marker))
                taken.append((idx, idx + len(m)))
            start = idx + 1
    return sorted(found)


def _english_word_positions(clause: str, words: list[str]) -> list[tuple[int, str]]:
    """Whole-word positions for English markers (avoids 'no' matching 'nose')."""
    found = []
    low = clause.lower()
    for w in words:
        for m in re.finditer(rf"(?<![a-z]){re.escape(w.lower())}(?![a-z])", low):
            found.append((m.start(), w))
    return sorted(found)


# --------------------------------------------------------------------------
# Concept matching
# --------------------------------------------------------------------------
# Malayalam chillu letters and the base consonant they revert to before a
# following vowel (e.g. the enclitic "-ഉം").
_CHILLU = {"ൺ": "ണ", "ൻ": "ന", "ർ": "ര",
           "ൽ": "ല", "ൾ": "ള", "ൿ": "ക"}


def _alias_variants(alias: str) -> list[str]:
    """Malayalam conjunction forms of an alias.

    The enclitic "-ഉം" ("and") rewrites the stem: കഫം + ഉം -> കഫവും, ചുമ + ഉം ->
    ചുമയും. A plain substring search for കഫം therefore misses കഫവും entirely,
    which silently drops a symptom from a list of symptoms.
    """
    variants = [alias]
    if alias.endswith("ം"):                 # കഫം -> കഫവും
        variants.append(alias[:-1] + "വും")
    elif alias and alias[-1] in _CHILLU:
        # A chillu is a consonant with no inherent vowel. Before the enclitic it
        # reverts to its base consonant: ശ്വാസം മുട്ടൽ -> ...മുട്ടലും. Without
        # this the symptom is simply not found when it appears in a list.
        variants.append(alias[:-1] + _CHILLU[alias[-1]] + "ും")
    elif alias and alias[-1] == "്":
        # Final virama elides before the enclitic: കഫക്കെട്ട് + ഉം -> കഫക്കെട്ടും.
        # Without the stem form the coordinated mention is silently missed.
        variants.append(alias[:-1] + "ും")
        # The same elision before the fused denial: ബുദ്ധിമുട്ട് + ഇല്ല ->
        # ബുദ്ധിമുട്ടില്ല ("no <stem>"). _suffix_negated then reads the surface
        # itself and records ABSENT; the idiom form ("...പറ്റുന്നില്ല") is a
        # different surface and is never generated here.
        variants.append(alias[:-1] + "ില്ല")
    elif alias and "ഀ" <= alias[-1] <= "ൿ":
        variants.append(alias + "യും")      # ചുമ -> ചുമയും
        variants.append(alias + "ും")
    return variants


def _concept_matches(clause: str, lex: dict) -> list[tuple[int, int, str, str]]:
    """(start, end, concept_id, surface) for every concept mentioned.

    Longest alias wins, so "നെഞ്ചുവേദന" is chest_pain rather than the generic
    "വേദന" pain. Generic concepts only fire where no specific one covered the span.
    """
    hits: list[tuple[int, int, str, str]] = []
    taken: list[tuple[int, int]] = []
    low = clause.lower()

    candidates: list[tuple[str, str, bool]] = []
    seen: set[tuple[str, str]] = set()
    for cid, spec in lex["concepts"].items():
        generic = bool(spec.get("generic"))
        for alias in spec.get("aliases", []):
            for variant in _alias_variants(_norm(alias)):
                if (variant, cid) not in seen:
                    seen.add((variant, cid))
                    candidates.append((variant, cid, generic))
        for alias in spec.get("english_aliases", []):
            if (alias.lower(), cid) not in seen:
                seen.add((alias.lower(), cid))
                candidates.append((alias.lower(), cid, generic))

    # Specific before generic, then longest first.
    candidates.sort(key=lambda c: (c[2], -len(c[0])))

    for alias, cid, _generic in candidates:
        is_english = bool(re.match(r"^[a-z0-9 '\-]+$", alias))
        haystack = low if is_english else clause
        start = 0
        while True:
            idx = haystack.find(alias, start)
            if idx < 0:
                break
            end = idx + len(alias)
            if is_english:
                before_ok = idx == 0 or not haystack[idx - 1].isalpha()
                after_ok = end >= len(haystack) or not haystack[end].isalpha()
            else:
                before_ok = after_ok = True
            if before_ok and after_ok and not any(
                    not (end <= a or idx >= b) for a, b in taken):
                hits.append((idx, end, cid, clause[idx:end]))
                taken.append((idx, end))
            start = idx + 1
    return sorted(hits)


def _suffix_negated(clause: str, start: int, end: int, lex: dict) -> bool:
    """True when negation is fused onto the concept word (പനിയില്ല).

    The negation morpheme must END a finite form (പനിയില്ല = fever + ഇല്ല).
    A suffix-prefix hit that keeps going (ഉണ്ടാകാ**ം**: കാം begins with the
    ാ of ില്ല) is a different, still-continuing verb form — probability, not
    denial. Requiring the boundary after the morpheme keeps ഉണ്ടാകാം POSSIBLE
    while പനിയില്ല stays ABSENT.
    """
    suffixes = [_norm(s) for s in lex.get("suffix_negations", [])]
    tail = clause[end:end + 12]
    for suffix in suffixes:
        if tail.startswith(suffix):
            after = tail[len(suffix):len(suffix) + 1]
            if after == "" or after in (" ", ".", ",", "?", "!", ";"):
                return True
    # The alias itself may already embed the negation (വേദനയില്ല).
    surface = clause[start:end]
    return any(surface.endswith(s) for s in suffixes)


# --------------------------------------------------------------------------
# Attribute extraction
# --------------------------------------------------------------------------
def _extract_duration(clause: str, lex: dict) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """Duration as (text, value, unit). Only when explicitly stated.

    Conversational temporal expressions are stored VERBATIM ("multiple times
    a week", "the whole week"): the speaker's own words are the evidence and
    no number is ever invented for a vague frequency.
    """
    for vague in lex.get("vague_durations", []):
        if _norm(vague) in clause:
            return vague, None, None

    # Conversational (English) temporal expressions — verbatim, never
    # normalized into artificial exact durations. ALL matches in the clause
    # are preserved ("on a regular basis, like, multiple times a week" keeps
    # both phrases: the speaker's own words are the evidence).
    hits: list[str] = []
    for pattern in lex.get("english_conversational_durations", []):
        for m in re.finditer(pattern, clause, re.IGNORECASE):
            phrase = m.group(0).lower()
            if phrase not in hits:
                hits.append(phrase)
    if hits:
        return "; ".join(hits), None, None

    numerals = {_norm(k): v for k, v in lex.get("numerals", {}).items()}
    units = {_norm(k): v for k, v in lex.get("duration_units", {}).items()}

    for unit_word, unit in sorted(units.items(), key=lambda kv: -len(kv[0])):
        idx = clause.find(unit_word)
        if idx < 0:
            continue
        before = clause[max(0, idx - 30):idx]
        for num_word, value in sorted(numerals.items(), key=lambda kv: -len(kv[0])):
            if before.rstrip().endswith(num_word) or num_word in before.split()[-2:]:
                return f"{value} {unit}", value, unit
        digits = re.findall(r"(\d+)\s*$", before.strip())
        if digits:
            return f"{int(digits[0])} {unit}", int(digits[0]), unit
        # Compound words like ഒരാഴ്ചയായി carry the number inside.
        for num_word, value in numerals.items():
            if num_word in unit_word:
                return f"{value} {unit}", value, unit

    m = re.search(r"(?:for|since)\s+(\d+)\s+(day|days|week|weeks|month|months|year|years)",
                  clause, re.IGNORECASE)
    if m:
        unit = m.group(2).lower().rstrip("s") + "s"
        return f"{int(m.group(1))} {unit}", int(m.group(1)), unit
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
             "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    m = re.search(rf"(?:for|since)\s+({'|'.join(words)})\s+"
                  r"(day|days|week|weeks|month|months|year|years)",
                  clause, re.IGNORECASE)
    if m:
        unit = m.group(2).lower().rstrip("s") + "s"
        return f"{words[m.group(1).lower()]} {unit}", words[m.group(1).lower()], unit
    return None, None, None


def _nearest_attribute(clause: str, position: int, mapping: dict,
                       window: int = 40) -> Optional[str]:
    """Attribute value whose trigger sits closest before the concept."""
    best, best_dist = None, window + 1
    for phrase, value in mapping.items():
        p = _norm(phrase)
        is_english = bool(re.match(r"^[a-z0-9 '\-]+$", p))
        hay = clause.lower() if is_english else clause
        needle = p.lower() if is_english else p
        for m in re.finditer(re.escape(needle), hay):
            if m.start() < position:
                dist = position - m.end()
                if 0 <= dist < best_dist:
                    best, best_dist = value, dist
    return best


# Onset evidence: the duration phrase is anchored by an onset construction —
# the unit word fused with ായി ("ദിവസമായി" = for/since 6 days, at a word
# boundary so ഉണ്ടായിരുന്നു does not match), ായിട്ട്, "മുതൽ" (since), or English
# "since". Without the anchor the duration is a plain duration, not an onset.
_ONSET_RE = re.compile(r"ായി(?=\s|$)|ായിട്ട|മുതൽ|[Ss]ince")

_FREQUENCY_NORMALIZED = {
    "എപ്പോഴും": "always",
    "എല്ലായ്പ്പോഴും": "always",
    "തുടർച്ചയായി": "continuous",
    "തുടർച്ചയായിട്ട്": "continuous",
    "ഇടയ്ക്കിടെ": "intermittent",
    "ഇടയ്ക്ക്": "intermittent",
    "പലപ്പോഴും": "often",
    "ചിലപ്പോൾ": "intermittent",
    "അപൂർവ്വമായി": "rare",
    "വല്ലപ്പോഴും": "rare",
    "ദിവസവും": "daily",
    "വീണ്ടും വീണ്ടും": "recurrent",
}


def _frequency(clause: str, lex: dict) -> tuple[Optional[str], Optional[str]]:
    """(freq_raw, freq_normalized). Normalized label only where justified:
    ചിലപ്പോൾ/ഇടയ്ക്കിടെ = intermittent (a frequency qualifier — the symptom
    EXISTS; it does not become uncertain).

    English rate expressions ("on a regular basis", "multiple times a week")
    are FREQUENCIES, not durations: they are collected verbatim — the
    speaker's own words are the evidence and no number is ever invented — and
    stored on attributes.frequency as a semicolon-joined list when several
    appear. A normalized label is only attached when the lexicon justifies
    one ("most days" -> intermittent); otherwise the verbatim phrase alone
    is the value.
    """
    for phrase in sorted(lex.get("frequency_markers", []), key=len, reverse=True):
        if _norm(phrase) in clause:
            return phrase, _FREQUENCY_NORMALIZED.get(phrase)
    for phrase in lex.get("english_recurrence_markers", []):
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", clause.lower()):
            return phrase, _FREQUENCY_NORMALIZED.get(phrase)
    return None, None


def _frequency_verbatim(clause: str, lex: dict) -> Optional[str]:
    """All verbatim English rate expressions in the clause, joined by "; ".

    Complements _frequency (which returns a single labeled hit): the verbatim
    collector finds every rate expression so "on a regular basis, like,
    multiple times a week" keeps BOTH phrases on attributes.frequency.
    """
    hits: list[str] = []
    for pattern in lex.get("english_frequency_markers", []):
        for m in re.finditer(pattern, clause, re.IGNORECASE):
            phrase = m.group(0).lower()
            if phrase not in hits:
                hits.append(phrase)
    return "; ".join(hits) if hits else None


def _norm_window(clause: str, start: int, end: int, lex: dict) -> list[str]:
    """Neighborhood window around a concept mention for scoped uncertainty.

    Uncertainty markers ("തോന്നുന്നു", "ഉറപ്പില്ല", "അറിയില്ാ" …) count as
    modifying the concept ONLY inside this window — clause-global matching
    would make a doctor's unrelated doubt deny/uncertain every symptom in the
    sentence. AFTER: from the mention's end to its next clause boundary;
    BEFORE: up to 25 chars back, cut at the previous clause boundary so an
    earlier sentence's doubt never reaches across.
    """
    boundaries = [0, len(clause)]
    # A polarity marker bounds the window only when it genuinely ENDS a
    # predication. An embedded copula ("പനി ആണെന്ന് തോന്നുന്നു": the ആണ് is
    # subordinated by എന്ന് before the perception verb) does not — the whole
    # stretch is one predication about the concept. Boundaries are recorded at
    # the marker's END so the predicate itself ("...എന്ന് തോന്നുന്നു") stays
    # INSIDE the preceding concept's window.
    for pos, m in (_find_markers(clause, lex["negation_markers"])
                   + _find_markers(clause, lex["presence_markers"])):
        if _is_clause_end(clause, pos + len(m), lex):
            boundaries.append(pos + len(m))
    for pos, w in _english_word_positions(
            clause, lex["english_negation_markers"] + lex["english_presence_markers"]):
        if _is_clause_end(clause, pos + len(w), lex):
            boundaries.append(pos + len(w))
    boundaries.sort()
    w_start = end
    for b in boundaries:
        if b >= end:
            w_start_end = b
            break
    else:
        w_start_end = len(clause)
    w_before_start = 0
    for b in reversed(boundaries):
        if b <= start:
            w_before_start = b
            break
    after = clause[w_start:w_start_end]
    before = clause[max(w_before_start, start - 25):start]
    return [after, before]


# --------------------------------------------------------------------------
# Status resolution
# --------------------------------------------------------------------------
def _clause_polarity_markers(clause: str, lex: dict) -> list[tuple[int, str]]:
    """Every polarity marker in the clause as (position, 'pos'|'neg')."""
    markers: list[tuple[int, str]] = []
    for pos, _m in _find_markers(clause, lex["negation_markers"]):
        markers.append((pos, "neg"))
    for pos, _m in _find_markers(clause, lex["presence_markers"]):
        if not any(abs(pos - p) < 2 for p, _ in markers):
            markers.append((pos, "pos"))
    english_neg = _english_word_positions(clause, lex["english_negation_markers"])
    english_pos = _english_word_positions(clause, lex["english_presence_markers"])
    for pos, _w in english_neg:
        markers.append((pos, "neg"))
    for pos, word in english_pos:
        # "don't have fever": the auxiliary is inside the negation, so it must
        # not register as a separate positive marker between the negation and
        # the concept - that would read as PRESENT.
        if any(0 <= pos - (npos + len(nw)) <= 2 for npos, nw in english_neg):
            continue
        markers.append((pos, "pos"))
    return sorted(markers)


def _tail_of(clause: str, lex: dict) -> Optional[str]:
    """Clause-final interrogative tail ("...എന്ന് അറിയില്ല", "...എന്ന് ഉറപ്പില്ല").

    Such a tail is a statement ABOUT the previous predication — the speaker
    does not know / is not sure whether the preceding clause's content holds —
    and must never be read as a negation of anything (ഉറപ്പില്ല embeds ഇല്ല
    but means "not sure", not "not present").
    """
    stripped = clause.rstrip(" .!?;")
    for phrase in sorted({*lex.get("unknown_markers", []),
                          *lex.get("english_unknown_markers", [])},
                         key=len, reverse=True):
        if stripped.endswith(_norm(phrase)):
            return phrase
    return None


def _status_for(clause: str, start: int, end: int, lex: dict,
                is_question: bool) -> tuple[str, float, Optional[str]]:
    """Decide the status of one concept mention within its clause.

    Returns (status, confidence, uncertainty_reason). All uncertainty/unknown
    matching is CONCEPT-SCOPED via _norm_window — never clause-global: an
    unrelated doubt in the same clause must not modify this concept, and a
    perception marker ("...എന്ന് തോന്നുന്നു") must reach the concept it
    follows while never crossing into the next predication.
    """
    reason = None
    after, before = _norm_window(clause, start, end, lex)
    window = after + (" " + before if before else "")

    # Clause-final "...എന്ന് അറിയില്ല / ഉറപ്പില്ല": the speaker is talking about
    # the preceding predication, not denying this concept. Tested FIRST so the
    # embedded ഇല്ല can never be misread as polarity.
    tail = _tail_of(clause, lex)
    if tail:
        return UNKNOWN, 0.8, f"interrogative_tail:{tail}"

    # "പനി ആണെന്ന് തോന്നുന്നു" — a perception/probability marker inside the
    # concept's own neighborhood makes THIS concept uncertain. A bare
    # experience verb ("ക്ഷീണം തോന്നുന്നുണ്ട്") is a PRESENCE marker and is
    # handled by polarity below; only explicit doubt words land here.
    for phrase in lex.get("uncertainty_markers", []):
        if _norm(phrase) in window:
            return POSSIBLE, 0.8, f"uncertainty_marker:{phrase}"
    for phrase in lex.get("english_uncertainty_markers", []):
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", window.lower()):
            return POSSIBLE, 0.8, f"uncertainty_marker:{phrase}"
    # "Sounds like" is inherently a paraphrase hedge about whatever predication
    # follows it ("it sounds like you were feeling all weak" — the clinician's
    # wording, not the patient's assertion), so a clause-level check is safe
    # for THIS marker alone; the other uncertainty markers stay window-scoped.
    if re.search(r"(?<![a-z])sounds like(?![a-z])", clause.lower()):
        return POSSIBLE, 0.7, "paraphrase_hedge:sounds like"

    if is_question:
        return QUESTION, 0.9, None

    # A hypothetical clause asserts nothing about the patient ("പനി
    # മാറിയില്ലെങ്കിൽ" literally embeds മാറി+ഇല്ല). This must outrank every
    # polarity/resolution marker inside it.
    if _is_conditional(clause, lex):
        return PRESENT, 0.5, None

    if _suffix_negated(clause, start, end, lex):
        return ABSENT, 0.95, None

    # Nearest FOLLOWING marker governs (Malayalam is verb-final); fall back to
    # the nearest preceding one for English word order ("no chest pain").
    # English fronted negation ("No injection was given") is decided by the
    # negation BEFORE the concept, not by the past-tense verb that follows it:
    # an English negation word anywhere before the mention outranks a later
    # presence marker, because Malayalam verb-final polarity has no analogue
    # of "No X was given" and the old order read such sentences as PRESENT.
    markers = _clause_polarity_markers(clause, lex)
    after_m = [(p, k) for p, k in markers if p >= end]
    before_m = [(p, k) for p, k in markers if p < start]
    before_neg = [p for p, k in before_m if k == "neg"]
    # English clause = latin script only (no Malayalam block characters).
    english_clause = not re.search(r"[\u0d00-\u0d7f]", clause)
    if english_clause and before_neg and (not after_m or after_m[0][1] != "neg"):
        last_neg = before_neg[-1]
        # The negation must govern the mention directly: a presence marker
        # between the negation and the concept ("not yet symptomatic in that
        # he still HAS some residual pain") re-asserts presence, so the early
        # "not" no longer denies it. Without this guard the residual-pain
        # sentence in a real medico-legal report was denied outright.
        pos_between = any(k == "pos" and last_neg < p < start for p, k in before_m)
        # "Not + QUALIFIER + concept" ("some not well-defined apprehension")
        # negates the QUALITY of the finding, not its existence — the finding
        # is still reported. The negation is scoped to the qualifier.
        between = clause[last_neg:start]
        if re.search(r"\b(?:not|no)\s+(?:well[- ]?defined|clear|significant|"
                     r"specific|certain|typical|substantial|major|severe)\b",
                     between, re.I):
            return PRESENT, 0.6, "negation_scoped_to_qualifier"
        # "not (quite|entirely|fully) over X": still HAS X ("still not quite
        # over the feeling of being totally alone").
        if re.search(r"\bnot\s+(?:quite|entirely|fully|completely)\s+over\b",
                     clause, re.I):
            return PRESENT, 0.7, "negation_scoped_to_recovery"
        # Reported-speech negation (FIRST-PERSON communication verbs only):
        # "I often don't say that I feel so tired" — the negation governs the
        # SPEECH ACT (a habit of not reporting), not the symptom. Belief
        # negations ("I don't think there is chest pain") deliberately do NOT
        # match: a doubted finding stays denied/uncertain.
        if re.search(r"\bI\s+(?:often\s+|usually\s+|sometimes\s+)?"
                     r"(?:don'?t|do not|never)\s+"
                     r"(?:say|tell|report|mention)\b", between, re.I):
            return PRESENT, 0.7, "negation_scoped_to_speech_act"
        # "not why…" / "not because…": the negation targets the CAUSE, not
        # the finding ("that's not why I was crying" — the crying happened).
        if re.search(r"\b(?:not|no)\s+(?:why|because|'cause|cause)\b",
                     between + clause[start:start + 12], re.I):
            return PRESENT, 0.7, "negation_scoped_to_cause"
        if not pos_between:
            return ABSENT, 0.9, None
    if after_m:
        # A FOLLOWING negation whose own object is a gerund/state ("without
        # being hospitalized") negates THAT state, not the preceding finding
        # ("off and on suicidal for several months without being hospitalized"
        # — the suicidal ideation is affirmed). Skip the gerund-object
        # negation and let the remaining markers decide.
        first_neg = next((p for p, k in after_m if k == "neg"), None)
        if first_neg is not None and re.match(
                r"(?:without|not)\s+being\b",
                clause[first_neg:first_neg + 20], re.I):
            rest = [(p, k) for p, k in after_m if (p, k) != (first_neg, "neg")]
            if rest:
                return (ABSENT if rest[0][1] == "neg" else PRESENT), 0.9, None
            return PRESENT, 0.75, None
        return (ABSENT if after_m[0][1] == "neg" else PRESENT), 0.9, None
    if before_m:
        return (ABSENT if before_m[-1][1] == "neg" else PRESENT), 0.75, None

    # A bare mention with no marker: recorded, but with low confidence.
    return PRESENT, 0.5, None


def _temporality(clause: str, lex: dict) -> tuple[Optional[str], bool]:
    """(temporality, is_recurrent)."""
    for phrase in lex.get("recurrence_markers", []):
        if _norm(phrase) in clause:
            return RECURRENT, True
    for phrase in lex.get("english_recurrence_markers", []):
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", clause.lower()):
            return RECURRENT, True

    best, best_pos = None, -1
    for phrase, value in lex.get("temporal_markers", {}).items():
        idx = clause.find(_norm(phrase))
        if idx >= 0 and idx > best_pos:
            best, best_pos = value, idx
    for phrase, value in lex.get("english_temporal_markers", {}).items():
        m = re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", clause.lower())
        if m and m.start() > best_pos:
            best, best_pos = value, m.start()

    if best is None:
        # Past-tense presence markers imply the past without a date word.
        for marker in ("ഉണ്ടായിരുന്നു", "വന്നിരുന്നു", "ഉണ്ടായിട്ടുണ്ട്", "ആയിരുന്നു", "ഛർദ്ദിച്ചു"):
            if _norm(marker) in clause:
                return PAST, False
    return best, False


# Statuses beyond the base four that the medication-modality layer uses.
# RESOLVED already exists; PAST marks a clearly historical ("used to") mention
# and CONSIDERED a proposed-but-not-taken treatment.
PAST_STATUS = "PAST"
CONSIDERED_STATUS = "CONSIDERED"


def _is_question(clause: str, lex: dict) -> bool:
    if "?" in clause:
        return True
    # Conversational questions often lose the question mark in ASR output;
    # an interrogative OPENER ("are you taking Prozac") still marks a
    # question. Prefix-anchored so a statement containing the words ("not
    # sure are you") is not misread.
    for opener in lex.get("english_question_openers", []):
        if clause.lower().startswith(opener):
            return True
    return any(_norm(q) in clause for q in lex["question_markers"] if q != "?")


def _is_conditional(clause: str, lex: dict) -> bool:
    """True for hypothetical clauses, which assert nothing about the patient.

    "come back in one week if the fever does not settle" contains a negation
    next to 'fever', but it is an instruction about a possible future, not a
    denial of a present fever. Treating it as a finding produced a note that
    simultaneously reported and denied the same symptom.
    """
    for phrase in lex.get("conditional_markers", []):
        if _norm(phrase) in clause:
            return True
    for phrase in lex.get("english_conditional_markers", []):
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", clause.lower()):
            return True
    return False


def _is_resolution(clause: str, lex: dict) -> bool:
    for phrase in lex.get("resolution_markers", []):
        if _norm(phrase) in clause:
            return True
    for phrase in lex.get("english_resolution_markers", []):
        if re.search(rf"(?<![a-z]){re.escape(phrase)}(?![a-z])", clause.lower()):
            return True
    return False


def _resolution_scope(clause: str, matches: list, lex: dict) -> set:
    """Concept spans in this clause that a resolution marker governs.

    A resolution verb resolves the concept it belongs to, not every concept
    that happens to share its clause. Malayalam is verb-final: the resolved
    subject precedes its resolution verb ("പനി മാറി", "fever is gone"), so a
    marker governs mentions BEFORE it. A resolution BEFORE a mention belongs
    to an earlier predication — after a finite/converb resolution form
    ("മാറിട്ടു", "മാറിയെങ്കിലും") Malayalam starts a new predication, which is
    exactly the leak that resolved fatigue standing next to resolved fever.
    Coordination sharing one final verb still works: both mentions precede it
    ("പനിയും ക്ഷീണവും മാറി"). A bare mention with no following resolution keeps
    its polarity-derived status.
    """
    if not matches:
        return set()
    res = _find_markers(clause, lex.get("resolution_markers", []))
    res += _english_word_positions(clause, lex.get("english_resolution_markers", []))
    if not res:
        return set()
    return {(start, end) for start, end, _cid, _surface in matches
            if any(p >= end for p, _m in res)}


def _resolution_clause_final(clause: str, lex: dict) -> bool:
    """True when the clause's final predicate is a resolution verb.

    "ഇപ്പോൾ മാറി" asserts only that something resolved and names no symptom —
    the ellipsis path must resolve the symptom just discussed. The finality
    guard keeps an incidental "മാറി" substring inside a longer future verb
    ("മാറിപ്പോകും", "മാറിക്കൊള്ളാം") from fabricating resolutions.
    """
    tail = clause.rstrip(" .!?;")
    found = _find_markers(tail, lex.get("resolution_markers", []))
    return any(p + len(m) >= len(tail) - 2 for p, m in found)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def normalize_clinical_text(text: str, lexicon: dict | None = None) -> list[ClinicalEntity]:
    """Extract structured clinical entities from Malayalam and/or English text.

    The input is never modified; this returns a parallel structure.
    """
    lex = lexicon or load_lexicon()
    entities: list[ClinicalEntity] = []
    if not text or not isinstance(text, str):
        return entities

    clauses = split_clauses(text, lex)
    previous_concepts: list[tuple[str, str]] = []   # (concept_id, surface)
    previous_source: str = ""                       # source_clause of the previous concept-bearing clause

    for position, clause in enumerate(clauses):
        clause_n = _norm(clause)
        question = _is_question(clause_n, lex)
        resolution = _is_resolution(clause_n, lex)
        conditional = _is_conditional(clause_n, lex)
        temporality, _recurrent = _temporality(clause_n, lex)
        duration_text, duration_value, duration_unit = _extract_duration(clause_n, lex)
        frequency_raw, frequency_norm = _frequency(clause_n, lex)

        matches = _concept_matches(clause_n, lex)

        # Ellipsis: "ഇന്നലെ പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ ഇല്ല" - the second clause names
        # no symptom but carries a polarity marker. It refers back to the
        # symptom just discussed. Without this the "ഇപ്പോൾ ഇല്ല" is dropped and
        # a resolved fever is recorded as still present. The same holds for a
        # clause whose whole predicate is a resolution verb ("ഇപ്പോൾ മാറി"):
        # it carries no polarity marker but still resolves the previous symptom.
        if not matches and previous_concepts:
            tail = _tail_of(clause_n, lex)
            polarity = _clause_polarity_markers(clause_n, lex)
            res_final = _resolution_clause_final(clause_n, lex)
            if tail:
                # "പനി ഉണ്ട്, പക്ഷേ ഉറപ്പില്ല" — a tail-only clause is a statement
                # ABOUT the previous predication: the speaker is not sure it
                # holds. Mutate that predication's entities to UNKNOWN (no
                # duplicate entity, and the doubt must outrank the earlier
                # PRESENT because it temporally follows it). The tail embeds
                # ഇല്ല, but means "not sure", never "not present".
                tail_cids = {cid for cid, _s in previous_concepts}
                for ent in entities:
                    if (ent.concept in tail_cids
                            and ent.source_clause == previous_source):
                        ent.status = UNKNOWN
                        ent.confidence = 0.8
                        ent.uncertainty_reason = f"interrogative_tail:{tail}"
            elif polarity and not question:
                for cid, surface in previous_concepts:
                    spec = lex["concepts"][cid]
                    status = ABSENT if polarity[0][1] == "neg" else PRESENT
                    entities.append(ClinicalEntity(
                        concept=cid, english=spec["english"], surface_text=surface,
                        status=RESOLVED if resolution else status,
                        temporality=temporality or CURRENT,
                        duration=duration_text, duration_value=duration_value,
                        duration_unit=duration_unit,
                        frequency=frequency_raw, frequency_normalized=frequency_norm,
                        confidence=0.7, source_clause=clause.strip(),
                        conditional=conditional,
                    ))
            elif res_final and not question:
                for cid, surface in previous_concepts:
                    spec = lex["concepts"][cid]
                    entities.append(ClinicalEntity(
                        concept=cid, english=spec["english"], surface_text=surface,
                        status=RESOLVED,
                        temporality=temporality or CURRENT,
                        duration=duration_text, duration_value=duration_value,
                        duration_unit=duration_unit,
                        frequency=frequency_raw, frequency_normalized=frequency_norm,
                        confidence=0.7, source_clause=clause.strip(),
                        conditional=conditional,
                    ))
            elif duration_text and not question:
                # Duration ellipsis: "പനി ഉണ്ടായിരുന്നു, ആറു ദിവസം ആയി" — a clause
                # that names no symptom but states how long it has been going on
                # refers to the symptom just discussed (Malayalam narration
                # commonly drops the subject). Attach the duration to the
                # previous concept instead of dropping it. Confined to
                # duration-only clauses with no polarity verb of their own, so
                # no status can ever ride across the boundary this way.
                for cid, surface in previous_concepts:
                    spec = lex["concepts"][cid]
                    for ent in entities:
                        if ent.concept == cid and ent.source_clause == previous_source:
                            ent.duration = ent.duration or duration_text
                            ent.duration_value = ent.duration_value or duration_value
                            ent.duration_unit = ent.duration_unit or duration_unit
                            if _ONSET_RE.search(clause_n):
                                ent.onset = ent.onset or duration_text
                            break
                    else:
                        entities.append(ClinicalEntity(
                            concept=cid, english=spec["english"], surface_text=surface,
                            status=PRESENT, temporality=temporality or CURRENT,
                            duration=duration_text, duration_value=duration_value,
                            duration_unit=duration_unit,
                            frequency=frequency_raw, frequency_normalized=frequency_norm,
                            onset=(duration_text if _ONSET_RE.search(clause_n) else None),
                            confidence=0.7, source_clause=clause.strip(),
                            conditional=conditional,
                        ))
            continue

        if matches:
            previous_concepts = [(cid, surface) for _s, _e, cid, surface in matches]
            previous_source = clause.strip()

        resolution_scope = _resolution_scope(clause_n, matches, lex)
        for start, end, cid, surface in matches:
            spec = lex["concepts"][cid]
            status, confidence, ureason = _status_for(clause_n, start, end, lex, question)

            # A concept whose name already means an absence ("loss of appetite")
            # must not be flipped to ABSENT by its own wording; likewise aliases
            # that literally contain a negation ("ശ്വാസം കിട്ടുന്നില്ല" = breath
            # not coming = the symptom is present). A true negation of the idiom
            # itself ("ശ്വാസംമുട്ടലില്ല" = no breathlessness) still stays ABSENT:
            # there the negation lands on the concept word, not inside the idiom.
            negative_idioms = {_norm(x) for x in spec.get("negative_idioms", [])}
            if status == ABSENT and (
                (spec.get("inherently_negative")
                 and _suffix_negated(clause_n, start, end, lex))
                or surface in negative_idioms
            ):
                status, confidence = PRESENT, 0.85

            # Resolution applies only to the concept span the resolution verb
            # governs (see _resolution_scope): fever's മാറി must not resolve
            # the fatigue named beside it. A conditional clause asserts nothing
            # about the patient, so it also never becomes a resolution.
            if ((start, end) in resolution_scope
                    and status in (PRESENT, ABSENT) and not conditional):
                status = RESOLVED
                confidence = max(confidence, 0.8)

            entities.append(ClinicalEntity(
                concept=cid,
                english=spec["english"],
                surface_text=surface,
                status=status,
                temporality=(RECURRENT if _recurrent else (temporality or CURRENT)),
                duration=duration_text,
                duration_value=duration_value,
                duration_unit=duration_unit,
                severity=_nearest_attribute(
                    clause_n, start,
                    {**lex.get("severity_markers", {}),
                     **lex.get("english_severity_markers", {})}),
                frequency=frequency_raw,
                frequency_normalized=frequency_norm,
                body_location=_nearest_attribute(
                    clause_n, start, lex.get("body_locations", {}), window=20),
                pain_quality=_nearest_attribute(
                    clause_n, start, lex.get("pain_quality", {}), window=30),
                onset=(duration_text if (duration_text and _ONSET_RE.search(clause_n))
                       else None),
                confidence=round(confidence, 2),
                source_clause=clause.strip(),
                conditional=conditional,
                uncertainty_reason=ureason,
            ))

    return _merge_across_clauses(entities)


def _merge_across_clauses(entities: list[ClinicalEntity]) -> list[ClinicalEntity]:
    """Resolve a concept stated across clauses.

    "ഇന്നലെ പനി ഉണ്ടായിരുന്നു, ഇപ്പോൾ ഇല്ല" is one finding (fever, RESOLVED), not a
    contradictory pair. Past-present followed by current-absent is exactly the
    definition of resolved.
    """
    by_concept: dict[str, list[ClinicalEntity]] = {}
    for e in entities:
        by_concept.setdefault(e.concept, []).append(e)

    merged: list[ClinicalEntity] = []
    for _cid, group in by_concept.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # A hypothetical mention must never outrank a real one. Without this,
        # "come back if the fever does not settle" beat the patient's actual
        # "I have a fever" and the note both reported and denied fever.
        asserted = [e for e in group if not e.conditional]
        if asserted:
            group = asserted

        past_present = [e for e in group
                        if e.status == PRESENT and e.temporality in (PAST, RECURRENT)]
        current_absent = [e for e in group
                          if e.status == ABSENT and e.temporality in (CURRENT, None)]
        # "പനി ഉണ്ട്, പക്ഷേ ഇപ്പോൾ ഇല്ല" — stated present, negated now, both
        # temporally current. The resolution word may be absent from the speech;
        # the meaning is still resolved. Same merger as the past-present form.
        current_present = [e for e in group
                           if e.status == PRESENT and e.temporality == CURRENT]
        if not past_present and current_present and current_absent:
            past_present = current_present
        if past_present and current_absent:
            base = past_present[0]
            base.status = RESOLVED
            base.temporality = PAST
            base.confidence = max(base.confidence, 0.85)
            # Keep every clause that contributed: downstream context rules
            # ("control ആണ്", resolution markers) may live in the second half,
            # and clinical audit needs the full evidence.
            base.source_clause = " | ".join(
                dict.fromkeys([e.source_clause for e in group]))
            # Carry a duration stated in either half.
            for e in group:
                if base.duration is None and e.duration:
                    base.duration = e.duration
                    base.duration_value = e.duration_value
                    base.duration_unit = e.duration_unit
                if base.uncertainty_reason is None and e.uncertainty_reason:
                    base.uncertainty_reason = e.uncertainty_reason
                if base.onset is None and e.onset:
                    base.onset = e.onset
            merged.append(base)
            continue

        # Otherwise prefer the most informative statement over a bare mention.
        rank = {QUESTION: 0, UNKNOWN: 1, POSSIBLE: 2, PRESENT: 3, ABSENT: 3, RESOLVED: 4}
        group.sort(key=lambda e: (rank.get(e.status, 0), e.confidence))
        best = group[-1]
        best.source_clause = " | ".join(
            dict.fromkeys(e.source_clause for e in group if e.source_clause)) \
            if best.source_clause else best.source_clause
        for e in group:
            if best.duration is None and e.duration:
                best.duration, best.duration_value, best.duration_unit = (
                    e.duration, e.duration_value, e.duration_unit)
            if best.severity is None and e.severity:
                best.severity = e.severity
            if best.uncertainty_reason is None and e.uncertainty_reason:
                best.uncertainty_reason = e.uncertainty_reason
        merged.append(best)

    return merged


def entities_to_english_summary(entities: list[ClinicalEntity]) -> str:
    """A normalized clinical representation — NOT a transcript.

    Never present this as what the patient said; it is a structured restatement.
    """
    present = [e for e in entities if e.status == PRESENT]
    absent = [e for e in entities if e.status == ABSENT]
    resolved = [e for e in entities if e.status == RESOLVED]
    possible = [e for e in entities if e.status == POSSIBLE]

    parts: list[str] = []
    if present:
        chunks = []
        for e in present:
            text = e.english
            if e.severity:
                text = f"{e.severity} {text}"
            if e.duration:
                text += f" for {e.duration}"
            if e.temporality == RECURRENT:
                text += " (recurrent)"
            chunks.append(text)
        parts.append(f"Reports {', '.join(chunks)}.")
    if resolved:
        parts.append(f"Resolved: {', '.join(e.english for e in resolved)}.")
    if possible:
        parts.append(f"Possible: {', '.join(e.english for e in possible)}.")
    if absent:
        parts.append(f"Denies {', '.join(e.english for e in absent)}.")
    return " ".join(parts)


def normalize_to_dicts(text: str, lexicon: dict | None = None) -> list[dict]:
    """Convenience wrapper returning plain dicts for JSON storage."""
    return [e.as_dict() for e in normalize_clinical_text(text, lexicon)]
