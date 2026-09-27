"""Deterministic de-identification for training candidates.

Scope and honesty (see docs/audio_lifecycle_security.md §3):

* This layer redacts the TRANSCRIPT COPY that accompanies a training
  candidate. It is deterministic (no model, no LLM): fixed pattern classes
  plus exact-name removal from the encounter's own metadata.
* Result status is TRANSCRIPT_DEIDENTIFIED. It is NEVER AUDIO_DEIDENTIFIED:
  the original recording may still contain names and identifiers acoustically,
  and no claim of voice anonymization is made anywhere.
* Pattern-based redaction is never perfect. Reviewers see the candidate at
  TRAINING_PENDING_REVIEW precisely so a human can catch residual identifiers
  before dataset inclusion.
"""

from __future__ import annotations

import re

# Fixed token used for every redaction so reviewers can spot redactions.
REDACTED = "[REDACTED]"

# Deterministic pattern classes (ordered; longest/most-specific first).
_PATTERNS = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("phone", re.compile(r"(?:\+?\d[\d\s\-()]{7,}\d)")),
    ("mrn", re.compile(r"\b(?:MRN|UHID|IP[-/ ]?NO|OP[-/ ]?NO|CASE\s+NO)"
                       r"[:\s#]*([A-Z0-9/-]{3,})\b", re.IGNORECASE)),
    ("date", re.compile(r"\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})"
                        r"|\b(?:\d{1,2}(?:st|nd|rd|th)?\s+"
                        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
                        r"[a-z]*\s+\d{2,4})\b", re.IGNORECASE)),
    ("long_number", re.compile(r"\b\d{6,}\b")),
]

_AGE = re.compile(r"\b([1-9]\d?\s*(?:years?|yrs?|y/o|yo)\s*old)\b", re.IGNORECASE)


def deidentify_transcript(text: str, names: list[str] | None = None) -> tuple[str, dict]:
    """Return (deidentified_text, provenance).

    `names` are exact strings removed verbatim (patient/doctor/hospital as
    recorded in the encounter metadata). Only exact, case-insensitive,
    word-boundary matches are removed — no fuzzy guessing.
    """
    provenance: dict = {"counts": {}, "names_removed": []}
    out = text or ""

    for name in sorted((names or []), key=len, reverse=True):
        token = (name or "").strip()
        if len(token) < 3:
            continue
        pattern = re.compile(r"(?<![\w])" + re.escape(token) + r"(?![\w])",
                             re.IGNORECASE)
        out, n = pattern.subn(REDACTED, out)
        if n:
            provenance["names_removed"].append({"length": len(token),
                                                "replacements": n})

    for label, pattern in _PATTERNS:
        out, n = pattern.subn(REDACTED, out)
        if n:
            provenance["counts"][label] = n

    # Stated ages are quasi-identifiers in small populations: mask the number,
    # keep the clinical fact that an age was discussed.
    out, n = _AGE.subn(f"{REDACTED} years old", out)
    if n:
        provenance["counts"]["age"] = n

    provenance["status"] = "TRANSCRIPT_DEIDENTIFIED"
    return out, provenance
