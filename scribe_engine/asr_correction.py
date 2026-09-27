"""Evidence-based ASR correction for the Malayalam pipeline.

Two layers, never mixed:

    RAW ASR TRANSCRIPT (immutable, always the record)
        -> asr_correction   VERIFIED ASR corruptions only, with provenance
        -> semantic normalization   colloquial meaning -> canonical concepts
           (scribe_engine/semantic.py + malayalam_clinical_semantics.json)

An ASR correction is added ONLY when the exact corruption was observed in a
frozen IndicConformer benchmark transcript paired with the known ground-truth
fixture script (benchmarks/malayalam/transcripts/ vs.
scribe_engine/stt/experimental/indic_malayalam/make_fixtures.py). Nothing is
guessed: an unverified surface form passes through unchanged. UNKNOWN beats
GUESS, and cosmetic correction is the lowest rung of the safety hierarchy —
below not inventing findings, negation, questions, temporality, subject and
dosage preservation.

Each applied correction carries provenance:
    {"raw": ..., "corrected": ..., "type": "verified_asr_correction",
     "reason": ..., "confidence": ...}
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "")

# ---------------------------------------------------------------------------
# Verified corruption table.
#
# Source of evidence: benchmarks/malayalam/transcripts/*.ml.txt (frozen
# IndicConformer CTC output on synthetic TTS fixtures) paired against the
# ground-truth spoken scripts in make_fixtures.py.
#
# NOTE ON THE BRIEFING'S EXAMPLE: the task briefing attributed
# "ചവദാഭയബഥ" <- spoken "blood pressure". The frozen transcripts show the
# opposite pairing: ചവദാഭയബഥ / ചവതാഭയപട appear exactly where "diabetes" was
# spoken (fixtures ml_chest_pain, ml_english_medical_terms), while spoken
# "blood pressure" reached us as "ശപ ഐതുക" (fixture ml_english_medical_terms).
# The table follows the evidence, not the briefing.
# ---------------------------------------------------------------------------

# Confidence tiers by number of independent observations of the same
# corruption -> same ground truth.
_REPEATED_OBSERVED = [
    # spoken "diabetes" (2 independent fixtures)
    {"raw": "ചവദാഭയബഥ", "corrected": "diabetes", "occurrences": 2},
    {"raw": "ചവതാഭയപട", "corrected": "diabetes", "occurrences": 1},
]

_SINGLE_OBSERVED = [
    # spoken "blood pressure" (fixture ml_english_medical_terms)
    {"raw": "ശപ ഐതുക", "corrected": "blood pressure", "occurrences": 1},
]

# Spoken "nasal spray" -> "ലാൻഡിലാണ്" (fixture ml_english_medical_terms).
# ലാൻഡിലാണ് is also a plausible real Malayalam phrase ("is on land"), so this
# correction is context-guarded: it applies only inside the observed pattern —
# an instruction sentence containing ഉപയോഗിക്കുക ("use [it]"), which is how the
# corrupted sentence actually read ("ഇത് ലാൻഡിലാണ് രാവിലെ ഉപയോഗിക്കുക").

# ---------------------------------------------------------------------------
# Manually verified REAL-consultation forms (human confirmed from the audio,
# 2026-09 sessions; evidence source: the real consultation recording itself).
# These are spoken colloquial Kerala forms the ASR transcribed faithfully —
# semantic aliases would be equally defensible, but routing them through the
# correction table keeps ONE evidence trail and gives them provenance objects
# (observed form / intended form / evidence / regression test).
# ---------------------------------------------------------------------------
_REAL_AUDIO_VERIFIED = [
    {"raw": "ചൊമ്മയുണ്ട്", "corrected": "ചുമയുണ്ട്", "occurrences": 1,
     "reason": "manually verified from real consultation audio (human listening; "
               "spoken colloquial ചൊമ്മ = ചുമ/cough)",
     "type": "real_audio_verified"},
    {"raw": "തലവനൊക്കെ", "corrected": "തലവേദനയൊക്കെ", "occurrences": 1,
     "reason": "manually verified from real consultation audio (human listening; "
               "spoken colloquial തലവൻ = തലവേദന/headache)",
     "type": "real_audio_verified"},
    {"raw": "കബക്കെട്ട്", "corrected": "കഫക്കെട്ട്", "occurrences": 1,
     "reason": "manually verified from real consultation audio (human listening; "
               "ASR b/c confusion for കഫക്കെട്ട്/congestion)",
     "type": "real_audio_verified"},
]
_CONTEXT_GUARDED_OBSERVED = [
    {
        "raw": "ലാൻഡിലാണ്",
        "corrected": "nasal spray",
        "occurrences": 1,
        "context_markers": ["ഉപയോഗിക്കുക"],
    },
]


def _corrections_from(entries: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in entries:
        conf = 0.95 if e["occurrences"] >= 2 else 0.85
        reason = e.get("reason") or (
            f"observed {e['occurrences']}x in frozen IndicConformer benchmark "
            f"transcripts vs. fixture ground truth")
        out[_norm(e["raw"])] = {
            "raw": e["raw"],
            "corrected": e["corrected"],
            "type": e.get("type") or "verified_asr_correction",
            "reason": reason,
            "confidence": conf,
            "context_markers": e.get("context_markers"),
        }
    return out


_VERIFIED = _corrections_from(_REPEATED_OBSERVED + _SINGLE_OBSERVED
                              + _REAL_AUDIO_VERIFIED
                              + _CONTEXT_GUARDED_OBSERVED)


@dataclass
class CorrectionResult:
    """Corrected text plus the provenance of every change made."""

    corrected_text: str
    corrections: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"corrected_text": self.corrected_text,
                "corrections_applied": self.corrections}


def correction_table() -> dict[str, dict]:
    """The verified table (for audit/tests). Never mutate the returned dict."""
    return _VERIFIED


def _context_allows(clause: str, entry: dict) -> bool:
    markers = entry.get("context_markers")
    if not markers:
        return True
    return any(_norm(m) in clause for m in markers)


def apply_corrections(text: str) -> CorrectionResult:
    """Apply verified ASR corrections to raw ASR output.

    The input is never modified. Each replacement is recorded with full
    provenance. Context-guarded corrections fire only when the observed
    surrounding pattern is present; otherwise the raw text passes through.
    """
    result = CorrectionResult(corrected_text=_norm(text))
    if not result.corrected_text:
        return result

    # Malayalam ASR output is a run-on string with no punctuation; treat each
    # clause-sized window (split on spaces) for context checks but correct on
    # the whole text so multi-word corruptions ("ശപ ഐതുക") match.
    corrected = result.corrected_text
    for raw, entry in _VERIFIED.items():
        if raw not in corrected:
            continue
        # Context check on a window around each occurrence.
        allowed_positions: list[int] = []
        start = 0
        while True:
            idx = corrected.find(raw, start)
            if idx < 0:
                break
            window = corrected[max(0, idx - 60):idx + len(raw) + 60]
            if _context_allows(window, entry):
                allowed_positions.append(idx)
            start = idx + 1
        if not allowed_positions:
            continue
        # Replace only the allowed occurrences, right-to-left so earlier
        # indices stay valid.
        for idx in reversed(allowed_positions):
            corrected = corrected[:idx] + entry["corrected"] + corrected[idx + len(raw):]
        result.corrections.append({
            "raw": entry["raw"],
            "corrected": entry["corrected"],
            "type": entry["type"],
            "reason": entry["reason"],
            "confidence": entry["confidence"],
            "count": len(allowed_positions),
        })
    result.corrected_text = corrected
    return result
