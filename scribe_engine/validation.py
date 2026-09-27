"""Transcript quality guardrails.

Runs between STT and clinical extraction. Its job is to refuse to pass a
transcript downstream when there is concrete evidence it does not represent what
was spoken — wrong language, wrong script, empty output, or the repetition
signature of a hallucinating decoder.

Design rules:

* It **flags**, it never repairs. Rewriting an uncertain word into a plausible
  medical term is exactly the failure mode this module exists to prevent.
* Every check is evidence-based and reported with the measurement that
  triggered it, so a clinician or engineer can see why.
* `severity` separates "show the clinician a warning" from "do not present this
  as a successful transcript".
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

# Non-Latin script blocks we expect never to appear in an English transcript.
# Presence of these is the clearest signal that STT decoded the wrong language.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("Devanagari", 0x0900, 0x097F),   # Hindi, Marathi
    ("Bengali", 0x0980, 0x09FF),
    ("Gurmukhi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Oriya", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
    ("Sinhala", 0x0D80, 0x0DFF),
    ("Thai", 0x0E00, 0x0E7F),
    ("Arabic", 0x0600, 0x06FF),       # also Urdu
    ("Hebrew", 0x0590, 0x05FF),
    ("Cyrillic", 0x0400, 0x04FF),
    ("Greek", 0x0370, 0x03FF),
    ("Han", 0x4E00, 0x9FFF),
    ("Hiragana", 0x3040, 0x309F),
    ("Katakana", 0x30A0, 0x30FF),
    ("Hangul", 0xAC00, 0xD7AF),
)

# Thresholds. Deliberately conservative: a false "invalid" costs a re-record,
# a false "valid" puts fabricated words in a clinical record.
# The script a transcript is expected to be written in, per language. The check
# is "is this the script this language uses", not "is this Latin": a Malayalam
# transcript is correct in Malayalam script and suspicious in Latin.
EXPECTED_SCRIPT = {
    "en": "Latin",
    "ml": "Malayalam",
    "hi": "Devanagari",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "bn": "Bengali",
}

MIN_CHARS = 8
MIN_WORDS = 3
# An English clinical transcript has no legitimate reason to contain Devanagari,
# Malayalam, Tamil or any other non-Latin script. A single such character means
# the decoder produced something that was not spoken, so it warns immediately
# rather than waiting for a percentage threshold.
NON_LATIN_WARN_COUNT = 1
NON_LATIN_INVALID_RATIO = 0.10    # this much is not an English transcript at all
REPEAT_PHRASE_MIN_WORDS = 3
REPEAT_PHRASE_INVALID_COUNT = 5   # same 3-gram 5+ times = decoder loop
REPEAT_RATIO_INVALID = 0.55       # share of text taken by one repeated phrase
SEGMENT_REPEAT_INVALID = 4        # identical consecutive segments


@dataclass
class TranscriptValidation:
    """Outcome of validating one transcript."""

    severity: str = "ok"                       # "ok" | "warning" | "invalid"
    codes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.severity == "ok"

    @property
    def is_invalid(self) -> bool:
        return self.severity == "invalid"

    @property
    def message(self) -> str:
        """Single human-readable line for the clinician."""
        if self.severity == "ok":
            return "Transcript passed automatic quality checks."
        if self.severity == "warning":
            return ("Transcription quality could not be fully verified. Please review "
                    "the audio and transcript before using this note. "
                    + " ".join(self.issues))
        return ("Transcription quality could not be verified. Please review the "
                "audio/transcript before using this note. " + " ".join(self.issues))

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "ok": self.ok,
            "codes": self.codes,
            "issues": self.issues,
            "metrics": self.metrics,
            "message": self.message,
        }

    def _add(self, severity: str, code: str, issue: str) -> None:
        order = {"ok": 0, "warning": 1, "invalid": 2}
        if order[severity] > order[self.severity]:
            self.severity = severity
        self.codes.append(code)
        self.issues.append(issue)


def _script_counts(text: str) -> dict[str, int]:
    counts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        cp = ord(ch)
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break
        else:
            if cp < 0x0250 or unicodedata.name(ch, "").startswith("LATIN"):
                counts["Latin"] += 1
            else:
                counts["Other"] += 1
    return dict(counts)


def _repeated_phrase(words: list[str]) -> tuple[str, int]:
    """Most-repeated n-gram and its count — the decoder-loop signature."""
    if len(words) < REPEAT_PHRASE_MIN_WORDS * 2:
        return "", 0
    grams = Counter(
        " ".join(words[i:i + REPEAT_PHRASE_MIN_WORDS])
        for i in range(len(words) - REPEAT_PHRASE_MIN_WORDS + 1)
    )
    phrase, count = grams.most_common(1)[0]
    return phrase, count


def _max_consecutive_segment_repeats(segments: list) -> int:
    best = run = 1
    previous = None
    for seg in segments or []:
        text = (seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")) or ""
        text = text.strip().lower()
        if not text:
            continue
        if text == previous:
            run += 1
            best = max(best, run)
        else:
            run = 1
        previous = text
    return best


def validate_transcript(
    text: str,
    expected_language: str = "en",
    reported_language: str | None = None,
    segments: list | None = None,
) -> TranscriptValidation:
    """Check a transcript for evidence it does not reflect the spoken audio."""
    result = TranscriptValidation()
    text = (text or "").strip()
    words = re.findall(r"\S+", text)

    result.metrics.update({
        "chars": len(text),
        "words": len(words),
        "expected_language": expected_language,
        "reported_language": reported_language,
        "segments": len(segments or []),
    })

    # -- empty / too short ------------------------------------------------
    if not text:
        result._add("invalid", "empty_transcript",
                    "The transcript is empty — no speech was recognised.")
        return result
    if len(text) < MIN_CHARS or len(words) < MIN_WORDS:
        result._add("invalid", "transcript_too_short",
                    f"Only {len(words)} word(s) were recognised.")

    # -- language reported by the provider --------------------------------
    if reported_language:
        base = reported_language.split("-")[0].lower()
        result.metrics["reported_language_base"] = base
        if base != expected_language.lower():
            result._add(
                "invalid", "unexpected_language",
                f"The speech engine reported '{reported_language}', not "
                f"'{expected_language}'.")

    # -- script analysis ---------------------------------------------------
    lang = expected_language.split("-")[0].lower()
    wanted = EXPECTED_SCRIPT.get(lang, "Latin")
    counts = _script_counts(text)
    letters = sum(counts.values())
    in_script = counts.get(wanted, 0)
    # Latin is tolerated inside an Indic transcript: clinical speech is
    # code-switched, and English drug names legitimately appear in Malayalam.
    tolerated = in_script + (counts.get("Latin", 0) if wanted != "Latin" else 0)
    unexpected = letters - tolerated
    ratio = (unexpected / letters) if letters else 0.0
    foreign = sorted(
        (name for name, n in counts.items()
         if n > 0 and name != wanted and not (wanted != "Latin" and name == "Latin")),
        key=lambda n: -counts[n],
    )
    result.metrics.update({
        "letters": letters,
        "expected_script": wanted,
        "expected_script_letters": in_script,
        "unexpected_script_letters": unexpected,
        "unexpected_script_ratio": round(ratio, 4),
        "scripts_detected": foreign,
    })

    if ratio >= NON_LATIN_INVALID_RATIO:
        result._add(
            "invalid", "wrong_script",
            f"{round(ratio * 100)}% of the text is in an unexpected script "
            f"({', '.join(foreign[:3])}); this is not a {lang} transcript.")
    elif unexpected >= NON_LATIN_WARN_COUNT:
        result._add(
            "warning", "mixed_script",
            f"{unexpected} character(s) in an unexpected script "
            f"({', '.join(foreign[:3])}) for a {lang} transcript.")

    # An Indic transcript with no Indic characters at all means the engine
    # decoded the wrong language even if it claimed otherwise.
    if wanted != "Latin" and letters and in_script == 0:
        result._add(
            "invalid", "expected_script_absent",
            f"No {wanted} characters found in a transcript expected to be {lang}.")

    # -- hallucination / decoder loop --------------------------------------
    phrase, count = _repeated_phrase([w.lower() for w in words])
    if phrase:
        covered = (count * REPEAT_PHRASE_MIN_WORDS) / max(len(words), 1)
        result.metrics.update({
            "top_repeated_phrase_count": count,
            "top_repeated_phrase_coverage": round(covered, 3),
        })
        if count >= REPEAT_PHRASE_INVALID_COUNT and covered >= REPEAT_RATIO_INVALID:
            result._add(
                "invalid", "pathological_repetition",
                f"A single phrase repeats {count} times and covers "
                f"{round(covered * 100)}% of the transcript, which indicates the "
                "speech engine looped rather than transcribing speech.")
        elif count >= REPEAT_PHRASE_INVALID_COUNT:
            result._add(
                "warning", "repeated_phrase",
                f"A phrase repeats {count} times; check the transcript for "
                "duplicated speech.")

    repeats = _max_consecutive_segment_repeats(segments or [])
    result.metrics["max_identical_consecutive_segments"] = repeats
    if repeats >= SEGMENT_REPEAT_INVALID:
        result._add(
            "invalid", "repeated_segments",
            f"{repeats} consecutive segments are identical, which indicates a "
            "stuck decoder rather than speech.")

    return result
