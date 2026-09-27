"""Compare an STT provider's transcript against known ground truth.

Answers the only question that matters for the demo: does the transcript say
what was actually spoken? Reports word error rate, which required clinical terms
survived, and what was invented.

    python scripts/transcription_check.py --provider current_faster_whisper
    python scripts/transcription_check.py --provider deepgram

Synthetic, non-PHI fixture only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIXTURE = ROOT / "tests" / "fixtures" / "english_consultation.wav"
TRUTH = ROOT / "tests" / "fixtures" / "english_consultation.truth.json"


# Spoken-vs-formatted equivalents. Deepgram's smart formatting writes "3 days"
# where whisper writes "three days"; both are faithful to the audio. Without
# this mapping the WER punishes formatting and the two providers cannot be
# compared on transcription accuracy, which is what actually matters clinically.
_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000",
}
_UNIT_WORDS = {"milligram": "mg", "milligrams": "mg", "mgs": "mg"}


def normalise(text: str) -> list[str]:
    """Lowercase, strip punctuation, fold number and unit spellings."""
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    folded: list[str] = []
    for tok in tokens:
        tok = _NUMBER_WORDS.get(tok, tok)
        tok = _UNIT_WORDS.get(tok, tok)
        folded.append(tok)
    # "five hundred" -> "500" so it matches the formatted "500".
    out: list[str] = []
    i = 0
    while i < len(folded):
        if (i + 1 < len(folded) and folded[i].isdigit()
                and folded[i + 1] in ("100", "1000")):
            out.append(str(int(folded[i]) * int(folded[i + 1])))
            i += 2
            continue
        out.append(folded[i])
        i += 1
    return out


def wer(reference: list[str], hypothesis: list[str]) -> tuple[float, dict]:
    """Levenshtein word error rate with a substitution/insertion/deletion split."""
    n, m = len(reference), len(hypothesis)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)

    # Backtrack for the error breakdown.
    i, j = n, m
    sub = ins = dele = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (
                0 if reference[i - 1] == hypothesis[j - 1] else 1):
            if reference[i - 1] != hypothesis[j - 1]:
                sub += 1
            i, j = i - 1, j - 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            ins += 1
            j -= 1
        else:
            dele += 1
            i -= 1
    return (d[n][m] / max(n, 1)), {"substitutions": sub, "insertions": ins,
                                   "deletions": dele, "reference_words": n,
                                   "hypothesis_words": m}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default=None,
                    help="current_faster_whisper | deepgram (default: production priority)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model", default="base")
    ap.add_argument("--audio", default=str(FIXTURE))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    truth = json.loads(TRUTH.read_text(encoding="utf-8"))
    audio = pathlib.Path(args.audio)
    if not audio.exists():
        print(f"fixture missing: {audio}")
        return 1

    from scribe_engine import ScribeEngine

    engine = ScribeEngine(
        whisper_model_size=args.model,
        device=args.device,
        stt_provider_id=args.provider,
        language="en",
    )
    t0 = time.perf_counter()
    result = engine.process_audio(audio)
    elapsed = round(time.perf_counter() - t0, 2)

    transcript = result["transcript"]["text"]
    meta = result["meta"]
    validation = result["validation"]

    ref = normalise(truth["full_text"])
    hyp = normalise(transcript)
    rate, breakdown = wer(ref, hyp)

    # Compare clinical terms on the folded token stream too, so "3 days" and
    # "three days" both satisfy the same requirement.
    low = transcript.lower()
    folded = " ".join(normalise(transcript))

    def _present(term: str) -> bool:
        return term.lower() in low or " ".join(normalise(term)) in folded

    present = [t for t in truth["must_contain"] if _present(t)]
    missing = [t for t in truth["must_contain"] if not _present(t)]

    # Groups accept equivalent renderings ("five hundred" vs "500"), so a miss
    # here is a genuine transcription error rather than a formatting choice.
    for group in truth.get("must_contain_any", []):
        label = group[0]
        if any(_present(variant) for variant in group):
            present.append(label)
        else:
            missing.append(label)
    total_terms = len(truth["must_contain"]) + len(truth.get("must_contain_any", []))

    # Words in the transcript that appear nowhere in the ground truth.
    ref_set = set(ref)
    invented = sorted({w for w in hyp if w not in ref_set and len(w) > 3})

    report = {
        "provider": meta.get("stt_provider"),
        "provider_selection": meta.get("provider_selection"),
        "requested_language": meta.get("requested_language"),
        "reported_language": meta.get("reported_language"),
        "audio_duration_s": meta.get("audio_duration_s"),
        "processing_seconds": elapsed,
        "realtime_factor": round((meta.get("audio_duration_s") or 0) / elapsed, 2) if elapsed else None,
        "validation_severity": validation.get("severity"),
        "validation_codes": validation.get("codes"),
        "word_error_rate": round(rate, 4),
        "accuracy_pct": round((1 - rate) * 100, 2),
        **breakdown,
        "segments": len(result["transcript"]["segments"]),
        "speaker_method": result["speakers"]["method"],
        "roles_known": result["speakers"]["roles_known"],
        "clinical_terms_present": present,
        "clinical_terms_missing": missing,
        "clinical_terms_score": f"{len(present)}/{total_terms}",
        "words_not_in_ground_truth": invented[:40],
        "transcript": transcript,
    }

    print("=" * 72)
    print(f"PROVIDER      : {report['provider']}  ({report['provider_selection']})")
    print(f"LANGUAGE      : requested={report['requested_language']} "
          f"reported={report['reported_language']}")
    print(f"TIMING        : {elapsed}s for {report['audio_duration_s']}s audio "
          f"(RTF {report['realtime_factor']})")
    print(f"VALIDATION    : {report['validation_severity']} {report['validation_codes']}")
    print(f"WORD ACCURACY : {report['accuracy_pct']}%  (WER {report['word_error_rate']})")
    print(f"                sub={breakdown['substitutions']} "
          f"ins={breakdown['insertions']} del={breakdown['deletions']} "
          f"ref_words={breakdown['reference_words']}")
    print(f"CLINICAL TERMS: {report['clinical_terms_score']}")
    if missing:
        print(f"  MISSING     : {missing}")
    print(f"SPEAKERS      : {report['speaker_method']} "
          f"(roles_known={report['roles_known']})")
    if invented:
        print(f"NOT IN TRUTH  : {invented[:20]}")
    print("=" * 72)
    print("TRANSCRIPT:")
    print(transcript)
    print("=" * 72)

    if args.out:
        pathlib.Path(args.out).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
