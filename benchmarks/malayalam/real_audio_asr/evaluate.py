# -*- coding: utf-8 -*-
"""Evaluation over the three raw transcripts: REAL audio, no human reference.

No WER is computable (no ground-truth transcript for the real consultation).
The evaluation is therefore structural and clinical-content based:
  - Malayalam script share (garbage detection proxy)
  - Devanagari/Latin noise tokens (multi-script soup)
  - code-switched English loanwords captured
  - known clinical terms present in the audio (fever, fatigue, head pain,
    cough/phlegm, eating difficulty), counted from BOTH transcripts' evidence
  - obvious garbage (Latin/Devanagari soup, fluent unrelated English)
Output feeds real_audio_asr_report.md; raw transcripts are never modified.
"""
import glob
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Clinical content attested by BOTH IndicConformer and faster-whisper-small
# independently recognizing related material, plus the typed demo reference
# (consultation 026358949f29: fever, bad cough, body pain, headache).
CLINICAL_MARKERS = {
    "fever": ["പനി"],
    "fatigue": ["ക്ഷീണ"],
    "head pain": ["തലവേദ", "തലവേദന", "തല വേദ"],
    "cough/phlegm": ["ചുമ", "കഫം", "കബ", "ചൊമ്മ"],
    "eating difficulty": ["കഴിക്കാൻ പറ്റ", "ഫുഡ്", "കഴിക്കാൻ"],
    "greeting (code-switch)": ["ഗുഡ് മോണിം", "good morning"],
    "sir (code-switch)": ["സർ", "sir"],
}

ML = re.compile(r"[\u0D00-\u0D7F]")
DEV = re.compile(r"[\u0900-\u097F]")
LATIN = re.compile(r"[A-Za-z]")


def stats(text: str) -> dict:
    words = [w for w in re.split(r"\s+", text) if w]
    if not words:
        return {"words": 0}
    ml = sum(bool(ML.search(w)) for w in words)
    dev = sum(bool(DEV.search(w)) for w in words)
    latin = sum(bool(LATIN.fullmatch(w)) for w in words)
    return {
        "words": len(words),
        "malayalam_script_words": ml,
        "malayalam_share": round(ml / len(words), 3),
        "devanagari_words": dev,
        "latin_words": latin,
    }


def clinical_hits(text: str) -> dict:
    return {k: [m for m in ms if m in text]
            for k, ms in CLINICAL_MARKERS.items()}


rows = {}
for path in sorted(glob.glob(os.path.join(HERE, "*.json"))):
    if os.path.basename(path) == "evaluation.json":
        continue
    data = json.load(open(path, encoding="utf-8"))
    entries = data.get("results", [data]) if isinstance(data, dict) else []
    for entry in entries:
        sub = entry.get("config", entry.get("attempt", ""))
        label = entry.get("provider", data.get("provider", "?")) + (f":{sub}" if sub else "")
        text = entry.get("raw_transcript") or ""
        rows[label] = {
            "status": entry.get("status") or ("ok" if text else "no-transcript"),
            "asr_time_seconds": entry.get("asr_time_seconds"),
            "stats": stats(text),
            "clinical": clinical_hits(text),
            "clinical_terms_hit": sum(1 for v in clinical_hits(text).values() if v),
            "transcript": text,
        }

with open(os.path.join(HERE, "evaluation.json"), "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=2)

for label, r in rows.items():
    s = r["stats"]
    if r["status"] != "ok":
        print(f"{label:28} {str(r['status']):8}")
        continue
    print(f"{label:28} {r['status']:8} ml-share={s.get('malayalam_share')} "
          f"dev={s.get('devanagari_words')} latin={s.get('latin_words')} "
          f"clinical={r['clinical_terms_hit']}/7")
print("saved -> evaluation.json")
