# -*- coding: utf-8 -*-
"""Real-audio verification: live ASR -> semantic layer, before/after statuses.

Transcribes the real Malayalam consultation through the running sidecar
(IndicConformer CTC, CPU) and runs the FIXED semantic layer over the raw
transcript. The 'before' column is the documented pre-fix behavior (from the
committed probe baseline); 'after' is computed live. Output: JSON printed to
stdout; nothing written into the repo.
"""
import io
import json
import sys
import uuid
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

SIDECAR = "http://127.0.0.1:8131/transcribe"
AUDIO = HERE.parent / "malayalam_speech_translation" / "previous_consultation.m4a"

BEFORE = {  # measured pre-fix (committed probe baseline)
    "fever": "RESOLVED",
    "fatigue": "RESOLVED",   # <-- the bug
}


def transcribe(path: Path) -> dict:
    data = path.read_bytes()
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: audio/mp4\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        SIDECAR, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    from scribe_engine.asr_correction import apply_corrections
    from scribe_engine.semantic import semantic_normalize

    result = transcribe(AUDIO)
    raw = result.get("malayalam_text") or result.get("transcript") or ""
    if isinstance(raw, dict):
        raw = raw.get("text") or ""
    corrected = apply_corrections(raw).corrected_text

    entities = semantic_normalize(corrected)
    out = {
        "audio": AUDIO.name,
        "asr_seconds": result.get("asr_seconds"),
        "raw_asr_transcript": raw,
        "corrected_transcript": corrected,
        "asr_corrections_applied": result.get("corrections") or [],
        "entities_after_fix": [
            {"concept": e.concept, "status": e.status,
             "temporality": e.temporality, "confidence": e.confidence,
             "source_clause": e.source_clause}
            for e in entities
        ],
        "before_vs_after": {
            c: {"before": BEFORE.get(c), "after": next(
                (e.status for e in entities if e.concept == c), "not extracted")}
            for c in ("fever", "fatigue")
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
