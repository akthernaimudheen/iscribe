# -*- coding: utf-8 -*-
"""Real-audio ASR benchmark: local faster-whisper on the real Malayalam consultation.

Uses the repository's existing provider (scribe_engine.transcription.transcribe)
unchanged. Four configurations: the demo default (base, auto-detect), base
forced-ml, small forced-ml, and large-v3 forced-ml. Raw output is saved verbatim
- no correction, no normalization.
"""
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO_ROOT)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

AUDIO = os.path.join(HERE, "real_consultation_16k.wav")
OUT_JSON = os.path.join(HERE, "faster_whisper.json")

from scribe_engine.transcription import transcribe  # noqa: E402

CONFIGS = [
    {"label": "base_auto", "model_size": "base", "language": None},
    {"label": "base_ml", "model_size": "base", "language": "ml"},
    {"label": "small_ml", "model_size": "small", "language": "ml"},
    {"label": "large_v3_ml", "model_size": "large-v3", "language": "ml"},
]

results = []
for cfg in CONFIGS:
    t0 = time.time()
    try:
        out = transcribe(AUDIO, model_size=cfg["model_size"], device="cpu",
                         compute_type="int8", language=cfg["language"])
        elapsed = round(time.time() - t0, 2)
        text = " ".join(s.text for s in out["segments"]).strip()
        results.append({
            "config": cfg["label"],
            "model_size": cfg["model_size"],
            "requested_language": cfg["language"] or "auto",
            "reported_language": out["language"],
            "asr_time_seconds": elapsed,
            "raw_transcript": text,
            "status": "ok",
        })
        print(f"[{cfg['label']}] {elapsed}s -> {text[:90]}")
    except Exception as e:  # record failures instead of hiding them
        results.append({
            "config": cfg["label"], "model_size": cfg["model_size"],
            "requested_language": cfg["language"] or "auto",
            "status": "failed", "error": str(e),
        })
        print(f"[{cfg['label']}] FAILED: {e}")

with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump({"audio": os.path.basename(AUDIO), "provider": "local faster-whisper",
               "results": results}, fh, ensure_ascii=False, indent=2)
print("saved ->", OUT_JSON)
