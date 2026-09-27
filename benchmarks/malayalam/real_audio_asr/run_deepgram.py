# -*- coding: utf-8 -*-
"""Real-audio ASR benchmark: Deepgram (existing repo provider) on the real audio.

Uses the repo's Deepgram path as-is. If DEEPGRAM_API_KEY is absent or the call
fails, the failure is recorded - not hidden.
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
OUT_JSON = os.path.join(HERE, "deepgram.json")

candidates = []
try:
    from scribe_engine.stt.deepgram_provider import DeepgramSTTProvider
    provider = DeepgramSTTProvider()
    if provider.is_configured:
        candidates.append(("DeepgramSTTProvider.transcribe",
                           lambda p: {"text": provider.transcribe(p).text,
                                      "language": provider.transcribe(p).language}))
    else:
        print("DeepgramSTTProvider present but not configured (no DEEPGRAM_API_KEY)")
except Exception as e:
    print(f"no repo deepgram provider importable: {e}")

import shutil  # noqa: E402

print("deepgram cli on PATH:", bool(shutil.which("deepgram")))
print("DEEPGRAM_API_KEY set:", bool(os.environ.get("DEEPGRAM_API_KEY")))


def call(fn, path):
    t0 = time.time()
    try:
        r = fn(path)
        if isinstance(r, dict):
            text = r.get("text") or r.get("transcript") or ""
            lang = r.get("language")
        elif isinstance(r, str):
            text, lang = r, None
        else:
            text, lang = str(r), None
        return {"status": "ok", "asr_time_seconds": round(time.time() - t0, 2),
                "reported_language": lang, "raw_transcript": text.strip()}
    except Exception as e:
        return {"status": "failed", "error": str(e)[:300]}


results = []
for name, fn in candidates:
    for attempt in (1, 2):
        r = call(fn, AUDIO)
        r["provider_fn"] = name
        r["attempt"] = attempt
        results.append(r)
        print(f"[{name} attempt {attempt}] {r['status']}: "
              f"{r.get('raw_transcript', r.get('error', ''))[:120]}")
        if r["status"] == "ok":
            break

with open(OUT_JSON, "w", encoding="utf-8") as fh:
    json.dump({"audio": os.path.basename(AUDIO), "provider": "deepgram",
               "results": results}, fh, ensure_ascii=False, indent=2)
print("saved ->", OUT_JSON)
