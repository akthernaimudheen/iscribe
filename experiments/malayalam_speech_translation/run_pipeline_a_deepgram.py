# -*- coding: utf-8 -*-
"""Pipeline A: the CURRENT baseline (Deepgram) on the real Malayalam audio.

A1 runs the production configuration exactly as the service would (nova-3-medical,
language=en, diarize) through the existing provider class.
A2 probes whether Deepgram can be asked for Malayalam at all (language=ml) --
an empirical question answered by the API, not assumed.

The API key is read from the environment/.env by the provider itself and is
never printed, never written to any artifact.
"""
import io
import json
import os
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "East Tippu Sulthan Road 4.m4a"
TAG = sys.argv[2] if len(sys.argv) > 2 else "new"
OUT = HERE / "results" / f"pipeline_a_deepgram_{TAG}.json"
sys.path.insert(0, str(ROOT))

# Load key presence from .env without printing values (service convention).
for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))


def main() -> int:
    from scribe_engine.stt.deepgram_provider import DeepgramSTTProvider

    results = {"audio": AUDIO.name, "pipeline": "deepgram_current",
               "errors": []}
    provider = DeepgramSTTProvider()

    if not provider.is_configured:
        results["errors"].append("DEEPGRAM_API_KEY absent - Pipeline A not runnable here")
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 1

    # A1: production config (English medical model on Malayalam audio -- what
    # the current routing does when language is not forced to ml).
    t0 = time.time()
    try:
        tr = provider.transcribe(AUDIO, language="en")
        results["A1_production_config"] = {
            "model": provider.model,
            "requested_language": "en",
            "detected_language": tr.language,
            "seconds": round(time.time() - t0, 2),
            "transcript": tr.text,
            "speaker_turns": (tr.provider_meta or {}).get("speaker_turns") or [],
            "diarized": (tr.provider_meta or {}).get("diarized"),
        }
    except Exception as exc:
        results["A1_production_config"] = {"failed": True}
        results["errors"].append(f"A1 {type(exc).__name__}: {exc}")

    # A2: explicit Malayalam request (empirical probe).
    t0 = time.time()
    try:
        tr2 = provider.transcribe(AUDIO, language="ml")
        results["A2_language_ml"] = {
            "model": provider.model,
            "requested_language": "ml",
            "detected_language": tr2.language,
            "seconds": round(time.time() - t0, 2),
            "transcript": tr2.text,
            "speaker_turns": (tr2.provider_meta or {}).get("speaker_turns") or [],
            "diarized": (tr2.provider_meta or {}).get("diarized"),
        }
    except Exception as exc:
        results["A2_language_ml"] = {"failed": True}
        results["errors"].append(f"A2 {type(exc).__name__}: {exc}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {k: results[k] for k in ("audio", "pipeline", "errors")}
    for tag in ("A1_production_config", "A2_language_ml"):
        blk = results.get(tag) or {}
        compact[tag] = {
            "detected_language": blk.get("detected_language"),
            "seconds": blk.get("seconds"),
            "failed": blk.get("failed", False),
            "transcript": blk.get("transcript", "")[:400],
        }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
