# -*- coding: utf-8 -*-
"""Pipeline B: Malayalam ASR (AI4Bharat IndicConformer) -> English translation
(AI4Bharat IndicTrans2), via the project's isolated sidecar process.

The sidecar is the isolated Indic stack already proven on this machine:
  - ASR:  ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large (local .nemo)
  - MT:   ai4bharat/indictrans2-indic-en-dist-200M (local snapshot)
It runs in its own Python 3.12 + NeMo-fork process on loopback; this runner
is production-independent and only reads its HTTP contract.

Output: results/pipeline_b_indic_asr_indictrans2.json
No credentials involved: the sidecar is tokenless and loopback-only.
"""
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
AUDIO = HERE / "East Tippu Sulthan Road 4.m4a"
OUT = HERE / "results" / "pipeline_b_indic_asr_indictrans2.json"
SIDECAR = "http://127.0.0.1:8131"


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    result = {
        "audio": AUDIO.name,
        "pipeline": "indic_asr_indictrans2",
        "asr_model": "ai4bharat/indicconformer_stt_ml_hybrid_ctc_rnnt_large",
        "translation_model": "ai4bharat/indictrans2-indic-en-dist-200M",
        "language": "ml",
        "target_language": "en",
        "processing_time_seconds": 0.0,
        "device": None,
        "malayalam_transcript": None,
        "english_translation": None,
        "errors": [],
    }

    # Readiness gate: report honestly instead of hanging.
    try:
        with urllib.request.urlopen(f"{SIDECAR}/ready", timeout=5) as r:
            ready = json.loads(r.read().decode())
        result["device"] = ready.get("device")
        result["asr_ready"] = ready.get("asr_ready")
        result["mt_ready"] = ready.get("mt_ready")
        if not ready.get("asr_ready"):
            result["errors"].append("sidecar ASR not ready (is_ready=false)")
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 1
    except Exception as exc:
        result["errors"].append(f"sidecar unreachable: {type(exc).__name__}: {exc}")
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    t0 = time.time()
    try:
        boundary = b"----iscribe-experiment-boundary"
        body = io.BytesIO()
        body.write(b"--" + boundary + b"\r\n")
        body.write(
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{AUDIO.name}"\r\nContent-Type: audio/mp4\r\n\r\n'.encode()
        )
        body.write(AUDIO.read_bytes())
        body.write(b"\r\n")
        body.write(b"--" + boundary + b"\r\n")
        body.write(b'Content-Disposition: form-data; name="translate"\r\n\r\ntrue\r\n')
        body.write(b"--" + boundary + b"--\r\n")
        req = urllib.request.Request(
            f"{SIDECAR}/transcribe",
            data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        result["errors"].append(f"sidecar HTTP {exc.code}: {exc.read()[:300]!r}")
        result["processing_time_seconds"] = round(time.time() - t0, 2)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["processing_time_seconds"] = round(time.time() - t0, 2)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    result["processing_time_seconds"] = round(time.time() - t0, 2)
    result["asr_seconds"] = payload.get("asr_seconds")
    result["translation_seconds"] = payload.get("translation_seconds")
    result["decoder"] = payload.get("decoder")
    result["asr_model_revision"] = payload.get("asr_revision")
    # Raw = exactly what the ASR produced. Corrected = verified-corruption layer.
    result["malayalam_transcript"] = payload.get("malayalam_text")
    result["malayalam_transcript_corrected"] = payload.get("corrected_text")
    result["asr_corrections_applied"] = payload.get("asr_corrections") or []
    result["english_translation"] = payload.get("english_text")
    result["translation_error"] = payload.get("translation_error")
    # No diarization in this stack: mark honestly.
    result["speaker_attribution"] = "SPEAKER_UNKNOWN (no diarization in IndicConformer stack)"

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "pipeline", "device", "processing_time_seconds", "asr_seconds",
        "translation_seconds", "errors")}, ensure_ascii=False, indent=2))
    print("--- malayalam (raw) ---")
    print(result["malayalam_transcript"])
    print("--- english ---")
    print(result["english_translation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
