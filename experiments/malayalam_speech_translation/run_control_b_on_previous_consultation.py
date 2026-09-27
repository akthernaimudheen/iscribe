# -*- coding: utf-8 -*-
"""Control: run the IDENTICAL Pipeline B stack on the previous consultation
recording (data/uploads/1701bd0281a4.m4a) that produced a good Malayalam
transcript earlier today. Isolates audio-quality vs model hypotheses."""
import io
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

ROOT = Path(__file__).parents[2]
AUDIO = ROOT / "data" / "uploads" / "1701bd0281a4.m4a"
OUT = Path(__file__).parent / "results" / "control_b_on_previous_consultation.json"
SIDECAR = "http://127.0.0.1:8131"


def main() -> int:
    OUT.parent.mkdir(exist_ok=True)
    result = {
        "audio": "data/uploads/1701bd0281a4.m4a (previous consultation, known-good control)",
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
    try:
        with urllib.request.urlopen(f"{SIDECAR}/ready", timeout=5) as r:
            ready = json.loads(r.read().decode())
        result["device"] = ready.get("device")
        if not ready.get("asr_ready"):
            result["errors"].append("sidecar ASR not ready")
            OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            return 1
    except Exception as exc:
        result["errors"].append(f"sidecar unreachable: {type(exc).__name__}")
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    t0 = time.time()
    try:
        boundary = b"----iscribe-experiment-boundary"
        body = io.BytesIO()
        body.write(b"--" + boundary + b"\r\n")
        body.write(
            f'Content-Disposition: form-data; name="file"; filename="{AUDIO.name}"\r\n'
            f"Content-Type: audio/mp4\r\n\r\n".encode())
        body.write(AUDIO.read_bytes())
        body.write(b"\r\n--" + boundary + b"\r\n")
        body.write(b'Content-Disposition: form-data; name="translate"\r\n\r\ntrue\r\n')
        body.write(b"--" + boundary + b"--\r\n")
        req = urllib.request.Request(
            f"{SIDECAR}/transcribe", data=body.getvalue(),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary.decode()}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode())
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["processing_time_seconds"] = round(time.time() - t0, 2)
        OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    result["processing_time_seconds"] = round(time.time() - t0, 2)
    result["asr_seconds"] = payload.get("asr_seconds")
    result["translation_seconds"] = payload.get("translation_seconds")
    result["malayalam_transcript"] = payload.get("malayalam_text")
    result["english_translation"] = payload.get("english_text")
    result["translation_error"] = payload.get("translation_error")
    result["speaker_attribution"] = "SPEAKER_UNKNOWN (no diarization in IndicConformer stack)"
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("processing_time_seconds", "asr_seconds",
                                             "translation_seconds", "errors")},
                     ensure_ascii=False, indent=2))
    print("--- malayalam (raw) ---")
    print(result["malayalam_transcript"])
    print("--- english ---")
    print(result["english_translation"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
