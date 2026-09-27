# -*- coding: utf-8 -*-
"""Pipeline C: direct speech -> English via faster-whisper task=translate.

faster-whisper (already in the project venv, 1.2.1) exposes Whisper's speech
translation natively: task="translate" decodes non-English speech directly to
English text with no separate MT stage. Runs on CPU (int8) to stay inside the
4 GB VRAM constraint; whisper small is the smallest variant with usable
multilingual translation quality.
"""
import io
import json
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
AUDIO = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "East Tippu Sulthan Road 4.m4a"
TAG = sys.argv[2] if len(sys.argv) > 2 else "new"
OUT = HERE / "results" / f"pipeline_c_whisper_{TAG}.json"


def main() -> int:
    from faster_whisper import WhisperModel

    results = {
        "audio": AUDIO.name,
        "pipeline": "faster_whisper_task_translate",
        "asr_model": "small (int8, CPU)",
        "target_language": "en",
        "device": "cpu",
        "errors": [],
    }

    t0 = time.time()
    try:
        model = WhisperModel("small", device="cpu", compute_type="int8")
        results["load_seconds"] = round(time.time() - t0, 2)
    except Exception as exc:
        results["errors"].append(f"load {type(exc).__name__}: {exc}")
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(results, indent=2))
        return 1

    t0 = time.time()
    try:
        segments, info = model.transcribe(str(AUDIO), task="translate", vad_filter=False)
        segs, texts = [], []
        for s in segments:
            segs.append({"start": round(s.start, 2), "end": round(s.end, 2),
                         "text": s.text.strip()})
            texts.append(s.text.strip())
        results["seconds"] = round(time.time() - t0, 2)
        results["detected_language"] = info.language
        results["language_probability"] = round(info.language_probability, 3)
        results["english_translation"] = " ".join(texts)
        results["segments"] = segs
    except Exception as exc:
        results["errors"].append(f"transcribe {type(exc).__name__}: {exc}")

    # A second pass WITHOUT translation to record what Whisper thinks the audio is.
    t0 = time.time()
    try:
        segments, info = model.transcribe(str(AUDIO), task="transcribe", vad_filter=False)
        results["seconds_transcribe"] = round(time.time() - t0, 2)
        results["native_transcription"] = " ".join(s.text.strip() for s in segments)
        results["detected_language_transcribe"] = info.language
    except Exception as exc:
        results["errors"].append(f"transcribe-native {type(exc).__name__}: {exc}")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {k: results[k] for k in ("pipeline", "detected_language",
                                       "language_probability", "seconds",
                                       "seconds_transcribe", "detected_language_transcribe",
                                       "english_translation", "native_transcription",
                                       "errors")}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
