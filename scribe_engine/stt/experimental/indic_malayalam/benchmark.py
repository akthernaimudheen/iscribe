"""Benchmark: Pipeline A (CurrentSTTProvider) vs Pipeline B (MalayalamIndicProvider).

Run from the ISOLATED experiment environment (see README.md), NOT the demo venv:

    python -m scribe_engine.stt.experimental.indic_malayalam.benchmark audio1.wav audio2.wav \
        --device cpu --out benchmark_results.json

Honesty rules enforced here:
- Only measured values are written; anything not measurable is recorded as "unavailable".
- No accuracy percentages. The terminology test reports factual term-presence
  counts (case-insensitive substring match against the English output), which is
  an observation, not a quality score.
- The Malayalam source transcript is always preserved in the output.
"""

import argparse
import json
import platform
import time
from datetime import datetime

import psutil

from scribe_engine.stt.current_provider import CurrentSTTProvider
from scribe_engine.stt.experimental.indic_malayalam.provider import MalayalamIndicProvider

# English medical terms the test fixtures are designed to contain after
# translation. Case-insensitive substring presence; NOT semantic accuracy.
TERMINOLOGY_CHECKLIST = [
    "fever", "headache", "cough", "vomit", "chest pain", "breath",
    "blood pressure", "diabetes", "tablet", "paracetamol",
    "three days", "five days", "two years", "morning", "evening",
]

_proc = psutil.Process()


def _rss_mb():
    try:
        return round(_proc.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _cpu_seconds():
    try:
        t = _proc.cpu_times()
        return round(t.user + t.system, 2)
    except Exception:
        return None


def _gpu_reset(device):
    """Zero the CUDA peak-memory counters so per-stage VRAM is attributable."""
    if device != "cuda":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _gpu_stats(device):
    """Peak VRAM for the work since the last reset. 'unavailable' off-CUDA."""
    if device != "cuda":
        return "not measured (CPU run)"
    try:
        import torch
        if not torch.cuda.is_available():
            return "cuda requested but unavailable"
        return {
            "name": torch.cuda.get_device_name(0),
            "vram_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 1048576, 1),
            "vram_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 1048576, 1),
            "vram_total_mib": round(
                torch.cuda.get_device_properties(0).total_memory / 1048576, 1),
        }
    except Exception as e:
        return f"unavailable ({type(e).__name__})"


def run_pipeline_a(path):
    rec = {"asr_time_seconds": None, "translation_time_seconds": 0.0, "transcript": None}
    p = CurrentSTTProvider()
    t0 = time.time()
    tr = p.transcribe(path)
    rec["asr_time_seconds"] = round(time.time() - t0, 2)
    rec["transcript"] = tr.text
    rec["language_detected"] = tr.language
    return rec


def run_pipeline_b(path, device="cpu"):
    rec = {"asr_time_seconds": None, "translation_time_seconds": None}
    p = MalayalamIndicProvider(device=device)
    t0 = time.time()
    tr = p.transcribe(path)
    total = round(time.time() - t0, 2)
    rec["asr_time_seconds"] = tr.provider_meta.get("asr_time_seconds")
    rec["translation_time_seconds"] = tr.provider_meta.get("translation_time_seconds")
    rec["total_time_seconds"] = total
    # Both languages are kept: the Malayalam source is the clinical record of
    # what was actually said; the English is the working text.
    rec["transcript"] = tr.source_transcript
    rec["english_translation"] = tr.translated_transcript
    rec["asr_model"] = tr.provider_meta.get("asr_model")
    rec["mt_model"] = tr.provider_meta.get("mt_model")
    # Revisions are recorded per fixture so a result can always be traced back
    # to the exact weights that produced it.
    try:
        from .asr import MODEL_REVISION as ASR_REV
        from .translation import MODEL_REVISION as MT_REV
        rec["asr_revision"] = ASR_REV or "unavailable"
        rec["mt_revision"] = MT_REV or "unavailable"
    except Exception:
        rec["asr_revision"] = rec["mt_revision"] = "unavailable"
    return rec


def terminology_check(english_text):
    if not english_text:
        return {"present": [], "missing": TERMINOLOGY_CHECKLIST, "present_count": 0}
    low = english_text.lower()
    present = [t for t in TERMINOLOGY_CHECKLIST if t in low]
    missing = [t for t in TERMINOLOGY_CHECKLIST if t not in low]
    return {"present": present, "missing": missing, "present_count": len(present)}


def benchmark_audio(path, device="cpu"):
    import subprocess

    def duration():
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=nw=1:nk=1", str(path)],
                capture_output=True, text=True, timeout=30, check=True,
            )
            return round(float(out.stdout.strip()), 2)
        except Exception:
            return "unavailable"

    entry = {
        "audio": str(path),
        "duration_seconds": duration(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "device": device,
        "python": platform.python_version(),
        "results": {"pipeline_A_current_faster_whisper": {}, "pipeline_B_indic_malayalam": {}},
    }

    # ---- Pipeline A ----
    try:
        rss0, cpu0 = _rss_mb(), _cpu_seconds()
        a = run_pipeline_a(path)
        a["peak_rss_delta_mb"] = "unavailable" if _rss_mb() is None or rss0 is None \
            else round((_rss_mb() or 0) - rss0, 1)
        a["cpu_seconds"] = "unavailable" if cpu0 is None else round(
            (_cpu_seconds() or 0) - cpu0, 2)
        a["english_translation"] = a["transcript"]
        a["terminology_presence"] = terminology_check(a["english_translation"])
        entry["results"]["pipeline_A_current_faster_whisper"] = a
    except Exception as e:
        entry["results"]["pipeline_A_current_faster_whisper"] = {"errors": [f"{type(e).__name__}: {e}"]}

    # ---- Pipeline B ----
    try:
        rss0, cpu0 = _rss_mb(), _cpu_seconds()
        _gpu_reset(device)
        b = run_pipeline_b(path, device=device)
        b["peak_rss_delta_mb"] = "unavailable" if _rss_mb() is None or rss0 is None \
            else round((_rss_mb() or 0) - rss0, 1)
        b["cpu_seconds"] = "unavailable" if cpu0 is None else round(
            (_cpu_seconds() or 0) - cpu0, 2)
        b["rss_mb"] = _rss_mb()
        b["gpu"] = _gpu_stats(device)
        b["terminology_presence"] = terminology_check(b.get("english_translation"))
        entry["results"]["pipeline_B_indic_malayalam"] = b
    except Exception as e:
        # A failure here is a real result, not a reason to abort the run: the
        # other fixtures still carry usable evidence.
        entry["results"]["pipeline_B_indic_malayalam"] = {
            "errors": [f"{type(e).__name__}: {e}"],
            "gpu": _gpu_stats(device),
        }

    # ---- realtime factors (wall-clock total / audio duration) ----
    for key, res in entry["results"].items():
        total = res.get("total_time_seconds")
        if total is None:
            t_asr = res.get("asr_time_seconds")
            total = t_asr if t_asr is not None else None
            if total is not None:
                res["total_time_seconds"] = total
        d = entry["duration_seconds"]
        if isinstance(total, (int, float)) and isinstance(d, (int, float)) and d > 0:
            res["realtime_factor"] = round(total / d, 2)
        else:
            res["realtime_factor"] = "unavailable"
        res.setdefault("cpu_percent", "unavailable")
        res.setdefault("gpu", "not measured (CPU run)" if device == "cpu" else "unavailable")

    return entry


def main():
    ap = argparse.ArgumentParser(description="Benchmark current STT vs Malayalam Indic pipeline")
    ap.add_argument("audio", nargs="+", help="Audio files (wav/mp3, 16 kHz preferred)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--out", default="benchmark_results.json")
    args = ap.parse_args()

    report = {
        "meta": {
            "generated": datetime.now().isoformat(timespec="seconds"),
            "device": args.device,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "note": "Pipeline A = CurrentSTTProvider (faster-whisper base). "
                    "Pipeline B = IndicConformer ml ASR + IndicTrans2 ml->en. "
                    "Terminology test = term presence counts, NOT accuracy.",
        },
        "audio": [benchmark_audio(p, args.device) for p in args.audio],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"wrote {args.out}")
    for entry in report["audio"]:
        print(f"\n=== {entry['audio']} ({entry['duration_seconds']}s) ===")
        for pk, res in entry["results"].items():
            if "errors" in res:
                print(f"  {pk}: ERROR {res['errors']}")
                continue
            print(f"  {pk}: total={res.get('total_time_seconds')}s rtf={res.get('realtime_factor')} "
                  f"terms={res.get('terminology_presence', {}).get('present_count')}/{len(TERMINOLOGY_CHECKLIST)}")


if __name__ == "__main__":
    main()
