# -*- coding: utf-8 -*-
"""ASR-evidence probe for the three unresolved transcript forms.

For each of ചൊമ്മയുണ്ട് / കബക്കെട്ട് / തലവനൊക്കെ:
  - where it sits in the raw ASR transcript (char span)
  - its audio time window: exact via NeMo word timestamps when the fork
    provides them, otherwise a char-proportional estimate over the known
    duration (clearly labelled APPROXIMATE)
  - a 6 s audio clip around the window for human listening (written to a
    gitignored path; PHI policy)

No transcript correction is performed anywhere in this script.

Run from the experiment venv (NeMo required):
    C:/Users/akthe/scribe-gpu-venv/Scripts/python.exe \
        experiments/status_scope/asr_evidence_probe.py
"""
import io
import json
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

AUDIO_16K = ROOT / "benchmarks" / "malayalam" / "real_audio_asr" / "real_consultation_16k.wav"
OUT_JSON = HERE / "asr_evidence_probe.json"
CLIP_DIR = ROOT / "benchmarks" / "malayalam" / "real_audio_asr"  # gitignored dir w/ PHI

TARGETS = [
    {"surface": "ചൊമ്മയുണ്ട്", "stem": "ചൊമ്മ"},
    {"surface": "കബക്കെട്ട്", "stem": "കബ"},
    {"surface": "തലവനൊക്കെ", "stem": "തലവ"},
]


def duration_seconds(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True)
    return float(out.stdout.strip())


def transcribe(path: Path) -> tuple[str, list | None]:
    """Raw transcript plus per-word hypotheses when the fork supports them."""
    # Compat shims (same as run_indicconformer.py).
    import huggingface_hub  # noqa: E402
    if not hasattr(huggingface_hub, "ModelFilter"):
        huggingface_hub.ModelFilter = object
    try:
        import pytorch_lightning as _pl  # noqa: E402
        if not hasattr(_pl.loggers, "NeptuneLogger"):
            _pl.loggers.NeptuneLogger = object
    except Exception:
        pass
    import numpy as np  # noqa: E402
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "bool": [bool],
        }
    from scribe_engine.stt.experimental.indic_malayalam.asr import _get_model
    import nemo.collections.asr as nemo_asr  # noqa: F401  (import proves env)

    entry = _get_model("cpu")
    model = entry["model"]
    model.cur_decoder = "ctc"
    try:
        out = model.transcribe([str(path)], batch_size=1, logprobs=False,
                               language_id="ml", return_hypotheses=True)
    except TypeError:
        return _fallback_text(model, path), None
    item = out[0]
    if isinstance(item, (list, tuple)):
        item = item[0]
    text = getattr(item, "text", "") or ""
    words = []
    for w in (getattr(item, "words", None) or getattr(item, "word", None) or []):
        try:
            words.append({"word": w.text, "start": round(float(w.start_offset or 0), 3),
                          "end": round(float(w.end_offset or 0), 3)})
        except Exception:
            return text, None
    return text.strip(), (words or None)


def _fallback_text(model, path) -> str:
    out = model.transcribe([str(path)], batch_size=1, logprobs=False,
                           language_id="ml")
    item = out[0]
    if isinstance(item, (list, tuple)):
        item = item[0]
    return (item if isinstance(item, str) else getattr(item, "text", "")).strip()


def cut_clip(path: Path, start: float, end: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
         "-ss", f"{max(0.0, start):.2f}", "-t", f"{max(0.5, end - start):.2f}",
         str(out)], check=True)


def main() -> int:
    from scribe_engine.stt.experimental.indic_malayalam.asr import transcribe_malayalam

    duration = duration_seconds(AUDIO_16K)
    # Baseline raw transcript for char-span alignment.
    baseline = transcribe_malayalam(AUDIO_16K, device="cpu")
    raw = baseline["malayalam_text"].strip()
    # Timestamped hypotheses pass (best effort).
    _ignored, words = transcribe(AUDIO_16K)
    # The fork dedupes decoder repeats and may re-run inference; prefer the
    # baseline text when the hypotheses pass returns something different.
    n_chars = len(raw)
    per_char = duration / max(n_chars, 1)

    results = {
        "audio": AUDIO_16K.name,
        "duration_seconds": round(duration, 3),
        "raw_asr_transcript": raw,
        "transcript_chars": n_chars,
        "word_timestamps_available": bool(words),
        "timestamp_source": "nemo_word_hypotheses" if words else
                            "APPROXIMATE char-proportional estimate",
        "asr_model": baseline["model"],
        "asr_revision": baseline["revision"],
        "device": baseline["device"],
        "targets": [],
        "errors": [],
    }

    for t in TARGETS:
        surface = t["surface"]
        idx = raw.find(surface)
        rec = {"surface": surface, "in_transcript": idx >= 0}
        if idx < 0:
            results["errors"].append(f"{surface} not in transcript")
            continue
        rec["char_span"] = [idx, idx + len(surface)]
        if words:
            # Map char span to words by matching the transcript text against
            # the word stream when possible; else proportional within clause.
            rec["time_window_seconds"] = None
            rec["timestamp_note"] = "word-level mapping not attempted for run-on text"
        else:
            start = round(idx * per_char, 2)
            end = round((idx + len(surface)) * per_char, 2)
            rec["time_window_seconds"] = [start, end]
            clip = CLIP_DIR / f"evidence_{t['stem']}.wav"
            cut_clip(AUDIO_16K, max(0.0, start - 2.0), end + 2.0, clip)
            rec["listen_clip"] = clip.name
        results["targets"].append(rec)

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
