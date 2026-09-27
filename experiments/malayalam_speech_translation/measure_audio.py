# -*- coding: utf-8 -*-
"""Objective audio-condition measurement: peak/RMS dBFS and silence share.

Distinguishes 'the model failed' from 'the recording is bad' before any
pipeline is blamed for garbage output. ffmpeg is already a project dependency.
"""
import io
import json
import subprocess
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
FILES = [HERE / "East Tippu Sulthan Road 4.m4a",
         HERE / "previous_consultation.m4a"]


def measure(path: Path) -> dict:
    out = {"file": path.name, "exists": path.exists()}
    if not path.exists():
        return out
    try:
        astats = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path),
             "-af", "astats=metadata=1:measure_overall=RMS_level+Peak_level:measure_perchannel=none",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        stats = {}
        for line in astats.stderr.splitlines():
            line = line.strip()
            if "RMS level dB:" in line:
                stats["rms_db"] = float(line.split(":")[1])
            elif "Peak level dB:" in line:
                stats["peak_db"] = float(line.split(":")[1])
        silencedetect = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(path),
             "-af", "silencedetect=noise=-40dB:d=0.8", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60)
        sil_total, dur = 0.0, None
        for line in silencedetect.stderr.splitlines():
            if "silence_duration" in line:
                sil_total += float(line.split("silence_duration:")[1])
            elif "Duration:" in line and dur is None:
                dur = float(line.split("Duration:")[1].strip().split(",")[0]) if False else dur
        out.update(stats)
        out["silence_seconds_ge0.8s_-40dB"] = round(sil_total, 2)
        if dur:
            out["silence_share"] = round(sil_total / dur, 3)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    import io
    print(json.dumps([measure(p) for p in FILES], ensure_ascii=False, indent=2))
