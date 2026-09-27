"""Audio input helpers.

The upstream repo has no dedicated audio module: audio handling is implicit in
whisper (`model.transcribe(path)`). We keep the same behavior (whisper-family
models decode via ffmpeg/PyAV internally and accept wav/mp3/m4a/flac/ogg/webm)
and only add explicit validation + duration probing for the demo UI.

Note: faster-whisper decodes with PyAV (bundled), so the ffmpeg *binary* is not
required for plain transcription; it is required if the optional pyannote
diarization backend is enabled.
"""

import subprocess
from pathlib import Path

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}


class AudioError(ValueError):
    """Raised for unsupported or unreadable audio input."""


def validate_audio(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise AudioError(f"Audio file not found: {p}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(
            f"Unsupported audio format '{p.suffix}'. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )
    return p


def probe_duration(path: str | Path) -> float | None:
    """Return duration in seconds using ffprobe, or None if unavailable."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=nw=1:nk=1",
                str(path),
            ],
            capture_output=True, text=True, timeout=30, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None
