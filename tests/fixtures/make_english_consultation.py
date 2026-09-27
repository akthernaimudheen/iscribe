"""Generate a realistic English doctor-patient consultation fixture.

Indian-English TTS voices (edge-tts), two speakers, with the ground-truth script
written alongside the audio. The ground truth is the point: without it there is
no way to tell a faithful transcript from a fluent invention.

Synthetic and non-PHI by construction. TTS is cleaner than clinic audio, so
results from it are an upper bound on real-world quality, not a prediction of it.

    python tests/fixtures/make_english_consultation.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import subprocess
import tempfile

import edge_tts

HERE = pathlib.Path(__file__).parent
OUT_WAV = HERE / "english_consultation.wav"
OUT_TRUTH = HERE / "english_consultation.truth.json"

DOCTOR_VOICE = "en-IN-PrabhatNeural"
PATIENT_VOICE = "en-IN-NeerjaNeural"

# Deliberately loaded with the things that break clinical STT: drug names,
# numbers with units, durations, an interruption, and a spelled-out dosage.
SCRIPT: list[tuple[str, str]] = [
    ("Doctor", "Good morning. Please sit down. What brings you in today?"),
    ("Patient", "Good morning doctor. I have been having a fever and a bad cough for the last three days."),
    ("Doctor", "Any headache or body pain along with that?"),
    ("Patient", "Yes, I have a headache, mostly in the evening. And I vomited once yesterday."),
    ("Doctor", "Are you taking any medication currently?"),
    ("Patient", "I took paracetamol twice, but the fever came back."),
    ("Doctor", "Do you have any known allergies?"),
    ("Patient", "No allergies doctor."),
    ("Doctor", "Any history of diabetes or high blood pressure?"),
    ("Patient", "My blood pressure was slightly high last year, but I am not on any tablets."),
    ("Doctor", "Let me check. Your temperature is one hundred and one degrees."),
    ("Patient", "Is it serious doctor?"),
    ("Doctor", "It looks like a viral infection. I will start you on paracetamol five hundred milligrams, twice a day after food, for five days."),
    ("Patient", "Should I take anything for the cough?"),
    ("Doctor", "Yes, take cetirizine ten milligrams at night. Drink plenty of fluids and come back in one week if the fever does not settle."),
    ("Patient", "Thank you doctor."),
]


async def _speak(text: str, voice: str, dest: pathlib.Path) -> None:
    await edge_tts.Communicate(text, voice).save(str(dest))


async def build() -> None:
    tmp = pathlib.Path(tempfile.mkdtemp())
    parts: list[pathlib.Path] = []
    for i, (speaker, line) in enumerate(SCRIPT):
        voice = DOCTOR_VOICE if speaker == "Doctor" else PATIENT_VOICE
        part = tmp / f"{i:02d}_{speaker}.mp3"
        await _speak(line, voice, part)
        parts.append(part)
        print(f"  {i:02d} {speaker:8} {line[:56]}...")

    # Concatenate, then resample to 16 kHz mono — what every STT engine wants.
    listing = tmp / "parts.txt"
    listing.write_text(
        "\n".join(f"file '{p.as_posix()}'" for p in parts), encoding="utf-8")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
         "-ar", "16000", "-ac", "1", str(OUT_WAV)],
        check=True, capture_output=True,
    )

    truth = {
        "description": "Synthetic English doctor-patient consultation, Indian English TTS voices.",
        "doctor_voice": DOCTOR_VOICE,
        "patient_voice": PATIENT_VOICE,
        "language": "en",
        "phi": False,
        "turns": [{"speaker": s, "text": t} for s, t in SCRIPT],
        "full_text": " ".join(t for _, t in SCRIPT),
        # What a faithful transcript must contain. Checked literally by the
        # accuracy harness; each is a term whose loss or mutation would matter
        # clinically.
        "must_contain": [
            "fever", "cough", "three days", "headache", "vomited",
            "paracetamol", "allergies", "blood pressure",
            "cetirizine", "five hundred", "ten milligrams",
            "five days", "one week", "viral",
        ],
    }
    OUT_TRUTH.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT_WAV}")
    print(f"wrote {OUT_TRUTH}")


if __name__ == "__main__":
    asyncio.run(build())
