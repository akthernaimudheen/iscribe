"""Generate SYNTHETIC Malayalam/English test fixtures with edge-tts (fixture generation only).

Scripted, synthetic doctor-patient sentences — no real patient data, no PHI.
edge-tts (free, keyless Microsoft cloud TTS) is used ONLY to synthesize these
non-PHI fixtures; it is NOT part of the scribe pipeline and no real or patient
audio ever passes through it. Concatenation + 16 kHz conversion run locally
via ffmpeg.

Usage:  python make_fixtures.py
Output: test_audio/*.mp3 + test_audio/*.wav (16 kHz mono)
"""

import asyncio
import pathlib
import subprocess

import edge_tts

VOICE_DOCTOR = "ml-IN-MidhunNeural"
VOICE_PATIENT = "ml-IN-SobhanaNeural"

# Each fixture: (filename, [(speaker, text), ...])
FIXTURES = [
    ("ml_fever_cough", [
        ("d", "നിങ്ങൾക്ക് എന്ത് പ്രശ്നമാണ് ഉള്ളത്?"),
        ("p", "എനിക്ക് മൂന്ന് ദിവസമായി പനിയും ചുമയും ഉണ്ട്."),
        ("d", "തലവേദനയോ ഛർദ്ദിയോ ഉണ്ടോ?"),
        ("p", "തലവേദനയും ഉണ്ട്, ഇന്നലെ ഛർദ്ദിയും ഉണ്ടായിരുന്നു."),
        ("d", "പാരസിറ്റമോൾ ടാബ്ലറ്റ് രാവിലെ വൈകുന്നേരം കഴിക്കുക."),
    ]),
    ("ml_english_medical_terms", [
        ("d", "നിങ്ങളുടെ blood pressure എത്രയാണ്?"),
        ("p", "എനിക്ക് രണ്ട് വർഷമായി diabetes ഉണ്ട്. ടാബ്ലറ്റ് കഴിക്കുന്നുണ്ട്."),
        ("d", "ഇത് nasal spray ആണ്. രാവിലെ ഉപയോഗിക്കുക."),
        ("p", "ശരി, ഡോക്ടർ."),
    ]),
    ("ml_chest_pain", [
        ("p", "എനിക്ക് നെഞ്ചുവേദനയും ശ്വാസം മുട്ടലും ഉണ്ട്."),
        ("d", "എത്ര നാൾ ആയി?"),
        ("p", "രണ്ട് വർഷമായി diabetes ഉള്ള ആളാണ്."),
        ("d", "ഉടനെ blood test ചെയ്യണം."),
    ]),
    ("manglish_codeswitch", [
        ("p", "Doctor, എനിക്ക് severe headache ആണ്, രണ്ട് ദിവസമായി."),
        ("d", "Allergy ഒന്നുമില്ലേ? Tablet മതിയാകും."),
        ("p", "Allergy ഇല്ല."),
    ]),
    ("en_only_control", [
        ("d", "Good morning, how can I help you today?"),
        ("p", "I have fever and cough for the last three days."),
        ("d", "Any headache or vomiting?"),
        ("p", "I have headache. Take paracetamol tablet twice daily."),
    ]),
]


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], check=True)


async def gen():
    out_dir = pathlib.Path(__file__).parent / "test_audio"
    out_dir.mkdir(exist_ok=True)
    for name, lines in FIXTURES:
        parts = []
        for i, (spk, text) in enumerate(lines):
            part = out_dir / f"_{name}_{i}_{spk}.mp3"
            tts = edge_tts.Communicate(text, VOICE_DOCTOR if spk == "d" else VOICE_PATIENT)
            await tts.save(str(part))
            parts.append(str(part))
        mp3 = out_dir / f"{name}.mp3"
        ffmpeg(" ".join(parts) and "-i", f"concat:{'|'.join(parts)}", "-c", "copy", str(mp3))
        wav = out_dir / f"{name}.wav"
        ffmpeg("-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav))
        for p in parts:
            pathlib.Path(p).unlink()
        print(f"{name}: {mp3.name} + {wav.name} OK")


if __name__ == "__main__":
    asyncio.run(gen())
