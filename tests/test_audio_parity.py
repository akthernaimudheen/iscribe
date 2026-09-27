"""Audio parity check: ScribeEngine.process_audio vs. the direct upstream-code probe.

The probe (_pipeline_probe_result.txt, produced by running upstream nlp.py logic +
faster-whisper directly on the sample audio) is the reference. The engine must
reproduce the useful content, with the single documented deviation: the
patient-first label-phase fix in diarization (which, by design, changes which
lines count as patient lines — that is the point of the fix).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scribe_engine import ScribeEngine


def main() -> int:
    audio = "upstream-scribe/Allergy final.mp3"
    engine = ScribeEngine()
    result = engine.process_audio(audio)

    t = result["transcript"]
    print(f"language={t['language']} segments={len(t['segments'])}")
    print(f"speakers: method={result['speakers']['method']} conf={result['speakers']['confidence']}")
    print(f"turns={len(result['speakers']['turns'])}")
    print("--- transcript head ---")
    print(t["text"][:400])
    print("--- note fields ---")
    for k, v in result["clinical_note"]["fields"].items():
        if isinstance(v, (str, int, float, list)):
            print(f"{k}: {v}")
    print("--- rx meds ---")
    print([m["name"] for m in result["prescription"]["fields"]["medications"]])

    # Parity assertions
    assert t["language"] == "en"
    assert len(t["segments"]) == 22, f"expected 22 segments like the probe, got {len(t['segments'])}"
    assert result["speakers"]["method"] == "alternating"
    # Documented deviation (diarization patient-first fix): the opening symptom
    # line is now labeled Patient, so patient-only symptom detection works —
    # the probe (verbatim upstream labeling) missed 'asthma' entirely.
    assert result["speakers"]["turns"][0]["speaker"] == "Patient"
    assert result["clinical_note"]["fields"]["symptoms_reported"] == "asthma"
    meds = [m["name"] for m in result["prescription"]["fields"]["medications"]]
    assert "allegra" in meds  # upstream lowercases before matching
    # The 'spray' hit from the probe came from a line the label-phase fix now
    # correctly assigns to the patient, so it legitimately drops out:
    assert "spray" not in meds
    # Structured NLP still finds no diagnosis on this noisy audio (upstream
    # weakness, honestly shown):
    assert result["clinical_note"]["fields"]["diagnoses"] == []
    print("PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
