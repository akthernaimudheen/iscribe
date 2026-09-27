# -*- coding: utf-8 -*-
"""Clinical-extraction comparison across pipelines.

Feeds each pipeline's ENGLISH output through the EXISTING production clinical
layer (scribe_engine.clinical.build_normalized_entities) to isolate where
clinical meaning survives: speech recognition vs translation vs extraction.
Also runs the Malayalam-native production path (IndicConformer ASR -> ml
semantic layer, no translation) on the control audio's Pipeline-B transcript
as the reference, and the same semantic layer over the ml transcript rendered
as pseudo-English via the ml->en concept dictionary is deliberately NOT done:
the production reference IS the ml semantic layer.

Outputs results/clinical_extraction_comparison.json. No ground truth is
invented; every entity is the extractor's real output on the real transcript.
"""
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
RESULTS = HERE / "results"
OUT = RESULTS / "clinical_extraction_comparison.json"
sys.path.insert(0, str(HERE.parents[1]))


def extract_english(text: str) -> dict:
    from scribe_engine.clinical import build_normalized_entities
    return build_normalized_entities(text)


def extract_ml_reference(ml_text: str) -> dict:
    """Production Malayalam path: semantic layer over the ml transcript."""
    from scribe_engine.semantic import semantic_normalize
    ents = semantic_normalize(ml_text)
    return {
        "layer": "scribe_engine.semantic (ml, production sidecar path)",
        "entities": [
            {"concept": e.concept, "status": e.status, "subject": e.subject,
             "temporality": e.temporality, "conditional": e.conditional,
             "surface_text": e.surface_text, "source_clause": e.source_clause,
             "context": e.context}
            for e in ents
        ],
    }


def compact(d: dict) -> dict:
    """Shrink an extraction result to its meaningful fields (real key names)."""
    if not d:
        return {"entities": []}
    out = {}
    for key in ("entities", "present", "resolved", "absent", "uncertain",
                "questioned", "family_history", "medications_discussed",
                "conditional", "normalized_summary"):
        if key in d and d[key] not in (None, [], ""):
            out[key] = d[key]
    return out


def main() -> int:
    res = {}

    # --- New audio (English speech, confirmed): A1 and C transcripts ---
    a1 = json.loads((RESULTS / "pipeline_a_deepgram.json").read_text(encoding="utf-8"))
    c_new = json.loads((RESULTS / "pipeline_c_whisper_translate.json").read_text(encoding="utf-8"))
    res["new_audio_english_speech"] = {
        "deepgram_A1": compact(extract_english(a1["A1_production_config"]["transcript"])),
        "whisper_C": compact(extract_english(c_new["english_translation"])),
    }

    # --- Control audio (real Malayalam): the three English renderings ---
    a1c = json.loads((RESULTS / "pipeline_a_deepgram_control.json").read_text(encoding="utf-8"))
    cc = json.loads((RESULTS / "pipeline_c_whisper_control.json").read_text(encoding="utf-8"))
    cb = json.loads((RESULTS / "control_b_on_previous_consultation.json").read_text(encoding="utf-8"))
    res["control_audio_malayalam"] = {
        "deepgram_A1_english_model": compact(extract_english(a1c["A1_production_config"]["transcript"])),
        "whisper_C_english_translation": compact(extract_english(cc["english_translation"])),
    }
    res["control_audio_malayalam_reference"] = extract_ml_reference(cb["malayalam_transcript"])

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
