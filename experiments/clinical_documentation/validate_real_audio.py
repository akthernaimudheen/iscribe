# -*- coding: utf-8 -*-
"""Real-audio end-to-end validation of the structured clinical documentation layer.

Runs the COMPLETE production chain on the real Malayalam consultation:

    audio -> sidecar ASR (IndicConformer CTC, CPU)
          -> verified ASR corrections (provenance kept)
          -> semantic extraction (existing layer, untouched)
          -> structured clinical documentation (new layer)
          -> rendered 14-section note

Prints sections A-G required by the briefing. Saves nothing into the repo;
stdout only. Audio never leaves the machine (localhost sidecar).
"""
import io
import json
import sys
import uuid
import urllib.request
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

SIDECAR = "http://127.0.0.1:8131/transcribe"
AUDIO = HERE.parent / "malayalam_speech_translation" / "previous_consultation.m4a"


def transcribe(path: Path) -> dict:
    data = path.read_bytes()
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"file\"; filename=\"{path.name}\"\r\n"
        f"Content-Type: audio/mp4\r\n\r\n"
    ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        SIDECAR, data=body, method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    from scribe_engine.asr_correction import apply_corrections
    from scribe_engine.clinical_documentation import (
        NOT_DOCUMENTED,
        build_structured_documentation,
        render_clinical_documentation,
    )
    from scribe_engine.semantic import semantic_normalize

    result = transcribe(AUDIO)
    raw = result.get("malayalam_text") or result.get("transcript") or ""
    if isinstance(raw, dict):
        raw = raw.get("text") or ""
    corrected = apply_corrections(raw).corrected_text

    entities = semantic_normalize(corrected)
    ents = [e.as_dict() for e in entities]
    # Roles: the sidecar path has no diarization; mirror production honesty.
    turns = [{"speaker": "Speaker 0", "text": corrected}]
    doc = build_structured_documentation(ents, turns, roles_known=False)

    # A. raw Malayalam transcript
    print("A. RAW ASR TRANSCRIPT")
    print(raw)
    print()
    # B. structured clinical representation
    print("B. STRUCTURED CLINICAL REPRESENTATION (per section)")
    for key, facts in doc["sections"].items():
        for f in facts:
            print(f"  [{f['evidence']:>16}] {key}: {f['text'][:80]}"
                  + (f"  <concept={f['concept']} status={f['status']} "
                     f"conf={f['confidence']}>" if f.get("concept") else ""))
    print()
    # C. generated note
    print("C. GENERATED NOTE")
    print(render_clinical_documentation(doc))
    # D. provenance per clinical fact
    print("D. PROVENANCE (concept-derived facts)")
    for f in doc["provenance"]:
        if f.get("concept"):
            print(f"  {f['concept']:<22} status={f.get('status') or '-':<10} "
                  f"conf={f.get('confidence')} speaker={f.get('source_speaker')} "
                  f"source={f.get('source_text', '')[:60]!r}")
    # E. not-documented list
    print()
    print("E. NOT DOCUMENTED (never discussed — not the same as denied)")
    for key, facts in doc["sections"].items():
        if facts and all(f["evidence"] == NOT_DOCUMENTED for f in facts):
            print(f"  - {key}")
    # F. unresolved/missed speech (known ASR-layer items from the evidence probe)
    print()
    print("F. UNRESOLVED / MISSED SPEECH (ASR-layer, previously classified)")
    for form, note in [
        ("ചൊമ്മയുണ്ട്", "colloquial 'chomma' (cough?) — needs human listening"),
        ("കബക്കെട്ട്", "ASR corruption candidate for കഫക്കെട്ട് (congestion)"),
        ("തലവനൊക്കെ", "colloquial 'thalavanokke' (head pain) — needs human listening"),
    ]:
        present = form in (raw or "")
        print(f"  - {form}: {'present in raw transcript' if present else 'not in this ASR run'} ({note})")
    # G. confidence values
    print()
    print("G. CONFIDENCE")
    for e in entities:
        print(f"  {e.concept:<22} {e.status:<10} {e.confidence}")

    out = {"raw": raw, "corrected": corrected,
           "documentation": doc, "note": render_clinical_documentation(doc)}
    Path("_real_audio_doc_result.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n(full JSON in _real_audio_doc_result.json — gitignored scratch)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
