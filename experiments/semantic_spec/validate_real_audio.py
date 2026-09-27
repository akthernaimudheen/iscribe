# -*- coding: utf-8 -*-
"""Real-audio validation of Malayalam Clinical Semantic Specification v1.

Live pipeline: sidecar ASR (IndicConformer CTC, CPU) → verified corrections
(provenance kept) → semantic layer → FACT GRAPH → anti-hallucination validator.

Expected verified concepts for the current recording (human-verified from audio):
    fever → RESOLVED, fatigue → PRESENT, headache → PRESENT,
    cough → PRESENT, congestion (phlegm) → PRESENT,
    difficulty_eating → PRESENT.
Prints concept/status/certainty/surface/source/speaker/confidence per concept,
plus anything missed. Raw transcript printed verbatim and never modified.
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
    from scribe_engine.fact_graph import build_fact_graph, validate_fact_graph
    from scribe_engine.semantic import semantic_normalize

    result = transcribe(AUDIO)
    raw = result.get("malayalam_text") or result.get("transcript") or ""
    if isinstance(raw, dict):
        raw = raw.get("text") or ""

    corr = apply_corrections(raw)
    corrected = corr.corrected_text
    entities = semantic_normalize(corrected)

    # Roles: sidecar ASR has no diarization — speaker stays UNKNOWN honestly.
    turns = [{"speaker": "UNKNOWN", "text": raw}]
    graph = build_fact_graph([e.as_dict() for e in entities], turns,
                             raw_text=raw, roles_known=False)
    report = validate_fact_graph(graph)

    print("RAW ASR TRANSCRIPT (verbatim, source of truth)")
    print(raw)
    print()
    print("VERIFIED CORRECTIONS APPLIED (separate copy, with provenance)")
    for c in corr.corrections:
        print(f"  {c['raw']} -> {c['corrected']}  [{c['type']}, conf={c['confidence']}]")
        print(f"    evidence: {c['reason']}")
    print()
    print("FACTS (concept | status | certainty | confidence | surface | source)")
    for f in graph.facts:
        print(f"  {f['concept']:<20} {f['status']:<10} {f['certainty']:<9} "
              f"{f['confidence']:<5} {f['surface_text']!r}  {f['source_clause'][:48]!r}"
              + (f"  [reason: {f['uncertainty_reason']}]" if f["uncertainty_reason"] else ""))
    print()
    print("SPEAKER: ", "UNKNOWN (sidecar ASR has no diarization — honest label)")
    print()
    print("ANTI-HALLUCINATION VALIDATOR:", "PASS" if report["valid"] else "FAIL")
    for v in report["violations"]:
        print("  violation:", v)

    expected = {"fever": "RESOLVED", "fatigue": "PRESENT", "headache": "PRESENT",
                "cough": "PRESENT", "phlegm": "PRESENT",
                "difficulty_eating": "PRESENT"}
    got = {f["concept"]: f["status"] for f in graph.facts}
    print()
    print("EXPECTED vs GOT")
    for concept, want in expected.items():
        print(f"  {concept:<20} want={want:<10} got={got.get(concept, 'MISSING')}")
    missed = [c for c, w in expected.items() if got.get(c) != w]
    print()
    print("MISSED:", missed or "none")
    print("EXTRA (beyond expected):",
          [c for c in got if c not in expected] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
