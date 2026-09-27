# -*- coding: utf-8 -*-
"""Deterministic note generator — real-audio validation.

Live pipeline: sidecar ASR (IndicConformer CTC, CPU) → verified corrections
→ semantic layer → FACT GRAPH → deterministic note renderer →
validate_note_against_fact_graph. Saves fact graph + note JSON artifacts
(gitignored) and prints the rendered note.

Expected verified concepts for the current recording (human-verified):
fever RESOLVED; fatigue/headache/cough/congestion/difficulty_eating PRESENT.
PMH/PSH/medications/allergies/family/social/exam/assessment/plan must stay
NOT_DOCUMENTED — this consultation documents none of them.
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
AUDIO = ROOT / "experiments" / "malayalam_speech_translation" / "previous_consultation.m4a"
GRAPH_OUT = HERE / "real_audio_fact_graph.json"
NOTE_OUT = HERE / "real_audio_note.json"


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
    from scribe_engine.note_generator import (render_deterministic_note,
                                              validate_note_against_fact_graph)
    from scribe_engine.semantic import semantic_normalize

    result = transcribe(AUDIO)
    raw = result.get("malayalam_text") or result.get("transcript") or ""
    if isinstance(raw, dict):
        raw = raw.get("text") or ""

    corr = apply_corrections(raw)
    entities = semantic_normalize(corr.corrected_text)
    # Sidecar ASR has no diarization — speaker stays UNKNOWN (honest label).
    turns = [{"speaker": "UNKNOWN", "text": raw}]
    graph = build_fact_graph([e.as_dict() for e in entities], turns,
                             raw_text=raw, roles_known=False)
    graph_report = validate_fact_graph(graph)

    note = render_deterministic_note(graph)
    note_report = validate_note_against_fact_graph(note["note"], graph)

    GRAPH_OUT.write_text(json.dumps(graph.as_dict(), ensure_ascii=False, indent=1),
                         encoding="utf-8")
    NOTE_OUT.write_text(json.dumps(note, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print("RAW ASR TRANSCRIPT (verbatim, unchanged):")
    print(raw)
    print()
    print("CORRECTIONS APPLIED (separate copy, provenance kept):")
    for c in corr.corrections:
        print(f"  {c['raw']} -> {c['corrected']} [{c['type']}]")
    print()
    print("FACT GRAPH:", "PASS" if graph_report["valid"] else "FAIL",
          f"({len(graph.facts)} facts)")
    for v_ in graph_report["violations"]:
        print("  graph violation:", v_)
    print("NOTE VALIDATOR:", "PASS" if note_report["valid"] else "FAIL")
    for v_ in note_report["violations"]:
        print("  note violation:", v_)
    print()
    print("=== GENERATED NOTE ===")
    print(note["note"])
    print()

    expected = {"fever": "RESOLVED", "fatigue": "PRESENT", "headache": "PRESENT",
                "cough": "PRESENT", "phlegm": "PRESENT",
                "difficulty_eating": "PRESENT"}
    got = {f["concept"]: f["status"] for f in graph.facts}
    ok = all(got.get(c) == s for c, s in expected.items()) and \
        not [c for c in got if c not in expected] and \
        graph_report["valid"] and note_report["valid"]
    print("EXPECTED vs GOT:", "MATCH" if all(got.get(c) == s for c, s in expected.items()) else "MISMATCH")
    for c, s in expected.items():
        print(f"  {c:<20} want={s:<10} got={got.get(c, 'MISSING')}")
    print("EXTRA:", [c for c in got if c not in expected] or "none")
    print("OVERALL:", "SUCCESS" if ok else "REVIEW NEEDED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
