"""Clinical Intelligence V2 — failure trace (Task 2 evidence).

Reads the frozen real-case transcripts and answers, for every wrong fact the
current pipeline produced, *which rule created it*:

  * per-mention extraction (normalization.split_clauses / _concept_matches /
    _status_for) — the evidence and status of each mention;
  * cross-clause merging — how mentions of one concept collapse into a single
    entity (the temporal-collapse failure);
  * the v1 note renderer's line-scanning rules (_assessment / _plan /
    _physical_exam / _pmh) — which pattern matched which line.

Run:  ./.venv/Scripts/python.exe experiments/clinical_intelligence_v2/trace_failures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scribe_engine import clinical_documentation as CD          # noqa: E402
from scribe_engine import normalization as N                    # noqa: E402
from scribe_engine.note_generator import (_assessment, _physical_exam,  # noqa: E402
                                          _plan, _pmh)
from scribe_engine.semantic import _merged_lexicon, semantic_normalize  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "english_real_cases"

CASES = {
    "referral_letter_orthopaedic": "orthopaedic referral letter",
    "medicolegal_jones": "medico-legal report (Jones)",
    "medicolegal_finton": "medico-legal report (Finton)",
}


def mentions(text: str) -> list[dict]:
    """Per-clause, per-mention records — the layer V2 rebuilds on."""
    lex = _merged_lexicon()
    out: list[dict] = []
    for clause in N.split_clauses(text, lex):
        cn = N._norm(clause)
        question = N._is_question(cn, lex)
        duration, _v, _u = N._extract_duration(cn, lex)
        for start, end, cid, surface in N._concept_matches(cn, lex):
            status, confidence, reason = N._status_for(cn, start, end, lex, question)
            out.append({
                "clause": clause.strip(),
                "concept": cid,
                "surface": surface,
                "status": status,
                "confidence": confidence,
                "question": question,
                "conditional": N._is_conditional(cn, lex),
                "resolution": N._is_resolution(cn, lex),
                "temporality": N._temporality(cn, lex)[0],
                "duration": duration,
                "uncertainty_reason": reason,
            })
    return out


def merged_entities(text: str) -> list[dict]:
    """What the pipeline actually consumes (post-merge)."""
    return [e.as_dict() for e in semantic_normalize(text)]


def v1_rules(graph: dict) -> dict:
    """Which line-scanning rule produced each v1 non-clinical section."""
    from scribe_engine.fact_graph import build_fact_graph
    fg = build_fact_graph(graph.get("entities", []), graph.get("turns", []),
                          raw_text=graph.get("raw_text", ""),
                          roles_known=graph.get("roles_known", True))
    assessment = _assessment(fg)
    hits = []
    if assessment:
        for t in fg.turns:
            text = t.get("text") or ""
            for p in CD._DIAGNOSIS_PAT_EN:
                if __import__("re").search(p, text, __import__("re").I):
                    hits.append({"pattern": p, "line": text})
    return {
        "assessment": assessment,
        "diagnosis_pattern_hits": hits,
        "plan": _plan(fg),
        "physical_exam": _physical_exam(fg),
        "pmh": _pmh(graph.get("facts") or []),
    }


def main() -> int:
    report = {}
    for name, label in CASES.items():
        path = FIXTURES / f"{name}.txt"
        text = path.read_text(encoding="utf-8")
        # process_text applies diarization to unlabelled text exactly as the
        # service does for a pasted transcript.
        from scribe_engine import ScribeEngine
        result = ScribeEngine().process_text(text, language="en")
        graph = {
            "entities": (result.get("normalized_clinical_entities") or {}).get("entities", []),
            "turns": (result.get("speakers") or {}).get("turns", []),
            "raw_text": text,
            "roles_known": (result.get("speakers") or {}).get("roles_known", True),
        }
        entry = {
            "label": label,
            "chars": len(text),
            "mentions": mentions(text),
            "merged_entities": merged_entities(text),
            "v1_sections": v1_rules(graph),
            "v1_note": (result.get("clinical_note_v2") or {}).get("note", ""),
        }
        report[name] = entry

        print("=" * 78)
        print(f"{name} — {label}")
        print(f"\n-- per-mention extraction ({len(entry['mentions'])} mentions) --")
        for m in entry["mentions"]:
            print(f"   {m['concept']:<22} {m['status']:<9} conf={m['confidence']:<5} "
                  f"surf={m['surface']!r}\n        clause: {m['clause'][:150]!r}")
        print(f"\n-- merged entities ({len(entry['merged_entities'])}) --")
        for e in entry["merged_entities"]:
            clauses = (e.get("source_clause") or "").split(" | ")
            print(f"   {e['concept']:<16} {e['status']:<9} dur={e['duration']!r} "
                  f"surface={e['surface_text']!r} from {len(clauses)} clause(s)")
            for c in clauses[:6]:
                print(f"        - {c[:130]!r}")
        print("\n-- v1 renderer rules --")
        print(json.dumps(entry["v1_sections"], indent=2, ensure_ascii=False)[:2500])

    out = ROOT / "experiments" / "clinical_intelligence_v2" / "baseline_trace.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
