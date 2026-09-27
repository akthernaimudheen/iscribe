"""Clinical Intelligence V2 — inspect the typed fact graph for the real cases.

Run:  ./.venv/Scripts/python.exe experiments/clinical_intelligence_v2/inspect_facts.py [case]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scribe_engine.clinical_facts import build_clinical_facts  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "english_real_cases"


def main() -> int:
    names = sys.argv[1:] or ["referral_letter_orthopaedic", "medicolegal_jones",
                             "medicolegal_finton"]
    for name in names:
        text = (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
        doc = build_clinical_facts(text)
        print("=" * 78)
        print(name, "|", doc["document"]["document_type"],
              f"conf={doc['document']['confidence']}",
              f"uncertain={doc['document']['uncertain']}")
        print("counts:", json.dumps(doc["counts"]["by_type"], ensure_ascii=False))
        print("sections:", json.dumps(doc["counts"]["by_section"], ensure_ascii=False))
        print("metadata:", json.dumps({k: v["value"] for k, v in doc["metadata"].items()},
                                      ensure_ascii=False))
        print("-- facts --")
        for f in doc["facts"]:
            attrs = {k: v for k, v in (f["attributes"] or {}).items() if v}
            print(f"  {f['fact_id']:<5} {f['fact_type']:<16} {f['status']:<11} "
                  f"{f['certainty']:<12} {f['temporal_context']:<12} "
                  f"{f['section']:<20} {str(f['english'])[:48]!r}")
            print(f"        src: {f['source_text'][:110]!r}")
            if attrs:
                print(f"        attrs: {json.dumps(attrs, ensure_ascii=False)[:220]}")
        print("-- elements --")
        for e in doc["elements"]:
            print(f"  {e['element']:<24} {e['text'][:70]!r}")
        print("validation:", json.dumps(doc["validation"]["violations"][:8], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
