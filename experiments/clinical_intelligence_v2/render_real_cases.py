"""Clinical Intelligence V2 — render the real cases end to end.

Run:  ./.venv/Scripts/python.exe experiments/clinical_intelligence_v2/render_real_cases.py [case]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scribe_engine.clinical_facts import build_clinical_facts  # noqa: E402
from scribe_engine.note_v2 import render_clinical_note          # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures" / "english_real_cases"


def main() -> int:
    names = sys.argv[1:] or ["referral_letter_orthopaedic", "medicolegal_jones",
                             "medicolegal_finton"]
    failures = 0
    for name in names:
        text = (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")
        doc = build_clinical_facts(text)
        out = render_clinical_note(doc)
        print("#" * 78)
        print(name)
        print(out["note"])
        v = out["validation"]
        print(f"validation: {'PASS' if v['valid'] else 'FAIL'} "
              f"({len(v['violations'])} violations)")
        for violation in v["violations"][:12]:
            print("   !", violation)
        failures += 0 if v["valid"] else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
