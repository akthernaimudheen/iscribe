"""Pre-deployment safety gate for the single-machine trial.

The watchdog restarts the service whenever git HEAD changes, so a push to
main IS a production deployment. This script is the safety gate to run
BEFORE pushing (or pulling on the server):

  1. fresh online backup of the production database
  2. full local test suite (skippable with --skip-tests for speed)
  3. fail-closed deployment report

It makes no attempt to roll back code by itself; see RUNBOOK section I for
the manual rollback which is two commands. The gate guarantees that a
restore point exists before any deploy, so a bad update can never leave you
with both broken code AND no safe copy of clinical data.

Usage:
    python scripts/predeploy_check.py [--skip-tests]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

FAILED = []


def step(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-tests", action="store_true",
                    help="skip the pytest run (backup still runs)")
    args = ap.parse_args()

    print(f"pre-deploy check @ {REPO_ROOT}")

    # 1. Fresh backup => a restore point exists before the deploy.
    backup = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "backup_db.py")],
        capture_output=True, text=True, timeout=300)
    step("online database backup", backup.returncode == 0,
         backup.stdout.strip() or backup.stderr.strip())

    # 2. Working tree must not carry surprises into the deploy.
    dirty = subprocess.run(["git", "status", "--porcelain"],
                           cwd=REPO_ROOT, capture_output=True, text=True)
    untracked_data = [
        line for line in dirty.stdout.splitlines()
        if line.strip() and not line.startswith("??")
    ]
    step("git tracked files clean", not untracked_data,
         f"{len(untracked_data)} modified tracked file(s)")

    # 3. Test suite.
    if not args.skip_tests:
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=3600)
        tail = (tests.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        step("test suite", tests.returncode == 0, tail[0])
    else:
        step("test suite", True, "SKIPPED by flag")

    print()
    if FAILED:
        print("DEPLOY BLOCKED - fix the failures above.")
        return 1
    print("READY TO DEPLOY: git push (or git pull on the host), then let the "
          "watchdog restart the service. Rollback: see runbook section I.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
