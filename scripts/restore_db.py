"""Verify a backup WITHOUT touching the live production database.

Opens the backup copy read-only, runs integrity_check, counts rows and
reports the newest consultation timestamp so an operator can eyeball
freshness. The live DB is never opened for writing.

Usage:
    python scripts/restore_db.py --verify <backup.db>
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def verify(path: Path) -> int:
    if not path.exists():
        print(f"FAIL: {path} does not exist")
        return 1
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            (integrity,) = con.execute("PRAGMA integrity_check").fetchone()
            tables = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                      for t in ("consultations", "users", "audit_events")}
            row = con.execute(
                "SELECT MAX(updated_at) FROM consultations").fetchone()
            newest = row[0] if row else None
        finally:
            con.close()
    except sqlite3.DatabaseError as exc:
        print(f"file             : {path}")
        print(f"integrity_check  : FAILED ({type(exc).__name__})")
        print("RESULT           : FAIL - not a usable SQLite database")
        return 1

    print(f"file             : {path}")
    print(f"size_bytes       : {path.stat().st_size}")
    print(f"integrity_check  : {integrity}")
    print(f"consultations    : {tables['consultations']}")
    print(f"users            : {tables['users']}")
    print(f"audit_events     : {tables['audit_events']}")
    print(f"newest_updated_at: {newest}")
    ok = integrity == "ok"
    print("RESULT           :", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", required=True, type=Path,
                    help="path to a backup .db to verify read-only")
    args = ap.parse_args()
    return verify(args.verify)


if __name__ == "__main__":
    raise SystemExit(main())
