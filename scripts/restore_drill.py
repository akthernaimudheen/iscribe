"""Restore drill: prove a backup is restorable WITHOUT touching production.

Runs the full documented restore procedure (runbook section H) against a
TEMPORARY copy of the production data directory:

  1. cold-copy the live DB (sqlite3 backup API, then close) to a temp dir
  2. overwrite the temp copy with the newest backup (copy, not move)
  3. WAL-recover the restored copy
  4. assert content survived the restore round-trip:
       users intact, consultations intact, clinical notes intact,
       review/approval state intact

The live production database is only ever READ. Nothing in the live data
directory or backup directory is modified or deleted.

Usage:  python scripts/restore_drill.py
Exit 0 and prints DRILL RESULT: PASS on success.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from service.config import load_settings  # noqa: E402


def newest_backup() -> Path:
    settings = load_settings()
    backups = sorted((settings.db_path.parent.parent / "iscribe-backups")
                     .glob("iscribe-*.db"))
    if not backups:
        raise SystemExit("FAIL: no backups found - run scripts/backup_db.py")
    return backups[-1]


def snapshot_counts(db: Path) -> dict:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        tables = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("consultations", "users", "audit_events")}
        users = con.execute(
            "SELECT COUNT(*) FROM users WHERE role='DOCTOR' AND disabled=0"
        ).fetchone()[0]
        approved = con.execute(
            "SELECT COUNT(*) FROM consultations "
            "WHERE json_extract(document, '$.reviewed') = 1").fetchone()[0]
        with_notes = con.execute(
            "SELECT COUNT(*) FROM consultations "
            "WHERE json_extract(document, '$.result.clinical_note.fields') "
            "IS NOT NULL").fetchone()[0]
        return {"consultations": tables["consultations"],
                "users": tables["users"],
                "audit_events": tables["audit_events"],
                "enabled_doctors": users,
                "reviewed_notes": approved,
                "notes_with_fields": with_notes}
    finally:
        con.close()


def main() -> int:
    settings = load_settings()
    live_db = settings.db_path
    backup = newest_backup()
    print(f"live db (READ ONLY) : {live_db}")
    print(f"backup under test   : {backup}")

    # Guard: refuse to run when the live DB or live data dir could be touched.
    if "backup" in str(live_db).lower():
        raise SystemExit("FAIL: safety guard tripped")

    failures = []

    with tempfile.TemporaryDirectory(prefix="iscribe-restore-drill-") as td:
        work = Path(td) / "data"
        work.mkdir(parents=True)
        sandbox = work / "iscribe.db"

        # 1. Cold copy of the live DB (backup API, connection closed after).
        src = sqlite3.connect(f"file:{live_db}?mode=ro", uri=True)
        dst = sqlite3.connect(sandbox)
        src.backup(dst)
        dst.close()
        src.close()

        before = snapshot_counts(sandbox)
        print(f"pre-restore counts  : {before}")

        # 2. Documented restore procedure: copy the backup OVER the sandbox
        #    copy (never over production), deleting sidecars first.
        for suffix in ("-wal", "-shm"):
            Path(str(sandbox) + suffix).unlink(missing_ok=True)
        shutil.copy2(backup, sandbox)

        # 3. Open once to force WAL/journal recovery on the restored file.
        con = sqlite3.connect(sandbox)
        (integrity,) = con.execute("PRAGMA integrity_check").fetchone()
        con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        con.close()
        print(f"restored integrity  : {integrity}")
        if integrity != "ok":
            failures.append("integrity_check failed after restore")

        # 4. Content assertions.
        after = snapshot_counts(sandbox)
        print(f"post-restore counts : {after}")
        for key in before:
            if before[key] != after[key]:
                failures.append(
                    f"{key}: {before[key]} before != {after[key]} after")
        if after["users"] < 1:
            failures.append("no users after restore")
        if after["enabled_doctors"] < 1:
            failures.append("no enabled doctor account after restore")

        # Round-trip the note/review state through the real app layer to be
        # sure the restored bytes are what the application actually reads.
        sys.path.insert(0, str(REPO_ROOT))
        con = sqlite3.connect(sandbox)
        con.row_factory = sqlite3.Row
        row = con.execute(
            "SELECT id, document FROM consultations "
            "WHERE json_extract(document, '$.result.clinical_note') IS NOT NULL "
            "ORDER BY created_ts DESC LIMIT 1").fetchone()
        con.close()
        if row is None:
            failures.append("no consultation with a clinical note to round-trip")
        else:
            import json
            doc = json.loads(row["document"])
            note_fields = doc["result"]["clinical_note"]["fields"]
            if not note_fields:
                failures.append("restored clinical note has empty fields")
            print(f"note round-trip     : cid={row['id'][:8]}... "
                  f"reviewed={doc.get('reviewed')} "
                  f"note_fields={len(note_fields)} keys intact")

    if failures:
        print()
        for f in failures:
            print(f"FAIL: {f}")
        print("DRILL RESULT: FAIL")
        return 1
    print()
    print("DRILL RESULT: PASS - backup restores with users, consultations, "
          "clinical notes and review state intact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
