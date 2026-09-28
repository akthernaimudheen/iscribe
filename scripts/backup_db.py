"""Online production database backup for the single-machine trial.

Design constraints (see docs/PRODUCTION_TRIAL_RUNBOOK.md):
  * The live database is NEVER touched by this script beyond SQLite's own
    online-backup API (no exclusive locks, no WAL surgery, no downtime).
  * Backups are timestamped copies of ONE file (the whole database -- users,
    consultations, notes, transcripts, jobs and the audit trail all live in
    the single iscribe.db, by design).
  * Every backup is verified with PRAGMA integrity_check before it counts.
  * Keeps a small rolling window (default 14) of verified backups.
  * Failures are visible: exit code 1 + a failure marker file the watchdog
    logs on its next cycle.
  * Backup files contain PHI and NEVER leave the machine: they are written
    OUTSIDE the repo tree, in a gitignored sibling directory, and are not
    reachable through the web server (the app only serves its static dir
    and API endpoints).

Usage:
    python scripts/backup_db.py [--backup-dir DIR] [--keep N]

Returns 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from service.config import load_settings  # noqa: E402

FAIL_MARKER = "LAST_BACKUP_FAILED.txt"


def _default_backup_dir() -> Path:
    """<repo>/iscribe-backups: inside the OneDrive-synced tree (free offsite
    sync for the trial) but outside every web-served directory, and fully
    gitignored (see /iscribe-backups/ in .gitignore) so git can never stage
    it. The runbook documents moving this to a second drive for the full
    3-2-1 pattern if the clinic wants it."""
    settings = load_settings()
    return settings.db_path.parent.parent / "iscribe-backups"


def _verify(db_path: Path) -> str:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        (result,) = con.execute("PRAGMA integrity_check").fetchone()
        return result
    finally:
        con.close()


def make_backup(backup_dir: Path | None = None, keep: int = 14) -> Path:
    settings = load_settings()
    db_path = settings.db_path
    dest_dir = Path(backup_dir) if backup_dir else _default_backup_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = dest_dir / f"iscribe-{stamp}.db"

    src = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        dst = sqlite3.connect(dest)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()

    # The copy inherits the live DB's WAL journal mode, so opening it leaves
    # -wal/-shm sidecars behind. After a clean close the WAL is always empty;
    # delete the sidecars so a backup is exactly ONE restorable file.
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(dest) + suffix)
        if sidecar.exists() and sidecar.stat().st_size == 0:
            sidecar.unlink()

    result = _verify(dest)
    if result != "ok":
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"backup failed integrity_check: {result}")

    rows = {}
    for table in ("consultations", "users", "audit_events"):
        con = sqlite3.connect(f"file:{dest}?mode=ro", uri=True)
        try:
            rows[table] = con.execute(
                f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        finally:
            con.close()
    (dest_dir / f"{dest.name}.manifest.txt").write_text(
        f"timestamp={stamp}\n"
        f"source=online sqlite3 backup API (live, no downtime)\n"
        f"integrity_check={result}\n"
        f"consultations={rows['consultations']}\n"
        f"users={rows['users']}\n"
        f"audit_events={rows['audit_events']}\n",
        encoding="utf-8",
    )

    backups = sorted(dest_dir.glob("iscribe-*.db"))
    for stale in backups[:-keep] if len(backups) > keep else []:
        for suffix in ("", ".manifest.txt", "-wal", "-shm"):
            Path(str(stale) + suffix).unlink(missing_ok=True)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup-dir", default=None,
                    help="override backup destination (default: <repo>/../iscribe-backups)")
    ap.add_argument("--keep", type=int, default=14,
                    help="rolling backups to retain (default 14)")
    args = ap.parse_args()

    fail_marker = REPO_ROOT / FAIL_MARKER
    try:
        dest = make_backup(
            Path(args.backup_dir) if args.backup_dir else None, args.keep)
    except Exception as exc:
        fail_marker.write_text(
            f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {type(exc).__name__}: {exc}\n",
            encoding="utf-8")
        print(f"BACKUP FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    fail_marker.unlink(missing_ok=True)
    print(f"backup ok: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
