"""Persistent consultation store (SQLite).

Replaces the demo's in-process ``STORAGE: dict``, which lost every consultation
on restart. SQLite is deliberate: the trial runs as a single process on one
machine, so a file-backed embedded database gives durability and crash-recovery
with zero operational surface (no server, no credentials, no network port).

The whole consultation record is stored as a JSON document, with the few fields
needed for listing/cleanup promoted to real columns. That keeps the engine's
result schema free to evolve without migrations.

PHI note: this database holds clinical content. It lives under the configured
data directory, is never exposed through a static mount, and is excluded from
version control.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

_SCHEMA = """
CREATE TABLE IF NOT EXISTS consultations (
    id            TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    created_ts    REAL NOT NULL,
    status        TEXT NOT NULL,
    audio_path    TEXT,
    audio_purged  INTEGER NOT NULL DEFAULT 0,
    document      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_consultations_created_ts
    ON consultations (created_ts DESC);
CREATE INDEX IF NOT EXISTS idx_consultations_audio
    ON consultations (audio_purged, created_ts);
CREATE TABLE IF NOT EXISTS documentation_jobs (
    job_id           TEXT PRIMARY KEY,
    consultation_id  TEXT NOT NULL,
    recording_key    TEXT NOT NULL,
    status           TEXT NOT NULL,
    stage            TEXT,
    error            TEXT,
    created_at       TEXT NOT NULL,
    created_ts       REAL NOT NULL,
    started_at       TEXT,
    completed_at     TEXT
);
-- Idempotency: one documentation outcome per (consultation, recording).
-- A FAILED job may be retried; that inserts a NEW row (history preserved)
-- and the old row keeps its terminal state.
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_idempotency
    ON documentation_jobs (consultation_id, recording_key, status)
    WHERE status IN ('queued', 'processing', 'completed');
CREATE INDEX IF NOT EXISTS idx_jobs_consultation
    ON documentation_jobs (consultation_id, created_ts DESC);

-- Append-only audit trail. There is deliberately NO update or delete code
-- path for this table anywhere in the codebase. Entries reference ids only —
-- never transcripts, notes, audio content or patient identifiers.
CREATE TABLE IF NOT EXISTS audit_events (
    event_id     TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,
    ts_epoch     REAL NOT NULL,
    action       TEXT NOT NULL,
    cid          TEXT,
    hospital_id  TEXT,
    actor        TEXT,
    detail       TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_cid
    ON audit_events (cid, ts_epoch);
CREATE INDEX IF NOT EXISTS idx_audit_action
    ON audit_events (action, ts_epoch);

-- Per-hospital retention/training policy. Row values override the environment
-- defaults where set; absence of authorization is denial (fail closed).
CREATE TABLE IF NOT EXISTS hospital_config (
    hospital_id            TEXT PRIMARY KEY,
    audio_retention_after_note_approval_hours INTEGER,
    hospital_training_enabled INTEGER NOT NULL DEFAULT 0,
    training_retention_period_hours INTEGER,
    training_data_requires_explicit_selection INTEGER NOT NULL DEFAULT 1,
    training_data_requires_hospital_authorization INTEGER NOT NULL DEFAULT 1,
    training_authorized    INTEGER NOT NULL DEFAULT 0,
    training_authorization_version TEXT,
    training_authorized_at TEXT,
    training_authorized_by TEXT,
    training_authorization_expiry REAL,
    training_policy_reference TEXT,
    training_consent_revoked_at TEXT,
    training_consent_revoked_by TEXT,
    updated_at             TEXT
);

-- Training dataset. Logically and physically separate from clinical audio:
-- its own rows, its own storage namespace (the vault), its own retention
-- deadline and its own deletion lifecycle. Never a flag on a clinical record.
CREATE TABLE IF NOT EXISTS training_records (
    training_record_id    TEXT PRIMARY KEY,
    source_consultation_id TEXT NOT NULL,
    hospital_id           TEXT NOT NULL,
    audio_path            TEXT NOT NULL,
    audio_filename        TEXT NOT NULL,
    sha256                TEXT NOT NULL,
    authorization_reference TEXT NOT NULL,
    retention_deadline    REAL NOT NULL,
    created_at            TEXT NOT NULL,
    created_ts            REAL NOT NULL,
    approved_for_training_at TEXT NOT NULL,
    approved_by           TEXT NOT NULL,
    selection_reason      TEXT,
    deletion_status       TEXT NOT NULL DEFAULT 'retained',
    deletion_attempts     INTEGER NOT NULL DEFAULT 0,
    last_deletion_error   TEXT,
    next_deletion_retry_ts REAL,
    revoked               INTEGER NOT NULL DEFAULT 0,
    -- Human review staging: a candidate is NOT in the training corpus until
    -- an authorized reviewer approves it. review_status: pending|approved|rejected.
    review_status         TEXT NOT NULL DEFAULT 'pending',
    reviewed_at           TEXT,
    reviewed_by           TEXT,
    review_note           TEXT,
    -- De-identification artifact (transcript copy only; audio is NEVER
    -- claimed de-identified): deidentification_status TRANSCRIPT_DEIDENTIFIED.
    deidentification_status TEXT,
    deidentified_transcript TEXT,
    deidentification_provenance TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_deadline
    ON training_records (deletion_status, retention_deadline);
CREATE INDEX IF NOT EXISTS idx_training_hospital
    ON training_records (hospital_id);
"""

# Additive migration columns for the consultations table (see _migrate).
_CONSULTATION_MIGRATION_COLUMNS = {
    "hospital_id": "TEXT",
    "doctor_id": "TEXT",
    "audio_state": "TEXT",
    "audio_deletion_due_ts": "REAL",
    "deletion_attempts": "INTEGER NOT NULL DEFAULT 0",
    "last_deletion_error": "TEXT",
    "next_deletion_retry_ts": "REAL",
    "approved_at": "TEXT",
    "approved_by": "TEXT",
    "review_started_at": "TEXT",
    "audio_sha256": "TEXT",
    "training_record_id": "TEXT",
    # Note approval gate: DRAFT -> REVIEWED (doctor edited/opened review)
    # -> APPROVED (explicit approval event). "Note generated" is never
    # "approved"; the retention countdown never starts from DRAFT/REVIEWED.
    "note_status": "TEXT",
    "approval_version": "INTEGER",
    # Retention policy snapshot stamped when the deletion deadline was
    # computed, so a later global policy change never silently rewrites a
    # historical retention decision.
    "retention_policy_id": "TEXT",
    "retention_policy_version": "TEXT",
    "deleted_at": "TEXT",
}


class ConsultationStore:
    """Thread-safe SQLite store for consultation documents."""

    def __init__(self, db_path: Path, hospital_id: str = "default"):
        self.db_path = Path(db_path)
        self.hospital_id = hospital_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,   # guarded by self._lock
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            # WAL keeps the polling UI's reads from blocking a write mid-job.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._migrate()
            self._conn.commit()

    def _migrate(self) -> None:
        """Additive, idempotent migration for pre-retention-layer databases.

        Adds the lifecycle/tenancy columns to consultations and the hospital
        column to documentation_jobs, then backfills safe values: every
        existing row belongs to this server's hospital, and existing audio is
        mapped onto the closest lifecycle state (never silently deletable).
        """
        existing = {r["name"] for r in self._conn.execute(
            "PRAGMA table_info(consultations)")}
        for column, decl in _CONSULTATION_MIGRATION_COLUMNS.items():
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE consultations ADD COLUMN {column} {decl}")
        job_cols = {r["name"] for r in self._conn.execute(
            "PRAGMA table_info(documentation_jobs)")}
        if "hospital_id" not in job_cols:
            self._conn.execute(
                "ALTER TABLE documentation_jobs ADD COLUMN hospital_id TEXT")
        # Backfills (idempotent).
        self._conn.execute(
            "UPDATE consultations SET hospital_id = ? WHERE hospital_id IS NULL",
            (self.hospital_id,))
        self._conn.execute(
            "UPDATE documentation_jobs SET hospital_id = ? WHERE hospital_id IS NULL",
            (self.hospital_id,))
        self._conn.execute(
            "UPDATE consultations SET audio_state = CASE "
            "WHEN audio_purged = 1 THEN 'DELETED' "
            "WHEN audio_path IS NULL THEN NULL "
            "WHEN status = 'completed' THEN 'DOCTOR_APPROVED' "
            "ELSE 'AWAITING_DOCTOR_REVIEW' END "
            "WHERE audio_state IS NULL")
        # Note gate backfill: legacy completed rows were explicitly signed off
        # in the UI, so they count as approved; everything else is a draft.
        self._conn.execute(
            "UPDATE consultations SET note_status = 'APPROVED' "
            "WHERE note_status IS NULL AND approved_at IS NOT NULL")
        self._conn.execute(
            "UPDATE consultations SET note_status = 'DRAFT' "
            "WHERE note_status IS NULL")

    # -- lifecycle --------------------------------------------------------
    def close(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA optimize")
                self._conn.commit()
            finally:
                self._conn.close()

    # -- writes -----------------------------------------------------------
    def create(self, record: dict) -> dict:
        with self._lock:
            self._conn.execute(
                "INSERT INTO consultations "
                "(id, created_at, updated_at, created_ts, status, audio_path, "
                "audio_purged, hospital_id, note_status, document) "
                "VALUES (?,?,?,?,?,?,0,?,'DRAFT',?)",
                (
                    record["id"],
                    record["created_at"],
                    record["created_at"],
                    time.time(),
                    record["status"],
                    record.get("audio_file"),
                    # Tenancy is stamped at creation from the server config.
                    record.get("hospital_id") or self.hospital_id,
                    json.dumps(record),
                ),
            )
            self._conn.commit()
        return record

    @contextmanager
    def mutate(self, cid: str) -> Iterator[dict | None]:
        """Read-modify-write a consultation atomically.

        Yields the record dict (or None if it does not exist); whatever the
        caller leaves in that dict is persisted when the block exits normally.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT document FROM consultations WHERE id = ?", (cid,)
            ).fetchone()
            if row is None:
                yield None
                return
            record = json.loads(row["document"])
            yield record
            self._conn.execute(
                "UPDATE consultations SET updated_at = ?, status = ?, audio_path = ?, "
                "document = ? WHERE id = ?",
                (
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    record.get("status", "unknown"),
                    record.get("audio_file"),
                    json.dumps(record),
                    cid,
                ),
            )
            self._conn.commit()

    def apply(self, cid: str, fn: Callable[[dict], None]) -> dict | None:
        """Convenience wrapper around :meth:`mutate`."""
        with self.mutate(cid) as record:
            if record is None:
                return None
            fn(record)
            return record

    def mark_audio_purged(self, cid: str) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT document FROM consultations WHERE id = ?", (cid,)
            ).fetchone()
            if row is None:
                return
            record = json.loads(row["document"])
            record["audio_file"] = None
            record["audio_purged"] = True
            self._conn.execute(
                "UPDATE consultations SET audio_path = NULL, audio_purged = 1, "
                "document = ?, updated_at = ? WHERE id = ?",
                (json.dumps(record), time.strftime("%Y-%m-%d %H:%M:%S"), cid),
            )
            self._conn.commit()

    # -- reads ------------------------------------------------------------
    def get(self, cid: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT document FROM consultations WHERE id = ?", (cid,)
            ).fetchone()
        return json.loads(row["document"]) if row else None

    def list(self, limit: int = 200) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT document FROM consultations ORDER BY created_ts DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(r["document"]) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) AS n FROM consultations"
            ).fetchone()["n"]

    def expired_audio(self, older_than_ts: float) -> list[tuple[str, str]]:
        """(id, audio_path) for records whose audio is past the retention window."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, audio_path FROM consultations "
                "WHERE audio_purged = 0 AND audio_path IS NOT NULL AND created_ts < ?",
                (older_than_ts,),
            ).fetchall()
        return [(r["id"], r["audio_path"]) for r in rows]

    def stuck_jobs(self) -> list[str]:
        """Consultations left mid-processing by a crash or restart."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM consultations WHERE status = 'processing'"
            ).fetchall()
        return [r["id"] for r in rows]

    # -- documentation jobs -------------------------------------------------
    def create_job(self, cid: str, recording_key: str) -> dict:
        """Create (or idempotently return) the active job for a recording.

        If an active job (queued/processing/completed) already exists for the
        same (consultation, recording), it is returned unchanged — submitting
        the same recording twice never duplicates documentation. A FAILED job
        does not block a new attempt: retrying inserts a fresh row.
        """
        now_ts = time.time()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO documentation_jobs "
                    "(job_id, consultation_id, recording_key, status, "
                    "hospital_id, created_at, created_ts) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (job_id, cid, recording_key, "queued",
                     self.hospital_id, now, now_ts),
                )
                self._conn.commit()
                created = True
            except sqlite3.IntegrityError:
                created = False
            if not created:
                row = self._conn.execute(
                    "SELECT job_id FROM documentation_jobs "
                    "WHERE consultation_id = ? AND recording_key = ? "
                    "AND status IN ('queued','processing','completed')",
                    (cid, recording_key),
                ).fetchone()
                job_id = row["job_id"]
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documentation_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        return dict(row) if row else None

    def latest_job(self, cid: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documentation_jobs "
                "WHERE consultation_id = ? ORDER BY created_ts DESC LIMIT 1",
                (cid,),
            ).fetchone()
        return dict(row) if row else None

    def update_job(self, job_id: str, **fields) -> None:
        allowed = {"status", "stage", "error", "started_at", "completed_at"}
        cols, vals = [], []
        for k, v in fields.items():
            if k in allowed:
                cols.append(f"{k} = ?")
                vals.append(v)
        if not cols:
            return
        vals.append(job_id)
        with self._lock:
            # The idempotency index allows only ONE active/terminal-success row
            # per (consultation, recording): status IN (queued, processing,
            # completed). A RETRY of a previously-completed recording creates a
            # new job row; when that retry succeeds, marking it 'completed'
            # would collide with the older completed row and raise
            # sqlite3.IntegrityError — a successful retry was therefore
            # IMPOSSIBLE (observed on prod: job 9046678a4209). A terminal
            # history row is demoted to 'superseded' first; the newest
            # completed job owns the terminal state, and the audit story (all
            # attempts retained) is unchanged.
            new_status = fields.get("status")
            if new_status == "completed":
                self._conn.execute(
                    "UPDATE documentation_jobs SET status = 'superseded' "
                    "WHERE consultation_id = (SELECT consultation_id FROM "
                    "documentation_jobs WHERE job_id = ?) "
                    "AND recording_key = (SELECT recording_key FROM "
                    "documentation_jobs WHERE job_id = ?) "
                    "AND status = 'completed' AND job_id != ?",
                    (job_id, job_id, job_id),
                )
            self._conn.execute(
                f"UPDATE documentation_jobs SET {', '.join(cols)} WHERE job_id = ?",
                vals,
            )
            self._conn.commit()

    def interrupted_jobs(self) -> list[str]:
        """Jobs left queued/processing by a crash or restart."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT job_id FROM documentation_jobs "
                "WHERE status IN ('queued', 'processing')"
            ).fetchall()
        return [r["job_id"] for r in rows]

    # -- audit trail (append-only) -------------------------------------------
    # No update or delete method exists for audit_events, by design.
    def append_audit(self, action: str, cid: str | None = None,
                     hospital_id: str | None = None, actor: str | None = None,
                     detail: dict | None = None) -> str:
        """Record one lifecycle event. IDs and enums only — callers must never
        pass transcripts, notes, audio content or patient identifiers."""
        event_id = uuid.uuid4().hex[:16]
        now_ts = time.time()
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_events "
                "(event_id, ts, ts_epoch, action, cid, hospital_id, actor, detail) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (
                    event_id,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now_ts)) + "Z",
                    now_ts,
                    action,
                    cid,
                    hospital_id or self.hospital_id,
                    actor,
                    json.dumps(detail or {}, sort_keys=True),
                ),
            )
            self._conn.commit()
        return event_id

    def audit_events(self, cid: str | None = None, action: str | None = None,
                     limit: int = 200) -> list[dict]:
        q = "SELECT * FROM audit_events"
        clauses, vals = [], []
        if cid:
            clauses.append("cid = ?")
            vals.append(cid)
        if action:
            clauses.append("action = ?")
            vals.append(action)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY ts_epoch ASC LIMIT ?"
        vals.append(limit)
        with self._lock:
            rows = self._conn.execute(q, vals).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["detail"] = json.loads(d["detail"] or "{}")
            except Exception:
                d["detail"] = {}
            out.append(d)
        return out

    # -- audio lifecycle columns ---------------------------------------------
    _LIFECYCLE_COLUMNS = frozenset({
        "audio_state", "audio_deletion_due_ts", "deletion_attempts",
        "last_deletion_error", "next_deletion_retry_ts",
        "approved_at", "approved_by", "review_started_at",
        "training_record_id", "audio_sha256",
        "note_status", "approval_version", "retention_policy_id",
        "retention_policy_version", "deleted_at",
    })

    def set_audio_lifecycle(self, cid: str, **fields) -> None:
        cols, vals = [], []
        for k, v in fields.items():
            if k not in self._LIFECYCLE_COLUMNS:
                raise ValueError(f"Unknown lifecycle column {k!r}")
            cols.append(f"{k} = ?")
            vals.append(v)
        if not cols:
            return
        vals.append(cid)
        with self._lock:
            self._conn.execute(
                f"UPDATE consultations SET {', '.join(cols)} WHERE id = ?", vals)
            self._conn.commit()

    def get_audio_lifecycle(self, cid: str) -> dict | None:
        """Lifecycle/tenancy columns for one consultation (API-safe fields)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT hospital_id, doctor_id, audio_state, "
                "audio_deletion_due_ts, deletion_attempts, last_deletion_error, "
                "next_deletion_retry_ts, approved_at, approved_by, "
                "review_started_at, audio_sha256, training_record_id, created_ts, "
                "note_status, approval_version, retention_policy_id, "
                "retention_policy_version, deleted_at "
                "FROM consultations WHERE id = ?", (cid,)
            ).fetchone()
        return dict(row) if row else None

    def audio_due_for_deletion(self, now_ts: float) -> list[dict]:
        """Clinical audio records whose deletion deadline has passed."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, audio_path, deletion_attempts FROM consultations "
                "WHERE audio_purged = 0 AND audio_path IS NOT NULL "
                "AND audio_state = 'SCHEDULED_FOR_DELETION' "
                "AND audio_deletion_due_ts IS NOT NULL "
                "AND audio_deletion_due_ts <= ? "
                "AND (next_deletion_retry_ts IS NULL "
                "     OR next_deletion_retry_ts <= ?)",
                (now_ts, now_ts),
            ).fetchall()
        return [dict(r) for r in rows]

    def audio_without_deadline(self) -> list[dict]:
        """Live audio with no deletion deadline in any pre-approval state.

        Used by the startup/sweep backfill so the configured unreviewed
        ceiling always applies, even when a crash interrupted the normal
        evaluation flow. Pending-job protection is applied by the caller.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM consultations "
                "WHERE audio_purged = 0 AND audio_path IS NOT NULL "
                "AND audio_deletion_due_ts IS NULL AND audio_state IN ("
                "'UPLOADED', 'PROCESSING', 'TRANSCRIBED', 'NOTE_READY', "
                "'AWAITING_DOCTOR_REVIEW')"
            ).fetchall()
        return [dict(r) for r in rows]

    def audio_failed_deletions(self, now_ts: float) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, audio_path, deletion_attempts FROM consultations "
                "WHERE audio_purged = 0 AND audio_path IS NOT NULL "
                "AND audio_state = 'DELETION_FAILED' "
                "AND next_deletion_retry_ts IS NOT NULL "
                "AND next_deletion_retry_ts <= ?",
                (now_ts,),
            ).fetchall()
        return [dict(r) for r in rows]

    def has_pending_job(self, cid: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM documentation_jobs "
                "WHERE consultation_id = ? AND status IN ('queued', 'processing') "
                "LIMIT 1",
                (cid,),
            ).fetchone()
        return row is not None

    # -- per-hospital policy ---------------------------------------------------
    _HOSPITAL_CONFIG_COLUMNS = frozenset({
        "audio_retention_after_note_approval_hours",
        "hospital_training_enabled", "training_retention_period_hours",
        "training_data_requires_explicit_selection",
        "training_data_requires_hospital_authorization",
        "training_authorized", "training_authorization_version",
        "training_authorized_at", "training_authorized_by",
        "training_authorization_expiry", "training_policy_reference",
        "training_consent_revoked_at", "training_consent_revoked_by",
    })

    # Closed vocabulary for the training-candidate review lifecycle.
    TRAINING_REVIEW_STATUSES = frozenset({"pending", "approved", "rejected"})

    def get_hospital_config(self, hospital_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM hospital_config WHERE hospital_id = ?",
                (hospital_id,),
            ).fetchone()
        return dict(row) if row else None

    def set_hospital_config(self, hospital_id: str, **fields) -> dict:
        unknown = set(fields) - self._HOSPITAL_CONFIG_COLUMNS
        if unknown:
            raise ValueError(f"Unknown hospital_config columns: {sorted(unknown)}")
        with self._lock:
            self._conn.execute(
                "INSERT INTO hospital_config (hospital_id, updated_at) VALUES (?,?) "
                "ON CONFLICT(hospital_id) DO NOTHING",
                (hospital_id, time.strftime("%Y-%m-%d %H:%M:%S")))
            for k, v in fields.items():
                self._conn.execute(
                    f"UPDATE hospital_config SET {k} = ?, updated_at = ? "
                    "WHERE hospital_id = ?",
                    (v, time.strftime("%Y-%m-%d %H:%M:%S"), hospital_id))
            self._conn.commit()
        return self.get_hospital_config(hospital_id)

    # -- training records (separate dataset) -----------------------------------
    def create_training_record(self, fields: dict) -> dict:
        with self._lock:
            self._conn.execute(
                "INSERT INTO training_records ("
                "training_record_id, source_consultation_id, hospital_id, "
                "audio_path, audio_filename, sha256, authorization_reference, "
                "retention_deadline, created_at, created_ts, "
                "approved_for_training_at, approved_by, selection_reason, "
                "deidentification_status, deidentified_transcript, "
                "deidentification_provenance) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    fields["training_record_id"],
                    fields["source_consultation_id"],
                    fields["hospital_id"],
                    fields["audio_path"],
                    fields["audio_filename"],
                    fields["sha256"],
                    fields["authorization_reference"],
                    fields["retention_deadline"],
                    fields["created_at"],
                    fields["created_ts"],
                    fields["approved_for_training_at"],
                    fields["approved_by"],
                    fields.get("selection_reason"),
                    fields.get("deidentification_status"),
                    fields.get("deidentified_transcript"),
                    fields.get("deidentification_provenance"),
                ),
            )
            self._conn.commit()
        return self.get_training_record(fields["training_record_id"])

    def get_training_record(self, training_record_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM training_records WHERE training_record_id = ?",
                (training_record_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_training_records(self, status: str | None = None) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM training_records WHERE deletion_status = ? "
                    "ORDER BY created_ts DESC",
                    (status,)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM training_records ORDER BY created_ts DESC"
                ).fetchall()
        return [dict(r) for r in rows]

    def due_training_records(self, now_ts: float) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT training_record_id, audio_path, deletion_attempts "
                "FROM training_records "
                "WHERE deletion_status IN ('retained', 'scheduled') "
                "AND retention_deadline <= ? "
                "AND (next_deletion_retry_ts IS NULL "
                "     OR next_deletion_retry_ts <= ?)",
                (now_ts, now_ts),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_training_review(self, training_record_id: str, status: str,
                            reviewer: str, note: str | None = None) -> dict | None:
        """Record a human review decision on a training candidate."""
        if status not in self.TRAINING_REVIEW_STATUSES:
            raise ValueError(f"Unknown review status {status!r}")
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM training_records WHERE training_record_id = ?",
                (training_record_id,)).fetchone()
            if not row:
                return None
            self._conn.execute(
                "UPDATE training_records SET review_status = ?, reviewed_at = ?, "
                "reviewed_by = ?, review_note = ? WHERE training_record_id = ?",
                (status, time.strftime("%Y-%m-%d %H:%M:%S"), reviewer,
                 note, training_record_id))
            self._conn.commit()
        return self.get_training_record(training_record_id)

    def update_training_record(self, training_record_id: str, **fields) -> None:
        allowed = {"deletion_status", "deletion_attempts",
                   "last_deletion_error", "next_deletion_retry_ts", "revoked",
                   "deidentification_status", "deidentified_transcript",
                   "deidentification_provenance"}
        cols, vals = [], []
        for k, v in fields.items():
            if k not in allowed:
                raise ValueError(f"Unknown training_record column {k!r}")
            cols.append(f"{k} = ?")
            vals.append(v)
        if not cols:
            return
        vals.append(training_record_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE training_records SET {', '.join(cols)} "
                "WHERE training_record_id = ?", vals)
            self._conn.commit()
