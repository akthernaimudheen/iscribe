"""Audio retention, consent and training-data separation.

This module owns the AUDIO LIFECYCLE for clinical recordings and the
authorization path into the separate TRAINING DATASET. It is deliberately
separate from the clinical pipeline: nothing here touches transcription,
corrections, facts or note generation.

Design rules enforced in code (not just documentation):

* Minimum retention. Defaults favour deletion. Audio is never retained
  "indefinitely" by default, and nothing is ever retained for training
  without a current, explicit, audited hospital authorization.
* Review-window invariant. Clinical audio is never scheduled for deletion
  while a documentation job is pending or before an explicit approval event
  (or the configured unreviewed ceiling — a policy value the hospital sets,
  never an internal default).
* Fail closed. Every ambiguous authorization state (no config row, unknown
  version, expired authorization, missing token, wrong hospital) is treated
  as DENIAL.
* Logical separation. Training data lives in its own table and its own
  storage namespace with its own retention deadline. A clinical record is
  never "marked training"; a training copy is its own record.
* Auditable. Every transition emits an append-only audit event carrying IDs
  and enums only — never transcripts, notes or audio content.

Storage namespaces:

    data/uploads/       IDENTIFIED CLINICAL AUDIO (session + token gated)
    data/training_vault/  IDENTIFIED TRAINING AUDIO (authorized environment)

There is NO automated de-identification of audio anywhere in this system;
training audio therefore remains identified clinical audio held inside the
authorized hospital environment. See docs/RETENTION_AND_DATA_FLOW.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

from .config import Settings
from .deidentify import deidentify_transcript
from .store import ConsultationStore

# --------------------------------------------------------------------------
# Lifecycle states (closed vocabulary — do not invent states elsewhere)
# --------------------------------------------------------------------------
UPLOADED = "UPLOADED"
PROCESSING = "PROCESSING"
TRANSCRIBED = "TRANSCRIBED"
NOTE_READY = "NOTE_READY"
AWAITING_DOCTOR_REVIEW = "AWAITING_DOCTOR_REVIEW"
DOCTOR_APPROVED = "DOCTOR_APPROVED"
SCHEDULED_FOR_DELETION = "SCHEDULED_FOR_DELETION"
DELETION_FAILED = "DELETION_FAILED"
DELETED = "DELETED"

# Deletion retry policy: bounded, with visible failures. A stuck file is a
# PHI-retention incident, not a quiet inconvenience.
DELETE_MAX_ATTEMPTS = 5
DELETE_RETRY_BASE_SECONDS = 900          # 15 minutes, doubling per attempt


def _now() -> float:
    return time.time()


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class RetentionPolicy:
    """Effective policy for one hospital: environment defaults overridden by
    the hospital's own configuration row where present.

    Defaults favour deletion:
      * audio deleted 24 h after approval (0 = immediately at approval),
      * never-reviewed audio deleted 168 h after upload (explicit ceiling),
      * training retention disabled, explicit selection required, hospital
        authorization required.
    """

    def __init__(self, settings: Settings, store: ConsultationStore,
                 hospital_id: str | None = None):
        self.settings = settings
        self.store = store
        self.hospital_id = hospital_id or settings.hospital_id

    def _cfg(self) -> dict | None:
        return self.store.get_hospital_config(self.hospital_id)

    def _override(self, cfg: dict | None, key: str, default):
        value = (cfg or {}).get(key)
        return default if value is None else value

    # -- clinical retention --------------------------------------------------
    @property
    def audio_retention_after_note_approval_hours(self) -> int:
        return int(self._override(
            self._cfg(), "audio_retention_after_note_approval_hours",
            self.settings.audio_retention_after_note_approval_hours))

    @property
    def audio_retention_unreviewed_hours(self) -> int:
        return int(self.settings.audio_retention_unreviewed_hours)

    # -- training retention ----------------------------------------------------
    @property
    def training_retention_enabled(self) -> bool:
        # The master switch is the AND of the deployment setting and the
        # hospital's own config (when a row exists). Either alone denying is
        # enough: fail closed.
        cfg = self._cfg()
        if cfg is not None:
            return bool(cfg.get("hospital_training_enabled")) and \
                bool(self.settings.training_retention_enabled)
        return bool(self.settings.training_retention_enabled)

    @property
    def training_retention_period_hours(self) -> int:
        return int(self._override(
            self._cfg(), "training_retention_period_hours",
            self.settings.training_retention_period_hours))

    @property
    def training_data_requires_explicit_selection(self) -> bool:
        return bool(self._override(
            self._cfg(), "training_data_requires_explicit_selection",
            self.settings.training_data_requires_explicit_selection))

    @property
    def training_data_requires_hospital_authorization(self) -> bool:
        return bool(self._override(
            self._cfg(), "training_data_requires_hospital_authorization",
            self.settings.training_data_requires_hospital_authorization))

    def as_public_dict(self) -> dict:
        """Redacted, UI-safe view of the effective policy. No secrets."""
        return {
            "hospital_id": self.hospital_id,
            "audio_retention_after_note_approval_hours":
                self.audio_retention_after_note_approval_hours,
            "audio_retention_unreviewed_hours":
                self.audio_retention_unreviewed_hours,
            "training_retention_enabled": self.training_retention_enabled,
            "training_retention_period_hours": self.training_retention_period_hours,
            "training_data_requires_explicit_selection":
                self.training_data_requires_explicit_selection,
            "training_data_requires_hospital_authorization":
                self.training_data_requires_hospital_authorization,
        }


class RetentionManager:
    """Executes the lifecycle: scheduling, deletion, and the training path."""

    def __init__(self, settings: Settings, store: ConsultationStore,
                 logger=None):
        self.settings = settings
        self.store = store
        self.logger = logger
        self.policy = RetentionPolicy(settings, store)

    # -- audit helper ----------------------------------------------------------
    def audit(self, action: str, cid: str | None = None, actor: str | None = None,
              **detail) -> None:
        # Unknown/absent facts are omitted entirely, never recorded as null:
        # a null request_id would read as "the provider returned an id".
        detail = {k: v for k, v in detail.items() if v is not None}
        self.store.append_audit(action, cid=cid,
                                hospital_id=self.policy.hospital_id,
                                actor=actor, detail=detail or None)
        if self.logger is not None:
            self.logger.info("event=audit_%s cid=%s", action, cid or "-")

    # -- lifecycle transitions ---------------------------------------------------
    def mark_uploaded(self, cid: str, audio_path: Path, actor: str = None) -> None:
        self.store.set_audio_lifecycle(cid, audio_state=UPLOADED)
        self.audit("audio_uploaded", cid, actor,
                   filename=audio_path.name)

    def mark_processing(self, cid: str) -> None:
        self.store.set_audio_lifecycle(cid, audio_state=PROCESSING)
        self.audit("transcription_requested", cid)

    def mark_note_ready(self, cid: str, provider: str | None = None) -> None:
        self.store.set_audio_lifecycle(cid, audio_state=NOTE_READY)
        self.audit("note_generated", cid, provider_used=provider or "unknown")
        self._evaluate(cid, reason="note_ready")

    def mark_review_started(self, cid: str) -> None:
        lifecycle = self.store.get_audio_lifecycle(cid) or {}
        if not lifecycle.get("review_started_at"):
            self.store.set_audio_lifecycle(cid, audio_state=AWAITING_DOCTOR_REVIEW,
                                           review_started_at=_stamp())
            self.audit("note_reviewed", cid)
            # Review opens the doctor's window; it never starts the deletion
            # countdown. The countdown starts at approval, or — only when the
            # hospital configured it — at the unreviewed ceiling, which the
            # sweep's backfill applies from upload time.

    def policy_identity(self, basis: str, hours: int) -> tuple[str, str]:
        """(policy_id, policy_version) snapshot for a retention decision.

        The version encodes the parameters actually used, so a later global
        policy change can never silently reinterpret a historical decision.
        """
        return self.policy.hospital_id, f"{basis}={int(hours)}h"

    def approve(self, cid: str, actor: str = None) -> None:
        """The explicit doctor approval event. Evaluates retention policy and
        either schedules deletion or (only through the fail-closed training
        path) moves audio into the authorized vault."""
        rec = self.store.get(cid)
        if not rec:
            raise KeyError(cid)
        # Text-only consultations have no audio; approval is still recorded.
        # _evaluate() no-ops when there is nothing to retain or delete.
        lifecycle = self.store.get_audio_lifecycle(cid) or {}
        version = int(lifecycle.get("approval_version") or 0) + 1
        self.store.set_audio_lifecycle(
            cid, audio_state=DOCTOR_APPROVED, note_status="APPROVED",
            approval_version=version,
            approved_at=_stamp(), approved_by=actor or "shared-session")
        self.audit("note_approved", cid, actor=actor, approval_version=version)

        self._evaluate(cid, reason="approval", actor=actor)

    # -- policy evaluation --------------------------------------------------------
    def _evaluate(self, cid: str, reason: str, actor: str = None) -> None:
        """Compute the deletion deadline for clinical audio and record it.

        Never deletes here; the sweep does the actual deletion so failures are
        recorded and retried in one place.
        """
        rec = self.store.get(cid) or {}
        if rec.get("audio_purged") or not rec.get("audio_file"):
            return
        # Lifecycle state lives in real columns (the JSON document never
        # mirrors them), so read the authoritative state from the columns.
        lifecycle = self.store.get_audio_lifecycle(cid) or {}
        state = lifecycle.get("audio_state")
        PRE_APPROVAL_STATES = (UPLOADED, PROCESSING, TRANSCRIBED,
                               NOTE_READY, AWAITING_DOCTOR_REVIEW)
        if state not in PRE_APPROVAL_STATES and state != DOCTOR_APPROVED:
            # Already scheduled/deleted — nothing to evaluate.
            return

        now = _now()
        if state == DOCTOR_APPROVED:
            hours = self.policy.audio_retention_after_note_approval_hours
            policy_id, policy_version = self.policy_identity(
                "after_approval", hours)
            due = now + max(0, hours) * 3600
            self.store.set_audio_lifecycle(
                cid, audio_state=SCHEDULED_FOR_DELETION,
                audio_deletion_due_ts=due,
                retention_policy_id=policy_id,
                retention_policy_version=policy_version)
            self.audit("retention_policy_evaluated", cid,
                       basis="after_approval", due_hours=hours,
                       deletion_due_ts=due, policy_version=policy_version)
        else:
            ceiling = self.policy.audio_retention_unreviewed_hours
            if ceiling <= 0:
                return
            # The ceiling anchors to UPLOAD time, not to when evaluation runs:
            # an approved-but-unreviewed recording can never outlive the
            # configured window the hospital agreed to.
            created_ts = float(lifecycle.get("created_ts") or now)
            due = created_ts + ceiling * 3600
            policy_id, policy_version = self.policy_identity(
                "unreviewed_ceiling", ceiling)
            self.store.set_audio_lifecycle(
                cid, audio_state=SCHEDULED_FOR_DELETION,
                audio_deletion_due_ts=due,
                retention_policy_id=policy_id,
                retention_policy_version=policy_version)
            self.audit("retention_policy_evaluated", cid,
                       basis="unreviewed_ceiling", due_hours=ceiling,
                       deletion_due_ts=due, policy_version=policy_version,
                       reason=reason)

    # -- deletion ----------------------------------------------------------------
    def sweep(self) -> dict:
        """Evaluate deadlines and execute due deletions (clinical + training).

        Called periodically by the service. Returns an operational summary
        (counts only — safe to log).
        """
        now = _now()
        summary = {"clinical_deleted": 0, "clinical_failed": 0,
                   "training_deleted": 0, "training_failed": 0,
                   "training_scheduled": 0}

        # Clinical audio.
        for row in self.store.audio_due_for_deletion(now) + \
                self.store.audio_failed_deletions(now):
            if self.store.has_pending_job(row["id"]):
                # Invariant: a doctor (or the pipeline) may still need the
                # audio. Deletion waits until the job finishes.
                continue
            if self._delete_clinical(row["id"], row["audio_path"],
                                     row["deletion_attempts"] or 0):
                summary["clinical_deleted"] += 1
            else:
                summary["clinical_failed"] += 1

        # Training records: schedule expiry and execute.
        for row in self.store.due_training_records(now):
            rec = self.store.get_training_record(row["training_record_id"])
            if rec and rec["deletion_status"] == "retained":
                self.store.update_training_record(
                    row["training_record_id"], deletion_status="scheduled")
                summary["training_scheduled"] += 1
            if self._delete_training(row["training_record_id"],
                                     row["audio_path"],
                                     row["deletion_attempts"] or 0):
                summary["training_deleted"] += 1
            else:
                summary["training_failed"] += 1
        return summary

    def _delete_clinical(self, cid: str, audio_path: str, attempts: int) -> bool:
        path = Path(audio_path)
        attempts += 1
        try:
            if path.exists():
                path.unlink()
            else:
                # The file is already gone. That IS deletion, but it must be
                # visible: recorded, not silently ignored.
                self.logger and self.logger.warning(
                    "event=audio_already_absent cid=%s", cid)
            self.store.mark_audio_purged(cid)
            self.store.set_audio_lifecycle(cid, audio_state=DELETED,
                                           audio_deletion_due_ts=None,
                                           last_deletion_error=None,
                                           next_deletion_retry_ts=None,
                                           deleted_at=_stamp())
            self.audit("audio_deleted", cid, attempts=attempts,
                       filename=path.name)
            return True
        except Exception as exc:
            error = f"{type(exc).__name__}"
            self.store.set_audio_lifecycle(
                cid, audio_state=DELETION_FAILED,
                deletion_attempts=attempts,
                last_deletion_error=error,
                next_deletion_retry_ts=_now() +
                DELETE_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)))
            self.audit("audio_deletion_failed", cid, attempts=attempts,
                       error_type=error)
            return False

    def _delete_training(self, training_record_id: str, audio_path: str,
                         attempts: int) -> bool:
        path = Path(audio_path)
        attempts += 1
        try:
            if path.exists():
                path.unlink()
            self.store.update_training_record(
                training_record_id, deletion_status="deleted",
                deletion_attempts=attempts, last_deletion_error=None,
                next_deletion_retry_ts=None)
            self.audit("training_record_deleted",
                       cid=None, attempts=attempts,
                       training_record_id=training_record_id,
                       filename=path.name)
            return True
        except Exception as exc:
            error = f"{type(exc).__name__}"
            self.store.update_training_record(
                training_record_id, deletion_status="deletion_failed",
                deletion_attempts=attempts, last_deletion_error=error,
                next_deletion_retry_ts=_now() +
                DELETE_RETRY_BASE_SECONDS * (2 ** max(0, attempts - 1)))
            self.audit("training_deletion_failed", attempts=attempts,
                       training_record_id=training_record_id,
                       error_type=error)
            return False

    # -- training review staging -----------------------------------------------
    def review_training_candidate(self, training_record_id: str, decision: str,
                                  reviewer: str, note: str | None = None) -> dict:
        """Human review gate before a candidate enters the training corpus.

        approve  -> the candidate joins the dataset (its own retention
                    deadline continues to govern deletion).
        reject   -> the candidate never enters the corpus: the vault audio is
                    deleted immediately and the record is marked rejected.
        """
        rec = self.store.get_training_record(training_record_id)
        if not rec:
            raise KeyError(training_record_id)
        if rec["review_status"] != "pending":
            raise PermissionError("candidate_already_reviewed")
        if decision == "approve":
            updated = self.store.set_training_review(
                training_record_id, "approved", reviewer, note)
            self.audit("training_candidate_approved",
                       cid=rec["source_consultation_id"],
                       training_record_id=training_record_id, actor=reviewer)
            return updated
        if decision == "reject":
            self.store.set_training_review(
                training_record_id, "rejected", reviewer, note)
            path = Path(rec["audio_path"])
            try:
                if path.exists():
                    path.unlink()
                self.store.update_training_record(
                    training_record_id, deletion_status="deleted",
                    deletion_attempts=rec["deletion_attempts"] + 1)
                self.audit("training_candidate_rejected",
                           cid=rec["source_consultation_id"],
                           training_record_id=training_record_id, actor=reviewer)
                self.audit("training_record_deleted",
                           training_record_id=training_record_id,
                           filename=path.name, reason="rejected_by_reviewer")
            except Exception as exc:
                self.store.update_training_record(
                    training_record_id,
                    deletion_status="deletion_failed",
                    deletion_attempts=rec["deletion_attempts"] + 1,
                    last_deletion_error=type(exc).__name__)
                self.audit("training_deletion_failed",
                           training_record_id=training_record_id,
                           error_type=type(exc).__name__)
                raise
            return self.store.get_training_record(training_record_id)
        raise ValueError(f"Unknown review decision {decision!r}")

    def withdraw_training_consent(self, hospital_id: str, actor: str) -> int:
        """Propagate consent withdrawal across training data.

        Candidates not yet in the dataset (pending review) are deleted
        outright. Dataset records are flagged revoked so the dataset/version
        policy removes them; the audit trail records everything. Returns the
        number of affected records. Deleting from an already-TRAINED model is
        NOT claimed possible anywhere.
        """
        affected = 0
        for rec in self.store.list_training_records():
            if rec["deletion_status"] in ("deleted", "deletion_failed"):
                continue
            if rec["review_status"] == "pending":
                # Never entered the dataset: remove the candidate entirely.
                try:
                    path = Path(rec["audio_path"])
                    if path.exists():
                        path.unlink()
                    self.store.update_training_record(
                        rec["training_record_id"], deletion_status="deleted",
                        revoked=1)
                    self.audit("training_record_deleted",
                               training_record_id=rec["training_record_id"],
                               filename=rec["audio_filename"],
                               reason="consent_withdrawn_before_review")
                except Exception as exc:
                    self.store.update_training_record(
                        rec["training_record_id"], revoked=1,
                        deletion_status="deletion_failed",
                        last_deletion_error=type(exc).__name__)
                    self.audit("training_deletion_failed",
                               training_record_id=rec["training_record_id"],
                               error_type=type(exc).__name__)
            else:
                # In the dataset: flag for removal by the dataset policy.
                self.store.update_training_record(
                    rec["training_record_id"], revoked=1)
                self.audit("training_dataset_withdrawal_flagged",
                           training_record_id=rec["training_record_id"])
            affected += 1
        return affected

    # -- training path (fail closed) ---------------------------------------------
    def training_authorization_error(self, hospital_id: str) -> str | None:
        """Return a denial reason, or None when a CURRENT authorization exists.

        Ambiguity is denial: a missing config row, an unknown version, an
        expired authorization or a mismatched hospital all deny.
        """
        policy = self.policy
        cfg = self.store.get_hospital_config(hospital_id)
        # Revocation overrides everything and is checked first: a hospital
        # that revoked participation must never ingest again, even if a stale
        # authorization row is still marked authorized.
        if cfg and cfg.get("training_consent_revoked_at"):
            return "training_consent_revoked"
        if not policy.training_retention_enabled:
            return "training_retention_disabled"
        if policy.training_data_requires_hospital_authorization:
            if not cfg or not cfg.get("training_authorized"):
                return "no_training_authorization"
            expiry = cfg.get("training_authorization_expiry")
            if expiry is None:
                return "training_authorization_has_no_expiry"
            if float(expiry) <= _now():
                return "training_authorization_expired"
            if not cfg.get("training_authorization_version"):
                return "training_authorization_version_unknown"
            if cfg.get("hospital_id") and cfg["hospital_id"] != hospital_id:
                return "hospital_mismatch"
        return None

    def create_training_copy(self, cid: str, actor: str,
                             selection_reason: str | None = None) -> dict:
        """Move the approved clinical audio into the training vault.

        A MOVE, not a copy: after this there is exactly one copy, inside the
        authorized namespace, with its own retention deadline. The clinical
        record is marked purged — the doctor has already approved the note,
        so the review window is closed by construction.
        """
        rec = self.store.get(cid)
        if not rec:
            raise KeyError(cid)
        lifecycle = self.store.get_audio_lifecycle(cid) or {}
        hospital_id = lifecycle.get("hospital_id") or self.policy.hospital_id
        if lifecycle.get("hospital_id") and \
                lifecycle["hospital_id"] != self.policy.hospital_id:
            raise PermissionError("hospital_mismatch")

        denial = self.training_authorization_error(hospital_id)
        if denial:
            self.audit("training_selection_rejected", cid, actor=actor,
                       reason=denial)
            raise PermissionError(denial)

        if self.policy.training_data_requires_explicit_selection:
            if not (selection_reason or "").strip():
                raise PermissionError("explicit_selection_required")
        # Even when policy allows unselected intake, a selection event with
        # the authorization reference is recorded — never silent promotion.

        cfg = self.store.get_hospital_config(hospital_id) or {}
        authorization_reference = (
            f"{hospital_id}:{cfg.get('training_authorization_version') or 'unknown'}"
            f":{cfg.get('training_authorized_at') or 'unknown'}"
        )

        audio_path = Path(rec["audio_file"])
        if not audio_path.exists():
            raise FileNotFoundError("clinical audio missing; cannot create "
                                    "training copy")

        vault = self.settings.training_vault_dir
        vault.mkdir(parents=True, exist_ok=True)
        training_record_id = uuid.uuid4().hex[:16]
        vault_path = vault / f"{training_record_id}_{audio_path.name}"
        shutil.move(str(audio_path), str(vault_path))

        try:
            digest = sha256_of(vault_path)
            # Deterministic de-identification of the TRANSCRIPT COPY only.
            # The audio itself is NOT claimed de-identified (see module docs):
            # the candidate enters review staging as TRANSCRIPT_DEIDENTIFIED.
            result = rec.get("result") or {}
            transcript_text = (result.get("transcript") or {}).get("text", "") or ""
            names = [v for v in (rec.get("patient_id"), rec.get("doctor"),
                                 rec.get("department")) if v]
            deid_text, deid_provenance = deidentify_transcript(
                transcript_text, names=names)
            record = self.store.create_training_record({
                "training_record_id": training_record_id,
                "source_consultation_id": cid,
                "hospital_id": hospital_id,
                "audio_path": str(vault_path),
                "audio_filename": audio_path.name,
                "sha256": digest,
                "authorization_reference": authorization_reference,
                "retention_deadline": _now() +
                    self.policy.training_retention_period_hours * 3600,
                "created_at": _stamp(),
                "created_ts": _now(),
                "approved_for_training_at": _stamp(),
                "approved_by": actor or "shared-session",
                "selection_reason": (selection_reason or "").strip() or None,
                "deidentification_status": deid_provenance["status"],
                "deidentified_transcript": deid_text,
                "deidentification_provenance": json.dumps(
                    deid_provenance, sort_keys=True),
            })
            self.audit("training_deidentification_started", cid,
                       training_record_id=training_record_id,
                       status=deid_provenance["status"])

            # The clinical copy no longer exists — it was moved, not copied.
            self.store.mark_audio_purged(cid)
            self.store.set_audio_lifecycle(
                cid, audio_state=DELETED, audio_deletion_due_ts=None,
                training_record_id=training_record_id)
            self.audit("training_copy_created", cid, actor=actor,
                       training_record_id=training_record_id,
                       authorization_reference=authorization_reference,
                       retention_deadline=record["retention_deadline"],
                       sha256=digest)
            self.audit("audio_deleted", cid, attempts=0,
                       filename=audio_path.name,
                       reason="moved_to_training_vault")
            return record
        except Exception:
            # Do not leave the audio in the vault without its record: move it
            # back so the clinical record stays truthful.
            if not audio_path.exists() and vault_path.exists():
                shutil.move(str(vault_path), str(audio_path))
            raise
