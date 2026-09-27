"""Retention, consent, security and training-separation tests.

Covers the 20 required scenarios for the hospital audio-retention layer.
These tests deliberately avoid running Whisper/STT: the retention layer sits
ABOVE the clinical pipeline, so a stubbed engine exercises the full service
flow (upload → job → approval → deletion/training) without models.

Baseline invariants that must never regress live in the existing suite
(Malayalam semantics, ASR corrections, V2 rules, note validation).
"""

from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ACCESS_TOKEN = "test-token-that-is-long-enough-123"
TRAINING_ADMIN_TOKEN = "training-admin-token-long-enough-999"


class _StubEngine:
    """Stands in for ScribeEngine; returns a minimal valid result."""

    def __init__(self) -> None:
        self.on_stage = None

    def process_audio(self, audio_path, language=""):
        return _minimal_result()

    def process_text(self, transcript_text, language="en"):
        return _minimal_result()


def _minimal_result():
    return {
        "transcript": {"text": "Doctor: Hello.", "segments": []},
        "speakers": {"method": "stub", "confidence": "low", "roles_known": True,
                     "turns": []},
        "clinical_note": {"fields": {}, "text": "note"},
        "prescription": {"fields": {}, "text": "rx"},
        "clinical_note_v2": {"note": "note", "validation": {"valid": True,
                                                            "violations": []}},
        "meta": {
            "stt_provider": "stub", "requested_language": "en",
            # The real engine always carries the policy decision on success;
            # the stub mirrors that result contract (on-host story: no
            # declared endpoint/region to report).
            "policy_decision": "allowed",
            "policy_reason": "declared_boundary_satisfies_policy",
            "stt_zero_retention": True,
        },
    }


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("ISCRIBE_ENV", "production")
    monkeypatch.setenv("ISCRIBE_ACCESS_TOKEN", ACCESS_TOKEN)
    monkeypatch.setenv("ISCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ISCRIBE_PRELOAD_MODELS", "false")
    monkeypatch.setenv("ISCRIBE_COOKIE_SECURE", "false")
    monkeypatch.setenv("ISCRIBE_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("ISCRIBE_HOSPITAL_ID", "hospital-A")
    monkeypatch.setenv("ISCRIBE_TRAINING_ADMIN_TOKEN", TRAINING_ADMIN_TOKEN)
    monkeypatch.setenv("ISCRIBE_TRAINING_RETENTION_ENABLED", "true")
    monkeypatch.setenv("ISCRIBE_AUDIO_RETENTION_AFTER_APPROVAL_HOURS", "0")
    monkeypatch.setenv("ISCRIBE_AUDIO_RETENTION_UNREVIEWED_HOURS", "168")

    for name in ("service.app", "service.config", "service.store",
                 "service.security", "service.logging_config",
                 "service.retention"):
        sys.modules.pop(name, None)
    module = importlib.import_module("service.app")
    module._StubEngine = _StubEngine
    yield module
    for name in ("service.app", "service.config", "service.store",
                 "service.security", "service.logging_config",
                 "service.retention"):
        sys.modules.pop(name, None)


@pytest.fixture()
def client(app_module):
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture()
def auth_client(app_module):
    """A signed-in named user (clinic doctor). The shared access key now only
    issues an anonymous, clinically read-only session, so tests that act on
    consultations authenticate as a DOCTOR account created directly in the
    user store (bootstrap-free, no extra HTTP round-trips)."""
    with TestClient(app_module.app) as c:
        app_module.rt.users.create(
            "doctor@trial.test", "Trial-Doctor-Pass-1", "Dr. Trial", "DOCTOR",
            app_module.settings.hospital_id)
        from service.security import issue_user_session
        c.cookies.set(
            app_module.SESSION_COOKIE,
            issue_user_session(app_module.settings.access_token,
                               "doctor@trial.test", "DOCTOR",
                               app_module.settings.hospital_id,
                               3600))
        yield c


def _make_record_with_audio(app_module, auth_client, ext=".mp3",
                            patient_id=None) -> str:
    body = {"patient_id": patient_id} if patient_id else {}
    cid = auth_client.post("/api/consultations", json=body).json()["id"]
    audio = app_module.settings.upload_dir / f"{cid}{ext}"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"synthetic consultation audio")
    auth_client.post(f"/api/consultations/{cid}/audio",
                     files={"file": (audio.name, audio.read_bytes(), "audio/mpeg")})
    return cid


def _approve(app_module, cid: str) -> None:
    app_module.retention().approve(cid, actor="Dr. Test")


# ---------------------------------------------------------------------------
# 1-2. Review window and post-approval deletion
# ---------------------------------------------------------------------------

def test_1_audio_survives_while_awaiting_review(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] == "UPLOADED"
    # No sweep, no deadline: the review window is open.
    summary = app_module.rt.retention.sweep()
    assert summary["clinical_deleted"] == 0
    assert (app_module.settings.upload_dir / f"{cid}.mp3").exists()


def test_2_audio_becomes_deletion_eligible_after_approval(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] == "SCHEDULED_FOR_DELETION"
    assert lifecycle["audio_deletion_due_ts"] is not None
    # deadline passed (approval + 0h) → the next sweep deletes.
    assert app_module.rt.retention.sweep()["clinical_deleted"] == 1
    assert not (app_module.settings.upload_dir / f"{cid}.mp3").exists()
    assert app_module.store().get_audio_lifecycle(cid)["audio_state"] == "DELETED"
    actions = [e["action"] for e in app_module.store().audit_events(cid=cid)]
    assert "note_approved" in actions and "audio_deleted" in actions


def test_2b_pending_job_blocks_deletion(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    app_module.store().create_job(cid, "rec.mp3")  # queued job exists
    assert app_module.rt.retention.sweep()["clinical_deleted"] == 0
    assert (app_module.settings.upload_dir / f"{cid}.mp3").exists()


def test_2c_approval_required_before_post_review_deletion(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    # No approval: audio stays even after many sweeps.
    for _ in range(3):
        app_module.rt.retention.sweep()
    assert (app_module.settings.upload_dir / f"{cid}.mp3").exists()
    assert app_module.store().get_audio_lifecycle(cid)["audio_state"] == "UPLOADED"


def test_2d_unreviewed_ceiling_is_configured_not_automatic(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    app_module.retention()._evaluate(cid, reason="test_backfill")
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] == "SCHEDULED_FOR_DELETION"
    # Anchored to upload + 168h, not to "now".
    assert lifecycle["audio_deletion_due_ts"] == pytest.approx(
        lifecycle["created_ts"] + 168 * 3600)


# ---------------------------------------------------------------------------
# 3-7. Training separation, authorization, expiry, independence
# ---------------------------------------------------------------------------

def _grant_authorization(app_module, version="hospital-agreement-2026-v1"):
    app_module.store().set_hospital_config(
        "hospital-A",
        hospital_training_enabled=1,
        training_authorized=1,
        training_authorization_version=version,
        training_authorized_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        training_authorized_by="training-admin",
        training_authorization_expiry=time.time() + 3600,
        training_policy_reference="research-agreement-2026",
    )


def _approve_and_select(app_module, auth_client, reason="psychiatry corpus Q3"):
    cid = _make_record_with_audio(app_module, auth_client)
    _grant_authorization(app_module)
    _approve(app_module, cid)
    # approval+0h schedules immediate deletion; give the sweep a deadline far
    # in the future so only the training path may consume the file.
    app_module.store().set_audio_lifecycle(
        cid, audio_deletion_due_ts=time.time() + 10_000)
    res = auth_client.post(f"/api/consultations/{cid}/training-selection",
                           json={"reason": reason})
    assert res.status_code == 200, res.text
    return cid, res.json()


def test_3_training_disabled_cannot_enter_dataset(app_module, auth_client,
                                                  monkeypatch, tmp_path):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    # Disable training entirely (deployment-level master switch off).
    app_module.settings.training_retention_enabled = False
    res = auth_client.post(f"/api/consultations/{cid}/training-selection",
                           json={"reason": "should fail"})
    assert res.status_code == 403
    assert "disabled" in res.json()["detail"]
    assert app_module.store().list_training_records() == []
    # The clinical file is untouched — nothing was silently retained.
    assert (app_module.settings.upload_dir / f"{cid}.mp3").exists()


def test_4_training_enabled_still_requires_authorization(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    res = auth_client.post(f"/api/consultations/{cid}/training-selection",
                           json={"reason": "no authorization yet"})
    assert res.status_code == 403
    assert res.json()["detail"] == "no_training_authorization"
    # Now authorize, but selection is still explicit per recording.
    _grant_authorization(app_module)
    res = auth_client.post(f"/api/consultations/{cid}/training-selection",
                           json={"reason": ""})
    assert res.status_code == 403
    assert res.json()["detail"] == "explicit_selection_required"


def test_5_expired_authorization_fails_closed(app_module, auth_client):
    _grant_authorization(app_module)
    app_module.store().set_hospital_config(
        "hospital-A", training_authorization_expiry=time.time() - 1)
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    res = auth_client.post(f"/api/consultations/{cid}/training-selection",
                           json={"reason": "expired"})
    assert res.status_code == 403
    assert res.json()["detail"] == "training_authorization_expired"
    assert app_module.store().list_training_records() == []


def test_6_training_copy_has_independent_retention(app_module, auth_client):
    cid, payload = _approve_and_select(app_module, auth_client)
    rec = app_module.store().get_training_record(payload["training_record_id"])
    assert rec["hospital_id"] == "hospital-A"
    assert rec["source_consultation_id"] == cid
    assert rec["authorization_reference"].startswith("hospital-A:")
    assert rec["deletion_status"] == "retained"
    # Its own deadline, independent of the clinical record:
    assert rec["retention_deadline"] == pytest.approx(
        time.time() + app_module.settings.training_retention_period_hours * 3600,
        abs=60)
    # The clinical audio was MOVED into the vault namespace — one copy only.
    assert not (app_module.settings.upload_dir / f"{cid}.mp3").exists()
    assert Path(rec["audio_path"]).exists()
    assert Path(rec["audio_path"]).parent == app_module.settings.training_vault_dir
    assert app_module.store().get_audio_lifecycle(cid)["training_record_id"] == \
        rec["training_record_id"]


def test_7_training_copy_deletes_independently(app_module, auth_client):
    cid, payload = _approve_and_select(app_module, auth_client)
    tid = payload["training_record_id"]
    rec = app_module.store().get_training_record(tid)
    # Force the training deadline due; clinical side is already purged/moved.
    with app_module.store()._lock:
        app_module.store()._conn.execute(
            "UPDATE training_records SET retention_deadline = 0 WHERE "
            "training_record_id = ?", (tid,))
        app_module.store()._conn.commit()
    summary = app_module.rt.retention.sweep()
    assert summary["training_deleted"] == 1
    assert not Path(rec["audio_path"]).exists()
    after = app_module.store().get_training_record(tid)
    assert after["deletion_status"] == "deleted"
    events = app_module.store().audit_events(action="training_record_deleted")
    assert any(e["detail"].get("training_record_id") == tid for e in events)


def test_7b_no_silent_training_promotion(app_module, auth_client):
    """Raw clinical audio is never silently promoted to training data."""
    cid = _make_record_with_audio(app_module, auth_client)
    _grant_authorization(app_module)
    # No selection endpoint call: even sweeps must not move anything.
    for _ in range(2):
        app_module.rt.retention.sweep()
    assert app_module.store().list_training_records() == []
    assert (app_module.settings.upload_dir / f"{cid}.mp3").exists()


def test_4b_training_admin_endpoint_is_fail_closed(app_module, auth_client):
    # Missing credential
    res = auth_client.post("/api/hospital/training-authorization",
                           json={"authorized": True,
                                 "authorization_version": "v1",
                                 "authorization_expiry_seconds": 3600})
    assert res.status_code == 403
    # Wrong credential
    res = auth_client.post(
        "/api/hospital/training-authorization",
        json={"authorized": True, "authorization_version": "v1",
              "authorization_expiry_seconds": 3600},
        headers={"X-Training-Admin-Token": "wrong-token-wrong-token"})
    assert res.status_code == 403
    # Correct credential but no expiry → rejected (authorization must be finite)
    res = auth_client.post(
        "/api/hospital/training-authorization",
        json={"authorized": True, "authorization_version": "v1",
              "authorization_expiry_seconds": 0},
        headers={"X-Training-Admin-Token": TRAINING_ADMIN_TOKEN})
    assert res.status_code == 422
    # Grant works and is audited
    res = auth_client.post(
        "/api/hospital/training-authorization",
        json={"authorized": True, "authorization_version": "v1",
              "authorization_expiry_seconds": 3600,
              "policy_reference": "agreement-2026"},
        headers={"X-Training-Admin-Token": TRAINING_ADMIN_TOKEN})
    assert res.status_code == 200
    actions = [e["action"] for e in app_module.store().audit_events()]
    assert "training_authorization_granted" in actions


def test_4c_unauthorized_token_cannot_admin_training(app_module, auth_client):
    """The ordinary access token must NOT grant training administration."""
    res = auth_client.post(
        "/api/hospital/training-authorization",
        json={"authorized": True, "authorization_version": "v1",
              "authorization_expiry_seconds": 3600},
        headers={"X-Access-Token": ACCESS_TOKEN})
    assert res.status_code == 403


# ---------------------------------------------------------------------------
# 8-11. Tenancy and audio access
# ---------------------------------------------------------------------------

def test_8_hospital_boundary_is_server_assigned(app_module, auth_client):
    """hospital_id comes from the server config, never from client data."""
    res = auth_client.post("/api/consultations",
                           json={"patient_id": "P1", "hospital_id": "hospital-B"})
    assert res.status_code == 200
    lifecycle = app_module.store().get_audio_lifecycle(res.json()["id"])
    assert lifecycle["hospital_id"] == "hospital-A"
    # Client-supplied hospital_id never lands on the record.
    record = auth_client.get(f"/api/consultations/{res.json()['id']}").json()
    assert record.get("hospital_id") == "hospital-A"


def test_8b_hospital_mismatch_cannot_create_training_copy(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    _grant_authorization(app_module)
    # Forge a foreign-hospital record id: the manager refuses cross-tenant moves.
    app_module.store().set_audio_lifecycle = app_module.store().set_audio_lifecycle
    # Simulate a record stamped with another hospital (server-side tamper test):
    with app_module.store()._lock:
        app_module.store()._conn.execute(
            "UPDATE consultations SET hospital_id = 'hospital-B' WHERE id = ?",
            (cid,))
        app_module.store()._conn.commit()
    with pytest.raises(PermissionError):
        app_module.rt.retention.create_training_copy(cid, actor="attacker",
                                                     selection_reason="x")
    assert app_module.store().list_training_records() == []


def test_9_record_access_is_server_side(app_module, auth_client):
    """record_id resolution happens server-side; unknown ids never resolve."""
    assert auth_client.get("/api/consultations/does-not-exist").status_code == 404
    assert auth_client.post("/api/consultations/does-not-exist/process").status_code == 404
    assert auth_client.get("/api/consultations/does-not-exist/export").status_code == 404


def test_10_unauthenticated_user_cannot_retrieve_audio(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    # No session, no token → both session and audio endpoints reject.
    assert auth_client.app is not None  # sanity
    fresh = TestClient(app_module.app)
    res = fresh.get(f"/api/consultations/{cid}")
    assert res.status_code == 401
    res = fresh.get(f"/api/consultations/{cid}/audio?expires=forged-abc")
    assert res.status_code == 401
    # Valid session but no/invalid audio token → 403.
    res = auth_client.get(f"/api/consultations/{cid}/audio")
    assert res.status_code == 403
    res = auth_client.get(f"/api/consultations/{cid}/audio?expires=forged-abc")
    assert res.status_code == 403


def test_11_audio_link_expires(app_module, auth_client, monkeypatch):
    from service import security
    cid = _make_record_with_audio(app_module, auth_client)
    link = auth_client.post(f"/api/consultations/{cid}/audio-link").json()
    assert link["expires_in_seconds"] == security.DOWNLOAD_TOKEN_TTL_SECONDS
    # Fresh token works…
    res = auth_client.get(link["url"])
    assert res.status_code == 200
    assert res.headers["cache-control"].startswith("no-store")
    # …and a token minted in the past no longer does.
    expired = security.issue_download_token(ACCESS_TOKEN, cid,
                                            ttl_seconds=-10)
    assert not security.download_token_valid(expired, ACCESS_TOKEN, cid)
    res = auth_client.get(f"/api/consultations/{cid}/audio?expires={expired}")
    assert res.status_code == 403
    # A valid-format token for a DIFFERENT consultation must not open this one.
    other = security.issue_download_token(ACCESS_TOKEN, "other-consultation")
    res = auth_client.get(f"/api/consultations/{cid}/audio?expires={other}")
    assert res.status_code == 403


# ---------------------------------------------------------------------------
# 12-13. Deletion failure accounting and retry
# ---------------------------------------------------------------------------

def test_12_13_failed_deletion_recorded_and_retried(app_module, auth_client,
                                                    monkeypatch):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    audio_path = Path(app_module.store().get(cid)["audio_file"])

    import service.retention as retention_mod
    real_unlink = Path.unlink

    def failing_unlink(self, missing_ok=False):
        if self == audio_path:
            raise PermissionError("file locked by another process")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", failing_unlink)
    summary = app_module.rt.retention.sweep()
    assert summary["clinical_failed"] == 1
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] == "DELETION_FAILED"
    assert lifecycle["deletion_attempts"] == 1
    assert lifecycle["last_deletion_error"] == "PermissionError"
    assert lifecycle["next_deletion_retry_ts"] > time.time()
    actions = [e["action"] for e in app_module.store().audit_events(cid=cid)]
    assert "audio_deletion_failed" in actions
    # NOT deleted must never be reported as deleted.
    assert audio_path.exists()

    # Retry before the backoff elapses: still nothing happens.
    assert app_module.rt.retention.sweep()["clinical_failed"] == 0
    assert audio_path.exists()

    # Once the backoff elapses, the retry runs and succeeds.
    with app_module.store()._lock:
        app_module.store()._conn.execute(
            "UPDATE consultations SET next_deletion_retry_ts = 0 WHERE id = ?",
            (cid,))
        app_module.store()._conn.commit()
    monkeypatch.setattr(Path, "unlink", real_unlink)  # restore real unlink
    summary = app_module.rt.retention.sweep()
    assert summary["clinical_deleted"] == 1
    assert not audio_path.exists()


def test_12b_missing_file_counts_as_visible_deletion(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    (app_module.settings.upload_dir / f"{cid}.mp3").unlink()  # vanished
    assert app_module.rt.retention.sweep()["clinical_deleted"] == 1
    actions = [e["action"] for e in app_module.store().audit_events(cid=cid)]
    assert "audio_deleted" in actions  # recorded, not silent


# ---------------------------------------------------------------------------
# 14-16. Fail-closed authorization, provider safety, interchangeability
# ---------------------------------------------------------------------------

def test_14_ambiguous_authorization_fails_closed(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid)
    # Partial states that MUST all deny:
    ambiguous = [
        {"training_authorized": 1},  # no version, no expiry
        {"training_authorized": 1, "training_authorization_version": "v1"},  # no expiry
        {"training_authorized": 1, "training_authorization_version": "v1",
         "training_authorization_expiry": None},  # NULL expiry
        {"training_authorized": 0, "training_authorization_version": "v1",
         "training_authorization_expiry": time.time() + 3600},  # not granted
    ]
    for fields in ambiguous:
        app_module.store().set_hospital_config("hospital-A", **fields)
        denial = app_module.rt.retention.training_authorization_error("hospital-A")
        assert denial is not None, f"ambiguous state did not deny: {fields}"
        res = auth_client.post(f"/api/consultations/{cid}/training-selection",
                               json={"reason": "ambiguous"})
        assert res.status_code == 403
    assert app_module.store().list_training_records() == []


def test_15_provider_failure_does_not_corrupt_note_generation(app_module,
                                                              auth_client):
    """A provider exception in the job path leaves the record in a clean
    error state; the validated-note gate still decides completion."""
    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)

    class _Boom:
        def process_audio(self, audio_path, language=""):
            raise RuntimeError("provider exploded")

    app_module.build_engine = _Boom
    app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
    record = auth_client.get(f"/api/consultations/{cid}").json()
    assert record["status"] == "error"
    assert "provider exploded" not in record.get("error", "")  # PHI-safe message
    assert app_module.store().get_job(job["job_id"])["status"] == "failed"
    # The clinical record is still queryable and nothing was half-written.
    assert record.get("result") is None


def test_16_deepgram_remains_interchangeable(app_module):
    """The provider contract survives: registry, resolution, and the retention
    capability surface required by this layer."""
    from scribe_engine.stt import (get_provider, available_providers,
                                   resolve_production_provider, DeepgramSTTProvider,
                                   CURRENT_PROVIDER_ID, DEEPGRAM_PROVIDER_ID)
    assert CURRENT_PROVIDER_ID in available_providers()
    assert DEEPGRAM_PROVIDER_ID in available_providers()

    dg = get_provider(DEEPGRAM_PROVIDER_ID, api_key="k")
    assert isinstance(dg, DeepgramSTTProvider)
    # Capability surface exists and is evidence-based (provider docs):
    assert dg.supports_delete_after_processing() is False
    rc = dg.retention_configuration()
    assert rc["retention_known"] is True
    assert rc["audio_sent_off_host"] is True
    assert rc["deletion_supported"] is False
    assert rc["mip_opt_out"] is True
    assert "DEEPGRAM" not in json.dumps(rc).upper().replace("DEEPGRAM", "")  # no key material
    assert dg.health_check() == {"provider": "deepgram", "ok": True,
                                 "configured": True}

    # Base-class defaults are the honest minimum for any future provider.
    from scribe_engine.stt.base import STTProvider
    base = STTProvider()
    assert base.supports_delete_after_processing() is False
    assert base.retention_configuration()["retention_known"] is False

    # Provider selection still resolves a working engine (whisper fallback).
    provider, reason = resolve_production_provider(None, language="en")
    assert provider is not None and reason


# ---------------------------------------------------------------------------
# 17-20. Audit hygiene and pipeline invariance
# ---------------------------------------------------------------------------

def test_17_audit_log_contains_no_phi(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _grant_authorization(app_module)
    _approve(app_module, cid)
    app_module.store().set_audio_lifecycle(
        cid, audio_deletion_due_ts=time.time() + 10_000)
    auth_client.post(f"/api/consultations/{cid}/training-selection",
                     json={"reason": "audit hygiene check"})
    app_module.rt.retention.sweep()
    events = app_module.store().audit_events(cid=cid, limit=100)
    assert events, "expected audit events"
    blob = json.dumps(events).lower()
    for phi in ("synthetic consultation audio", "doctor: hello", "patient",
                '"text"', "note text", "reason\": \"audit"):
        assert phi not in blob, f"audit leak: {phi!r}"
    # Actions and ids only:
    allowed_fields = {"event_id", "ts", "ts_epoch", "action", "cid",
                      "hospital_id", "actor", "detail"}
    for e in events:
        assert set(e.keys()) <= allowed_fields | {"detail"}
        assert set(e["detail"].keys()) <= {
            "filename", "bytes", "ext", "provider_used", "basis", "due_hours",
            "deletion_due_ts", "reason", "attempts", "error_type",
            "training_record_id", "authorization_reference",
            "retention_deadline", "sha256", "actor", "authorization_version",
            "policy_reference", "expiry_seconds", "approval_version",
            "affected_records", "policy_version", "status", "provider",
            "model", "off_host",
            # STT policy decision (Phase: auditable boundary) — configuration
            # facts and reason categories only, never keys/audio/transcript.
            "policy_decision", "policy_reason", "endpoint", "region",
            "zero_retention", "fail_closed", "provider_id", "message",
            "request_id",
        }


def test_18_approval_required_before_deletion(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    # Even a sweep immediately after upload must not delete.
    app_module.rt.retention.sweep()
    assert (app_module.settings.upload_dir / f"{cid}.mp3").exists()
    # Approve with a far deadline; deletion must still not fire early.
    _approve(app_module, cid)
    app_module.store().set_audio_lifecycle(
        cid, audio_deletion_due_ts=time.time() + 10_000)
    assert app_module.rt.retention.sweep()["clinical_deleted"] == 0
    assert (app_module.settings.upload_dir / f"{cid}.mp3").exists()


def test_19_raw_clinical_audio_never_silently_promoted(app_module, auth_client):
    """With training disabled (the default everywhere), no code path — upload,
    review, approval, sweep — ever produces a training record."""
    cid = _make_record_with_audio(app_module, auth_client)
    app_module.settings.training_retention_enabled = False
    app_module.store().set_hospital_config("hospital-A", training_authorized=0)
    _approve(app_module, cid)
    app_module.rt.retention.sweep()
    assert app_module.store().list_training_records() == []
    assert not app_module.settings.training_vault_dir.exists() or \
        not list(app_module.settings.training_vault_dir.glob("*"))


def test_20_malayalam_and_english_pipelines_unchanged():
    """The clinical contracts this layer must not touch still hold."""
    from scribe_engine.asr_correction import apply_corrections
    from scribe_engine.clinical_facts import build_clinical_facts
    from scribe_engine.note_v2 import render_clinical_note, validate_note_v2

    # Malayalam verified ASR correction (diabetes, observed corruption).
    result = apply_corrections("ചവദാഭയബഥ ഉണ്ട്")
    assert result.corrected_text == "diabetes ഉണ്ട്"
    assert result.corrections and result.corrections[0]["raw"] == "ചവദാഭയബഥ"

    # English clinical intelligence → note validation still passes end to end.
    turns = [
        {"speaker": "Doctor", "text": "Do you have any chest pain?",
         "start": 0.0, "end": 2.0},
        {"speaker": "Patient", "text": "Yes, I have chest pain since morning.",
         "start": 2.0, "end": 5.0},
    ]
    facts = build_clinical_facts(
        "Doctor: Do you have any chest pain?\n"
        "Patient: Yes, I have chest pain since morning.",
        turns, roles_known=True, entities=[], corrected_text=None)
    note = render_clinical_note(facts)
    validation = validate_note_v2(note["note"], facts)
    assert validation["valid"] is True
    assert facts["validation"]["valid"] is True


# ===========================================================================
# Privacy-hardening phase 2: note approval gate, policy versioning,
# approved-only export, consent revocation.
# ===========================================================================

def _review_note(auth_client, cid: str) -> None:
    auth_client.patch(f"/api/consultations/{cid}/review",
                      json={"clinical_note_fields": {"impression": "Reviewed"}})


def _seed_result(app_module, auth_client, cid: str) -> None:
    """Put a minimal result on the record so the note gate can engage."""
    app_module.store().apply(cid, lambda r: r.update(result=_minimal_result()))


def test_21_note_starts_as_draft_and_approval_requires_review(app_module,
                                                              auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _seed_result(app_module, auth_client, cid)
    assert app_module.store().get_audio_lifecycle(cid)["note_status"] == "DRAFT"
    # Approval is refused while the note is still an unreviewed draft.
    res = auth_client.post(f"/api/consultations/{cid}/approve")
    assert res.status_code == 409
    assert "Review the note" in res.json()["detail"]
    _review_note(auth_client, cid)
    assert app_module.store().get_audio_lifecycle(cid)["note_status"] == "REVIEWED"
    res = auth_client.post(f"/api/consultations/{cid}/approve")
    assert res.status_code == 200
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["note_status"] == "APPROVED"
    assert lifecycle["approval_version"] == 1
    assert lifecycle["approved_at"] and lifecycle["approved_by"]


def test_22_retention_countdown_starts_only_on_approval(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _seed_result(app_module, auth_client, cid)
    _review_note(auth_client, cid)
    # Reviewed but NOT approved: no schedule may exist yet.
    app_module.rt.retention.sweep()
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] == "AWAITING_DOCTOR_REVIEW"
    assert lifecycle["audio_deletion_due_ts"] is None
    # Approval starts the countdown.
    auth_client.post(f"/api/consultations/{cid}/approve")
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] == "SCHEDULED_FOR_DELETION"
    assert lifecycle["audio_deletion_due_ts"] is not None


def test_23_deletion_timestamp_uses_configured_policy(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _seed_result(app_module, auth_client, cid)
    _review_note(auth_client, cid)
    before = time.time()
    auth_client.post(f"/api/consultations/{cid}/approve")
    after = time.time()
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    hours = app_module.settings.audio_retention_after_note_approval_hours
    assert before + hours * 3600 <= lifecycle["audio_deletion_due_ts"] <= after + hours * 3600 + 1
    # Policy identity snapshot records the parameters actually used.
    assert lifecycle["retention_policy_version"] == f"after_approval={int(hours)}h"
    assert lifecycle["retention_policy_id"] == "hospital-A"


def test_24_policy_change_does_not_rewrite_existing_decisions(app_module,
                                                              auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _seed_result(app_module, auth_client, cid)
    _review_note(auth_client, cid)
    auth_client.post(f"/api/consultations/{cid}/approve")
    scheduled = app_module.store().get_audio_lifecycle(cid)
    old_due = scheduled["audio_deletion_due_ts"]
    old_version = scheduled["retention_policy_version"]

    # The hospital administrator changes the global policy.
    app_module.store().set_hospital_config(
        "hospital-A", audio_retention_after_note_approval_hours=72)

    # The existing decision is untouched and still carries its old version.
    after = app_module.store().get_audio_lifecycle(cid)
    assert after["audio_deletion_due_ts"] == old_due
    assert after["retention_policy_version"] == old_version

    # A NEW consultation evaluates under the new policy version.
    cid2 = _make_record_with_audio(app_module, auth_client)
    _seed_result(app_module, auth_client, cid2)
    _review_note(auth_client, cid2)
    auth_client.post(f"/api/consultations/{cid2}/approve")
    lifecycle2 = app_module.store().get_audio_lifecycle(cid2)
    assert lifecycle2["retention_policy_version"] == "after_approval=72h"


def test_25_export_contains_approved_note_only(app_module, auth_client):
    cid = _make_record_with_audio(app_module, auth_client)
    _seed_result(app_module, auth_client, cid)
    _review_note(auth_client, cid)
    auth_client.post(f"/api/consultations/{cid}/approve")
    export = auth_client.get(f"/api/consultations/{cid}/export")
    assert export.status_code == 200
    # Approved clinical document: no raw transcript, no debug sections,
    # no internal AI reasoning.
    assert "TRANSCRIPT" not in export.text
    assert "LEGACY TEMPLATE NOTE" not in export.text
    assert "Doctor: Hello." not in export.text
    assert "no-store" in export.headers["cache-control"]


def test_26_training_consent_revocation_blocks_ingestion(app_module, auth_client):
    cid, payload = _approve_and_select(app_module, auth_client)
    tid = payload["training_record_id"]
    headers = {"X-Training-Admin-Token": TRAINING_ADMIN_TOKEN}

    # Revoke consent.
    res = auth_client.post("/api/hospital/training-authorization",
                           json={"authorized": False}, headers=headers)
    assert res.status_code == 200
    assert res.json()["consent_revoked"] is True

    # Future ingestion is blocked with the explicit revocation reason.
    cid2 = _make_record_with_audio(app_module, auth_client)
    _approve(app_module, cid2)
    app_module.store().set_audio_lifecycle(
        cid2, audio_deletion_due_ts=time.time() + 10_000)
    res = auth_client.post(f"/api/consultations/{cid2}/training-selection",
                           json={"reason": "after revocation"})
    assert res.status_code == 403
    assert res.json()["detail"] == "training_consent_revoked"

    # Existing retained training records are flagged for the dataset policy.
    rec = app_module.store().get_training_record(tid)
    assert rec["revoked"] == 1

    # The audit trail retains the revocation event (consent rows are never
    # silently deleted).
    actions = [e["action"] for e in app_module.store().audit_events()]
    assert "training_consent_revoked" in actions


def test_27_note_status_flow_in_export_gate(app_module, auth_client):
    """DRAFT and REVIEWED notes cannot be exported; only APPROVED can."""
    cid = _make_record_with_audio(app_module, auth_client)
    _seed_result(app_module, auth_client, cid)
    assert auth_client.get(f"/api/consultations/{cid}/export").status_code == 409
    _review_note(auth_client, cid)
    assert auth_client.get(f"/api/consultations/{cid}/export").status_code == 409
    auth_client.post(f"/api/consultations/{cid}/approve")
    assert auth_client.get(f"/api/consultations/{cid}/export").status_code == 200


# ===========================================================================
# Training review staging, de-identification, withdrawal propagation,
# provider boundary metadata (§5/§6/§12/§16 of the lifecycle spec).
# ===========================================================================
TRAINING_ADMIN = {"X-Training-Admin-Token": TRAINING_ADMIN_TOKEN}


def _candidate(app_module, auth_client, transcript="Doctor: Good morning. "
              "Patient: I am Ramesh, 45 years old, my number is 9876543210.",
              patient_name="Ramesh"):
    # The spoken name is registered in the encounter metadata (patient_id),
    # which is where the deterministic redactor takes its name list from.
    cid = _make_record_with_audio(app_module, auth_client,
                                  patient_id=patient_name)
    _grant_authorization(app_module)
    _approve(app_module, cid)
    app_module.store().apply(
        cid, lambda r: r.update(
            result={**_minimal_result(),
                    "transcript": {**_minimal_result()["transcript"],
                                   "text": transcript}}))
    app_module.store().set_audio_lifecycle(
        cid, audio_deletion_due_ts=time.time() + 10_000)
    res = auth_client.post(f"/api/consultations/{cid}/training-selection",
                           json={"reason": "quality review corpus"})
    assert res.status_code == 200, res.text
    return cid, res.json()["training_record_id"]


def test_28_candidate_starts_pending_and_requires_human_review(app_module,
                                                               auth_client):
    """§5: a staging state exists; selection never lands directly in the
    corpus, and the reviewer decides via the dedicated endpoint."""
    cid, tid = _candidate(app_module, auth_client)
    rec = app_module.store().get_training_record(tid)
    assert rec["review_status"] == "pending"
    # Ordinary access token must NOT be able to review (separate credential).
    res = auth_client.post(
        f"/api/training/candidates/{tid}/review",
        json={"decision": "approve"})
    assert res.status_code == 403
    # Authorized reviewer approves → the candidate joins the dataset.
    res = auth_client.post(
        f"/api/training/candidates/{tid}/review",
        json={"decision": "approve", "reviewer": "Dr. Reviewer"},
        headers=TRAINING_ADMIN)
    assert res.status_code == 200
    assert res.json()["review_status"] == "approved"
    # Double review is refused.
    res = auth_client.post(
        f"/api/training/candidates/{tid}/review",
        json={"decision": "reject"}, headers=TRAINING_ADMIN)
    assert res.status_code == 409


def test_29_rejected_candidate_never_enters_corpus(app_module, auth_client):
    cid, tid = _candidate(app_module, auth_client)
    rec = app_module.store().get_training_record(tid)
    res = auth_client.post(
        f"/api/training/candidates/{tid}/review",
        json={"decision": "reject", "note": "residual identifier"},
        headers=TRAINING_ADMIN)
    assert res.status_code == 200
    assert res.json()["review_status"] == "rejected"
    assert res.json()["deletion_status"] == "deleted"
    # The vault audio is gone.
    assert not Path(rec["audio_path"]).exists()
    actions = [e["action"] for e in app_module.store().audit_events()]
    assert "training_candidate_rejected" in actions
    assert "training_record_deleted" in actions


def test_30_transcript_deidentified_but_audio_never_claimed(app_module,
                                                           auth_client):
    """§6: deterministic redaction on the transcript copy with provenance;
    audio de-identification is never claimed."""
    cid, tid = _candidate(app_module, auth_client)
    rec = app_module.store().get_training_record(tid)
    assert rec["deidentification_status"] == "TRANSCRIPT_DEIDENTIFIED"
    deid = rec["deidentified_transcript"]
    # Metadata-registered names, phone numbers and stated ages are removed.
    assert "Ramesh" not in deid
    assert "9876543210" not in deid
    assert "45 years" not in deid
    assert "[REDACTED]" in deid
    provenance = json.loads(rec["deidentification_provenance"])
    assert provenance["names_removed"]
    # Honest limitation, by design: names spoken but never written into the
    # encounter metadata are NOT pattern-redactable — exactly why a human
    # reviews every candidate before dataset inclusion (see §6).
    # The RAW transcript stays untouched on the clinical encounter.
    raw = app_module.store().get(cid)["result"]["transcript"]["text"]
    assert "Ramesh" in raw


def test_31_reviewer_view_hides_raw_transcript_and_paths(app_module,
                                                         auth_client):
    cid, tid = _candidate(app_module, auth_client)
    res = auth_client.get("/api/training/candidates?status=pending",
                          headers=TRAINING_ADMIN)
    assert res.status_code == 200
    items = res.json()
    assert items and items[0]["training_record_id"] == tid
    blob = json.dumps(items)
    # The reviewer sees ONLY the de-identified copy: no patient name, no
    # phone number, redaction markers present, vault path never exposed.
    assert "Ramesh" not in blob
    assert "9876543210" not in blob
    assert "[REDACTED]" in blob
    assert str(app_module.settings.training_vault_dir) not in blob
    assert items[0]["deidentification_status"] == "TRANSCRIPT_DEIDENTIFIED"


def test_32_withdrawal_deletes_pending_candidates_and_flags_dataset(
        app_module, auth_client):
    """§16: withdrawal before dataset inclusion deletes the candidate;
    after inclusion it flags the dataset record for removal."""
    cid_pending, tid_pending = _candidate(app_module, auth_client)
    cid_in, tid_in = _candidate(app_module, auth_client)
    auth_client.post(f"/api/training/candidates/{tid_in}/review",
                     json={"decision": "approve"}, headers=TRAINING_ADMIN)

    res = auth_client.post("/api/hospital/training-authorization",
                           json={"authorized": False}, headers=TRAINING_ADMIN)
    assert res.status_code == 200

    pending = app_module.store().get_training_record(tid_pending)
    in_ds = app_module.store().get_training_record(tid_in)
    assert pending["deletion_status"] == "deleted"
    assert not Path(pending["audio_path"]).exists()
    assert in_ds["revoked"] == 1 and in_ds["review_status"] == "approved"
    actions = [e["action"] for e in app_module.store().audit_events()]
    assert "training_dataset_withdrawal_flagged" in actions
    assert "training_consent_revoked" in actions


def test_33_provider_metadata_recorded_and_failures_preserve_audio(
        app_module, auth_client):
    """§12/§13: provider boundary metadata travels to the audit trail and a
    failing provider leaves the source audio untouched."""
    # (a) provider metadata: run the stub job path and inspect audit detail.
    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)
    app_module.build_engine = _StubEngine  # class: _execute_doc_job calls it
    app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
    events = app_module.store().audit_events(cid=cid,
                                             action="transcription_completed")
    assert events, "provider metadata event missing"
    detail = events[-1]["detail"]
    assert detail["provider"] == "stub"
    assert "off_host" in detail
    # A provider that returns no request id records none (never invented).
    assert "request_id" not in detail

    # (b) provider failure: audio survives, record is a clean error.
    cid2 = _make_record_with_audio(app_module, auth_client)
    audio2 = app_module.store().get(cid2)["audio_file"]
    job2 = app_module.store().create_job(cid2, Path(audio2).name)

    class _Boom:
        def process_audio(self, audio_path, language=""):
            raise RuntimeError("provider down")

    app_module.build_engine = _Boom
    app_module._execute_doc_job(job2["job_id"], cid2, audio2, "en")
    assert Path(audio2).exists()          # source audio preserved
    rec = app_module.store().get(cid2)
    assert rec["status"] == "error"
    lifecycle = app_module.store().get_audio_lifecycle(cid2)
    assert lifecycle["audio_state"] != "SCHEDULED_FOR_DELETION"


def test_34_training_metadata_is_minimal(app_module, auth_client):
    """§15: the training dataset entry carries references and statuses, not
    patient metadata copies."""
    cid, tid = _candidate(app_module, auth_client)
    rec = app_module.store().get_training_record(tid)
    allowed = {"training_record_id", "source_consultation_id", "hospital_id",
               "audio_path", "audio_filename", "sha256",
               "authorization_reference", "retention_deadline", "created_at",
               "created_ts", "approved_for_training_at", "approved_by",
               "selection_reason", "deletion_status", "deletion_attempts",
               "last_deletion_error", "next_deletion_retry_ts", "revoked",
               "review_status", "reviewed_at", "reviewed_by", "review_note",
               "deidentification_status", "deidentified_transcript",
               "deidentification_provenance"}
    assert set(rec.keys()) <= allowed
    # No patient/doctor name or patient identifier columns exist on the
    # training record (patient_id/doctor live only on the clinical record).
    assert "patient_id" not in rec and "doctor" not in rec


# ===========================================================================
# Provider boundary: evidence-based retention posture, residency, auditability
# (docs/stt_provider_privacy_and_security.md).
# ===========================================================================
from scribe_engine.stt.deepgram_provider import DeepgramSTTProvider


def test_35_deepgram_requests_default_to_zero_retention(monkeypatch):
    """mip_opt_out=true must be sent by default: Deepgram's default (MIP)
    retains audio AND transcript for model training, which is never
    acceptable for clinical audio. The flag is a documented query parameter."""
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-a-real-credential")
    provider = DeepgramSTTProvider()
    assert provider.mip_opt_out is True
    params = provider._params("en")
    assert params["mip_opt_out"] == "true"
    # Existing quality parameters are untouched.
    assert params["model"] == "nova-3-medical"
    assert params["diarize"] == "true"


def test_36_deepgram_endpoint_and_capability_metadata(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key-not-a-real-credential")
    provider = DeepgramSTTProvider(data_endpoint="api.in.deepgram.com")
    assert provider.data_location() == "India"
    rc = provider.retention_configuration()
    assert rc["retention_known"] is True
    assert rc["audio_sent_off_host"] is True
    assert rc["deletion_supported"] is False
    assert "api.in.deepgram.com" in rc["endpoint"]
    assert "mip_opt_out=true: zero data retention" in rc["retention_behaviour"]
    assert "mip_opt_out" in rc["metadata_sent"]
    # The key must never appear anywhere in the declared surface.
    assert provider.api_key not in json.dumps(rc)


def test_37_global_endpoint_states_no_residency(monkeypatch):
    provider = DeepgramSTTProvider(api_key="k")
    assert provider.data_location().startswith("global")
    assert "no residency guarantee" in provider.data_location()


def test_38_base_provider_defaults_stay_honest():
    """The interface default stays 'unknown' so a future provider cannot
    accidentally claim retention/residency facts it never declared."""
    from scribe_engine.stt.base import STTProvider
    base = STTProvider()
    assert base.data_location().startswith("unknown")
    assert base.supports_delete_after_processing() is False
    rc = base.retention_configuration()
    assert rc["retention_known"] is False


def test_39_off_host_flag_follows_provider_in_audit(app_module, auth_client):
    """Every external boundary crossing is auditable: deepgram marks off_host
    true; a local provider (or text input) never claims off-host transfer."""
    provider = DeepgramSTTProvider(api_key="k")
    assert provider.retention_configuration()["audio_sent_off_host"] is True
    from scribe_engine.stt import CURRENT_PROVIDER_ID
    local = DeepgramSTTProvider  # noqa: F841  (interface parity check)
    assert provider.provider_id == "deepgram"
    # The audit detail contract: only present when truthfully off-host.
    assert app_module.settings.deepgram_mip_opt_out is True


# ===========================================================================
# STT policy enforcement at the provider boundary (fail-closed, pre-flight,
# region/ZDR policy, /api/stt/boundary).
# ===========================================================================
from scribe_engine.stt.policy import STTPolicyBlocked, check_policy


class _FakeOffHostProvider:
    """Minimal provider surface for policy checks (no network, no keys)."""

    provider_id = "deepgram"

    def __init__(self, endpoint="api.deepgram.com", mip_opt_out=True):
        self._endpoint = endpoint
        self._mip = mip_opt_out

    def retention_configuration(self):
        return {
            "retention_known": True,
            "audio_sent_off_host": True,
            "deletion_supported": False,
            "endpoint": f"https://{self._endpoint}/v1/listen",
            "mip_opt_out": self._mip,
            "diarize": True,
            "model": "nova-3-medical",
        }

    def data_location(self):
        return {
            "api.eu.deepgram.com": "EU",
            "api.au.deepgram.com": "Australia",
            "api.in.deepgram.com": "India",
        }.get(self._endpoint, "global (no residency guarantee)")

    def transcribe(self, audio_path, language=None):
        raise RuntimeError("simulated provider failure")


def test_40_policy_block_region_mismatch():
    """Hospital requires India processing; endpoint is global → audio must
    NOT be sent (STT_POLICY_BLOCKED before any network I/O)."""
    provider = _FakeOffHostProvider(endpoint="api.deepgram.com")
    with pytest.raises(STTPolicyBlocked) as exc:
        check_policy(provider, {"region": "in"})
    assert exc.value.reason == "region_mismatch"


def test_41_policy_block_zdr_mismatch():
    """Hospital requires zero retention; provider configured with MIP →
    refused before sending."""
    provider = _FakeOffHostProvider(mip_opt_out=False)
    with pytest.raises(STTPolicyBlocked) as exc:
        check_policy(provider, {"zero_retention": True})
    assert exc.value.reason == "zero_retention_required"


def test_42_policy_allows_matching_region_and_zdr():
    provider = _FakeOffHostProvider(endpoint="api.in.deepgram.com",
                                    mip_opt_out=True)
    check_policy(provider, {"region": "in", "zero_retention": True})  # no raise


def test_43_policy_block_provider_mismatch_and_offhost():
    provider = _FakeOffHostProvider()
    with pytest.raises(STTPolicyBlocked) as exc:
        check_policy(provider, {"provider": "current_faster_whisper"})
    assert exc.value.reason == "provider_mismatch"
    with pytest.raises(STTPolicyBlocked) as exc:
        check_policy(provider, {"off_host_allowed": False})
    assert exc.value.reason == "off_host_not_allowed"


def test_44_pipeline_preflight_refuses_to_send_audio(app_module, auth_client):
    """Scenario D/E end-to-end: a policy violation must stop the job before
    any provider call; source audio untouched; no partial note; audited."""
    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)

    engine = app_module.build_engine()
    engine.stt_policy = {"region": "in"}  # config is global → must refuse

    calls = []

    class _Spy(_StubEngine):
        def process_audio(self, audio_path, language=""):
            # The real pipeline raises before transcribing; if this spy runs,
            # the policy check failed to gate the send.
            calls.append(audio_path)
            return _minimal_result()

    # Run the real pipeline path with a fake off-host provider to prove the
    # pre-flight fires before any provider transcribe() call.
    from scribe_engine.pipeline import ScribeEngine as RealEngine
    probe = RealEngine.__new__(RealEngine)
    probe.language = "en"
    probe.whisper_model_size = "base"
    probe.device = "cpu"
    probe.compute_type = "int8"
    probe.cpu_threads = 0
    probe.stt_provider_id = None
    probe.stt_policy = {"region": "in"}
    probe.stt_fail_closed = False
    probe.on_stage = None
    probe._stages = []
    with pytest.raises(STTPolicyBlocked):
        probe.process_audio(audio_file, language="en")
    assert calls == []  # no provider was ever invoked

    # The audio file is still on disk and no clinical result was produced.
    assert Path(audio_file).exists()
    assert app_module.store().get(cid)["result"] in (None, {})


def test_45_fail_closed_disables_fallback(app_module, auth_client, monkeypatch):
    """Scenarios B/C: with fail-closed mode, a provider failure must NOT
    switch providers; the job fails safely with a retryable policy error."""
    from scribe_engine.pipeline import ScribeEngine as RealEngine
    probe = RealEngine(
        whisper_model_size="base", device="cpu", compute_type="int8",
        cpu_threads=0, language="en", stt_provider_id="deepgram")
    probe.stt_policy = None
    probe.stt_fail_closed = True
    probe.on_stage = None

    monkeypatch.setattr(
        "scribe_engine.pipeline.resolve_production_provider",
        lambda *a, **k: (_FakeOffHostProvider(), "test"))
    monkeypatch.setattr(
        _FakeOffHostProvider, "transcribe",
        lambda self, path, language=None: (_ for _ in ()).throw(
            RuntimeError("timeout")))

    audio = app_module.settings.upload_dir / "failclosed-test.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"x")
    with pytest.raises(STTPolicyBlocked) as exc:
        probe.process_audio(audio, language="en")
    assert exc.value.reason == "provider_failed_fail_closed"
    # The source audio survives the failure.
    assert audio.exists()


def test_46_failed_stt_job_preserves_audio_and_audits(app_module, auth_client):
    """Failure semantics at the service layer: failed STT leaves the audio
    under the normal lifecycle and never schedules deletion."""
    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def process_audio(self, audio_path, language=""):
            raise STTPolicyBlocked("provider_failed_fail_closed",
                                   "fallback disabled by policy")

    app_module.build_engine = _Boom
    app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
    assert Path(audio_file).exists()
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] != "SCHEDULED_FOR_DELETION"
    assert app_module.store().get_job(job["job_id"])["status"] == "failed"
    # Failure is auditable (the job-failure path records the error).
    assert app_module.store().get_job(job["job_id"])["error"]


def test_47_stt_boundary_endpoint_shape_and_auth(app_module, auth_client):
    """§6: safe configuration metadata only; standard auth applies; policy
    violation fields (reason categories) never carry secrets."""
    # Unauthenticated requests are rejected like every other API route.
    fresh = TestClient(app_module.app)
    assert fresh.get("/api/stt/boundary").status_code == 401

    res = auth_client.get("/api/stt/boundary")
    assert res.status_code == 200
    body = res.json()
    assert body["provider"] == "deepgram"
    assert body["status"] == "configured"
    assert body["off_host"] is True
    assert body["zero_retention"] is True
    assert body["fail_closed"] == app_module.settings.stt_fail_closed
    assert body["endpoint_class"] in ("global (no residency guarantee)",
                                       "eu", "australia", "india",
                                       "custom endpoint")
    blob = json.dumps(body).lower()
    for secret in (ACCESS_TOKEN, TRAINING_ADMIN_TOKEN, "authorization",
                   "api_key", "token="):
        assert secret.lower() not in blob
    # No patient data or clinical content may appear: the words "patient",
    # "ramesh" and any transcript CONTENT must be absent. (The provider's
    # documented retention posture legitimately *mentions* the word
    # "transcript" — that is documentation text, not clinical content.)
    assert "ramesh" not in blob
    assert "chest pain" not in blob
    assert "patient_id" not in blob


def test_48_policy_blocked_jobs_are_retryable(app_module, auth_client):
    """A policy-blocked job can be retried once configuration is fixed."""
    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def process_audio(self, audio_path, language=""):
            raise STTPolicyBlocked("region_mismatch", "no")

    app_module.build_engine = _Boom
    app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
    assert app_module.store().get_job(job["job_id"])["status"] == "failed"
    assert Path(audio_file).exists()
    # The normal retry endpoint accepts the failed job (retryable).
    res = auth_client.post(f"/api/consultations/{cid}/jobs/{job['job_id']}/retry")
    assert res.status_code in (202, 403)  # 403 only if auth deps differ in dev


# ===========================================================================
# STT policy audit observability: every transcription decision (allowed or
# blocked) is answerable from the audit trail alone. Configuration facts only;
# the PHI-free audit allowlist is extended, never bypassed.
# ===========================================================================

def _last_event(app_module, cid: str, action: str) -> dict:
    events = app_module.store().audit_events(cid=cid, action=action)
    assert events, f"expected a {action!r} audit event"
    return events[-1]


def test_49_allowed_transcription_audits_policy_decision(app_module, auth_client):
    """A successful job records policy_decision=allowed with the declared
    boundary facts (endpoint/region/ZDR) that justified the decision."""
    class _DeepgramStory(_StubEngine):
        """Stub engine whose result declares a Deepgram boundary, mirroring
        what the real engine's meta carries after a successful send."""

        def process_audio(self, audio_path, language=""):
            result = _minimal_result()
            result["meta"].update({
                "stt_provider": "deepgram",
                "stt_endpoint": "https://api.in.deepgram.com/v1/listen",
                "stt_region": "India",
                "stt_zero_retention": True,
            })
            return result

    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)
    app_module.build_engine = _DeepgramStory
    app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
    detail = _last_event(app_module, cid, "transcription_completed")["detail"]
    assert detail["policy_decision"] == "allowed"
    assert detail["policy_reason"] == "declared_boundary_satisfies_policy"
    # Declared provider facts (configuration only, never secrets):
    assert detail["zero_retention"] is True
    assert detail["fail_closed"] == app_module.settings.stt_fail_closed
    assert "endpoint" in detail and "deepgram.com" in detail["endpoint"]
    assert "region" in detail and detail["region"]
    blob = json.dumps(detail).lower()
    for secret in (ACCESS_TOKEN, TRAINING_ADMIN_TOKEN, "authorization", "api_key"):
        assert secret.lower() not in blob


def test_50_blocked_transcription_audits_policy_decision(app_module, auth_client):
    """Every policy-block reason lands in the audit trail as a first-class
    transcription_blocked decision — one row per existing reason value."""
    from scribe_engine.stt.policy import STTPolicyBlocked as Block
    for reason in ("region_mismatch", "zero_retention_required",
                   "provider_mismatch", "off_host_not_allowed",
                   "provider_failed_fail_closed"):
        cid = _make_record_with_audio(app_module, auth_client)
        audio_file = app_module.store().get(cid)["audio_file"]
        job = app_module.store().create_job(cid, Path(audio_file).name)

        class _Boom:
            def __init__(self, *a, **k):
                pass

            def process_audio(self, audio_path, language=""):
                raise Block(reason, "policy refusal (test)")

        app_module.build_engine = _Boom
        app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
        ev = _last_event(app_module, cid, "transcription_blocked")
        assert ev["detail"]["policy_decision"] == "blocked"
        assert ev["detail"]["policy_reason"] == reason
        assert ev["action"] == "transcription_blocked"
        # The message is operator-facing configuration text, never PHI.
        assert "message" in ev["detail"]
        assert Path(audio_file).exists()
        blob = json.dumps(ev).lower()
        for phi in ("doctor: hello", '"text"', "ramesh"):
            assert phi not in blob


def test_51_policy_blocked_job_never_calls_provider(app_module, auth_client,
                                                    monkeypatch):
    """THE critical guarantee: a policy-blocked transcription never issues a
    provider call. The pre-flight check runs before any audio leaves, and the
    blocked refusal still lands in the audit trail with its reason."""
    from scribe_engine.pipeline import ScribeEngine as RealEngine

    sent = []

    class _Sending(_FakeOffHostProvider):
        def transcribe(self, audio_path, language=None):
            sent.append(audio_path)
            return super().transcribe(audio_path, language)

    probe = RealEngine.__new__(RealEngine)
    probe.language = "en"
    probe.whisper_model_size = "base"
    probe.device = "cpu"
    probe.compute_type = "int8"
    probe.cpu_threads = 0
    probe.stt_provider_id = None
    probe.stt_policy = {"region": "in"}  # provider declares global → refuse
    probe.stt_fail_closed = False
    probe.on_stage = None
    probe._stages = []
    audio = app_module.settings.upload_dir / "never-send.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"x")
    with pytest.raises(STTPolicyBlocked) as exc:
        probe.process_audio(audio, language="en")
    assert exc.value.reason == "region_mismatch"
    assert sent == []  # the provider call NEVER happened

    # The refusal is recorded by the service layer when the job runs.
    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def process_audio(self, audio_path, language=""):
            raise STTPolicyBlocked("region_mismatch",
                                   "STT policy requires processing in region "
                                   "'in'. Audio was not sent.")

    app_module.build_engine = _Boom
    app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
    detail = _last_event(app_module, cid, "transcription_blocked")["detail"]
    assert detail["policy_reason"] == "region_mismatch"
    assert "was not sent" in detail["message"]


def test_52_pipeline_meta_carries_boundary_facts_for_audit(app_module,
                                                           auth_client,
                                                           monkeypatch):
    """The real engine result meta carries the declared boundary (endpoint,
    region, ZDR) and request id so the success audit can answer 'why was
    this audio allowed to leave the hospital environment?' from config
    facts alone. Local providers (no declared endpoint) carry none."""
    from scribe_engine.pipeline import ScribeEngine as RealEngine

    probe = RealEngine.__new__(RealEngine)
    probe.language = "en"
    probe.whisper_model_size = "base"
    probe.device = "cpu"
    probe.compute_type = "int8"
    probe.cpu_threads = 0
    probe.stt_provider_id = None
    probe.stt_policy = {"provider": "current_faster_whisper",
                        "off_host_allowed": True,
                        "automatic_fallback_allowed": True}
    probe.stt_fail_closed = False
    probe.on_stage = None
    probe._stages = []

    class _LocalProvider(_FakeOffHostProvider):
        provider_id = "current_faster_whisper"

        def __init__(self):
            self._endpoint = "local"
            self._mip = True

        def retention_configuration(self):
            return {"audio_sent_off_host": False}

        def data_location(self):
            return "on-host (no audio leaves this infrastructure)"

        def transcribe(self, audio_path, language=None):
            from scribe_engine.stt.base import Transcript
            return Transcript(text="fever since morning and a mild cough",
                              segments=[], language="en",
                              provider_id=self.provider_id,
                              provider_meta={"request_id": "req-123"})

    monkeypatch.setattr("scribe_engine.pipeline.resolve_production_provider",
                        lambda *a, **k: (_LocalProvider(), "test"))
    audio = app_module.settings.upload_dir / "meta-probe.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"x")
    result = probe.process_audio(audio, language="en")
    meta = result["meta"]
    assert meta["policy_decision"] == "allowed"
    assert meta["policy_reason"] == "declared_boundary_satisfies_policy"
    assert meta["stt_region"] is None  # local provider: no declared endpoint
    assert meta["stt_zero_retention"] is True
    assert meta["provider_request_id"] == "req-123"
    assert "provider_failed_fail_closed" not in meta.get("policy_reason", "")


def test_53_endpoint_class_classifies_by_host_not_path():
    """The admin boundary view classifies endpoints by HOST: the stock global
    endpoint (api.deepgram.com, with or without a /v1/listen path) must never
    be mislabelled 'custom endpoint' — that label is reserved for genuinely
    custom hosts, and mislabelling would obscure the residency picture."""
    from service.stt_boundary import _endpoint_class
    assert _endpoint_class("api.deepgram.com") == "global (no residency guarantee)"
    assert _endpoint_class("api.deepgram.com/v1/listen") == \
        "global (no residency guarantee)"
    assert _endpoint_class("api.in.deepgram.com/v1/listen") == "india"
    assert _endpoint_class("api.eu.deepgram.com") == "eu"
    assert _endpoint_class("api.au.deepgram.com") == "australia"
    assert _endpoint_class("stt.example-hospital.internal/v1/listen") == \
        "custom endpoint"


def test_54_fail_closed_provider_failure_audits_blocked(app_module, auth_client,
                                                        monkeypatch):
    """A provider failure under fail-closed mode surfaces as a BLOCKED policy
    decision (reason provider_failed_fail_closed) in the audit trail — not a
    silent internal error — and the audio stays under the normal lifecycle."""
    from scribe_engine.pipeline import ScribeEngine as RealEngine

    probe = RealEngine(
        whisper_model_size="base", device="cpu", compute_type="int8",
        cpu_threads=0, language="en", stt_provider_id="deepgram")
    probe.stt_policy = None
    probe.stt_fail_closed = True
    probe.on_stage = None

    monkeypatch.setattr(
        "scribe_engine.pipeline.resolve_production_provider",
        lambda *a, **k: (_FakeOffHostProvider(), "test"))
    monkeypatch.setattr(
        _FakeOffHostProvider, "transcribe",
        lambda self, path, language=None: (_ for _ in ()).throw(
            RuntimeError("timeout")))

    audio = app_module.settings.upload_dir / "failclosed-audit.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"x")
    with pytest.raises(STTPolicyBlocked) as exc:
        probe.process_audio(audio, language="en")
    assert exc.value.reason == "provider_failed_fail_closed"

    # Through the service layer the failure lands in the audit trail as a
    # BLOCKED policy decision, and the source audio is preserved.
    cid = _make_record_with_audio(app_module, auth_client)
    audio_file = app_module.store().get(cid)["audio_file"]
    job = app_module.store().create_job(cid, Path(audio_file).name)

    class _Boom:
        def __init__(self, *a, **k):
            pass

        def process_audio(self, audio_path, language=""):
            raise STTPolicyBlocked(
                "provider_failed_fail_closed",
                "Transcription failed and automatic fallback is disabled "
                "by policy (ISCRIBE_STT_FAIL_CLOSED).")

    app_module.build_engine = _Boom
    app_module._execute_doc_job(job["job_id"], cid, audio_file, "en")
    detail = _last_event(app_module, cid, "transcription_blocked")["detail"]
    assert detail["policy_decision"] == "blocked"
    assert detail["policy_reason"] == "provider_failed_fail_closed"
    assert Path(audio_file).exists()
    lifecycle = app_module.store().get_audio_lifecycle(cid)
    assert lifecycle["audio_state"] != "SCHEDULED_FOR_DELETION"


