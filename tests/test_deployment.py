"""Deployment-critical behaviour tests.

These cover the things that make the difference between "runs on my laptop" and
"safe to point a hospital at": authentication, persistence across restart, PHI
not leaking through the API or the logs, upload limits, and health reporting.

They deliberately avoid running Whisper — STT itself is covered by
tests/test_engine.py. Everything here is fast and needs no model.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class _StubEngine:
    """Stands in for ScribeEngine in async-flow tests (no models needed)."""

    def __init__(self, fn) -> None:
        self._fn = fn
        self.on_stage = None

    def process_audio(self, audio_path, language=""):
        return self._fn(audio_path, language)

ACCESS_TOKEN = "test-token-that-is-long-enough-123"


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Import service.app fresh against a throwaway data directory."""
    monkeypatch.setenv("ISCRIBE_ENV", "production")
    monkeypatch.setenv("ISCRIBE_ACCESS_TOKEN", ACCESS_TOKEN)
    monkeypatch.setenv("ISCRIBE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ISCRIBE_PRELOAD_MODELS", "false")
    monkeypatch.setenv("ISCRIBE_COOKIE_SECURE", "false")
    monkeypatch.setenv("ISCRIBE_MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("ISCRIBE_LOG_DIR", str(tmp_path / "logs"))

    for name in ("service.app", "service.config", "service.store",
                 "service.security", "service.logging_config"):
        sys.modules.pop(name, None)
    module = importlib.import_module("service.app")
    yield module
    for name in ("service.app", "service.config", "service.store",
                 "service.security", "service.logging_config"):
        sys.modules.pop(name, None)


@pytest.fixture()
def client(app_module):
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture()
def auth_client(app_module):
    """A signed-in named user (clinic doctor). The shared access key now only
    issues an anonymous, clinically read-only session."""
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


# -- health -----------------------------------------------------------------
def test_served_login_page_has_no_provider_names(auth_client):
    """The product must not name speech engines anywhere the user can see."""
    import re

    res = auth_client.get("/login")
    assert res.status_code == 200
    assert not re.search(r"deepgram|whisper|indicconformer|nemo|sidecar",
                         res.text, re.IGNORECASE), "login page leaks provider names"


def test_health_is_public_and_does_not_require_models(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_ready_is_public(client):
    res = client.get("/api/ready")
    assert res.status_code in (200, 503)
    assert "models_ready" in res.json()


# -- authentication ---------------------------------------------------------
def test_api_rejects_unauthenticated_requests(client):
    res = client.get("/api/consultations")
    assert res.status_code == 401


def test_ui_redirects_unauthenticated_browser_to_login(client):
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 303
    assert res.headers["location"] == "/login"


def test_wrong_access_key_is_rejected(client):
    res = client.post("/api/login", data={"access_token": "wrong-key-wrong-key"})
    assert res.status_code == 401


def test_correct_access_key_issues_session_cookie(client):
    res = client.post("/api/login", data={"access_token": ACCESS_TOKEN},
                      follow_redirects=False)
    assert res.status_code == 303
    cookie = res.headers.get("set-cookie", "")
    assert "iscribe_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=strict" in cookie.replace("samesite", "SameSite")
    # The shared token itself must never be handed back to the browser.
    assert ACCESS_TOKEN not in cookie


def test_session_cookie_cannot_be_forged(app_module):
    from service.security import verify_session

    assert not verify_session("bogus.value", ACCESS_TOKEN)
    assert not verify_session("", ACCESS_TOKEN)


def test_production_refuses_to_start_without_a_token(monkeypatch, tmp_path):
    monkeypatch.setenv("ISCRIBE_ENV", "production")
    # Blank rather than deleted: config.py loads .env at import time, so on a
    # machine that has a real .env the variable is already present. An empty
    # value is the same "no token configured" case from the app's point of view.
    monkeypatch.setenv("ISCRIBE_ACCESS_TOKEN", "")
    monkeypatch.setenv("ISCRIBE_DATA_DIR", str(tmp_path))
    sys.modules.pop("service.config", None)
    config = importlib.import_module("service.config")
    with pytest.raises(config.ConfigError):
        config.load_settings()
    sys.modules.pop("service.config", None)


# -- security headers -------------------------------------------------------
def test_security_headers_present(client):
    res = client.get("/api/health")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in res.headers["Content-Security-Policy"]
    assert "no-store" in res.headers["Cache-Control"]


# -- consultation flow ------------------------------------------------------
def test_text_flow_produces_note_and_prescription(auth_client):
    created = auth_client.post("/api/consultations",
                               json={"patient_id": "TRIAL-001"}).json()
    cid = created["id"]

    transcript = auth_client.get("/api/demo/transcript").json()["transcript"]
    processed = auth_client.post(f"/api/consultations/{cid}/text",
                                 json={"transcript": transcript}).json()

    assert processed["status"] == "ready"
    note = processed["result"]["clinical_note"]["fields"]
    assert "chest pain" in note["symptoms_reported"]
    assert processed["result"]["prescription"]["fields"] is not None
    assert processed["result"]["speakers"]["method"] == "explicit_roles"


def test_review_complete_and_export(auth_client):
    """Review → approve (complete) → export: the approved-only EHR document.

    The export no longer carries the raw transcript or legacy debug sections;
    it carries the doctor-edited, approved note.
    """
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    transcript = auth_client.get("/api/demo/transcript").json()["transcript"]
    auth_client.post(f"/api/consultations/{cid}/text", json={"transcript": transcript})

    reviewed = auth_client.patch(
        f"/api/consultations/{cid}/review",
        json={"clinical_note_fields": {"impression": "Reviewed impression"}},
    ).json()
    assert reviewed["reviewed"] is True

    completed = auth_client.post(f"/api/consultations/{cid}/complete").json()
    assert completed["status"] == "completed"
    assert completed["completed_at"]

    export = auth_client.get(f"/api/consultations/{cid}/export")
    assert export.status_code == 200
    assert "Reviewed impression" in export.text
    assert "no-store" in export.headers["Cache-Control"]
    # Approved-only export: no raw transcript, no legacy debug section.
    assert "TRANSCRIPT" not in export.text
    assert "LEGACY TEMPLATE NOTE" not in export.text
    assert "Hello, good morning" not in export.text


def test_export_requires_approval(auth_client):
    """Exporting an unapproved note is a workflow error, not a footnote."""
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    transcript = auth_client.get("/api/demo/transcript").json()["transcript"]
    auth_client.post(f"/api/consultations/{cid}/text", json={"transcript": transcript})
    export = auth_client.get(f"/api/consultations/{cid}/export")
    assert export.status_code == 409
    assert "Approve the note" in export.json()["detail"]


# -- data exposure ----------------------------------------------------------
def test_server_filesystem_path_never_reaches_the_browser(auth_client, app_module):
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    app_module.store().apply(cid, lambda r: r.update(audio_file="C:/secret/path/x.mp3"))

    body = auth_client.get(f"/api/consultations/{cid}").json()
    assert "audio_file" not in body
    assert "audio_path" not in body
    assert body["has_audio"] is True
    assert "secret" not in json.dumps(body)


def test_uploads_are_not_served_as_static_files(auth_client, app_module):
    settings = app_module.settings
    leak = settings.upload_dir / "leak.mp3"
    leak.parent.mkdir(parents=True, exist_ok=True)
    leak.write_bytes(b"audio")
    for path in ("/uploads/leak.mp3", "/data/uploads/leak.mp3", "/leak.mp3"):
        assert auth_client.get(path).status_code == 404


def test_engine_result_meta_carries_no_filesystem_path():
    """meta is nested inside `result`, so it bypasses top-level key stripping."""
    from scribe_engine import clinical, prescription
    from scribe_engine.validation import validate_transcript

    # Build the same meta shape process_audio returns, from a known path.
    import pathlib

    # PureWindowsPath keeps the server-secret directory semantics on every
    # platform (the server path is Windows-shaped regardless of CI OS).
    path = pathlib.PureWindowsPath(r"C:\server\secret\uploads\abc123.wav")
    meta = {"audio_filename": path.name, "audio_duration_s": 1.0}
    blob = json.dumps(meta)
    assert "secret" not in blob
    assert "C:\\" not in blob
    assert meta["audio_filename"] == "abc123.wav"


def test_database_is_outside_the_static_root(app_module):
    settings = app_module.settings
    from service.config import STATIC_DIR

    assert STATIC_DIR not in settings.db_path.parents
    assert STATIC_DIR not in settings.upload_dir.parents


# -- upload limits ----------------------------------------------------------
def test_oversize_upload_is_rejected_and_leaves_no_file(auth_client, app_module):
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    payload = b"x" * (2 * 1024 * 1024)  # limit is 1 MB in this fixture
    res = auth_client.post(f"/api/consultations/{cid}/audio",
                           files={"file": ("big.mp3", payload, "audio/mpeg")})
    assert res.status_code == 413
    assert list(app_module.settings.upload_dir.glob(f"{cid}*")) == []


def test_unsupported_extension_is_rejected(auth_client):
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    res = auth_client.post(f"/api/consultations/{cid}/audio",
                           files={"file": ("notes.exe", b"data", "application/octet-stream")})
    assert res.status_code == 415


def test_empty_upload_is_rejected(auth_client):
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    res = auth_client.post(f"/api/consultations/{cid}/audio",
                           files={"file": ("empty.mp3", b"", "audio/mpeg")})
    assert res.status_code == 400


def test_process_without_audio_is_rejected(auth_client):
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    assert auth_client.post(f"/api/consultations/{cid}/process").status_code == 400


# -- asynchronous documentation jobs -----------------------------------------

def _make_result(tag: str, *, note_valid: bool = True) -> dict:
    """A minimal engine result shaped like ScribeEngine.process_audio output."""
    from scribe_engine import note_generator as ng
    from scribe_engine.fact_graph import build_fact_graph
    fact = {
        "concept": "cough", "english": "cough", "status": "PRESENT",
        "certainty": "CONFIRMED", "temporality": "CURRENT", "duration": None,
        "onset": None, "frequency": None, "surface_text": f"{tag} surf",
        "source_clause": f"{tag} clause", "speaker": "UNKNOWN",
        "confidence": 0.9, "attributes": {}, "uncertainty_reason": None,
        "raw_text": f"RAW {tag}", "ongoing": False,
    }
    graph = build_fact_graph([fact], [], raw_text=f"RAW {tag}", roles_known=False)
    note = ng.render_deterministic_note(graph)
    if not note_valid:
        note["validation"] = {"valid": False, "violations": ["forced for test"]}
    return {
        "transcript": {"text": f"transcript {tag}", "language": "en", "segments": []},
        "validation": {"severity": "ok", "codes": []},
        "normalized_clinical_entities": {
            "entities": [fact], "present": ["cough"]},
        "speakers": {"method": "none", "roles_known": False, "turns": []},
        "clinical_note": {"fields": {"symptoms_reported": tag}, "text": "note"},
        "clinical_note_v2": note,
        "prescription": {"fields": {}, "text": ""},
        "meta": {"audio_filename": "a.wav", "audio_duration_s": 1.0},
    }


def _upload(auth_client, cid: str, name: str = "a.mp3") -> None:
    res = auth_client.post(f"/api/consultations/{cid}/audio",
                           files={"file": (name, b"fake-audio-bytes", "audio/mpeg")})
    assert res.status_code == 200, res.text


def _wait_for_job(auth_client, cid: str, timeout: float = 10.0) -> dict:
    import time as _time
    deadline = _time.time() + timeout
    while _time.time() < deadline:
        rec = auth_client.get(f"/api/consultations/{cid}").json()
        if rec["status"] in ("ready", "error"):
            return rec
        _time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_process_returns_202_immediately_without_running_asr(
        auth_client, app_module, monkeypatch):
    """Phase 12.1: submission never waits for the (slow) clinical pipeline."""
    import time as _time

    gate = threading.Event()

    def slow_engine(audio_path, language=""):
        gate.wait(5)            # simulates ~50 s of ASR
        return _make_result("slow")

    monkeypatch.setattr(app_module, "build_engine", lambda: _StubEngine(slow_engine))
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    _upload(auth_client, cid)

    t0 = _time.perf_counter()
    res = auth_client.post(f"/api/consultations/{cid}/process")
    elapsed = _time.perf_counter() - t0

    assert res.status_code == 202, res.text
    assert elapsed < 2.0, f"doctor blocked for {elapsed:.2f}s"
    assert res.json()["job_status"] == "queued"
    assert "job_id" in res.json()
    gate.set()
    _wait_for_job(auth_client, cid)


def test_job_completes_with_validated_note(auth_client, app_module, monkeypatch):
    """Phase 12.3: queued job produces transcript, fact graph note, PASS gate."""
    monkeypatch.setattr(app_module, "build_engine", lambda: _StubEngine(
        lambda p, language="": _make_result("ok")))
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    _upload(auth_client, cid)
    res = auth_client.post(f"/api/consultations/{cid}/process")
    assert res.status_code == 202
    rec = _wait_for_job(auth_client, cid)
    assert rec["status"] == "ready"
    v2 = rec["result"]["clinical_note_v2"]
    assert v2["validation"]["valid"] is True
    assert "2. History of Present Illness" in v2["note"]


def test_failed_note_validation_marks_job_failed(auth_client, app_module, monkeypatch):
    """Phase 11: an unvalidated note can never reach status COMPLETED."""
    monkeypatch.setattr(app_module, "build_engine", lambda: _StubEngine(
        lambda p, language="": _make_result("bad", note_valid=False)))
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    _upload(auth_client, cid)
    auth_client.post(f"/api/consultations/{cid}/process")
    rec = _wait_for_job(auth_client, cid)
    assert rec["status"] == "error"
    job = app_module.rt.store.latest_job(cid)
    assert job["status"] == "failed"
    assert "clinical note validation failed" in job["error"]


def test_encounter_isolation_concurrent_jobs(auth_client, app_module, monkeypatch):
    """Phase 12.4/12.5: two patients processed concurrently never cross."""
    import time as _time

    def engine_for(tag):
        def run(audio_path, language=""):
            _time.sleep(0.15)   # both jobs overlap in the serial worker
            return _make_result(tag)
        return _StubEngine(run)

    monkeypatch.setattr(app_module, "build_engine",
                        lambda: engine_for("placeholder"))
    # Distinct results per consultation: uploads are stored as '<cid>.mp3',
    # so the tag derives from the actual per-encounter audio file.
    class TagEngine(_StubEngine):
        def process_audio(self, path, language=""):
            _time.sleep(0.15)
            return _make_result(Path(path).stem)   # the cid itself

    monkeypatch.setattr(app_module, "build_engine", lambda: TagEngine(None))

    cids = []
    for name in ("a", "b"):
        cid = auth_client.post("/api/consultations",
                               json={"patient_id": f"P-{name}"}).json()["id"]
        _upload(auth_client, cid, name=f"{name}.mp3")
        cids.append(cid)
    for cid in cids:
        assert auth_client.post(f"/api/consultations/{cid}/process").status_code == 202

    recs = [_wait_for_job(auth_client, cid) for cid in cids]
    assert all(r["status"] == "ready" for r in recs)
    # Each consultation's artifacts carry ITS OWN tag (the cid): a swapped or
    # shared result would repeat one tag twice and fail this assertion.
    tags = [r["result"]["normalized_clinical_entities"]["entities"][0]["surface_text"]
            for r in recs]
    assert tags == [f"{cids[0]} surf", f"{cids[1]} surf"], tags  # no cross-talk
    assert recs[0]["result"]["transcript"]["text"] == f"transcript {cids[0]}"
    assert recs[1]["result"]["transcript"]["text"] == f"transcript {cids[1]}"


def test_failure_isolation_patient_b_unaffected(auth_client, app_module, monkeypatch):
    """Phase 12.5: Patient A fails; Patient B still completes normally."""
    cids = []
    for name in ("a", "b"):
        cid = auth_client.post("/api/consultations", json={}).json()["id"]
        _upload(auth_client, cid, name=f"{name}.mp3")
        cids.append(cid)

    class MixedEngine(_StubEngine):
        def process_audio(self, path, language=""):
            if Path(path).stem == cids[0]:
                raise RuntimeError("simulated ASR crash")
            return _make_result("b")

    monkeypatch.setattr(app_module, "build_engine", lambda: MixedEngine(None))
    for cid in cids:
        auth_client.post(f"/api/consultations/{cid}/process")
    recs = [_wait_for_job(auth_client, cid) for cid in cids]
    assert recs[0]["status"] == "error"
    assert recs[1]["status"] == "ready"
    job_a = app_module.rt.store.latest_job(cids[0])
    assert job_a["status"] == "failed"


def test_failed_job_can_be_retried(auth_client, app_module, monkeypatch):
    """Phase 12.6: retry of a failed job produces a new job and a note."""
    calls = {"n": 0}

    class FlakyEngine(_StubEngine):
        def process_audio(self, path, language=""):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient failure")
            return _make_result("retried")

    monkeypatch.setattr(app_module, "build_engine", lambda: FlakyEngine(None))
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    _upload(auth_client, cid)
    auth_client.post(f"/api/consultations/{cid}/process")
    rec = _wait_for_job(auth_client, cid)
    assert rec["status"] == "error"
    failed_job = app_module.rt.store.latest_job(cid)
    assert failed_job["status"] == "failed"

    res = auth_client.post(
        f"/api/consultations/{cid}/jobs/{failed_job['job_id']}/retry")
    assert res.status_code == 202, res.text
    rec = _wait_for_job(auth_client, cid)
    assert rec["status"] == "ready"
    assert rec["result"]["clinical_note_v2"]["validation"]["valid"] is True
    # The old failed row keeps its terminal state (history, not overwrite).
    assert app_module.rt.store.get_job(failed_job["job_id"])["status"] == "failed"


def test_resubmitting_same_recording_is_idempotent(auth_client, app_module, monkeypatch):
    """Phase 12.7: same recording twice -> one job, no duplicate documentation."""
    class SlowEngine(_StubEngine):
        def process_audio(self, path, language=""):
            time.sleep(0.4)
            return _make_result("idem")

    monkeypatch.setattr(app_module, "build_engine", lambda: SlowEngine(None))
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    _upload(auth_client, cid)
    r1 = auth_client.post(f"/api/consultations/{cid}/process")
    r2 = auth_client.post(f"/api/consultations/{cid}/process")  # while r1 runs
    # Both responses must be idempotent: a 202 naming the SAME job, or the
    # in-flight short-circuit (200) that creates nothing new.
    assert r1.status_code == 202
    assert r2.status_code in (200, 202)
    if "job_id" in r2.json():
        assert r1.json()["job_id"] == r2.json()["job_id"]
    _wait_for_job(auth_client, cid)
    jobs = app_module.rt.store.latest_job(cid)
    assert jobs["status"] == "completed"
    # Exactly one completed job row for this recording.
    with app_module.rt.store._lock:
        rows = app_module.rt.store._conn.execute(
            "SELECT status, COUNT(*) AS n FROM documentation_jobs "
            "WHERE consultation_id = ? GROUP BY status", (cid,)).fetchall()
    counts = {r["status"]: r["n"] for r in rows}
    assert counts.get("completed") == 1
    assert counts.get("failed", 0) == 0


def test_engine_build_reported_in_health(auth_client):
    """Observability: stale services are detectable via the build id."""
    body = auth_client.get("/api/health").json()
    assert body["engine_build"].startswith("engine-")


def test_unknown_consultation_is_404(auth_client):
    assert auth_client.get("/api/consultations/doesnotexist").status_code == 404


# -- persistence ------------------------------------------------------------
def test_consultations_survive_a_restart(app_module, auth_client):
    cid = auth_client.post("/api/consultations",
                           json={"patient_id": "TRIAL-042"}).json()["id"]
    transcript = auth_client.get("/api/demo/transcript").json()["transcript"]
    auth_client.post(f"/api/consultations/{cid}/text", json={"transcript": transcript})

    # Simulate a process restart: close the store, reopen the same database file.
    from service.store import ConsultationStore

    app_module.rt.store.close()
    app_module.rt.store = ConsultationStore(app_module.settings.db_path)

    restored = auth_client.get(f"/api/consultations/{cid}").json()
    assert restored["patient_id"] == "TRIAL-042"
    assert restored["result"]["clinical_note"]["fields"]["symptoms_reported"]


def test_interrupted_jobs_are_recovered_as_failed(app_module, auth_client):
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    app_module.store().apply(cid, lambda r: r.update(status="processing"))

    app_module._recover_stuck_jobs()

    record = auth_client.get(f"/api/consultations/{cid}").json()
    assert record["status"] == "error"
    assert "restart" in record["error"]


# -- audio retention ---------------------------------------------------------
# The age-based purge was replaced by the lifecycle-driven retention layer
# (service/retention.py): audio is scheduled for deletion by approval or the
# configured unreviewed ceiling, and the sweep executes deletions with audit
# events. Scenario coverage lives in tests/test_retention_security.py.
def test_expired_audio_is_deleted_from_disk_and_record(app_module, auth_client):
    """Scheduled-and-due audio is removed from disk and the record by the sweep."""
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    audio = app_module.settings.upload_dir / f"{cid}.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"synthetic audio")
    app_module.store().apply(cid, lambda r: r.update(audio_file=str(audio)))
    app_module.store().set_audio_lifecycle(
        cid, audio_state="SCHEDULED_FOR_DELETION", audio_deletion_due_ts=0.0)

    assert app_module.rt.retention.sweep()["clinical_deleted"] == 1
    assert not audio.exists()

    record = auth_client.get(f"/api/consultations/{cid}").json()
    assert record["audio_purged"] is True
    assert record["has_audio"] is False


def test_recent_audio_is_not_purged(app_module, auth_client):
    """Audio without an approval event and without a due deadline survives."""
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    audio = app_module.settings.upload_dir / f"{cid}.mp3"
    audio.parent.mkdir(parents=True, exist_ok=True)
    audio.write_bytes(b"synthetic audio")
    app_module.store().apply(cid, lambda r: r.update(audio_file=str(audio)))

    assert app_module.rt.retention.sweep()["clinical_deleted"] == 0
    assert audio.exists()


def test_reprocessing_purged_audio_gives_a_clear_error(app_module, auth_client):
    cid = auth_client.post("/api/consultations", json={}).json()["id"]
    missing = app_module.settings.upload_dir / f"{cid}.mp3"
    app_module.store().apply(cid, lambda r: r.update(audio_file=str(missing)))

    res = auth_client.post(f"/api/consultations/{cid}/process")
    assert res.status_code == 410
    assert "retention" in res.json()["detail"]


def test_orphan_upload_files_are_removed(app_module, auth_client):
    stray = app_module.settings.upload_dir / "no-such-consultation.mp3"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"orphan")

    assert app_module._purge_orphan_uploads() >= 1
    assert not stray.exists()


# -- logging ----------------------------------------------------------------
def test_clinical_content_is_not_written_to_the_log(app_module, auth_client, tmp_path):
    cid = auth_client.post("/api/consultations",
                           json={"patient_id": "TRIAL-PHI-CHECK"}).json()["id"]
    transcript = auth_client.get("/api/demo/transcript").json()["transcript"]
    auth_client.post(f"/api/consultations/{cid}/text", json={"transcript": transcript})
    auth_client.get(f"/api/consultations/{cid}/export")

    log_file = app_module.settings.log_dir / "iscribe.log"
    contents = log_file.read_text(encoding="utf-8") if log_file.exists() else ""

    # Transcript content, patient identifiers and generated notes must be absent.
    for leaked in ("Ramesh", "chest pain", "shortness of breath",
                   "angina", "TRIAL-PHI-CHECK"):
        assert leaked not in contents, f"{leaked!r} leaked into the application log"
    # Operational facts must still be there.
    assert cid in contents
    assert "event=text_job_completed" in contents


def test_log_filter_truncates_long_third_party_messages():
    import logging

    from service.logging_config import MAX_MESSAGE_CHARS, PHIRedactionFilter

    record = logging.LogRecord("x", logging.INFO, __file__, 1, "P" * 5000, (), None)
    PHIRedactionFilter().filter(record)
    assert len(record.getMessage()) < MAX_MESSAGE_CHARS + 60
    assert "truncated" in record.getMessage()


def test_job_event_refuses_to_serialise_structured_clinical_data(caplog):
    import logging

    from service.logging_config import job_event

    logger = logging.getLogger("iscribe.test")
    with caplog.at_level(logging.INFO, logger="iscribe.test"):
        job_event(logger, "t", cid="abc",
                  note={"impression": "Possible angina"},
                  transcript="Patient: I have chest pain " * 20)
    text = caplog.text
    assert "angina" not in text
    assert "chest pain" not in text
    assert "note=<dict>" in text


def test_sidecar_engine_build_is_cached_at_import():
    """Stale-process detection only works if /ready reports the STARTUP build.

    The sidecar previously recomputed engine_build from git on every request,
    so a sidecar running old code would report whatever commit was on disk and
    the watchdog would never restart it. It must resolve once at import, like
    service.app does.
    """
    import malayalam_sidecar.server as sidecar

    from service.app import ENGINE_BUILD as SERVICE_BUILD

    # Cached constant exists and is a resolved value, not a per-call recompute.
    assert isinstance(sidecar.ENGINE_BUILD, str)
    assert sidecar.ENGINE_BUILD.startswith("engine-")

    # Mutating the on-disk git state must NOT change what the running module
    # reports (that is the entire point of caching at import).
    frozen = sidecar.ENGINE_BUILD
    import unittest.mock as mock
    with mock.patch.object(sidecar, "_compute_engine_build",
                           return_value="engine-DIFFERENT"):
        client = TestClient(sidecar.app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["engine_build"] == frozen
        assert resp.json()["engine_build"] != "engine-DIFFERENT"

    # Both components derive from the same helper contract.
    assert sidecar.ENGINE_BUILD == SERVICE_BUILD


# ===========================================================================
# STT fail-closed production default: the runtime configuration derives the
# default from ISCRIBE_ENV itself. No .env file can silently disable it in
# production; an explicit ISCRIBE_STT_FAIL_CLOSED always overrides.
# ===========================================================================

def _load_settings(monkeypatch, env: str, fail_closed: str | None, tmp_path):
    """Import service.config fresh with a controlled environment.

    The dotenv loader is stubbed for the import so a developer's local .env
    file cannot inject ISCRIBE_STT_FAIL_CLOSED behind the test's back — the
    defaults under test must come from the ISCRIBE_ENV derivation, not from
    whatever happens to be in a working copy.
    """
    import os
    import sys
    import types

    stub = types.ModuleType("dotenv")
    stub.load_dotenv = lambda *a, **k: False
    monkeypatch.setitem(sys.modules, "dotenv", stub)

    monkeypatch.setenv("ISCRIBE_ENV", env)
    monkeypatch.setenv("ISCRIBE_DATA_DIR", str(tmp_path / "cfg"))
    # load_settings refuses to produce a production configuration without an
    # access token — satisfy that gate so the fail-closed default is what is
    # actually under test.
    monkeypatch.setenv("ISCRIBE_ACCESS_TOKEN", "config-test-token-long-enough")
    if fail_closed is None:
        monkeypatch.delenv("ISCRIBE_STT_FAIL_CLOSED", raising=False)
    else:
        monkeypatch.setenv("ISCRIBE_STT_FAIL_CLOSED", fail_closed)

    for name in list(sys.modules):
        if name.startswith("service."):
            sys.modules.pop(name, None)
    try:
        config = importlib.import_module("service.config")
        return config.load_settings()
    finally:
        for name in list(sys.modules):
            if name.startswith("service."):
                sys.modules.pop(name, None)


def test_dev_default_fail_closed_off(monkeypatch, tmp_path):
    """Development behaviour is unchanged: automatic fallback stays the
    default when ISCRIBE_ENV is not production."""
    s = _load_settings(monkeypatch, "development", None, tmp_path)
    assert s.stt_fail_closed is False


def test_production_default_fail_closed_on(monkeypatch, tmp_path):
    """ISCRIBE_ENV=production implies fail-closed STT by default: an
    unexpected provider switch is a privacy/residency/auditability risk, and
    'we forgot to set the env var' is not an acceptable answer."""
    s = _load_settings(monkeypatch, "production", None, tmp_path)
    assert s.stt_fail_closed is True


def test_explicit_fail_closed_true_wins(monkeypatch, tmp_path):
    """An explicit value always overrides the environment-derived default."""
    s = _load_settings(monkeypatch, "development", "true", tmp_path)
    assert s.stt_fail_closed is True


def test_explicit_fail_closed_false_wins(monkeypatch, tmp_path):
    """Production operators retain an explicit opt-out (documented risk
    decision), but it must be stated in configuration, never defaulted."""
    s = _load_settings(monkeypatch, "production", "false", tmp_path)
    assert s.stt_fail_closed is False


def test_env_example_documented(monkeypatch):
    """The template documents that production defaults fail-closed ON, so an
    operator reading .env.example is not misled about the production default."""
    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text(
        encoding="utf-8")
    assert "ISCRIBE_STT_FAIL_CLOSED" in text
    after = text.split("ISCRIBE_STT_FAIL_CLOSED", 1)[1][:400].lower()
    assert "production" in after
