"""Accounts, roles and clinic-isolation tests.

Covers the production requirements that the shared-key trial deployment did
not have: per-user authentication, ADMIN/DOCTOR/STAFF roles, clinic tenancy
enforced server-side on every clinical read/write, rate limiting on auth and
processing, and audit events carrying a real actor identity.

These tests run against a fresh app module with two clinics' worth of data to
prove the isolation boundary, not just the happy path.
"""

from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BOOTSTRAP_CODE = "bootstrap-code-long-enough-for-trial"
ALPHA_KEY = "clinic-alpha-access-key-123"
BETA_KEY = "clinic-beta-access-key-456"


@pytest.fixture()
def env_two_clinics(tmp_path, monkeypatch):
    """Two .env-shaped clinic configs sharing one throwaway home."""
    def make(env_name: str, key: str, clinic: str):
        monkeypatch.setenv("ISCRIBE_ENV", "production")
        monkeypatch.setenv("ISCRIBE_ACCESS_TOKEN", key)
        monkeypatch.setenv("ISCRIBE_BOOTSTRAP_ADMIN_CODE", BOOTSTRAP_CODE)
        monkeypatch.setenv("ISCRIBE_DATA_DIR", str(tmp_path / env_name / "data"))
        monkeypatch.setenv("ISCRIBE_HOSPITAL_ID", clinic)
        monkeypatch.setenv("ISCRIBE_PRELOAD_MODELS", "false")
        monkeypatch.setenv("ISCRIBE_COOKIE_SECURE", "false")
        monkeypatch.setenv("ISCRIBE_LOG_DIR", str(tmp_path / env_name / "logs"))

    # Clinic A lives in `app_module`-style fresh imports below; clinic B is
    # simulated by re-importing with different env (see cross_clinic fixture).
    yield make


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Fresh service.app as a production clinic (clinic-alpha)."""
    monkeypatch.setenv("ISCRIBE_ENV", "production")
    monkeypatch.setenv("ISCRIBE_ACCESS_TOKEN", ALPHA_KEY)
    monkeypatch.setenv("ISCRIBE_BOOTSTRAP_ADMIN_CODE", BOOTSTRAP_CODE)
    monkeypatch.setenv("ISCRIBE_DATA_DIR", str(tmp_path / "alpha" / "data"))
    monkeypatch.setenv("ISCRIBE_HOSPITAL_ID", "clinic-alpha")
    monkeypatch.setenv("ISCRIBE_PRELOAD_MODELS", "false")
    monkeypatch.setenv("ISCRIBE_COOKIE_SECURE", "false")
    monkeypatch.setenv("ISCRIBE_LOG_DIR", str(tmp_path / "alpha" / "logs"))
    for name in ("service.app", "service.config", "service.store",
                 "service.security", "service.logging_config",
                 "service.retention", "service.users"):
        sys.modules.pop(name, None)
    module = importlib.import_module("service.app")
    yield module
    for name in ("service.app", "service.config", "service.store",
                 "service.security", "service.logging_config",
                 "service.retention", "service.users"):
        sys.modules.pop(name, None)


@pytest.fixture()
def client(app_module):
    with TestClient(app_module.app) as c:
        yield c


def _bootstrap_admin(client) -> dict:
    res = client.post("/api/auth/bootstrap", json={
        "email": "admin@alpha.test", "password": "AdminPass-Long-1",
        "display_name": "Alpha Admin", "bootstrap_code": BOOTSTRAP_CODE})
    assert res.status_code == 200, res.text
    return res.json()


def _login(client, email: str, password: str):
    res = client.post("/api/login", data={"email": email, "password": password},
                      follow_redirects=False)
    assert res.status_code == 303, res.text
    return res


def _admin_client(client) -> TestClient:
    """Turn the ONE shared TestClient into a signed-in admin (identity is a
    cookie on the client; no second lifespan is started)."""
    _bootstrap_admin(client)
    _login(client, "admin@alpha.test", "AdminPass-Long-1")
    return client


def _fresh_client(app_module) -> TestClient:
    """A cookie-jar-only client WITHOUT entering the app lifespan again (the
    module-scoped lifespan state in rt is already up from `client`)."""
    return TestClient(app_module.app)


# ---------------------------------------------------------------------------
# Bootstrap + registration
# ---------------------------------------------------------------------------
def test_bootstrap_requires_the_server_code(client):
    res = client.post("/api/auth/bootstrap", json={
        "email": "admin@alpha.test", "password": "AdminPass-Long-1",
        "display_name": "X", "bootstrap_code": "wrong-code"})
    assert res.status_code == 403


def test_bootstrap_creates_exactly_one_admin(client):
    _bootstrap_admin(client)
    # Second bootstrap is refused even with the right code.
    res = client.post("/api/auth/bootstrap", json={
        "email": "second@alpha.test", "password": "AdminPass-Long-2",
        "display_name": "Y", "bootstrap_code": BOOTSTRAP_CODE})
    assert res.status_code == 409


def test_weak_passwords_are_rejected(client):
    res = client.post("/api/auth/bootstrap", json={
        "email": "admin@alpha.test", "password": "short",
        "display_name": "X", "bootstrap_code": BOOTSTRAP_CODE})
    assert res.status_code == 422


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------
def test_login_with_good_credentials_issues_named_session(app_module, client):
    _bootstrap_admin(client)
    res = _login(client, "admin@alpha.test", "AdminPass-Long-1")
    cookie = res.headers.get("set-cookie", "")
    assert "iscribe_session=" in cookie and "HttpOnly" in cookie


def test_login_with_bad_credentials_is_401(client):
    _bootstrap_admin(client)
    res = client.post("/api/login", data={"email": "admin@alpha.test",
                                          "password": "WrongPass-Long-1"},
                      follow_redirects=False)
    assert res.status_code == 401
    # The response must not discriminate between wrong password and unknown
    # user (no account enumeration).
    res2 = client.post("/api/login", data={"email": "nobody@alpha.test",
                                           "password": "WrongPass-Long-1"},
                       follow_redirects=False)
    assert res2.status_code == 401


def test_disabled_account_cannot_log_in(client, app_module):
    _admin_client(client)
    client.post("/api/users", json={
        "email": "doc@alpha.test", "password": "DocPass-Long-1",
        "display_name": "Dr. Who", "role": "DOCTOR"})
    res = client.patch("/api/users/doc@alpha.test", json={"disabled": True})
    assert res.status_code == 200
    fresh = _fresh_client(app_module)
    r = fresh.post("/api/login", data={"email": "doc@alpha.test",
                                       "password": "DocPass-Long-1"},
                   follow_redirects=False)
    assert r.status_code == 401


def test_login_is_rate_limited(client):
    _bootstrap_admin(client)
    for _ in range(5):
        client.post("/api/login", data={"email": "admin@alpha.test",
                                        "password": "Definitely-Wrong-1"},
                    follow_redirects=False)
    res = client.post("/api/login", data={"email": "admin@alpha.test",
                                          "password": "AdminPass-Long-1"},
                      follow_redirects=False)
    assert res.status_code == 429
    assert res.headers.get("retry-after")


def test_shared_key_session_is_readonly_for_clinical_signoff(app_module, client):
    """The break-glass key still opens the door but can never sign off."""
    # Admin sets up a reviewable consultation first.
    admin = _admin_client(client)
    cid = admin.post("/api/consultations", json={}).json()["id"]
    from tests.test_retention_security import _minimal_result  # type: ignore
    app_module.store().apply(cid, lambda r: r.update(
        result=_minimal_result(), status="ready"))
    app_module.store().set_audio_lifecycle(cid, note_status="REVIEWED")
    # A separate browser signs in with the shared key.
    shared = _fresh_client(app_module)
    res = shared.post("/api/login", data={"access_token": ALPHA_KEY},
                      follow_redirects=False)
    assert res.status_code == 303
    # Shared-key session cannot approve...
    res = shared.post(f"/api/consultations/{cid}/approve")
    assert res.status_code == 403
    # ...nor export.
    app_module.store().set_audio_lifecycle(cid, note_status="APPROVED")
    res = shared.get(f"/api/consultations/{cid}/export")
    assert res.status_code == 403


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------
def test_role_gates_on_signoff_and_user_management(app_module, client):
    admin = _admin_client(client)
    admin.post("/api/users", json={
        "email": "staff@alpha.test", "password": "StaffPass-Long-1",
        "display_name": "S", "role": "STAFF"})
    admin.post("/api/users", json={
        "email": "doc@alpha.test", "password": "DocPass-Long-1",
        "display_name": "D", "role": "DOCTOR"})
    staff = _fresh_client(app_module)
    _login(staff, "staff@alpha.test", "StaffPass-Long-1")
    doc = _fresh_client(app_module)
    _login(doc, "doc@alpha.test", "DocPass-Long-1")
    # STAFF cannot create users or list users.
    assert staff.post("/api/users", json={
        "email": "x@alpha.test", "password": "StaffPass-Long-1",
        "display_name": "X", "role": "STAFF"}).status_code == 403
    assert staff.get("/api/users").status_code == 403
    # STAFF cannot even sign off; DOCTOR can.
    cid = admin.post("/api/consultations", json={}).json()["id"]
    from tests.test_retention_security import _minimal_result  # type: ignore
    app_module.store().apply(cid, lambda r: r.update(
        result=_minimal_result(), status="ready"))
    app_module.store().set_audio_lifecycle(cid, note_status="REVIEWED")
    assert staff.post(f"/api/consultations/{cid}/approve").status_code == 403
    assert doc.post(f"/api/consultations/{cid}/approve").status_code == 200


# ---------------------------------------------------------------------------
# Clinic isolation (the mandatory boundary)
# ---------------------------------------------------------------------------
def test_user_cannot_read_another_clinics_consultation(app_module, client):
    """A clinic-beta session cannot even learn that an alpha cid exists."""
    admin = _admin_client(client)
    cid = admin.post("/api/consultations", json={}).json()["id"]
    # Forge is impossible (HMAC), so simulate clinic-beta with a session
    # minted by the same signing key but a different hospital id — a stolen
    # cookie from another deployment must fail, and a forged hospital_id in a
    # request body must be ignored.
    from service.security import issue_user_session
    other = _fresh_client(app_module)
    forged = issue_user_session(app_module.settings.access_token,
                                "doctor@beta.test", "DOCTOR", "clinic-beta",
                                3600)
    other.cookies.set(app_module.SESSION_COOKIE, forged)
    assert other.get(f"/api/consultations/{cid}").status_code == 404
    # And the list shows nothing from the other clinic.
    listed = other.get("/api/consultations").json()
    assert all(r["id"] != cid for r in listed)


def test_created_consultation_is_stamped_with_session_clinic(app_module, client):
    from service.security import issue_user_session
    admin = _admin_client(client)
    forged = issue_user_session(app_module.settings.access_token,
                                "doctor@beta.test", "DOCTOR", "clinic-beta",
                                3600)
    beta = _fresh_client(app_module)
    beta.cookies.set(app_module.SESSION_COOKIE, forged)
    # A beta-session consultation gets clinic-beta tenancy, NOT the body's or
    # the server's default hospital — the session is the tenancy source.
    cid = beta.post("/api/consultations", json={}).json()["id"]
    rec = app_module.store().get(cid)
    assert rec["hospital_id"] == "clinic-beta"
    # The alpha admin cannot see or approve it.
    assert admin.get(f"/api/consultations/{cid}").status_code == 404
    assert admin.post(f"/api/consultations/{cid}/approve").status_code == 404


# ---------------------------------------------------------------------------
# Audit: named actors
# ---------------------------------------------------------------------------
def test_audit_events_carry_named_actor(client):
    admin = _admin_client(client)
    cid = admin.post("/api/consultations", json={}).json()["id"]
    events = [e for e in admin.app_module_store_audit(cid) if True] \
        if hasattr(admin, "app_module_store_audit") else []
    # (direct store read via the module fixture keeps this simple)
    events = admin_store_events(client, cid)
    created = [e for e in events if e["action"] == "consultation_created"]
    assert created and created[-1]["actor"] == "admin@alpha.test"
    blob = str(created[-1]).lower()
    for phi in ("transcript", "chest pain", "ramesh"):
        assert phi not in blob


def admin_store_events(client, cid):
    """Read the audit trail through the app under test's own store."""
    import service.app as app_ref
    return app_ref.store().audit_events(cid=cid)
