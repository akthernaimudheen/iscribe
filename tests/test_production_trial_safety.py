"""Production trial safety regression tests.

Covers the guarantees added for the controlled real-world consultation trial:

* online database backup (verified copy, rolling prune, visible failures)
* restore round-trip through a real SQLite file
* access-log scrubbing so short-lived audio tokens never hit disk
* security boundaries on the public health endpoints and static file serving
* deployment safety artifacts exist (runbook, gitignore for backups)

These tests never touch the production database: backup/drill functions take
explicit paths, so every test operates on throwaway SQLite files in tmp_path.
"""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


backup_db = _load_script("backup_db")
restore_db = _load_script("restore_db")


# Reuse the deployment suite's app fixtures (fresh throwaway app per test).
from tests.test_deployment import app_module, client, auth_client  # noqa: E402,F401


def _make_db(path: Path, consultations: int = 2) -> None:
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE consultations (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, created_ts REAL NOT NULL,
            status TEXT NOT NULL, audio_path TEXT,
            audio_purged INTEGER NOT NULL DEFAULT 0,
            document TEXT NOT NULL);
        CREATE TABLE users (
            email TEXT PRIMARY KEY, password_hash TEXT NOT NULL,
            display_name TEXT, role TEXT NOT NULL, hospital_id TEXT,
            disabled INTEGER NOT NULL DEFAULT 0, created_at TEXT);
        CREATE TABLE audit_events (
            event_id TEXT PRIMARY KEY, ts TEXT, ts_epoch REAL, action TEXT,
            cid TEXT, hospital_id TEXT, actor TEXT, detail TEXT);
    """)
    for i in range(consultations):
        con.execute("INSERT INTO consultations VALUES (?,?,?,?,?,?,?,?)",
                    (f"c{i}", "t", "t", 1.0 * i, "completed", None, 0,
                     '{"reviewed": true, "result": {"clinical_note": '
                     '{"fields": {"impression": "x"}}}}'))
    con.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                ("d@x.test", "hash", "Dr", "DOCTOR", "default", 0, "t"))
    con.execute("INSERT INTO audit_events VALUES (?,?,?,?,?,?,?,?)",
                ("e1", "t", 1.0, "note_exported", "c0", "default", "d@x.test", "{}"))
    con.commit()
    con.close()


# -- backup mechanism --------------------------------------------------------

def test_backup_creates_verified_copy_with_manifest(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    db = live / "iscribe.db"
    _make_db(db)
    backup_db.load_settings = lambda: type(
        "S", (), {"db_path": db})()
    dest = backup_db.make_backup(tmp_path / "backups", keep=5)
    assert dest.exists()
    manifest = (tmp_path / "backups" / f"{dest.name}.manifest.txt").read_text()
    assert "integrity_check=ok" in manifest
    assert "consultations=2" in manifest
    assert "users=1" in manifest


def test_backup_prunes_to_rolling_window(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    db = live / "iscribe.db"
    _make_db(db)
    backup_db.load_settings = lambda: type("S", (), {"db_path": db})()
    bdir = tmp_path / "backups"
    for _ in range(6):
        import time
        time.sleep(1.05)  # timestamped filenames need distinct seconds
        backup_db.make_backup(bdir, keep=3)
    remaining = list(bdir.glob("iscribe-*.db"))
    assert len(remaining) == 3


def test_backup_failure_leaves_visible_marker(tmp_path, monkeypatch):
    live = tmp_path / "live"
    live.mkdir()
    db = live / "iscribe.db"
    db.write_text("this is not a database")
    backup_db.load_settings = lambda: type("S", (), {"db_path": db})()
    marker = tmp_path / "m.txt"
    monkeypatch.setattr(backup_db, "FAIL_MARKER", "m.txt")
    monkeypatch.setattr(backup_db, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["backup_db.py"])
    rc = backup_db.main()
    assert rc == 1
    assert marker.exists()
    assert "BACKUP FAILED" not in marker.read_text()  # marker holds detail, not the banner
    assert marker.read_text().strip() != ""


# -- restore round-trip -------------------------------------------------------

def test_restore_round_trip_preserves_content(tmp_path):
    live = tmp_path / "live"
    live.mkdir()
    db = live / "iscribe.db"
    _make_db(db)
    backup_db.load_settings = lambda: type("S", (), {"db_path": db})()
    dest = backup_db.make_backup(tmp_path / "backups", keep=5)

    # Simulate drift on the "live" copy, then restore the backup over a
    # throwaway copy (the documented procedure, minus production paths).
    con = sqlite3.connect(db)
    con.execute("UPDATE users SET disabled=1")
    con.commit()
    con.close()

    restored = tmp_path / "restored.db"
    import shutil
    shutil.copy2(dest, restored)
    con = sqlite3.connect(restored)
    (integrity,) = con.execute("PRAGMA integrity_check").fetchone()
    con.close()
    assert integrity == "ok"
    con = sqlite3.connect(restored)
    enabled = con.execute(
        "SELECT COUNT(*) FROM users WHERE disabled=0").fetchone()[0]
    consultations = con.execute(
        "SELECT COUNT(*) FROM consultations").fetchone()[0]
    notes = con.execute(
        "SELECT COUNT(*) FROM consultations WHERE "
        "json_extract(document, '$.result.clinical_note.fields') IS NOT NULL"
    ).fetchone()[0]
    reviewed = con.execute(
        "SELECT COUNT(*) FROM consultations WHERE "
        "json_extract(document, '$.reviewed') = 1").fetchone()[0]
    con.close()
    assert enabled == 1
    assert consultations == 2
    assert notes == 2
    assert reviewed == 2


def test_restore_db_script_verifies_readonly(tmp_path, capsys):
    db = tmp_path / "b.db"
    _make_db(db)
    assert restore_db.verify(db) == 0
    out = capsys.readouterr().out
    assert "RESULT           : PASS" in out
    assert "consultations    : 2" in out


def test_restore_db_script_fails_on_garbage(tmp_path, capsys):
    db = tmp_path / "bad.db"
    db.write_text("garbage")
    assert restore_db.verify(db) == 1
    assert "FAIL" in capsys.readouterr().out


# -- access-log query scrubbing ----------------------------------------------

def test_access_log_scrubber_strips_audio_tokens():
    from service.logging_config import AccessLogQueryScrubFilter

    f = AccessLogQueryScrubFilter()
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        '%s - "%s %s HTTP/%s" %d', None, None)
    record.args = ("1.2.3.4:0", "GET",
                   "/api/consultations/abc/audio?expires=ME-secret123", "1.1", 200)
    assert f.filter(record)
    rendered = record.getMessage()
    assert "ME-secret123" not in rendered
    assert "/api/consultations/abc/audio" in rendered


def test_access_log_scrubber_handles_plain_strings():
    from service.logging_config import AccessLogQueryScrubFilter

    f = AccessLogQueryScrubFilter()
    record = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1,
        'GET /x?expires=ME-tok HTTP/1.1 200', (), None)
    assert f.filter(record)
    assert "ME-tok" not in record.getMessage()


def test_access_log_scrubber_is_idempotent():
    from service.logging_config import attach_access_log_scrubber

    logger = logging.getLogger("uvicorn.access")
    before = len(logger.filters)
    attach_access_log_scrubber()
    attach_access_log_scrubber()
    assert len(logger.filters) == before + 1  # exactly one added across both calls


# -- security boundaries on public endpoints ----------------------------------

def test_health_endpoint_reveals_no_secrets(client):
    body = client.get("/api/health").json().items()
    text = str(dict(body)).lower()
    for banned in ("token", "key", "password", "secret", "bearer"):
        assert banned not in text


def test_ready_endpoint_reveals_no_secrets(client):
    body = str(client.get("/api/ready").json()).lower()
    for banned in ("token", "api_key=", "password", "secret"):
        assert banned not in body
    # Presence booleans only.
    assert "deepgram_configured" in body


def test_static_serving_root_stays_inside_static_dir(client, app_module):
    # Unauthenticated static requests are redirected to the login page (303);
    # path traversal must never become a file disclosure (404/redirect, never 200).
    res = client.get("/", follow_redirects=False)
    assert res.status_code == 303
    traversal = client.get("/..%2f..%2f.env", follow_redirects=False)
    assert traversal.status_code in (301, 303, 404)


def test_env_and_db_are_not_web_accessible(client):
    # Unauthenticated: the access middleware redirects to /login (303) or
    # rejects (401/403). The file content must never be served (200).
    for forbidden in ("/.env", "/data/iscribe.db", "/iscribe-backups/",
                      "/.git/config"):
        res = client.get(forbidden, follow_redirects=False)
        assert res.status_code in (301, 303, 401, 403, 404), \
            f"{forbidden} reachable!"


def test_env_and_db_are_not_web_accessible_even_when_signed_in(auth_client):
    # The stronger case: a fully signed-in doctor must get 404 for these
    # paths too - a session cookie must not turn the server into a file
    # server for anything outside the static directory.
    for forbidden in ("/.env", "/data/iscribe.db", "/iscribe-backups/",
                      "/.git/config"):
        res = auth_client.get(forbidden, follow_redirects=False)
        assert res.status_code == 404, f"{forbidden} served with a session!"


# -- deployment safety artifacts ----------------------------------------------

def test_runbook_exists_and_documents_core_procedures():
    runbook = REPO_ROOT / "docs" / "PRODUCTION_TRIAL_RUNBOOK.md"
    assert runbook.exists()
    text = runbook.read_text(encoding="utf-8").lower()
    for topic in ("backup", "restore", "rollback", "health", "tunnel",
                  "windows reboot", "crash", "security incident"):
        assert topic in text, f"runbook missing section: {topic}"


def test_gitignore_covers_backups_and_runtime_data():
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/iscribe-backups/" in gitignore
    assert "*.db" in gitignore
    assert ".env" in gitignore
