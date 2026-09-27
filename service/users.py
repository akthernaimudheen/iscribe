"""Per-user accounts: email/password authentication, roles, clinic tenancy.

The shared access key remains as a break-glass bootstrap mechanism (it issues
an anonymous session), but real audit attribution and clinic isolation need
per-user identity. Passwords are hashed with PBKDF2-HMAC-SHA256 (stdlib only —
no new runtime dependency for a clinical deployment), each user in exactly one
clinic (hospital_id), roles: ADMIN | DOCTOR | STAFF.

A user can only ever see consultations of their own clinic: the clinic id is
server-assigned from the session identity on every read/write, never from the
client. See service/app.py::_get for the enforcement point.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

ROLES = ("ADMIN", "DOCTOR", "STAFF")
# Approval/finalize/export are clinical sign-off: doctors (and admins) only.
CLINICAL_SIGNOFF_ROLES = ("ADMIN", "DOCTOR")
# PBKDF2 iteration count. 600k follows the OWASP PBKDF2-HMAC-SHA256
# guidance; verification cost stays ~0.3s, acceptable for clinic-scale login
# rates (and the login endpoint is rate-limited on top).
_PBKDF2_ITERATIONS = 600_000
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    email         TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    display_name  TEXT NOT NULL,
    role          TEXT NOT NULL CHECK (role IN ('ADMIN','DOCTOR','STAFF')),
    hospital_id   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    created_by    TEXT,
    disabled      INTEGER NOT NULL DEFAULT 0,
    disabled_by   TEXT,
    disabled_at   TEXT
);
"""


def hash_password(password: str) -> str:
    """Return 'pbkdf2_sha256$<iterations>$<salt>$<hash>' for a password."""
    if len(password or "") < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters")
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Constant-time password verification against a stored hash."""
    try:
        scheme, iterations, salt, expected = (stored or "").split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", (password or "").encode(), salt.encode(), int(iterations))
        return hmac.compare_digest(digest.hex(), expected)
    except (ValueError, TypeError):
        return False


def valid_email(email: str) -> bool:
    return bool(_EMAIL_RE.match((email or "").strip().lower()))


class UserStore:
    """Thread-safe SQLite user store (separate connection, same data dir).

    WAL mode allows the user store and consultation store to sit on the same
    database file without blocking each other.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            self.db_path, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes -------------------------------------------------------------
    def create(self, email: str, password: str, display_name: str,
               role: str, hospital_id: str, created_by: str | None = None) -> dict:
        email = (email or "").strip().lower()
        if not valid_email(email):
            raise ValueError("A valid email address is required")
        if role not in ROLES:
            raise ValueError(f"Role must be one of: {', '.join(ROLES)}")
        display_name = (display_name or "").strip()
        if not display_name:
            raise ValueError("Display name is required")
        record = {
            "email": email,
            "password_hash": hash_password(password),
            "display_name": display_name,
            "role": role,
            "hospital_id": hospital_id,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": created_by,
            "disabled": 0,
        }
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO users (email, password_hash, display_name, role, "
                    "hospital_id, created_at, created_by, disabled) "
                    "VALUES (?,?,?,?,?,?,?,0)",
                    (email, record["password_hash"], display_name, role,
                     hospital_id, record["created_at"], created_by),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                raise ValueError("An account with this email already exists") from exc
        return self.get(email)

    def set_disabled(self, email: str, disabled: bool,
                     by: str | None = None) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE users SET disabled = ?, disabled_by = ?, disabled_at = ? "
                "WHERE email = ?",
                (1 if disabled else 0,
                 by if disabled else None,
                 time.strftime("%Y-%m-%d %H:%M:%S") if disabled else None,
                 (email or "").strip().lower()))
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get(email)

    def set_password(self, email: str, password: str) -> dict | None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE users SET password_hash = ? WHERE email = ?",
                (hash_password(password), (email or "").strip().lower()))
            self._conn.commit()
            if cur.rowcount == 0:
                return None
        return self.get(email)

    # -- reads ---------------------------------------------------------------
    def get(self, email: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM users WHERE email = ?",
                ((email or "").strip().lower(),)).fetchone()
        return dict(row) if row else None

    def list(self, include_disabled: bool = True) -> list[dict]:
        q = ("SELECT email, display_name, role, hospital_id, created_at, "
             "created_by, disabled, disabled_at FROM users")
        if not include_disabled:
            q += " WHERE disabled = 0"
        q += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(q).fetchall()
        return [dict(r) for r in rows]

    def count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"]

    # -- authentication --------------------------------------------------------
    def authenticate(self, email: str, password: str) -> dict | None:
        """Return the user record only for valid, enabled credentials.

        A disabled account is indistinguishable from a wrong password at this
        boundary (both -> None); the caller may re-check disabled separately
        to give a precise message AFTER rate limiting.
        """
        user = self.get(email)
        if not user or user.get("disabled"):
            # Burn comparable time so disabled accounts are not timing-orchestra.
            verify_password(password or "", hash_password(
                "timing-equalizer-not-a-real-password"))
            return None
        if not verify_password(password or "", user["password_hash"]):
            return None
        return user
