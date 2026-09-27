"""Access control and response hardening.

The trial is gated by a single shared access token. That is a deliberate
trade-off: it is not per-clinician identity (so it cannot support real audit
attribution), but it needs no user database, no password reset flow and no
identity provider, which keeps the first deployment small enough to stand up
safely. The token must be rotated at the end of the trial.

The browser exchanges the token once at /login for an HMAC-signed, HttpOnly,
SameSite=Strict session cookie, so the token itself is never kept in
localStorage and never appears in a URL.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse

SESSION_COOKIE = "iscribe_session"
TOKEN_HEADER = "X-Access-Token"

# Reachable without a session: liveness/readiness for the supervisor and the
# tunnel, the login page and its endpoint, the one-time admin bootstrap
# (gated by its own server-side code), and the stylesheets. No clinical data
# is served from any of these.
PUBLIC_PATHS = frozenset({
    "/api/health",
    "/api/ready",
    "/login",
    "/api/login",
    "/api/auth/bootstrap",
    "/styles.css",
    "/app.js",
    "/favicon.ico",
})


def _signing_key(access_token: str) -> bytes:
    """Derive the cookie-signing key from the access token.

    Separate from the token itself so a leaked cookie cannot be turned back
    into the shared token.
    """
    return hashlib.sha256(b"iscribe.session.v1:" + access_token.encode()).digest()


def issue_session(access_token: str, ttl_seconds: int) -> str:
    """Anonymous (shared-key) session: expiry payload only.

    Kept for the break-glass/bootstrap login; carries no identity.
    """
    expiry = int(time.time()) + ttl_seconds
    payload = str(expiry).encode()
    mac = hmac.new(_signing_key(access_token), payload, hashlib.sha256).digest()
    return f"{base64.urlsafe_b64encode(payload).decode()}.{base64.urlsafe_b64encode(mac).decode()}"


def issue_user_session(access_token: str, email: str, role: str,
                       hospital_id: str, ttl_seconds: int) -> str:
    """Named session: '<expiry>|<email>|<role>|<hospital_id>', HMAC-signed.

    The hospital id inside the session is the tenancy boundary: it is read
    back server-side on every request and NEVER accepted from client data.
    The signing key is server config, so a browser cannot mint or alter a
    session (role escalation would need the server's signing key).
    """
    expiry = int(time.time()) + ttl_seconds
    payload = f"{expiry}|{email}|{role}|{hospital_id}".encode()
    mac = hmac.new(_signing_key(access_token), payload, hashlib.sha256).digest()
    return f"{base64.urlsafe_b64encode(payload).decode()}.{base64.urlsafe_b64encode(mac).decode()}"


def _verify(cookie_value: str, access_token: str) -> bytes | None:
    """Return the raw payload when the signature and expiry are valid."""
    try:
        raw_payload, raw_mac = cookie_value.split(".", 1)
        payload = base64.urlsafe_b64decode(raw_payload)
        mac = base64.urlsafe_b64decode(raw_mac)
    except Exception:
        return None
    expected = hmac.new(_signing_key(access_token), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(mac, expected):
        return None
    try:
        expiry = int(payload.split(b"|", 1)[0])
    except ValueError:
        expiry = int(payload)
    return payload if expiry > int(time.time()) else None


def verify_session(cookie_value: str, access_token: str) -> bool:
    return _verify(cookie_value, access_token) is not None


def session_identity(cookie_value: str, access_token: str,
                     default_hospital_id: str = "default") -> dict:
    """Return {email, role, hospital_id, named} for a valid session cookie.

    Legacy/shared-key sessions (expiry-only payload) are anonymous: they are
    bound to this server's hospital and carry no identity for audit, and
    clinical sign-off is refused for them (see CLINICAL_SIGNOFF_ROLES in
    app.py). Returns {} for an invalid/expired cookie.
    """
    payload = _verify(cookie_value or "", access_token)
    if payload is None:
        return {}
    parts = payload.decode(errors="replace").split("|")
    if len(parts) != 4:
        return {"email": None, "role": "SHARED", "display_name": None,
                "hospital_id": default_hospital_id, "named": False}
    _expiry, email, role, hospital_id = parts
    return {"email": email or None, "role": role or "SHARED",
            "display_name": None, "hospital_id": hospital_id or default_hospital_id,
            "named": bool(email)}


def token_matches(candidate: str, access_token: str) -> bool:
    if not candidate or not access_token:
        return False
    return secrets.compare_digest(candidate, access_token)


# ---------------------------------------------------------------------------
# Expiring, consultation-bound download tokens.
#
# Raw audio is served only through /api/consultations/{cid}/audio with BOTH a
# valid session AND a short-lived token bound to that exact consultation.
# A leaked or guessed URL for one recording must never unlock another, and
# must stop working quickly. Tokens carry no content, only ids and expiry.
# ---------------------------------------------------------------------------
DOWNLOAD_TOKEN_TTL_SECONDS = 900  # 15 minutes

def _download_key(access_token: str) -> bytes:
    return hashlib.sha256(b"iscribe.audio-link.v1:" + access_token.encode()).digest()


def issue_download_token(access_token: str, cid: str,
                         ttl_seconds: int = DOWNLOAD_TOKEN_TTL_SECONDS) -> str:
    """Return '<expiry>-<mac>' bound to one consultation id.

    A non-positive ttl mints an already-expired token (useful for tests and
    immediate revocation); the endpoint itself always uses the default.
    """
    expiry = str(int(time.time()) + ttl_seconds)
    mac = hmac.new(_download_key(access_token),
                   f"{cid}.{expiry}".encode(), hashlib.sha256).hexdigest()
    return f"{expiry}-{mac}"


def download_token_valid(token_value: str, access_token: str, cid: str) -> bool:
    """True only for an unexpired token issued for THIS cid."""
    if not token_value:
        return False
    try:
        expiry_str, mac = token_value.split("-", 1)
        expiry = int(expiry_str)
    except (ValueError, AttributeError):
        return False
    expected = hmac.new(_download_key(access_token),
                        f"{cid}.{expiry}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(mac, expected):
        return False
    return expiry > int(time.time())


class AccessControlMiddleware(BaseHTTPMiddleware):
    """Rejects any request without a valid session cookie or access token."""

    def __init__(self, app, settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request, call_next):
        settings = self.settings
        if not settings.auth_required:
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        cookie = request.cookies.get(SESSION_COOKIE, "")
        if cookie and verify_session(cookie, settings.access_token):
            return await call_next(request)

        header = request.headers.get(TOKEN_HEADER, "")
        if token_matches(header, settings.access_token):
            return await call_next(request)

        if path.startswith("/api/"):
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
        return RedirectResponse("/login", status_code=303)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline browser hardening; also stops clinical responses being cached."""

    def __init__(self, app, settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        # Clinical content must not sit in a shared/browser cache.
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store, private")
        if self.settings.is_production:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response


LOGIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>iScribe — Sign in</title>
<link rel="stylesheet" href="/styles.css">
</head>
<body class="login-body">
  <form class="login-card" method="POST" action="/api/login">
    <div class="login-brand"><span class="brand-mark">&#10010;</span> iScribe</div>
    <p class="hint">Sign in with your clinic account.</p>
    <input type="email" name="email" placeholder="Email"
           autocomplete="username" autofocus required>
    <input type="password" name="password" placeholder="Password"
           autocomplete="current-password" required>
    <button class="btn btn-primary" type="submit">Sign in</button>
    __ERROR__
    <details class="login-alt"><summary>Sign in with an access key</summary>
      <input type="password" name="access_token" placeholder="Access key"
             autocomplete="off">
      <p class="hint">Break-glass access: sessions issued this way cannot
      approve or finalize clinical notes.</p>
    </details>
    <p class="login-foot">AI medical scribe — clinical trial deployment.
    Every note requires clinician review.</p>
  </form>
</body>
</html>
"""


def login_page(error: str = "") -> str:
    block = f'<p class="login-error">{error}</p>' if error else ""
    return LOGIN_PAGE.replace("__ERROR__", block)
