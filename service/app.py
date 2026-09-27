"""iScribe service — FastAPI application.

Sits between the browser UI and ScribeEngine. Deployment-relevant behaviour:

* Consultations persist in SQLite (``service.store``), so a restart or crash no
  longer destroys in-flight clinical work.
* Models are warmed at startup in a background thread; ``/api/ready`` reports
  when transcription is actually possible, so the first clinician of the day is
  not the one who pays the model-load cost.
* Processing runs in a bounded worker pool. Whisper is memory-hungry relative to
  the trial host, so concurrency is capped (default 1) and further requests queue
  rather than competing for RAM.
* Uploaded audio is streamed to disk under a size cap and purged on a retention
  timer; it is never served from a static mount.

PHI policy: no patient identifier, transcript, note or prescription is written to
the application log — see ``service.logging_config``.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
    FileResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from scribe_engine import ScribeEngine
from scribe_engine.audio import SUPPORTED_EXTENSIONS, AudioError
from scribe_engine.clinical import render_clinical_note_text
from scribe_engine.prescription import render_prescription_text
from scribe_engine.stt.base import SpeechLanguageUnavailable

from scribe_engine.stt.deepgram_provider import REGIONAL_ENDPOINTS
from scribe_engine.stt.policy import STTPolicyBlocked

from .config import STATIC_DIR, load_settings
from .export_text import _export_text
from .retention import RetentionManager, sha256_of, DELETE_MAX_ATTEMPTS


def _engine_build() -> str:
    """Deployed build identity for stale-service detection (best effort).

    Derived from the scribe_engine package git revision when available.
    Never contains clinical content; safe to expose via /api/health.
    """
    try:
        import subprocess
        pkg_dir = Path(scribe_engine.__file__).parent
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=pkg_dir,
            capture_output=True, text=True, timeout=5,
        )
        if rev.returncode == 0 and rev.stdout.strip():
            return f"engine-{rev.stdout.strip()}"
    except Exception:
        pass
    return "engine-unknown"


import scribe_engine  # noqa: E402  (build id needs the package object)
ENGINE_BUILD = _engine_build()
from .logging_config import Timer, configure_logging, job_event
from .ratelimit import RateLimiter
from .security import (
    SESSION_COOKIE,
    AccessControlMiddleware,
    SecurityHeadersMiddleware,
    issue_session,
    issue_user_session,
    session_identity,
    login_page,
    token_matches,
    issue_download_token,
    download_token_valid,
    DOWNLOAD_TOKEN_TTL_SECONDS,
)
from .stt_boundary import stt_boundary_view, stt_policy as _effective_stt_policy
from .store import ConsultationStore
from .users import CLINICAL_SIGNOFF_ROLES, ROLES, UserStore

settings = load_settings()
settings.ensure_dirs()
logger = configure_logging(settings.log_level, settings.log_dir)

SHUTDOWN_GRACE_SECONDS = 30
QUEUE_TIMEOUT_SECONDS = 1800


# --------------------------------------------------------------------------
# Runtime state
# --------------------------------------------------------------------------
class Runtime:
    """Process-wide mutable state, owned by the lifespan handler."""

    def __init__(self) -> None:
        self.store: ConsultationStore | None = None
        self.retention: RetentionManager | None = None
        self.users: UserStore | None = None
        self.auth_limiter: RateLimiter | None = None
        self.processing_limiter: RateLimiter | None = None
        self.models_ready = False
        self.model_error: str | None = None
        self.model_load_seconds: float | None = None
        self.started_at = time.time()
        self.startup_seconds: float | None = None
        self.job_slots = threading.Semaphore(settings.max_concurrent_jobs)
        self.active_jobs = 0
        self.jobs_lock = threading.Lock()
        self.shutdown = threading.Event()
        self.job_queue: "queue.Queue[tuple[str, str, str]] | None" = None
        self.job_worker: threading.Thread | None = None


rt = Runtime()


def store() -> ConsultationStore:
    if rt.store is None:  # pragma: no cover - only reachable outside lifespan
        raise RuntimeError("Consultation store is not initialised")
    return rt.store


def retention() -> RetentionManager:
    if rt.retention is None:  # pragma: no cover - only reachable outside lifespan
        raise RuntimeError("Retention manager is not initialised")
    return rt.retention


def users() -> UserStore:
    if rt.users is None:  # pragma: no cover - only reachable outside lifespan
        raise RuntimeError("User store is not initialised")
    return rt.users


# --------------------------------------------------------------------------
# Session identity + tenancy/authorization helpers
# --------------------------------------------------------------------------
def _current_identity(request: Request) -> dict:
    """Resolve the session identity for this request.

    The hospital_id ALWAYS comes from the signed session (or, for anonymous
    shared-key sessions, from the server configuration) — never from any
    client-supplied field. This is the clinic-isolation boundary.
    """
    cookie = request.cookies.get(SESSION_COOKIE, "")
    ident = session_identity(cookie, settings.access_token,
                             default_hospital_id=settings.hospital_id)
    if ident:
        return ident
    # Fall back to the X-Access-Token header path (API/scripts), same tenancy
    # rule: anonymous, server-config hospital.
    if token_matches(request.headers.get("X-Access-Token", ""),
                     settings.access_token):
        return {"email": None, "role": "SHARED", "display_name": None,
                "hospital_id": settings.hospital_id, "named": False}
    return {"email": None, "role": "SHARED", "display_name": None,
            "hospital_id": settings.hospital_id, "named": False}


def _require_named_user(request: Request) -> dict:
    """Identity that may act clinically: a signed-in user account."""
    ident = _current_identity(request)
    if not ident.get("named"):
        raise HTTPException(
            403, "Sign in with a user account to perform this action "
                 "(shared-key sessions cannot modify clinical records)")
    return ident


def _require_role(request: Request, *roles: str) -> dict:
    ident = _require_named_user(request)
    if ident["role"] not in roles:
        raise HTTPException(403, "Your role is not permitted to perform this action")
    return ident


def _own_clinic(request: Request, record: dict) -> dict:
    """Tenant-isolation gate: the record must belong to the session's clinic.

    Every clinical-data endpoint resolves its record through this check, so a
    consultation from another clinic is indistinguishable from a missing one.
    """
    ident = _current_identity(request)
    if record.get("hospital_id") and \
            record["hospital_id"] != ident["hospital_id"]:
        raise HTTPException(404, "Consultation not found")
    return ident


def build_engine() -> ScribeEngine:
    """A fresh engine per job.

    The engine carries a mutable ``on_stage`` callback bound to one
    consultation, so sharing a single instance across concurrent jobs would
    cross-report progress. The expensive Whisper model is cached inside
    ``scribe_engine.transcription``, so constructing an engine is cheap.
    """
    # DEEPGRAM_API_KEY is read from the process environment by the provider
    # itself. Mirroring it here (never logging it) keeps the key out of every
    # call signature and out of the consultation record.
    if settings.deepgram_api_key:
        os.environ.setdefault("DEEPGRAM_API_KEY", settings.deepgram_api_key)
    # Regional endpoint + MIP opt-out are passed through the environment so
    # every construction site (engine, readiness probe, fallback) inherits the
    # hospital's residency/retention configuration.
    os.environ.setdefault("ISCRIBE_DEEPGRAM_ENDPOINT",
                          REGIONAL_ENDPOINTS.get(settings.deepgram_region,
                                                 settings.deepgram_region))
    os.environ.setdefault("ISCRIBE_DEEPGRAM_MIP_OPT_OUT",
                          "true" if settings.deepgram_mip_opt_out else "false")
    engine = ScribeEngine(
        whisper_model_size=settings.whisper_model,
        device=settings.whisper_device,
        stt_provider_id=settings.stt_provider or None,
        compute_type=settings.whisper_compute_type,
        cpu_threads=settings.whisper_cpu_threads,
        language=settings.language,
    )
    # Hospital STT policy: pre-flight provider-boundary check + fail-closed
    # fallback mode. The pipeline enforces both at the provider boundary.
    engine.stt_policy = _effective_stt_policy(settings)
    engine.stt_fail_closed = settings.stt_fail_closed
    return engine


# --------------------------------------------------------------------------
# Startup / shutdown
# --------------------------------------------------------------------------
def _warm_models() -> None:
    """Load Whisper + spaCy once, off the request path."""
    timer = Timer()
    timer.__enter__()
    try:
        from scribe_engine.transcription import _get_model

        # Same cache key the job path uses, so the model is loaded exactly once.
        _get_model(
            settings.whisper_model,
            settings.whisper_device,
            settings.whisper_compute_type,
            settings.whisper_cpu_threads,
        )
        job_event(logger, "model_loaded", component="whisper",
                  model=settings.whisper_model, device=settings.whisper_device,
                  compute_type=settings.whisper_compute_type, seconds=timer.seconds)

        from scribe_engine import clinical

        clinical._get_nlp()
        job_event(logger, "model_loaded", component="spacy", seconds=timer.seconds)

        rt.model_load_seconds = timer.seconds
        rt.models_ready = True
        rt.model_error = None
        job_event(logger, "models_ready", seconds=timer.seconds)
    except Exception as exc:
        rt.model_error = f"{type(exc).__name__}: {exc}"
        rt.models_ready = False
        logger.error("event=model_load_failed error_type=%s", type(exc).__name__)
        logger.exception("model preload failed")


def _purge_orphan_uploads() -> int:
    """Remove upload files with no consultation row (e.g. crashed mid-upload)."""
    removed = 0
    now = time.time()
    for path in settings.upload_dir.glob("*"):
        if not path.is_file():
            continue
        try:
            if path.suffix == ".part":
                if path.stat().st_mtime < now - 3600:
                    path.unlink(missing_ok=True)
                    removed += 1
                continue
            record = store().get(path.stem)
            if record is None or record.get("audio_file") != str(path):
                path.unlink(missing_ok=True)
                removed += 1
        except Exception as exc:
            # Name the failure type: a silent warning here would hide a cleanup
            # that has stopped deleting audio, which is a PHI-retention problem.
            logger.warning("event=orphan_purge_failed error_type=%s", type(exc).__name__)
    return removed


def _retention_backfill() -> None:
    """Give every unapproved audio record a deletion deadline.

    Safety net for records whose pipeline died before any policy evaluation:
    the configured unreviewed ceiling applies from now, the record is scheduled,
    and the sweep handles it like any other deadline. Records with a pending
    job are left alone — the doctor may still need them.
    """
    for row in store().audio_without_deadline():
        if store().has_pending_job(row["id"]):
            continue
        retention()._evaluate(row["id"], reason="unreviewed_ceiling_backfill")


def _cleanup_loop() -> None:
    """Lifecycle sweep: evaluate deadlines, execute due deletions (clinical and
    training) with per-record failure accounting, and remove orphan uploads."""
    while not rt.shutdown.is_set():
        try:
            _retention_backfill()
            summary = rt.retention.sweep()
            orphans = _purge_orphan_uploads()
            if any(summary.values()) or orphans:
                job_event(logger, "cleanup_run", orphans_removed=orphans, **summary)
        except Exception as exc:
            logger.warning("event=cleanup_failed error_type=%s", type(exc).__name__)
        rt.shutdown.wait(settings.retention_sweep_seconds)


def _recover_stuck_jobs() -> None:
    """Fail jobs left mid-processing by a previous crash/restart."""
    def mark(record: dict) -> None:
        record["status"] = "error"
        record["active_stage"] = None
        record["error"] = ("Processing was interrupted by a server restart. "
                           "Re-run processing for this consultation.")

    for cid in store().stuck_jobs():
        store().apply(cid, mark)
        job_event(logger, "job_recovered_as_failed", cid=cid)
    for job_id in store().interrupted_jobs():
        store().update_job(
            job_id, status="failed",
            error="Interrupted by a server restart. Retry from the dashboard.")
        job_event(logger, "doc_job_recovered_as_failed", job_id=job_id)


@asynccontextmanager
async def lifespan(app: FastAPI):
    boot = Timer()
    boot.__enter__()

    settings.ensure_dirs()
    settings.training_vault_dir.mkdir(parents=True, exist_ok=True)
    rt.store = ConsultationStore(settings.db_path,
                                 hospital_id=settings.hospital_id)
    rt.users = UserStore(settings.db_path)
    rt.auth_limiter = RateLimiter(settings.auth_rate_limit,
                                  settings.auth_rate_window_seconds)
    rt.processing_limiter = RateLimiter(
        settings.processing_rate_limit,
        settings.processing_rate_window_seconds)
    rt.retention = RetentionManager(settings, rt.store, logger=logger)
    _recover_stuck_jobs()

    job_event(logger, "startup", **{k: v for k, v in settings.redacted().items()
                                    if not isinstance(v, (list, dict))})
    if settings.auth_required and not settings.is_production:
        # Development only: a token nobody can see is useless, and there is no
        # PHI here. Production never reaches this branch — it refuses to start
        # without an explicitly configured token.
        logger.warning("event=dev_auth_token_generated token=%s "
                       "hint=set_ISCRIBE_ACCESS_TOKEN_to_choose_your_own",
                       settings.access_token)

    if settings.preload_models:
        threading.Thread(target=_warm_models, name="model-warmup", daemon=True).start()
    else:
        rt.models_ready = True  # loaded lazily on first use

    threading.Thread(target=_cleanup_loop, name="cleanup", daemon=True).start()

    # Serial documentation worker: the existing bounded-concurrency model
    # (default 1) as a durable queue. The clinical pipeline itself is unchanged.
    rt.job_queue = queue.Queue()
    rt.job_worker = threading.Thread(
        target=_doc_worker_loop, name="doc-worker", daemon=True)
    rt.job_worker.start()

    rt.startup_seconds = boot.seconds
    job_event(logger, "startup_complete", seconds=rt.startup_seconds)

    try:
        yield
    finally:
        job_event(logger, "shutdown_begin", active_jobs=rt.active_jobs)
        rt.shutdown.set()
        deadline = time.time() + SHUTDOWN_GRACE_SECONDS
        while rt.active_jobs > 0 and time.time() < deadline:
            await asyncio.sleep(0.5)
        if rt.active_jobs > 0:
            job_event(logger, "shutdown_forced", abandoned_jobs=rt.active_jobs)
        if rt.store is not None:
            rt.store.close()
        job_event(logger, "shutdown_complete")


app = FastAPI(
    title="iScribe API",
    version="1.0.0",
    lifespan=lifespan,
    # The interactive docs expose the full clinical schema; off in production.
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
    openapi_url=None if settings.is_production else "/openapi.json",
)

# Middleware added last runs first: security headers wrap everything, then
# access control, then CORS.
app.add_middleware(SecurityHeadersMiddleware, settings=settings)
app.add_middleware(AccessControlMiddleware, settings=settings)
if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "X-Access-Token"],
    )


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class ConsultationCreate(BaseModel):
    patient_id: str = Field(default="TRIAL-001", max_length=64)
    doctor: str = Field(default="Dr.", max_length=120)
    department: str = Field(default="General Medicine", max_length=120)
    consultation_type: str = Field(default="General Consultation", max_length=120)
    # Per-consultation language. English is the default and the only
    # supported production path; "ml" is opt-in and experimental, never
    # selected automatically and never inferred from the audio.
    language: str = Field(default="", max_length=8)


class TrainingSelectionRequest(BaseModel):
    # Why this recording was selected for the training dataset. Required
    # whenever the hospital policy demands explicit selection (the default).
    reason: str = Field(default="", max_length=500)


class TrainingAuthorizationRequest(BaseModel):
    authorized: bool
    authorization_version: str = Field(default="", max_length=64)
    # Authorization lifetime in seconds; required so an authorization can
    # never be granted without an expiry (fail closed).
    authorization_expiry_seconds: int = Field(default=0, ge=0)
    policy_reference: str = Field(default="", max_length=200)


class TextProcessRequest(BaseModel):
    transcript: str = Field(max_length=200_000)


class ReviewPatch(BaseModel):
    clinical_note_fields: dict | None = None
    prescription_fields: dict | None = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
# Server-side-only keys: filesystem paths must never reach the browser.
_PRIVATE_KEYS = ("audio_path", "audio_file")


def _public(record: dict) -> dict:
    out = {k: v for k, v in record.items() if k not in _PRIVATE_KEYS}
    out["has_audio"] = bool(record.get("audio_file"))
    return out


def _get(cid: str, request: Request | None = None) -> dict:
    """Fetch a consultation; when a request is supplied, enforce clinic
    isolation (a record from another clinic is a 404, never a leak)."""
    record = store().get(cid)
    if not record:
        raise HTTPException(404, "Consultation not found")
    if request is not None:
        _own_clinic(request, record)
    # Lifecycle/tenancy live in real columns (the retention layer queries
    # them); surface the API-safe ones on the record for the UI.
    lifecycle = store().get_audio_lifecycle(cid) or {}
    record.update(lifecycle)
    return record


def _progress_callback(cid: str):
    """Persist real engine stage events as they happen."""
    def cb(entry: dict) -> None:
        def mutate(record: dict) -> None:
            if entry["status"] == "running":
                record["active_stage"] = entry["stage"]
            elif entry["status"] == "done" and record.get("active_stage") == entry["stage"]:
                record["active_stage"] = None
            record.setdefault("stages", []).append(entry)

        try:
            store().apply(cid, mutate)
        except Exception:
            logger.warning("event=stage_persist_failed cid=%s", cid)
    return cb


    result = record.get("result") or {}
    note = result.get("clinical_note", {})
    rx = result.get("prescription", {})
    v2 = result.get("clinical_note_v2") or {}
    v2_valid = bool(v2.get("validation", {}).get("valid"))
    speakers = result.get("speakers", {})
    meta = result.get("meta", {})
    validation = result.get("validation", {})
    reviewed = "yes" if record.get("reviewed") else "NO - not yet reviewed"
    lines = [
        "iSCRIBE - CONSULTATION RECORD",
        "Trial software. Clinician review and sign-off required before clinical use.",
        "",
    ]
    # A quality warning belongs at the top of the document, not buried.
    if validation and validation.get("severity") != "ok" and not approved_only:
        lines += [
            "*" * 60,
            f"TRANSCRIPT QUALITY: {str(validation.get('severity', '?')).upper()}",
            validation.get("message", ""),
            "*" * 60,
            "",
        ]
    lines += [
        f"Consultation : {record['id']}",
        f"Patient      : {record['patient_id']}",
        f"Doctor       : {record['doctor']}",
        f"Department   : {record['department']}",
        f"Type         : {record['consultation_type']}",
        f"Generated    : {record.get('completed_at') or record.get('created_at')}",
        f"Reviewed     : {reviewed}",
        f"Language     : {meta.get('reported_language') or 'en'} "
        f"(requested: {meta.get('requested_language') or 'en'})",
        f"STT engine   : {meta.get('stt_provider', 'n/a')}",
        f"Speakers     : {speakers.get('method', 'n/a')} "
        f"(confidence: {speakers.get('confidence', 'n/a')}; "
        f"roles {'identified' if speakers.get('roles_known') else 'NOT identified'})",
        "",
        "=" * 60,
        "TRANSCRIPT",
        "=" * 60,
        result.get("transcript", {}).get("text", "(no transcript)"),
        "",
        "=" * 60,
        # ONE authoritative clinical note: the validated fact-graph note when
        # it validated; otherwise the legacy template text. The legacy text is
        # never shown as a competing note — when V2 is authoritative it is
        # appended below, clearly labelled as superseded.
        (v2.get("note") if v2_valid and v2.get("note")
         else note.get("text", "(no clinical note)")),
        "",
        "=" * 60,
        rx.get("text", "(no prescription)"),
        "",
        "-" * 60,
        'Fields reading "Not mentioned" were not stated in this consultation.',
        "Nothing in this note is inferred or filled in from a template.",
        "Verify every field against the transcript before signing.",
    ]
    if v2_valid and note.get("text") and not approved_only:
        lines += [
            "",
            "-" * 60,
            "LEGACY TEMPLATE NOTE — superseded; reconciliation/debug only, "
            "not for clinical use",
            "-" * 60,
            note["text"],
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Auth routes
# --------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
def login_form():
    if not settings.auth_required:
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(login_page())


def _set_session_cookie(response, token_value: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        token_value,
        max_age=settings.session_hours * 3600,
        httponly=True,
        samesite="strict",
        secure=settings.cookie_secure,
        path="/",
    )


@app.post("/api/login")
def login(request: Request, email: str = Form(default=""),
          password: str = Form(default=""),
          access_token: str = Form(default="")):
    """Dual-mode sign-in: a named user account (email/password) or the
    break-glass shared access key (anonymous, clinically read-only).

    Rate-limited per source IP BEFORE any credential work.
    """
    if not settings.auth_required:
        return RedirectResponse("/", status_code=303)
    client = request.client.host if request.client else "unknown"
    limiter = rt.auth_limiter
    if limiter is not None and not limiter.allow(f"login:{client}"):
        retry = int(limiter.retry_after(f"login:{client}")) or 60
        job_event(logger, "login_rate_limited", source_ip=client)
        return HTMLResponse(
            login_page(f"Too many sign-in attempts. Try again in {retry}s."),
            status_code=429, headers={"Retry-After": str(retry)})

    # Path 1: named user account.
    if email.strip():
        user = users().authenticate(email, password)
        if user is None:
            job_event(logger, "login_failed", source_ip=client, method="password")
            return HTMLResponse(login_page("Incorrect email or password."),
                                status_code=401)
        job_event(logger, "login_ok", source_ip=client, method="password",
                  user=user["email"], role=user["role"])
        store().append_audit("user_login", hospital_id=user["hospital_id"],
                             actor=user["email"],
                             detail={"method": "password"})
        response = RedirectResponse("/", status_code=303)
        _set_session_cookie(response, issue_user_session(
            settings.access_token, user["email"], user["role"],
            user["hospital_id"], settings.session_hours * 3600))
        return response

    # Path 2: break-glass shared key → anonymous session (no clinical writes).
    if access_token and token_matches(access_token, settings.access_token):
        job_event(logger, "login_ok", source_ip=client, method="shared_key")
        store().append_audit("user_login", hospital_id=settings.hospital_id,
                             detail={"method": "shared_key"})
        response = RedirectResponse("/", status_code=303)
        _set_session_cookie(response, issue_session(
            settings.access_token, settings.session_hours * 3600))
        return response

    job_event(logger, "login_failed", source_ip=client, method="none")
    return HTMLResponse(login_page("Enter your email and password."),
                        status_code=401)


@app.get("/api/me")
def me(request: Request):
    """The signed-in identity for the UI topbar. No secrets, no tokens."""
    ident = _current_identity(request)
    return {"email": ident.get("email"), "role": ident.get("role"),
            "named": bool(ident.get("named")),
            "demo_mode": settings.demo_mode,
            "real_consultation_trial": settings.real_consultation_trial}


@app.post("/api/logout")
def logout():
    response = JSONResponse({"status": "signed_out"})
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


# --------------------------------------------------------------------------
# Account administration (bootstrap + user management)
# --------------------------------------------------------------------------
class BootstrapAdminRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=256)
    display_name: str = Field(max_length=120)
    bootstrap_code: str = Field(max_length=256)


class UserCreateRequest(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(min_length=1, max_length=256)
    display_name: str = Field(max_length=120)
    role: str


class UserDisabledPatch(BaseModel):
    disabled: bool


def _public_user(u: dict) -> dict:
    """Account fields safe to return; never the password hash."""
    return {k: u.get(k) for k in ("email", "display_name", "role",
                                  "hospital_id", "created_at", "disabled")}


@app.post("/api/auth/bootstrap")
def bootstrap_admin(body: BootstrapAdminRequest, request: Request):
    """Create the FIRST admin account, exactly once.

    Gated by ISCRIBE_BOOTSTRAP_ADMIN_CODE (server-side secret). Refuses when
    any admin already exists — this is a bootstrap, not a backdoor.
    """
    if not settings.auth_required:
        raise HTTPException(403, "Accounts are managed by the deployment")
    if not settings.bootstrap_admin_code:
        raise HTTPException(
            403, "Bootstrap is disabled: set ISCRIBE_BOOTSTRAP_ADMIN_CODE "
                 "on the server to create the first admin")
    if not token_matches(body.bootstrap_code, settings.bootstrap_admin_code):
        job_event(logger, "bootstrap_rejected",
                  source_ip=request.client.host if request.client else "unknown")
        raise HTTPException(403, "Invalid bootstrap code")
    if any(u["role"] == "ADMIN" for u in users().list()):
        raise HTTPException(409, "An administrator account already exists")
    try:
        user = users().create(body.email, body.password, body.display_name,
                              "ADMIN", settings.hospital_id)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    store().append_audit("user_created", hospital_id=settings.hospital_id,
                         actor=user["email"],
                         detail={"role": "ADMIN", "bootstrap": True})
    job_event(logger, "bootstrap_admin_created", user=user["email"])
    return _public_user(user)


@app.get("/api/users")
def list_users(request: Request):
    _require_role(request, "ADMIN")
    return [_public_user(u) for u in users().list()]


@app.post("/api/users")
def create_user(body: UserCreateRequest, request: Request):
    """ADMIN-only account creation, bound to the admin's own clinic."""
    admin = _require_role(request, "ADMIN")
    if body.role not in ROLES:
        raise HTTPException(422, f"Role must be one of: {', '.join(ROLES)}")
    try:
        user = users().create(body.email, body.password, body.display_name,
                              body.role, admin["hospital_id"],
                              created_by=admin["email"])
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    store().append_audit("user_created", hospital_id=admin["hospital_id"],
                         actor=admin["email"],
                         detail={"new_user": user["email"], "role": user["role"]})
    return _public_user(user)


@app.patch("/api/users/{email}")
def set_user_disabled(email: str, body: UserDisabledPatch, request: Request):
    """Enable/disable an account (ADMIN only, never your own account)."""
    admin = _require_role(request, "ADMIN")
    target = (email or "").strip().lower()
    if target == admin["email"]:
        raise HTTPException(409, "You cannot disable your own account")
    user = users().get(target)
    if not user or user["hospital_id"] != admin["hospital_id"]:
        raise HTTPException(404, "User not found")
    updated = users().set_disabled(target, body.disabled, by=admin["email"])
    store().append_audit("user_status_changed", hospital_id=admin["hospital_id"],
                         actor=admin["email"],
                         detail={"target": target, "disabled": body.disabled})
    return _public_user(updated)


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------
@app.get("/api/health")
def health():
    """Liveness: the process is up and serving. Never gated by auth."""
    return {
        "status": "ok",
        "version": app.version,
        "engine_build": ENGINE_BUILD,
        "uptime_s": round(time.time() - rt.started_at, 1),
    }


@app.get("/api/ready")
def ready():
    """Readiness: transcription is actually possible right now."""
    from scribe_engine.stt import resolve_production_provider

    try:
        active_provider, selection_reason = resolve_production_provider(
            settings.stt_provider or None,
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            cpu_threads=settings.whisper_cpu_threads,
        )
        provider_id = active_provider.provider_id
    except Exception:
        provider_id, selection_reason = "unavailable", "provider resolution failed"

    payload = {
        "status": "ready" if rt.models_ready else ("error" if rt.model_error else "loading"),
        "models_ready": rt.models_ready,
        "model_load_s": rt.model_load_seconds,
        "language": settings.language,
        "stt_provider": provider_id,
        "provider_selection": selection_reason,
        # Presence only. The key itself is never returned.
        "deepgram_configured": bool(settings.deepgram_api_key
                                    or os.environ.get("DEEPGRAM_API_KEY")),
        "whisper_model": settings.whisper_model,
        "device": settings.whisper_device,
        "active_jobs": rt.active_jobs,
        "max_concurrent_jobs": settings.max_concurrent_jobs,
    }
    if rt.model_error:
        payload["error"] = rt.model_error
    return JSONResponse(payload, status_code=200 if rt.models_ready else 503)


# --------------------------------------------------------------------------
# Consultations
# --------------------------------------------------------------------------
@app.post("/api/consultations")
def create_consultation(body: ConsultationCreate, request: Request):
    ident = _current_identity(request)
    cid = uuid.uuid4().hex[:12]
    record = {
        "id": cid,
        "patient_id": body.patient_id or "TRIAL-001",
        "doctor": body.doctor or "Dr.",
        "department": body.department or "General Medicine",
        "consultation_type": body.consultation_type or "General Consultation",
        "language": (body.language or settings.language).lower(),
        "status": "created",
        "active_stage": None,
        "stages": [],
        "audio_file": None,
        "audio_purged": False,
        "source": None,
        "result": None,
        "error": None,
        "reviewed": False,
        "completed_at": None,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        # Tenancy is server-assigned from the session identity: a client can
        # never name its hospital (request bodies are never trusted for it).
        "hospital_id": ident["hospital_id"],
        "created_by": ident.get("email"),
    }
    store().create(record)
    store().append_audit("consultation_created", cid=cid,
                         hospital_id=record["hospital_id"],
                         actor=ident.get("email"))
    job_event(logger, "consultation_created", cid=cid)
    return _public(record)


@app.get("/api/consultations")
def list_consultations(request: Request):
    """Only this session's clinic's consultations are visible — the
    clinic-isolation rule applied to the list, not just detail reads."""
    ident = _current_identity(request)
    mine = [r for r in store().list()
            if (r.get("hospital_id") or settings.hospital_id) == ident["hospital_id"]]
    return [_public(r) for r in mine]


@app.get("/api/consultations/{cid}")
def get_consultation(cid: str, request: Request):
    return _public(_get(cid, request))


@app.post("/api/consultations/{cid}/audio")
async def upload_audio(request: Request, cid: str,
                       file: UploadFile = File(...)):
    ident = _current_identity(request)
    _get(cid, request)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        # Browser MediaRecorder produces .webm; anything unrecognised is rejected
        # rather than guessed, so the engine never sees an undecodable file.
        if suffix:
            raise HTTPException(
                415,
                f"Unsupported audio format '{suffix}'. Supported: "
                + ", ".join(sorted(SUPPORTED_EXTENSIONS)),
            )
        suffix = ".webm"

    # Stream to a .part file so an over-size or aborted upload never becomes a
    # usable consultation, and never sits fully in RAM.
    final_path = settings.upload_dir / f"{cid}{suffix}"
    part_path = settings.upload_dir / f"{cid}{suffix}.part"
    written = 0
    try:
        with part_path.open("wb") as fh:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > settings.max_upload_bytes:
                    raise HTTPException(
                        413, f"Audio exceeds the {settings.max_upload_mb} MB upload limit."
                    )
                fh.write(chunk)
    except HTTPException:
        part_path.unlink(missing_ok=True)
        job_event(logger, "upload_rejected", cid=cid, reason="too_large", bytes=written)
        raise
    except Exception:
        part_path.unlink(missing_ok=True)
        logger.exception("upload failed")
        raise HTTPException(500, "Upload failed")

    if written == 0:
        part_path.unlink(missing_ok=True)
        raise HTTPException(400, "Uploaded audio file is empty")

    part_path.replace(final_path)

    def mutate(rec: dict) -> None:
        rec["audio_file"] = str(final_path)
        rec["audio_purged"] = False
        rec["status"] = "audio_ready"
        rec["source"] = "upload"

    store().apply(cid, mutate)
    # Integrity anchor: the digest is stored but the bytes are NOT copied into
    # the audit trail — ids and hashes only.
    store().set_audio_lifecycle(cid, audio_sha256=sha256_of(final_path),
                                audio_state="UPLOADED")
    retention().audit("audio_uploaded", cid, actor=ident.get("email"),
                      bytes=written, ext=suffix, filename=final_path.name)
    job_event(logger, "audio_uploaded", cid=cid, bytes=written, ext=suffix)
    return _public(_get(cid, request))


def _doc_worker_loop() -> None:
    """Serial background worker: runs queued documentation jobs to completion.

    Failure isolation is per job: an exception in one consultation's job never
    stops the worker loop, so Patient B is unaffected by Patient A's failure.
    """
    while not rt.shutdown.is_set():
        try:
            job_id, cid, audio_path, language = rt.job_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            _execute_doc_job(job_id, cid, audio_path, language)
        except Exception:  # never let one job kill the worker
            logger.exception("event=doc_worker_unexpected_error job_id=%s cid=%s", job_id, cid)
            store().update_job(job_id, status="failed", error="Internal worker error")
            store().apply(cid, lambda r: r.update(
                status="error", active_stage=None, error="Processing failed"))
        finally:
            rt.job_queue.task_done()


def _execute_doc_job(job_id: str, cid: str, audio_path: str, language: str) -> None:
    """Run the existing clinical pipeline for one job; gate completion on the
    fact-graph note validator."""
    store().update_job(job_id, status="processing", started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    retention().mark_processing(cid)
    timer = Timer()
    timer.__enter__()
    try:
        engine = build_engine()
        engine.on_stage = _progress_callback(cid)
        result = engine.process_audio(audio_path, language=language)

        # Clinical-safety gate: the async path may only complete when the
        # deterministic note validates against the fact graph (Phase 11).
        v2 = (result.get("clinical_note_v2") or {})
        v2_valid = bool(v2.get("validation", {}).get("valid"))
        if result.get("clinical_note_v2") is not None and not v2_valid:
            violations = v2["validation"].get("violations", [])
            raise RuntimeError(f"clinical note validation failed: {violations}")

        def finish(record: dict) -> None:
            record["result"] = result
            record["status"] = "ready"
            record["active_stage"] = None

        store().apply(cid, finish)
        store().update_job(job_id, status="completed", stage=None,
                           completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        meta = result.get("meta", {})
        retention().mark_note_ready(cid, provider=meta.get("stt_provider"))
        # Provider boundary metadata: what left our infrastructure, to whom,
        # with which request id and timing. No retention guarantee is implied.
        # The policy decision travels alongside the declared facts that were
        # validated before the audio was sent, so the audit trail answers
        # "why was this audio allowed to leave the hospital environment?"
        # Configuration only — no keys, no transcript, no patient data.
        retention().audit(
            "transcription_completed", cid,
            provider=meta.get("stt_provider"),
            model=meta.get("whisper_model"),
            request_id=meta.get("provider_request_id"),
            elapsed_s=meta.get("elapsed_s"),
            off_host=meta.get("stt_provider") == "deepgram",
            policy_decision=meta.get("policy_decision") or "allowed",
            policy_reason=meta.get("policy_reason")
                          or "declared_boundary_satisfies_policy",
            endpoint=meta.get("stt_endpoint"),
            region=meta.get("stt_region"),
            zero_retention=meta.get("stt_zero_retention"),
            fail_closed=settings.stt_fail_closed,
        )
        validation = result.get("validation", {})
        job_event(
            logger, "doc_job_completed", cid=cid, job_id=job_id,
            elapsed_s=timer.seconds,
            audio_duration_s=meta.get("audio_duration_s"),
            segments=len(result.get("transcript", {}).get("segments", [])),
            speaker_method=result.get("speakers", {}).get("method"),
            roles_known=result.get("speakers", {}).get("roles_known"),
            stt_provider=meta.get("stt_provider"),
            provider_selection=meta.get("provider_selection"),
            requested_language=meta.get("requested_language"),
            reported_language=meta.get("reported_language"),
            note_validator="pass" if v2_valid else "skipped",
            validation=validation.get("severity"),
            validation_codes=",".join(validation.get("codes", [])) or None,
        )
    except AudioError as exc:
        store().update_job(job_id, status="failed", error=str(exc),
                           completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        store().apply(cid, lambda r: r.update(
            status="error", active_stage=None, error=str(exc)))
        job_event(logger, "doc_job_failed", cid=cid, job_id=job_id,
                  reason="audio_error", elapsed_s=timer.seconds)
    except Exception as exc:
        # A policy refusal (pre-flight boundary check or fail-closed provider
        # failure) is audited as a first-class BLOCKED decision: the trail
        # answers "why was this transcription blocked?" without log digging.
        # The refusal itself is operator-worded configuration text, never PHI.
        if isinstance(exc, STTPolicyBlocked):
            retention().audit(
                "transcription_blocked", cid,
                policy_decision="blocked",
                policy_reason=exc.reason,
                provider_id=exc.provider_id,
                message=str(exc),
            )

        # Validation-gate failures are our own RuntimeError with a diagnosable,
        # PHI-free message (concept labels / pattern names only) — keep it in
        # the job row so a rejected note can be reviewed. Engine exceptions can
        # embed transcript text, so their message is not stored verbatim;
        # provider/transport errors are the exception (API status only).
        gate_failure = isinstance(exc, RuntimeError) and str(exc).startswith(
            "clinical note validation failed")
        safe = gate_failure or exc.__class__.__name__ in (
            "DeepgramError", "TranscriptionError", "SpeechLanguageUnavailable")
        logger.error("event=doc_job_failed cid=%s job_id=%s error_type=%s elapsed_s=%s detail=%s",
                     cid, job_id, type(exc).__name__, timer.seconds,
                     str(exc)[:200] if safe else "<suppressed>")
        message = str(exc) if safe else (
            f"Processing failed ({type(exc).__name__}). "
            "Check the server log for details.")
        store().update_job(job_id, status="failed", error=message,
                           completed_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        store().apply(cid, lambda r: r.update(
            status="error", active_stage=None, error=message))
        job_event(logger, "doc_job_failed", cid=cid, job_id=job_id,
                  reason=type(exc).__name__, elapsed_s=timer.seconds)


def _run_job(cid: str, audio_path: str, language: str = "") -> None:
    """Worker body: bounded by the job semaphore, runs off the event loop."""
    if not rt.job_slots.acquire(timeout=QUEUE_TIMEOUT_SECONDS):
        store().apply(cid, lambda r: r.update(
            status="error", active_stage=None,
            error="Server busy: the job could not start in time. Please retry."))
        job_event(logger, "job_timed_out_in_queue", cid=cid)
        return

    with rt.jobs_lock:
        rt.active_jobs += 1
    timer = Timer()
    timer.__enter__()
    try:
        store().apply(cid, lambda r: r.update(status="processing", stages=[], error=None))
        job_event(logger, "job_started", cid=cid)

        engine = build_engine()
        engine.on_stage = _progress_callback(cid)
        result = engine.process_audio(audio_path, language=language)

        def finish(record: dict) -> None:
            record["result"] = result
            record["status"] = "ready"
            record["active_stage"] = None

        store().apply(cid, finish)
        meta = result.get("meta", {})
        validation = result.get("validation", {})
        job_event(
            logger, "job_completed", cid=cid,
            elapsed_s=timer.seconds,
            audio_duration_s=meta.get("audio_duration_s"),
            segments=len(result.get("transcript", {}).get("segments", [])),
            speaker_method=result.get("speakers", {}).get("method"),
            roles_known=result.get("speakers", {}).get("roles_known"),
            stt_provider=meta.get("stt_provider"),
            provider_selection=meta.get("provider_selection"),
            requested_language=meta.get("requested_language"),
            reported_language=meta.get("reported_language"),
            # Codes only — never the transcript that triggered them.
            validation=validation.get("severity"),
            validation_codes=",".join(validation.get("codes", [])) or None,
        )
    except AudioError as exc:
        store().apply(cid, lambda r: r.update(
            status="error", active_stage=None, error=str(exc)))
        job_event(logger, "job_failed", cid=cid, reason="audio_error", elapsed_s=timer.seconds)
    except Exception as exc:
        # Engine exceptions can embed transcript text, so the message is not
        # logged in general. Provider/transport errors are the exception:
        # they carry API status only (never the key - see DeepgramSTTProvider)
        # and without them a failed consultation is undiagnosable.
        safe = exc.__class__.__name__ in (
            "DeepgramError", "TranscriptionError", "SpeechLanguageUnavailable")
        logger.error("event=job_failed cid=%s error_type=%s elapsed_s=%s detail=%s",
                     cid, type(exc).__name__, timer.seconds,
                     str(exc)[:200] if safe else "<suppressed>")
        message = str(exc) if isinstance(exc, SpeechLanguageUnavailable) else (
            f"Processing failed ({type(exc).__name__}). "
            "Check the server log for details.")
        store().apply(cid, lambda r: r.update(
            status="error", active_stage=None, error=message))
    finally:
        with rt.jobs_lock:
            rt.active_jobs -= 1
        rt.job_slots.release()


@app.post("/api/consultations/{cid}/process")
def process(cid: str, request: Request):
    """Queue processing; poll GET /api/consultations/{cid} for real stage progress.

    Rate-limited per identity: STT is usage-based, so job submission is a
    metered action even for authenticated sessions.
    """
    ident = _current_identity(request)
    limiter = rt.processing_limiter
    if limiter is not None:
        key = f"process:{ident.get('email') or request.client.host if request.client else 'anon'}"
        if not limiter.allow(key):
            retry = int(limiter.retry_after(key)) or 60
            raise HTTPException(429, f"Too many processing requests; retry in {retry}s",
                                headers={"Retry-After": str(retry)})
    record = _get(cid, request)
    if record["status"] in ("processing", "queued"):
        return _public(record)
    if not record.get("audio_file"):
        raise HTTPException(400, "Upload audio first")
    if not Path(record["audio_file"]).exists():
        raise HTTPException(
            410, "The audio for this consultation has been deleted by the retention "
                 "policy. Upload it again to reprocess.")
    if settings.preload_models and not rt.models_ready:
        detail = (f"Speech model failed to load: {rt.model_error}" if rt.model_error
                  else "Speech model is still loading - try again in a few seconds.")
        raise HTTPException(503, detail)
    if rt.shutdown.is_set():
        raise HTTPException(503, "Server is shutting down")

    recording_key = Path(record["audio_file"]).name
    store().apply(cid, lambda r: r.update(status="queued", error=None, stages=[]))
    job = store().create_job(cid, recording_key)
    rt.job_queue.put((job["job_id"], cid, record["audio_file"],
                      record.get("language") or settings.language))
    retention().audit("transcription_requested", cid, actor=ident.get("email"))
    job_event(logger, "doc_job_queued", cid=cid, job_id=job["job_id"],
              queue_depth=rt.job_queue.qsize())
    return JSONResponse(status_code=202, content=_public(_get(cid, request)) | {
        "job_id": job["job_id"],
        "job_status": job["status"],
    })


@app.post("/api/consultations/{cid}/jobs/{job_id}/retry")
def retry_job(cid: str, job_id: str, request: Request):
    """Retry a FAILED documentation job.

    Safe by construction: the failed row keeps its terminal state and a fresh
    queued row is created; the clinical pipeline re-runs in full, so the note
    can only ever be produced by a validated pass.
    """
    record = _get(cid, request)
    job = store().get_job(job_id)
    if not job or job["consultation_id"] != cid:
        raise HTTPException(404, "Job not found")
    if job["status"] != "failed":
        raise HTTPException(409, f"Job is {job['status']}; only failed jobs can be retried")
    if not record.get("audio_file") or not Path(record["audio_file"]).exists():
        raise HTTPException(410, "Audio is no longer available; upload it again")
    if rt.shutdown.is_set():
        raise HTTPException(503, "Server is shutting down")

    store().apply(cid, lambda r: r.update(status="queued", error=None, stages=[]))
    new_job = store().create_job(cid, job["recording_key"])
    rt.job_queue.put((new_job["job_id"], cid, record["audio_file"],
                      record.get("language") or settings.language))
    job_event(logger, "doc_job_retried", cid=cid, job_id=new_job["job_id"],
              previous_job=job_id)
    return JSONResponse(status_code=202, content=_public(_get(cid, request)) | {
        "job_id": new_job["job_id"], "job_status": new_job["status"]})


@app.post("/api/consultations/{cid}/text")
async def process_text(cid: str, body: TextProcessRequest, request: Request):
    """Plain-text flow: process a Doctor:/Patient: transcript directly.

    spaCy runs for several seconds, so it goes to a worker thread rather than
    blocking the event loop (and therefore every other clinician's polling).
    """
    record = _get(cid, request)
    if not body.transcript.strip():
        raise HTTPException(400, "Transcript is empty")

    timer = Timer()
    timer.__enter__()
    try:
        result = await asyncio.to_thread(
            build_engine().process_text, body.transcript,
            record.get("language") or settings.language)
    except Exception as exc:
        logger.error("event=text_job_failed cid=%s error_type=%s", cid, type(exc).__name__)
        raise HTTPException(500, f"Processing failed ({type(exc).__name__})") from exc

    def mutate(record: dict) -> None:
        record["source"] = "text"
        record["result"] = result
        record["status"] = "ready"
        record["active_stage"] = None

    store().apply(cid, mutate)
    job_event(logger, "text_job_completed", cid=cid, elapsed_s=timer.seconds,
              chars=len(body.transcript))
    return _public(_get(cid, request))


@app.patch("/api/consultations/{cid}/review")
def review(cid: str, body: ReviewPatch, request: Request):
    ident = _require_named_user(request)
    record = _get(cid, request)
    if not record.get("result"):
        raise HTTPException(400, "Nothing to review yet")

    def mutate(rec: dict) -> None:
        note = rec["result"]["clinical_note"]
        rx = rec["result"]["prescription"]
        if body.clinical_note_fields:
            note["fields"].update(body.clinical_note_fields)
        if body.prescription_fields:
            rx["fields"].update(body.prescription_fields)
        # Re-render from the edited fields. Without this the export would still
        # carry the text block frozen at generation time, silently discarding
        # the doctor's corrections on the document that reaches the record.
        note["text"] = render_clinical_note_text(note["fields"])
        rx["text"] = render_prescription_text(rx["fields"])
        rec["reviewed"] = True

    store().apply(cid, mutate)
    # Doctor edited the note: the note is now REVIEWED (never approved).
    store().set_audio_lifecycle(cid, note_status="REVIEWED")
    retention().mark_review_started(cid)
    store().append_audit("note_edited", cid=cid,
                         hospital_id=record.get("hospital_id"),
                         actor=ident.get("email"))
    job_event(logger, "consultation_reviewed", cid=cid)
    return _public(_get(cid, request))


class ApprovalRequest(BaseModel):
    # Attribution is best-effort under the shared-token model; it records WHO
    # the session claims to be, not a verified identity.
    approved_by: str = Field(default="", max_length=120)


@app.post("/api/consultations/{cid}/approve")
def approve(cid: str, body: ApprovalRequest | None = None, request: Request = None):
    """Explicit doctor approval of the reviewed note.

    This is the retention-relevant clinical event: it closes the review
    window, records the approval (note_status=APPROVED + approval_version),
    evaluates the retention policy and either schedules deletion or makes
    the audio eligible for the (separately authorized) training path.
    Approval is refused while the note is still an unreviewed draft.
    """
    # Clinical sign-off requires a named, role-authorized identity: the
    # break-glass shared-key session cannot approve notes.
    ident = _require_role(request, *CLINICAL_SIGNOFF_ROLES)
    record = _get(cid, request)
    if not record.get("result"):
        raise HTTPException(400, "Process a consultation first")
    if store().has_pending_job(cid):
        raise HTTPException(409, "A documentation job is still running")
    lifecycle = store().get_audio_lifecycle(cid) or {}
    if (lifecycle.get("note_status") or "DRAFT") == "DRAFT":
        raise HTTPException(409, "Review the note before approving it")
    retention().mark_review_started(cid)
    # A verified session identity outranks any client-supplied attribution.
    actor = ident.get("email") or (body.approved_by if body else None)
    retention().approve(cid, actor=actor)
    store().append_audit("note_approved", cid=cid,
                         hospital_id=record.get("hospital_id"),
                         actor=ident.get("email"))
    return _public(_get(cid, request))


@app.post("/api/consultations/{cid}/complete")
def complete(cid: str, request: Request):
    """Completing IS the explicit doctor sign-off — same role gate as approve."""
    ident = _require_role(request, *CLINICAL_SIGNOFF_ROLES)
    record = _get(cid, request)
    if not record.get("result"):
        raise HTTPException(400, "Process a consultation first")
    if store().has_pending_job(cid):
        raise HTTPException(409, "A documentation job is still running")
    lifecycle = store().get_audio_lifecycle(cid) or {}
    # "Note generated" is never "approved": completing requires the doctor to
    # have reviewed (edited) the note; the approval event itself follows here.
    if (lifecycle.get("note_status") or "DRAFT") == "DRAFT":
        raise HTTPException(409, "Review (and edit) the note before completing it")

    def mutate(rec: dict) -> None:
        rec["status"] = "completed"
        rec["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    store().apply(cid, mutate)
    # Completing the consultation IS the explicit doctor sign-off: the note
    # transitions to APPROVED and the retention policy is evaluated now.
    retention().mark_review_started(cid)
    retention().approve(cid, actor=ident.get("email") or record.get("doctor"))
    store().append_audit("note_approved", cid=cid,
                         hospital_id=record.get("hospital_id"),
                         actor=ident.get("email"),
                         detail={"via": "complete"})
    job_event(logger, "consultation_completed", cid=cid, reviewed=record.get("reviewed", False))
    return _public(_get(cid, request))


@app.get("/api/consultations/{cid}/export")
def export(cid: str, request: Request):
    """Export the APPROVED note for the hospital's EHR workflow.

    The primary workflow is review → approve → download. The export carries
    the approved note only: no raw audio reference, no draft AI reasoning,
    no internal validation/debug sections. Exporting an unapproved note is a
    workflow error, not a policy footnote. Export is clinical sign-off: named,
    role-authorized sessions only.
    """
    ident = _require_role(request, *CLINICAL_SIGNOFF_ROLES)
    record = _get(cid, request)
    lifecycle = store().get_audio_lifecycle(cid) or {}
    if (lifecycle.get("note_status") or "DRAFT") != "APPROVED":
        raise HTTPException(409, "Approve the note before exporting it")
    store().append_audit("note_exported", cid=cid,
                         hospital_id=record.get("hospital_id"),
                         actor=ident.get("email"))
    job_event(logger, "consultation_exported", cid=cid)
    return PlainTextResponse(
        _export_text(record, approved_only=True),
        headers={
            "Content-Disposition": f'attachment; filename="consultation_{cid}.txt"',
            "Cache-Control": "no-store, private",
        },
    )


# --------------------------------------------------------------------------
# Audio access: session + short-lived consultation-bound token.
# --------------------------------------------------------------------------
@app.post("/api/consultations/{cid}/audio-link")
def audio_link(cid: str, request: Request):
    """Issue a short-lived download URL for this consultation's audio.

    Raw audio is NOT on a static mount and has no permanent URL: the link is
    signed, bound to this cid, and expires (15 minutes). The UI calls this
    when the doctor wants to listen back during review.
    """
    record = _get(cid, request)
    if not record.get("audio_file") or record.get("audio_purged"):
        raise HTTPException(404, "Audio is not available")
    if not Path(record["audio_file"]).exists():
        raise HTTPException(404, "Audio is not available")
    token = issue_download_token(settings.access_token, cid)
    return {
        "url": f"/api/consultations/{cid}/audio?expires={token}",
        "expires_in_seconds": DOWNLOAD_TOKEN_TTL_SECONDS,
    }


@app.get("/api/consultations/{cid}/audio")
def get_audio(cid: str, request: Request, expires: str = ""):
    """Serve raw audio only to a session that ALSO holds a valid, unexpired,
    cid-bound token. Fails closed on any absence or mismatch."""
    ident = _current_identity(request)
    record = _get(cid, request)
    if not record.get("audio_file") or record.get("audio_purged"):
        raise HTTPException(404, "Audio is not available")
    if not download_token_valid(expires, settings.access_token, cid):
        raise HTTPException(403, "Audio link is invalid or expired")
    path = Path(record["audio_file"])
    if not path.exists():
        raise HTTPException(404, "Audio is not available")
    store().append_audit("audio_accessed", cid=cid,
                         hospital_id=record.get("hospital_id"),
                         actor=ident.get("email"))
    return FileResponse(
        path,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store, private",
            "Content-Disposition": f'inline; filename="{path.name}"',
        },
    )


# --------------------------------------------------------------------------
# Retention policy + training authorization (fail-closed endpoints).
# --------------------------------------------------------------------------
@app.get("/api/retention/policy")
def retention_policy():
    """Effective redacted policy for the UI. Contains no secrets."""
    return retention().policy.as_public_dict()


@app.get("/api/stt/boundary")
def stt_boundary():
    """Admin-facing STT provider boundary view (configuration metadata only).

    Shows which provider is active, where it processes audio, and its
    documented retention posture — so the hospital's privacy team can verify
    the boundary without reading source. Never returns keys, credentials or
    patient data. Uses the standard session/token authentication model.
    """
    return stt_boundary_view(settings)


def _require_training_admin(request: Request) -> None:
    """Training administration needs a SEPARATE credential, fail closed."""
    if not settings.training_admin_token:
        raise HTTPException(403, "Training administration is disabled")
    supplied = request.headers.get("X-Training-Admin-Token", "")
    if not token_matches(supplied, settings.training_admin_token):
        raise HTTPException(403, "Training administration is disabled")


@app.post("/api/hospital/training-authorization")
def training_authorization(request: Request, body: TrainingAuthorizationRequest):
    """Grant or revoke hospital-level training authorization.

    This event is deliberately separate from 'the hospital uses the product'.
    It requires its own credential, an explicit version, and a finite expiry —
    an authorization without an expiry is rejected rather than defaulted.
    """
    _require_training_admin(request)
    if body.authorized:
        if not body.authorization_version.strip():
            raise HTTPException(422, "authorization_version is required")
        if body.authorization_expiry_seconds <= 0:
            raise HTTPException(422, "a finite authorization_expiry_seconds is required")
        row = store().set_hospital_config(
            settings.hospital_id,
            training_authorized=1,
            training_authorization_version=body.authorization_version.strip(),
            training_authorized_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            training_authorized_by="training-admin",
            training_authorization_expiry=time.time() + body.authorization_expiry_seconds,
            training_policy_reference=body.policy_reference.strip() or None,
            hospital_training_enabled=1 if settings.training_retention_enabled else 0,
        )
        retention().audit("training_authorization_granted", actor="training-admin",
                          authorization_version=body.authorization_version.strip(),
                          policy_reference=body.policy_reference.strip() or None,
                          expiry_seconds=body.authorization_expiry_seconds)
    else:
        # Revocation is an auditable event of its own: the consent row keeps
        # its history (never simply deleted), future ingestion is blocked by
        # the revocation guard, and withdrawal propagates: pending candidates
        # are deleted outright, dataset records are flagged for removal.
        store().set_hospital_config(
            settings.hospital_id,
            training_authorized=0,
            training_consent_revoked_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            training_consent_revoked_by="training-admin",
        )
        affected = retention().withdraw_training_consent(
            settings.hospital_id, "training-admin")
        retention().audit("training_consent_revoked", actor="training-admin",
                          affected_records=affected)
        row = store().get_hospital_config(settings.hospital_id) or {}
    return {"hospital_id": settings.hospital_id,
            "training_authorized": bool(row.get("training_authorized")),
            "consent_revoked": bool(row.get("training_consent_revoked_at"))}


@app.post("/api/consultations/{cid}/training-selection")
def training_selection(cid: str, body: TrainingSelectionRequest,
                       request: Request):
    """Explicitly select an APPROVED consultation's audio for the training
    dataset. Every guard fails closed; the clinical copy is moved (not copied)
    into the authorized vault namespace with its own retention deadline."""
    ident = _require_named_user(request)
    record = _get(cid, request)
    if record.get("audio_state") not in ("DOCTOR_APPROVED",
                                          "SCHEDULED_FOR_DELETION") \
            or record.get("audio_purged"):
        raise HTTPException(409,
                            "Audio can only enter the training dataset after "
                            "the note has been explicitly approved and before "
                            "it has been deleted")
    if not record.get("audio_file"):
        raise HTTPException(404, "Audio is not available")
    try:
        rec = retention().create_training_copy(
            cid, actor=ident.get("email") or record.get("doctor") or "shared-session",
            selection_reason=body.reason)
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    return {
        "training_record_id": rec["training_record_id"],
        "retention_deadline": rec["retention_deadline"],
        "review_status": rec["review_status"],
        "deletion_status": rec["deletion_status"],
    }


# --------------------------------------------------------------------------
# Training candidate review staging (authorized reviewer only).
# A candidate is NOT in the training corpus until a human approves it.
# --------------------------------------------------------------------------
def _candidate_public(rec: dict) -> dict:
    """Reviewer-safe view: ids, statuses, the DE-IDENTIFIED transcript and
    provenance. Never the raw transcript, never a vault filesystem path."""
    try:
        provenance = json.loads(rec.get("deidentification_provenance") or "{}")
    except Exception:
        provenance = {}
    return {
        "training_record_id": rec["training_record_id"],
        "source_consultation_id": rec["source_consultation_id"],
        "hospital_id": rec["hospital_id"],
        "authorization_reference": rec["authorization_reference"],
        "sha256": rec["sha256"],
        "deidentification_status": rec.get("deidentification_status"),
        "deidentified_transcript": rec.get("deidentified_transcript"),
        "deidentification_provenance": provenance,
        "review_status": rec["review_status"],
        "reviewed_at": rec.get("reviewed_at"),
        "reviewed_by": rec.get("reviewed_by"),
        "selection_reason": rec.get("selection_reason"),
        "retention_deadline": rec["retention_deadline"],
        "deletion_status": rec["deletion_status"],
        "revoked": rec.get("revoked"),
        "created_at": rec["created_at"],
    }


@app.get("/api/training/candidates")
def training_candidates(request: Request, status: str = "pending"):
    _require_training_admin(request)
    if status not in ("pending", "approved", "rejected", "all"):
        raise HTTPException(422, "Unknown status filter")
    rows = store().list_training_records()
    rows = [r for r in rows
            if status == "all" or r["review_status"] == status]
    return [_candidate_public(r) for r in rows]


class TrainingReviewRequest(BaseModel):
    decision: str = Field(pattern="^(approve|reject)$")
    reviewer: str = Field(default="training-admin", max_length=120)
    note: str = Field(default="", max_length=500)


@app.post("/api/training/candidates/{training_record_id}/review")
def review_training_candidate(training_record_id: str,
                              request: Request,
                              body: TrainingReviewRequest):
    _require_training_admin(request)
    try:
        rec = retention().review_training_candidate(
            training_record_id, body.decision,
            reviewer=body.reviewer or "training-admin",
            note=body.note.strip() or None)
    except KeyError:
        raise HTTPException(404, "Training candidate not found")
    except PermissionError as exc:
        raise HTTPException(409, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return _candidate_public(rec)


@app.get("/api/demo/transcript")
def demo_transcript():
    """Bundled synthetic transcript for smoke-testing. Contains no real PHI."""
    return {
        "label": "synthetic test data",
        "transcript": (
            "Doctor: Hello, good morning! How can I help you today?\n"
            "Patient: Good morning, doctor. I'm Ramesh. I've been feeling some chest pain "
            "and shortness of breath.\n"
            "Doctor: When did it start?\n"
            "Patient: I think it started around 5 days ago.\n"
            "Doctor: Have you experienced any other symptoms?\n"
            "Patient: I've also had fatigue and a bit of nausea.\n"
            "Doctor: Based on what you're telling me, it looks like angina. How old are you?\n"
            "Patient: I am 45 years old.\n"
            "Doctor: Got it. We'll run a few more tests to confirm."
        ),
    }


# --------------------------------------------------------------------------
# Error handling
# --------------------------------------------------------------------------
@app.exception_handler(AudioError)
async def audio_error_handler(request: Request, exc: AudioError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    """Never leak a stack trace or engine message to the browser."""
    logger.error("event=unhandled_error path=%s error_type=%s",
                 request.url.path, type(exc).__name__)
    logger.exception("unhandled error")
    return JSONResponse(
        {"detail": "Internal server error. The incident was logged."}, status_code=500
    )


# --------------------------------------------------------------------------
# Static UI — mounted last so /api/* wins. Resolved from the package directory,
# not the process working directory.
# --------------------------------------------------------------------------
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
