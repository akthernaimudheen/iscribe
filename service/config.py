"""Runtime configuration for iScribe, read from the environment.

Every deployment-specific value lives here so nothing is hard-coded in the
application and no secret is ever committed. Defaults are chosen so that a bare
`python -m service` still runs the local development demo exactly as before;
production behaviour is opt-in via ISCRIBE_ENV=production, which additionally
*requires* an access token rather than defaulting to an insecure one.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

# Repository root — resolved from this file so the app no longer depends on the
# process working directory (the old code used the relative path
# "service/static" and only started when cwd happened to be the repo root).
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Load .env from the repository root if present. Real environment variables
# always win, so a container's injected secrets are never overwritten by a file
# that happened to be left in the image.
try:  # pragma: no cover - optional convenience dependency
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=False)
except ImportError:
    pass


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a safe configuration."""


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _env_list(name: str) -> list[str]:
    raw = _env(name)
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    # -- deployment -------------------------------------------------------
    env: str = "development"
    host: str = "127.0.0.1"
    port: int = 8123

    # -- access control ---------------------------------------------------
    # A single shared token gates the whole app. Adequate for a small, named
    # hospital trial; it is NOT per-user identity and must be rotated after.
    access_token: str = ""
    auth_required: bool = False
    session_hours: int = 12
    cookie_secure: bool = True
    cors_origins: list[str] = field(default_factory=list)

    # -- storage ----------------------------------------------------------
    data_dir: Path = BASE_DIR / "data"
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    db_path: Path = BASE_DIR / "data" / "iscribe.db"
    log_dir: Path = BASE_DIR / "data" / "logs"
    # Separate storage namespace for authorized training audio. Never written
    # to unless a valid, audited hospital training authorization exists.
    training_vault_dir: Path = BASE_DIR / "data" / "training_vault"

    # -- tenancy -----------------------------------------------------------
    # Server-assigned hospital boundary. Consultations inherit this from the
    # server, never from client-supplied request data, so a forged hospital_id
    # in a request body cannot cross a tenant boundary.
    hospital_id: str = "default"

    # -- limits -----------------------------------------------------------
    max_upload_mb: int = 100
    max_concurrent_jobs: int = 1

    # -- clinical audio retention (lifecycle-driven, favours deletion) ------
    # After explicit doctor approval the audio becomes deletion-eligible and is
    # deleted at approval + this many hours (0 = immediately at approval).
    audio_retention_after_note_approval_hours: int = 24
    # Ceiling for audio the doctor never approves: kept this many hours from
    # upload so a review is still possible, then deleted. 0 disables the
    # ceiling (audio would wait for approval indefinitely) — allowed for
    # development, discouraged in production.
    audio_retention_unreviewed_hours: int = 168
    # How often the lifecycle sweep evaluates deadlines and executes deletions.
    retention_sweep_seconds: int = 900

    # -- training retention (fail-closed; every default denies) -------------
    # Training retention exists only when the hospital has explicitly enabled
    # it AND a current authorization exists AND the recording is explicitly
    # selected (when required). Enabling this never retains anything by itself.
    training_retention_enabled: bool = False
    # Training copies have their OWN expiry, counted from training-copy
    # creation — independent of the clinical record's deletion.
    training_retention_period_hours: int = 24 * 365
    training_data_requires_explicit_selection: bool = True
    training_data_requires_hospital_authorization: bool = True
    # Separate credential that may grant/revoke hospital training
    # authorization. Unset => the authorization endpoint fails closed.
    training_admin_token: str = ""

    # -- engine -----------------------------------------------------------
    # Empty = automatic production priority (Deepgram when configured, else
    # faster-whisper). Set explicitly to pin one provider.
    stt_provider: str = ""
    # The production path is English-only and never auto-detects.
    language: str = "en"
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-3-medical"
    # Regional data-residency endpoint (global|eu|au|in or a full host).
    # global = api.deepgram.com, which carries NO residency guarantee.
    deepgram_region: str = "global"
    # Deepgram Model Improvement Program opt-out (zero data retention).
    # Default ON: provider documentation states requests participate in MIP
    # by default and are retained for training; opted-out requests are
    # "retained only for the duration needed to process the request".
    deepgram_mip_opt_out: bool = True
    whisper_model: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    # 0 = library default (every core). Capped by default so transcription
    # cannot starve the event loop that serves the UI.
    whisper_cpu_threads: int = 4
    preload_models: bool = True

    # -- STT policy (provider boundary) -------------------------------------
    # Fail-closed: when enabled, a provider failure NEVER falls back to
    # another provider — the job fails safely and the source audio stays
    # under the normal retention lifecycle. Unexpected provider switching is
    # a privacy/residency/auditability risk in hospital deployments.
    # The dataclass default stays False (library/CLI behaviour); the SERVICE
    # derives its production default from ISCRIBE_ENV in load_settings().
    stt_fail_closed: bool = False
    # Pre-flight policy check: the provider's declared boundary (retention
    # posture, region, off-host) must satisfy this hospital's policy before
    # any audio is sent. ON by default: refusing to send is always safe.
    stt_policy_enforced: bool = True

    # -- logging ----------------------------------------------------------
    log_level: str = "INFO"

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.upload_dir, self.db_path.parent, self.log_dir):
            path.mkdir(parents=True, exist_ok=True)

    def redacted(self) -> dict:
        """Safe-to-log view: never includes the token itself."""
        return {
            "env": self.env,
            "host": self.host,
            "port": self.port,
            "auth_required": self.auth_required,
            "access_token_set": bool(self.access_token),
            "cors_origins": self.cors_origins or ["<same-origin only>"],
            "data_dir": str(self.data_dir),
            "max_upload_mb": self.max_upload_mb,
            "hospital_id": self.hospital_id,
            "audio_retention_after_note_approval_hours":
                self.audio_retention_after_note_approval_hours,
            "audio_retention_unreviewed_hours": self.audio_retention_unreviewed_hours,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "training_retention_enabled": self.training_retention_enabled,
            "training_retention_period_hours": self.training_retention_period_hours,
            "training_data_requires_explicit_selection":
                self.training_data_requires_explicit_selection,
            "training_data_requires_hospital_authorization":
                self.training_data_requires_hospital_authorization,
            "training_admin_token_set": bool(self.training_admin_token),
            "training_vault_dir": str(self.training_vault_dir),
            "stt_provider": self.stt_provider or "<auto: deepgram then faster-whisper>",
            "stt_fail_closed": self.stt_fail_closed,
            "stt_policy_enforced": self.stt_policy_enforced,
            "language": self.language,
            "deepgram_configured": bool(self.deepgram_api_key),
            "deepgram_model": self.deepgram_model,
            "deepgram_region": self.deepgram_region,
            "deepgram_mip_opt_out": self.deepgram_mip_opt_out,
            "whisper_model": self.whisper_model,
            "whisper_device": self.whisper_device,
            "whisper_compute_type": self.whisper_compute_type,
            "whisper_cpu_threads": self.whisper_cpu_threads,
            "preload_models": self.preload_models,
        }


def load_settings() -> Settings:
    env = (_env("ISCRIBE_ENV", "development") or "development").lower()
    if env not in ("development", "production"):
        raise ConfigError("ISCRIBE_ENV must be 'development' or 'production'")
    is_prod = env == "production"

    data_dir = Path(_env("ISCRIBE_DATA_DIR") or (BASE_DIR / "data")).resolve()
    upload_dir = Path(_env("ISCRIBE_UPLOAD_DIR") or (data_dir / "uploads")).resolve()
    db_path = Path(_env("ISCRIBE_DB_PATH") or (data_dir / "iscribe.db")).resolve()
    log_dir = Path(_env("ISCRIBE_LOG_DIR") or (data_dir / "logs")).resolve()
    training_vault_dir = Path(_env("ISCRIBE_TRAINING_VAULT_DIR")
                              or (data_dir / "training_vault")).resolve()

    token = _env("ISCRIBE_ACCESS_TOKEN", "") or ""
    auth_required = _env_bool("ISCRIBE_AUTH_REQUIRED", is_prod)

    if auth_required and not token:
        if is_prod:
            # Refuse to serve PHI without a gate. Fail loudly at startup rather
            # than quietly generating a token nobody knows.
            raise ConfigError(
                "ISCRIBE_ACCESS_TOKEN is required when ISCRIBE_ENV=production. "
                "Generate one with: python -c \"import secrets; print(secrets.token_urlsafe(32))\""
            )
        token = secrets.token_urlsafe(32)

    if token and len(token) < 16:
        raise ConfigError("ISCRIBE_ACCESS_TOKEN must be at least 16 characters")

    return Settings(
        env=env,
        host=_env("ISCRIBE_HOST", "0.0.0.0" if is_prod else "127.0.0.1"),
        port=_env_int("ISCRIBE_PORT", 8123),
        access_token=token,
        auth_required=auth_required,
        session_hours=_env_int("ISCRIBE_SESSION_HOURS", 12),
        cookie_secure=_env_bool("ISCRIBE_COOKIE_SECURE", is_prod),
        cors_origins=_env_list("ISCRIBE_CORS_ORIGINS"),
        data_dir=data_dir,
        upload_dir=upload_dir,
        db_path=db_path,
        log_dir=log_dir,
        training_vault_dir=training_vault_dir,
        hospital_id=(_env("ISCRIBE_HOSPITAL_ID", "default") or "default"),
        max_upload_mb=_env_int("ISCRIBE_MAX_UPLOAD_MB", 100),
        audio_retention_after_note_approval_hours=_env_int(
            "ISCRIBE_AUDIO_RETENTION_AFTER_APPROVAL_HOURS", 24),
        audio_retention_unreviewed_hours=_env_int(
            "ISCRIBE_AUDIO_RETENTION_UNREVIEWED_HOURS", 168),
        retention_sweep_seconds=_env_int("ISCRIBE_RETENTION_SWEEP_SECONDS", 900),
        training_retention_enabled=_env_bool("ISCRIBE_TRAINING_RETENTION_ENABLED", False),
        training_retention_period_hours=_env_int(
            "ISCRIBE_TRAINING_RETENTION_PERIOD_HOURS", 24 * 365),
        training_data_requires_explicit_selection=_env_bool(
            "ISCRIBE_TRAINING_REQUIRES_EXPLICIT_SELECTION", True),
        training_data_requires_hospital_authorization=_env_bool(
            "ISCRIBE_TRAINING_REQUIRES_HOSPITAL_AUTHORIZATION", True),
        training_admin_token=_env("ISCRIBE_TRAINING_ADMIN_TOKEN", "") or "",
        max_concurrent_jobs=_env_int("ISCRIBE_MAX_CONCURRENT_JOBS", 1),
        stt_provider=_env("ISCRIBE_STT_PROVIDER", "") or "",
        language=_env("ISCRIBE_LANGUAGE", "en"),
        deepgram_api_key=_env("DEEPGRAM_API_KEY", "") or "",
        deepgram_model=_env("ISCRIBE_DEEPGRAM_MODEL", "nova-3-medical"),
        deepgram_region=(_env("ISCRIBE_DEEPGRAM_REGION", "global") or "global").lower(),
        deepgram_mip_opt_out=_env_bool("ISCRIBE_DEEPGRAM_MIP_OPT_OUT", True),
        whisper_model=_env("ISCRIBE_WHISPER_MODEL", "base"),
        whisper_device=_env("ISCRIBE_WHISPER_DEVICE", "cpu"),
        whisper_compute_type=_env("ISCRIBE_WHISPER_COMPUTE_TYPE", "int8"),
        whisper_cpu_threads=_env_int("ISCRIBE_WHISPER_CPU_THREADS", 4),
        preload_models=_env_bool("ISCRIBE_PRELOAD_MODELS", True),
        # Fail-closed is the production default, derived from ISCRIBE_ENV so a
        # deployment never silently enables automatic provider fallback just
        # because the variable was never written down. An explicit
        # ISCRIBE_STT_FAIL_CLOSED=true|false always overrides the default.
        stt_fail_closed=_env_bool("ISCRIBE_STT_FAIL_CLOSED", is_prod),
        stt_policy_enforced=_env_bool("ISCRIBE_STT_POLICY_ENFORCED", True),
        log_level=(_env("ISCRIBE_LOG_LEVEL", "INFO") or "INFO").upper(),
    )
