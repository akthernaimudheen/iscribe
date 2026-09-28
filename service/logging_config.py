"""Operational logging for a clinical deployment.

Policy: logs carry *operational* facts only — consultation ids (server-generated
random hex, not derived from any patient identifier), job stage names, durations,
byte counts, status codes and error types. They must never carry patient names,
transcripts, clinical notes, prescriptions or audio content.

Two mechanisms enforce that:

1. Application code logs through :func:`job_event`, which accepts only scalar
   operational fields.
2. :class:`PHIRedactionFilter` is a backstop on every handler. Third-party
   libraries are not audited line by line, so any record whose message looks
   like it carries transcript-sized free text is truncated, and the known
   content-bearing loggers are silenced below WARNING.
"""

from __future__ import annotations

import logging
import logging.handlers
import time
from pathlib import Path

# Loggers that are known to echo recognised speech or model content at INFO.
# Kept at WARNING so their operational failures still surface.
CONTENT_BEARING_LOGGERS = (
    "faster_whisper",
    "ctranslate2",
    "speechbrain",
    "nemo_logger",
    "transformers",
)

MAX_MESSAGE_CHARS = 300


class AccessLogQueryScrubFilter(logging.Filter):
    """Strip query strings from uvicorn access-log records.

    The audio replay endpoint issues short-lived, cid-bound tokens as a query
    parameter (``?expires=...``). Uvicorn's default access log records the
    full request target, so every audio replay would persist a usable token
    into a plain-text log file. Access logs never need query strings, so the
    filter drops them wholesale. Applies to the structured-args records
    uvicorn emits by default and to pre-formatted strings as a fallback.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.args:
            args = list(record.args)
            for i, arg in enumerate(args):
                if isinstance(arg, str) and "?" in arg and " " not in arg:
                    args[i] = arg.split("?", 1)[0]
            record.args = tuple(args)
            return True
        if isinstance(record.msg, str) and "?" in record.msg:
            import re

            record.msg = re.sub(r"\?[^\s\"]*", "", record.msg)
        return True


def attach_access_log_scrubber() -> None:
    """Attach the query scrubber to uvicorn's access logger (idempotent).

    Called from the app lifespan so it runs AFTER uvicorn has configured its
    own loggers (config at import time would be overwritten by uvicorn.run).
    """
    access_logger = logging.getLogger("uvicorn.access")
    for f in access_logger.filters:
        if isinstance(f, AccessLogQueryScrubFilter):
            return
    access_logger.addFilter(AccessLogQueryScrubFilter())


class PHIRedactionFilter(logging.Filter):
    """Backstop: truncate over-long log messages from third-party libraries.

    A long free-text log line is the shape PHI leakage takes. Our own records
    are short and structured, so truncation costs nothing operationally.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        if len(message) > MAX_MESSAGE_CHARS:
            record.msg = message[:MAX_MESSAGE_CHARS] + f"... [truncated {len(message)} chars]"
            record.args = ()
        return True


def configure_logging(level: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    numeric = getattr(logging, level.upper(), logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    redaction = PHIRedactionFilter()

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    console.addFilter(redaction)

    # 5 files x 5 MB is weeks of operational logs at trial volume.
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "iscribe.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    file_handler.addFilter(redaction)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric)
    root.addHandler(console)
    root.addHandler(file_handler)

    for name in CONTENT_BEARING_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # The HF hub client logs full model URLs at INFO on every load.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    return logging.getLogger("iscribe")


def job_event(logger: logging.Logger, event: str, cid: str | None = None, **fields) -> None:
    """Log one operational event as key=value pairs.

    Only scalars are accepted; anything else is replaced with its type name so a
    stray dict of clinical fields cannot be serialised into the log.
    """
    parts = [f"event={event}"]
    if cid:
        parts.append(f"cid={cid}")
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            text = str(value)
            if len(text) > 120:
                text = f"<{type(value).__name__}:{len(text)}chars>"
            parts.append(f"{key}={text}")
        else:
            parts.append(f"{key}=<{type(value).__name__}>")
    logger.info(" ".join(parts))


class Timer:
    """Context manager returning elapsed seconds, for job timing logs."""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = round(time.perf_counter() - self._t0, 2)
        return False

    @property
    def seconds(self) -> float:
        return round(time.perf_counter() - self._t0, 2)
