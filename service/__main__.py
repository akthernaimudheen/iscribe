"""Production entrypoint: ``python -m service``.

Runs uvicorn with the settings resolved from the environment, so the same
command works in development and production and no host/port is hard-coded.

Single worker by design. The Whisper model cache, the job semaphore and the
SQLite connection are all per-process; multiple workers would multiply RAM by
the number of workers (~400 MB each) and let jobs bypass the concurrency cap.
Scale by moving to a bigger host, not by adding workers.
"""

from __future__ import annotations

import sys

import uvicorn

from .config import ConfigError, load_settings


def main() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(f"iScribe configuration error: {exc}", file=sys.stderr)
        return 2

    print(f"iScribe {settings.env} -> http://{settings.host}:{settings.port}", file=sys.stderr)
    uvicorn.run(
        "service.app:app",
        host=settings.host,
        port=settings.port,
        workers=1,
        log_level=settings.log_level.lower(),
        access_log=True,
        proxy_headers=True,          # behind the Cloudflare tunnel / any reverse proxy
        forwarded_allow_ips="*",
        timeout_graceful_shutdown=45,
        # Transcription is CPU-heavy and briefly starves the event loop. The
        # 5 s default would close the UI's polling connection mid-job and make a
        # successful transcription look like a failure to the clinician.
        timeout_keep_alive=60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
