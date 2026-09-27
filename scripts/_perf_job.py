"""Drive one real transcription job against the running server (perf harness).

Used by the deployment performance measurement. Synthetic audio only.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import httpx  # noqa: E402

from smoke_test import load_access_key  # noqa: E402

BASE = "http://127.0.0.1:8123"
AUDIO = Path(__file__).resolve().parent.parent / "uploads" / "939dcfab957e.mp3"

with httpx.Client(base_url=BASE, timeout=300.0, follow_redirects=False) as c:
    c.post("/api/login", data={"access_token": load_access_key()})
    c.headers.update({"X-Access-Token": load_access_key()})
    cid = c.post("/api/consultations", json={"patient_id": "PERF-TEST"}).json()["id"]
    with AUDIO.open("rb") as fh:
        c.post(f"/api/consultations/{cid}/audio",
               files={"file": (AUDIO.name, fh, "audio/mpeg")})
    t0 = time.perf_counter()
    c.post(f"/api/consultations/{cid}/process")
    while time.perf_counter() - t0 < 600:
        time.sleep(1.0)
        try:
            cur = c.get(f"/api/consultations/{cid}").json()
        except httpx.HTTPError:
            continue
        if cur["status"] in ("ready", "error"):
            meta = (cur.get("result") or {}).get("meta", {})
            print(f"status={cur['status']} "
                  f"elapsed_s={round(time.perf_counter() - t0, 2)} "
                  f"audio_s={meta.get('audio_duration_s')} "
                  f"engine_s={meta.get('elapsed_s')}")
            break
    else:
        print("status=timeout")
