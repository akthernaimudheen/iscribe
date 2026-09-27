"""End-to-end production smoke test for a running iScribe deployment.

Exercises the real clinical workflow against a live server: sign in, create a
consultation, upload audio, transcribe, extract, review, complete, export, and
reopen. Also checks the things that must NOT happen — secrets reaching the
browser, PHI reaching the log, clinical endpoints answering without a session.

Use synthetic audio only. Anything uploaded is processed and stored for real.

    python scripts/smoke_test.py --audio path/to/synthetic.mp3
    python scripts/smoke_test.py --base-url https://example.trycloudflare.com

The access key is read from .env (or ISCRIBE_ACCESS_TOKEN) and is never printed.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

# Content that appears in the synthetic sample and must never reach the log.
PHI_CANARIES = ("allergies", "energies", "Ramesh", "chest pain", "angina")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def load_access_key() -> str:
    key = os.environ.get("ISCRIBE_ACCESS_TOKEN", "").strip()
    if key:
        return key
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("ISCRIBE_ACCESS_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8123")
    parser.add_argument("--audio", default=None,
                        help="Synthetic audio file. Skips the STT flow if omitted.")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="Seconds to wait for transcription.")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    key = load_access_key()
    print(f"\niScribe smoke test -> {base}\n")

    timings: dict[str, float] = {}

    # ---- 0. server reachable ------------------------------------------
    print("0. Server")
    with httpx.Client(base_url=base, timeout=30.0, follow_redirects=False) as anon:
        try:
            health = anon.get("/api/health")
        except Exception as exc:
            check("server reachable", False, f"{type(exc).__name__}: {exc}")
            return summarise()
        check("health endpoint", health.status_code == 200, f"HTTP {health.status_code}")

        # Wait for the warm-up to finish. During it the process is CPU-bound and
        # answers slowly, which is exactly why /api/ready exists: a proxy or
        # supervisor should not route clinicians here until it returns 200.
        ready_deadline = time.time() + 180
        ready_body: dict = {}
        while time.time() < ready_deadline:
            try:
                ready_body = anon.get("/api/ready", timeout=60.0).json()
            except Exception:
                ready_body = {}
            if ready_body.get("models_ready") or ready_body.get("status") == "error":
                break
            print("       waiting for the speech model to load...", flush=True)
            time.sleep(5)
        check("speech model ready", ready_body.get("models_ready") is True,
              f"model={ready_body.get('whisper_model')} device={ready_body.get('device')} "
              f"load={ready_body.get('model_load_s')}s")
        # Provider honesty: Deepgram when a key is configured, local
        # faster-whisper otherwise. Either is correct; silent switching is not.
        expected_provider = ("deepgram" if ready_body.get("deepgram_configured")
                             else "current_faster_whisper")
        check("STT provider matches configuration",
              ready_body.get("stt_provider") == expected_provider,
              f"{ready_body.get('stt_provider')} (expected {expected_provider})")

        # ---- 1. access control ----------------------------------------
        print("\n1. Access control")
        check("clinical API rejects anonymous callers",
              anon.get("/api/consultations").status_code == 401)
        check("UI redirects anonymous browser to sign-in",
              anon.get("/").status_code == 303)
        check("wrong key rejected",
              anon.post("/api/login", data={"access_token": "wrong-key-value-x"}).status_code == 401)

    if not key:
        check("access key available to the test", False, "set ISCRIBE_ACCESS_TOKEN or .env")
        return summarise()

    # Generous timeout: the server is single-process and CPU-bound while a job
    # runs, so an unrelated request can legitimately queue behind it.
    with httpx.Client(base_url=base, timeout=180.0, follow_redirects=False) as client:
        login = client.post("/api/login", data={"access_token": key})
        check("sign-in with correct key", login.status_code == 303, f"HTTP {login.status_code}")
        set_cookie = login.headers.get("set-cookie", "")
        check("session cookie is HttpOnly", "httponly" in set_cookie.lower())
        check("access key not echoed to the browser", key not in set_cookie)
        if login.status_code != 303:
            return summarise()

        if "secure" in set_cookie.lower() and base.startswith("http://"):
            # Correct behaviour, not a defect: a Secure cookie must not be sent
            # over plain HTTP, so the session legitimately does not work here.
            check("Secure cookie not sent over plain HTTP (expected)",
                  client.get("/api/consultations").status_code == 401,
                  "cookie session is exercised by the https:// run")
            client.headers.update({"X-Access-Token": key})
        else:
            check("session cookie authenticates subsequent requests",
                  client.get("/api/consultations").status_code == 200)

        # ---- 1b. named clinical session ---------------------------------
        # The shared key is break-glass: it must NOT be able to review,
        # approve or export. Clinical actions need a named DOCTOR account.
        # The smoke doctor is created out-of-band on the host (README:
        # 'bootstrap the admin, then create users'); its credentials come
        # from the environment when provided.
        doctor_email = os.environ.get("SMOKE_DOCTOR_EMAIL", "").strip()
        doctor_password = os.environ.get("SMOKE_DOCTOR_PASSWORD", "").strip()
        if doctor_email and doctor_password:
            doctor = httpx.Client(base_url=base, timeout=180.0,
                                  follow_redirects=False)
            dlogin = doctor.post("/api/login", data={"email": doctor_email,
                                                     "password": doctor_password})
            check("doctor sign-in (named account)", dlogin.status_code == 303,
                  f"HTTP {dlogin.status_code}")
            if dlogin.status_code == 303:
                # The break-glass key may open consultations (read-only) but
                # must never reach clinical sign-off: approve/complete/export.
                probe_cid = client.post("/api/consultations", json={}).json()["id"]
                check("shared-key session CANNOT approve (read-only by design)",
                      client.post(f"/api/consultations/{probe_cid}/approve").status_code
                      in (403, 401),
                      "break-glass key must stay clinically read-only")
                client.cookies.update(doctor.cookies)
                check("doctor session authenticates clinical actions",
                      client.get("/api/consultations").status_code == 200)
            doctor.close()
        else:
            check("doctor credentials provided (SMOKE_DOCTOR_EMAIL/PASSWORD)",
                  False, "clinical sign-off steps will be skipped")

        check("UI reachable once signed in", client.get("/").status_code == 200)
        check("app.js reachable once signed in", client.get("/app.js").status_code == 200)

        # ---- 2. create consultation -----------------------------------
        print("\n2. Create consultation")
        created = client.post("/api/consultations", json={
            "patient_id": "SMOKE-TEST-001",
            "doctor": "Dr. Smoke Test",
            "department": "General Medicine",
            "consultation_type": "General Consultation",
        })
        check("create consultation", created.status_code == 200, f"HTTP {created.status_code}")
        record = created.json()
        cid = record["id"]
        print(f"       consultation id: {cid}")
        check("no server path in response",
              "audio_file" not in record and "audio_path" not in record)

        listing = client.get("/api/consultations").json()
        check("appears in dashboard", any(c["id"] == cid for c in listing))

        # ---- 3. text flow ---------------------------------------------
        print("\n3. Clinical extraction (text flow)")
        transcript = client.get("/api/demo/transcript").json()["transcript"]
        t0 = time.perf_counter()
        text_res = client.post(f"/api/consultations/{cid}/text", json={"transcript": transcript})
        timings["text_flow_s"] = round(time.perf_counter() - t0, 2)
        check("process transcript", text_res.status_code == 200, f"HTTP {text_res.status_code}")
        body = text_res.json()
        note = body["result"]["clinical_note"]["fields"]
        rx = body["result"]["prescription"]["fields"]
        check("clinical note generated", bool(note.get("symptoms_reported")),
              str(note.get("symptoms_reported"))[:60])
        check("prescription/plan generated", "medications" in rx)
        check("speakers identified",
              body["result"]["speakers"]["method"] == "explicit_roles",
              body["result"]["speakers"]["method"])
        print(f"       text flow: {timings['text_flow_s']}s")

        # ---- 4. review -------------------------------------------------
        print("\n4. Review")
        marker = "SMOKE-TEST-EDIT-MARKER"
        reviewed = client.patch(f"/api/consultations/{cid}/review", json={
            "clinical_note_fields": {"impression": marker},
        })
        if not doctor_email:
            # No named account configured: the shared-key session is correctly
            # refused. Verify the refusal and skip the clinical sign-off steps.
            check("review refused to a break-glass session (expected)",
                  reviewed.status_code == 403,
                  "set SMOKE_DOCTOR_EMAIL/PASSWORD to exercise review/export")
            print("\n(Clinical sign-off steps skipped: no named doctor "
                  "credentials — see SMOKE_DOCTOR_EMAIL/SMOKE_DOCTOR_PASSWORD)")
            return summarise()
        check("save review edits", reviewed.status_code == 200)
        check("edits persisted",
              reviewed.json()["result"]["clinical_note"]["fields"]["impression"] == marker)

        # ---- 5. complete + export --------------------------------------
        print("\n5. Complete and export")
        completed = client.post(f"/api/consultations/{cid}/complete")
        check("complete consultation", completed.json().get("status") == "completed")

        export = client.get(f"/api/consultations/{cid}/export")
        check("export downloads", export.status_code == 200)
        check("export carries review edits", marker in export.text)
        check("export marks the approved document",
              "approved" in export.text.lower())
        check("export marked as downloadable",
              "attachment" in export.headers.get("content-disposition", ""))
        check("export not cacheable", "no-store" in export.headers.get("cache-control", ""))

        # ---- 6. reopen -------------------------------------------------
        print("\n6. Reopen / refresh behaviour")
        reopened = client.get(f"/api/consultations/{cid}").json()
        check("reopen shows completed state", reopened["status"] == "completed")
        check("reopen still has the note",
              reopened["result"]["clinical_note"]["fields"]["impression"] == marker)

        # ---- 7. error handling -----------------------------------------
        print("\n7. Error handling")
        check("unknown consultation -> 404",
              client.get("/api/consultations/nosuchthing").status_code == 404)
        empty = client.post("/api/consultations", json={}).json()["id"]
        check("process without audio -> 400",
              client.post(f"/api/consultations/{empty}/process").status_code == 400)
        bad = client.post(f"/api/consultations/{empty}/audio",
                          files={"file": ("x.exe", b"nope", "application/octet-stream")})
        check("unsupported format -> 415", bad.status_code == 415, f"HTTP {bad.status_code}")
        blank = client.post(f"/api/consultations/{empty}/audio",
                            files={"file": ("x.mp3", b"", "audio/mpeg")})
        check("empty upload -> 400", blank.status_code == 400, f"HTTP {blank.status_code}")

        # ---- 8. audio flow ---------------------------------------------
        if args.audio:
            audio_path = Path(args.audio)
            print(f"\n8. Speech-to-text flow ({audio_path.name})")
            if not audio_path.exists():
                check("audio fixture exists", False, str(audio_path))
            else:
                acid = client.post("/api/consultations", json={
                    "patient_id": "SMOKE-TEST-AUDIO", "doctor": "Dr. Smoke Test",
                }).json()["id"]

                t0 = time.perf_counter()
                with audio_path.open("rb") as fh:
                    up = client.post(f"/api/consultations/{acid}/audio",
                                     files={"file": (audio_path.name, fh, "audio/mpeg")},
                                     timeout=300.0)
                timings["upload_s"] = round(time.perf_counter() - t0, 2)
                check("upload audio", up.status_code == 200, f"HTTP {up.status_code}")
                check("upload response hides server path", "audio_file" not in up.json())
                print(f"       upload: {timings['upload_s']}s "
                      f"({audio_path.stat().st_size/1024:.0f} KB)")

                t0 = time.perf_counter()
                started = client.post(f"/api/consultations/{acid}/process")
                check("start processing", started.status_code == 200,
                      f"HTTP {started.status_code}")

                final, seen_stages = None, set()
                poll_failures = 0
                deadline = time.time() + args.timeout
                while time.time() < deadline:
                    time.sleep(1.0)
                    try:
                        cur = client.get(f"/api/consultations/{acid}").json()
                    except httpx.HTTPError:
                        # A busy server can drop an idle keep-alive connection
                        # mid-job; that is not a job failure.
                        poll_failures += 1
                        if poll_failures > 15:
                            break
                        continue
                    poll_failures = 0
                    for stage in cur.get("stages", []):
                        seen_stages.add(stage["stage"])
                    if cur["status"] in ("ready", "error"):
                        final = cur
                        break
                if poll_failures:
                    print(f"       (recovered from {poll_failures} transient poll errors)")
                timings["process_s"] = round(time.perf_counter() - t0, 2)

                if final is None:
                    check("processing completes", False, f"timed out after {args.timeout}s")
                elif final["status"] == "error":
                    check("processing completes", False, final.get("error", "?"))
                else:
                    check("processing completes", True, f"{timings['process_s']}s")
                    check("real stage progress reported",
                          {"transcribe", "speakers", "clinical"} <= seen_stages,
                          ",".join(sorted(seen_stages)))
                    result = final["result"]
                    meta = result.get("meta", {})
                    check("transcript produced",
                          len(result["transcript"]["text"]) > 50,
                          f"{len(result['transcript']['text'])} chars, "
                          f"{len(result['transcript']['segments'])} segments")
                    check("STT provider is the production default",
                          result["transcript"]["stt_provider"] == "current_faster_whisper",
                          result["transcript"]["stt_provider"])
                    check("clinical note generated from audio",
                          bool(result["clinical_note"]["fields"]["symptoms_reported"]))
                    check("prescription/plan generated from audio",
                          "medications" in result["prescription"]["fields"])
                    audio_s = meta.get("audio_duration_s")
                    if audio_s:
                        timings["audio_duration_s"] = round(audio_s, 1)
                        timings["realtime_factor"] = round(audio_s / timings["process_s"], 2)
                        print(f"       audio {audio_s:.0f}s processed in "
                              f"{timings['process_s']}s "
                              f"({timings['realtime_factor']}x realtime)")
                    export2 = client.get(f"/api/consultations/{acid}/export")
                    check("audio consultation exports", export2.status_code == 200)
                    check("unreviewed export is flagged",
                          "not yet reviewed" in export2.text.lower())
        else:
            print("\n8. Speech-to-text flow — SKIPPED (pass --audio)")

        # ---- 9. sign out -----------------------------------------------
        print("\n9. Session")
        client.post("/api/logout")
        client.headers.pop("X-Access-Token", None)
        check("sign-out invalidates the session",
              client.get("/api/consultations").status_code == 401)

    # ---- 10. log hygiene ----------------------------------------------
    print("\n10. Log hygiene")
    log_file = Path(os.environ.get("ISCRIBE_LOG_DIR") or (ROOT / "data" / "logs")) / "iscribe.log"
    if not log_file.exists():
        check("application log found", False, str(log_file))
    else:
        contents = log_file.read_text(encoding="utf-8", errors="replace")
        leaked = [c for c in PHI_CANARIES if c.lower() in contents.lower()]
        check("no transcript/clinical content in log", not leaked,
              f"leaked: {leaked}" if leaked else "clean")
        check("no patient identifier in log", "SMOKE-TEST-001" not in contents)
        check("access key not in log", bool(key) and key not in contents)
        check("operational events present", "event=job_completed" in contents
              or "event=text_job_completed" in contents)

    if timings:
        print("\nMeasured timings:")
        for k, v in timings.items():
            print(f"  {k:24} {v}")

    return summarise()


def summarise() -> int:
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [name for name, ok, _ in results if not ok]
    print(f"\n{'=' * 62}")
    print(f"{passed}/{len(results)} checks passed")
    if failed:
        print("\nFAILED:")
        for name in failed:
            print(f"  - {name}")
    print("=" * 62)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
