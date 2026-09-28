"""Integration regression: Review Consultation flow on projected fields.

Pins the full clinician loop against the single-source (fact-graph) fields:

    create -> process (text flow) -> structured fields populated
    -> edit a field -> save -> reload shows the edit
    -> complete -> export carries the reviewed value

and the authorization boundary: the break-glass shared-key session stays
clinically read-only at every step of this loop.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.test_deployment import ACCESS_TOKEN, app_module  # noqa: F401


REFERRAL_TRANSCRIPT = """Doctor: This is a referral letter. Re: Mr A Sample.
Doctor: Thank you for referring this gentleman regarding pain in the left medial foot arch.
Doctor: The pain began after buying tight footwear and has been fluctuating since March.
Doctor: No benefit was obtained from physiotherapy. The pain worsens after resuming exercise.
Doctor: On examination there is no bony tenderness. There is tenderness over the left medial arch.
Doctor: I suspect plantar fasciitis.
Doctor: A steroid injection would benefit from consideration.
Doctor: I would be grateful if you could refer him to a consultant orthopaedic surgeon for a specialist opinion.
Doctor: Solicitors reference: HW / CE. Date of birth: 06/22/1972. Telephone: 07700900123."""


def _shared_key_client(app_module) -> TestClient:
    c = TestClient(app_module.app)
    login = c.post("/api/login", data={"access_token": ACCESS_TOKEN},
                   follow_redirects=False)
    assert login.status_code == 303
    return c


def test_review_flow_end_to_end_on_projected_fields(app_module):
    with TestClient(app_module.app) as c:
        from service.security import issue_user_session
        c.cookies.set(app_module.SESSION_COOKIE, issue_user_session(
            app_module.settings.access_token, "doctor@trial.test", "DOCTOR",
            app_module.settings.hospital_id, 3600))

        # 1. create + process (text flow)
        cid = c.post("/api/consultations",
                     json={"patient_id": "PROJ-001"}).json()["id"]
        processed = c.post(f"/api/consultations/{cid}/text",
                           json={"transcript": REFERRAL_TRANSCRIPT}).json()
        assert processed["status"] == "ready"

        # 2. structured fields populated from the fact graph
        note = processed["result"]["clinical_note"]
        assert note.get("fields_source") == "clinical_facts_v2"
        fields = note["fields"]
        assert "left medial foot arch" in fields["symptoms_reported"].lower()
        assert "tight footwear" in fields["onset_trigger"].lower()
        assert fields["duration"].lower() != "buying tight"
        assert "no bony tenderness" in fields["physical_findings"].lower()
        assert fields["diagnoses"], "suspected diagnosis must populate diagnoses"
        assert fields["diagnoses"][0]["status"] == "suspected"
        assert not any("no diagnosis" in w.lower() for w in note["warnings"])

        # 3. clinician edits a field and saves
        edited_diagnoses = [{"name": "plantar fasciitis",
                             "status": "confirmed", "confirmed": True}]
        reviewed = c.patch(
            f"/api/consultations/{cid}/review",
            json={"clinical_note_fields": {
                "impression": "Plantar fasciitis (clinician confirmed on review)",
                "diagnoses": edited_diagnoses,
            }},
        ).json()
        assert reviewed["reviewed"] is True
        assert reviewed["result"]["clinical_note"]["fields"][
            "impression"] == "Plantar fasciitis (clinician confirmed on review)"
        # The saved text block was re-rendered from the edited fields.
        assert "clinician confirmed" in reviewed["result"]["clinical_note"]["text"]

        # 4. reopening review shows the saved edit
        reopened = c.get(f"/api/consultations/{cid}").json()
        assert reopened["result"]["clinical_note"]["fields"][
            "impression"] == "Plantar fasciitis (clinician confirmed on review)"

        # 5. complete preserves the reviewed values
        completed = c.post(f"/api/consultations/{cid}/complete").json()
        assert completed["status"] == "completed"

        # 6. export carries the clinician-reviewed value
        export = c.get(f"/api/consultations/{cid}/export")
        assert export.status_code == 200
        assert "Plantar fasciitis (clinician confirmed on review)" in export.text
        # The export also still carries the projected clinical facts the
        # doctor did not change (nothing silently dropped).
        assert "no bony tenderness" in export.text.lower()
        assert "orthopaedic surgeon" in export.text.lower()


def test_shared_key_cannot_review_or_sign_off(app_module):
    # One client for the whole test: the doctor sets up the consultation,
    # then the SAME client swaps to the break-glass shared-key session.
    # (A second TestClient would outlive the first lifespan and hit a closed
    # SQLite connection at teardown.)
    with TestClient(app_module.app) as c:
        from service.security import issue_user_session
        c.cookies.set(app_module.SESSION_COOKIE, issue_user_session(
            app_module.settings.access_token, "doctor@trial.test", "DOCTOR",
            app_module.settings.hospital_id, 3600))
        cid = c.post("/api/consultations", json={}).json()["id"]
        c.post(f"/api/consultations/{cid}/text",
               json={"transcript": REFERRAL_TRANSCRIPT})

        # Swap the session cookie to the break-glass shared key.
        login = c.post("/api/login", data={"access_token": ACCESS_TOKEN},
                       follow_redirects=False)
        assert login.status_code == 303

        # The shared key can look, but never touch clinical sign-off:
        assert c.patch(
            f"/api/consultations/{cid}/review",
            json={"clinical_note_fields": {"impression": "tamper"}},
        ).status_code in (401, 403)
        assert c.post(f"/api/consultations/{cid}/complete").status_code in (401, 403)
        assert c.get(f"/api/consultations/{cid}/export").status_code in (401, 403)


def test_complete_refuses_unreviewed_projection_fields(app_module):
    """Approve still requires an explicit review step — projection or not."""
    with TestClient(app_module.app) as c:
        from service.security import issue_user_session
        c.cookies.set(app_module.SESSION_COOKIE, issue_user_session(
            app_module.settings.access_token, "doctor@trial.test", "DOCTOR",
            app_module.settings.hospital_id, 3600))
        cid = c.post("/api/consultations", json={}).json()["id"]
        c.post(f"/api/consultations/{cid}/text",
               json={"transcript": REFERRAL_TRANSCRIPT})
        # No PATCH /review first: complete must refuse (409) — the fields being
        # well-populated must never let the clinician skip sign-off.
        resp = c.post(f"/api/consultations/{cid}/complete")
        assert resp.status_code in (409, 400)
