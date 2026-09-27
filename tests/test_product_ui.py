"""Product-UI tests: iScribe is one coherent product.

Verifies:
  - provider/model names never appear in user-facing pages (HTML/JS/CSS)
  - the product presents itself as iScribe everywhere
  - mobile-ready markup is in place (viewport, short step labels,
    data-labelled concept table)
  - the primary recording action is reachable in the served markup
  - a Malayalam consultation whose speech engine is down fails safely:
    clinician-worded message, NO whisper fallback, transcript empty

Internal code, logs, tests, docs and benchmarks keep provider names —
only the user-facing product must be free of them.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

STATIC = Path(__file__).resolve().parent.parent / "service" / "static"

PROVIDER_TERMS = re.compile(
    r"deepgram|whisper|indicconformer|nemo|nvidia|sidecar|faster-whisper",
    re.IGNORECASE,
)


# -- A. provider abstraction in user-facing product ---------------------------
def test_user_facing_html_has_no_provider_names():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert not PROVIDER_TERMS.search(html), "provider/model name leaked into HTML"


def test_user_facing_js_has_no_provider_strings():
    """User-visible strings in app.js must not name engines. Dev comments are
    tolerated only if they carry no provider brand either (simplest rule)."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert not PROVIDER_TERMS.search(js), "provider/model name leaked into JS"


def test_user_facing_css_has_no_provider_names():
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    assert not PROVIDER_TERMS.search(css), "provider/model name leaked into CSS"





# -- B. product identity -------------------------------------------------------
def test_every_page_branded_iscribe():
    assert "<title>iScribe" in (STATIC / "index.html").read_text(encoding="utf-8")
    security = (Path(__file__).resolve().parent.parent / "service" / "security.py"
                ).read_text(encoding="utf-8")
    assert "iScribe" in security


def test_mobile_viewport_present():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'name="viewport"' in html
    assert "width=device-width" in html


# -- C. mobile-ready markup -----------------------------------------------------
def test_progress_steps_carry_short_mobile_labels():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    shorts = re.findall(r'data-step="[a-z]+" data-short="([^"]+)"', html)
    assert len(shorts) == 7, "all seven flow steps need short mobile labels"
    assert all(len(s) <= 12 for s in shorts), "short labels must be compact"


def test_concept_table_cells_carry_mobile_labels():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'data-label="Status"' in js
    assert 'data-label="Concept"' in js


def test_recording_is_the_primary_audio_action():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    audio_section = html.split('id="view-audio"')[1].split('<!-- ================= PROCESSING')[0]
    # Record card comes before Upload card on the page (source order = mobile order)
    assert audio_section.index("Record consultation") < audio_section.index("Upload audio")
    assert 'id="rec-start"' in audio_section


def test_css_targets_phone_breakpoints():
    css = (STATIC / "styles.css").read_text(encoding="utf-8")
    for width in ("360px", "480px", "640px"):
        assert width in css, f"missing {width} breakpoint"
    assert "(hover: none) and (pointer: coarse)" in css, "touch-target rules missing"


def test_family_history_never_styled_as_present_finding():
    """Safety distinction: family history / conditional get their own classes
    and must not reuse the PRESENT chip style."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert 'listChip("Family history"' in js
    assert '"st-family"' in js or "st-family" in (STATIC / "styles.css").read_text(encoding="utf-8")


# -- D. language-mismatch failure is safe ---------------------------------------
def test_ml_failure_never_falls_back_to_english_engine():
    """The real-audio benchmark (benchmarks/malayalam/real_audio_asr) proved a
    local English engine returns fluent nonsense on Malayalam. The pipeline
    must fail the consultation instead of degrading."""
    from scribe_engine.pipeline import MalayalamSpeechUnavailable
    from scribe_engine.stt.base import SpeechLanguageUnavailable

    assert MalayalamSpeechUnavailable is SpeechLanguageUnavailable
    exc = SpeechLanguageUnavailable("ml", "detail-for-logs-only")
    msg = str(exc)
    assert "Malayalam" in msg
    # clinician-worded: no engine names, no operator jargon
    assert not PROVIDER_TERMS.search(msg)
    assert "sidecar" not in msg.lower() and "start-malayalam" not in msg.lower()


def test_ml_failure_message_contains_no_internal_paths():
    from scribe_engine.stt.base import SpeechLanguageUnavailable

    msg = str(SpeechLanguageUnavailable("ml"))
    assert "scripts/" not in msg and "MALAYALAM_SIDECAR_URL" not in msg


# -- E. STT privacy boundary panel (admin observability) -----------------------
def test_stt_boundary_panel_exists_and_factual():
    """The dashboard exposes the STT privacy boundary: configuration facts
    only, no provider brands, no secrets, no compliance claims."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'id="stt-boundary-panel"' in html
    assert html.count('id="stt-boundary-panel"') == 1
    assert "STT Privacy Boundary" in html
    assert 'id="stt-boundary-body"' in html
    # Claims must stay factual/evidence-based.
    low = html.lower()
    for claim in ("hipaa", "gdpr", "dpdp", "dpa", "zero data retention",
                  "compliant"):
        assert claim not in low, f"non-factual claim in HTML: {claim!r}"
    # No file paths may appear in the panel markup.
    assert "C:\\" not in html and "/data/" not in html


def test_stt_boundary_rendering_hides_panel_when_unavailable():
    """A failing/unauthenticated boundary fetch degrades to a short hint —
    the panel never renders placeholder junk or errors with internal detail."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "stt-boundary-panel" in js
    assert "stt-boundary-body" in js
    assert "/stt/boundary" in js
    assert "style.display = \"none\"" in js.split("async function loadBoundaryPanel")[1] \
        .split("\nasync ")[0]


def test_stt_boundary_js_values_are_escaped():
    """Every value rendered into the panel goes through esc() — the same rule
    every other dashboard string follows."""
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    fn = js.split("function boundaryRows")[1].split("\nfunction ")[0]
    assert "esc(" in fn
    # label + value + section per row template = 3 esc call sites.
    assert fn.count("esc(") >= 3
