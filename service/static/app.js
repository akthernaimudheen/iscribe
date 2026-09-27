/* iScribe demo frontend. Vanilla JS, no build step.
   Talks only to the iScribe API. Transcription is handled entirely by the
   service — the product never names or selects speech engines. */

const API = ""; // same origin
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];

const state = {
  consultation: null,
  audioReady: false,
  pollTimer: null,
  recorder: null,
  chunks: [],
  recordSeconds: 0,
  timerInterval: null,
};

/* ---------------- navigation ---------------- */

const STEP_ORDER = ["dashboard", "consult", "audio", "processing", "results", "review", "export"];

function show(view) {
  $$(".view").forEach(v => v.classList.remove("active"));
  $(`#view-${view}`).classList.add("active");
  const idx = STEP_ORDER.indexOf(view);
  $$("#steps .step").forEach((el, i) => {
    el.classList.toggle("active", i === idx);
    el.classList.toggle("done", i < idx);
  });
  window.scrollTo(0, 0);
}

$$("[data-nav]").forEach(el =>
  el.addEventListener("click", () => show(el.dataset.nav))
);

/* ---------------- api helper ---------------- */

async function api(path, options = {}) {
  const res = await fetch(`/api${path}`, options);
  if (res.status === 401) {
    // Session expired or never established — back to the sign-in page.
    window.location.href = "/login";
    throw new Error("Session expired");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  return res.json();
}

/* ---------------- readiness ---------------- */

async function refreshReadiness() {
  const el = $("#ready-chip");
  if (!el) return true;
  try {
    const res = await fetch("/api/ready");
    const info = await res.json();
    if (info.models_ready) {
      el.textContent = "Transcription ready";
      el.className = "ready-chip ok";
      return true;
    }
    if (info.status === "error") {
      el.textContent = "Transcription is unavailable — contact the study team";
      el.className = "ready-chip bad";
    } else {
      el.textContent = "Starting transcription service… audio processing will be available shortly";
      el.className = "ready-chip warn";
      setTimeout(refreshReadiness, 3000);
    }
  } catch {
    el.textContent = "Server unreachable";
    el.className = "ready-chip bad";
  }
  return false;
}

$("#btn-signout")?.addEventListener("click", async () => {
  try { await fetch("/api/logout", { method: "POST" }); } catch {}
  window.location.href = "/login";
});

/* ---------------- dashboard ---------------- */

async function refreshDashboard() {
  try {
    const list = await api("/consultations");
    const el = $("#consult-list");
    if (!list.length) {
      el.innerHTML = `<div class="card hint">No consultations yet. Click “＋ New Consultation” to begin.</div>`;
      return;
    }
    el.innerHTML = list.map(c => `
      <div class="card">
        <h2>${esc(c.patient_id)} <span class="hint">· ${esc(c.consultation_type)}</span></h2>
        <p class="hint">${esc(c.doctor)} · ${esc(c.department)}<br>
           Status: <b>${esc(c.status)}</b> · ${esc(c.created_at)}</p>
        <button class="btn" data-open="${c.id}">Open</button>
      </div>`).join("");
    $$("[data-open]").forEach(b =>
      b.addEventListener("click", () => openConsultation(b.dataset.open))
    );
  } catch (e) {
    $("#consult-list").innerHTML = `<div class="card error-box">API unreachable: ${esc(e.message)}</div>`;
  }
}

async function openConsultation(cid) {
  state.consultation = await api(`/consultations/${cid}`);
  state.audioReady = !!(state.consultation.has_audio || state.consultation.result);
  $("#audio-consult-chip").textContent = chipText();
  $("#results-consult-chip").textContent = chipText();
  if (state.consultation.status === "ready" || state.consultation.status === "completed") {
    renderResults();
    show("results");
  } else if (state.consultation.status === "processing" || state.consultation.status === "queued") {
    show("processing");
    poll();
  } else {
    show("audio");
  }
}

function chipText() {
  const c = state.consultation;
  return c ? `${c.patient_id} · ${c.doctor}` : "";
}

/* ---------------- new consultation ---------------- */

$("#btn-new").addEventListener("click", () => show("consult"));

// Async documentation: from the processing screen the doctor immediately
// moves on — the background worker finishes this consultation's note.
$("#btn-next-patient")?.addEventListener("click", () => {
  stopPolling();
  show("consult");
});

$("#f-language")?.addEventListener("change", (e) => {
  const warn = $("#lang-warning");
  if (warn) warn.style.display = e.target.value === "ml" ? "block" : "none";
});

$("#btn-create").addEventListener("click", async () => {
  const body = {
    patient_id: $("#f-patient").value.trim() || "TRIAL-001",
    doctor: $("#f-doctor").value.trim() || "Dr.",
    department: $("#f-dept").value,
    consultation_type: $("#f-type").value,
    language: $("#f-language") ? $("#f-language").value : "en",
  };
  state.consultation = await api("/consultations", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  state.audioReady = false;
  $("#audio-consult-chip").textContent = chipText();
  $("#results-consult-chip").textContent = chipText();
  resetAudioCards();
  show("audio");
});

/* ---------------- audio: recording ---------------- */

function resetAudioCards() {
  state.audioReady = false;
  $("#rec-result").classList.add("hidden");
  $("#upload-result").classList.add("hidden");
  $("#btn-process").disabled = true;
  $("#rec-timer").textContent = "00:00:00";
  $("#rec-status").textContent = "Microphone idle";
  $("#file-input").value = "";
}

const fmt = (s) => {
  const h = String(Math.floor(s / 3600)).padStart(2, "0");
  const m = String(Math.floor((s % 3600) / 60)).padStart(2, "0");
  const sec = String(s % 60).padStart(2, "0");
  return `${h}:${m}:${sec}`;
};

$("#rec-start").addEventListener("click", async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.chunks = [];
    state.recordSeconds = 0;
    state.recorder = new MediaRecorder(stream);
    state.recorder.ondataavailable = (e) => e.data.size && state.chunks.push(e.data);
    state.recorder.onstop = onRecordingStopped;
    state.recorder.start();
    $("#rec-status").textContent = "Recording…";
    setRecButtons("recording");
    state.timerInterval = setInterval(() => {
      state.recordSeconds++;
      $("#rec-timer").textContent = fmt(state.recordSeconds);
    }, 1000);
  } catch (e) {
    $("#rec-status").textContent = `Microphone unavailable: ${e.message}`;
  }
});

$("#rec-pause").addEventListener("click", () => {
  state.recorder?.pause();
  clearInterval(state.timerInterval);
  $("#rec-status").textContent = "Paused";
  setRecButtons("paused");
});

$("#rec-resume").addEventListener("click", () => {
  state.recorder?.resume();
  state.timerInterval = setInterval(() => {
    state.recordSeconds++;
    $("#rec-timer").textContent = fmt(state.recordSeconds);
  }, 1000);
  $("#rec-status").textContent = "Recording…";
  setRecButtons("recording");
});

$("#rec-stop").addEventListener("click", () => {
  clearInterval(state.timerInterval);
  state.recorder?.stop();
});

function setRecButtons(mode) {
  $("#rec-start").disabled = mode !== "idle" && mode !== "stopped";
  $("#rec-pause").disabled = mode !== "recording";
  $("#rec-resume").disabled = mode !== "paused";
  $("#rec-stop").disabled = mode === "idle" || mode === "stopped" || mode === "paused";
}

async function onRecordingStopped() {
  state.recorder.stream.getTracks().forEach(t => t.stop());
  setRecButtons("stopped");
  const blob = new Blob(state.chunks, { type: state.recorder.mimeType || "audio/webm" });
  await uploadAudioBlob(blob, `recording_${Date.now()}.webm`);
  $("#rec-result").classList.remove("hidden");
}

async function uploadAudioBlob(blob, filename) {
  const fd = new FormData();
  fd.append("file", blob, filename);
  state.consultation = await api(`/consultations/${state.consultation.id}/audio`, {
    method: "POST",
    body: fd,
  });
  state.audioReady = true;
  $("#btn-process").disabled = false;
}

/* ---------------- audio: upload ---------------- */

$("#file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("#rec-status").textContent = "Uploading…";
  try {
    await uploadAudioBlob(file, file.name);
    $("#upload-name").textContent = file.name;
    $("#upload-result").classList.remove("hidden");
    $("#rec-status").textContent = "Microphone idle";
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
  }
});

/* ---------------- audio: demo transcript ---------------- */

const MALAYALAM_SAMPLE =
  "ഡോക്ടറെ, എനിക്ക് മൂന്ന് ദിവസമായി പനി ഉണ്ട്. ചുമയും കഫവും ഉണ്ട്. " +
  "തലവേദനയും ഉണ്ട്. പക്ഷേ ശ്വാസം മുട്ടൽ ഇല്ല.";

$("#btn-demo-text").addEventListener("click", async () => {
  const demo = await api("/demo/transcript");
  $("#paste-transcript").value = demo.transcript;
});

$("#btn-demo-ml")?.addEventListener("click", () => {
  $("#paste-transcript").value = MALAYALAM_SAMPLE;
});

$("#btn-process-text").addEventListener("click", async () => {
  const transcript = $("#paste-transcript").value.trim();
  const box = $("#text-error");
  box.classList.add("hidden");
  if (!transcript) {
    box.textContent = "Paste a transcript first.";
    box.classList.remove("hidden");
    return;
  }
  try {
    state.consultation = await api(`/consultations/${state.consultation.id}/text`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript }),
    });
  } catch (e) {
    box.textContent = e.message;
    box.classList.remove("hidden");
    return;
  }
  state.audioReady = true;
  $("#btn-process").disabled = true; // already processed
  renderResults();
  show("results");
});

/* ---------------- processing ---------------- */

$("#btn-process").addEventListener("click", async () => {
  $("#process-error").classList.add("hidden");
  try {
    state.consultation = await api(`/consultations/${state.consultation.id}/process`, { method: "POST" });
  } catch (e) {
    const box = $("#process-error");
    box.textContent = e.message;
    box.classList.remove("hidden");
    show("processing");
    return;
  }
  show("processing");
  poll();
});

const STAGE_LABELS = {
  prepare: "Preparing audio",
  transcribe: "Transcribing",
  speakers: "Identifying speakers",
  clinical: "Generating clinical note & prescription",
};

function renderStages(stages, activeStage, status) {
  for (const key of Object.keys(STAGE_LABELS)) {
    const li = $(`#stage-list li[data-stage="${key}"]`);
    const done = stages.some(s => s.stage === key && s.status === "done");
    const running = (activeStage === key) || stages.some(s => s.stage === key && s.status === "running");
    li.className = done ? "done" : (running ? "running" : "");
    li.querySelector(".dot").textContent = done ? "✓" : (running ? "●" : "○");
  }
}

// Transcription saturates the CPU, so a poll can time out or lose its
// connection while the job is running perfectly well. Tolerate a run of
// failures before telling the clinician anything is wrong — giving up on the
// first blip would report a successful transcription as a failure.
const POLL_INTERVAL_MS = 1000;
const POLL_MAX_CONSECUTIVE_FAILURES = 15;

function stopPolling() {
  clearInterval(state.pollTimer);
  state.pollTimer = null;
}

function poll() {
  clearInterval(state.pollTimer);
  let failures = 0;
  state.pollTimer = setInterval(async () => {
    try {
      const c = await api(`/consultations/${state.consultation.id}`);
      failures = 0;
      state.consultation = c;
      renderStages(c.stages || [], c.active_stage, c.status);
      if (c.status === "ready") {
        clearInterval(state.pollTimer);
        renderResults();
        show("results");
      } else if (c.status === "error") {
        clearInterval(state.pollTimer);
        const box = $("#process-error");
        box.textContent = c.error || "Processing failed";
        box.classList.remove("hidden");
        show("processing");
      }
    } catch (e) {
      failures++;
      if (failures >= POLL_MAX_CONSECUTIVE_FAILURES) {
        clearInterval(state.pollTimer);
        const box = $("#process-error");
        box.textContent =
          `Lost contact with the server while processing (${e.message}). ` +
          `The job may still have finished — reopen this consultation from the dashboard to check.`;
        box.classList.remove("hidden");
        show("processing");
      }
    }
  }, POLL_INTERVAL_MS);
}

/* ---------------- results / review / export ---------------- */

function renderResults() {
  const c = state.consultation;
  const r = c.result;
  if (!r) return;

  const m = r.meta || {};
  const language = r.transcript.language || "en";
  const languageLabel = language === "ml" ? "Malayalam" : (language === "en" ? "English" : language);
  $("#results-meta").innerHTML =
    `Language: <b>${esc(languageLabel)}</b>` +
    (r.speakers.roles_known === false
      ? ` <span class="warn-inline">Speaker roles not identified — verify who said what</span>`
      : "") +
    (m.elapsed_s ? ` · ${esc(m.elapsed_s)}s` : "");

  // Transcript quality banner. Shown before the transcript so a clinician sees
  // the warning before reading content that may be wrong.
  const vb = $("#validation-banner");
  const v = r.validation;
  if (vb) {
    if (v && v.severity && v.severity !== "ok") {
      vb.textContent = v.message || "Transcription quality could not be verified.";
      vb.className = v.severity === "invalid" ? "validation-banner bad" : "validation-banner warn";
      vb.classList.remove("hidden");
    } else {
      vb.classList.add("hidden");
    }
  }

  // Speaker turns. Anonymous diarization labels ("Speaker 0") carry no
  // identity, so the transcript presents them as a separated meta label
  // ("Speaker 1") — clearly apart from the spoken text, never merged into it.
  // Roles are never fabricated: a stored transcript keeps exactly the roles it
  // was recorded with.
  const ANON_SPEAKER_RE = /^Speaker\s*\d+$/i;
  $("#transcript-view").innerHTML = (r.speakers.turns || [])
    .map(t => {
      const isAnon = ANON_SPEAKER_RE.test(t.speaker || "");
      const label = isAnon ? t.speaker.replace(/\s+/g, " ") : t.speaker;
      const cls = isAnon ? "speaker-anon"
        : `speaker-${esc(t.speaker.toLowerCase().replace(/[^a-z0-9]/g, "-"))}`;
      return `<div class="turn"><span class="${cls}">${esc(label)}:</span> ${esc(t.text)}</div>`;
    })
    .join("") || `<div class="hint">(no speech recognised)</div>`;

  renderNormalized(r.normalized_clinical_entities);
  renderNoteV2(r.clinical_note_v2);
  // clinical_note_v2 is the ONE authoritative clinical note. The legacy
  // clinical_note / prescription extractions stay in the payload for API
  // compatibility but are shown only inside a clearly-labelled debug panel,
  // never as a competing note.
  const legacyPanel = $("#legacy-extraction-panel");
  if (legacyPanel) {
    legacyPanel.classList.remove("hidden");
    renderFields($("#note-fields"), r.clinical_note.fields, false);
    renderFields($("#rx-fields"), r.prescription.fields, false);
  }
  renderReview();
  $("#export-preview").textContent = "Loading…";
  loadExportPreview();
  renderRetentionStatus("#export-retention-status");
}

/* Render the validated deterministic clinical note (clinical_note_v2).
   Read-only: the doctor-facing edit/review flow continues to use the classic
   field template until the v2 sections are wired into review/export. */
function renderNoteV2(v2) {
  const panel = $("#note-v2-panel");
  if (!panel) return;
  if (!v2 || !v2.note) { panel.classList.add("hidden"); return; }
  const badge = (v2.validation && v2.validation.valid === true)
    ? `<span class="norm-chip st-present">Fact graph validated</span>`
    : `<span class="norm-chip st-absent">Validation failed</span>`;
  panel.innerHTML =
    `<h2>Clinical Note ${badge}</h2>` +
    `<p class="hint">Rendered deterministically from the evidence-grounded fact
      graph. Every line traces to the transcript — see provenance below the
      note.</p>` +
    `<div class="note-v2">${esc(v2.note)}</div>`;
  panel.classList.remove("hidden");
}

function renderNormalized(n) {
  const box = $("#normalized-panel");
  if (!box) return;
  if (!n || !(n.entities || []).length) { box.classList.add("hidden"); return; }
  const chip = (label, items, cls) =>
    items && items.length
      ? `<div class="norm-row"><span class="norm-label ${cls}">${label}</span>` +
        items.map(i => `<span class="norm-chip ${cls}">${esc(i)}</span>`).join("") + `</div>`
      : "";
  const listChip = (label, items, cls) =>
    items && items.length ? chip(label, items, cls) : "";
  const rows = (n.entities || []).map(e => `
    <tr>
      <td data-label="Concept">${esc(e.english)}</td>
      <td data-label="Status"><span class="norm-chip st-${esc(e.status.toLowerCase())}">${esc(e.status)}</span></td>
      <td data-label="When">${esc(e.temporality || "-")}</td>
      <td data-label="Duration">${esc(e.duration || "-")}</td>
      <td data-label="Severity">${esc(e.severity || "-")}</td>
      <td class="norm-src" data-label="Said as">${esc(e.surface_text)}</td>
      <td data-label="Conf.">${esc(e.confidence)}</td>
    </tr>`).join("");
  box.innerHTML =
    `<h2>Clinical concepts identified</h2>` +
    chip("Reported", n.present, "st-present") +
    chip("Denied", n.absent, "st-absent") +
    chip("Resolved", n.resolved, "st-resolved") +
    chip("Asked about", n.questioned, "st-question") +
    chip("Uncertain", n.uncertain, "st-possible") +
    listChip("Family history", n.family_history, "st-family") +
    listChip("Medications mentioned", n.medications_discussed, "st-med") +
    listChip("If-then only (not findings)", n.conditional, "st-conditional") +
    (n.normalized_summary
      ? `<p class="norm-summary"><b>Normalized restatement</b> — this is a structured
         summary, <b>not</b> the transcript: ${esc(n.normalized_summary)}</p>` : "") +
    `<table class="norm-table"><thead><tr>
       <th>Concept</th><th>Status</th><th>When</th><th>Duration</th>
       <th>Severity</th><th>Said as</th><th>Conf.</th></tr></thead>
     <tbody>${rows}</tbody></table>`;
  box.classList.remove("hidden");
}

function renderFields(container, fields, editable) {
  container.innerHTML = "";
  for (const [k, v] of Object.entries(fields)) {
    const div = document.createElement("div");
    div.className = "field";
    const label = document.createElement("span");
    label.className = "field-key";
    label.textContent = k.replace(/_/g, " ");
    div.appendChild(label);

    const isTemplate = typeof v === "string" && v.includes("[template]");
    if (editable) {
      const ta = document.createElement("textarea");
      ta.dataset.key = k;
      ta.value = typeof v === "string" ? v : JSON.stringify(v);
      div.appendChild(ta);
    } else {
      const span = document.createElement("span");
      if (isTemplate) span.className = "template";
      span.textContent = Array.isArray(v)
        ? (v.length ? v.join(", ") : "—")
        : (typeof v === "object" ? JSON.stringify(v) : String(v ?? "—"));
      div.appendChild(span);
    }
    container.appendChild(div);
  }
}

function renderReview() {
  const r = state.consultation.result;
  // The review screen edits the legacy template fields (backward-compatible
  // review/export path). The validated fact-graph note above is read-only:
  // editing it would mean editing clinical text without the evidence checks.
  renderFields($("#review-note-fields"), r.clinical_note.fields, true);
  renderFields($("#review-rx-fields"), r.prescription.fields, true);
  renderRetentionStatus("#review-retention-status");
}

$("#btn-to-review").addEventListener("click", () => show("review"));

$("#btn-save-review").addEventListener("click", async () => {
  const collect = (container) => {
    const out = {};
    $$(`${container} textarea`).forEach(ta => {
      try { out[ta.dataset.key] = JSON.parse(ta.value); }
      catch { out[ta.dataset.key] = ta.value; }
    });
    return out;
  };
  const body = {
    clinical_note_fields: collect("#review-note-fields"),
    prescription_fields: collect("#review-rx-fields"),
  };
  state.consultation = await api(`/consultations/${state.consultation.id}/review`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  renderResults();
  show("results");
});

/* ---------------- audio retention status ---------------- */

function retentionStatusText(c) {
  const s = c.audio_state;
  if (c.audio_purged) {
    return c.training_record_id
      ? ("Audio retained only in the hospital's authorized training dataset " +
         "(independently scheduled for deletion).")
      : ("Audio has been deleted according to the retention policy. The " +
         "approved note remains available for export.");
  }
  if (s === "SCHEDULED_FOR_DELETION") {
    const due = c.audio_deletion_due_ts ? new Date(c.audio_deletion_due_ts * 1000) : null;
    if (c.approved_at) {
      return due
        ? (`Note approved. Audio will be deleted around ${due.toLocaleString()} ` +
           "according to the hospital retention policy.")
        : "Note approved. Audio will be deleted according to the hospital " +
          "retention policy.";
    }
    return due
      ? (`Audio is available for review until ${due.toLocaleString()} per the ` +
         "hospital retention policy. Approving the note starts the " +
         "post-approval deletion window.")
      : ("Audio is available for the review window per the hospital " +
         "retention policy. Approving the note starts the post-approval " +
         "deletion window.");
  }
  if (s === "DOCTOR_APPROVED" || (c.note_status === "APPROVED" && s !== "SCHEDULED_FOR_DELETION"))
    return "Note approved. Audio will be deleted according to the clinic " +
      "retention policy.";
  if (s === "DOCTOR_APPROVED") return "Note approved. Audio deletion is being scheduled.";
  if (s === "AWAITING_DOCTOR_REVIEW")
    return "Audio is still available — the note is awaiting doctor review. " +
      "It will be deleted after approval (or at the policy deadline if the " +
      "note is never approved).";
  return "Audio is available for this consultation's review window.";
}

async function renderRetentionStatus(rootId) {
  const el = $(rootId);
  if (!el) return;
  const c = state.consultation;
  if (!c) { el.textContent = ""; return; }
  el.textContent = retentionStatusText(c);
  const box = $("#review-audio-box");
  const player = $("#review-audio-player");
  if (rootId !== "#review-retention-status" || !box || !player) return;
  if (c.audio_purged || !c.has_audio) { box.classList.add("hidden"); return; }
  try {
    // The audio URL is short-lived and bound to this consultation; it is
    // requested only when the doctor is on the review screen.
    const link = await api(`/consultations/${c.id}/audio-link`, { method: "POST" });
    player.src = link.url;
    box.classList.remove("hidden");
  } catch {
    box.classList.add("hidden");
  }
}

$("#btn-complete").addEventListener("click", async () => {
  const errBox = $("#review-error");
  if (errBox) { errBox.classList.add("hidden"); errBox.textContent = ""; }
  try {
    state.consultation = await api(`/consultations/${state.consultation.id}/complete`, { method: "POST" });
    $("#export-banner").classList.remove("hidden");
    show("export");
    refreshDashboard();
  } catch (e) {
    // e.g. "Review (and edit) the note before completing it" — show it where
    // the doctor is looking instead of failing silently.
    if (errBox) { errBox.textContent = e.message; errBox.classList.remove("hidden"); }
    else throw e;
  }
});

async function loadExportPreview() {
  const res = await fetch(`/api/consultations/${state.consultation.id}/export`);
  $("#export-preview").textContent = await res.text();
}

$("#btn-download").addEventListener("click", () => {
  // set href at click-time so it always points at the latest consultation
  $("#btn-download").href = `/api/consultations/${state.consultation.id}/export`;
});

$("#btn-print").addEventListener("click", () => {
  const w = window.open("", "_blank");
  w.document.write(`<pre>${esc($("#export-preview").textContent)}</pre>`);
  w.print();
});

/* ---------------- STT privacy boundary (dashboard) ---------------- */

async function loadBoundaryPanel() {
  const panel = $("#stt-boundary-panel");
  const body = $("#stt-boundary-body");
  if (!panel || !body) return;
  try {
    const b = await api("/stt/boundary");
    $("#stt-boundary-rows").innerHTML = boundaryRows(b).join("");
  } catch (e) {
    // Unauthenticated or unavailable: say nothing more than that. No error
    // text can carry internal detail, so the panel simply hides.
    panel.style.display = "none";
  }
}

function boundaryRows(b) {
  const yesNo = v => (v ? "Enabled" : "Disabled");
  const status = (b && b.provider && b.status === "configured") ? "ALLOWED" : "BLOCKED";
  return [
    ["STT Provider", "Provider", b.provider_name || b.provider || "not configured"],
    ["STT Provider", "Model", b.model || "not reported"],
    ["Processing Location", "Region", b.data_location || b.region || "unknown"],
    ["Processing Location", "Endpoint", b.endpoint_class || b.endpoint || "unknown"],
    ["Processing Location", "Data location", b.data_location || "unknown"],
    ["Privacy", "Zero retention", yesNo(b.zero_retention)],
    ["Privacy", "Off-host processing", b.off_host ? "Allowed" : "Blocked"],
    ["Privacy", "Automatic fallback", yesNo(!!(b.policy && b.policy.automatic_fallback_allowed))],
    ["Policy", "Policy status", status],
    ["Policy", "Fail-closed", yesNo(b.fail_closed)],
    ["Policy", "Policy enforcement", yesNo(b.policy_enforced)],
  ].map(([section, label, value]) =>
    `<tr data-section="${esc(section)}"><th scope="row">${esc(label)}</th><td>${esc(value)}</td></tr>`);
}

/* ---------------- utils ---------------- */

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

refreshDashboard();
refreshReadiness();
loadBoundaryPanel();
