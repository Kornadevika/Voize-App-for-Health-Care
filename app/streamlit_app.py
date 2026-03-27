"""
app/streamlit_app.py

Streamlit UI — calls FastAPI for all predictions.
Run: streamlit run app/streamlit_app.py
"""

import json
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="voize | Healthcare Pipeline",
    page_icon="🏥",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #0a0e1a; color: #e2e8f0; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #1e2740; }

.card        { background:#111827; border:1px solid #1e2740; border-radius:12px; padding:20px 24px; margin-bottom:14px; }
.card-green  { border-color:#10b981; box-shadow:0 0 0 1px #10b98120; }
.card-blue   { border-color:#3b82f6; box-shadow:0 0 0 1px #3b82f620; }

.lbl  { font-family:'IBM Plex Mono',monospace; font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#64748b; }
.ttl  { font-size:17px; font-weight:600; color:#f1f5f9; margin:4px 0 12px; }

.tag         { display:inline-block; padding:2px 8px; border-radius:4px; font-size:12px; font-weight:600; margin:2px; font-family:'IBM Plex Mono',monospace; }
.PATIENT     { background:#1e3a5f; color:#60a5fa; }
.SYMPTOM     { background:#3b1f1f; color:#f87171; }
.MEDICATION  { background:#1a2f1a; color:#4ade80; }
.VITAL       { background:#2d1b4e; color:#c084fc; }
.VALUE       { background:#1e3040; color:#38bdf8; }
.DATE        { background:#2a2010; color:#fbbf24; }
.PROCEDURE   { background:#1a2a2a; color:#2dd4bf; }

.tx-box      { background:#0d1117; border:1px solid #1e2740; border-radius:8px; padding:14px 18px; font-size:15px; line-height:1.9; color:#e2e8f0; }
.f-row       { display:flex; justify-content:space-between; padding:9px 0; border-bottom:1px solid #1e274025; }
.f-key       { font-size:12px; color:#64748b; font-family:'IBM Plex Mono',monospace; }
.f-val       { font-size:14px; font-weight:500; }
.m-box       { background:#111827; border:1px solid #1e2740; border-radius:8px; padding:16px; text-align:center; }
.m-val       { font-family:'IBM Plex Mono',monospace; font-size:26px; font-weight:600; }
.m-lbl       { font-size:11px; color:#64748b; text-transform:uppercase; letter-spacing:1px; }
.sec-hdr     { color:#64748b; font-size:10px; margin:10px 0 3px; font-family:'IBM Plex Mono',monospace; letter-spacing:2px; text-transform:uppercase; }

.stButton>button    { background:#2563eb!important; color:white!important; border:none!important; border-radius:8px!important; font-weight:600!important; width:100%; }
.stTextArea textarea{ background:#0d1117!important; border:1px solid #1e2740!important; color:#e2e8f0!important; }
hr { border-color:#1e2740!important; }
</style>
""", unsafe_allow_html=True)

ICONS = {
    "PATIENT":   "👤",
    "SYMPTOM":   "🤒",
    "MEDICATION":"💊",
    "VITAL":     "📊",
    "VALUE":     "🔢",
    "DATE":      "🕐",
    "PROCEDURE": "🩺",
}

EXAMPLES = {
    "🤒 Fever + Refused meds":
        "Patient Müller had a fever of 38.5 degrees this morning and refused ibuprofen 400mg. Complained of knee pain. Blood pressure 130 over 85.",
    "🩺 Post-procedure check":
        "Wound dressing changed on Mrs Schmidt's left leg this evening. Temperature 37.2 degrees, pulse 68 bpm. Paracetamol 500mg administered.",
    "📊 Vitals and glucose":
        "Mr Weber morning vitals: blood pressure 145 over 90. Blood glucose 8.4 mmol. Metformin administered after breakfast. Temperature 36.8 degrees.",
    "💊 Multiple medications":
        "Patient Hoffmann refused ramipril and aspirin this morning. Nausea and dizziness reported. Oxygen saturation 97 percent.",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 voize Pipeline")
    st.markdown(
        '<div style="color:#64748b;font-size:13px;line-height:1.7">'
        "NER model loaded from pkl file.<br>"
        "Audio handled by Whisper-small.<br>"
        "DVC tracks model versions."
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # API health
    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        if h.get("status") == "healthy":
            st.markdown(
                '<div style="color:#10b981;font-size:13px;margin-bottom:8px">● API connected</div>',
                unsafe_allow_html=True,
            )
            for k, v in [
                ("NER",    f"pkl ({h.get('pkl_size_mb','??')} MB)"),
                ("ASR",    h.get("asr", "whisper-small")),
                ("Device", h.get("device", "cpu")),
            ]:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 0;font-size:12px;">'
                    f'<span style="color:#64748b">{k}</span>'
                    f'<span style="color:#10b981;font-family:IBM Plex Mono,monospace">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="color:#fbbf24;font-size:13px">⚠ API loading...</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        st.markdown(
            '<div style="color:#f87171;font-size:13px">● API not reachable</div>',
            unsafe_allow_html=True,
        )
        st.code("make run-api\n# or\nmake docker-up", language="bash")

    st.markdown("---")

    # Metrics
    try:
        m = requests.get(f"{API_URL}/metrics", timeout=3).json()
        if "ner" in m:
            st.markdown("**Eval Metrics** *(from Colab)*")
            ner = m["ner"].get("overall", {})
            for k, v in ner.items():
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 0;font-size:13px;">'
                    f'<span style="color:#64748b">NER {k.upper()}</span>'
                    f'<span style="color:#10b981;font-family:IBM Plex Mono,monospace">{v}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            lat = m["ner"].get("latency_ms")
            if lat:
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'padding:3px 0;font-size:13px;">'
                    f'<span style="color:#64748b">Latency</span>'
                    f'<span style="color:#3b82f6;font-family:IBM Plex Mono,monospace">{lat} ms</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        st.markdown(
            '<div style="color:#475569;font-size:12px">'
            "Metrics appear after Colab training."
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("**Pipeline**")
    for step in [
        "🎙️ Audio → Whisper → Text",
        "🔍 Text → DistilBERT NER",
        "📋 Entities → Care JSON",
    ]:
        st.markdown(
            f'<div style="font-size:12px;color:#94a3b8;padding:3px 0">{step}</div>',
            unsafe_allow_html=True,
        )


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="background:#111827;border:1px solid #1e2740;'
    'border-radius:14px;padding:24px 28px;margin-bottom:20px;">'
    '<div style="font-size:22px;font-weight:700;color:#f1f5f9">'
    '🏥 Healthcare Documentation Pipeline</div>'
    '<div style="font-size:14px;color:#64748b;margin-top:4px">'
    'Caregiver speech → structured care note · '
    '<span style="color:#3b82f6">pkl model · DVC tracked · CI/CD deployed</span>'
    '</div></div>',
    unsafe_allow_html=True,
)


# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("### Input")
tab_text, tab_audio = st.tabs(["📝 Text / Transcript", "🎙️ Upload Audio"])

with tab_text:
    col_ex, col_in = st.columns([1, 2])
    with col_ex:
        st.markdown('<div class="lbl" style="margin-bottom:8px">Examples</div>', unsafe_allow_html=True)
        for lbl, txt in EXAMPLES.items():
            if st.button(lbl, key=lbl):
                st.session_state["txt"] = txt
                st.rerun()
    with col_in:
        transcript_input = st.text_area(
            "transcript",
            value=st.session_state.get("txt", list(EXAMPLES.values())[0]),
            height=140,
            label_visibility="collapsed",
        )

with tab_audio:
    audio_file = st.file_uploader(
        "Upload audio",
        type=["wav", "mp3", "m4a", "ogg"],
        label_visibility="collapsed",
    )
    if audio_file:
        st.audio(audio_file)
        st.info(
            "Audio → Whisper transcribes → NER extracts entities → structured JSON",
            icon="ℹ️",
        )

st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("▶  Run Pipeline")


# ── Helpers ───────────────────────────────────────────────────────────────────

def call_api(text=None, audio=None):
    try:
        if audio:
            resp = requests.post(
                f"{API_URL}/predict/audio",
                files={"file": (audio.name, audio.getvalue(), "audio/wav")},
                timeout=60,
            )
        else:
            resp = requests.post(
                f"{API_URL}/predict/text",
                json={"text": text},
                timeout=30,
            )
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        st.error("❌ API not reachable. Run: `make run-api` or `make docker-up`")
    except requests.HTTPError as e:
        st.error(f"❌ API error {e.response.status_code}: {e.response.text}")
    return None


def highlight(transcript: str, entities: list) -> str:
    if not entities:
        return f'<div class="tx-box">{transcript}</div>'

    positioned = []
    for ent in entities:
        idx = transcript.lower().find(ent["text"].lower())
        if idx != -1:
            positioned.append({**ent, "start": idx, "end": idx + len(ent["text"])})
    positioned.sort(key=lambda x: x["start"])

    parts, last = [], 0
    for ent in positioned:
        s, e = ent["start"], ent["end"]
        if s > last:
            parts.append(transcript[last:s])
        parts.append(
            f'<span class="tag {ent["label"]}" '
            f'title="{ent["label"]} ({ent["confidence"]:.0%})">'
            f'{ICONS.get(ent["label"],"")} {transcript[s:e]}'
            f'<sup style="font-size:9px;opacity:.5;margin-left:2px">'
            f'{ent["label"]}</sup></span>'
        )
        last = e
    if last < len(transcript):
        parts.append(transcript[last:])
    return f'<div class="tx-box">{"".join(parts)}</div>'


def field_row(key, val, color="#f1f5f9"):
    if not val:
        return
    st.markdown(
        f'<div class="f-row">'
        f'<span class="f-key">{key}</span>'
        f'<span class="f-val" style="color:{color}">{val}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_care_form(doc: dict):
    st.markdown('<div class="card card-green">', unsafe_allow_html=True)
    st.markdown('<div class="lbl">Stage 3 — Output</div>', unsafe_allow_html=True)
    st.markdown('<div class="ttl">📋 Care Documentation</div>', unsafe_allow_html=True)

    field_row("Patient",     doc.get("patient_name"),     "#60a5fa")
    field_row("Time",        doc.get("observation_time"), "#fbbf24")

    vitals = doc.get("vitals", {})
    if vitals:
        st.markdown('<div class="sec-hdr">Vital Signs</div>', unsafe_allow_html=True)
        field_row("Temperature",   vitals.get("temperature_celsius") and f'{vitals["temperature_celsius"]} °C',  "#c084fc")
        field_row("Blood Pressure",vitals.get("blood_pressure"),  "#c084fc")
        field_row("Pulse",         vitals.get("pulse_bpm") and f'{vitals["pulse_bpm"]} bpm',            "#c084fc")
        field_row("Blood Glucose", vitals.get("blood_glucose_mmol") and f'{vitals["blood_glucose_mmol"]} mmol/L', "#c084fc")
        field_row("O₂ Saturation", vitals.get("oxygen_saturation_pct") and f'{vitals["oxygen_saturation_pct"]}%', "#c084fc")

    symptoms = doc.get("symptoms", [])
    if symptoms:
        st.markdown('<div class="sec-hdr">Symptoms</div>', unsafe_allow_html=True)
        st.markdown(
            " ".join(f'<span class="tag SYMPTOM">{s}</span>' for s in symptoms),
            unsafe_allow_html=True,
        )

    meds = doc.get("medications", [])
    if meds:
        st.markdown('<div class="sec-hdr">Medications</div>', unsafe_allow_html=True)
        for med in meds:
            c = "#4ade80" if med["administered"] else "#f87171"
            t = "✓ Administered" if med["administered"] else "✗ Refused"
            d = f' · {med["dosage"]}' if med.get("dosage") else ""
            field_row(f'{med["name"]}{d}', t, c)

    procs = doc.get("procedures", [])
    if procs:
        st.markdown('<div class="sec-hdr">Procedures</div>', unsafe_allow_html=True)
        st.markdown(
            " ".join(f'<span class="tag PROCEDURE">{p}</span>' for p in procs),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run:
    st.markdown("---")
    use_audio = audio_file is not None

    if not use_audio and not transcript_input.strip():
        st.warning("Enter a transcript or upload audio.")
        st.stop()

    with st.spinner("Running pipeline..."):
        result = call_api(
            text=transcript_input if not use_audio else None,
            audio=audio_file if use_audio else None,
        )

    if not result:
        st.stop()

    entities = result.get("entities", [])
    care_doc = result.get("care_documentation", {})
    timings  = result.get("timings_ms", {})
    shown_tx = result.get("transcript") or transcript_input

    st.markdown("### Results")

    # Stage 1 — ASR (only for audio)
    if result.get("transcript"):
        st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
        st.markdown('<div class="lbl">Stage 1 — ASR · Whisper-small</div>', unsafe_allow_html=True)
        st.markdown('<div class="ttl">🎙️ Transcript</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="tx-box">{result["transcript"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:12px;color:#64748b;margin-top:6px">'
            f'⏱ {timings.get("asr", 0):.0f} ms</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # Stage 2 — NER
    st.markdown('<div class="card card-blue">', unsafe_allow_html=True)
    st.markdown('<div class="lbl">Stage 2 — NER · DistilBERT (pkl)</div>', unsafe_allow_html=True)
    st.markdown('<div class="ttl">🔍 Entities in transcript</div>', unsafe_allow_html=True)
    st.markdown(highlight(shown_tx, entities), unsafe_allow_html=True)

    if entities:
        tags = " ".join(
            f'<span class="tag {e["label"]}">'
            f'{ICONS.get(e["label"],"")} {e["label"]}</span>'
            for e in entities
        )
        st.markdown(f'<div style="margin-top:8px">{tags}</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div style="font-size:12px;color:#64748b;margin-top:6px">'
        f'⏱ {timings.get("ner", 0):.0f} ms · {len(entities)} entities</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Stage 3 — Structured output
    col_form, col_json = st.columns(2)
    with col_form:
        render_care_form(care_doc)
    with col_json:
        st.markdown('<div class="card card-green">', unsafe_allow_html=True)
        st.markdown('<div class="lbl">Raw JSON Output</div>', unsafe_allow_html=True)
        st.markdown('<div class="ttl">{ } Machine-readable</div>', unsafe_allow_html=True)
        st.json(care_doc)
        st.markdown("</div>", unsafe_allow_html=True)

    # Timings
    st.markdown("---")
    st.markdown("### ⏱ Latency")
    colors = {"asr": "#3b82f6", "ner": "#8b5cf6", "total": "#f59e0b"}
    cols = st.columns(len(timings))
    for col, (stage, ms) in zip(cols, timings.items()):
        with col:
            st.markdown(
                f'<div class="m-box">'
                f'<div class="m-val" style="color:{colors.get(stage,"#3b82f6")}">'
                f'{ms:.0f} ms</div>'
                f'<div class="m-lbl">{stage}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        '<div style="text-align:center;padding:48px;background:#111827;'
        'border:1px dashed #1e2740;border-radius:16px;margin-top:24px;">'
        '<div style="font-size:40px;margin-bottom:12px">🎙️ → 📋</div>'
        '<div style="font-size:16px;color:#94a3b8">'
        'Select an example or paste a transcript</div>'
        '<div style="font-size:13px;color:#475569;margin-top:8px">'
        'Then click ▶ Run Pipeline</div>'
        '</div>',
        unsafe_allow_html=True,
    )
