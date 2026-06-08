# app.py
# Streamlit UI for the AI Image Detector — Stage 1 metadata analysis.

import os
import sys
import json
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image

import pipeline
import logger
from config import AUDIT_LOG_PATH, STAGE1_THRESHOLDS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1117 0%, #1a1d27 100%);
    border-right: 1px solid #2a2d3e;
}
.stApp { background: #0d0f18; }

.score-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #252840 100%);
    border: 1px solid #3a3d5c;
    border-radius: 16px;
    padding: 28px 32px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.score-number {
    font-size: 72px;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 8px;
}
.score-label {
    font-size: 13px;
    color: #8b8fa8;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
.verdict-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 100px;
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-top: 14px;
}
.signal-row {
    background: #1a1d2e;
    border: 1px solid #2a2d3e;
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #a0a3b8;
    font-family: 'SF Mono', 'Fira Code', monospace;
}
.log-entry {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 12px;
    color: #a0a3b8;
    padding: 4px 0;
    border-bottom: 1px solid #1e2030;
    line-height: 1.6;
}
.next-stage-box {
    background: #12151f;
    border: 1px solid #2a4a3e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-top: 8px;
}
[data-testid="stMetricValue"] { color: #e8eaf6 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score < STAGE1_THRESHOLDS["likely_real"]:
        return "#22c55e"
    elif score < STAGE1_THRESHOLDS["uncertain"]:
        return "#f59e0b"
    elif score < STAGE1_THRESHOLDS["probably_ai"]:
        return "#f97316"
    else:
        return "#ef4444"


def _verdict_style(verdict: str) -> str:
    return {
        "LIKELY_REAL":  "background:#14532d; color:#4ade80; border:1px solid #16a34a",
        "UNCERTAIN":    "background:#451a03; color:#fbbf24; border:1px solid #d97706",
        "PROBABLY_AI":  "background:#431407; color:#fb923c; border:1px solid #ea580c",
        "LIKELY_AI":    "background:#450a0a; color:#f87171; border:1px solid #dc2626",
    }.get(verdict, "background:#252840; color:#a0a3b8; border:1px solid #3a3d5c")


def _load_audit_log() -> list:
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    entries = []
    with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return list(reversed(entries))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔍 AI Image Detector")
    st.markdown("<p style='color:#6b6f88; font-size:13px;'>Stage 1 — Metadata Analysis</p>",
                unsafe_allow_html=True)
    st.divider()

    page = st.radio(
        "Navigation",
        ["Analyze Image", "Audit Log"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
    <div style='color:#4b4f68; font-size:11px; line-height:2;'>
    <b style='color:#6b6f88'>Stage 1 score bands</b><br>
    🟢 0.00 – 0.30 &nbsp; LIKELY_REAL<br>
    🟡 0.30 – 0.60 &nbsp; UNCERTAIN<br>
    🟠 0.60 – 0.85 &nbsp; PROBABLY_AI<br>
    🔴 0.85 – 1.00 &nbsp; LIKELY_AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:20px; color:#4b4f68; font-size:11px; line-height:1.9;'>
    <b style='color:#6b6f88'>Detected signals</b><br>
    🔴 Software tag (Midjourney, SD…)<br>
    🔴 PNG params chunk (A1111/ComfyUI)<br>
    🔴 XMP AI claim (Firefly, DALL-E)<br>
    🔴 C2PA manifest<br>
    🟢 Real camera EXIF (Canon, Sony…)
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analyze Image
# ══════════════════════════════════════════════════════════════════════════════
if page == "Analyze Image":
    st.markdown("## Analyze Image")
    st.markdown("<p style='color:#6b6f88;'>Upload an image to run Stage 1 metadata & EXIF analysis.</p>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "webp", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded:
        col_img, col_gap, col_result = st.columns([1, 0.05, 1.6])

        with col_img:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True, caption=uploaded.name)
            st.markdown(f"""
            <div style='background:#1a1d2e; border:1px solid #2a2d3e; border-radius:12px; padding:16px; margin-top:8px;'>
                <div style='font-size:11px; color:#6b6f88; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>File info</div>
                <div style='color:#a0a3b8; font-size:13px; line-height:2;'>
                    <b>Name:</b> {uploaded.name}<br>
                    <b>Size:</b> {round(uploaded.size / 1024, 1)} KB<br>
                    <b>Format:</b> {img.format or uploaded.type}<br>
                    <b>Dimensions:</b> {img.width} × {img.height}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_result:
            if st.button("🔍 Run Stage 1 Analysis", use_container_width=True, type="primary"):
                with st.spinner("Analysing metadata…"):
                    suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded.getvalue())
                        tmp_path = tmp.name
                    try:
                        result = pipeline.run(tmp_path)
                    finally:
                        os.unlink(tmp_path)
                st.session_state["last_result"] = result

            result = st.session_state.get("last_result")

            if result:
                score   = result["stage1_ai_score"]
                verdict = result["stage1_verdict"]
                color   = _score_color(score)

                # ── Score card ────────────────────────────────────────────────
                st.markdown(f"""
                <div class='score-card'>
                    <div class='score-number' style='color:{color};'>{score:.2f}</div>
                    <div class='score-label'>Stage 1 AI Score</div>
                    <div class='verdict-badge' style='{_verdict_style(verdict)}'>{verdict}</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                m1, m2, m3 = st.columns(3)
                m1.metric("Processing", f"{result['processing_time_ms']}ms")
                m2.metric("Format", result.get("format", "?"))
                m3.metric("Signals", len(result.get("stage1_signals", [])))

                st.markdown("")

                # ── Signals ───────────────────────────────────────────────────
                with st.expander("🔎 Detected Signals", expanded=True):
                    c1, c2 = st.columns(2)

                    sw_det = result.get("ai_software_detected", False)
                    sw_nm  = result.get("ai_software_name") or "N/A"
                    c1.markdown(f"**AI Software Tag:** {'🔴 ' + sw_nm.upper() if sw_det else '✅ Not detected'}")

                    sd = result.get("sd_png_params_found", False)
                    c1.markdown(f"**SD/ComfyUI PNG Params:** {'🔴 Found' if sd else '✅ Not found'}")

                    xmp = result.get("xmp_ai_claim_found", False)
                    c1.markdown(f"**XMP AI Claim:** {'🔴 Found' if xmp else '✅ Not found'}")

                    c2pa = result.get("has_c2pa", False)
                    c1.markdown(f"**C2PA Manifest:** {'⚠️ Found' if c2pa else '✅ Not found'}")

                    c2.markdown(f"**EXIF Score:** `{result.get('exif_score', 0):.2f}`")
                    rcm = result.get("real_camera_make", False)
                    c2.markdown(f"**Real Camera Make:** {'✅ ' + str(result.get('camera_make', '')) if rcm else '⚠️ Not identified'}")
                    c2.markdown(f"**Camera Model:** `{result.get('camera_model') or 'N/A'}`")
                    c2.markdown(f"**Has GPS:** {'Yes' if result.get('has_gps') else 'No'}")

                    signals = result.get("stage1_signals", [])
                    if signals:
                        st.markdown("---")
                        st.markdown("**Signal breakdown:**")
                        for sig in signals:
                            icon  = "🔴" if sig["direction"] == "ai" else "🟢"
                            badge = f"w={sig['weight']:+.2f}"
                            st.markdown(
                                f"<div class='signal-row'>{icon} <b>{sig['signal']}</b> <span style='color:#4b4f68'>({badge})</span> — {sig['detail'][:90]}</div>",
                                unsafe_allow_html=True
                            )

                    ee = result.get("early_exit")
                    if ee:
                        ev = result.get("early_exit_verdict", "")
                        st.info(f"⚡ Early exit triggered → **{ev}** (Stage 2 would be skipped)")

                # ── Pipeline log ──────────────────────────────────────────────
                with st.expander("📜 Pipeline Log"):
                    for line in result.get("pipeline_log", []):
                        st.markdown(f"<div class='log-entry'>{line}</div>", unsafe_allow_html=True)

                # ── next_stage_input (open-ended hook) ────────────────────────
                with st.expander("🔌 next_stage_input — Ready for Stage 2"):
                    st.markdown(
                        "<p style='color:#6b6f88; font-size:12px; margin-bottom:8px;'>"
                        "This payload is pre-packaged and ready to send to any external API "
                        "(Hive, Google SafeSearch, a custom model, etc.).</p>",
                        unsafe_allow_html=True
                    )
                    st.code(
                        json.dumps(result.get("next_stage_input", {}), indent=2, default=str),
                        language="json"
                    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Audit Log
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    st.markdown("## Audit Log")
    st.markdown("<p style='color:#6b6f88;'>Every pipeline run is recorded here. Most recent first.</p>",
                unsafe_allow_html=True)

    entries = _load_audit_log()

    if not entries:
        st.info("No pipeline runs recorded yet.")
    else:
        verdicts = [e.get("stage1_verdict") for e in entries]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Runs",    len(entries))
        s2.metric("LIKELY_REAL",   verdicts.count("LIKELY_REAL"))
        s3.metric("UNCERTAIN",     verdicts.count("UNCERTAIN"))
        s4.metric("LIKELY_AI",     verdicts.count("LIKELY_AI"))

        st.markdown("")

        for entry in entries:
            score   = entry.get("stage1_ai_score", 0)
            verdict = entry.get("stage1_verdict", "unknown")
            color   = _score_color(score)
            ran_at  = entry.get("ran_at", "")[:19].replace("T", " ")

            with st.expander(
                f"{os.path.basename(entry.get('image_path', 'unknown'))}  —  "
                f"Score: {score:.2f}  |  {verdict}  |  {ran_at} UTC"
            ):
                tab1, tab2 = st.tabs(["Summary", "Raw JSON"])

                with tab1:
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**Score:** `{score:.4f}`")
                    c1.markdown(f"**Verdict:** `{verdict}`")
                    c1.markdown(f"**Processing:** `{entry.get('processing_time_ms')}ms`")
                    c2.markdown(f"**AI Software:** `{entry.get('ai_software_name') or 'N/A'}`")
                    c2.markdown(f"**SD Params:** `{entry.get('sd_png_params_found')}`")
                    c2.markdown(f"**C2PA:** `{entry.get('has_c2pa')}`")
                    c3.markdown(f"**EXIF Score:** `{entry.get('exif_score')}`")
                    c3.markdown(f"**Camera Make:** `{entry.get('camera_make') or 'N/A'}`")
                    c3.markdown(f"**Format:** `{entry.get('format')}` {entry.get('width')}×{entry.get('height')}")

                with tab2:
                    st.code(json.dumps(entry, indent=2, default=str), language="json")
