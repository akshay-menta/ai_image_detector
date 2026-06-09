# poc_app.py
# ─────────────────────────────────────────────────────────────────────────────
# AI Image Detector — Full 3-Stage POC
# Stage 1: Metadata + EXIF analysis (free, local)
# Stage 2: Winston AI image detection API
# Stage 3: Confidence router → AUTO_APPROVE / HUMAN_REVIEW / AUTO_REJECT
# ─────────────────────────────────────────────────────────────────────────────

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

import pipeline          # Stage 1 orchestrator
import stage2_winston    # Stage 2 Winston AI image detection
import stage3_router     # Stage 3 Router
import logger
from config import AUDIT_LOG_PATH, STAGE1_THRESHOLDS

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector — POC",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    box-sizing: border-box;
}

.stApp { background: #080b14; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c0f1c 0%, #0f1525 100%);
    border-right: 1px solid #1e2540;
}

/* ── Header ── */
.poc-header {
    background: linear-gradient(135deg, #0f1525 0%, #1a2035 50%, #0f2040 100%);
    border: 1px solid #1e3560;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.poc-header::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,179,237,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.poc-title {
    font-size: 28px;
    font-weight: 800;
    color: #e8f4fd;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.poc-subtitle {
    font-size: 14px;
    color: #6b7fa8;
    margin: 0;
    font-weight: 400;
}

/* ── Stage cards ── */
.stage-card {
    background: #0f1525;
    border: 1px solid #1e2540;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s;
}
.stage-card.active { border-color: #3b5bdb; }
.stage-card.done-green { border-color: #22c55e; }
.stage-card.done-red { border-color: #ef4444; }
.stage-card.done-yellow { border-color: #f59e0b; }

.stage-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a6fa5;
    margin-bottom: 10px;
}
.stage-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #3b5bdb;
}
.stage-dot.green { background: #22c55e; }
.stage-dot.red { background: #ef4444; }
.stage-dot.yellow { background: #f59e0b; }
.stage-dot.gray { background: #4a6fa5; }

.stage-title {
    font-size: 17px;
    font-weight: 700;
    color: #dde6f7;
    margin: 0 0 4px 0;
}
.stage-desc {
    font-size: 12px;
    color: #5a6d8a;
    margin: 0;
}

/* ── Score ── */
.score-display {
    text-align: center;
    padding: 20px;
}
.score-big {
    font-size: 68px;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -2px;
}
.score-label {
    font-size: 11px;
    color: #4a6fa5;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

/* ── Verdict badge ── */
.verdict-pill {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.8px;
}

/* ── Action badge ── */
.action-card {
    border-radius: 14px;
    padding: 20px 24px;
    text-align: center;
    margin: 12px 0;
}
.action-title {
    font-size: 22px;
    font-weight: 800;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.action-desc {
    font-size: 12px;
    opacity: 0.75;
}

/* ── Pipeline bar ── */
.pipeline-bar {
    display: flex;
    align-items: center;
    gap: 0;
    margin: 20px 0 28px;
    padding: 0 4px;
}
.pipe-step {
    flex: 1;
    text-align: center;
    padding: 12px 6px;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 600;
    color: #4a6fa5;
    background: #0f1525;
    border: 1px solid #1e2540;
    position: relative;
}
.pipe-step.active {
    color: #93c5fd;
    background: #1a2540;
    border-color: #3b5bdb;
}
.pipe-step.complete {
    color: #4ade80;
    background: #0d1f14;
    border-color: #22c55e;
}
.pipe-arrow {
    color: #2a3558;
    font-size: 18px;
    padding: 0 6px;
    flex-shrink: 0;
}

/* ── Signal rows ── */
.sig-row {
    background: #0c1120;
    border: 1px solid #1a2540;
    border-radius: 8px;
    padding: 9px 14px;
    margin-bottom: 6px;
    font-size: 12px;
    font-family: 'SF Mono', 'Fira Code', monospace;
    color: #8a9fc0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.sig-icon { font-size: 14px; flex-shrink: 0; }
.sig-name { font-weight: 600; color: #c0cce0; }
.sig-detail { color: #506080; font-size: 11px; }
.sig-weight { margin-left: auto; font-weight: 700; font-size: 11px; }

/* ── Reasoning box ── */
.reasoning-box {
    background: linear-gradient(135deg, #0c1520 0%, #111e30 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
}
.reasoning-label {
    font-size: 10px;
    font-weight: 700;
    color: #3b6ea5;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
}
.reasoning-text {
    font-size: 14px;
    color: #a8c4e0;
    line-height: 1.7;
}

/* ── Metric chips ── */
.metric-chip {
    background: #0c1120;
    border: 1px solid #1a2540;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-val { font-size: 22px; font-weight: 700; color: #e0eaff; }
.metric-key { font-size: 11px; color: #3b5b7a; text-transform: uppercase; letter-spacing: 1px; margin-top: 2px; }

/* ── Dividers ── */
.section-divider {
    border: none;
    border-top: 1px solid #1a2540;
    margin: 20px 0;
}

/* ── File Uploader — complete restyle ── */
[data-testid="stFileUploader"] {
    width: 100%;
}
[data-testid="stFileUploader"] section {
    background: linear-gradient(135deg, #0c1120 0%, #0f1830 100%) !important;
    border: 2px dashed #253560 !important;
    border-radius: 16px !important;
    padding: 40px 24px !important;
    text-align: center !important;
    transition: border-color 0.25s, background 0.25s !important;
    cursor: pointer !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: #3b5bdb !important;
    background: linear-gradient(135deg, #0f1528 0%, #131d3a 100%) !important;
}
/* Hide the raw "Browse files" button text area and replace with clean look */
[data-testid="stFileUploader"] section > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 10px !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
/* The inner upload button — make it look premium */
[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 8px !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div span {
    font-size: 14px !important;
    color: #6b8ab8 !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] div small {
    font-size: 12px !important;
    color: #3b5570 !important;
}
/* The actual "Browse files" button inside */
[data-testid="stFileUploader"] button {
    background: linear-gradient(135deg, #1e2e50 0%, #263660 100%) !important;
    border: 1px solid #3b5bdb !important;
    border-radius: 10px !important;
    color: #93c5fd !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 8px 22px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s !important;
    cursor: pointer !important;
    margin-top: 4px !important;
}
[data-testid="stFileUploader"] button:hover {
    background: linear-gradient(135deg, #253860 0%, #2e4478 100%) !important;
    border-color: #60a5fa !important;
    color: #bfdbfe !important;
}
/* Uploaded file name pill */
[data-testid="stFileUploaderFile"] {
    background: #0c1120 !important;
    border: 1px solid #1e2d50 !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    margin-top: 8px !important;
}
[data-testid="stFileUploaderFile"] span {
    color: #93c5fd !important;
    font-size: 13px !important;
}

/* ── Hide streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stMetricValue"] { color: #e0eaff !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score < 0.30: return "#22c55e"
    if score < 0.60: return "#f59e0b"
    if score < 0.85: return "#f97316"
    return "#ef4444"

def _verdict_style(verdict: str) -> str:
    return {
        "LIKELY_REAL":  "background:#0d2218; color:#4ade80; border:1px solid #166534",
        "UNCERTAIN":    "background:#1c1400; color:#fbbf24; border:1px solid #854d0e",
        "PROBABLY_AI":  "background:#1e0d00; color:#fb923c; border:1px solid #9a3412",
        "LIKELY_AI":    "background:#1e0505; color:#f87171; border:1px solid #991b1b",
    }.get(verdict, "background:#1a2035; color:#a0b4cc; border:1px solid #2a3558")

def _action_style(action: str) -> tuple[str, str]:
    return {
        "AUTO_APPROVE": ("background:linear-gradient(135deg,#0d2218,#0f3020); color:#4ade80; border:2px solid #166534",
                         "✅ AUTO APPROVED"),
        "HUMAN_REVIEW": ("background:linear-gradient(135deg,#1c1400,#1e1a00); color:#fbbf24; border:2px solid #854d0e",
                         "👁 HUMAN REVIEW"),
        "AUTO_REJECT":  ("background:linear-gradient(135deg,#1e0505,#200808); color:#f87171; border:2px solid #991b1b",
                         "❌ AUTO REJECTED"),
    }.get(action, ("background:#1a2035; color:#a0b4cc; border:2px solid #2a3558", "⚠️ UNKNOWN"))

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

def _dot_class(color: str) -> str:
    return {"green": "green", "red": "red", "yellow": "yellow"}.get(color, "gray")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 16px; border-bottom: 1px solid #1a2540; margin-bottom: 20px;'>
        <div style='font-size: 20px; font-weight: 800; color: #e0eaff; letter-spacing: -0.5px;'>🔬 AI Detector</div>
        <div style='font-size: 11px; color: #3b5b7a; margin-top: 3px; text-transform: uppercase; letter-spacing: 1px;'>3-Stage POC Pipeline</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "nav",
        ["🔍 Analyze Image", "📋 Audit Log", "ℹ️ How It Works"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#1a2540; margin: 16px 0;'>", unsafe_allow_html=True)

    # API Key input
    st.markdown("<div style='font-size:11px; color:#3b5b7a; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Winston AI Key (Stage 2)</div>", unsafe_allow_html=True)
    winston_key = st.text_input(
        "winston_key",
        value=os.environ.get("WINSTON_API_KEY", ""),
        type="password",
        placeholder="winston_...",
        label_visibility="collapsed",
    )
    if winston_key:
        st.markdown("<div style='color:#4ade80; font-size:11px;'>✓ Winston AI Stage 2 enabled</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='color:#f59e0b; font-size:11px;'>⚠ Stage 2 disabled (Stage 1 only)</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2540; margin: 16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:11px; color:#3b5b7a; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Score bands</div>
    <div style='font-size:12px; color:#5a7090; line-height:2.2;'>
    🟢 &nbsp;0.00 – 0.30 &nbsp;&nbsp; LIKELY_REAL<br>
    🟡 &nbsp;0.30 – 0.60 &nbsp;&nbsp; UNCERTAIN<br>
    🟠 &nbsp;0.60 – 0.85 &nbsp;&nbsp; PROBABLY_AI<br>
    🔴 &nbsp;0.85 – 1.00 &nbsp;&nbsp; LIKELY_AI
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1a2540; margin: 16px 0;'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:11px; color:#3b5b7a; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;'>Routing actions</div>
    <div style='font-size:12px; color:#5a7090; line-height:2.2;'>
    ✅ &nbsp;Score &lt; 0.30 &nbsp;→&nbsp; Auto Approve<br>
    👁 &nbsp;Score 0.30–0.85 &nbsp;→&nbsp; Human Review<br>
    ❌ &nbsp;Score &gt; 0.85 &nbsp;→&nbsp; Auto Reject
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Analyze Image
# ══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Analyze Image":

    st.markdown("""
    <div class='poc-header'>
        <div class='poc-title'>🔬 AI Image Detection Pipeline</div>
        <div class='poc-subtitle'>3-stage pipeline · Stage 1 free metadata check → Stage 2 Winston AI → Stage 3 confidence router</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom: 0px;'>
        <div style='
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: linear-gradient(135deg, #0c1120 0%, #0f1830 100%);
            border: 2px dashed #253560;
            border-radius: 16px;
            padding: 32px 24px 12px;
            margin-bottom: -12px;
            position: relative;
            z-index: 1;
        '>
            <div style='font-size: 36px; margin-bottom: 10px;'>🖼️</div>
            <div style='font-size: 15px; font-weight: 600; color: #7090b8; margin-bottom: 4px;'>Drop your image here</div>
            <div style='font-size: 12px; color: #3b5570; margin-bottom: 14px;'>JPG · PNG · WEBP · TIFF · BMP</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png", "webp", "tiff", "bmp"],
        label_visibility="collapsed",
    )

    if not uploaded:
        # Empty state
        st.markdown("""
        <div style='text-align:center; padding: 60px 20px; color:#2a3a58;'>
            <div style='font-size: 48px; margin-bottom: 16px;'>📂</div>
            <div style='font-size: 18px; font-weight: 600; color: #3a5078;'>Upload an image to begin</div>
            <div style='font-size: 13px; margin-top: 8px;'>JPG, PNG, WEBP, TIFF, BMP supported</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        col_img, col_main = st.columns([1, 1.8], gap="large")

        with col_img:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True, caption=uploaded.name)
            st.markdown(f"""
            <div class='metric-chip' style='text-align:left; margin-top:12px;'>
                <div style='font-size:11px; color:#3b5b7a; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>File Info</div>
                <div style='font-size:13px; color:#8aa0c0; line-height:2;'>
                    <b style='color:#b0c4de;'>Name:</b> {uploaded.name}<br>
                    <b style='color:#b0c4de;'>Size:</b> {round(uploaded.size / 1024, 1)} KB<br>
                    <b style='color:#b0c4de;'>Format:</b> {img.format or uploaded.type.split('/')[-1].upper()}<br>
                    <b style='color:#b0c4de;'>Dimensions:</b> {img.width} × {img.height}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_main:
            run_btn = st.button(
                "🚀 Run Full Detection Pipeline",
                use_container_width=True,
                type="primary",
            )

            if run_btn:
                suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                try:
                    # ── Stage 1 ──────────────────────────────────────────────
                    with st.spinner("⚙️ Stage 1 · Metadata & EXIF analysis…"):
                        s1_result = pipeline.run(tmp_path)

                    s1_score = s1_result["stage1_ai_score"]
                    early_exit = s1_result.get("early_exit", False)

                    # ── Stage 2 ──────────────────────────────────────────────
                    s2_result = None
                    if early_exit:
                        st.info(f"⚡ **Early exit** — Stage 1 is definitive ({s1_result.get('early_exit_verdict')}). Stage 2 skipped to save API cost.")
                        final_score = s1_score
                    elif winston_key:
                        with st.spinner("🔬 Stage 2 · Winston AI image detection…"):
                            s2_result = stage2_winston.run(tmp_path, s1_result, winston_key)
                        s2_score = s2_result.get("stage2_ai_score")
                        if s2_score is not None:
                            # Blend: Stage 2 × 75% + Stage 1 × 25%
                            final_score = round(s2_score * 0.75 + s1_score * 0.25, 4)
                        else:
                            final_score = s1_score
                    else:
                        st.warning("⚠️ No Winston AI key — running Stage 1 only. Add your key in the sidebar for the full pipeline.")
                        final_score = s1_score

                    # ── Stage 3 ──────────────────────────────────────────────
                    route = stage3_router.route(final_score, s1_result, s2_result)

                    st.session_state["poc_result"] = {
                        "s1": s1_result,
                        "s2": s2_result,
                        "route": route,
                        "final_score": final_score,
                        "early_exit": early_exit,
                    }

                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

            # ── Display results ───────────────────────────────────────────────
            r = st.session_state.get("poc_result")

            if r:
                s1 = r["s1"]
                s2 = r["s2"]
                route = r["route"]
                final = r["final_score"]
                color = _score_color(final)
                verdict = route["verdict"]
                action = route["action"]

                # ── Pipeline progress bar ─────────────────────────────────────
                s2_done = s2 is not None and s2.get("stage2_ai_score") is not None
                st.markdown(f"""
                <div class='pipeline-bar'>
                    <div class='pipe-step complete'>
                        <div>Stage 1</div>
                        <div style='font-size:10px; color:#22c55e; margin-top:2px;'>Metadata ✓</div>
                    </div>
                    <div class='pipe-arrow'>→</div>
                    <div class='pipe-step {"complete" if s2_done else "active"}'>
                        <div>Stage 2</div>
                        <div style='font-size:10px; margin-top:2px; color:{"#22c55e" if s2_done else "#f59e0b"};'>
                            {"Winston AI ✓" if s2_done else ("Skipped" if r["early_exit"] else "Stage 1 only")}
                        </div>
                    </div>
                    <div class='pipe-arrow'>→</div>
                    <div class='pipe-step complete'>
                        <div>Stage 3</div>
                        <div style='font-size:10px; color:#22c55e; margin-top:2px;'>Router ✓</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Final score ───────────────────────────────────────────────
                c_score, c_verdict = st.columns([1, 1])
                with c_score:
                    st.markdown(f"""
                    <div style='background:#0c1120; border:1px solid #1a2540; border-radius:14px; padding:20px; text-align:center;'>
                        <div style='font-size:56px; font-weight:800; color:{color}; line-height:1; letter-spacing:-2px;'>{final:.2f}</div>
                        <div style='font-size:10px; color:#3b5b7a; text-transform:uppercase; letter-spacing:1.5px; margin-top:6px;'>Final AI Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with c_verdict:
                    st.markdown(f"""
                    <div style='background:#0c1120; border:1px solid #1a2540; border-radius:14px; padding:20px; text-align:center; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center;'>
                        <div style='padding:8px 20px; border-radius:100px; font-size:13px; font-weight:700; {_verdict_style(verdict)}'>{verdict}</div>
                        <div style='font-size:10px; color:#3b5b7a; text-transform:uppercase; letter-spacing:1.5px; margin-top:10px;'>Verdict</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("")

                # ── Stage 3 Action card ───────────────────────────────────────
                astyle, alabel = _action_style(action)
                st.markdown(f"""
                <div class='action-card' style='{astyle}'>
                    <div class='action-title'>{alabel}</div>
                    <div class='action-desc'>{route["routing_reason"]}</div>
                </div>
                """, unsafe_allow_html=True)

                # ── Score breakdown ───────────────────────────────────────────
                s1_score = s1["stage1_ai_score"]
                s2_score = (s2 or {}).get("stage2_ai_score")

                cols = st.columns(3)
                cols[0].markdown(f"""
                <div class='metric-chip'>
                    <div class='metric-val' style='color:{_score_color(s1_score)};'>{s1_score:.2f}</div>
                    <div class='metric-key'>Stage 1 Score</div>
                </div>
                """, unsafe_allow_html=True)
                cols[1].markdown(f"""
                <div class='metric-chip'>
                    <div class='metric-val' style='color:{_score_color(s2_score or 0.5)};'>{f"{s2_score:.2f}" if s2_score is not None else "N/A"}</div>
                    <div class='metric-key'>Stage 2 Score</div>
                </div>
                """, unsafe_allow_html=True)
                cols[2].markdown(f"""
                <div class='metric-chip'>
                    <div class='metric-val' style='color:{color};'>{final:.2f}</div>
                    <div class='metric-key'>Blended Final</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("")

                # ── Winston AI Analysis (if Stage 2 ran) ─────────────────────
                if s2 and s2.get("stage2_reasoning"):
                    human_score = s2.get("stage2_human_score")
                    hs_label = f" · Human Score: {human_score:.1f}/100" if human_score is not None else ""
                    st.markdown(f"""
                    <div class='reasoning-box'>
                        <div class='reasoning-label'>🔬 Winston AI Analysis (Stage 2){hs_label}</div>
                        <div class='reasoning-text'>{s2["stage2_reasoning"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Stage 1 Signals ───────────────────────────────────────────
                with st.expander("🔎 Stage 1 — Detected Signals", expanded=False):
                    signals = s1.get("stage1_signals", [])
                    if signals:
                        for sig in signals:
                            icon = "🔴" if sig["direction"] == "ai" else "🟢"
                            wt = f"{sig['weight']:+.2f}"
                            wt_color = "#f87171" if sig["direction"] == "ai" else "#4ade80"
                            st.markdown(f"""
                            <div class='sig-row'>
                                <span class='sig-icon'>{icon}</span>
                                <span class='sig-name'>{sig["signal"]}</span>
                                <span class='sig-detail'>{sig["detail"][:80]}</span>
                                <span class='sig-weight' style='color:{wt_color};'>{wt}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("<div style='color:#3b5b7a; font-size:13px;'>No signals detected.</div>", unsafe_allow_html=True)

                    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"**AI Software:** {'🔴 ' + (s1.get('ai_software_name') or '').upper() if s1.get('ai_software_detected') else '✅ None'}")
                    c1.markdown(f"**SD PNG Params:** {'🔴 Found' if s1.get('sd_png_params_found') else '✅ None'}")
                    c1.markdown(f"**XMP AI Claim:** {'🔴 Found' if s1.get('xmp_ai_claim_found') else '✅ None'}")
                    c2.markdown(f"**C2PA Manifest:** {'⚠️ Found' if s1.get('has_c2pa') else '✅ None'}")
                    c2.markdown(f"**EXIF Score:** `{s1.get('exif_score', 0):.2f}`")
                    c2.markdown(f"**Real Camera:** {'✅ ' + str(s1.get('camera_make', '')) if s1.get('real_camera_make') else '⚠️ Not found'}")
                    c3.markdown(f"**Camera Model:** `{s1.get('camera_model') or 'N/A'}`")
                    c3.markdown(f"**Has GPS:** {'Yes' if s1.get('has_gps') else 'No'}")
                    c3.markdown(f"**Processing:** `{s1.get('processing_time_ms', '?')}ms`")

                # ── Pipeline log ──────────────────────────────────────────────
                with st.expander("📜 Pipeline Log (Stage 1 notes)"):
                    for line in s1.get("pipeline_log", []):
                        st.markdown(
                            f"<div style='font-family: monospace; font-size:11px; color:#4a6a88; padding:2px 0; border-bottom:1px solid #0f1520;'>{line}</div>",
                            unsafe_allow_html=True
                        )

                # ── next_stage_input ──────────────────────────────────────────
                with st.expander("🔌 Raw Pipeline Output (JSON)"):
                    output = {
                        "stage1": {k: v for k, v in s1.items() if k != "pipeline_log"},
                        "stage2": s2,
                        "stage3_route": route,
                        "final_score": final,
                    }
                    st.code(json.dumps(output, indent=2, default=str), language="json")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: Audit Log
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Audit Log":
    st.markdown("""
    <div class='poc-header'>
        <div class='poc-title'>📋 Audit Log</div>
        <div class='poc-subtitle'>Every pipeline run is recorded here — append-only JSONL trail</div>
    </div>
    """, unsafe_allow_html=True)

    entries = _load_audit_log()

    if not entries:
        st.info("No pipeline runs recorded yet. Analyze an image to get started.")
    else:
        verdicts = [e.get("stage1_verdict") for e in entries]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Runs", len(entries))
        s2.metric("LIKELY_REAL", verdicts.count("LIKELY_REAL"))
        s3.metric("UNCERTAIN",   verdicts.count("UNCERTAIN") + verdicts.count("PROBABLY_AI"))
        s4.metric("LIKELY_AI",   verdicts.count("LIKELY_AI"))

        st.markdown("")

        for entry in entries:
            score   = entry.get("stage1_ai_score", 0)
            verdict = entry.get("stage1_verdict", "unknown")
            color   = _score_color(score)
            ran_at  = entry.get("ran_at", "")[:19].replace("T", " ")

            with st.expander(
                f"{os.path.basename(entry.get('image_path', 'unknown'))}  ·  "
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
                    c3.markdown(f"**Camera:** `{entry.get('camera_make') or 'N/A'}`")
                    c3.markdown(f"**Format:** `{entry.get('format')}` {entry.get('width')}×{entry.get('height')}")
                with tab2:
                    st.code(json.dumps(entry, indent=2, default=str), language="json")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: How It Works
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ How It Works":
    st.markdown("""
    <div class='poc-header'>
        <div class='poc-title'>ℹ️ How the Pipeline Works</div>
        <div class='poc-subtitle'>A layered tool pipeline with confidence routing — built for accuracy and cost efficiency</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='stage-card done-green'>
        <div class='stage-badge'><span class='stage-dot green'></span> Stage 1</div>
        <div class='stage-title'>Metadata Agent — Free & Local</div>
        <div class='stage-desc'>Runs entirely on your machine. Zero API cost. Under 200ms.</div>
        <hr class='section-divider'>
        <div style='font-size:13px; color:#8aa0c0; line-height:1.9;'>
            <b style='color:#b0c4de;'>EXIF Richness Check:</b> Scores 12 camera-specific fields (Make, Model, ISO, GPS, Lens, etc.). Real cameras have rich EXIF; AI tools usually have none.<br>
            <b style='color:#b0c4de;'>Software Tag Detection:</b> AI tools like Midjourney, DALL-E, and Stable Diffusion often write their name directly into the EXIF Software field.<br>
            <b style='color:#b0c4de;'>PNG Chunk Analysis:</b> Stable Diffusion and ComfyUI embed full generation parameters (prompt, seed, steps, sampler) into PNG text chunks.<br>
            <b style='color:#b0c4de;'>XMP Metadata:</b> Adobe Firefly and DALL-E embed AI provenance claims in XMP namespaces.<br>
            <b style='color:#b0c4de;'>C2PA Manifest:</b> DALL-E and modern tools embed cryptographic content credentials. If found — definitive proof, pipeline stops here.
        </div>
    </div>

    <div class='stage-card done-yellow'>
        <div class='stage-badge'><span class='stage-dot yellow'></span> Stage 2</div>
        <div class='stage-title'>Winston AI — Purpose-built image detection model</div>
        <div class='stage-desc'>Runs only when Stage 1 is inconclusive. 99.98% accuracy, 0.5% false-positive rate.</div>
        <hr class='section-divider'>
        <div style='font-size:13px; color:#8aa0c0; line-height:1.9;'>
            <b style='color:#b0c4de;'>When it runs:</b> Only when Stage 1 score is in the 0.30–0.85 gray zone — no definitive metadata signal found.<br>
            <b style='color:#b0c4de;'>How it works:</b> Sends the image to Winston AI's <code style="color:#93c5fd;">POST /v2/image-detection</code> endpoint. Their proprietary model is trained specifically on AI-vs-real image classification at scale.<br>
            <b style='color:#b0c4de;'>Human Score:</b> Winston returns a 0–100 "Human Score". 100 = definitely real. 0 = definitely AI. We invert this to an AI probability for blending.<br>
            <b style='color:#b0c4de;'>Score blending:</b> Final score = Winston AI × 75% + Stage 1 EXIF × 25%. Consistently outperforms either signal alone.<br>
            <b style='color:#b0c4de;'>Cost:</b> 300 credits per call (~$0.02). Only runs on images that Stage 1 cannot conclusively resolve.
        </div>
    </div>

    <div class='stage-card done-red'>
        <div class='stage-badge'><span class='stage-dot red'></span> Stage 3</div>
        <div class='stage-title'>Confidence Router — Auto or Human decision</div>
        <div class='stage-desc'>Applies threshold logic. Every decision is logged to an append-only audit trail.</div>
        <hr class='section-divider'>
        <div style='font-size:13px; color:#8aa0c0; line-height:1.9;'>
            <b style='color:#4ade80;'>✅ Auto Approve (score &lt; 0.30):</b> Strong real-camera signals. No human needed.<br>
            <b style='color:#fbbf24;'>👁 Human Review (0.30–0.85):</b> Gray zone. Flagged for a human to decide. Typically 10–20% of images.<br>
            <b style='color:#f87171;'>❌ Auto Reject (score &gt; 0.85):</b> Strong AI signals. Rejected automatically.<br>
            <b style='color:#b0c4de;'>Cost efficiency:</b> Stage 1 is free on all images. Stage 2 API only runs on ~60–70% of images. Human review only on the gray zone ~10–20%.
        </div>
    </div>

    <div style='background:#0c1120; border:1px solid #1a2540; border-radius:14px; padding:20px 24px; margin-top:8px;'>
        <div style='font-size:13px; font-weight:700; color:#b0c4de; margin-bottom:12px;'>💰 Cost Model (per 1,000 images)</div>
        <div style='font-size:13px; color:#8aa0c0; line-height:2;'>
            C2PA credential found → Pipeline stops at Stage 1. <b style='color:#4ade80;'>API Cost: $0</b><br>
            Strong EXIF (real photo) → Stage 2 score dominated by low Stage 1. Fast auto-approve. <b style='color:#4ade80;'>1 Winston API call (~$0.02)</b><br>
            No EXIF, uncertain → Stage 2 runs, score blended, human review if gray zone. <b style='color:#fbbf24;'>1 Winston call + reviewer time</b><br>
            High-confidence AI → Winston returns high score. Auto reject. <b style='color:#f87171;'>1 Winston call, no human</b><br>
            <b style='color:#b0c4de;'>Blended estimate per 1,000 images: ~$2–$6 total (Stage 1 free on all; Stage 2 only on gray-zone ~60–70%)</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
