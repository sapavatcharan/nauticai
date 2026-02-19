"""
NautiCAI - Autonomous Subsea Inspection Platform
Enhancing Underwater Awareness for Maritime Safety
v3.0 - Smart Detection Engine
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import os
from ultralytics import YOLO
from underwater_augment import apply_full_underwater_simulation
from report_gen import generate_report

st.set_page_config(
    page_title="NautiCAI — Explore Safer Seas",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;600;700;900&family=Inter:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html, body, .stApp { background: #010913 !important; font-family: 'Space Grotesk', sans-serif; color: #B8CDD8; }
.stApp::before { content: ''; position: fixed; inset: 0; background: radial-gradient(ellipse 120% 60% at 0% 100%, rgba(0,210,180,0.06) 0%, transparent 55%), radial-gradient(ellipse 80% 60% at 100% 0%, rgba(0,100,255,0.07) 0%, transparent 55%); pointer-events: none; z-index: 0; }
.stApp::after { content: ''; position: fixed; inset: 0; background-image: linear-gradient(rgba(0,180,160,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(0,180,160,0.025) 1px, transparent 1px); background-size: 60px 60px; pointer-events: none; z-index: 0; }
section[data-testid="stSidebar"] { background: linear-gradient(160deg, #010D1E 0%, #010913 100%) !important; border-right: 1px solid rgba(0,210,180,0.1) !important; min-width: 280px !important; max-width: 280px !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
.brand-block { padding: 28px 20px 24px; background: linear-gradient(135deg, rgba(0,210,180,0.08), rgba(0,100,255,0.06)); border-bottom: 1px solid rgba(0,210,180,0.1); position: relative; overflow: hidden; }
.brand-name { font-family: 'Orbitron', monospace; font-size: 1.5rem; font-weight: 900; letter-spacing: 3px; background: linear-gradient(135deg, #00D4B4, #0088FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.brand-tagline { font-size: 0.62rem; letter-spacing: 2.5px; text-transform: uppercase; color: rgba(0,210,180,0.5); margin-top: 6px; }
.brand-version { position: absolute; top: 12px; right: 16px; font-size: 0.6rem; color: rgba(0,210,180,0.35); letter-spacing: 2px; font-family: 'Orbitron', monospace; }
.sidebar-section { padding: 16px 16px 8px; font-size: 0.6rem; letter-spacing: 3px; text-transform: uppercase; color: rgba(0,210,180,0.4); font-weight: 600; border-top: 1px solid rgba(0,210,180,0.06); margin-top: 8px; }
section[data-testid="stSidebar"] label { color: #5A7A90 !important; font-size: 0.72rem !important; letter-spacing: 1px !important; font-weight: 500 !important; }
.hero { padding: 50px 0 36px; text-align: center; }
.hero-tag { display: inline-block; font-size: 0.62rem; letter-spacing: 5px; text-transform: uppercase; color: rgba(0,210,180,0.7); border: 1px solid rgba(0,210,180,0.2); border-radius: 100px; padding: 5px 18px; margin-bottom: 20px; background: rgba(0,210,180,0.04); font-family: 'Orbitron', monospace; }
.hero-h1 { font-family: 'Orbitron', monospace; font-size: 4rem; font-weight: 900; line-height: 1; color: #FFFFFF; text-shadow: 0 0 60px rgba(0,210,180,0.15); margin-bottom: 6px; }
.hero-h1 .accent { background: linear-gradient(135deg, #00D4B4 0%, #00AAFF 50%, #0066FF 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero-sub { font-size: 0.9rem; color: #3A5A70; max-width: 520px; margin: 14px auto 0; line-height: 1.7; font-weight: 300; }
.status-row { display: flex; justify-content: center; gap: 10px; margin: 24px 0 4px; flex-wrap: wrap; }
.s-pill { display: inline-flex; align-items: center; gap: 7px; padding: 7px 16px; border-radius: 100px; font-size: 0.68rem; letter-spacing: 1px; font-weight: 500; border: 1px solid rgba(0,210,180,0.15); background: rgba(0,210,180,0.04); color: rgba(0,210,180,0.8); text-transform: uppercase; }
.s-dot { width: 5px; height: 5px; border-radius: 50%; background: #00D4B4; box-shadow: 0 0 6px #00D4B4; animation: blink 2s infinite; }
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 1px solid rgba(0,210,180,0.1) !important; gap: 0 !important; padding: 0 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #2A4A60 !important; font-size: 0.75rem !important; font-weight: 600 !important; letter-spacing: 2px !important; text-transform: uppercase !important; padding: 16px 28px !important; border: none !important; border-bottom: 2px solid transparent !important; transition: all 0.3s !important; }
.stTabs [aria-selected="true"] { color: #00D4B4 !important; border-bottom: 2px solid #00D4B4 !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 28px 0 !important; }
.stFileUploader > div { background: rgba(0,210,180,0.02) !important; border: 1.5px dashed rgba(0,210,180,0.2) !important; border-radius: 16px !important; }
.sec-label { font-size: 0.62rem; letter-spacing: 4px; text-transform: uppercase; color: rgba(0,210,180,0.55); font-weight: 600; margin-bottom: 10px; display: flex; align-items: center; gap: 10px; }
.sec-label::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(0,210,180,0.2), transparent); }
.det-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px,1fr)); gap: 12px; margin-top: 16px; }
.det-card { background: linear-gradient(160deg, rgba(8,20,38,0.95), rgba(4,12,24,0.98)); border-radius: 16px; padding: 20px 16px; text-align: center; border: 1px solid rgba(255,255,255,0.04); position: relative; overflow: hidden; transition: all 0.4s; }
.det-card:hover { transform: translateY(-4px) scale(1.02); }
.det-card.c { border-top: 2px solid #E63946; }
.det-card.w { border-top: 2px solid #F4A261; }
.det-card.n { border-top: 2px solid #00D4B4; }
.det-icon { font-size: 1.8rem; margin-bottom: 8px; display: block; }
.det-name { font-size: 0.8rem; font-weight: 600; color: #E8F0F8; margin-bottom: 10px; }
.det-pct { font-family: 'Orbitron', monospace; font-size: 1.8rem; font-weight: 700; color: white; line-height: 1; }
.det-sub { font-size: 0.58rem; letter-spacing: 2px; color: #2A4A60; text-transform: uppercase; margin-top: 4px; }
.det-badge { display: inline-block; font-size: 0.58rem; letter-spacing: 2px; font-weight: 700; text-transform: uppercase; padding: 3px 10px; border-radius: 100px; margin-bottom: 10px; }
.b-c { background: rgba(230,57,70,0.12); color: #E63946; border: 1px solid rgba(230,57,70,0.25); }
.b-w { background: rgba(244,162,97,0.12); color: #F4A261; border: 1px solid rgba(244,162,97,0.25); }
.b-n { background: rgba(0,212,180,0.12); color: #00D4B4; border: 1px solid rgba(0,212,180,0.25); }
.metric-strip { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin: 8px 0 24px; }
.m-card { background: rgba(8,20,38,0.8); border: 1px solid rgba(255,255,255,0.04); border-radius: 14px; padding: 22px 16px 18px; text-align: center; position: relative; overflow: hidden; transition: all 0.3s; }
.m-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 14px 14px 0 0; }
.m-card.mt::before { background: linear-gradient(90deg, #00D4B4, #0088FF); }
.m-card.mc::before { background: linear-gradient(90deg, #E63946, #FF6B6B); }
.m-card.mw::before { background: linear-gradient(90deg, #F4A261, #FFD166); }
.m-card.mn::before { background: linear-gradient(90deg, #00D4B4, #00B4A0); }
.m-val { font-family: 'Orbitron', monospace; font-size: 2.6rem; font-weight: 700; line-height: 1; margin-bottom: 6px; }
.m-card.mt .m-val { color: #E8F4FF; }
.m-card.mc .m-val { color: #E63946; }
.m-card.mw .m-val { color: #F4A261; }
.m-card.mn .m-val { color: #00D4B4; }
.m-lbl { font-size: 0.6rem; letter-spacing: 2.5px; text-transform: uppercase; color: #2A4A60; font-weight: 600; }
.log-item { display: flex; align-items: center; justify-content: space-between; padding: 13px 18px; border-radius: 10px; background: rgba(8,20,38,0.6); border: 1px solid rgba(255,255,255,0.03); margin-bottom: 5px; }
.log-name { font-size: 0.85rem; font-weight: 600; color: #C8D8E8; }
.timeline-item { display: flex; gap: 14px; padding: 12px 0; border-bottom: 1px solid rgba(0,210,180,0.05); align-items: flex-start; }
.timeline-dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 5px; flex-shrink: 0; }
.timeline-dot.c { background: #E63946; box-shadow: 0 0 8px rgba(230,57,70,0.5); }
.timeline-dot.w { background: #F4A261; box-shadow: 0 0 8px rgba(244,162,97,0.5); }
.timeline-dot.n { background: #00D4B4; box-shadow: 0 0 8px rgba(0,212,180,0.5); }
.timeline-title { font-size: 0.82rem; font-weight: 600; color: #C8D8E8; }
.timeline-meta { font-size: 0.68rem; color: #2A4A60; margin-top: 2px; }
.timeline-ts { font-family: 'Orbitron', monospace; font-size: 0.65rem; color: #1A3A50; white-space: nowrap; }
.img-wrap { border-radius: 14px; overflow: hidden; border: 1px solid rgba(0,210,180,0.1); box-shadow: 0 20px 60px rgba(0,0,0,0.4); }
.stButton > button { background: linear-gradient(135deg, #00C4A8, #0088CC) !important; color: white !important; border: none !important; border-radius: 10px !important; padding: 13px 28px !important; font-size: 0.75rem !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; width: 100% !important; transition: all 0.3s !important; }
.stDownloadButton > button { background: linear-gradient(135deg, #0055AA, #003388) !important; color: white !important; border: none !important; border-radius: 10px !important; width: 100% !important; font-weight: 700 !important; letter-spacing: 2px !important; text-transform: uppercase !important; font-size: 0.75rem !important; }
.stTextInput input { background: rgba(8,20,38,0.8) !important; border: 1px solid rgba(0,210,180,0.12) !important; border-radius: 10px !important; color: #C8D8E8 !important; font-size: 0.85rem !important; }
.stProgress > div > div { background: linear-gradient(90deg, #00D4B4, #0088FF) !important; border-radius: 100px !important; }
[data-testid="metric-container"] { background: rgba(8,20,38,0.7) !important; border: 1px solid rgba(0,210,180,0.08) !important; border-radius: 12px !important; padding: 14px 16px !important; }
[data-testid="metric-container"] label { color: #2A4A60 !important; font-size: 0.7rem !important; letter-spacing: 1.5px !important; text-transform: uppercase !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #E8F4FF !important; font-family: 'Orbitron', monospace !important; font-size: 1.6rem !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: rgba(0,210,180,0.2); border-radius: 2px; }
.footer { text-align: center; padding: 40px 0 20px; margin-top: 50px; border-top: 1px solid rgba(0,210,180,0.06); }
.footer-brand { font-family: 'Orbitron', monospace; font-size: 0.7rem; letter-spacing: 4px; color: rgba(0,210,180,0.3); margin-bottom: 6px; }
.footer-text { font-size: 0.65rem; letter-spacing: 2.5px; text-transform: uppercase; color: #1A3A50; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding-top: 0 !important; max-width: 1200px !important; }
.stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
SEVERITY = {
    'corrosion':     ('CRITICAL', 'c', 'b-c'),
    'damage':        ('CRITICAL', 'c', 'b-c'),
    'free_span':     ('CRITICAL', 'c', 'b-c'),
    'marine_growth': ('WARNING',  'w', 'b-w'),
    'debris':        ('WARNING',  'w', 'b-w'),
    'healthy':       ('NORMAL',   'n', 'b-n'),
    'anode':         ('NORMAL',   'n', 'b-n'),
}
ICONS = {
    'corrosion': '⚠️', 'damage': '🔧', 'free_span': '🔴',
    'marine_growth': '🌿', 'debris': '🗑️', 'healthy': '✅', 'anode': '🔋'
}

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "weights", "best.pt")

@st.cache_resource
def load_model(p):
    return YOLO(p) if os.path.exists(p) else YOLO('yolov8n.pt')

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [('anomaly_log', []), ('det_counts', {}), ('last_img_id', None), ('pdf_bytes', None)]:
    if k not in st.session_state:
        st.session_state[k] = v

model   = load_model(model_path)
m_label = "Custom YOLOv8s" if os.path.exists(model_path) else "YOLOv8n Baseline"

# ── Smart log helper ──────────────────────────────────────────────────────────
DIFF_THRESHOLD = 0.30  # 30% gap = different instance

def smart_log(cn, cf, ts, frame_bytes, class_tracker):
    if cn not in class_tracker:
        st.session_state.anomaly_log.append({
            "class_name":  cn, "confidence": cf,
            "timestamp": ts, "frame_bytes": frame_bytes
        })
        st.session_state.det_counts[cn] = st.session_state.det_counts.get(cn, 0) + 1
        class_tracker[cn] = [cf]
        return True
    else:
        logged_confs = class_tracker[cn]
        if all(abs(cf - prev) >= DIFF_THRESHOLD for prev in logged_confs):
            st.session_state.anomaly_log.append({
                "class_name": cn, "confidence": cf,
                "timestamp": ts, "frame_bytes": frame_bytes
            })
            st.session_state.det_counts[cn] = st.session_state.det_counts.get(cn, 0) + 1
            class_tracker[cn].append(cf)
            return True
    return False

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand-block">
        <div class="brand-version">v3.0</div>
        <div class="brand-name">NautiCAI</div>
        <div class="brand-tagline">Explore Safer Seas Now</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Detection</div>', unsafe_allow_html=True)
    conf = st.slider("Confidence Threshold", 0.10, 1.0, 0.25, 0.05)

    st.markdown('<div class="sidebar-section">Environment</div>', unsafe_allow_html=True)
    sim_on = st.toggle("Underwater Simulation", False)
    if sim_on:
        turb = st.select_slider("Turbidity Level", ['low', 'medium', 'high'], 'medium')
        snow = st.checkbox("Marine Snow Particles", True)
    else:
        turb, snow = 'medium', False

    st.markdown('<div class="sidebar-section">Mission</div>', unsafe_allow_html=True)
    m_name = st.text_input("Name",     "Subsea Inspection Mission")
    m_op   = st.text_input("Operator", "NautiCAI Operator")
    m_rov  = st.text_input("ROV ID",   "ROV-NautiCAI-01")
    m_loc  = st.text_input("Location", "Offshore Location")

    st.divider()
    c1, c2 = st.columns(2)
    total_n    = len(st.session_state.anomaly_log)
    critical_n = sum(1 for x in st.session_state.anomaly_log if x['class_name'] in ['corrosion', 'damage', 'free_span'])
    c1.metric("Detections", total_n)
    c2.metric("🔴 Critical", critical_n)

    if st.button("⟳  Reset Session"):
        st.session_state.anomaly_log = []
        st.session_state.det_counts  = {}
        st.session_state.last_img_id = None
        st.session_state.pdf_bytes   = None
        st.rerun()

    st.success("✅ Custom YOLOv8s loaded") if os.path.exists(model_path) else st.warning("⚠️ Using YOLOv8n baseline")

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-tag">AI Intelligence Layer · Maritime Safety</div>
    <div class="hero-h1">Nauti<span class="accent">CAI</span></div>
    <div class="hero-sub">Preventing underwater hazards for safer ports, coasts and oceans — real-time AI detection for ROV &amp; AUV fleets</div>
    <div class="status-row">
        <span class="s-pill"><span class="s-dot"></span>System Online</span>
        <span class="s-pill"><span class="s-dot"></span>{m_label}</span>
        <span class="s-pill"><span class="s-dot"></span>Ports · Coasts · Offshore</span>
        <span class="s-pill"><span class="s-dot"></span>Digital Twin Foundation</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸   Image Detection", "🎥   Video Analysis", "📊   Mission Report"])

# ── TAB 1: IMAGE ─────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="sec-label">Upload Underwater Image</div>', unsafe_allow_html=True)
    img_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")

    if img_file:
        is_new = (img_file.file_id != st.session_state.last_img_id)
        raw    = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img    = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        proc   = apply_full_underwater_simulation(img, turb, snow) if sim_on else img.copy()

        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown('<div class="sec-label">Original Feed</div>', unsafe_allow_html=True)
            st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.spinner("Running YOLOv8 inference..."):
            res = model.predict(proc, conf=conf, verbose=False)
            ann = res[0].plot()

        with col2:
            st.markdown('<div class="sec-label">AI Detection Output</div>', unsafe_allow_html=True)
            st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
            st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        boxes = res[0].boxes
        if boxes is not None and len(boxes) > 0:
            st.markdown('<br><div class="sec-label">Detections</div>', unsafe_allow_html=True)
            cards_html  = '<div class="det-grid">'
            ann_rgb     = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
            _, buf      = cv2.imencode('.jpg', ann_rgb)
            frame_bytes = buf.tobytes()

            for box in boxes:
                cls_id   = int(box.cls[0])
                cls_name = model.names[cls_id]
                pct      = float(box.conf[0])
                sev, card_cls, badge_cls = SEVERITY.get(cls_name, ('WARNING', 'w', 'b-w'))
                icon = ICONS.get(cls_name, '🔍')
                cards_html += f"""<div class="det-card {card_cls}">
                    <span class="det-badge {badge_cls}">{sev}</span>
                    <span class="det-icon">{icon}</span>
                    <div class="det-name">{cls_name.replace('_', ' ').title()}</div>
                    <div class="det-pct">{pct:.0%}</div>
                    <div class="det-sub">Confidence</div>
                </div>"""
                if is_new:
                    st.session_state.det_counts[cls_name] = st.session_state.det_counts.get(cls_name, 0) + 1

            if is_new:
                img_tracker = {}
                ts = time.strftime('%H:%M:%S')
                for box in boxes:
                    cn = model.names[int(box.cls[0])]
                    cf = float(box.conf[0])
                    smart_log(cn, cf, ts, frame_bytes, img_tracker)
                st.session_state.last_img_id = img_file.file_id
                st.session_state.pdf_bytes   = None

            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)
        else:
            st.success("✅ No anomalies detected — surface appears healthy.")

# ── TAB 2: VIDEO ─────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="sec-label">Upload Mission Video</div>', unsafe_allow_html=True)
    vid_file = st.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'], label_visibility="collapsed")

    if vid_file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(vid_file.read())
        tfile.flush()
        cap    = cv2.VideoCapture(tfile.name)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps    = max(cap.get(cv2.CAP_PROP_FPS), 1)
        dur    = frames / fps

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Frames", f"{frames:,}")
        c2.metric("Frame Rate",   f"{fps:.1f} FPS")
        c3.metric("Duration",     f"{dur:.1f}s")
        c4.metric("Classes",      f"{len(model.names)}")

        ca, cb = st.columns(2)
        with ca: skip = st.slider("Process every N frames", 1, 15, 8)
        with cb: maxf = st.slider("Max frames to scan", 10, 200, 60)

        if st.button("▶   Start Video Analysis"):
            placeholder = st.empty()
            prog        = st.progress(0)
            status_box  = st.empty()
            live_log    = st.empty()

            fc = pc = 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Smart tracker: {class_name: best_confidence}
            # Each class is logged ONCE with its best confidence frame
            # If same class found again with much higher confidence → UPGRADE the snapshot
            class_tracker = {}  # {class_name: [list of logged confidences]}

            while cap.isOpened() and pc < maxf:
                ret, frame = cap.read()
                if not ret:
                    break
                fc += 1
                if fc % skip != 0:
                    continue

                if sim_on:
                    frame = apply_full_underwater_simulation(frame, turb, snow)

                res         = model.predict(frame, conf=conf, verbose=False)
                ann         = res[0].plot()
                current_sec = fc / fps

                placeholder.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), use_container_width=True)

                if res[0].boxes and len(res[0].boxes) > 0:
                    # Encode frame once per detection cycle
                    ann_rgb     = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
                    _, buf      = cv2.imencode('.jpg', ann_rgb)
                    frame_bytes = buf.tobytes()
                    mm, ss      = int(current_sec // 60), int(current_sec % 60)
                    ts          = f"{mm:02d}:{ss:02d}"

                    # Smart log — same class skipped unless 30%+ confidence difference
                    for box in res[0].boxes:
                        cn = model.names[int(box.cls[0])]
                        cf = float(box.conf[0])
                        smart_log(cn, cf, ts, frame_bytes, class_tracker)

                pc += 1
                prog.progress(min(pc / maxf, 1.0))

                # Live status bar
                status_box.markdown(
                    f"<small style='color:#2A4A60;letter-spacing:1px'>"
                    f"SCANNING {pc}/{maxf} FRAMES · "
                    f"UNIQUE ANOMALIES FOUND: {len(class_tracker)} · "
                    f"TIMESTAMP: {int(current_sec//60):02d}:{int(current_sec%60):02d}"
                    f"</small>",
                    unsafe_allow_html=True
                )

                # Live badges of found anomalies
                if class_tracker:
                    log_html = "<div style='display:flex;flex-wrap:wrap;gap:8px;margin-top:8px;'>"
                    for cls, confs in class_tracker.items():
                        sev, _, badge = SEVERITY.get(cls, ('WARNING', 'w', 'b-w'))
                        icon = ICONS.get(cls, '🔍')
                        log_html += f"<span class='det-badge {badge}'>{icon} {cls.replace("_"," ").title()} ×{len(confs)}</span>"
                    log_html += "</div>"
                    live_log.markdown(log_html, unsafe_allow_html=True)

            cap.release()
            try:
                os.unlink(tfile.name)
            except:
                pass

            st.session_state.pdf_bytes = None

            # Final summary
            unique = len(class_tracker)
            crit   = sum(1 for c in class_tracker if SEVERITY.get(c, ('W',))[0] == 'CRITICAL')
            warn   = sum(1 for c in class_tracker if SEVERITY.get(c, ('W',))[0] == 'WARNING')
            norm   = sum(1 for c in class_tracker if SEVERITY.get(c, ('N',))[0] == 'NORMAL')

            if unique == 0:
                st.success("✅ Scan complete — No anomalies detected. Structure appears healthy.")
            else:
                parts = [f"🔍 **{unique} unique anomaly type(s) found**"]
                if crit: parts.append(f"🔴 **{crit} CRITICAL**")
                if warn: parts.append(f"🟡 **{warn} WARNING**")
                if norm: parts.append(f"🟢 **{norm} NORMAL**")
                msg = " · ".join(parts)
                st.error(msg) if crit else st.warning(msg)

# ── TAB 3: MISSION REPORT ────────────────────────────────────────────────────
with tab3:
    log      = st.session_state.anomaly_log
    total    = len(log)
    critical = sum(1 for x in log if x['class_name'] in ['corrosion', 'damage', 'free_span'])
    warnings = sum(1 for x in log if x['class_name'] in ['marine_growth', 'debris'])
    normal   = sum(1 for x in log if x['class_name'] in ['healthy', 'anode'])

    st.markdown(f"""
    <div class="metric-strip">
        <div class="m-card mt"><div class="m-val">{total}</div><div class="m-lbl">Unique Anomalies</div></div>
        <div class="m-card mc"><div class="m-val">{critical}</div><div class="m-lbl">Critical</div></div>
        <div class="m-card mw"><div class="m-val">{warnings}</div><div class="m-lbl">Warnings</div></div>
        <div class="m-card mn"><div class="m-val">{normal}</div><div class="m-lbl">Normal</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.det_counts:
        col_a, col_b = st.columns([1, 1], gap="large")

        with col_a:
            st.markdown('<div class="sec-label">Breakdown by Class</div>', unsafe_allow_html=True)
            for cls, cnt in sorted(st.session_state.det_counts.items(), key=lambda x: -x[1]):
                sev, _, badge = SEVERITY.get(cls, ('WARNING', 'w', 'b-w'))
                icon = ICONS.get(cls, '🔍')
                pct  = (cnt / max(total, 1)) * 100
                st.markdown(f"""<div class="log-item">
                    <span class="log-name">{icon}&nbsp; {cls.replace('_',' ').title()}</span>
                    <span style='font-size:0.7rem;color:#2A4A60'>{pct:.1f}%</span>
                    <span class="det-badge {badge}" style="margin:0">{sev}</span>
                </div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="sec-label">Detection Timeline</div>', unsafe_allow_html=True)
            for item in log:
                cn   = item['class_name']
                sev, card_cls, badge = SEVERITY.get(cn, ('WARNING', 'w', 'b-w'))
                icon = ICONS.get(cn, '🔍')
                st.markdown(f"""
                <div class="timeline-item">
                    <div class="timeline-dot {card_cls}"></div>
                    <div style="flex:1">
                        <div class="timeline-title">{icon} {cn.replace('_',' ').title()}</div>
                        <div class="timeline-meta">{item['confidence']:.0%} confidence &nbsp;
                            <span class="det-badge {badge}" style="padding:1px 7px;font-size:0.55rem">{sev}</span>
                        </div>
                    </div>
                    <div class="timeline-ts">{item['timestamp']}</div>
                </div>
                """, unsafe_allow_html=True)

    # Best snapshots - one per anomaly type
    if log:
        st.markdown('<br><div class="sec-label">Best Snapshot Per Anomaly Type</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, item in enumerate(log):
            with cols[i % 3]:
                if item.get('frame_bytes'):
                    frame_arr = np.frombuffer(item['frame_bytes'], dtype=np.uint8)
                    frame_img = cv2.imdecode(frame_arr, cv2.IMREAD_COLOR)
                    st.markdown('<div class="img-wrap">', unsafe_allow_html=True)
                    st.image(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                icon = ICONS.get(item['class_name'], '🔍')
                sev, _, badge = SEVERITY.get(item['class_name'], ('WARNING', 'w', 'b-w'))
                st.markdown(
                    f"<div style='text-align:center;font-size:0.68rem;color:#2A4A60;margin-top:6px;letter-spacing:1px'>"
                    f"{icon} <b style='color:#8AAABB'>{item['class_name'].replace('_',' ').title()}</b>"
                    f" · <span style='font-family:Orbitron,monospace'>{item['confidence']:.0%}</span>"
                    f" · {item['timestamp']}</div>",
                    unsafe_allow_html=True
                )

    # Generate PDF
    st.markdown('<br><div class="sec-label">Generate Report</div>', unsafe_allow_html=True)
    if st.button("📄   Generate PDF Inspection Report"):
        if not log:
            st.warning("Run detection on an image or video first.")
        else:
            with st.spinner("Compiling inspection report..."):
                st.session_state.pdf_bytes = generate_report(
                    anomaly_log=log,
                    mission_name=m_name,
                    operator_name=m_op,
                    vessel_id=m_rov,
                    location=m_loc
                )

    if st.session_state.get('pdf_bytes'):
        st.download_button(
            "⬇️   Download PDF Report",
            st.session_state.pdf_bytes,
            f"nauticai_{time.strftime('%Y%m%d_%H%M%S')}.pdf",
            "application/pdf"
        )
        st.success("✅ Report ready — one page per unique anomaly type, best frame captured!")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-brand">NAUTICAI</div>
    <div class="footer-text">Explore Safer Seas Now · www.nauticai-ai.com · Enhancing Underwater Awareness for Maritime Safety</div>
</div>
""", unsafe_allow_html=True)