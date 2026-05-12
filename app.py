"""
TrustGuard — Fraud Analyst Dashboard
=====================================
Streamlit Cloud deployment for the AI Fraud Detection System.
Track A — Deliverable 3

IMPORTANT: This file does NOT modify any existing project file.
It imports from rag_module.py and loads saved artefacts from outputs/.
"""

import os, sys, json, time, io, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
from PIL import Image

warnings.filterwarnings("ignore")

# ── Resolve project root so relative imports work on Streamlit Cloud ──────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Import and patch rag_module AFTER ROOT is defined ─────────────────────────
import rag_module as _rm
_rm.RAG_CONFIG["CHROMA_DB_PATH"] = os.path.join(ROOT, "chroma_db")

def _patch_rag_module():
    return _rm

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustGuard AI · Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Cybersecurity / Hacker Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&family=Orbitron:wght@400;600;700;900&display=swap');

/* ── GLOBAL RESET ── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #080c10 !important;
    color: #c8d8e8 !important;
}
.main { background-color: #080c10 !important; }
.block-container { padding-top: 0 !important; max-width: 100% !important; }

/* ── HIDE DEFAULT STREAMLIT ELEMENTS ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #080c10; }
::-webkit-scrollbar-thumb { background: #00ff41; border-radius: 2px; }

/* ── TOP NAV BAR ── */
.tg-navbar {
    position: sticky;
    top: 0;
    z-index: 9999;
    background: rgba(8,12,16,0.97);
    border-bottom: 1px solid #00ff4133;
    padding: 0 2rem;
    display: flex;
    align-items: center;
    gap: 0;
    height: 58px;
    backdrop-filter: blur(12px);
    margin-bottom: 0;
}
.tg-navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-right: 2.5rem;
    flex-shrink: 0;
}
.tg-navbar-logo {
    font-family: 'Orbitron', monospace;
    font-size: 1.15rem;
    font-weight: 900;
    color: #00ff41 !important;
    letter-spacing: 2px;
    text-shadow: 0 0 20px #00ff4166;
    white-space: nowrap;
}
.tg-navbar-version {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #4a7a5a;
    letter-spacing: 1px;
    margin-top: 2px;
}
.tg-nav-links {
    display: flex;
    align-items: center;
    gap: 2px;
    flex: 1;
}
.tg-nav-btn {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #6a8a7a;
    padding: 6px 14px;
    border: 1px solid transparent;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.18s ease;
    white-space: nowrap;
    background: none;
    text-decoration: none;
    display: inline-block;
}
.tg-nav-btn:hover { color: #00ff41; border-color: #00ff4133; background: #00ff4108; }
.tg-nav-btn.active {
    color: #00ff41;
    border-color: #00ff4155;
    background: #00ff4112;
    text-shadow: 0 0 8px #00ff4144;
}
.tg-navbar-status {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
    flex-shrink: 0;
}
.tg-status-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #00ff41;
    animation: pulse-green 2s infinite;
}
@keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 #00ff4166; opacity: 1; }
    50% { box-shadow: 0 0 0 5px #00ff4100; opacity: 0.7; }
}
.tg-status-text {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #00ff41;
    letter-spacing: 1px;
}
.tg-threshold-inline {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #ff6b35;
    border: 1px solid #ff6b3533;
    padding: 3px 10px;
    border-radius: 3px;
    background: #ff6b3508;
    margin-left: 16px;
}

/* ── PAGE WRAPPER ── */
.tg-page { padding: 1.8rem 2rem 3rem 2rem; }

/* ── SECTION HEADERS ── */
.tg-page-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e8f4e8;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 0.2rem;
    text-shadow: 0 0 30px #00ff4122;
}
.tg-page-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #3a6a4a;
    letter-spacing: 1px;
    margin-bottom: 1.8rem;
}
.tg-section-title {
    font-family: 'Orbitron', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    color: #00ff41;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding-bottom: 8px;
    margin-bottom: 14px;
    border-bottom: 1px solid #00ff4122;
}
.tg-divider {
    border: none;
    border-top: 1px solid #0f1f14;
    margin: 1.5rem 0;
}

/* ── CARDS ── */
.tg-card {
    background: #0a0f0d;
    border: 1px solid #1a3020;
    border-radius: 6px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.tg-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ff4133, transparent);
}
.tg-card-danger {
    border-color: #ff000033;
    background: #0f0a0a;
}
.tg-card-danger::before { background: linear-gradient(90deg, transparent, #ff000055, transparent); }
.tg-card-success {
    border-color: #00ff4133;
    background: #0a0f0a;
}
.tg-card-success::before { background: linear-gradient(90deg, transparent, #00ff4155, transparent); }
.tg-card-warning {
    border-color: #ff6b3533;
    background: #0f0d0a;
}
.tg-card-warning::before { background: linear-gradient(90deg, transparent, #ff6b3555, transparent); }
.tg-card-info {
    border-color: #00aaff22;
    background: #080c10;
}

/* ── METRIC CARDS ── */
.tg-metric {
    background: #080c10;
    border: 1px solid #1a2a1a;
    border-radius: 6px;
    padding: 1.4rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.tg-metric:hover { border-color: #00ff4133; }
.tg-metric-val {
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 6px;
}
.tg-metric-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #3a5a4a;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.tg-metric-sub {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: #2a4a3a;
    margin-top: 4px;
}

/* Metric colors */
.val-green  { color: #00ff41; text-shadow: 0 0 20px #00ff4155; }
.val-red    { color: #ff2020; text-shadow: 0 0 20px #ff202055; }
.val-orange { color: #ff6b35; text-shadow: 0 0 20px #ff6b3555; }
.val-cyan   { color: #00aaff; text-shadow: 0 0 20px #00aaff55; }
.val-white  { color: #e8f4e8; }

/* ── RISK BADGES ── */
.badge {
    display: inline-block;
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 12px;
    border-radius: 3px;
    text-transform: uppercase;
}
.badge-critical { background: #ff000020; color: #ff2020; border: 1px solid #ff000055; }
.badge-high     { background: #ff6b3520; color: #ff6b35; border: 1px solid #ff6b3555; }
.badge-medium   { background: #ffaa0020; color: #ffaa00; border: 1px solid #ffaa0055; }
.badge-low      { background: #00ff4120; color: #00ff41; border: 1px solid #00ff4155; }

/* ── SCANLINE EFFECT (decorative) ── */
.tg-scanline {
    position: relative;
}
.tg-scanline::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,255,65,0.015) 2px, rgba(0,255,65,0.015) 4px
    );
    pointer-events: none;
    border-radius: inherit;
}

/* ── TERMINAL BOX ── */
.tg-terminal {
    background: #040806;
    border: 1px solid #00ff4122;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #00ff41;
    line-height: 1.7;
    position: relative;
}
.tg-terminal-header {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #00ff4115;
}
.tg-terminal-dot { width: 8px; height: 8px; border-radius: 50%; }
.dot-red   { background: #ff5f57; }
.dot-amber { background: #ffbd2e; }
.dot-green { background: #28c840; }
.tg-terminal-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    color: #3a6a4a;
    letter-spacing: 2px;
    margin-left: 6px;
}
.tg-prompt { color: #00ff41; }
.tg-prompt-dim { color: #2a5a3a; }

/* ── FEATURE TABLE ── */
.tg-feature-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #0f1f14;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
}
.tg-feature-name { color: #4a7a5a; }
.tg-feature-val  { color: #00ff41; }

/* ── PROBABILITY BAR ── */
.tg-prob-bar-wrap { margin: 1rem 0; }
.tg-prob-bar-bg {
    background: #0f1a12;
    border: 1px solid #1a3020;
    border-radius: 3px;
    height: 10px;
    position: relative;
    overflow: hidden;
}
.tg-prob-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.6s ease;
    position: relative;
}
.tg-prob-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 3px;
    background: rgba(255,255,255,0.4);
}
.tg-prob-labels {
    display: flex;
    justify-content: space-between;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.62rem;
    color: #2a4a3a;
    margin-top: 4px;
}

/* ── STREAMLIT WIDGET OVERRIDES ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > div {
    background: #040806 !important;
    border: 1px solid #1a3020 !important;
    border-radius: 4px !important;
    color: #00ff41 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #00ff4155 !important;
    box-shadow: 0 0 0 2px #00ff4115 !important;
}
.stSelectbox > div > div { background: #040806 !important; }
label, .stSelectbox label, .stNumberInput label, .stTextInput label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #3a6a4a !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
.stSlider > div > div > div { background: #00ff4133 !important; }
.stSlider [data-baseweb="slider"] { background: #0f1a12 !important; }
.stSlider .st-bx { color: #00ff41 !important; }

/* ── BUTTONS ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid #00ff4155 !important;
    color: #00ff41 !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border-radius: 4px !important;
    padding: 0.55rem 1.4rem !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: #00ff4115 !important;
    border-color: #00ff41 !important;
    box-shadow: 0 0 15px #00ff4130 !important;
}
.stButton > button[kind="primary"] {
    background: #00ff4115 !important;
    border-color: #00ff41 !important;
    box-shadow: 0 0 20px #00ff4125 !important;
}
.stDownloadButton > button {
    background: transparent !important;
    border: 1px solid #00aaff44 !important;
    color: #00aaff !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    border-radius: 4px !important;
}
.stDownloadButton > button:hover {
    background: #00aaff10 !important;
    border-color: #00aaff !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: #040806 !important;
    border: 1px dashed #1a3020 !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #00ff4144 !important; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #1a3020 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #3a6a4a !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
    padding: 8px 18px !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #00ff41 !important;
    border-bottom-color: #00ff41 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding-top: 1.2rem !important;
}

/* ── EXPANDER ── */
.streamlit-expanderHeader {
    background: #040806 !important;
    border: 1px solid #1a3020 !important;
    border-radius: 4px !important;
    color: #3a6a4a !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
}
.streamlit-expanderContent {
    background: #040806 !important;
    border: 1px solid #1a3020 !important;
    border-top: none !important;
}

/* ── DATAFRAME / TABLE ── */
[data-testid="stDataFrame"] {
    border: 1px solid #1a3020 !important;
    border-radius: 6px !important;
}
.dataframe { background: #040806 !important; color: #c8d8e8 !important; }

/* ── METRICS (native st.metric) ── */
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    color: #00ff41 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #3a6a4a !important;
    font-size: 0.65rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricDelta"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── ALERTS ── */
.stAlert {
    background: #040806 !important;
    border-radius: 4px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}
.stSuccess { border-left: 3px solid #00ff41 !important; }
.stWarning { border-left: 3px solid #ff6b35 !important; }
.stError   { border-left: 3px solid #ff2020 !important; }
.stInfo    { border-left: 3px solid #00aaff !important; }

/* ── CODE BLOCKS ── */
.stCodeBlock { background: #040806 !important; border: 1px solid #1a3020 !important; }
code { color: #00ff41 !important; font-family: 'Share Tech Mono', monospace !important; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: #00ff41 !important; }

/* ── CHECKBOX / RADIO ── */
.stCheckbox label, .stRadio label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #4a7a5a !important;
    letter-spacing: 1px !important;
}

/* ── PROGRESS ── */
.stProgress > div > div > div { background: #00ff41 !important; }
.stProgress > div > div { background: #0f1a12 !important; }

/* ── GRID LINES BACKGROUND ── */
.tg-grid-bg {
    background-image:
        linear-gradient(rgba(0,255,65,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,65,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
}

/* ── GLOW SEPARATOR ── */
.tg-glow-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ff4133, #00ff4166, #00ff4133, transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── API KEY INPUT ── */
.tg-key-input {
    background: #040806;
    border: 1px solid #1a3020;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    margin-bottom: 1rem;
}

/* ── ABOUT CHECKLIST ── */
.tg-check-item {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #4a7a5a;
    padding: 5px 0;
    border-bottom: 1px solid #0f1f14;
    display: flex;
    align-items: center;
    gap: 10px;
}
.tg-check-item .check-icon { color: #00ff41; font-size: 0.9rem; }

/* ── COLUMNS GAP FIX ── */
[data-testid="column"] { padding: 0 0.4rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS / PATHS
# ─────────────────────────────────────────────────────────────────────────────
DEPLOY_DIR   = os.path.join(ROOT, "outputs", "deployment")
MODELS_DIR   = os.path.join(ROOT, "outputs", "models")
METRICS_DIR  = os.path.join(ROOT, "outputs", "metrics")
PLOTS_DIR    = os.path.join(ROOT, "outputs", "plots")
ABLATION_DIR = os.path.join(ROOT, "outputs", "ablation")

FEATURES = ["step","amount","oldbalanceOrg","newbalanceOrig",
            "oldbalanceDest","newbalanceDest","balanceDiff","amount_ratio",
            "type_CASH_OUT","type_DEBIT","type_PAYMENT","type_TRANSFER"]

FEATURE_LABELS = {
    "step": "Time Step", "amount": "Amount (PKR)",
    "oldbalanceOrg": "Sender Old Balance", "newbalanceOrig": "Sender New Balance",
    "oldbalanceDest": "Recipient Old Balance", "newbalanceDest": "Recipient New Balance",
    "balanceDiff": "Balance Difference", "amount_ratio": "Amount Ratio",
    "type_CASH_OUT": "Type: CASH_OUT", "type_DEBIT": "Type: DEBIT",
    "type_PAYMENT": "Type: PAYMENT", "type_TRANSFER": "Type: TRANSFER",
}

RISK_COLORS = {
    "CRITICAL": "#ff2020", "HIGH": "#ff6b35",
    "MEDIUM": "#ffaa00", "LOW": "#00ff41"
}

NAV_PAGES = [
    ("predict",     "⬡ PREDICT",      "Single Transaction"),
    ("batch",       "⬡ BATCH",        "CSV Analysis"),
    ("dataset",     "⬡ DATASET",      "Data & Imbalance"),
    ("performance", "⬡ PERFORMANCE",  "Model Metrics"),
    ("ablation",    "⬡ ABLATION",     "Study"),
    ("about",       "⬡ ABOUT",        "System Info"),
]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "predict"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.5

# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────────────────
nav_html = """
<div class="tg-navbar tg-grid-bg">
    <div class="tg-navbar-brand">
        <div>
            <div class="tg-navbar-logo">🛡 TRUSTGUARD</div>
            <div class="tg-navbar-version">v1.0 · AI FRAUD DETECTION</div>
        </div>
    </div>
    <div class="tg-nav-links">
"""
for page_id, label, _ in NAV_PAGES:
    active_class = "active" if st.session_state.page == page_id else ""
    nav_html += f'<span class="tg-nav-btn {active_class}" onclick="void(0)">{label}</span>'

nav_html += f"""
    </div>
    <div class="tg-navbar-status">
        <span class="tg-threshold-inline">THR: {st.session_state.threshold:.0%}</span>
        <div class="tg-status-dot"></div>
        <span class="tg-status-text">SYSTEM ONLINE</span>
    </div>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# ── Nav buttons (hidden behind the visual nav) ──
nav_cols = st.columns(len(NAV_PAGES))
for i, (page_id, label, _) in enumerate(NAV_PAGES):
    with nav_cols[i]:
        if st.button(label, key=f"nav_{page_id}", use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

page = st.session_state.page

# ── Threshold & API key in a compact control strip below nav ──
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([2, 1, 1])
with ctrl_col1:
    threshold = st.slider(
        "FRAUD THRESHOLD", 0.1, 0.9,
        st.session_state.threshold, 0.01,
        help="Probability cutoff to classify as fraud",
        key="threshold_slider"
    )
    st.session_state.threshold = threshold
with ctrl_col2:
    api_input = st.text_input(
        "OPENAI API KEY",
        type="password",
        placeholder="sk-proj-...",
        key="openai_api_key",
        help="For RAG regulatory justification"
    )
with ctrl_col3:
    if api_input:
        st.success("KEY ACTIVE")
    elif st.session_state.get("openai_api_key", ""):
        st.success("KEY FROM SECRETS ✓")
    else:
        st.warning("NO KEY — RAG DISABLED")

st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="[ LOADING EMBEDDING MODELS... ]")
def load_embedding_models():
    from sentence_transformers import SentenceTransformer, CrossEncoder
    em = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    try:
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except:
        reranker = None
    return em, reranker

embed_model, reranker = load_embedding_models()

@st.cache_resource(show_spinner="[ LOADING MODEL ARTEFACTS... ]")
def load_deployment_model():
    import joblib
    model  = joblib.load(os.path.join(DEPLOY_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(DEPLOY_DIR, "scaler.pkl"))
    with open(os.path.join(DEPLOY_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

import base64
def show_img(path, width="100%"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:{width}; border-radius:4px; border:1px solid #1a3020;">',
            unsafe_allow_html=True
        )
    else:
        st.warning(f"[ MISSING: `{path}` ]")

@st.cache_resource(show_spinner="[ LOADING ALL MODELS... ]")
def load_all_models():
    import joblib
    models = {}
    for name in ["xgboost", "random_forest", "neural_network", "logistic_regression"]:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return models, scaler

@st.cache_data(show_spinner=False)
def load_model_comparison():
    path = os.path.join(METRICS_DIR, "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    return None

@st.cache_data(show_spinner=False)
def load_ablation():
    path = os.path.join(ABLATION_DIR, "ablation_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else None

def get_openai_key():
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key: return key
    except Exception:
        pass
    return st.session_state.get("openai_api_key", "")

def risk_badge(tier: str) -> str:
    return f'<span class="badge badge-{tier.lower()}">{tier}</span>'

def risk_color(tier: str) -> str:
    return RISK_COLORS.get(tier, "#00ff41")

def get_risk_tier(prob: float) -> str:
    if prob >= 0.85: return "CRITICAL"
    if prob >= 0.65: return "HIGH"
    if prob >= 0.50: return "MEDIUM"
    return "LOW"

def mpl_dark_style(fig, ax_list=None):
    """Apply dark theme to matplotlib figures."""
    fig.patch.set_facecolor("#040806")
    axes = ax_list if ax_list else (fig.get_axes() or [])
    for ax in axes:
        ax.set_facecolor("#040806")
        ax.tick_params(colors="#3a6a4a", labelsize=8)
        ax.xaxis.label.set_color("#3a6a4a")
        ax.yaxis.label.set_color("#3a6a4a")
        ax.title.set_color("#00ff41")
        for spine in ax.spines.values():
            spine.set_color("#1a3020")
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# XAI — SHAP + fallback
# ─────────────────────────────────────────────────────────────────────────────
def build_xai_figure(model, X_scaled: np.ndarray, feature_names: list, threshold: float = 0.5):
    try:
        import shap
        import numpy as np
        masker = shap.maskers.Independent(X_scaled, max_samples=50)
        model_type = type(model).__name__
        if "XGB" in model_type or "GBM" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, feature_perturbation="interventional")

        shap_values = explainer(X_scaled)
        sv = shap_values[0]

        fig, ax = plt.subplots(figsize=(8, 5))
        mpl_dark_style(fig, [ax])

        vals   = sv.values if hasattr(sv, "values") else sv
        base   = float(sv.base_values if hasattr(sv, "base_values") else explainer.expected_value)
        order  = np.argsort(np.abs(vals))[::-1]
        top_n  = min(12, len(order))
        idx    = order[:top_n][::-1]
        fnames = [feature_names[i] for i in idx]
        fvals  = vals[idx]

        colors = ["#ff2020" if v > 0 else "#00ff41" for v in fvals]
        bars   = ax.barh(fnames, fvals, color=colors, edgecolor="#040806", linewidth=0.8, alpha=0.85)
        ax.axvline(0, color="#1a3020", linewidth=1.5, linestyle="--")
        ax.set_xlabel("SHAP value  ·  impact on fraud probability", fontsize=8, color="#3a6a4a")
        ax.set_title(f"SHAP EXPLANATION  ·  BASE: {base:.3f}", fontsize=10,
                     color="#00ff41", fontweight="bold", fontfamily="monospace")

        for bar, v in zip(bars, fvals):
            ax.text(v + (0.003 if v >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
                    f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7, color="#c8d8e8", fontfamily="monospace")

        for spine in ax.spines.values(): spine.set_color("#1a3020")
        plt.tight_layout()
        return fig, "SHAP (TreeExplainer)"

    except Exception:
        plt.close("all")
        pass

    fig, ax = plt.subplots(figsize=(8, 5))
    mpl_dark_style(fig, [ax])

    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.ones(len(feature_names)) / len(feature_names)

    order   = np.argsort(importances)
    fnames  = [feature_names[i] for i in order]
    fimps   = importances[order]
    palette = [f"#{int(35 + 200*(v/max(fimps))):02x}ff{int(35 + 65*(v/max(fimps))):02x}" for v in fimps]
    ax.barh(fnames, fimps, color=palette, edgecolor="#040806", linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Feature importance (gain)", fontsize=8, color="#3a6a4a")
    ax.set_title("FEATURE IMPORTANCE  ·  XGBOOST GAIN", fontsize=10,
                 color="#00ff41", fontweight="bold", fontfamily="monospace")

    for spine in ax.spines.values(): spine.set_color("#1a3020")
    plt.tight_layout()
    return fig, "XGBoost Feature Importance (fallback)"

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_transaction(model, scaler, row: dict, threshold: float = 0.5):
    X = np.array([[row.get(f, 0.0) for f in FEATURES]], dtype=float)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    return prob, prob >= threshold, X_scaled

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT TRANSACTION
# ─────────────────────────────────────────────────────────────────────────────
if page == "predict":
    model, scaler, meta = load_deployment_model()

    st.markdown("""
    <div class="tg-page-title">TRANSACTION ANALYSIS</div>
    <div class="tg-page-sub">> SINGLE TRANSACTION FRAUD SCORING — REAL-TIME INFERENCE WITH XAI</div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="tg-section-title">// INPUT PARAMETERS</div>', unsafe_allow_html=True)

        tx_type = st.selectbox(
            "Transaction Type",
            ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"],
            help="Select the payment channel"
        )
        step   = st.number_input("Time Step (hour)", min_value=1, max_value=744, value=200)
        amount = st.number_input("Amount (PKR)", min_value=0.0, value=180000.0, step=1000.0, format="%.2f")

        st.markdown("**// SENDER BALANCES**")
        c1, c2 = st.columns(2)
        with c1:
            old_orig = st.number_input("Before Transfer", min_value=0.0, value=200000.0, step=1000.0, format="%.2f")
        with c2:
            new_orig = st.number_input("After Transfer",  min_value=0.0, value=20000.0,  step=1000.0, format="%.2f")

        st.markdown("**// RECIPIENT BALANCES**")
        c3, c4 = st.columns(2)
        with c3:
            old_dest = st.number_input("Before Receipt", min_value=0.0, value=0.0, step=1000.0, format="%.2f")
        with c4:
            new_dest = st.number_input("After Receipt",  min_value=0.0, value=0.0, step=1000.0, format="%.2f")

        tx_id = st.text_input("Transaction ID", value="TXN-001", placeholder="e.g. TXN-001")

    # Build feature dict
    balance_diff = old_orig - new_orig
    amount_ratio = (amount / old_orig) if old_orig > 0 else 0.0
    row = {
        "step": step, "amount": amount,
        "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "balanceDiff": balance_diff, "amount_ratio": amount_ratio,
        "type_CASH_OUT": 1 if tx_type == "CASH_OUT" else 0,
        "type_DEBIT":    1 if tx_type == "DEBIT"    else 0,
        "type_PAYMENT":  1 if tx_type == "PAYMENT"  else 0,
        "type_TRANSFER": 1 if tx_type == "TRANSFER" else 0,
    }

    with col_right:
        st.markdown('<div class="tg-section-title">// FEATURE VECTOR</div>', unsafe_allow_html=True)
        st.markdown('<div class="tg-terminal"><div class="tg-terminal-header">'
                    '<div class="tg-terminal-dot dot-red"></div>'
                    '<div class="tg-terminal-dot dot-amber"></div>'
                    '<div class="tg-terminal-dot dot-green"></div>'
                    '<span class="tg-terminal-title">FEATURE_PREVIEW.DAT</span>'
                    '</div>', unsafe_allow_html=True)
        for k in FEATURES:
            val = row[k]
            label = FEATURE_LABELS.get(k, k)
            st.markdown(
                f'<div class="tg-feature-row">'
                f'<span class="tg-feature-name">{label}</span>'
                f'<span class="tg-feature-val">{val:,.4f}</span>'
                f'</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
    run_btn = st.button("▶  EXECUTE FRAUD ANALYSIS", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("[ RUNNING INFERENCE ENGINE... ]"):
            prob, is_fraud, X_scaled = predict_transaction(model, scaler, row, threshold)
            risk_tier = get_risk_tier(prob)

        # ── Result metric cards ──────────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        val_class = "val-red" if is_fraud else "val-green"

        with r1:
            st.markdown(f"""
            <div class="tg-metric tg-scanline">
                <div class="tg-metric-val {val_class}">{prob:.1%}</div>
                <div class="tg-metric-label">FRAUD PROBABILITY</div>
                <div class="tg-metric-sub">threshold @ {threshold:.0%}</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            verdict_txt = "FRAUD" if is_fraud else "CLEAN"
            st.markdown(f"""
            <div class="tg-metric tg-scanline">
                <div class="tg-metric-val {val_class}" style="font-size:1.6rem">
                    {"⛔" if is_fraud else "✅"} {verdict_txt}
                </div>
                <div class="tg-metric-label">VERDICT</div>
                <div class="tg-metric-sub">model decision</div>
            </div>""", unsafe_allow_html=True)

        with r3:
            badge_map = {"CRITICAL": "val-red", "HIGH": "val-orange", "MEDIUM": "val-orange", "LOW": "val-green"}
            st.markdown(f"""
            <div class="tg-metric tg-scanline">
                <div class="tg-metric-val {badge_map.get(risk_tier,'val-white')}" style="font-size:1.6rem">
                    {risk_tier}
                </div>
                <div class="tg-metric-label">RISK TIER</div>
                <div class="tg-metric-sub">4-level classification</div>
            </div>""", unsafe_allow_html=True)

        with r4:
            confidence = abs(prob - 0.5) * 2
            st.markdown(f"""
            <div class="tg-metric tg-scanline">
                <div class="tg-metric-val val-cyan">{confidence:.1%}</div>
                <div class="tg-metric-label">MODEL CONFIDENCE</div>
                <div class="tg-metric-sub">distance from boundary</div>
            </div>""", unsafe_allow_html=True)

        # ── Probability bar ──────────────────────────────────────────────────
        bar_color = f"linear-gradient(90deg, #00ff41, {'#ff2020' if prob > 0.5 else '#ffaa00'})"
        prob_pct = prob * 100
        thr_pct  = threshold * 100
        st.markdown(f"""
        <div class="tg-prob-bar-wrap">
            <div class="tg-prob-bar-bg">
                <div class="tg-prob-bar-fill" style="width:{prob_pct:.1f}%; background:{bar_color};"></div>
                <div style="position:absolute;top:0;left:{thr_pct:.1f}%;height:100%;width:2px;background:#ff6b35;"></div>
            </div>
            <div class="tg-prob-labels">
                <span>0%  LEGITIMATE</span>
                <span style="position:absolute;left:{thr_pct:.1f}%;transform:translateX(-50%);color:#ff6b35;">
                    ▲ THR {threshold:.0%}
                </span>
                <span>FRAUD  100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── XAI + RAG side-by-side ───────────────────────────────────────────
        st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
        st.markdown('<div class="tg-section-title">// XAI ENGINE  +  REGULATORY JUSTIFICATION</div>',
                    unsafe_allow_html=True)
        xai_col, rag_col = st.columns([1, 1], gap="large")

        with xai_col:
            st.markdown('<div class="tg-section-title">[ SHAP EXPLAINABILITY ]</div>',
                        unsafe_allow_html=True)
            with st.spinner("[ COMPUTING SHAP VALUES... ]"):
                xai_fig, xai_method = build_xai_figure(model, X_scaled, FEATURES, threshold)
            st.pyplot(xai_fig, use_container_width=True)
            st.caption(f"Method: `{xai_method}`")
            plt.close(xai_fig)

        with rag_col:
            st.markdown('<div class="tg-section-title">[ SBP REGULATORY REPORT ]</div>',
                        unsafe_allow_html=True)
            openai_key = get_openai_key()
            if not openai_key:
                st.markdown("""
                <div class="tg-terminal">
                    <div class="tg-terminal-header">
                        <div class="tg-terminal-dot dot-red"></div>
                        <div class="tg-terminal-dot dot-amber"></div>
                        <div class="tg-terminal-dot dot-green"></div>
                        <span class="tg-terminal-title">RAG_MODULE.PY</span>
                    </div>
                    <span class="tg-prompt-dim">$ </span>rag_pipeline --init<br>
                    <span style="color:#ff6b35">⚠ ERROR: OPENAI_API_KEY not found in environment</span><br>
                    <span class="tg-prompt-dim">$ </span>regulatory_report --status<br>
                    <span style="color:#ff2020">✗ RAG pipeline disabled — add API key to continue</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("[ QUERYING SBP REGULATORY KB... ]"):
                    try:
                        rag_mod = _patch_rag_module()
                        rag_result = rag_mod.rag_pipeline_for_streamlit(
                            fraud_probability=prob,
                            features=row,
                            transaction_id=tx_id or "TXN-STREAMLIT",
                            openai_api_key=openai_key,
                            embed_model=embed_model,
                            reranker=reranker,
                        )
                        st.markdown(rag_result.response_text)
                        if rag_result.sources:
                            with st.expander("[ RETRIEVED SBP SOURCES ]"):
                                for s in rag_result.sources:
                                    st.markdown(f"`▸` {s}")
                    except Exception as e:
                        st.error(f"[ RAG ERROR ]: {e}")

        # ── Export report ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Export Report")

        report_lines = [
            f"TrustGuard Fraud Analysis Report",
            "=" * 40,
            f"Transaction ID  : {tx_id}",
            f"Type            : {tx_type}",
            f"Amount (PKR)    : {amount:,.2f}",
            f"Fraud Probability: {prob:.4f} ({prob:.1%})",
            f"Risk Tier       : {risk_tier}",
            f"Verdict         : {'FRAUD' if is_fraud else 'LEGITIMATE'}",
            f"Threshold Used  : {threshold}",
            f"Model           : {meta.get('model_type','XGBoost')} | AUC {meta.get('test_auc',0.9995):.4f}",
            "",
            "Features:",
        ]
        for k in FEATURES:
            report_lines.append(
                f"  {FEATURE_LABELS.get(k,k):35s}: {row[k]:>12.4f}"
            )
        report_text = "\n".join(report_lines)
        st.code(report_text, language=None)
        
        st.download_button(
            "⬇️ Download Report (.txt)",
            data=report_text,
            file_name=f"trustguard_{tx_id}_{risk_tier}.txt",
            mime="text/plain",
        )



        xai_buf = io.BytesIO()

        try:
            xai_fig.savefig(
                xai_buf,
                format="png",
                dpi=150,
                bbox_inches="tight"
            )
            xai_buf.seek(0)
            
            st.download_button(
                "⬇️ Download XAI Chart (.png)",
                xai_buf,
                file_name=f"trustguard_xai_{tx_id}.png",
                mime="image/png",
            )
        except Exception as e:
            st.warning(f"Could not generate XAI download: {e}")
        finally:
            plt.close("all")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — BATCH CSV ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "batch":
    model, scaler, meta = load_deployment_model()

    st.markdown("""
    <div class="tg-page-title">BATCH ANALYSIS</div>
    <div class="tg-page-sub">> BULK TRANSACTION SCORING — CSV UPLOAD — UP TO 500K ROWS</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tg-terminal">
        <div class="tg-terminal-header">
            <div class="tg-terminal-dot dot-red"></div>
            <div class="tg-terminal-dot dot-amber"></div>
            <div class="tg-terminal-dot dot-green"></div>
            <span class="tg-terminal-title">BATCH_INFERENCE.PY</span>
        </div>
        <span class="tg-prompt">$ </span>required_cols = [step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type]<br>
        <span class="tg-prompt">$ </span>optional_cols = [isFraud]  <span class="tg-prompt-dim"># for ground truth comparison</span><br>
        <span class="tg-prompt">$ </span>max_rows = 500_000  <span class="tg-prompt-dim"># memory safety cap</span>
    </div>
    """, unsafe_allow_html=True)

    template_df = pd.DataFrame([{
        "step": 200, "amount": 180000, "oldbalanceOrg": 200000,
        "newbalanceOrig": 20000, "oldbalanceDest": 0, "newbalanceDest": 0,
        "type": "CASH_OUT", "isFraud": 1,
    },{
        "step": 50, "amount": 1500, "oldbalanceOrg": 50000,
        "newbalanceOrig": 48500, "oldbalanceDest": 10000, "newbalanceDest": 11500,
        "type": "PAYMENT", "isFraud": 0,
    }])
    st.download_button("⬇ DOWNLOAD CSV TEMPLATE", template_df.to_csv(index=False),
                       "template.csv", "text/csv")

    st.markdown('<br>', unsafe_allow_html=True)
    uploaded = st.file_uploader("[ DROP TRANSACTIONS CSV ]", type=["csv"])

    if uploaded:
        ROW_LIMIT = 500_000
        raw = pd.read_csv(uploaded, nrows=ROW_LIMIT)
        st.info(f"[ LOADED ] **{len(raw):,}** rows × {len(raw.columns)} columns")
        if len(raw) == ROW_LIMIT:
            st.warning(f"[ CAP ] Truncated at {ROW_LIMIT:,} rows for memory safety.")

        with st.expander("[ PREVIEW — FIRST 5 ROWS ]"):
            st.dataframe(raw.head(), use_container_width=True)

        def prep_batch(df):
            d = df.copy()
            for t in ["CASH_OUT","DEBIT","PAYMENT","TRANSFER"]:
                d[f"type_{t}"] = (d["type"].str.upper() == t).astype(int)
            d["balanceDiff"]  = d["oldbalanceOrg"] - d["newbalanceOrig"]
            d["amount_ratio"] = d["amount"] / d["oldbalanceOrg"].replace(0, np.nan)
            d["amount_ratio"] = d["amount_ratio"].fillna(0)
            missing = [f for f in FEATURES if f not in d.columns]
            for m in missing: d[m] = 0
            return d

        with st.spinner("[ BATCH INFERENCE IN PROGRESS... ]"):
            df_prep = prep_batch(raw)
            CHUNK = 50_000
            probs = np.concatenate([
                model.predict_proba(scaler.transform(
                    df_prep[FEATURES].iloc[i:i+CHUNK].values.astype(float)
                ))[:,1]
                for i in range(0, len(df_prep), CHUNK)
            ])
            preds = (probs >= threshold).astype(int)

        df_out = raw.copy()
        df_out["fraud_probability"] = np.round(probs, 4)
        df_out["prediction"]        = preds
        df_out["risk_tier"]         = [get_risk_tier(p) for p in probs]

        n_fraud = int(preds.sum())

        # ── Summary metric cards ─────────────────────────────────────────────
        s1, s2, s3, s4 = st.columns(4)
        metrics_data = [
            (s1, f"{len(df_out):,}",        "TOTAL TRANSACTIONS", "val-white"),
            (s2, f"{n_fraud:,}",             "FLAGGED AS FRAUD",   "val-red"),
            (s3, f"{probs.mean():.3f}",      "AVG FRAUD PROB",     "val-orange"),
            (s4, f"{probs.max():.3f}",       "MAX FRAUD PROB",     "val-red"),
        ]
        for col, val, label, cls in metrics_data:
            with col:
                st.markdown(f"""
                <div class="tg-metric tg-scanline">
                    <div class="tg-metric-val {cls}">{val}</div>
                    <div class="tg-metric-label">{label}</div>
                    <div class="tg-metric-sub">{n_fraud/len(df_out):.1%} fraud rate</div>
                </div>""", unsafe_allow_html=True)

        # ── Risk tier bar chart ──────────────────────────────────────────────
        tier_counts = df_out["risk_tier"].value_counts().reindex(
            ["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0)

        fig_t, ax_t = plt.subplots(figsize=(9, 3))
        mpl_dark_style(fig_t, [ax_t])
        colors_t = [RISK_COLORS[t] for t in tier_counts.index]
        bars_t = ax_t.bar(tier_counts.index, tier_counts.values,
                          color=colors_t, edgecolor="#040806", linewidth=1.5, width=0.55, alpha=0.85)
        ax_t.set_title("TRANSACTIONS BY RISK TIER", fontsize=10, color="#00ff41",
                       fontweight="bold", fontfamily="monospace")
        for sp in ax_t.spines.values(): sp.set_color("#1a3020")
        for i, (idx, v) in enumerate(tier_counts.items()):
            ax_t.text(i, v + 0.3, str(v), ha="center", va="bottom",
                      fontsize=9, color=colors_t[i], fontweight="bold", fontfamily="monospace")
        plt.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)

        # ── Results table ────────────────────────────────────────────────────
        st.markdown('<div class="tg-section-title">// RESULTS TABLE — FIRST 100 ROWS</div>',
                    unsafe_allow_html=True)
        display_cols = [c for c in ["step","amount","type","fraud_probability",
                                    "prediction","risk_tier","isFraud"] if c in df_out.columns]
        st.dataframe(df_out[display_cols].head(100), use_container_width=True, hide_index=True)

        # ── Ground truth comparison ──────────────────────────────────────────
        if "isFraud" in df_out.columns:
            from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
            gt = df_out["isFraud"].values
            try:
                auc    = roc_auc_score(gt, probs)
                report = classification_report(gt, preds, output_dict=True)
                st.markdown('<div class="tg-section-title">// GROUND TRUTH COMPARISON</div>',
                            unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                gt_metrics = [
                    (m1, f"{auc:.4f}",                          "AUC-ROC",       "val-green"),
                    (m2, f"{report['1']['precision']:.4f}",     "PRECISION",     "val-cyan"),
                    (m3, f"{report['1']['recall']:.4f}",        "RECALL",        "val-orange"),
                    (m4, f"{report['1']['f1-score']:.4f}",      "F1 SCORE",      "val-green"),
                ]
                for col, val, label, cls in gt_metrics:
                    with col:
                        st.markdown(f"""
                        <div class="tg-metric tg-scanline">
                            <div class="tg-metric-val {cls}">{val}</div>
                            <div class="tg-metric-label">{label}</div>
                        </div>""", unsafe_allow_html=True)

                cm = confusion_matrix(gt, preds)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 3.5))
                mpl_dark_style(fig_cm, [ax_cm])
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d",
                            cmap=mcolors.LinearSegmentedColormap.from_list("tg", ["#040806","#00ff41"]),
                            xticklabels=["LEGIT","FRAUD"], yticklabels=["LEGIT","FRAUD"],
                            ax=ax_cm, linewidths=0.5, linecolor="#1a3020",
                            annot_kws={"fontfamily":"monospace","fontsize":11,"color":"#e8f4e8"})
                ax_cm.set_title("CONFUSION MATRIX", fontsize=9, color="#00ff41", fontfamily="monospace")
                ax_cm.tick_params(colors="#3a6a4a", labelsize=8)
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            except Exception:
                pass

        csv_out = df_out.to_csv(index=False)
        st.download_button("⬇ DOWNLOAD RESULTS CSV", csv_out,
                           "trustguard_batch_results.csv", "text/csv",
                           use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — DATASET & IMBALANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "dataset":
    st.markdown("""
    <div class="tg-page-title">DATASET INTELLIGENCE</div>
    <div class="tg-page-sub">> PAYSIM CORPUS — EDA — CLASS IMBALANCE MITIGATION</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tg-terminal">
        <div class="tg-terminal-header">
            <div class="tg-terminal-dot dot-red"></div>
            <div class="tg-terminal-dot dot-amber"></div>
            <div class="tg-terminal-dot dot-green"></div>
            <span class="tg-terminal-title">DATASET_SUMMARY.JSON</span>
        </div>
        <span class="tg-prompt-dim">$</span> <span class="tg-prompt">dataset</span>.describe()<br>
        &nbsp;&nbsp;source    : PaySim synthetic financial transactions<br>
        &nbsp;&nbsp;rows      : 6,362,620<br>
        &nbsp;&nbsp;fraud_rate: 0.13%  <span class="tg-prompt-dim"># severely imbalanced</span><br>
        &nbsp;&nbsp;features  : 18<br>
        &nbsp;&nbsp;timespan  : 30 simulated days (744 steps)
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stat_data = [
        (c1, "6.36M",  "TOTAL ROWS",    "val-white"),
        (c2, "8,213",  "FRAUD CASES",   "val-red"),
        (c3, "0.13%",  "FRAUD RATE",    "val-orange"),
        (c4, "18",     "FEATURES",      "val-cyan"),
    ]
    for col, val, label, cls in stat_data:
        with col:
            st.markdown(f"""
            <div class="tg-metric tg-scanline">
                <div class="tg-metric-val {cls}">{val}</div>
                <div class="tg-metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
    st.markdown('<div class="tg-section-title">// EXPLORATORY DATA ANALYSIS</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "TRANSACTION TYPES", "AMOUNT DIST.", "CORRELATIONS", "FRAUD PATTERNS", "FEATURE IMPORTANCE"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction Type Distribution**")
            show_img(os.path.join(PLOTS_DIR, "transaction_types_distribution.png"))
        with c2:
            st.markdown("**Fraud by Transaction Type**")
            show_img(os.path.join(PLOTS_DIR, "fraud_by_transaction_type.png"))
        st.markdown("**Detailed Fraud by Type**")
        show_img(os.path.join(PLOTS_DIR, "fraud_by_type_detailed.png"))

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction Amount Distribution**")
            show_img(os.path.join(PLOTS_DIR, "transaction_amount_distribution.png"))
        with c2:
            st.markdown("**Fraud vs Normal Amounts**")
            show_img(os.path.join(PLOTS_DIR, "fraud_vs_normal_transaction_amounts.png"))
        st.markdown("**Amount Distribution by Type & Fraud Status**")
        show_img(os.path.join(PLOTS_DIR, "amount_distribution_by_type_fraud.png"))

    with tab3:
        st.markdown("**Correlation Heatmap**")
        show_img(os.path.join(PLOTS_DIR, "correlation_heatmap.png"))
        st.markdown("**Models Comparison Heatmap**")
        show_img(os.path.join(PLOTS_DIR, "models_comparison_heatmap.png"))
        st.markdown("**EDA Feature Distributions**")
        show_img(os.path.join(PLOTS_DIR, "eda_feature_distributions.png"))

    with tab4:
        st.markdown("**Top Fraud Patterns — Mean Feature Comparison**")
        show_img(os.path.join(PLOTS_DIR, "top_fraud_patterns.png"))
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transaction Volume Over Time**")
            show_img(os.path.join(PLOTS_DIR, "transaction_volume_over_time.png"))
        with c2:
            st.markdown("**EDA Class Distribution**")
            show_img(os.path.join(PLOTS_DIR, "eda_class_distribution.png"))

    with tab5:
        st.markdown("**Feature Importance (XGBoost)**")
        show_img(os.path.join(PLOTS_DIR, "feature_importance_1.png"))
        st.markdown("**Feature Importance Comparison — All Models**")
        show_img(os.path.join(PLOTS_DIR, "feature_importance_comparison.png"))

    st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
    st.markdown('<div class="tg-section-title">// CLASS IMBALANCE MITIGATION</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="tg-card tg-card-warning">
        <b style="color:#ff6b35;">PROBLEM:</b>
        <span style="font-family:'Share Tech Mono',monospace;font-size:0.82rem;color:#c8d8e8;">
        0.13% fraud rate → naive model achieves 99.87% accuracy predicting everything as LEGITIMATE.
        Accuracy is a useless metric. TrustGuard deploys a 2-stage strategy:
        <b style="color:#00ff41">Fraud Simulation</b> + <b style="color:#00ff41">SMOTE Oversampling</b>,
        evaluated via Precision-Recall AUC.
        </span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Class Distribution (Before Balancing)**")
        show_img(os.path.join(PLOTS_DIR, "class_imbalance.png"))
    with c2:
        st.markdown("**Fraud Distribution**")
        show_img(os.path.join(PLOTS_DIR, "fraud_distribution.png"))

    s1, s2, s3 = st.columns(3)
    stage_data = [
        (s1, "~0.13%", "① ORIGINAL TRAIN",       "Raw PaySim fraud rate",             "val-red"),
        (s2, "~1.26%", "② AFTER SIMULATION",      "5% TRANSFER/CASH_OUT injected",     "val-orange"),
        (s3, "~23%",   "③ AFTER SMOTE",           "sampling_strategy=0.3 applied",     "val-green"),
    ]
    for col, val, label, sub, cls in stage_data:
        with col:
            st.markdown(f"""
            <div class="tg-metric tg-scanline">
                <div class="tg-metric-val {cls}">{val}</div>
                <div class="tg-metric-label">{label}</div>
                <div class="tg-metric-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="tg-card">
        <div class="tg-section-title">// FRAUD SIMULATION ENGINE</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#c8d8e8;line-height:1.8;">
        ▸ Samples 5% of TRANSFER & CASH_OUT transactions<br>
        ▸ Simulates full account drain: amount = oldbalanceOrg<br>
        ▸ Sets newbalanceOrig = 0, recomputes ratios<br>
        ▸ Labels injected rows as fraud<br>
        ▸ Applied BEFORE SMOTE — realistic behaviour grounding
        </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="tg-card">
        <div class="tg-section-title">// SMOTE OVERSAMPLING</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#c8d8e8;line-height:1.8;">
        ▸ sampling_strategy = 0.3 (minority → 30% of majority)<br>
        ▸ Applied ONLY to training set — test stays real data<br>
        ▸ Inside K-Fold CV via ImbPipeline — no leakage<br>
        ▸ Tested: 0.0, 0.3 (Best ★), 0.5 in ablation study<br>
        ▸ 0.1 had negligible effect; 0.5 caused bloat
        </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**SMOTE Ratio Ablation — AUPRC, Recall, Precision vs ratio**")
    show_img(os.path.join(ABLATION_DIR, "ablation_smote_trend.png"))
    st.markdown("**Cost-Benefit Analysis**")
    show_img(os.path.join(PLOTS_DIR, "cost_benefit_analysis.png"))

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "performance":
    st.markdown("""
    <div class="tg-page-title">MODEL PERFORMANCE</div>
    <div class="tg-page-sub">> COMPARATIVE ANALYSIS — 4 MODELS — TEST SET METRICS</div>
    """, unsafe_allow_html=True)

    comp_df = load_model_comparison()
    if comp_df is not None:
        st.markdown('<div class="tg-section-title">// ALL MODELS — TEST METRICS</div>',
                    unsafe_allow_html=True)
        disp_cols = ["Model","Test Precision","Test Recall","Test F1","Test AUC-ROC","Test Avg Prec"]
        st.dataframe(comp_df[[c for c in disp_cols if c in comp_df.columns]],
                     use_container_width=True, hide_index=True)

        metrics = ["Test Precision","Test Recall","Test F1","Test AUC-ROC"]
        avail   = [m for m in metrics if m in comp_df.columns]
        fig_b, ax_b = plt.subplots(figsize=(10, 4))
        mpl_dark_style(fig_b, [ax_b])
        x       = np.arange(len(avail))
        width   = 0.18
        colors_m = ["#00ff41", "#00aaff", "#ff6b35", "#ff2020"]
        for i, (_, row_m) in enumerate(comp_df.iterrows()):
            vals = [row_m[m] for m in avail]
            ax_b.bar(x + i*width, vals, width, label=row_m["Model"],
                     color=colors_m[i % len(colors_m)], edgecolor="#040806",
                     linewidth=0.8, alpha=0.85)
        ax_b.set_xticks(x + width*1.5)
        ax_b.set_xticklabels([m.replace("Test ","") for m in avail],
                              fontsize=9, color="#3a6a4a", fontfamily="monospace")
        ax_b.set_ylim(0, 1.1)
        ax_b.set_title("MODEL COMPARISON — TEST METRICS", fontsize=11,
                        color="#00ff41", fontweight="bold", fontfamily="monospace")
        ax_b.legend(fontsize=8, framealpha=0.3, facecolor="#040806",
                    edgecolor="#1a3020", labelcolor="#c8d8e8")
        for sp in ax_b.spines.values(): sp.set_color("#1a3020")
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
    st.markdown('<div class="tg-section-title">// VISUALISATIONS</div>', unsafe_allow_html=True)

    plot_map = {
        "ROC & PR Curves":              "roc_pr_curves.png",
        "Confusion Matrices":           "confusion_matrices_all_models.png",
        "Feature Importance":           "feature_importance_comparison.png",
        "Fraud Probability Dist.":      "fraud_prob_distribution_all_models.png",
        "Cost-Benefit Analysis":        "cost_benefit_analysis.png",
        "Error Analysis":               "error_analysis.png",
        "Model Comparison All Metrics": "model_comparison_all_metrics.png",
        "Classification Report Heatmap":"classification_report_heatmap.png",
        "Model Metrics Lineplot":       "model_metrics_lineplot.png",
        "Models Comparison Heatmap":    "models_comparison_heatmap.png",
    }
    cols = st.columns(2)
    for i, (title, fname) in enumerate(plot_map.items()):
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(Image.open(fpath), use_container_width=True)

    xgb_path = os.path.join(METRICS_DIR, "xgboost_metrics.json")
    if os.path.exists(xgb_path):
        with open(xgb_path) as f:
            xgb_m = json.load(f)
        st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
        st.markdown('<div class="tg-section-title">// XGBOOST (DEPLOYED) — FULL METRICS</div>',
                    unsafe_allow_html=True)
        m_cols = st.columns(5)
        keys = [("test_auc_roc","AUC-ROC"),("test_recall","RECALL"),
                ("test_precision","PRECISION"),("test_f1","F1"),("test_avg_prec","AVG PREC")]
        cls_list = ["val-green","val-orange","val-cyan","val-green","val-cyan"]
        for j, ((k, lbl), cls) in enumerate(zip(keys, cls_list)):
            with m_cols[j]:
                st.markdown(f"""
                <div class="tg-metric tg-scanline">
                    <div class="tg-metric-val {cls}">{xgb_m.get(k, 0):.4f}</div>
                    <div class="tg-metric-label">{lbl}</div>
                </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ablation":
    st.markdown("""
    <div class="tg-page-title">ABLATION STUDY</div>
    <div class="tg-page-sub">> COMPONENT-LEVEL PIPELINE ANALYSIS — CONTRIBUTION QUANTIFICATION</div>
    """, unsafe_allow_html=True)

    ab_df = load_ablation()
    if ab_df is not None:
        st.dataframe(ab_df, use_container_width=True, hide_index=True)

        fig_a, axes = plt.subplots(1, 2, figsize=(12, 5))
        mpl_dark_style(fig_a, list(axes))

        pal = ["#00ff41" if "full" in c.lower() or "current" in c.lower()
               else "#1a4a2a" for c in ab_df["Condition"]]

        axes[0].barh(ab_df["Condition"], ab_df["CV F1"],
                     color=pal, edgecolor="#040806", linewidth=0.8, alpha=0.85)
        axes[0].set_title("CV F1 BY CONDITION", fontsize=10, color="#00ff41",
                           fontweight="bold", fontfamily="monospace")
        axes[0].set_xlim(0, 1.1)
        for sp in axes[0].spines.values(): sp.set_color("#1a3020")

        axes[1].barh(ab_df["Condition"], ab_df["Test AUC"],
                     color=pal, edgecolor="#040806", linewidth=0.8, alpha=0.85)
        axes[1].set_title("TEST AUC BY CONDITION", fontsize=10, color="#00ff41",
                           fontweight="bold", fontfamily="monospace")
        min_auc = ab_df["Test AUC"].min()
        axes[1].set_xlim(max(0, min_auc - 0.01), 1.002)
        for sp in axes[1].spines.values(): sp.set_color("#1a3020")

        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)

    st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
    st.markdown('<div class="tg-section-title">// ABLATION PLOTS</div>', unsafe_allow_html=True)

    ab_plots = {
        "Ablation Summary":  "ablation_study.png",
        "Ablation Heatmap":  "ablation_heatmap.png",
        "SMOTE Trend":       "ablation_smote_trend.png",
        "PR Scatter":        "ablation_pr_scatter.png",
        "Delta (Δ) Chart":   "ablation_delta.png",
    }
    ab_cols = st.columns(2)
    for i, (title, fname) in enumerate(ab_plots.items()):
        fpath = os.path.join(ABLATION_DIR, fname)
        if os.path.exists(fpath):
            with ab_cols[i % 2]:
                st.markdown(f"**{title}**")
                show_img(fpath)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "about":
    st.markdown("""
    <div class="tg-page-title">SYSTEM INFORMATION</div>
    <div class="tg-page-sub">> TRUSTGUARD AI — ARCHITECTURE — DELIVERABLE 3 COMPLIANCE</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="tg-terminal">
        <div class="tg-terminal-header">
            <div class="tg-terminal-dot dot-red"></div>
            <div class="tg-terminal-dot dot-amber"></div>
            <div class="tg-terminal-dot dot-green"></div>
            <span class="tg-terminal-title">SYSTEM_INFO.SH</span>
        </div>
        <span class="tg-prompt">$ </span>cat /etc/trustguard/description<br>
        TrustGuard is an AI-powered fraud detection system built on the PaySim synthetic<br>
        financial transaction dataset. It combines classical ML, deep learning, XAI<br>
        explainability (SHAP), and a RAG pipeline grounded in SBP regulatory documents.<br><br>
        <span class="tg-prompt">$ </span>trustguard --version<br>
        <span style="color:#00aaff">TrustGuard v1.0 · Track A · Deliverable 3 · XGBoost Deployment</span>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="tg-card">
        <div class="tg-section-title">// SYSTEM ARCHITECTURE</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.78rem;color:#c8d8e8;line-height:2;">
        ▸ <span style="color:#00ff41">DATA</span>: PaySim synthetic transactions (6.3M rows)<br>
        ▸ <span style="color:#00ff41">FEATURES</span>: balance diffs, amount ratios, one-hot type<br>
        ▸ <span style="color:#00ff41">MODELS</span>: XGBoost ★, Random Forest, Neural Net, LR<br>
        ▸ <span style="color:#00ff41">IMBALANCE</span>: SMOTE + fraud simulation augmentation<br>
        ▸ <span style="color:#00ff41">XAI</span>: SHAP TreeExplainer (fallback: feature importance)<br>
        ▸ <span style="color:#00ff41">RAG</span>: ChromaDB + sentence-transformers + BM25 hybrid<br>
        ▸ <span style="color:#00ff41">LLM</span>: GPT-4o-mini via OpenAI API<br>
        ▸ <span style="color:#00ff41">CORPUS</span>: SBP AML/CFT, Branchless Banking, SME PRs
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="tg-section-title">// DEPLOYED MODEL METADATA</div>',
                    unsafe_allow_html=True)
        _, _, meta = load_deployment_model()
        st.json(meta)

    st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
    st.markdown('<div class="tg-section-title">// DELIVERABLE 3 COMPLIANCE CHECKLIST</div>',
                unsafe_allow_html=True)

    checks = [
        ("✓", "Model hosted and serving real-time inference"),
        ("✓", "Frontend communicates with backend model"),
        ("✓", "CSV upload + batch inference supported"),
        ("✓", "XAI heatmap (SHAP / feature importance)"),
        ("✓", "RAG regulatory justification beside heatmap"),
        ("✓", "Fraud probability + confidence score displayed"),
        ("✓", "Top navigation bar with section routing"),
        ("✓", "Export / download reports (.txt + .png)"),
        ("✓", "Model performance metrics dashboard"),
        ("✓", "Ablation study visualisation"),
    ]
    for icon, text in checks:
        st.markdown(f"""
        <div class="tg-check-item">
            <span class="check-icon">{icon}</span>
            <span>{text}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="tg-glow-line">', unsafe_allow_html=True)
    st.markdown('<div class="tg-section-title">// RAG KNOWN LIMITATIONS</div>',
                unsafe_allow_html=True)
    st.warning("""
    ⚠ Requires a valid OpenAI API key (gpt-4o-mini). Without it, the regulatory report section is skipped.
    ⚠ ChromaDB stores ~100 SBP document chunks. Ensure chroma_db/ is committed to the repo.
    ⚠ If OpenAI rate-limits are hit, the RAG section will display a rate-limit error.
    """)
