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
# In app.py — simplified now that rag_module resolves its own path
import rag_module as _rm   # no patching needed

warnings.filterwarnings("ignore")

# ── Resolve project root so relative imports work on Streamlit Cloud ──────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Patch rag_module's hard-coded Windows path BEFORE importing it ────────────
import importlib, types

def _patch_rag_module():
    """Load rag_module but override CHROMA_DB_PATH to a relative path."""
    import rag_module as _rm
    _rm.RAG_CONFIG["CHROMA_DB_PATH"] = os.path.join(ROOT, "chroma_db")
    return _rm

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TrustGuard · Fraud Analyst Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS  (Doodle-inspired palette from skill.md: primary #49B6E5, secondary #263D5B)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Delius+Swash+Caps&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Delius Swash Caps', cursive; }
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #263D5B 0%, #1a2d44 100%);
    color: #f0f4f8;
}
[data-testid="stSidebar"] * { color: #f0f4f8 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 4px 0; }

/* Cards */
.tg-card {
    background: #ffffff;
    border: 2px solid #49B6E5;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 3px 3px 0px #49B6E540;
}
.tg-card-danger {
    border-color: #DC2626;
    box-shadow: 3px 3px 0px #DC262640;
}
.tg-card-warning {
    border-color: #D97706;
    box-shadow: 3px 3px 0px #D9770640;
}
.tg-card-success {
    border-color: #16A34A;
    box-shadow: 3px 3px 0px #16A34A40;
}

/* Risk badges */
.badge-critical { background:#DC2626; color:#fff; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:700; }
.badge-high     { background:#D97706; color:#fff; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:700; }
.badge-medium   { background:#49B6E5; color:#fff; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:700; }
.badge-low      { background:#16A34A; color:#fff; padding:3px 10px; border-radius:20px; font-size:.8rem; font-weight:700; }

/* Probability gauge container */
.gauge-wrap { text-align:center; padding: 1rem 0; }
.gauge-pct  { font-size: 3rem; font-weight:800; line-height:1; }
.gauge-label{ font-size: .9rem; color: #666; margin-top: 4px; }

/* Section headers */
h1, h2, h3 { font-family: 'Delius Swash Caps', cursive !important; }
.section-title {
    font-size: 1.1rem; font-weight: 700;
    color: #263D5B; border-bottom: 2px dashed #49B6E5;
    padding-bottom: 4px; margin-bottom: 12px;
}

/* Table */
.stDataFrame { border: 1px solid #49B6E530; border-radius: 8px; }

/* Buttons */
.stButton > button {
    background: #263D5B; color: #fff;
    border-radius: 10px; border: 2px solid #49B6E5;
    font-family: 'Delius Swash Caps', cursive;
    transition: background .2s;
}
.stButton > button:hover { background: #49B6E5; color: #fff; }

/* Top header bar */
.topbar {
    background: linear-gradient(90deg, #263D5B, #49B6E5);
    color: white; padding: .6rem 1.5rem;
    border-radius: 12px; margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 1rem;
}
.topbar h1 { margin:0; font-size:1.6rem; color:white !important; }
.topbar span { font-size:.85rem; opacity:.85; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS / PATHS
# ─────────────────────────────────────────────────────────────────────────────
DEPLOY_DIR  = os.path.join(ROOT, "outputs", "deployment")
MODELS_DIR  = os.path.join(ROOT, "outputs", "models")
METRICS_DIR = os.path.join(ROOT, "outputs", "metrics")
PLOTS_DIR   = os.path.join(ROOT, "outputs", "plots")
ABLATION_DIR= os.path.join(ROOT, "outputs", "ablation")

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
    "CRITICAL": "#DC2626", "HIGH": "#D97706",
    "MEDIUM": "#49B6E5", "LOW": "#16A34A"
}

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — LOAD ARTEFACTS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artefacts…")
def load_deployment_model():
    import joblib
    model  = joblib.load(os.path.join(DEPLOY_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(DEPLOY_DIR, "scaler.pkl"))
    with open(os.path.join(DEPLOY_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

@st.cache_resource(show_spinner="Loading all model artefacts…")
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
    """Try st.secrets first, then session_state (sidebar input)."""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return st.session_state.get("openai_api_key", "")

def risk_badge(tier: str) -> str:
    return f'<span class="badge-{tier.lower()}">{tier}</span>'

def risk_color(tier: str) -> str:
    return RISK_COLORS.get(tier, "#263D5B")

def get_risk_tier(prob: float) -> str:
    if prob >= 0.85: return "CRITICAL"
    if prob >= 0.65: return "HIGH"
    if prob >= 0.50: return "MEDIUM"
    return "LOW"

# ─────────────────────────────────────────────────────────────────────────────
# XAI — SHAP + fallback
# ─────────────────────────────────────────────────────────────────────────────
def build_xai_figure(model, X_scaled: np.ndarray, feature_names: list, threshold: float = 0.5):
    """
    Try SHAP waterfall for per-transaction XAI.
    On OOM / import error, fall back to XGBoost feature-importance bar chart.
    Returns (fig, method_used).
    """
    # ── Try SHAP ──────────────────────────────────────────────────────────
    try:
        import shap
        # Use TreeExplainer only for tree models (fast, low-memory)
        model_type = type(model).__name__
        if "XGB" in model_type or "GBM" in model_type:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.Explainer(model, feature_perturbation="interventional")

        shap_values = explainer(X_scaled)
        sv = shap_values[0]  # first (only) row

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#f8fbff")
        ax.set_facecolor("#f8fbff")

        vals   = sv.values if hasattr(sv, "values") else sv
        base   = float(sv.base_values if hasattr(sv, "base_values") else explainer.expected_value)

        # Sort by absolute contribution
        order  = np.argsort(np.abs(vals))[::-1]
        top_n  = min(12, len(order))
        idx    = order[:top_n][::-1]
        fnames = [feature_names[i] for i in idx]
        fvals  = vals[idx]

        colors = ["#DC2626" if v > 0 else "#16A34A" for v in fvals]
        bars   = ax.barh(fnames, fvals, color=colors, edgecolor="white", linewidth=0.6)
        ax.axvline(0, color="#263D5B", linewidth=1.2, linestyle="--", alpha=0.5)
        ax.set_xlabel("SHAP Value (impact on fraud probability)", fontsize=9, color="#263D5B")
        ax.set_title(f"SHAP Explanation  ·  Base: {base:.3f}", fontsize=11,
                     color="#263D5B", fontweight="bold")
        ax.tick_params(labelsize=8, colors="#263D5B")
        for spine in ax.spines.values(): spine.set_visible(False)

        # Add value annotations
        for bar, v in zip(bars, fvals):
            ax.text(v + (0.003 if v >= 0 else -0.003), bar.get_y() + bar.get_height()/2,
                    f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=7, color="#263D5B")

        plt.tight_layout()
        return fig, "SHAP (TreeExplainer)"

    except Exception:
        pass  # fall through to fallback

    # ── Fallback: XGBoost built-in feature importances ────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#f8fbff")
    ax.set_facecolor("#f8fbff")

    try:
        importances = model.feature_importances_
    except AttributeError:
        importances = np.ones(len(feature_names)) / len(feature_names)

    order      = np.argsort(importances)
    fnames     = [feature_names[i] for i in order]
    fimps      = importances[order]
    palette    = plt.cm.Blues(np.linspace(0.35, 0.9, len(fimps)))
    ax.barh(fnames, fimps, color=palette, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (gain)", fontsize=9, color="#263D5B")
    ax.set_title("Feature Importance (XGBoost Gain)", fontsize=11,
                 color="#263D5B", fontweight="bold")
    ax.tick_params(labelsize=8, colors="#263D5B")
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    return fig, "XGBoost Feature Importance (fallback)"

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_transaction(model, scaler, row: dict, threshold: float = 0.5):
    """
    row: dict with keys = FEATURES
    Returns (fraud_prob, is_fraud, scaled_array)
    """
    X = np.array([[row.get(f, 0.0) for f in FEATURES]], dtype=float)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    return prob, prob >= threshold, X_scaled

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ TrustGuard")
    st.markdown("**Fraud Analyst Dashboard**")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🔍 Predict Transaction",
        "📂 Batch CSV Analysis",
        "📊 Model Performance",
        "🔬 Ablation Study",
        "ℹ️ About"
    ])

    st.markdown("---")
    st.markdown("#### OpenAI API Key")
    st.caption("For RAG regulatory justification. Tries `st.secrets` first.")
    api_input = st.text_input("Paste key (optional)", type="password",
                               placeholder="sk-proj-...", key="openai_api_key")
    if api_input:
        st.success("Key stored for this session.")
    elif get_openai_key():
        st.success("Key loaded from Secrets ✓")
    else:
        st.warning("No key — RAG will be skipped.")

    st.markdown("---")
    st.markdown("#### Decision Threshold")
    threshold = st.slider("Fraud threshold", 0.1, 0.9, 0.5, 0.01,
                          help="Probability cutoff to classify as fraud")

    st.markdown("---")
    st.caption("TrustGuard v1.0 · Track A · Deliverable 3")

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div>
        <h1>🛡️ TrustGuard — Fraud Analyst Dashboard</h1>
        <span>AI-powered fraud detection with XAI explainability & SBP regulatory justification</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PREDICT TRANSACTION
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔍 Predict Transaction":
    model, scaler, meta = load_deployment_model()

    st.markdown("### 🔍 Predict a Single Transaction")
    st.markdown("Enter transaction details manually to get an instant fraud verdict with XAI + regulatory justification.")

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Transaction Details</div>', unsafe_allow_html=True)

        tx_type = st.selectbox("Transaction Type",
            ["CASH_OUT", "TRANSFER", "PAYMENT", "DEBIT", "CASH_IN"],
            help="Select the payment channel")
        step   = st.number_input("Time Step (hour)", min_value=1, max_value=744, value=200)
        amount = st.number_input("Amount (PKR)", min_value=0.0, value=180000.0, step=1000.0,
                                  format="%.2f")

        st.markdown("**Sender Balances**")
        c1, c2 = st.columns(2)
        with c1:
            old_orig = st.number_input("Before Transfer", min_value=0.0, value=200000.0,
                                        step=1000.0, format="%.2f")
        with c2:
            new_orig = st.number_input("After Transfer", min_value=0.0, value=20000.0,
                                        step=1000.0, format="%.2f")

        st.markdown("**Recipient Balances**")
        c3, c4 = st.columns(2)
        with c3:
            old_dest = st.number_input("Before Receipt", min_value=0.0, value=0.0,
                                        step=1000.0, format="%.2f")
        with c4:
            new_dest = st.number_input("After Receipt", min_value=0.0, value=0.0,
                                        step=1000.0, format="%.2f")

        tx_id = st.text_input("Transaction ID (optional)", value="TXN-001",
                               placeholder="e.g. TXN-001")

    # Build feature dict
    balance_diff  = old_orig - new_orig
    amount_ratio  = (amount / old_orig) if old_orig > 0 else 0.0
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
        st.markdown('<div class="section-title">Feature Preview</div>', unsafe_allow_html=True)
        preview_df = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(k, k) for k in FEATURES],
            "Value":   [f"{row[k]:,.4f}" for k in FEATURES],
        })
        st.dataframe(preview_df, use_container_width=True, hide_index=True, height=280)

    st.markdown("---")
    run_btn = st.button("🚀 Run Fraud Analysis", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Running inference…"):
            prob, is_fraud, X_scaled = predict_transaction(model, scaler, row, threshold)
            risk_tier = get_risk_tier(prob)

        # ── Results row ──────────────────────────────────────────────────
        r1, r2, r3, r4 = st.columns(4)
        card_class = "tg-card-danger" if is_fraud else "tg-card-success"
        with r1:
            st.markdown(f"""
            <div class="tg-card {card_class}">
                <div class="gauge-wrap">
                    <div class="gauge-pct" style="color:{risk_color(risk_tier)}">
                        {prob:.1%}
                    </div>
                    <div class="gauge-label">Fraud Probability</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with r2:
            verdict = "⛔ FRAUD" if is_fraud else "✅ LEGITIMATE"
            v_color = "#DC2626" if is_fraud else "#16A34A"
            st.markdown(f"""
            <div class="tg-card {card_class}">
                <div class="gauge-wrap">
                    <div class="gauge-pct" style="color:{v_color};font-size:1.8rem">
                        {verdict}
                    </div>
                    <div class="gauge-label">Verdict (threshold {threshold:.0%})</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="tg-card">
                <div class="gauge-wrap">
                    <div class="gauge-pct" style="color:{risk_color(risk_tier)};font-size:2rem">
                        {risk_badge(risk_tier)}
                    </div>
                    <div class="gauge-label">Risk Tier</div>
                </div>
            </div>""", unsafe_allow_html=True)
        with r4:
            confidence = abs(prob - 0.5) * 2
            st.markdown(f"""
            <div class="tg-card">
                <div class="gauge-wrap">
                    <div class="gauge-pct" style="color:#263D5B;font-size:2rem">
                        {confidence:.1%}
                    </div>
                    <div class="gauge-label">Model Confidence</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # ── XAI + RAG side-by-side ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 XAI Heatmap  +  Regulatory Justification")
        xai_col, rag_col = st.columns([1, 1], gap="large")

        with xai_col:
            st.markdown('<div class="section-title">🧠 Explainability Heatmap</div>',
                        unsafe_allow_html=True)
            with st.spinner("Generating XAI…"):
                xai_fig, xai_method = build_xai_figure(model, X_scaled, FEATURES, threshold)
            st.pyplot(xai_fig, use_container_width=True)
            st.caption(f"Method: **{xai_method}**")
            plt.close(xai_fig)

        with rag_col:
            st.markdown('<div class="section-title">📜 SBP Regulatory Report</div>',
                        unsafe_allow_html=True)
            openai_key = get_openai_key()
            if not openai_key:
                st.warning("No OpenAI API key — RAG skipped. Add key in sidebar or Streamlit Secrets.")
            else:
                with st.spinner("Querying SBP regulatory knowledge base…"):
                    try:
                        rag_mod = _patch_rag_module()
                        rag_result = rag_mod.rag_pipeline_for_streamlit(
                            fraud_probability=prob,
                            features=row,
                            transaction_id=tx_id or "TXN-STREAMLIT",
                            openai_api_key=openai_key,
                        )
                        st.markdown(f"""
                        <div class="tg-card">
                            <b>Transaction:</b> {rag_result.transaction_id} &nbsp;
                            {risk_badge(rag_result.risk_tier)}<br/>
                            <b>Fraud Prob:</b> {rag_result.fraud_probability:.1%} &nbsp;|&nbsp;
                            <b>Grounding:</b> {rag_result.grounding_score:.0%} &nbsp;|&nbsp;
                            <b>Latency:</b> {rag_result.latency_seconds:.1f}s
                        </div>""", unsafe_allow_html=True)
                        st.markdown(rag_result.response_text)

                        if rag_result.sources:
                            with st.expander("📚 Retrieved SBP Sources"):
                                for s in rag_result.sources:
                                    st.markdown(f"- {s}")

                    except Exception as e:
                        st.error(f"RAG error: {e}")
                        st.info("Check OpenAI key validity and ChromaDB path.")

        # ── Probability bar ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### Fraud Probability Gauge")
        fig_g, ax_g = plt.subplots(figsize=(10, 1.2))
        fig_g.patch.set_alpha(0)
        ax_g.set_facecolor("#f0f4f8")
        cmap  = mcolors.LinearSegmentedColormap.from_list("rg", ["#16A34A","#D97706","#DC2626"])
        ax_g.barh(["Fraud prob"], [1], color="#e8edf2", height=0.6)
        ax_g.barh(["Fraud prob"], [prob], color=cmap(prob), height=0.6)
        ax_g.axvline(threshold, color="#263D5B", linewidth=2, linestyle="--")
        ax_g.set_xlim(0, 1)
        ax_g.set_xlabel("Probability", fontsize=9)
        ax_g.text(threshold + 0.01, 0, f"threshold {threshold:.0%}", fontsize=8,
                  va="center", color="#263D5B")
        ax_g.text(prob, 0, f" {prob:.1%}", fontsize=9, va="center",
                  color="white" if prob > 0.4 else "#263D5B", fontweight="bold")
        for sp in ax_g.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_g, use_container_width=True)
        plt.close(fig_g)

        # ── Export report ─────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("#### 📥 Export Report")
        report_lines = [
            f"TrustGuard Fraud Analysis Report",
            f"=" * 40,
            f"Transaction ID  : {tx_id}",
            f"Type            : {tx_type}",
            f"Amount (PKR)    : {amount:,.2f}",
            f"Fraud Probability: {prob:.4f} ({prob:.1%})",
            f"Risk Tier       : {risk_tier}",
            f"Verdict         : {'FRAUD' if is_fraud else 'LEGITIMATE'}",
            f"Threshold Used  : {threshold}",
            f"Model           : {meta.get('model_type','XGBoost')} | AUC {meta.get('test_auc',0.9995):.4f}",
            f"",
            f"Features:",
            *[f"  {FEATURE_LABELS.get(k,k):35s}: {row[k]:>12.4f}" for k in FEATURES],
        ]
        report_text = "\n".join(report_lines)
        st.download_button(
            "⬇️ Download Report (.txt)",
            data=report_text,
            file_name=f"trustguard_{tx_id}_{risk_tier}.txt",
            mime="text/plain",
        )

        # Save XAI figure for download
        xai_buf = io.BytesIO()
        xai_fig2, _, = build_xai_figure(model, X_scaled, FEATURES, threshold)
        xai_fig2.savefig(xai_buf, format="png", dpi=150, bbox_inches="tight")
        xai_buf.seek(0)
        plt.close(xai_fig2)
        st.download_button(
            "⬇️ Download XAI Chart (.png)",
            data=xai_buf,
            file_name=f"trustguard_xai_{tx_id}.png",
            mime="image/png",
        )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — BATCH CSV ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📂 Batch CSV Analysis":
    model, scaler, meta = load_deployment_model()

    st.markdown("### 📂 Batch Transaction Analysis")
    st.markdown("""
    Upload a CSV with transaction data. Required columns:
    `step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type`

    Optional: `isFraud` for ground-truth comparison.
    """)

    # CSV template download
    template_df = pd.DataFrame([{
        "step": 200, "amount": 180000, "oldbalanceOrg": 200000,
        "newbalanceOrig": 20000, "oldbalanceDest": 0, "newbalanceDest": 0,
        "type": "CASH_OUT", "isFraud": 1,
    },{
        "step": 50, "amount": 1500, "oldbalanceOrg": 50000,
        "newbalanceOrig": 48500, "oldbalanceDest": 10000, "newbalanceDest": 11500,
        "type": "PAYMENT", "isFraud": 0,
    }])
    st.download_button("⬇️ Download CSV Template", template_df.to_csv(index=False),
                       "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded:
        raw = pd.read_csv(uploaded)
        st.info(f"Loaded **{len(raw):,}** rows × {len(raw.columns)} columns")

        # Preview
        with st.expander("Preview raw data (first 5 rows)"):
            st.dataframe(raw.head(), use_container_width=True)

        # Feature engineering
        def prep_batch(df):
            d = df.copy()
            for t in ["CASH_OUT","DEBIT","PAYMENT","TRANSFER"]:
                d[f"type_{t}"] = (d["type"].str.upper() == t).astype(int)
            d["balanceDiff"]  = d["oldbalanceOrg"]  - d["newbalanceOrig"]
            d["amount_ratio"] = d["amount"] / d["oldbalanceOrg"].replace(0, np.nan)
            d["amount_ratio"].fillna(0, inplace=True)
            missing = [f for f in FEATURES if f not in d.columns]
            for m in missing: d[m] = 0
            return d

        with st.spinner("Running batch inference…"):
            df_prep = prep_batch(raw)
            X = df_prep[FEATURES].values.astype(float)
            X_s = scaler.transform(X)
            probs  = model.predict_proba(X_s)[:, 1]
            preds  = (probs >= threshold).astype(int)

        df_out = raw.copy()
        df_out["fraud_probability"] = np.round(probs, 4)
        df_out["prediction"]        = preds
        df_out["risk_tier"]         = [get_risk_tier(p) for p in probs]

        # Summary stats
        n_fraud = int(preds.sum())
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Transactions", f"{len(df_out):,}")
        s2.metric("Flagged as Fraud", f"{n_fraud:,}", delta=f"{n_fraud/len(df_out):.1%}")
        s3.metric("Avg Fraud Prob", f"{probs.mean():.3f}")
        s4.metric("Max Fraud Prob", f"{probs.max():.3f}")

        # Risk tier chart
        tier_counts = df_out["risk_tier"].value_counts().reindex(
            ["CRITICAL","HIGH","MEDIUM","LOW"], fill_value=0)

        fig_t, ax_t = plt.subplots(figsize=(8, 3))
        fig_t.patch.set_facecolor("#f8fbff")
        ax_t.set_facecolor("#f8fbff")
        colors = [RISK_COLORS[t] for t in tier_counts.index]
        ax_t.bar(tier_counts.index, tier_counts.values, color=colors,
                 edgecolor="white", linewidth=0.8, width=0.6)
        ax_t.set_title("Transactions by Risk Tier", fontsize=11, color="#263D5B", fontweight="bold")
        ax_t.tick_params(colors="#263D5B", labelsize=9)
        for sp in ax_t.spines.values(): sp.set_visible(False)
        for i, (idx, v) in enumerate(tier_counts.items()):
            ax_t.text(i, v + 0.3, str(v), ha="center", va="bottom",
                      fontsize=9, color="#263D5B", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_t, use_container_width=True)
        plt.close(fig_t)

        # Results table
        st.markdown("#### Results Table (first 100 rows)")
        display_cols = [c for c in ["step","amount","type","fraud_probability","prediction","risk_tier","isFraud"]
                        if c in df_out.columns]
        st.dataframe(df_out[display_cols].head(100), use_container_width=True, hide_index=True)

        # Accuracy if ground truth present
        if "isFraud" in df_out.columns:
            from sklearn.metrics import (classification_report, roc_auc_score,
                                         confusion_matrix)
            gt = df_out["isFraud"].values
            try:
                auc = roc_auc_score(gt, probs)
                report = classification_report(gt, preds, output_dict=True)
                st.markdown("#### Ground Truth Comparison")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("AUC-ROC", f"{auc:.4f}")
                m2.metric("Precision (fraud)", f"{report['1']['precision']:.4f}")
                m3.metric("Recall (fraud)",    f"{report['1']['recall']:.4f}")
                m4.metric("F1 (fraud)",        f"{report['1']['f1-score']:.4f}")

                # Confusion matrix
                cm = confusion_matrix(gt, preds)
                fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
                import seaborn as sns
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"],
                            ax=ax_cm)
                ax_cm.set_title("Confusion Matrix", fontsize=10, color="#263D5B")
                plt.tight_layout()
                st.pyplot(fig_cm, use_container_width=True)
                plt.close(fig_cm)
            except Exception:
                pass

        # Download results
        csv_out = df_out.to_csv(index=False)
        st.download_button("⬇️ Download Results CSV", csv_out,
                           "trustguard_batch_results.csv", "text/csv",
                           use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown("### 📊 Model Performance Dashboard")

    comp_df = load_model_comparison()
    if comp_df is not None:
        st.markdown("#### All Models — Test Metrics")
        disp_cols = ["Model","Test Precision","Test Recall","Test F1","Test AUC-ROC","Test Avg Prec"]
        st.dataframe(comp_df[[c for c in disp_cols if c in comp_df.columns]],
                     use_container_width=True, hide_index=True)

        # Radar / grouped bar
        metrics = ["Test Precision","Test Recall","Test F1","Test AUC-ROC"]
        avail   = [m for m in metrics if m in comp_df.columns]
        fig_b, ax_b = plt.subplots(figsize=(10, 4))
        fig_b.patch.set_facecolor("#f8fbff")
        ax_b.set_facecolor("#f8fbff")
        x     = np.arange(len(avail))
        width = 0.18
        colors_m = ["#49B6E5","#263D5B","#D97706","#DC2626"]
        for i, (_, row_m) in enumerate(comp_df.iterrows()):
            vals = [row_m[m] for m in avail]
            ax_b.bar(x + i*width, vals, width, label=row_m["Model"],
                     color=colors_m[i % len(colors_m)], edgecolor="white", linewidth=0.6)
        ax_b.set_xticks(x + width*1.5)
        ax_b.set_xticklabels([m.replace("Test ","") for m in avail], fontsize=9, color="#263D5B")
        ax_b.set_ylim(0, 1.08)
        ax_b.set_title("Model Comparison — Test Metrics", fontsize=12, color="#263D5B", fontweight="bold")
        ax_b.legend(fontsize=8, framealpha=0.4)
        ax_b.tick_params(colors="#263D5B", labelsize=9)
        for sp in ax_b.spines.values(): sp.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    st.markdown("---")
    st.markdown("#### Saved Visualisations")

    plot_map = {
        "ROC & PR Curves":           "roc_pr_curves.png",
        "Confusion Matrices":         "confusion_matrices_all_models.png",
        "Feature Importance":         "feature_importance_comparison.png",
        "Fraud Probability Dist.":    "fraud_prob_distribution_all_models.png",
        "Cost-Benefit Analysis":      "cost_benefit_analysis.png",
        "Classification Report Heatmap": "classification_report_heatmap.png",
        "Model Metrics Lineplot":     "model_metrics_lineplot.png",
        "Models Comparison Heatmap":  "models_comparison_heatmap.png",
    }
    cols = st.columns(2)
    for i, (title, fname) in enumerate(plot_map.items()):
        fpath = os.path.join(PLOTS_DIR, fname)
        if os.path.exists(fpath):
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(fpath, use_container_width=True)

    # XGBoost metrics detail
    xgb_path = os.path.join(METRICS_DIR, "xgboost_metrics.json")
    if os.path.exists(xgb_path):
        with open(xgb_path) as f:
            xgb_m = json.load(f)
        st.markdown("---")
        st.markdown("#### XGBoost (Deployed Model) — Full Metrics")
        m_cols = st.columns(5)
        keys = [("test_auc_roc","AUC-ROC"),("test_recall","Recall"),
                ("test_precision","Precision"),("test_f1","F1"),("test_avg_prec","Avg Prec")]
        for j, (k, lbl) in enumerate(keys):
            m_cols[j].metric(lbl, f"{xgb_m.get(k, 0):.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ABLATION STUDY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 Ablation Study":
    st.markdown("### 🔬 Ablation Study")
    st.markdown("Component-level analysis of the pipeline — showing which parts contribute most.")

    ab_df = load_ablation()
    if ab_df is not None:
        st.dataframe(ab_df, use_container_width=True, hide_index=True)

        # Bar chart — F1 comparison
        fig_a, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig_a.patch.set_facecolor("#f8fbff")
        for ax in axes: ax.set_facecolor("#f8fbff")

        pal = ["#49B6E5" if "full" in c.lower() or "current" in c.lower()
               else "#263D5B" for c in ab_df["Condition"]]

        # CV F1
        axes[0].barh(ab_df["Condition"], ab_df["CV F1"], color=pal,
                     edgecolor="white", linewidth=0.6)
        axes[0].set_title("CV F1 by Condition", fontsize=10, color="#263D5B", fontweight="bold")
        axes[0].set_xlim(0, 1.1)
        axes[0].tick_params(labelsize=7, colors="#263D5B")
        for sp in axes[0].spines.values(): sp.set_visible(False)

        # Test AUC
        axes[1].barh(ab_df["Condition"], ab_df["Test AUC"], color=pal,
                     edgecolor="white", linewidth=0.6)
        axes[1].set_title("Test AUC by Condition", fontsize=10, color="#263D5B", fontweight="bold")
        axes[1].set_xlim(0.98, 1.002)
        axes[1].tick_params(labelsize=7, colors="#263D5B")
        for sp in axes[1].spines.values(): sp.set_visible(False)

        plt.tight_layout()
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)

    st.markdown("---")
    st.markdown("#### Ablation Plots")
    ab_plots = {
        "Ablation Summary":     "ablation_study.png",
        "Ablation Heatmap":     "ablation_heatmap.png",
        "SMOTE Trend":          "ablation_smote_trend.png",
        "PR Scatter":           "ablation_pr_scatter.png",
        "Delta (Δ) Chart":      "ablation_delta.png",
    }
    ab_cols = st.columns(2)
    for i, (title, fname) in enumerate(ab_plots.items()):
        fpath = os.path.join(ABLATION_DIR, fname)
        if os.path.exists(fpath):
            with ab_cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(fpath, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.markdown("### ℹ️ About TrustGuard")
    st.markdown("""
    <div class="tg-card">
    <b>TrustGuard</b> is an AI-powered fraud detection system built on the PaySim synthetic financial
    transaction dataset. It combines classical ML, deep learning, XAI explainability (SHAP), and a
    Retrieval-Augmented Generation (RAG) pipeline grounded in SBP (State Bank of Pakistan)
    regulatory documents.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### System Architecture")
        st.markdown("""
        - **Data**: PaySim synthetic transactions (6.3M rows)
        - **Feature Engineering**: balance diffs, amount ratios, one-hot type encoding
        - **Models**: XGBoost ★, Random Forest, Neural Network, Logistic Regression
        - **Class Imbalance**: SMOTE + fraud simulation augmentation
        - **XAI**: SHAP TreeExplainer (fallback: feature importance)
        - **RAG**: ChromaDB + sentence-transformers + BM25 hybrid retrieval + GPT-4o-mini
        - **Regulatory Corpus**: SBP AML/CFT, Branchless Banking, SME PRs documents
        """)
    with c2:
        st.markdown("#### Deployed Model (XGBoost)")
        _, _, meta = load_deployment_model()
        st.json(meta)

    st.markdown("---")
    st.markdown("#### Deliverable 3 Checklist")
    checks = [
        ("✅", "Model hosted and serving real-time inference"),
        ("✅", "Frontend communicates with backend model"),
        ("✅", "CSV upload + batch inference supported"),
        ("✅", "XAI heatmap (SHAP / feature importance)"),
        ("✅", "RAG regulatory justification beside heatmap"),
        ("✅", "Fraud probability + confidence score displayed"),
        ("✅", "Sidebar navigation"),
        ("✅", "Export / download reports"),
        ("✅", "Model performance metrics dashboard"),
        ("✅", "Ablation study visualisation"),
    ]
    for icon, text in checks:
        st.markdown(f"{icon} {text}")

    st.markdown("---")
    st.markdown("#### RAG Known Limitations")
    st.warning("""
    - Requires a valid **OpenAI API key** (gpt-4o-mini). Without it, the regulatory report section is skipped.
    - ChromaDB stores ~100 SBP document chunks. For best results, ensure `chroma_db/` is committed to the repo.
    - If OpenAI rate-limits are hit (as seen in `TXN-CRITICAL-001.txt`), the RAG section will show a rate-limit error.
    """)
