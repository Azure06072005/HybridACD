"""
Page 3: Results Dashboard
Experimental data: 5 models × Basic vs HybridACD on 242 questions
"""

import streamlit as st
import sys, os, json
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT, DEMO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_utils import init_session, apply_api_config, SIDEBAR_CSS, render_sidebar_api
from translations import t

st.set_page_config(page_title="Results Dashboard", page_icon="📊", layout="wide")
init_session()

st.markdown(SIDEBAR_CSS + """
<style>
.kpi-card { background: var(--panel2); border: 1px solid var(--line); border-radius: 12px; padding: 20px; text-align: center; }
.kpi-val { font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
.kpi-label { font-size: 0.8rem; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.info-box { background: var(--blue-dim); border-left: 4px solid var(--blue); padding: 16px; border-radius: 0 8px 8px 0; margin-bottom: 24px; font-size: 0.95rem; color: var(--text); line-height: 1.5; }
.info-box b { color: var(--blue); }
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

render_sidebar_api("Results Dashboard")

st.markdown(f"""
<div class="page-header">
  <div class="page-title">{t("results_title")}</div>
  <div class="page-sub">{t("results_sub")}</div>
</div>
""", unsafe_allow_html=True)

# ── Data Loading Placeholder ─────────────────────────────────────────────
# Normally loads from src/data/forecasts/Recent_evaluation. We use the previous mock/loading logic here.
@st.cache_data
def load_data():
    return {
        "gpt-4o-mini": {"display": "GPT-4o-mini", "b": {"avs": 0.0307, "bs": 0.205}, "h": {"avs": 0.0007, "bs": 0.200}},
        "gemini-2.5-flash": {"display": "Gemini 2.5 Flash", "b": {"avs": 0.1116, "bs": 0.141}, "h": {"avs": 0.0087, "bs": 0.130}},
        "mistral-medium": {"display": "Mistral Medium", "b": {"avs": 0.0792, "bs": 0.202}, "h": {"avs": 0.0087, "bs": 0.167}},
        "mistral-small": {"display": "Mistral Small", "b": {"avs": 0.0740, "bs": 0.211}, "h": {"avs": 0.0023, "bs": 0.202}},
        "minimax": {"display": "MiniMax M3", "b": {"avs": 0.1266, "bs": 0.123}, "h": {"avs": 0.0005, "bs": 0.116}}
    }
data = load_data()

# ── KPIs ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#34d399">-95.5%</div><div class="kpi-label">{t("avg_avs_reduction")}</div></div>', unsafe_allow_html=True)
with c2: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#60a5fa">MiniMax M3</div><div class="kpi-label">{t("best_brier")}</div></div>', unsafe_allow_html=True)
with c3: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#f472b6">GPT-4o-mini</div><div class="kpi-label">{t("best_avs")}</div></div>', unsafe_allow_html=True)
with c4: st.markdown(f'<div class="kpi-card"><div class="kpi-val" style="color:#fbbf24">242</div><div class="kpi-label">{t("ground_truth_q")}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([t("tab_avs"), t("tab_brier"), t("tab_raw")])

PLOT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(17,24,39,0.5)", font=dict(color="#9ca3af"), margin=dict(l=40, r=20, t=40, b=40), xaxis=dict(gridcolor="rgba(255,255,255,0.05)"), yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))

with tab1:
    st.markdown(f"""
    <div class="info-box">
        {t("avs_desc")}
    </div>
    """, unsafe_allow_html=True)
    
    models = [d["display"] for d in data.values()]
    b_avs = [d["b"]["avs"] for d in data.values()]
    h_avs = [d["h"]["avs"] for d in data.values()]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Basic Forecaster", x=models, y=b_avs, marker_color="#f87171"))
    fig.add_trace(go.Bar(name="HybridACD", x=models, y=h_avs, marker_color="#34d399"))
    fig.update_layout(**PLOT_LAYOUT, barmode="group", yaxis_title="AVS (Lower is better)", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown(f"""
    <div class="info-box">
        {t("brier_desc")}
    </div>
    """, unsafe_allow_html=True)
    
    b_bs = [d["b"]["bs"] for d in data.values()]
    h_bs = [d["h"]["bs"] for d in data.values()]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name="Basic Forecaster", x=models, y=b_bs, marker_color="#fbbf24"))
    fig2.add_trace(go.Bar(name="HybridACD", x=models, y=h_bs, marker_color="#60a5fa"))
    fig2.update_layout(**PLOT_LAYOUT, barmode="group", yaxis_title="Brier Score (Lower is better)", height=400)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown(f"#### {t('perf_matrix')}")
    df = pd.DataFrame([{
        "Model": d["display"],
        "Basic AVS": f"{d['b']['avs']:.4f}",
        "HybridACD AVS": f"{d['h']['avs']:.4f}",
        "Basic Brier": f"{d['b']['bs']:.3f}",
        "HybridACD Brier": f"{d['h']['bs']:.3f}"
    } for d in data.values()])
    st.dataframe(df, use_container_width=True, hide_index=True)

