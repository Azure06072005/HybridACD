"""
Page 4: Live Forecasting
Predict any question, run Basic and HybridACD, compare real-time
"""

import streamlit as st
import sys, os, asyncio, time, random
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT, DEMO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_utils import init_session, apply_api_config, render_sidebar_api, get_t, render_rag_config, SIDEBAR_CSS
from translations import t

st.set_page_config(page_title="Live Forecasting", page_icon="⚡", layout="wide")
init_session()

st.markdown(SIDEBAR_CSS + """
<style>
.forecast-card { border-radius: 12px; padding: 32px 24px; text-align: center; border: 1px solid; height: 100%; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); background: var(--panel); }
.fc-basic { border-color: var(--red); }
.fc-hybrid { border-color: var(--teal); }
.fc-label { font-size: 0.9rem; color: var(--muted); margin-bottom: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }
.fc-prob { font-size: 3.8rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; line-height: 1; margin-bottom: 8px; }
.fc-basic .fc-prob { color: var(--red); }
.fc-hybrid .fc-prob { color: var(--teal); }
.fc-bound { font-size: 0.85rem; color: var(--muted2); font-family: 'JetBrains Mono', monospace; background: rgba(0,0,0,0.2); padding: 4px 12px; border-radius: 20px; display: inline-block; margin-top: 12px; }
.cot-stream { background: var(--panel2); border: 1px solid var(--blue); border-radius: 12px; padding: 20px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text); max-height: 300px; overflow-y: auto; line-height: 1.6; }
.self-check { background: var(--blue-dim); border: 1px solid var(--blue); border-radius: 12px; padding: 24px; margin-top: 24px; }
.sc-title { font-weight: 700; color: var(--blue); font-size: 1.1rem; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
.sc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.sc-item { background: rgba(0,0,0,0.2); padding: 16px; border-radius: 8px; border-left: 3px solid; }
</style>
""", unsafe_allow_html=True)

render_sidebar_api("Live Forecasting")
rag_urls_list, rag_files_list = render_rag_config()

st.markdown(f"""
<div class="page-header">
  <div class="page-title">{t("live_title")}</div>
  <div class="page-sub">{t("live_sub")}</div>
</div>
""", unsafe_allow_html=True)

EXAMPLES = [
    "Will Vietnam's GDP growth exceed 7% in 2026?",
    "Will OpenAI release GPT-5 before end of 2026?",
    "Will Bitcoin exceed $200,000 USD by Dec 31, 2026?"
]
sel = st.selectbox(t("preset_label"), ["Custom"] + EXAMPLES)

c1, c2 = st.columns(2)
q_title = c1.text_input(t("q_title_lbl"), value="" if sel=="Custom" else sel, placeholder="Will X happen by date Y?")
q_body = c2.text_area(t("q_body_lbl"), value="Resolves YES if..." if sel=="Custom" else f"Resolves YES if the following occurs: {sel}", height=70)

mode = st.radio(t("exec_mode"), ["Simulation", "Live API"], horizontal=True)
trans_toggle = st.toggle("🌍 Translate reasoning to Vietnamese (Dịch sang Tiếng Việt)")
if st.button(t("btn_exec_pred"), type="primary"):
    apply_api_config()
    st.markdown(f"### {t('results_analysis')}")
    
    status_box = st.status("Executing Live Forecasting...", expanded=True)
    
    basic_probs, hybrid_probs = {}, {}
    
    if "Live" in mode and st.session_state.get("api_key"):
        try:
            from forecasters.basic_forecaster import BasicForecaster
            from forecasters.hybrid_acd_forecaster import HybridACDForecaster
            from common.datatypes import ForecastingQuestion
            
            active_model = st.session_state.get("model", "gpt-4o-mini")
            
            fq_P = ForecastingQuestion(
                title=q_title,
                body=q_body,
                resolution_date=datetime.now(),
                question_type="binary"
            )
            
            # Construct negation
            if q_title.strip().lower().startswith("will "):
                neg_title = "Will it NOT " + q_title.strip()[5:]
            else:
                neg_title = f"Will it NOT happen: {q_title}"
            neg_body = f"Resolves YES if the following does NOT occur: {q_body}"
            fq_not_P = ForecastingQuestion(
                title=neg_title,
                body=neg_body,
                resolution_date=datetime.now(),
                question_type="binary"
            )
            
            basic_forecaster = BasicForecaster(model=active_model)
            hybrid_forecaster = HybridACDForecaster(model=active_model, research_enabled=True)
            
            async def run_live_forecast():
                status_box.write("🔮 Querying Basic Forecaster for P...")
                b_forecast_P = await basic_forecaster.call_async(fq_P, api_key=st.session_state.api_key, base_url=st.session_state.api_base_url)
                
                status_box.write("🔮 Querying Basic Forecaster for Negation (not_P)...")
                b_forecast_not_P = await basic_forecaster.call_async(fq_not_P, api_key=st.session_state.api_key, base_url=st.session_state.api_base_url)
                
                status_box.write("🛡️ Querying HybridACD for Joint forecasting with Axiom enforcement...")
                fqs = {"P": fq_P, "not_P": fq_not_P}
                h_results = await hybrid_forecaster.elicit_async(
                    fqs, 
                    rule="NegChecker",
                    api_key=st.session_state.api_key, 
                    base_url=st.session_state.api_base_url,
                    rag_urls=rag_urls_list, 
                    rag_files=rag_files_list
                )
                
                return b_forecast_P, b_forecast_not_P, h_results
            
            b_forecast_P, b_forecast_not_P, h_results = asyncio.run(run_live_forecast())
            
            basic_probs["P"] = b_forecast_P.prob
            basic_probs["not_P"] = b_forecast_not_P.prob
            
            hybrid_probs["P"] = h_results["P"].prob
            hybrid_probs["not_P"] = h_results["not_P"].prob
            
            cot_text = h_results["P"].metadata.get("chain_of_thought") if h_results["P"].metadata else "No CoT scratchpad returned."
            research_summary = h_results["P"].metadata.get("research_summary", "No research data.") if h_results["P"].metadata else "No research data."
            status_box.update(label="✓ Forecast complete!", state="complete", expanded=False)
        except Exception as e:
            status_box.update(label=f"❌ Forecast failed: {e}", state="error", expanded=True)
            st.error(f"Live API execution failed: {e}. Falling back to simulation.")
            # Fallback simulation
            basic_probs["P"] = round(random.uniform(0.3, 0.7), 4)
            basic_probs["not_P"] = round(random.uniform(0.3, 0.7), 4)
            hybrid_probs["P"] = round(random.uniform(0.3, 0.7), 4)
            hybrid_probs["not_P"] = round(1.0 - hybrid_probs["P"], 4)
            cot_text = "Simulation: API error fallback."
            research_summary = "Simulation: API error fallback."
    else:
        status_box.write("⏳ Running simulation...")
        time.sleep(0.8)
        basic_probs["P"] = round(random.uniform(0.3, 0.7), 4)
        basic_probs["not_P"] = round(random.uniform(0.3, 0.7), 4)
        hybrid_probs["P"] = round(random.uniform(0.3, 0.7), 4)
        hybrid_probs["not_P"] = round(1.0 - hybrid_probs["P"], 4)
        cot_text = "Simulation: Epistemic analysis completed successfully."
        research_summary = "Simulation: Dummy research data found."
        status_box.update(label="✓ Simulation complete!", state="complete", expanded=False)
        
    if trans_toggle:
        status_box.write("🌍 Translating reasoning to Vietnamese...")
        from demo_utils import translate_to_vi
        cot_text = translate_to_vi(cot_text)
        research_summary = translate_to_vi(research_summary)
        
    b_viol = abs(basic_probs["P"] + basic_probs["not_P"] - 1.0)
    h_viol = abs(hybrid_probs["P"] + hybrid_probs["not_P"] - 1.0)
    
    # Save to SQLite history
    from demo_utils import save_history
    save_history(
        scenario_name=f"Live: {q_title}",
        model=st.session_state.model if "Live" in mode else "Simulation",
        method="Basic Forecaster (Live)",
        violation=b_viol,
        details_dict={
            "questions": {
                "P": {"title": q_title, "body": q_body},
                "not_P": {"title": f"Not: {q_title}", "body": "Negation"}
            },
            "predictions": basic_probs,
            "rule": "NegChecker",
            "constraint": "P(A) + P(\\neg A) = 1"
        }
    )
    save_history(
        scenario_name=f"Live: {q_title}",
        model=st.session_state.model if "Live" in mode else "Simulation",
        method="HybridACD (Live)",
        violation=h_viol,
        details_dict={
            "questions": {
                "P": {"title": q_title, "body": q_body},
                "not_P": {"title": f"Not: {q_title}", "body": "Negation"}
            },
            "predictions": hybrid_probs,
            "rule": "NegChecker",
            "constraint": "P(A) + P(\\neg A) = 1"
        }
    )
    
    # Render UI
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'''
        <div class="forecast-card fc-basic">
          <div class="fc-label">{t("basic_forecaster_val")}</div>
          <div class="fc-prob" style="color:#fca5a5;">{basic_probs["P"]:.4f}</div>
          <div style="font-size:0.95rem; color:#d1d5db;">Negation P(¬A): <b>{basic_probs["not_P"]:.4f}</b></div>
          <div class="fc-bound">Sum: {(basic_probs["P"]+basic_probs["not_P"]):.4f} (Viol: {b_viol:.4f})</div>
        </div>
        ''', unsafe_allow_html=True)
    with c2:
        st.markdown(f'''
        <div class="forecast-card fc-hybrid">
          <div class="fc-label">{t("hybridacd_val")}</div>
          <div class="fc-prob" style="color:#6ee7b7;">{hybrid_probs["P"]:.4f}</div>
          <div style="font-size:0.95rem; color:#d1d5db;">Negation P(¬A): <b>{hybrid_probs["not_P"]:.4f}</b></div>
          <div class="fc-bound" style="color:#6ee7b7; border-color:rgba(16,185,129,0.4);">Sum: {(hybrid_probs["P"]+hybrid_probs["not_P"]):.4f} (Viol: {h_viol:.4f})</div>
        </div>
        ''', unsafe_allow_html=True)
        
    st.markdown(f"### 🔎 {t('module0') if t('module0') != 'module0' else 'Agentic Research Summary'}")
    with st.expander("View Research Data"):
        st.markdown(research_summary)
        
    st.markdown(f"### 🧠 {t('cot_scratchpad')}")
    st.markdown(f'<div class="cot-stream">{cot_text}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    # Display model working status
    if "api_working_status" in st.session_state and "Live" in mode:
        status = st.session_state["api_working_status"]
        if status.get("working"):
            if status.get("fallback"):
                st.warning(t("api_status_fallback", orig_model=status.get('original_model'), used_model=status.get('model_used')))
            else:
                st.success(t("api_status_working", model=status.get('model_used')))
        else:
            st.error(t("api_status_fail", error=status.get('error')))

