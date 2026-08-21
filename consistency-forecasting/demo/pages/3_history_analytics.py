"""
Page 3: History & Research Analytics
View past runs, compare violations across models/methods, and export data.
"""

import streamlit as st
import sys, os, json
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT, DEMO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_utils import init_session, get_history, clear_history, SIDEBAR_CSS, render_sidebar_api
from translations import t

st.set_page_config(page_title="Research Analytics", page_icon="📊", layout="wide")
init_session()

st.markdown(SIDEBAR_CSS + """
<style>
.stat-card {
    background: var(--panel2);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stat-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--teal);
    margin-bottom: 4px;
    font-family: 'JetBrains Mono', monospace;
}
.stat-label {
    font-size: 0.85rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.detail-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 10px;
}
.detail-table th, .detail-table td {
    border: 1px solid var(--line);
    padding: 8px 12px;
    font-size: 0.85rem;
    text-align: left;
}
.detail-table th {
    background: var(--panel);
    color: var(--blue);
}
</style>
""", unsafe_allow_html=True)

render_sidebar_api("Research Analytics")

st.markdown(f"""
<div class="page-header">
  <div class="page-title">{t("history_title")}</div>
  <div class="page-sub">{t("history_sub")}</div>
</div>
""", unsafe_allow_html=True)

# Fetch history
df = get_history()

if df.empty:
    st.info(t("no_history"))
else:
    # KPI Section
    st.markdown(t("key_metrics"))
    c1, c2, c3, c4 = st.columns(4)
    
    total_runs = len(df)
    models_tested = df['model'].nunique()
    
    # Calculate average violations
    hybrid_df = df[df['method'].str.contains("Hybrid", case=False, na=False)]
    basic_df = df[df['method'].str.contains("Basic", case=False, na=False)]
    
    avg_basic_viol = basic_df['violation'].mean() if not basic_df.empty else 0.0
    avg_hybrid_viol = hybrid_df['violation'].mean() if not hybrid_df.empty else 0.0
    
    c1.markdown(f"""
    <div class="stat-card">
      <div class="stat-val" style="color:#c4b5fd;">{total_runs}</div>
      <div class="stat-label">{t("total_runs")}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c2.markdown(f"""
    <div class="stat-card">
      <div class="stat-val" style="color:#60a5fa;">{models_tested}</div>
      <div class="stat-label">{t("llms_evaluated")}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c3.markdown(f"""
    <div class="stat-card">
      <div class="stat-val" style="color:#f87171;">{avg_basic_viol:.4f}</div>
      <div class="stat-label">{t("avg_basic_viol")}</div>
    </div>
    """, unsafe_allow_html=True)
    
    c4.markdown(f"""
    <div class="stat-card">
      <div class="stat-val" style="color:#34d399;">{avg_hybrid_viol:.4f}</div>
      <div class="stat-label">{t("avg_hybrid_viol")}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison Chart
    st.markdown(t("consistency_by_method"))
    chart_data = df.groupby(['method'])['violation'].mean().reset_index()
    st.bar_chart(data=chart_data, x='method', y='violation', color='method')
    
    st.markdown("---")
    
    # History Table
    st.markdown(t("detailed_history"))
    
    # Export CSV
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=t("export_csv"),
        data=csv_data,
        file_name=f"consistency_forecasting_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render table with expanders for details
    for idx, row in df.iterrows():
        status_emoji = "🟢" if row['violation'] <= 0.001 else ("🟡" if row['violation'] <= 0.05 else "🔴")
        expander_title = f"{status_emoji} [{row['timestamp']}] {row['scenario_name']} | {row['model']} | {row['method']} | Violation: {row['violation']:.4f}"
        
        with st.expander(expander_title):
            try:
                details = json.loads(row['details_json'])
                
                # Render questions and predictions
                st.markdown(t("constraint_evaluated"))
                st.code(details.get("constraint", "N/A"))
                
                st.markdown(t("individual_element_details"))
                
                # Build HTML table
                table_html = f"""
                <table class="detail-table">
                  <thead>
                    <tr>
                      <th>{t("element")}</th>
                      <th>{t("forecast_question")}</th>
                      <th>{t("assigned_prob")}</th>
                    </tr>
                  </thead>
                  <tbody>
                """
                
                q_dict = details.get("questions", {})
                p_dict = details.get("predictions", {})
                reasonings_dict = details.get("reasonings", {})
                
                for key in q_dict.keys():
                    q_val = q_dict[key]
                    p_val = p_dict.get(key, "N/A")
                    r_val = reasonings_dict.get(key, "No reasoning logged.")
                    prob_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
                    
                    table_html += f"""
                    <tr>
                      <td><b>{key}</b></td>
                      <td>
                        <b>{q_val.get('title', '')}</b><br>
                        <span style="color:#94a3b8; font-size:0.8rem;">{q_val.get('body', '')}</span>
                        <details style="margin-top:8px; font-size:0.85rem; color:#cbd5e1;">
                          <summary style="cursor:pointer; color:#60a5fa;">{t("view_reasoning")}</summary>
                          <div style="background:rgba(0,0,0,0.2); padding:10px; border-radius:6px; margin-top:6px; white-space:pre-wrap; font-family:monospace;">{r_val}</div>
                        </details>
                      </td>
                      <td style="font-family:monospace; font-size:1rem; color:#6ee7b7;">{prob_str}</td>
                    </tr>
                    """
                    
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Failed to parse details: {e}")
                st.json(row['details_json'])

    st.markdown("---")
    st.markdown(t("db_management"))
    if st.button(t("clear_history_btn"), type="secondary", use_container_width=True):
        clear_history()
        st.success(t("clear_success"))
        st.rerun()

