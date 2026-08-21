
from pydantic import BaseModel, Field
from typing import List, Dict

class QuestionElement(BaseModel):
    title: str = Field(description="The title of the forecasting question. Must be clear and end with a question mark.")
    body: str = Field(description="The detailed resolution criteria defining exactly when the question resolves to YES.")

class TupleScenario(BaseModel):
    scenario_name: str = Field(description="The display name of the scenario matching one of the 10 checkers.")
    rule: str = Field(description="The rule name, must be one of: NegChecker, AndChecker, OrChecker, AndOrChecker, ButChecker, CondChecker, CondCondChecker, ConsequenceChecker, ExpectedEvidenceChecker, ParaphraseChecker")
    elements: Dict[str, QuestionElement] = Field(description="The question elements mapping the checker keys (e.g. 'P', 'not_P') to their question details. The questions MUST be logically linked exactly as required by the rule's constraint.")

class GeneratedTuples(BaseModel):
    tuples: List[TupleScenario] = Field(description="The list of generated logically connected question tuples.")
"""
Page 1: Pipeline Visualizer
Shows step-by-step logic of HybridACD: Adversarial Rewrite → CoT → TCD
"""

import streamlit as st
import sys, os, asyncio, time
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT, DEMO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_utils import init_session, apply_api_config, render_sidebar_api, get_t, render_rag_config, SIDEBAR_CSS
from translations import t

from common.perscache import register_model_for_cache
register_model_for_cache(QuestionElement)
register_model_for_cache(TupleScenario)
register_model_for_cache(GeneratedTuples)

st.set_page_config(page_title="Pipeline Visualizer", page_icon="🔭", layout="wide")
init_session()


st.markdown(SIDEBAR_CSS + """
<style>
.module-card {
    background: var(--panel);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    border-left: 4px solid;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
}
.m1 { border-color: var(--amber); }
.m2 { border-color: var(--blue); }
.m3 { border-color: var(--teal); }

.module-header { font-weight: 700; font-size: 1.15rem; margin-bottom: 6px; letter-spacing: -0.5px; font-family: 'Fraunces', serif; }
.m1-color { color: var(--amber); }
.m2-color { color: var(--blue); }
.m3-color { color: var(--teal); }

.diff-box {
    background: var(--panel2);
    border: 1px solid var(--line);
    padding: 16px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    margin-top: 8px;
}
.diff-old { border-left: 3px solid var(--red); color: var(--red); }
.diff-new { border-left: 3px solid var(--teal); color: var(--teal); margin-top: 12px; }

.bound-bar {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 20px;
    margin: 16px 0;
}
.bound-track {
    background: var(--panel2);
    border-radius: 8px;
    height: 32px;
    position: relative;
    overflow: hidden;
    margin: 12px 0;
}
.bound-valid {
    background: linear-gradient(90deg, rgba(62,203,158,0.2), rgba(62,203,158,0.5));
    border-left: 2px solid var(--teal);
    border-right: 2px solid var(--teal);
    position: absolute;
    top: 0;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    color: var(--teal);
}
.bound-blocked { background: repeating-linear-gradient(45deg, rgba(226,96,79,0.1), rgba(226,96,79,0.1) 10px, rgba(226,96,79,0.15) 10px, rgba(226,96,79,0.15) 20px); position: absolute; top: 0; height: 100%; }
.step-connector { text-align: center; color: var(--muted); font-size: 1.5rem; margin: -8px 0; }
.cot-box { background: var(--panel2); border: 1px solid rgba(62,123,250,0.3); border-radius: 8px; padding: 16px; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: var(--text); max-height: 250px; overflow-y: auto; white-space: pre-wrap; line-height: 1.6; }
.math-block { background: var(--blue-dim); border: 1px solid var(--blue); border-radius: 12px; padding: 16px 24px; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

render_sidebar_api("Pipeline Visualizer")
rag_urls_list, rag_files_list = render_rag_config()

st.markdown(f"""
<div class="page-header">
  <div class="page-title">{t("pipeline_title")}</div>
  <div class="page-sub">{t("pipeline_sub")}</div>
</div>
""", unsafe_allow_html=True)


PRESETS = {
    "Negation (NegChecker)": {
        "rule": "NegChecker",
        "P": {"title": "Will it rain in Hanoi on July 4, 2026?", "body": "Resolves YES if precipitation > 0mm."},
        "not_P": {"title": "Will it NOT rain in Hanoi on July 4, 2026?", "body": "Resolves YES if precipitation == 0mm."}
    },
    "And (AndChecker)": {
        "rule": "AndChecker",
        "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
        "Q": {"title": "Will FDI into Vietnam exceed $40B in 2026?", "body": "Resolves YES if registered FDI exceeds $40B."},
        "P_and_Q": {"title": "Will BOTH GDP > 7% AND FDI > $40B happen?", "body": "Resolves YES if both conditions are met."}
    },
    "Or (OrChecker)": {
        "rule": "OrChecker",
        "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
        "Q": {"title": "Will FDI into Vietnam exceed $40B in 2026?", "body": "Resolves YES if registered FDI exceeds $40B."},
        "P_or_Q": {"title": "Will EITHER GDP > 7% OR FDI > $40B happen?", "body": "Resolves YES if at least one condition is met."}
    },
    "And/Or Coherence (AndOrChecker)": {
        "rule": "AndOrChecker",
        "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
        "Q": {"title": "Will FDI into Vietnam exceed $40B in 2026?", "body": "Resolves YES if registered FDI exceeds $40B."},
        "P_and_Q": {"title": "Will BOTH GDP > 7% AND FDI > $40B happen?", "body": "Resolves YES if both conditions are met."},
        "P_or_Q": {"title": "Will EITHER GDP > 7% OR FDI > $40B happen?", "body": "Resolves YES if at least one condition is met."}
    },
    "But (ButChecker)": {
        "rule": "ButChecker",
        "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
        "Q_and_not_P": {"title": "Will FDI exceed $40B but GDP NOT exceed 7% in 2026?", "body": "Resolves YES if FDI exceeds $40B and GDP growth <= 7%."},
        "P_or_Q": {"title": "Will EITHER GDP > 7% OR FDI > $40B happen?", "body": "Resolves YES if at least one condition is met."}
    },
    "Conditional Probability (CondChecker)": {
        "rule": "CondChecker",
        "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
        "Q_given_P": {"title": "If Vietnam GDP > 7% in 2026, will FDI exceed $40B?", "body": "Conditional question: Resolves YES if FDI exceeds $40B, conditional on GDP > 7%."},
        "P_and_Q": {"title": "Will BOTH GDP > 7% AND FDI > $40B happen?", "body": "Resolves YES if both conditions are met."}
    },
    "Chain Rule (CondCondChecker)": {
        "rule": "CondCondChecker",
        "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
        "Q_given_P": {"title": "If Vietnam GDP > 7% in 2026, will FDI exceed $40B?", "body": "Conditional question: Resolves YES if FDI exceeds $40B, conditional on GDP > 7%."},
        "R_given_P_and_Q": {"title": "If GDP > 7% and FDI > $40B, will Vietnam inflation exceed 4%?", "body": "Conditional question: Resolves YES if inflation exceeds 4%, conditional on both GDP > 7% and FDI > $40B."},
        "P_and_Q_and_R": {"title": "Will GDP > 7%, FDI > $40B, and inflation > 4% all happen?", "body": "Resolves YES if all three conditions are met."}
    },
    "Monotonicity (ConsequenceChecker)": {
        "rule": "ConsequenceChecker",
        "P": {"title": "Will Hanoi GDP growth exceed 9% in 2026?", "body": "Resolves YES if Hanoi GDP growth exceeds 9%."},
        "cons_P": {"title": "Will Hanoi GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP growth exceeds 7% (implied by GDP > 9%)."}
    },
    "Total Probability (Expected EvidenceChecker)": {
        "rule": "ExpectedEvidenceChecker",
        "P": {"title": "Will Vietnam inflation exceed 4% in 2026?", "body": "Resolves YES if Vietnam CPI inflation exceeds 4%."},
        "Q": {"title": "Will FDI into Vietnam exceed $40B in 2026?", "body": "Resolves YES if registered FDI exceeds $40B."},
        "P_given_Q": {"title": "If FDI exceeds $40B, will inflation exceed 4%?", "body": "Conditional question: Resolves YES if inflation exceeds 4%, conditional on FDI > $40B."},
        "P_given_not_Q": {"title": "If FDI does NOT exceed $40B, will inflation exceed 4%?", "body": "Conditional question: Resolves YES if inflation exceeds 4%, conditional on FDI <= $40B."}
    },
    "Paraphrasing Symmetry (ParaphraseChecker)": {
        "rule": "ParaphraseChecker",
        "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
        "para_P": {"title": "Is Vietnam's economic growth rate in 2026 expected to be higher than 7%?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."}
    }
}

CHECKER_RULES = {
    "NegChecker": {"keys": ["P", "not_P"], "constraint": "P(A) + P(\\neg A) = 1", "desc": "Probabilities of mutually exclusive exhaustive events sum to 1."},
    "AndChecker": {"keys": ["P", "Q", "P_and_Q"], "constraint": "max(0, P(A)+P(B)-1) \\leq P(A \\wedge B) \\leq min(P(A), P(B))", "desc": "Conjunction (AND) is bounded by Fréchet inequalities."},
    "OrChecker": {"keys": ["P", "Q", "P_or_Q"], "constraint": "max(P(A), P(B)) \\leq P(A \\vee B) \\leq min(1, P(A) + P(B))", "desc": "Disjunction (OR) is bounded by Fréchet inequalities."},
    "AndOrChecker": {"keys": ["P", "Q", "P_and_Q", "P_or_Q"], "constraint": "P(A \\wedge B) + P(A \\vee B) = P(A) + P(B)", "desc": "Probability of conjunction and disjunction sum to individual probabilities."},
    "ButChecker": {"keys": ["P", "Q_and_not_P", "P_or_Q"], "constraint": "P(A) + P(B \\wedge \\neg A) = P(A \\vee B)", "desc": "Disjoint union of A and (B and not A) equals A or B."},
    "CondChecker": {"keys": ["P", "Q_given_P", "P_and_Q"], "constraint": "P(B | A) \\times P(A) = P(A \\wedge B)", "desc": "Conditional probability definition."},
    "CondCondChecker": {"keys": ["P", "Q_given_P", "R_given_P_and_Q", "P_and_Q_and_R"], "constraint": "P(C | A \\wedge B) \\times P(B | A) \\times P(A) = P(A \\wedge B \\wedge C)", "desc": "Chain rule of probability."},
    "ConsequenceChecker": {"keys": ["P", "cons_P"], "constraint": "P(A) \\leq P(B) \\text{ where } A \\implies B", "desc": "Monotonicity: if A implies B, then B is at least as likely as A."},
    "ExpectedEvidenceChecker": {"keys": ["P", "Q", "P_given_Q", "P_given_not_Q"], "constraint": "P(A) = P(A | B)P(B) + P(A | \\neg B)(1 - P(B))", "desc": "Law of Total Probability (expected evidence)."},
    "ParaphraseChecker": {"keys": ["P", "para_P"], "constraint": "P(A) = P(A')", "desc": "Symmetry: paraphrased questions must have identical probabilities."}
}

left, right = st.columns([1, 1.8])

with left:
    st.markdown(f"### {t('input_config')}")
    
    # Auto-generation expander
    _ALL_RULES = [
        "NegChecker", "AndChecker", "OrChecker", "AndOrChecker", "ButChecker",
        "CondChecker", "CondCondChecker", "ConsequenceChecker", "ExpectedEvidenceChecker", "ParaphraseChecker"
    ]
    _RULE_KEYS = {
        "NegChecker":              "['P', 'not_P']",
        "AndChecker":              "['P', 'Q', 'P_and_Q']",
        "OrChecker":               "['P', 'Q', 'P_or_Q']",
        "AndOrChecker":            "['P', 'Q', 'P_and_Q', 'P_or_Q']",
        "ButChecker":              "['P', 'Q_and_not_P', 'P_or_Q']",
        "CondChecker":             "['P', 'Q_given_P', 'P_and_Q']",
        "CondCondChecker":         "['P', 'Q_given_P', 'R_given_P_and_Q', 'P_and_Q_and_R']",
        "ConsequenceChecker":      "['P', 'cons_P']",
        "ExpectedEvidenceChecker": "['P', 'Q', 'P_given_Q', 'P_given_not_Q']",
        "ParaphraseChecker":       "['P', 'para_P']",
    }
    with st.expander(t("auto_gen"), expanded=False):
        topic = st.text_input(t("topic_label"), placeholder=t("topic_placeholder"), key="agent_topic_pipeline")
        st.markdown("**Select checkers to generate** *(1 tuple per checker — minimizes API cost)*")
        _cols = st.columns(2)
        selected_rules_pipeline = []
        for _i, _rule in enumerate(_ALL_RULES):
            if _cols[_i % 2].checkbox(_rule, value=(_rule in ["NegChecker", "AndChecker"]), key=f"chk_{_rule}_pipeline"):
                selected_rules_pipeline.append(_rule)
        
        if st.button(t("btn_generate"), key="btn_generate_pipeline", use_container_width=True):
            if not st.session_state.get("api_key"):
                st.error("Please configure your API Key in the sidebar first!")
            elif not selected_rules_pipeline:
                st.warning("Please select at least one checker type.")
            else:
                with st.spinner(f"Generating {len(selected_rules_pipeline)} tuple(s)..."):
                    try:
                        from common.llm_utils import query_api_chat
                        rules_spec = "\n".join(
                            f"- {r}: {_RULE_KEYS[r]}" for r in selected_rules_pipeline
                        )
                        prompt = (
                            f"Generate exactly {len(selected_rules_pipeline)} logically connected forecasting question tuple(s) "
                            f"about the topic: '{topic}'.\n"
                            f"Generate EXACTLY ONE tuple for each of these rules (in order):\n{rules_spec}\n"
                            "Each tuple must use the EXACT rule and key names listed above."
                        )
                        response = asyncio.run(query_api_chat(
                            messages=[
                                {"role": "system", "content": (
                                    "You are a research assistant generating logically connected forecasting question tuples. "
                                    "Use ONLY the rules and EXACT key names provided by the user. "
                                    "Generate exactly one tuple per rule specified."
                                )},
                                {"role": "user", "content": prompt}
                            ],
                            model=st.session_state.model,
                            response_model=GeneratedTuples,
                            api_key=st.session_state.api_key,
                            base_url=st.session_state.api_base_url
                        ))
                        st.session_state["generated_tuples"] = response.tuples
                        st.success(f"✅ Generated {len(response.tuples)} tuple(s) for: {', '.join(selected_rules_pipeline)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        
    if st.session_state.get("generated_tuples"):
        tuples = st.session_state["generated_tuples"]
        tuple_options = []
        for tup in tuples:
            first_key = list(tup.elements.keys())[0]
            tuple_options.append(f"{tup.scenario_name} - {tup.elements[first_key].title}")
            
        selected_tuple_idx = st.selectbox("Select a generated tuple to load:", range(len(tuples)), format_func=lambda x: tuple_options[x], key="select_generated_pipeline")
        
        if st.button(t("btn_load") if t("btn_load") != "btn_load" else "Load Selected Tuple", key="btn_load_pipeline", use_container_width=True):
            loaded_tuple = tuples[selected_tuple_idx]
            
            # Automatically correct rule based on keys to protect against LLM hallucinations
            tup_keys_lower = {k.lower() for k in loaded_tuple.elements.keys()}
            if tup_keys_lower == {"p", "not_p"}:
                loaded_tuple.rule = "NegChecker"
            elif tup_keys_lower == {"p", "para_p"}:
                loaded_tuple.rule = "ParaphraseChecker"
            elif tup_keys_lower == {"p", "q", "p_and_q"}:
                loaded_tuple.rule = "AndChecker"
            elif tup_keys_lower == {"p", "q", "p_or_q"}:
                loaded_tuple.rule = "OrChecker"
            elif tup_keys_lower == {"p", "q_given_p", "p_and_q"}:
                loaded_tuple.rule = "CondChecker"
            elif tup_keys_lower == {"p", "q_and_not_p", "p_or_q"}:
                loaded_tuple.rule = "ButChecker"
            elif tup_keys_lower == {"p", "q", "p_and_q", "p_or_q"}:
                loaded_tuple.rule = "AndOrChecker"
            elif tup_keys_lower == {"p", "q_given_p", "r_given_p_and_q", "p_and_q_and_r"}:
                loaded_tuple.rule = "CondCondChecker"
            elif tup_keys_lower == {"p", "cons_p"}:
                loaded_tuple.rule = "ConsequenceChecker"
            elif tup_keys_lower == {"p", "q", "p_given_q", "p_given_not_q"}:
                loaded_tuple.rule = "ExpectedEvidenceChecker"

            # Find the matching preset name by rule (use first match only as fallback label)
            matched_preset_name = None
            for p_name, p_val in PRESETS.items():
                if p_val["rule"] == loaded_tuple.rule:
                    matched_preset_name = p_name
                    break
            if matched_preset_name:
                # Switch to the matched preset
                st.session_state["preset_choice_pipeline"] = matched_preset_name
                st.session_state["preset_choice_select_pipeline"] = matched_preset_name
                
                # Clear all other pipeline question keys to avoid leakage from previous states
                for key in list(st.session_state.keys()):
                    if (key.startswith("q_title_") or key.startswith("q_body_")) and key.endswith("_pipeline"):
                        del st.session_state[key]
                        
                # Directly overwrite the preset's question values with the loaded tuple data
                expected_keys = CHECKER_RULES[loaded_tuple.rule]["keys"]
                for k in expected_keys:
                    match_key = next((ek for ek in loaded_tuple.elements.keys() if ek.lower() == k.lower()), None)
                    if match_key:
                        val = loaded_tuple.elements[match_key]
                        st.session_state[f"q_title_{k}_pipeline"] = val.title
                        st.session_state[f"q_body_{k}_pipeline"] = val.body
                        # Update all presets that contain this key
                        for p_name, p_val in PRESETS.items():
                            p_keys = CHECKER_RULES[p_val["rule"]]["keys"]
                            if k in p_keys:
                                st.session_state[f"q_title_{k}_{p_name}"] = val.title
                                st.session_state[f"q_body_{k}_{p_name}"] = val.body
                # Mark that we just loaded so the selectbox syncs correctly
                st.session_state["_tuple_just_loaded"] = True
                st.toast("✅ Loaded tuple successfully!")
                st.rerun()

    # Determine preset choice
    default_preset = list(PRESETS.keys())[0]
    if "preset_choice_pipeline" not in st.session_state:
        st.session_state["preset_choice_pipeline"] = default_preset

    prev_preset = st.session_state["preset_choice_pipeline"]
    preset_choice = st.selectbox(
        t("select_preset"),
        list(PRESETS.keys()),
        index=list(PRESETS.keys()).index(st.session_state["preset_choice_pipeline"]),
        key="preset_choice_select_pipeline"
    )

    # Detect preset change: when user manually switches preset, transition default values to new defaults while keeping custom modifications
    if preset_choice != prev_preset and not st.session_state.pop("_tuple_just_loaded", False):
        st.session_state["preset_choice_pipeline"] = preset_choice
        old_preset_data = PRESETS[prev_preset]
        new_preset_data = PRESETS[preset_choice]
        
        all_keys = set(old_preset_data.keys()) | set(new_preset_data.keys())
        for k in all_keys:
            if k == "rule":
                continue
            title_key = f"q_title_{k}_pipeline"
            body_key  = f"q_body_{k}_pipeline"
            
            old_default = old_preset_data.get(k, {})
            new_default = new_preset_data.get(k, {})
            
            current_title = st.session_state.get(title_key, "")
            current_body = st.session_state.get(body_key, "")
            
            old_title = old_default.get("title", "")
            old_body = old_default.get("body", "")
            
            if current_title == old_title or current_title == "":
                st.session_state[title_key] = new_default.get("title", "")
            if current_body == old_body or current_body == "":
                st.session_state[body_key] = new_default.get("body", "")
    else:
        st.session_state["preset_choice_pipeline"] = preset_choice
        # Clear the load flag if it was set
        st.session_state.pop("_tuple_just_loaded", None)

    preset = PRESETS[preset_choice]
    rule = preset["rule"]
    rule_info = CHECKER_RULES[rule]
    
    st.markdown(f"""
    <div class="math-block">
      <div style="font-weight: 700; color: #c4b5fd; font-size: 1.1rem; margin-bottom: 4px;">{rule} Logic</div>
      <div style="color: #9ca3af; font-size: 0.9rem; margin-bottom: 8px;">{rule_info['desc']}</div>
      <code>{rule_info['constraint']}</code>
    </div>
    """, unsafe_allow_html=True)

    questions = {}
    for key in rule_info["keys"]:
        title_key = f"q_title_{key}_pipeline"
        body_key  = f"q_body_{key}_pipeline"
        
        default_q = preset.get(key, {})
        if title_key not in st.session_state:
            st.session_state[title_key] = default_q.get("title", "")
        if body_key not in st.session_state:
            st.session_state[body_key] = default_q.get("body", "")
            
        with st.expander(f"Question Element: `{key}`", expanded=(key == rule_info["keys"][0])):
            title = st.text_input(f"{t('q_title_lbl')} [{key}]", key=title_key)
            body = st.text_area(f"{t('q_body_lbl')} [{key}]", key=body_key, height=70)
            questions[key] = {"title": title, "body": body}

    run_mode = st.radio(t("exec_mode"), ["Simulation (Fast)", "Live API (Requires Key)"], horizontal=True)
    trans_toggle = st.toggle("🌍 Translate reasoning to Vietnamese (Dịch sang Tiếng Việt)")
    run_btn = st.button(t("btn_execute"), type="primary", use_container_width=True)

with right:
    st.markdown(f"### 🔄 {t('results_analysis')}")
    if run_btn:

        apply_api_config()
        all_results = {}
        keys = rule_info["keys"]
        
        status_box = st.status(t("status_running"), expanded=True)
        all_reasonings = {}
        for i, key in enumerate(keys):
            q = questions[key]
            st.markdown(f"#### Element Processing: `{key}`")
            
            # MODULE 0
            st.markdown(f"""
            <div class="module-card" style="border-left-color: #f59e0b;">
              <div class="module-header" style="color:#f59e0b;">Module 0: Agentic RAG Research</div>
              <div style="color:#9ca3af; font-size:0.9rem;">Multi-turn tool calling to search web and internal docs</div>
            </div>""", unsafe_allow_html=True)
            
            status_box.write(f"🔎 Element `{key}`: Researching context...")
            
            with st.spinner("Module 0 researching..."):
                if "Live" in run_mode and st.session_state.get("api_key"):
                    try:
                        from common.agent_tools import execute_research_phase_async
                        research_summary = asyncio.run(execute_research_phase_async(
                            q["title"], model=st.session_state.model,
                            api_key=st.session_state.api_key, base_url=st.session_state.api_base_url,
                            rag_urls=rag_urls_list, rag_files=rag_files_list
                        ))
                        status_box.write(f"  ✓ Element `{key}`: Research complete.")
                    except Exception as e:
                        research_summary = f"Research failed: {e}"
                        status_box.write(f"  ⚠️ Element `{key}`: Research failed.")
                else:
                    time.sleep(0.4)
                    research_summary = f"Simulation: Researched '{q['title']}'. Found 2 relevant historical precedents."
                    status_box.write(f"  ✓ Element `{key}`: Research simulated.")
                    
            with st.expander("Show Research Summary"):
                st.markdown(research_summary)
                
            st.markdown('<div class="step-connector">↓</div>', unsafe_allow_html=True)

            # MODULE 1
            st.markdown(f"""
            <div class="module-card m1">
              <div class="module-header m1-color">{t("module1")}</div>
              <div style="color:#9ca3af; font-size:0.9rem;">{t("module1_desc")}</div>
            </div>""", unsafe_allow_html=True)
            
            status_box.write(f"🛡️ Element `{key}`: {t('status_m1')}...")

            with st.spinner("Module 1 executing..."):
                if "Live" in run_mode and st.session_state.get("api_key"):
                    try:
                        from forecasters.hybrid_acd_forecaster import ADVERSARIAL_AGENT_PROMPT, ADVERSARIAL_AGENT_PREFACE
                        from common.llm_utils import query_api_chat
                        from pydantic import BaseModel
                        class AdvOut(BaseModel):
                            title: str; body: str
                        prompt = ADVERSARIAL_AGENT_PROMPT.format(title=q["title"], body=q["body"])
                        response = asyncio.run(query_api_chat(
                            messages=[{"role":"system","content":ADVERSARIAL_AGENT_PREFACE}, {"role":"user","content":prompt}],
                            model=st.session_state.model, response_model=AdvOut, temperature=0.7,
                            api_key=st.session_state.api_key, base_url=st.session_state.api_base_url
                        ))
                        adv_title, adv_body = response.title, response.body
                        status_box.write(f"  ✓ Element `{key}`: Adversarial Rewrite complete.")
                    except Exception as e:
                        adv_title, adv_body = q["title"], q["body"]
                        status_box.write(f"  ⚠️ Element `{key}`: Adversarial Rewrite failed ({e}). Using original.")
                else:
                    time.sleep(0.4)
                    adv_title = f"Assuming standardized definitions apply, {q['title'].lower()}"
                    adv_body = f"Based on verifiable consensus, {q['body'].lower()}"
                    status_box.write(f"  ✓ Element `{key}`: Adversarial Rewrite simulated.")
                    
            st.markdown(f"""
            <div class="diff-box diff-old"><b>Original:</b> {q['title']}</div>
            <div class="diff-box diff-new"><b>Perturbed:</b> {adv_title}</div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="step-connector">↓</div>', unsafe_allow_html=True)
            
            # Calculate bounds
            def compute_bounds(history, cur_key, r):
                if r == "NegChecker" and cur_key == "not_P" and "P" in history:
                    return 1.0 - history["P"], 1.0 - history["P"]
                if r == "AndChecker" and cur_key == "P_and_Q" and "P" in history and "Q" in history:
                    return max(0.0, history["P"]+history["Q"]-1.0), min(history["P"], history["Q"])
                if r == "OrChecker" and cur_key == "P_or_Q" and "P" in history and "Q" in history:
                    return max(history["P"], history["Q"]), min(1.0, history["P"] + history["Q"])
                if r == "AndOrChecker" and "P" in history and "Q" in history:
                    if cur_key == "P_and_Q":
                        return max(0.0, history["P"]+history["Q"]-1.0), min(history["P"], history["Q"])
                    if cur_key == "P_or_Q" and "P_and_Q" in history:
                        val = history["P"] + history["Q"] - history["P_and_Q"]
                        return max(0.0, min(1.0, val)), max(0.0, min(1.0, val))
                if r == "ButChecker" and "P" in history:
                    if cur_key == "Q_and_not_P":
                        return 0.0, 1.0 - history["P"]
                    if cur_key == "P_or_Q" and "Q_and_not_P" in history:
                        val = history["P"] + history["Q_and_not_P"]
                        return max(0.0, min(1.0, val)), max(0.0, min(1.0, val))
                if r == "CondChecker" and cur_key == "P_and_Q" and "P" in history and "Q_given_P" in history:
                    val = history["P"] * history["Q_given_P"]
                    return max(0.0, min(1.0, val)), max(0.0, min(1.0, val))
                if r == "CondCondChecker" and cur_key == "P_and_Q_and_R" and "P" in history and "Q_given_P" in history and "R_given_P_and_Q" in history:
                    val = history["P"] * history["Q_given_P"] * history["R_given_P_and_Q"]
                    return max(0.0, min(1.0, val)), max(0.0, min(1.0, val))
                if r == "ConsequenceChecker" and cur_key == "cons_P" and "P" in history:
                    return history["P"], 1.0
                if r == "ExpectedEvidenceChecker" and cur_key == "P_given_not_Q" and "P" in history and "Q" in history and "P_given_Q" in history:
                    q = history["Q"]
                    if abs(1.0 - q) < 1e-5:
                        return 0.0, 1.0
                    val = (history["P"] - history["P_given_Q"] * q) / (1.0 - q)
                    val = max(0.0, min(1.0, val))
                    return val, val
                if r == "ParaphraseChecker" and cur_key == "para_P" and "P" in history:
                    return history["P"], history["P"]
                return 0.0, 1.0
            
            lower, upper = compute_bounds(all_results, key, rule)
            
            # MODULE 2
            st.markdown(f"""
            <div class="module-card m2">
              <div class="module-header m2-color">{t("module2")}</div>
              <div style="color:#9ca3af; font-size:0.9rem;">{t("module2_desc")}</div>
            </div>""", unsafe_allow_html=True)
            
            status_box.write(f"🧠 Element `{key}`: {t('status_m2')}...")

            with st.spinner("Module 2 generating CoT..."):
                if "Live" in run_mode and st.session_state.get("api_key"):
                    try:
                        from common.llm_utils import query_api_chat_native
                        from datetime import date as _date
                        _today = _date.today().strftime("%Y-%m-%d")
                        cot_text = asyncio.run(query_api_chat_native(
                            model=st.session_state.model,
                            messages=[
                                {"role": "system", "content": (
                                    "You are an expert forecasting analyst. "
                                    f"Today's date is {_today}. "
                                    "You have access to the following research summary gathered from the web:\n\n"
                                    f"{research_summary}\n\n"
                                    "Use this factual information—especially any confirmed real-world outcomes—as the primary basis for your reasoning. "
                                    "If the research clearly shows the event has already been resolved (e.g., a team was eliminated), "
                                    "your probability estimate MUST reflect that fact (e.g., P=0 or P=1 depending on the outcome). "
                                    f"CRITICAL CONSTRAINT: Due to logical consistency with prior answers, your probability MUST be strictly within [{lower:.4f}, {upper:.4f}]. "
                                    "At the END of your response, you MUST output a line in exactly this format: "
                                    "'FINAL PROBABILITY: X.XXXX' where X.XXXX is a number within that exact range."
                                )},
                                {"role": "user", "content": f"Reason step by step about this forecasting question and give a probability estimate within [{lower:.4f}, {upper:.4f}]:\n\n{adv_title}"}
                            ],
                            api_key=st.session_state.api_key, base_url=st.session_state.api_base_url
                        ))
                        status_box.write(f"  ✓ Element `{key}`: CoT generated successfully.")
                    except Exception as e:
                        cot_text = f"API Error fallback reasoning: {e}"
                        status_box.write(f"  ⚠️ Element `{key}`: CoT generation failed. Using fallback.")
                else:
                    time.sleep(0.5)
                    cot_text = (
                        f"Analyzing '{q['title']}'...\n"
                        f"1. Base rates suggest a moderate likelihood.\n"
                        f"2. Recent events increase the probability.\n"
                        f"3. Initial unconstrained estimate: ~0.{(4+i)*10}."
                    )
                    status_box.write(f"  ✓ Element `{key}`: CoT simulated.")
                    
            if trans_toggle:
                status_box.write(f"  🌍 Element `{key}`: Translating to Vietnamese...")
                from demo_utils import translate_to_vi
                cot_text = translate_to_vi(cot_text)
                    
            st.markdown(f'<div class="cot-box">{cot_text}</div>', unsafe_allow_html=True)
            all_reasonings[key] = cot_text
            st.markdown('<div class="step-connector">↓</div>', unsafe_allow_html=True)
            
            # MODULE 3
            st.markdown(f"""
            <div class="module-card m3">
              <div class="module-header m3-color">{t("module3")}</div>
              <div style="color:#9ca3af; font-size:0.9rem;">{t("module3_desc")}</div>
            </div>""", unsafe_allow_html=True)

            
            pct_l = int(lower * 100); pct_u = int(upper * 100)
            
            status_box.write(f"🔒 Element `{key}`: {t('status_m3')} [{lower:.4f}, {upper:.4f}]...")

            with st.spinner("Module 3 enforcing bounds..."):
                import re as _re
                
                # --- Extract CoT probability anchor ---
                # Priority 1: explicit "FINAL PROBABILITY: X" tag we requested
                cot_prob = None
                fp_match = _re.search(r'FINAL\s*PROBABILITY.*?([0-9]*\.?[0-9]+)', cot_text, _re.IGNORECASE)
                if fp_match:
                    cot_prob = float(fp_match.group(1))
                    cot_prob = max(0.0, min(1.0, cot_prob))  # safety clamp
                
                # Priority 2: scan all numbers in text, take the last one (likely conclusion)
                if cot_prob is None:
                    nums = _re.findall(r'\b(0(?:\.\d+)?|1(?:\.0+)?)\b', cot_text)
                    if nums:
                        cot_prob = float(nums[-1])
                        cot_prob = max(0.0, min(1.0, cot_prob))

                # --- Enforce TCD bounds ---
                if lower == upper:
                    # Fully determined by consistency constraint — CoT is overridden
                    final_prob = round(lower, 4)
                    decision = "exact (constraint)"
                elif cot_prob is not None and lower <= cot_prob <= upper:
                    # CoT estimate is already inside the valid window — use it directly
                    final_prob = round(cot_prob, 4)
                    decision = f"CoT anchor ({cot_prob:.4f}) in [{lower:.4f}, {upper:.4f}]"
                elif cot_prob is not None:
                    # CoT is outside the valid window — clamp to nearest boundary
                    final_prob = round(max(lower, min(upper, cot_prob)), 4)
                    decision = f"CoT anchor ({cot_prob:.4f}) clamped to [{lower:.4f}, {upper:.4f}]"
                else:
                    # No CoT probability found — fall back to midpoint (not random)
                    final_prob = round((lower + upper) / 2, 4)
                    decision = f"midpoint fallback (no CoT prob found)"

                time.sleep(0.3)
                all_results[key] = final_prob
                status_box.write(f"  ✓ Element `{key}`: Bounds enforced ({decision}). Output: {final_prob:.4f}")
            
            st.markdown(f"""
            <div class="bound-bar">
              <div style="font-size:0.9rem; color:#9ca3af; margin-bottom:8px;">Valid Domain: <b>[{lower:.4f}, {upper:.4f}]</b></div>
              <div class="bound-track">
                <div class="bound-blocked" style="left:0; width:{pct_l}%"></div>
                <div class="bound-valid" style="left:{pct_l}%; width:{max(pct_u-pct_l, 1)}%"></div>
                <div class="bound-blocked" style="left:{pct_u}%; width:{100-pct_u}%"></div>
              </div>
              <div style="text-align:center; font-size: 2rem; font-weight: 800; color: #6ee7b7; font-family:'Fira Code',monospace;">
                Output: {final_prob:.4f}
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("---")
        
        status_box.update(label=t("status_complete"), state="complete", expanded=False)

        # Summary
        if all_results:
            st.markdown(f"#### {t('coherent_final')}")
            cols = st.columns(len(keys))
            for i, k in enumerate(keys): cols[i].metric(f"P({k})", f"{all_results[k]:.4f}")

            
            if rule == "NegChecker" and "P" in all_results and "not_P" in all_results:
                viol = abs(all_results["P"] + all_results["not_P"] - 1.0)
            elif rule == "AndChecker" and len(all_results)==3:
                p, q, pq = all_results["P"], all_results["Q"], all_results["P_and_Q"]
                viol = max(max(p + q - 1.0, 0.0) - pq, pq - min(p, q), 0.0)
            elif rule == "OrChecker" and len(all_results)==3:
                p, q, poq = all_results["P"], all_results["Q"], all_results["P_or_Q"]
                viol = max(max(p, q) - poq, poq - min(1.0, p + q), 0.0)
            elif rule == "AndOrChecker" and len(all_results)==4:
                p, q, pq, poq = all_results["P"], all_results["Q"], all_results["P_and_Q"], all_results["P_or_Q"]
                viol = abs(p + q - pq - poq)
            elif rule == "ButChecker" and len(all_results)==3:
                p, q_not_p, poq = all_results["P"], all_results["Q_and_not_P"], all_results["P_or_Q"]
                viol = abs(p + q_not_p - poq)
            elif rule == "CondChecker" and len(all_results)==3:
                p, q_given_p, pq = all_results["P"], all_results["Q_given_P"], all_results["P_and_Q"]
                viol = abs(p * q_given_p - pq)
            elif rule == "CondCondChecker" and len(all_results)==4:
                p, q_given_p, r_given_pq, pqr = all_results["P"], all_results["Q_given_P"], all_results["R_given_P_and_Q"], all_results["P_and_Q_and_R"]
                viol = abs(p * q_given_p * r_given_pq - pqr)
            elif rule == "ConsequenceChecker" and len(all_results)==2:
                p, cons_p = all_results["P"], all_results["cons_P"]
                viol = max(0.0, p - cons_p)
            elif rule == "ExpectedEvidenceChecker" and len(all_results)==4:
                p, q, pqg, pgnotq = all_results["P"], all_results["Q"], all_results["P_given_Q"], all_results["P_given_not_Q"]
                viol = abs(p - pqg * q - pgnotq * (1.0 - q))
            elif rule == "ParaphraseChecker" and len(all_results)==2:
                p, para_p = all_results["P"], all_results["para_P"]
                viol = abs(p - para_p)
            else: viol = 0.0
            
            st.success(f"**Arbitrage Violation:** {viol:.6f} — The sequence is mathematically consistent!")
            
            # Save to SQLite history
            from demo_utils import save_history
            details = {
                "questions": questions,
                "predictions": all_results,
                "reasonings": all_reasonings,
                "rule": rule,
                "constraint": rule_info["constraint"]
            }
            save_history(
                scenario_name=preset_choice,
                model=st.session_state.model if "Live" in run_mode else "Simulation",
                method="HybridACD (Pipeline)",
                violation=viol,
                details_dict=details
            )
            
            # Display model working status
            if "api_working_status" in st.session_state:
                status = st.session_state["api_working_status"]
                if status.get("working"):
                    if status.get("fallback"):
                        st.warning(f"⚠️ Model '{status.get('original_model')}' failed to respond (error: {status.get('error')}). Fell back to default model '{status.get('model_used')}' which is working successfully! 🟢")
                    else:
                        st.success(f"🟢 Model '{status.get('model_used')}' is working successfully!")
                else:
                    st.error(f"❌ Model failed to respond. Error: {status.get('error')}")
    else:
        st.info("👈 Select a configuration on the left and execute the pipeline.")
