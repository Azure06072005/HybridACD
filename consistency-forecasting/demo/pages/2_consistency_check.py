"""
Page 2: Live Consistency Checker
Select a logic rule, input a tuple, run Basic vs HybridACD side-by-side
"""

import streamlit as st
import sys, os, asyncio, time, random
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT, DEMO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_utils import init_session, apply_api_config, render_sidebar_api, get_t, render_rag_config, SIDEBAR_CSS
from translations import t

from pydantic import BaseModel, Field
from typing import List, Dict
from common.perscache import register_model_for_cache

class QuestionElement(BaseModel):
    title: str = Field(description="The title of the forecasting question. Must be clear and end with a question mark.")
    body: str = Field(description="The detailed resolution criteria defining exactly when the question resolves to YES.")

class TupleScenario(BaseModel):
    scenario_name: str = Field(description="The display name of the scenario matching one of the 10 checkers.")
    rule: str = Field(description="The rule name, must be one of: NegChecker, AndChecker, OrChecker, AndOrChecker, ButChecker, CondChecker, CondCondChecker, ConsequenceChecker, ExpectedEvidenceChecker, ParaphraseChecker")
    elements: Dict[str, QuestionElement] = Field(description="The question elements mapping the checker keys (e.g. 'P', 'not_P') to their question details. The questions MUST be logically linked exactly as required by the rule's constraint.")

class GeneratedTuples(BaseModel):
    tuples: List[TupleScenario] = Field(description="The list of generated logically connected question tuples.")

register_model_for_cache(QuestionElement)
register_model_for_cache(TupleScenario)
register_model_for_cache(GeneratedTuples)

st.set_page_config(page_title="Consistency Checker", page_icon="🔍", layout="wide")
init_session()


st.markdown(SIDEBAR_CSS + """
<style>
.rule-card { background: var(--blue-dim); border: 1px solid var(--blue); border-radius: 12px; padding: 20px; margin-bottom: 20px; }
.rule-title { font-size: 1.2rem; font-weight: 700; color: var(--blue); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.rule-math { font-family: 'JetBrains Mono', monospace; color: var(--text); font-size: 1rem; margin-top: 8px; background: rgba(0,0,0,0.2); padding: 8px 12px; border-radius: 6px; display: inline-block; }

.compare-box { border-radius: 12px; padding: 24px; text-align: center; height: 100%; border: 1px solid; background: var(--panel2); }
.box-basic { border-color: var(--red); }
.box-hybrid { border-color: var(--teal); }
.box-title { font-size: 1.1rem; font-weight: 700; margin-bottom: 16px; text-transform: uppercase; letter-spacing: 1px; }
.box-basic .box-title { color: var(--red); }
.box-hybrid .box-title { color: var(--teal); }
.prob-val { font-size: 2.5rem; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin: 12px 0; }
.chip { padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 700; display: inline-block; }
.chip-bad { background: var(--red-dim); color: var(--red); border: 1px solid var(--red); }
.chip-good { background: var(--teal-dim); color: var(--teal); border: 1px solid var(--teal); }
</style>
""", unsafe_allow_html=True)

render_sidebar_api("Consistency Checker")
rag_urls_list, rag_files_list = render_rag_config()

st.markdown(f"""
<div class="page-header">
  <div class="page-title">{t("checker_title")}</div>
  <div class="page-sub">{t("checker_sub")}</div>
</div>
""", unsafe_allow_html=True)


CHECKER_DEFS = {
    "Negation (P + ¬P = 1)": {
        "rule": "NegChecker", "keys": ["P", "not_P"],
        "constraint": "P(A) + P(\\neg A) = 1", "emoji": "⊕",
        "desc": "The probability of an event and its complement must sum to 1.",
        "example": {
            "P": {"title": "Will it rain tomorrow?", "body": "Resolves YES if precipitation occurs."},
            "not_P": {"title": "Will it NOT rain tomorrow?", "body": "Resolves YES if NO precipitation occurs."}
        }
    },
    "Conjunction (AND)": {
        "rule": "AndChecker", "keys": ["P", "Q", "P_and_Q"],
        "constraint": "max(0, P(A)+P(B)-1) \\leq P(A \\wedge B) \\leq min(P(A), P(B))", "emoji": "∧",
        "desc": "Probability of conjunction (AND) is bounded by Fréchet inequalities.",
        "example": {
            "P": {"title": "Will Hanoi GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP growth > 7%."},
            "Q": {"title": "Will HCMC GDP growth exceed 7% in 2026?", "body": "Resolves YES if HCMC GDP growth > 7%."},
            "P_and_Q": {"title": "Will BOTH Hanoi and HCMC GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP > 7% AND HCMC GDP > 7%."}
        }
    },
    "Disjunction (OR)": {
        "rule": "OrChecker", "keys": ["P", "Q", "P_or_Q"],
        "constraint": "max(P(A), P(B)) \\leq P(A \\vee B) \\leq min(1, P(A) + P(B))", "emoji": "∨",
        "desc": "Probability of disjunction (OR) is bounded by Fréchet inequalities.",
        "example": {
            "P": {"title": "Will Hanoi GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP growth > 7%."},
            "Q": {"title": "Will HCMC GDP growth exceed 7% in 2026?", "body": "Resolves YES if HCMC GDP growth > 7%."},
            "P_or_Q": {"title": "Will EITHER Hanoi or HCMC GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP > 7% OR HCMC GDP > 7%."}
        }
    },
    "Conjunction & Disjunction": {
        "rule": "AndOrChecker", "keys": ["P", "Q", "P_and_Q", "P_or_Q"],
        "constraint": "P(A \\wedge B) + P(A \\vee B) = P(A) + P(B)", "emoji": "📊",
        "desc": "Probability of conjunction and disjunction sum to individual probabilities.",
        "example": {
            "P": {"title": "Will Hanoi GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP growth > 7%."},
            "Q": {"title": "Will HCMC GDP growth exceed 7% in 2026?", "body": "Resolves YES if HCMC GDP growth > 7%."},
            "P_and_Q": {"title": "Will BOTH Hanoi and HCMC GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP > 7% AND HCMC GDP > 7%."},
            "P_or_Q": {"title": "Will EITHER Hanoi or HCMC GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP > 7% OR HCMC GDP > 7%."}
        }
    },
    "But (Disjoint Union)": {
        "rule": "ButChecker", "keys": ["P", "Q_and_not_P", "P_or_Q"],
        "constraint": "P(A) + P(B \\wedge \\neg A) = P(A \\vee B)", "emoji": "➖",
        "desc": "Disjoint union of A and (B and not A) equals A or B.",
        "example": {
            "P": {"title": "Will Vietnam GDP growth exceed 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth > 7%."},
            "Q_and_not_P": {"title": "Will Vietnam GDP NOT exceed 7% BUT inflation exceed 4% in 2026?", "body": "Resolves YES if GDP <= 7% AND inflation > 4%."},
            "P_or_Q": {"title": "Will EITHER Vietnam GDP exceed 7% OR inflation exceed 4% in 2026?", "body": "Resolves YES if GDP > 7% OR inflation > 4%."}
        }
    },
    "Conditional Probability": {
        "rule": "CondChecker", "keys": ["P", "Q_given_P", "P_and_Q"],
        "constraint": "P(B | A) \\times P(A) = P(A \\wedge B)", "emoji": "🔑",
        "desc": "Conditional probability definition: P(B | A) * P(A) = P(A and B).",
        "example": {
            "P": {"title": "Will Vietnam GDP growth exceed 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth > 7%."},
            "Q_given_P": {"title": "If Vietnam GDP growth exceeds 7%, will Hanoi GDP exceed 8%?", "body": "Conditional on GDP > 7%, does Hanoi GDP > 8%?"},
            "P_and_Q": {"title": "Will Vietnam GDP exceed 7% AND Hanoi GDP exceed 8%?", "body": "Resolves YES if both occur."}
        }
    },
    "Chain Rule": {
        "rule": "CondCondChecker", "keys": ["P", "Q_given_P", "R_given_P_and_Q", "P_and_Q_and_R"],
        "constraint": "P(C | A \\wedge B) \\times P(B | A) \\times P(A) = P(A \\wedge B \\wedge C)", "emoji": "⛓️",
        "desc": "Chain rule of probability for three events.",
        "example": {
            "P": {"title": "Will Vietnam GDP growth exceed 7% in 2026?", "body": "Resolves YES if GDP > 7%."},
            "Q_given_P": {"title": "If GDP > 7%, will FDI exceed $40B?", "body": "Conditional on GDP > 7%."},
            "R_given_P_and_Q": {"title": "If GDP > 7% and FDI > $40B, will inflation exceed 4%?", "body": "Conditional on both GDP > 7% and FDI > $40B."},
            "P_and_Q_and_R": {"title": "Will GDP > 7%, FDI > $40B, and inflation > 4% in 2026?", "body": "Resolves YES if all three occur."}
        }
    },
    "Monotonicity (Implication)": {
        "rule": "ConsequenceChecker", "keys": ["P", "cons_P"],
        "constraint": "P(A) \\leq P(B) \\text{ where } A \\implies B", "emoji": "➔",
        "desc": "If event A implies event B, then B must be at least as likely as A.",
        "example": {
            "P": {"title": "Will Hanoi GDP growth exceed 9% in 2026?", "body": "Resolves YES if Hanoi GDP growth > 9%."},
            "cons_P": {"title": "Will Hanoi GDP growth exceed 7% in 2026?", "body": "Resolves YES if Hanoi GDP growth > 7% (implied by GDP > 9%)."}
        }
    },
    "Law of Total Probability": {
        "rule": "ExpectedEvidenceChecker", "keys": ["P", "Q", "P_given_Q", "P_given_not_Q"],
        "constraint": "P(A) = P(A | B)P(B) + P(A | \\neg B)(1 - P(B))", "emoji": "⚖️",
        "desc": "Expected evidence (Law of Total Probability).",
        "example": {
            "P": {"title": "Will Vietnam inflation exceed 4% in 2026?", "body": "Resolves YES if inflation > 4%."},
            "Q": {"title": "Will FDI into Vietnam exceed $40B in 2026?", "body": "Resolves YES if registered FDI > $40B."},
            "P_given_Q": {"title": "If FDI exceeds $40B, will inflation exceed 4%?", "body": "Conditional on FDI > $40B."},
            "P_given_not_Q": {"title": "If FDI does NOT exceed $40B, will inflation exceed 4%?", "body": "Conditional on FDI <= $40B."}
        }
    },
    "Paraphrase (P = P')": {
        "rule": "ParaphraseChecker", "keys": ["P", "para_P"],
        "constraint": "P(A) = P(A')", "emoji": "≡",
        "desc": "Semantically equivalent questions must receive identical probabilities.",
        "example": {
            "P": {"title": "Will Vietnam GDP > 7% in 2026?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."},
            "para_P": {"title": "Is Vietnam's economic growth rate in 2026 expected to be higher than 7%?", "body": "Resolves YES if Vietnam GDP growth exceeds 7%."}
        }
    }
}

left, right = st.columns([1, 1.8])

# Helper for bounds calculation during simulation
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

with left:
    st.markdown(f"### ⚙️ {t('input_config')}")
    
    # Auto-generation expander
    _ALL_RULES_CHK = [
        "NegChecker", "AndChecker", "OrChecker", "AndOrChecker", "ButChecker",
        "CondChecker", "CondCondChecker", "ConsequenceChecker", "ExpectedEvidenceChecker", "ParaphraseChecker"
    ]
    _RULE_KEYS_CHK = {
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
        topic = st.text_input(t("topic_label"), placeholder=t("topic_placeholder"), key="agent_topic_checker")
        st.markdown("**Select checkers to generate**")
        _cols_chk = st.columns(2)
        selected_rules_checker = []
        for _i, _rule in enumerate(_ALL_RULES_CHK):
            if _cols_chk[_i % 2].checkbox(_rule, value=(_rule in ["NegChecker", "AndChecker"]), key=f"chk_{_rule}_checker"):
                selected_rules_checker.append(_rule)
        
        if st.button(t("btn_generate"), key="btn_generate_checker", use_container_width=True):
            if not st.session_state.get("api_key"):
                st.error("Please configure your API Key in the sidebar first!")
            elif not selected_rules_checker:
                st.warning("Please select at least one checker type.")
            else:
                with st.spinner(f"Generating {len(selected_rules_checker)} tuple(s)..."):

                    try:
                        from common.llm_utils import query_api_chat
                        rules_spec = "\n".join(
                            f"- {r}: {_RULE_KEYS_CHK[r]}" for r in selected_rules_checker
                        )
                        prompt = (
                            f"Generate exactly {len(selected_rules_checker)} logically connected forecasting question tuple(s) "
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
                        st.success(f"✅ Generated {len(response.tuples)} tuple(s) for: {', '.join(selected_rules_checker)}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        
    if st.session_state.get("generated_tuples"):
        tuples = st.session_state["generated_tuples"]
        tuple_options = []
        for tup in tuples:
            first_key = list(tup.elements.keys())[0]
            tuple_options.append(f"{tup.scenario_name} - {tup.elements[first_key].title}")
            
        selected_tuple_idx = st.selectbox("Select a generated tuple to load:", range(len(tuples)), format_func=lambda x: tuple_options[x], key="select_generated_checker")
        
        if st.button(t("btn_load") if t("btn_load") != "btn_load" else "Load Selected Tuple", key="btn_load_checker", use_container_width=True):

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

            preset_name = None
            # Find the display name in CHECKER_DEFS that matches loaded_tuple.rule
            for p_name, p_val in CHECKER_DEFS.items():
                if p_val["rule"] == loaded_tuple.rule:
                    preset_name = p_name
                    break
            if preset_name:
                st.session_state["preset_choice_checker"] = preset_name
                st.session_state["preset_choice_select_checker"] = preset_name
                
                # Clear all other checker question keys to avoid leakage from previous states
                for key in list(st.session_state.keys()):
                    if (key.startswith("q_title_") or key.startswith("q_body_")) and key.endswith("_checker"):
                        del st.session_state[key]
                        
                expected_keys = CHECKER_DEFS[preset_name]["keys"]
                for k in expected_keys:
                    match_key = next((ek for ek in loaded_tuple.elements.keys() if ek.lower() == k.lower()), None)
                    if match_key:
                        val = loaded_tuple.elements[match_key]
                        st.session_state[f"q_title_{k}_checker"] = val.title
                        st.session_state[f"q_body_{k}_checker"] = val.body
                        # Update all presets that contain this key
                        for p_name, p_val in CHECKER_DEFS.items():
                            p_keys = p_val["keys"]
                            if k in p_keys:
                                st.session_state[f"q_title_{k}_{p_val['rule']}_checker"] = val.title
                                st.session_state[f"q_body_{k}_{p_val['rule']}_checker"] = val.body
                st.toast("Loaded tuple successfully!")
                st.rerun()

    default_preset = list(CHECKER_DEFS.keys())[0]
    if "preset_choice_checker" not in st.session_state:
        st.session_state["preset_choice_checker"] = default_preset
        
    prev_preset = st.session_state["preset_choice_checker"]
    selected = st.selectbox(t("select_preset"), list(CHECKER_DEFS.keys()), index=list(CHECKER_DEFS.keys()).index(st.session_state["preset_choice_checker"]), key="preset_choice_select_checker")

    # Detect preset change: transition default values while keeping custom questions
    if selected != prev_preset and not st.session_state.pop("_tuple_just_loaded", False):
        st.session_state["preset_choice_checker"] = selected
        old_preset_data = CHECKER_DEFS[prev_preset]["example"]
        new_preset_data = CHECKER_DEFS[selected]["example"]
        
        all_keys = set(old_preset_data.keys()) | set(new_preset_data.keys())
        for k in all_keys:
            title_key = f"q_title_{k}_checker"
            body_key  = f"q_body_{k}_checker"
            
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
        st.session_state["preset_choice_checker"] = selected
        st.session_state.pop("_tuple_just_loaded", None)

    rule_def = CHECKER_DEFS[selected]
    
    st.markdown(f"""
    <div class="rule-card">
      <div class="rule-title"><span>{rule_def['emoji']}</span> {rule_def['rule']}</div>
      <div style="color: #cbd5e1; font-size: 0.9rem;">{rule_def['desc']}</div>
      <div class="rule-math">{rule_def['constraint']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    questions = {}
    for k in rule_def["keys"]:
        title_key = f"q_title_{k}_checker"
        body_key = f"q_body_{k}_checker"
        
        ex = rule_def["example"][k]
        if title_key not in st.session_state:
            st.session_state[title_key] = ex["title"]
        if body_key not in st.session_state:
            st.session_state[body_key] = ex["body"]
            
        t_val = st.text_input(f"{t('q_title_lbl')} [{k}]", key=title_key)
        b_val = st.text_area(f"{t('q_body_lbl')} [{k}]", key=body_key, height=70)
        questions[k] = {"title": t_val, "body": b_val}
        
    run_mode = st.radio(t("exec_mode"), ["Simulation", "Live API"], horizontal=True)
    trans_toggle = st.toggle("🌍 Translate reasoning to Vietnamese (Dịch sang Tiếng Việt)")
    run_btn = st.button(t("btn_run_eval"), type="primary", use_container_width=True)


with right:
    st.markdown(f"{t('results_analysis')}")
    if run_btn:
        apply_api_config()
        keys = rule_def["keys"]
        basic_probs, hybrid_probs = {}, {}
        
        status_box = st.status(t("status_running"), expanded=True)

        
        basic_reasonings, hybrid_reasonings = {}, {}
        if "Live" in run_mode and st.session_state.get("api_key"):
            try:
                from forecasters.basic_forecaster import BasicForecaster
                from forecasters.hybrid_acd_forecaster import HybridACDForecaster
                from common.datatypes import ForecastingQuestion
                
                active_model = st.session_state.get("model", "gpt-4o-mini")
                
                fqs = {}
                for k in keys:
                    fqs[k] = ForecastingQuestion(
                        title=questions[k]["title"],
                        body=questions[k]["body"],
                        resolution_date=datetime.now(),
                        question_type="binary"
                    )
                    
                basic_forecaster = BasicForecaster(model=active_model)
                hybrid_forecaster = HybridACDForecaster(model=active_model, research_enabled=True)
                
                async def run_check():
                    status_box.write("🔮 Querying Basic Forecaster (Direct Elicitation)...")
                    b_results = {}
                    for k in keys:
                        status_box.write(f"  • Direct elicitation for `{k}`...")
                        b_forecast = await basic_forecaster.call_async(fqs[k], api_key=st.session_state.api_key, base_url=st.session_state.api_base_url)
                        b_results[k] = b_forecast
                    
                    status_box.write("🛡️ Querying HybridACD (Joint Elicitation with TCD)...")
                    h_results = await hybrid_forecaster.elicit_async(
                        fqs, 
                        rule=rule_def["rule"],
                        api_key=st.session_state.api_key, 
                        base_url=st.session_state.api_base_url,
                        rag_urls=rag_urls_list,
                        rag_files=rag_files_list
                    )
                    
                    return b_results, h_results
                
                b_results, h_results = asyncio.run(run_check())
                
                for k in keys:
                    basic_probs[k] = b_results[k].prob
                    hybrid_probs[k] = h_results[k].prob
                    
                    basic_reasonings[k] = "Basic Forecaster was queried directly for a probability value (no scratchpad reasoning)."
                    h_cot = h_results[k].metadata.get("chain_of_thought") if h_results[k].metadata else "No reasoning returned."
                    hybrid_reasonings[k] = h_cot
                
                status_box.update(label="✓ Evaluation complete!", state="complete", expanded=False)
            except Exception as e:
                status_box.update(label=f"❌ Evaluation failed: {e}", state="error", expanded=True)
                st.error(f"Live API execution failed: {e}. Falling back to simulation.")
                
                # Fallback simulation
                for k in keys:
                    base_prob = round(random.uniform(0.3, 0.7), 4)
                    basic_probs[k] = base_prob
                    basic_reasonings[k] = "Simulation: Basic Forecaster analyzed the question directly."
                
                for k in keys:
                    lower, upper = compute_bounds(hybrid_probs, k, rule_def["rule"])
                    if lower == upper:
                        hybrid_probs[k] = round(lower, 4)
                    else:
                        hybrid_probs[k] = round((lower + upper) / 2, 4)  # use midpoint, not random
                    hybrid_reasonings[k] = f"Simulation: HybridACD analyzed '{questions[k]['title']}' and reasoned that the probability lies within the logical constraints of {rule_def['rule']}."
        else:
            status_box.write("⏳ Running simulation...")
            for k in keys:
                time.sleep(0.3)
                base_prob = round(0.5, 4)  # neutral starting point for simulation
                basic_probs[k] = base_prob
                basic_reasonings[k] = "Simulation: Basic Forecaster analyzed the question directly."

            for k in keys:
                lower, upper = compute_bounds(hybrid_probs, k, rule_def["rule"])
                if lower == upper:
                    hybrid_probs[k] = round(lower, 4)
                else:
                    hybrid_probs[k] = round((lower + upper) / 2, 4)  # use midpoint, not random
                hybrid_reasonings[k] = f"Simulation: HybridACD analyzed '{questions[k]['title']}' and reasoned that the probability lies within the logical constraints of {rule_def['rule']}."
            
        if trans_toggle:
            status_box.write("🌍 Translating reasoning to Vietnamese...")
            from demo_utils import translate_to_vi
            for k in keys:
                hybrid_reasonings[k] = translate_to_vi(hybrid_reasonings[k])
                basic_reasonings[k] = translate_to_vi(basic_reasonings[k])
                
        status_box.update(label=t("status_complete"), state="complete", expanded=False)

                    
        def calc_viol(probs, rule):
            if rule == "NegChecker" and "P" in probs and "not_P" in probs:
                return abs(probs["P"] + probs["not_P"] - 1.0)
            elif rule == "AndChecker" and len(probs)==3:
                p, q, pq = probs["P"], probs["Q"], probs["P_and_Q"]
                return max(max(p + q - 1.0, 0.0) - pq, pq - min(p, q), 0.0)
            elif rule == "OrChecker" and len(probs)==3:
                p, q, poq = probs["P"], probs["Q"], probs["P_or_Q"]
                return max(max(p, q) - poq, poq - min(1.0, p + q), 0.0)
            elif rule == "AndOrChecker" and len(probs)==4:
                p, q, pq, poq = probs["P"], probs["Q"], probs["P_and_Q"], probs["P_or_Q"]
                return abs(p + q - pq - poq)
            elif rule == "ButChecker" and len(probs)==3:
                p, q_not_p, poq = probs["P"], probs["Q_and_not_P"], probs["P_or_Q"]
                return abs(p + q_not_p - poq)
            elif rule == "CondChecker" and len(probs)==3:
                p, q_given_p, pq = probs["P"], probs["Q_given_P"], probs["P_and_Q"]
                return abs(p * q_given_p - pq)
            elif rule == "CondCondChecker" and len(probs)==4:
                p, q_given_p, r_given_pq, pqr = probs["P"], probs["Q_given_P"], probs["R_given_P_and_Q"], probs["P_and_Q_and_R"]
                return abs(p * q_given_p * r_given_pq - pqr)
            elif rule == "ConsequenceChecker" and len(probs)==2:
                p, cons_p = probs["P"], probs["cons_P"]
                return max(0.0, p - cons_p)
            elif rule == "ExpectedEvidenceChecker" and len(probs)==4:
                p, q, pqg, pgnotq = probs["P"], probs["Q"], probs["P_given_Q"], probs["P_given_not_Q"]
                return abs(p - pqg * q - pgnotq * (1.0 - q))
            elif rule == "ParaphraseChecker" and len(probs)==2:
                p, para_p = probs["P"], probs["para_P"]
                return abs(p - para_p)
            return 0.0
            
        b_viol = calc_viol(basic_probs, rule_def["rule"])
        h_viol = calc_viol(hybrid_probs, rule_def["rule"])
        
        # Save to SQLite history
        from demo_utils import save_history
        save_history(
            scenario_name=selected,
            model=st.session_state.model if "Live" in run_mode else "Simulation",
            method="Basic Forecaster",
            violation=b_viol,
            details_dict={
                "questions": questions,
                "predictions": basic_probs,
                "reasonings": basic_reasonings,
                "rule": rule_def["rule"],
                "constraint": rule_def["constraint"]
            }
        )
        save_history(
            scenario_name=selected,
            model=st.session_state.model if "Live" in run_mode else "Simulation",
            method="HybridACD",
            violation=h_viol,
            details_dict={
                "questions": questions,
                "predictions": hybrid_probs,
                "reasonings": hybrid_reasonings,
                "rule": rule_def["rule"],
                "constraint": rule_def["constraint"]
            }
        )
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="compare-box box-basic">
              <div class="box-title">{t("basic_forecaster_val")}</div>
              <div style="color:#9ca3af; font-size:0.85rem;">Standard Prompting</div>
              <div style="margin-top:20px; font-size:0.9rem; color:#d1d5db;">{t("violation_score")}</div>
              <div class="prob-val" style="color:#fca5a5;">{b_viol:.4f}</div>
              <div class="chip {'chip-bad' if b_viol > 0.01 else 'chip-good'}">
                {('❌ ' + t('violated')) if b_viol > 0.01 else ('✅ ' + t('adherent'))}
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="compare-box box-hybrid">
              <div class="box-title">{t("hybridacd_val")}</div>
              <div style="color:#9ca3af; font-size:0.85rem;">TCD mathematically enforced</div>
              <div style="margin-top:20px; font-size:0.9rem; color:#d1d5db;">{t("violation_score")}</div>
              <div class="prob-val" style="color:#6ee7b7;">{h_viol:.4f}</div>
              <div class="chip {'chip-bad' if h_viol > 0.001 else 'chip-good'}">
                {('⚠️ ' + t('violated')) if h_viol > 0.001 else ('✅ ' + t('adherent'))}
              </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown(f"#### {t('individual_forecasts')}")

        for k in keys:
            cols = st.columns([1, 1, 1])
            cols[0].markdown(f"**`{k}`**")
            cols[1].metric("Basic", f"{basic_probs[k]:.4f}")
            cols[2].metric("HybridACD", f"{hybrid_probs[k]:.4f}")
            
        st.markdown("---")
        st.markdown("### 📝 AI Reasoning Output (Verify Logic & Recency)")
        for k in keys:
            with st.expander(f"📖 View step-by-step reasoning for element `{k}`: {questions[k]['title']}"):
                st.markdown("**HybridACD Reasoning:**")
                st.info(hybrid_reasonings.get(k, "No reasoning available."))
                st.markdown("**Basic Forecaster Output:**")
                st.code(f"Probability: {basic_probs[k]:.4f}\n{basic_reasonings.get(k, '')}")
            
        st.markdown("---")
        # Display model working status
        if "api_working_status" in st.session_state and "Live" in run_mode:
            status = st.session_state["api_working_status"]
            if status.get("working"):
                if status.get("fallback"):
                    st.warning(f"⚠️ Model '{status.get('original_model')}' failed to respond (error: {status.get('error')}). Fell back to default model '{status.get('model_used')}' which is working successfully! 🟢")
                else:
                    st.success(f"🟢 Model '{status.get('model_used')}' is working successfully!")
            else:
                st.error(f"❌ Model failed to respond. Error: {status.get('error')}")
    else:
        st.info("👈 Configure evaluation on the left.")
