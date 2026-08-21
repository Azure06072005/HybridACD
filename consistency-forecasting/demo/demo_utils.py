"""
Shared utilities for HybridACD demo — config injection, session init, and database history
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
import streamlit as st

CONFIG_FILE = ".demo_config.json"
DB_FILE = "demo/demo_history.db"

# Lazy import translation function to avoid circular import if any
def get_t():
    from translations import t
    return t


def init_db():
    """Initialize the SQLite database and create the history table if it doesn't exist."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                scenario_name TEXT,
                model TEXT,
                method TEXT,
                violation REAL,
                details_json TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database initialization error: {e}")

def save_history(scenario_name: str, model: str, method: str, violation: float, details_dict: dict):
    """Save a forecast run to the history database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        timestamp = datetime.now().isoformat()
        details_json = json.dumps(details_dict, ensure_ascii=False)
        cursor.execute("""
            INSERT INTO history (timestamp, scenario_name, model, method, violation, details_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (timestamp, scenario_name, model, method, violation, details_json))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to save history: {e}")

def get_history() -> pd.DataFrame:
    """Retrieve the forecast history as a pandas DataFrame."""
    try:
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql_query("SELECT id, timestamp, scenario_name, model, method, violation, details_json FROM history ORDER BY id DESC", conn)
        conn.close()
        if not df.empty:
            df['timestamp'] = df['timestamp'].apply(lambda x: datetime.fromisoformat(x).strftime('%Y-%m-%d %H:%M:%S'))
        return df
    except Exception as e:
        print(f"Failed to retrieve history: {e}")
        return pd.DataFrame()

def clear_history():
    """Clear all records from the history database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to clear history: {e}")

def init_session():
    """Initialize session state from cache file or defaults."""
    defaults = {
        "api_key": os.getenv("OPENAI_API_KEY", ""),
        "api_base_url": os.getenv("OPENAI_BASE_URL", "https://llm.wokushop.com/v1"),
        "model": "gpt-4o-mini",
        "no_cache": True,
        "language": "vi",
    }

    
    # Load from cache file if exists
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                saved_config = json.load(f)
                defaults.update(saved_config)
        except Exception:
            pass

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
            
    # Add translation_model to defaults if not present
    if "translation_model" not in st.session_state:
        st.session_state["translation_model"] = "gpt-5.2"
            
    init_db()

def save_config_to_cache():
    """Save the current API settings to a local JSON cache file."""
    config_to_save = {
        "api_key": st.session_state.get("api_key", ""),
        "api_base_url": st.session_state.get("api_base_url", ""),
        "model": st.session_state.get("model", "gpt-4o-mini"),
        "translation_model": st.session_state.get("translation_model", "gpt-5.2"),
        "no_cache": st.session_state.get("no_cache", True),
        "rag_urls": st.session_state.get("rag_urls", ""),
        "rag_files": st.session_state.get("rag_files", ""),
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config_to_save, f)
    except Exception as e:
        print(f"Error saving config: {e}")

def apply_api_config():
    """Inject API config from session_state into os.environ (excluding sensitive credentials)."""
    # Note: We do NOT set OPENAI_API_KEY or OPENAI_BASE_URL globally to prevent session crosstalk.
    # Instead, we pass them dynamically.
    os.environ["OPENAI_INSTRUCTOR_MODE"] = "MD_JSON"
    os.environ["NO_CACHE"] = "True" if st.session_state.get("no_cache", True) else "False"
    if not st.session_state.get("no_cache", True):
        os.environ["LOCAL_CACHE"] = ".cache/"
    
    os.environ["MAX_CONCURRENT_QUERIES"] = "2"
    os.environ["OPENAI_JSON_STRICT"] = "False"
    os.environ["DISABLE_COSTLY"] = "True"
    os.environ["USE_LOGFIRE"] = "False"
    os.environ["LOGFIRE_IGNORE_NO_CONFIG"] = "True"

SIDEBAR_CSS = """
<style>
/* Hide Streamlit default sidebar navigation */
div[data-testid="stSidebarNav"] {
    display: none !important;
}

/* Sidebar Refinement */
section[data-testid="stSidebar"] {
    background: #10141D !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 1rem 0;
}
.sidebar-logo {
    text-align: center;
    padding: 10px 0 20px 0;
}
.sidebar-logo h2 {
    color: #3E7BFA;
    margin: 0;
    font-weight: 600;
    font-family: 'Fraunces', serif;
    font-size: 1.8rem;
    letter-spacing: -0.02em;
}
.sidebar-logo .subtitle {
    font-size: 0.75rem;
    color: #8892A6;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 2px;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.nav-section-title {
    font-size: 0.8rem;
    color: #5B6474;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace;
}
/* Main App Typography and Backgrounds */
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg:#0A0E14; --panel:#10141D; --panel2:#151A24;
    --line:rgba(255,255,255,0.08); --line-strong:rgba(255,255,255,0.16);
    --text:#E7EAF0; --muted:#8892A6; --muted2:#5B6474;
    --blue:#3E7BFA; --blue-dim:#1F3A66;
    --amber:#E8A548; --amber-dim:#4A3A1E;
    --teal:#3ECB9E; --teal-dim:#153B31;
    --red:#E2604F; --red-dim:#3E1F19;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}
.stApp {
    background: #0A0E14 !important;
    color: #E7EAF0 !important;
}
.page-header {
    background: #10141D;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 32px 40px;
    margin-bottom: 32px;
}
.page-title {
    font-family: 'Fraunces', serif;
    font-size: 2.2rem;
    font-weight: 500;
    letter-spacing: -0.01em;
    color: #E7EAF0;
    margin-bottom: 8px;
}
.page-sub {
    color: #8892A6;
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.5;
}
</style>
"""

def render_sidebar_api(page_title: str = ""):
    """Render the refined sidebar navigation and config."""
    t = get_t()
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <h2>HybridACD</h2>
            <div class="subtitle">Research Demo</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Removed Language Selector

        st.markdown("---")
        
        st.markdown(f'<div class="nav-section-title">{t("sidebar_nav")}</div>', unsafe_allow_html=True)
        st.page_link("app.py", label=t("nav_home"), icon="🏠")
        st.page_link("pages/1_pipeline_demo.py", label=t("nav_pipeline"), icon="🔭")
        st.page_link("pages/2_consistency_check.py", label=t("nav_consistency"), icon="🔍")
        st.page_link("pages/3_results_comparison.py", label=t("nav_results"), icon="📊")
        st.page_link("pages/3_history_analytics.py", label=t("nav_history"), icon="⏳")
        st.page_link("pages/4_live_forecast.py", label=t("nav_live"), icon="⚡")
        
        st.markdown("---")
        
        st.markdown(f'<div class="nav-section-title">{t("sidebar_config")}</div>', unsafe_allow_html=True)
        
        new_key = st.text_input(t("api_key"), value=st.session_state.api_key, type="password", key=f"key_{page_title}")
        new_url = st.text_input(t("api_url"), value=st.session_state.api_base_url, key=f"url_{page_title}")
        new_model = st.text_input(t("api_model"), value=st.session_state.model, key=f"model_{page_title}")
        new_trans_model = st.text_input("Translation Model", value=st.session_state.get("translation_model", "gpt-5.2"), key=f"trans_model_{page_title}")
        new_cache = st.toggle(t("api_cache"), value=st.session_state.get("no_cache", True), key=f"cache_{page_title}")
        
        # Auto-save and apply on change
        if (new_key != st.session_state.api_key or 
            new_url != st.session_state.api_base_url or 
            new_model != st.session_state.model or 
            new_trans_model != st.session_state.get("translation_model", "") or
            new_cache != st.session_state.get("no_cache", True)):
            st.session_state.api_key = new_key
            st.session_state.api_base_url = new_url
            st.session_state.model = new_model
            st.session_state.translation_model = new_trans_model
            st.session_state.no_cache = new_cache
            save_config_to_cache()
            apply_api_config()
            st.toast(t("toast_settings_updated"))

def render_rag_config():
    """Render RAG configuration UI in the sidebar and return the config."""
    with st.sidebar:
        st.markdown("---")
        st.markdown('<div class="nav-section-title">Agentic RAG Config</div>', unsafe_allow_html=True)
        rag_urls_str = st.text_area("RAG URLs (mỗi dòng 1 link)", value="")
        rag_files_upload = st.file_uploader("RAG Local Files (Upload)", accept_multiple_files=True, type=["txt", "md", "json", "csv"])
        
        rag_urls_list = [u.strip() for u in rag_urls_str.split("\n") if u.strip()]
        rag_files_list = []
        if rag_files_upload:
            for f in rag_files_upload:
                try:
                    content = f.read().decode("utf-8")
                    rag_files_list.append({"name": f.name, "content": content})
                except Exception as e:
                    st.error(f"Cannot read {f.name}: {e}")
                    
        return rag_urls_list, rag_files_list
            
        # Always apply config to ensure env vars are set on page reload
        apply_api_config()
        
        st.markdown("---")
        st.caption("DS391 - LLM | UIT | 2026")

def translate_to_vi(text: str) -> str:
    """Uses the selected translation model to translate English text into Vietnamese."""
    import openai
    model = st.session_state.get("translation_model", "gpt-5.2")
    base_url = st.session_state.get("api_base_url", "https://llm.wokushop.com/v1")
    api_key = st.session_state.get("api_key", "")
    
    if not api_key:
        return "⚠️ Lỗi: Chưa cấu hình API Key để dịch thuật."
        
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional translator. Translate the following text into natural Vietnamese. Preserve all markdown formatting, formulas, and technical accuracy. Output ONLY the translated text."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ Lỗi dịch thuật: {str(e)}"
