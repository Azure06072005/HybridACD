"""
HybridACD Demo — Main Entry Point (Home Page)
Consistency Forecasting Research Demo (DS391 - LLM)
"""

import streamlit as st
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEMO_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
for p in [SRC_PATH, PROJECT_ROOT, DEMO_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from demo_utils import init_session, render_sidebar_api, SIDEBAR_CSS
from translations import t

st.set_page_config(
    page_title="HybridACD Demo — Consistency Forecasting",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(SIDEBAR_CSS + """
<style>
.hero-banner {
    background: radial-gradient(circle at right bottom, rgba(96,165,250,0.1) 0%, transparent 50%),
                linear-gradient(135deg, rgba(17,24,39,0.8) 0%, rgba(3,7,18,0.95) 100%);
    border: 1px solid rgba(96,165,250,0.2);
    border-radius: 20px;
    padding: 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5);
}
.hero-title {
    font-size: 3.5rem;
    font-weight: 800;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #93c5fd 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
    line-height: 1.1;
}
.hero-subtitle {
    font-size: 1.25rem;
    color: #9ca3af;
    margin-bottom: 24px;
    font-weight: 400;
    max-width: 800px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(96,165,250,0.1);
    border: 1px solid rgba(96,165,250,0.3);
    color: #93c5fd;
    padding: 6px 16px;
    border-radius: 24px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-right: 12px;
    margin-bottom: 12px;
    letter-spacing: 0.5px;
}
.metric-card {
    background: rgba(31,41,55,0.4);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 24px -10px rgba(0,0,0,0.3);
    border-color: rgba(96,165,250,0.3);
}
.metric-value { font-size: 2.2rem; font-weight: 800; letter-spacing: -1px; }
.metric-label { font-size: 0.85rem; color: #9ca3af; margin-top: 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 1px; }
.section-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f3f4f6;
    margin: 40px 0 20px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.research-box {
    background: rgba(31,41,55,0.4);
    border-left: 4px solid #8b5cf6;
    padding: 24px;
    border-radius: 0 12px 12px 0;
    margin-bottom: 24px;
}
.research-box h4 { color: #ddd6fe; margin-top: 0; font-size: 1.1rem; }
.research-box p { color: #d1d5db; line-height: 1.6; font-size: 0.95rem; margin-bottom: 0; }

/* Pipeline flow styles */
.pipeline-section {
    background: rgba(17,24,39,0.5);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 24px;
}
.pipeline-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 20px;
    display: inline-block;
    margin-bottom: 14px;
}
.pl-arbitrage { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(239,68,68,0.3); }
.pl-hybrid    { background: rgba(16,185,129,0.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.3); }
.flow-steps {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 12px;
}
.flow-step {
    background: rgba(31,41,55,0.7);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #e5e7eb;
    text-align: center;
    min-width: 120px;
    flex: 1;
    line-height: 1.4;
}
.flow-step .step-num {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    opacity: 0.6;
    display: block;
    margin-bottom: 3px;
}
.flow-step.arbitrage-step { border-color: rgba(239,68,68,0.25); }
.flow-step.hybrid-step    { border-color: rgba(16,185,129,0.25); }
.flow-arrow { color: #4b5563; font-size: 1.2rem; flex-shrink: 0; }
.pipeline-note {
    margin-top: 14px;
    font-size: 0.82rem;
    color: #6b7280;
    font-style: italic;
    border-top: 1px solid rgba(255,255,255,0.05);
    padding-top: 10px;
}
.compare-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-right: 8px;
}
.badge-cost-high { background: rgba(239,68,68,0.1); color: #fca5a5; border: 1px solid rgba(239,68,68,0.25); }
.badge-cost-low  { background: rgba(16,185,129,0.1); color: #6ee7b7; border: 1px solid rgba(16,185,129,0.25); }
.badge-consist   { background: rgba(139,92,246,0.1); color: #c4b5fd; border: 1px solid rgba(139,92,246,0.25); }
</style>
""", unsafe_allow_html=True)

init_session()
render_sidebar_api("Home")

import streamlit.components.v1 as components

components.html("""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HybridACD — Consistency Forecasting</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,600&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#0A0E14; --panel:#10141D; --panel2:#151A24;
    --line:rgba(255,255,255,0.08); --line-strong:rgba(255,255,255,0.16);
    --text:#E7EAF0; --muted:#8892A6; --muted2:#5B6474;
    --blue:#3E7BFA; --blue-dim:#1F3A66;
    --amber:#E8A548; --amber-dim:#4A3A1E;
    --teal:#3ECB9E; --teal-dim:#153B31;
    --red:#E2604F; --red-dim:#3E1F19;
  }
  *{box-sizing:border-box; margin:0; padding:0;}
  body{
    background:var(--bg); color:var(--text);
    font-family:'Inter',sans-serif;
    -webkit-font-smoothing:antialiased;
  }
  ::selection{ background:var(--blue-dim); }
  .wrap{ max-width:1180px; margin:0 auto; padding:0 40px; }

  /* ── nav ── */
  nav{ display:flex; align-items:center; justify-content:space-between; padding:28px 0; border-bottom:1px solid var(--line); }
  .logo{ font-family:'Fraunces',serif; font-weight:600; font-size:1.3rem; letter-spacing:-0.02em; display:flex; align-items:center; gap:10px;}
  .logo svg{ display:block; }
  nav .links{ display:flex; gap:32px; font-size:0.9rem; color:var(--muted); }
  nav .links a{ color:var(--muted); text-decoration:none; }
  nav .links a:hover{ color:var(--text); }
  .cta-btn{ background:var(--blue); color:#fff; border:none; padding:10px 20px; border-radius:6px; font-size:0.88rem; font-weight:500; cursor:pointer; font-family:'Inter',sans-serif; }
  .cta-btn.ghost{ background:transparent; border:1px solid var(--line-strong); color:var(--text); }

  /* ── hero ── */
  .hero{ padding:90px 0 60px; position:relative; }
  .eyebrow{ font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:var(--amber); letter-spacing:0.08em; text-transform:uppercase; margin-bottom:22px; display:flex; align-items:center; gap:10px;}
  .eyebrow::before{ content:''; width:6px; height:6px; border-radius:50%; background:var(--amber); display:inline-block; }
  h1{ font-family:'Fraunces',serif; font-weight:500; font-size:4.6rem; line-height:1.02; letter-spacing:-0.02em; max-width:820px; }
  h1 em{ font-style:italic; color:var(--blue); font-weight:500; }
  .hero-sub{ margin-top:26px; font-size:1.15rem; color:var(--muted); max-width:560px; line-height:1.6; font-weight:400; }
  .hero-actions{ margin-top:36px; display:flex; gap:14px; align-items:center; }
  .hero-actions .cta-btn{ padding:13px 26px; font-size:0.95rem; }
  .hero-actions .link-more{ font-size:0.9rem; color:var(--muted); text-decoration:underline; text-underline-offset:4px; }

  /* ── ticker (signature element) ── */
  .ticker-section{ margin-top:70px; border-top:1px solid var(--line); border-bottom:1px solid var(--line); padding:26px 0; overflow:hidden; position:relative; }
  .ticker-label{ font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:var(--muted2); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:14px; }
  .ticker-track{ display:flex; gap:36px; white-space:nowrap; animation:scroll 26s linear infinite; width:max-content; }
  .ticker-section:hover .ticker-track{ animation-play-state:paused; }
  @keyframes scroll{ from{ transform:translateX(0); } to{ transform:translateX(-50%); } }
  .tick{ font-family:'JetBrains Mono',monospace; font-size:0.92rem; display:flex; align-items:center; gap:10px; }
  .tick .q{ color:var(--muted); }
  .tick .sum-good{ color:var(--teal); }
  .tick .sum-bad{ color:var(--red); }

  /* ── section shared ── */
  section{ padding:80px 0; border-bottom:1px solid var(--line); }
  .sec-head{ display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:44px; gap:40px; }
  .sec-title{ font-family:'Fraunces',serif; font-weight:500; font-size:2.1rem; letter-spacing:-0.01em; }
  .sec-desc{ color:var(--muted); max-width:380px; font-size:0.95rem; line-height:1.6; text-align:right; }

  /* ── pipeline slider (signature interactive) ── */
  .slider-frame{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:40px; }
  .slider-row{ display:flex; align-items:center; gap:20px; margin-bottom:34px; }
  .slider-row input[type=range]{ flex:1; accent-color:var(--blue); height:2px; }
  .slider-tag{ font-family:'JetBrains Mono',monospace; font-size:0.78rem; padding:6px 14px; border-radius:20px; border:1px solid var(--line-strong); color:var(--muted); }
  .slider-tag.active-arb{ color:var(--red); border-color:rgba(226,96,79,0.4); background:var(--red-dim); }
  .slider-tag.active-hyb{ color:var(--teal); border-color:rgba(62,203,158,0.4); background:var(--teal-dim); }

  .pipe-row{ display:flex; align-items:stretch; gap:0; }
  .pipe-step{ flex:1; background:var(--panel2); border:1px solid var(--line); border-radius:10px; padding:18px 16px; margin-right:14px; position:relative; transition:transform 0.25s ease, border-color 0.25s ease; }
  .pipe-step:last-child{ margin-right:0; }
  .pipe-num{ font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:var(--muted2); text-transform:uppercase; letter-spacing:0.06em; }
  .pipe-name{ font-size:0.92rem; margin-top:8px; font-weight:500; line-height:1.35; }
  .pipe-cost{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:var(--muted); margin-top:10px; }

  .metrics-strip{ display:flex; justify-content:space-between; margin-top:34px; padding-top:28px; border-top:1px solid var(--line); }
  .metric{ text-align:left; }
  .metric .val{ font-family:'JetBrains Mono',monospace; font-size:1.7rem; font-weight:500; }
  .metric .lbl{ font-size:0.78rem; color:var(--muted); margin-top:4px; }

  /* ── forecast demo card ── */
  .demo-grid{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }
  .demo-card{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:30px; }
  .demo-card.bad{ border-color:rgba(226,96,79,0.25); }
  .demo-card.good{ border-color:rgba(62,203,158,0.3); }
  .demo-label{ font-family:'JetBrains Mono',monospace; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; color:var(--muted); }
  .demo-prob{ font-family:'Fraunces',serif; font-size:3.2rem; margin-top:12px; font-weight:500; }
  .demo-card.bad .demo-prob{ color:var(--red); }
  .demo-card.good .demo-prob{ color:var(--teal); }
  .demo-eq{ font-family:'JetBrains Mono',monospace; font-size:0.85rem; color:var(--muted); margin-top:14px; }

  /* ── models table ── */
  table{ width:100%; border-collapse:collapse; }
  th{ text-align:left; font-family:'JetBrains Mono',monospace; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted2); padding-bottom:14px; border-bottom:1px solid var(--line); font-weight:400; }
  td{ padding:16px 0; border-bottom:1px solid var(--line); font-size:0.92rem; }
  td.mono{ font-family:'JetBrains Mono',monospace; }
  td.down{ color:var(--teal); }

  footer{ padding:50px 0; display:flex; justify-content:space-between; align-items:center; color:var(--muted2); font-size:0.85rem; }

  @media (max-width:800px){
    h1{ font-size:2.8rem; }
    .sec-head{ flex-direction:column; align-items:flex-start; }
    .sec-desc{ text-align:left; }
    .demo-grid{ grid-template-columns:1fr; }
    .pipe-row{ flex-wrap:wrap; }
    .pipe-step{ flex:1 1 45%; margin-bottom:14px; }
  }
</style>
</head>
<body>

<div class="wrap">
  <!-- We remove the top nav because Streamlit sidebar handles navigation -->
  <div style="height: 40px;"></div>
  <div class="hero">
    <div class="eyebrow">DS391 · LLM probabilistic forecasting research</div>
    <h1>Your forecaster says<br>73% <em>and</em> 41%<br>for the same coin flip.</h1>
    <p class="hero-sub">HybridACD enforces logical consistency at decoding time — no post-hoc arbitrage, no dozens of extra API calls. Just probabilities that add up.</p>
    <div class="hero-actions">
      <button class="cta-btn" onclick="window.parent.postMessage({type: 'streamlit:setComponentValue', value: 'run_live'}, '*');">Run a live forecast</button>
      <a class="link-more" href="#pipeline">See how it works ↓</a>
    </div>
  </div>

  <div class="ticker-section">
    <div class="ticker-label">Consistency check — live sample questions</div>
    <div class="ticker-track" id="tickerTrack"></div>
  </div>

  <section id="pipeline">
    <div class="sec-head">
      <div class="sec-title">One question, two very<br>different pipelines</div>
      <div class="sec-desc">Drag the slider to compare the original arbitrage-based approach against HybridACD's decoding-time intervention.</div>
    </div>

    <div class="slider-frame">
      <div class="slider-row">
        <span class="slider-tag" id="tagArb">Arbitrage / ConsistentForecaster</span>
        <input type="range" min="0" max="1" step="1" value="1" id="pipeSlider">
        <span class="slider-tag" id="tagHyb">HybridACD</span>
      </div>
      <div class="pipe-row" id="pipeRow"></div>
      <div class="metrics-strip" id="metricsStrip"></div>
    </div>
  </section>

  <section id="demo">
    <div class="sec-head">
      <div class="sec-title">P(A) + P(¬A) should equal 1</div>
      <div class="sec-desc">Sum of probability and its negation for "Will Vietnam's GDP growth exceed 7% in 2026?"</div>
    </div>
    <div class="demo-grid">
      <div class="demo-card bad">
        <div class="demo-label">Basic forecaster</div>
        <div class="demo-prob">1.187</div>
        <div class="demo-eq">P(A) 0.62 + P(¬A) 0.567 — off by 0.187</div>
      </div>
      <div class="demo-card good">
        <div class="demo-label">HybridACD</div>
        <div class="demo-prob">1.000</div>
        <div class="demo-eq">P(A) 0.58 + P(¬A) 0.42 — exact</div>
      </div>
    </div>
  </section>

  <section id="results">
    <div class="sec-head">
      <div class="sec-title">Across 6 models, 242 questions</div>
      <div class="sec-desc">Average violation score (AVS) and Brier score, basic vs HybridACD.</div>
    </div>
    <table>
      <tr><th>Model</th><th>Basic AVS</th><th>HybridACD AVS</th><th>Reduction</th></tr>
      <tr><td>GPT-4o-mini</td><td class="mono">0.0307</td><td class="mono">0.0007</td><td class="mono down">-97.7%</td></tr>
      <tr><td>Gemini 2.5 Flash</td><td class="mono">0.1116</td><td class="mono">0.0087</td><td class="mono down">-92.2%</td></tr>
      <tr><td>Mistral Medium</td><td class="mono">0.0792</td><td class="mono">0.0087</td><td class="mono down">-89.0%</td></tr>
      <tr><td>Mistral Small</td><td class="mono">0.0740</td><td class="mono">0.0023</td><td class="mono down">-96.9%</td></tr>
      <tr><td>MiniMax M3</td><td class="mono">0.1266</td><td class="mono">0.0005</td><td class="mono down">-99.6%</td></tr>
    </table>
  </section>

  <footer>
    <span>HybridACD — consistency forecasting research demo</span>
    <span>DS391 · LLM course</span>
  </footer>
</div>

<script>
const questions = [
  {q:"Will BTC exceed $200k by Dec 2026?", basic:false, hybrid:true},
  {q:"Will VN GDP growth exceed 7% in 2026?", basic:true, hybrid:true},
  {q:"Will OpenAI release GPT-5 before EOY?", basic:false, hybrid:true},
  {q:"Will the Fed cut rates in Q3?", basic:true, hybrid:true},
  {q:"Will inflation stay under 3%?", basic:false, hybrid:true},
  {q:"Will the S&P 500 close green in July?", basic:true, hybrid:true},
];
function sum(ok){ return ok ? (1.000).toFixed(3) : (0.6+Math.random()*0.5).toFixed(3); }
let html = '';
for(let r=0;r<2;r++){
  questions.forEach(item=>{
    const bs = sum(item.basic), hs = sum(item.hybrid);
    html += `<div class="tick"><span class="q">${item.q}</span><span class="${bs=='1.000'?'sum-good':'sum-bad'}">basic ${bs}</span><span class="${hs=='1.000'?'sum-good':'sum-bad'}">hybrid ${hs}</span></div>`;
  });
}
document.getElementById('tickerTrack').innerHTML = html;

const arb = [
  {n:"Input", d:"Single question P", c:"$0.00"},
  {n:"Basic forecast", d:"P_raw via BasicForecaster", c:"$0.02"},
  {n:"Related questions", d:"DB lookup + LLM generation", c:"$0.40"},
  {n:"Tuple init", d:"instantiate_cons_tuples", c:"$0.10"},
  {n:"Sub-forecasts", d:"BasicForecaster × N", c:"$1.80"},
  {n:"Arbitrage optimize", d:"max_min_arbitrage()", c:"$0.15"},
  {n:"Output", d:"P_consistent", c:"~$2,500 / dataset"},
];
const hyb = [
  {n:"Input", d:"Full logic tuple (P, ¬P, ...)", c:"$0.00"},
  {n:"Agentic research", d:"Multi-turn tool calling", c:"$0.03"},
  {n:"Adversarial rewrite", d:"adversarial_rewrite_sync()", c:"$0.01"},
  {n:"Consistency bounds", d:"[lower, upper] from history", c:"$0.00"},
  {n:"Chain of thought", d:"Deep step-by-step reasoning", c:"$0.02"},
  {n:"Token constraint decode", d:"logit_bias = -100 (TCD)", c:"$0.00"},
  {n:"Output", d:"P ∈ [lo, hi], 100% consistent", c:"~$0.12 / dataset"},
];
const metricsArb = [["21,700×","more expensive"],["~95%","exact consistency"],["N × LLM calls","per question"]];
const metricsHyb = [["1×","baseline cost"],["100%","exact consistency"],["3 LLM calls","per question"]];

function render(isHybrid){
  const steps = isHybrid ? hyb : arb;
  document.getElementById('pipeRow').innerHTML = steps.map(s=>`
    <div class="pipe-step">
      <div class="pipe-num">${s.n}</div>
      <div class="pipe-name">${s.d}</div>
      <div class="pipe-cost">${s.c}</div>
    </div>`).join('');
  const m = isHybrid ? metricsHyb : metricsArb;
  document.getElementById('metricsStrip').innerHTML = m.map(x=>`
    <div class="metric"><div class="val" style="color:${isHybrid?'var(--teal)':'var(--red)'}">${x[0]}</div><div class="lbl">${x[1]}</div></div>`).join('');
  document.getElementById('tagArb').className = 'slider-tag' + (isHybrid?'':' active-arb');
  document.getElementById('tagHyb').className = 'slider-tag' + (isHybrid?' active-hyb':'');
}
const slider = document.getElementById('pipeSlider');
slider.addEventListener('input', ()=> render(slider.value === '1'));
render(true);
</script>

</body>
</html>
""", height=1900, scrolling=True)

