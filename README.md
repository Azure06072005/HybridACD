# HybridACD

**Adversarial Constraint Decoding for Logically Consistent Large Language Model Forecasting**

> Kiet Tran Anh · Faculty of Information Science and Engineering, University of Information Technology, VNU-HCM
> 23520820@gm.uit.edu.vn

HybridACD is a decoding-time intervention that makes LLM probabilistic forecasters logically consistent — without retraining, without expensive post-hoc calibration, and without the Goodhart-collapse risk of optimizing directly against a consistency metric.

---

## Table of Contents

- [Motivation](#motivation)
- [What HybridACD Does](#what-hybridacd-does)
- [Architecture](#architecture)
- [Consistency Rules](#consistency-rules)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Experimental Results](#experimental-results)
- [Limitations & Known Issues](#limitations--known-issues)
- [Project Discipline & Contributing](#project-discipline--contributing)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [References](#references)

---

## Motivation

Large language models are increasingly deployed as forecasting agents, assigning a probability p ∈ [0, 1] to future events. Good calibration on individual questions is not the same as **logical consistency across sets of related questions**. A model can have an excellent Brier Score while still violating basic probability axioms:

- If P(A) = 0.70, then P(¬A) must equal 0.30 — not 0.45 or 0.20.
- If P(A ∧ B) is predicted, it must lie within `[max(0, p_A + p_B − 1), min(p_A, p_B)]`.
- Conditional chains (`P(A∧B) = P(A)·P(B|A)`) must hold algebraically, not just approximately.

These aren't cosmetic issues. An inconsistent set of probability predictions is formally equivalent to a **Dutch Book vulnerability**: an adversary can construct a betting portfolio that guarantees profit regardless of the outcome. Consistency is therefore a *necessary condition* for treating an LLM forecaster as a trustworthy decision-making system, not an optional nicety.

Existing fixes are unsatisfying:
- **Post-hoc arbitrage calibration** (e.g. `ArbitrageForecaster`) reduces violations but requires many repeated model calls — the paper estimates ~$2,500 for N=4 rounds on a modest question set.
- **Fine-tuning against a consistency metric** risks **Goodhart's Law**: the model learns to trivially satisfy the metric by collapsing toward neutral predictions (P ≈ 0.5), destroying real forecasting value.

HybridACD sidesteps both failure modes by intervening **only at the final probability-extraction step**, after the model has already reasoned freely.

## What HybridACD Does

HybridACD combines two independent mechanisms:

1. **Adversarial Rewriting** — an auxiliary model rewrites each forecasting question into a syntactically different but semantically identical variant (subordinate clauses, entity substitution, negation raising, nested conditioning). This discourages the forecaster from relying on surface pattern-matching rather than genuine reasoning, and is especially important for rules like **Negation** and **Paraphrase**, where models are prone to lexical-cue shortcuts.
2. **Token-Constrained Decoding (TCD)** — given the predictions already made earlier in a consistency set, HybridACD algebraically derives a valid interval `[ℓ, u]` for the current question from the applicable consistency rule, then constrains the model's output to that interval via a logit-bias mask over the discretized probability token set (`{0.00, 0.01, ..., 1.00}`). Tokens outside `[ℓ, u]` receive a bias of −100; tokens inside are untouched. **If the provider API doesn't support logit bias, or biased-response parsing fails, HybridACD falls back to a deterministic clip:** `p̂ = clip(p_raw, ℓ, u)`. Either path guarantees the final output respects the derived bound.

Critically, the constraint is applied **only to the final numeric output**, not to the reasoning trace — the model still benefits from unconstrained Chain-of-Thought before its answer is clipped/biased into the valid range.

```
Adversarial Input Agent  →  free-form CoT reasoning  →  Token-Constrained Decoding  →  p̂ᵢ ∈ [ℓᵢ, uᵢ]
```

## Architecture

```
for each question Qᵢ in a consistency set:
    Qᵢ' = rewrite(Qᵢ)                     # adversarial rewriting, semantics preserved
    [ℓᵢ, uᵢ] = bounds(H, keyᵢ, rule)       # algebraic bound from prior predictions H
    coti = model(Qᵢ')                     # unconstrained chain-of-thought
    p̂ᵢ = extract(coti, logit_bias=B)      # TCD: bias out-of-bound probability tokens
    p̂ᵢ = clip(p̂ᵢ, ℓᵢ, uᵢ)                 # deterministic fallback guarantee
    H[keyᵢ] = p̂ᵢ                          # feed forward for subsequent bounds
return {p̂₁, ..., p̂ₖ}
```

Predictions within a consistency set are processed **sequentially** — order matters, because later bounds depend on earlier resolved predictions (e.g. `P` and `Q|P` must be known before bounding `P∧Q`).

## Consistency Rules

HybridACD implements ten probabilistic consistency rules, each converted into an algebraic bound `[ℓ, u]`:

| # | Rule | Constraint | Key Set |
|---|------|------------|---------|
| 1 | Negation | `p_P + p_¬P = 1` | `{P, ¬P}` |
| 2 | Paraphrase | `p_P = p_P'` | `{P, P'}` |
| 3 | Entailment | `P ⊨ B ⇒ p_P ≤ p_B` | `{P, B}` |
| 4 | And | `max(0, p+q−1) ≤ p_(P∧Q) ≤ min(p,q)` | `{P, Q, P∧Q}` |
| 5 | Or | `max(p,q) ≤ p_(P∨Q) ≤ min(1, p+q)` | `{P, Q, P∨Q}` |
| 6 | But | `p_(P∨Q) = p_P + p_(Q∧¬P)` | `{P, Q∧¬P, P∨Q}` |
| 7 | Conditional | `p_(P∧Q) = p_P · p_(Q|P)` | `{P, Q|P, P∧Q}` |
| 8 | Nested Conditional | `p_(P∧Q∧R) = p_P · p_(Q|P) · p_(R|P∧Q)` | `{P, Q|P, R|P∧Q, P∧Q∧R}` |
| 9 | And-Or | `p_P + p_Q = p_(P∨Q) + p_(P∧Q)` | `{P, Q, P∧Q, P∨Q}` |
| 10 | Expected Evidence | `p_P = p_Q·p_(P|Q) + (1−p_Q)·p_(P|¬Q)` | `{P, Q, P|Q, P|¬Q}` |

These follow directly from the consistency-check framework of Paleka et al. (ICLR 2025) that this project builds on.

## Repository Structure

```
HybridACD/
├── AGENTS.md                      # agent/session workflow, hard constraints, WIP=1 discipline
├── DECISIONS.md                   # architecture decision records (ADRs)
├── feature_list.json              # feature states: not_started / active / passing / blocked
├── claude-progress.md             # session log (Claude)
├── gemini-progress.md             # session log (Gemini)
├── init.sh                        # bootstrap: install deps, run tests, verify harness
├── docs/
│   ├── architecture.md            # system structure, module boundaries
│   ├── verification.md            # exact test/lint/smoke-run/full-run commands
│   └── conventions.md             # code style, naming, project-specific patterns
├── HybridACD_en.pdf                # source report (English)
├── HybridACD_vi.pdf                # source report (Vietnamese)
├── Consistency checks for LLM forecasters.pdf   # foundational paper (Paleka et al.)
└── consistency-forecasting/       # actual codebase (fork of dpaleka/consistency-forecasting)
    ├── src/
    │   ├── forecasters/
    │   │   ├── hybrid_acd_forecaster.py   # HybridACDForecaster: adversarial rewrite + TCD
    │   │   ├── basic_forecaster.py        # baseline: single-shot prompted forecast
    │   │   ├── cot_forecaster.py          # baseline: chain-of-thought forecast
    │   │   └── consistent_forecaster.py   # baseline: post-hoc arbitrage calibration
    │   ├── static_checks/                 # Checker classes per consistency rule
    │   ├── common/                        # shared datatypes, LLM API wrappers
    │   ├── evaluation.py                  # runs a forecaster over tuple sets, computes AVS
    │   ├── ground_truth_run.py            # runs on resolved questions, computes Brier Score
    │   └── data/
    │       ├── tuples/                    # consistency question sets (scraped, 2028, ...)
    │       └── forecasts/                 # per-(forecaster, model) output dirs, stats_*.json
    └── tests/
        └── test_hybrid_acd_forecaster.py  # bound-math + config round-trip tests
```

## Getting Started

```bash
git clone https://github.com/Azure06072005/HybridACD.git
cd HybridACD
./init.sh
```

`init.sh` will:
1. Install Python dependencies (`pip install -r requirements.txt --break-system-packages`).
2. Copy `.env.example` → `.env` (fill in API keys before running any live evaluation).
3. Run the offline unit test suite (bound-math + config tests — no API key required).
4. Run `ruff check` against the forecaster module.
5. Confirm the harness state files (`AGENTS.md`, `feature_list.json`, progress logs) exist.

You must fill in `consistency-forecasting/.env` (`OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `OPENROUTER_API_KEY`, etc.) before running anything that calls a real model.

## Usage

**Unit tests only (no live API calls):**
```bash
pytest consistency-forecasting/tests/test_hybrid_acd_forecaster.py -v
```

**Smoke run** (tiny sample, real model call — run this before any full evaluation):
```bash
python src/evaluation.py \
  --tuple_dir src/data/tuples/scraped \
  --num_lines 5 --run --async -k all \
  -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster \
  -o model=<model-id> \
  --output_dir src/data/forecasts/_smoke_run

python src/evaluation.py --load src/data/forecasts/_smoke_run
cat src/data/forecasts/_smoke_run/stats_summary.json
```
Pass criterion: exits 0 and `stats_summary.json` contains a non-null `AVS` field.

**Full evaluation** (matches the paper's reporting scale — real cost, run only after a clean smoke run):
```bash
python src/evaluation.py \
  --tuple_dir src/data/tuples/scraped \
  --num_lines 200 --run --async -k all \
  -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster \
  -o model=<model-id> \
  --output_dir src/data/forecasts/HybridACD_<model>_run
```

**Ablation** (isolate each component, matches paper Table 5):
```bash
# TCD only
-o adversarial_enabled=False -o tcd_enabled=True
# Adversarial only
-o adversarial_enabled=True -o tcd_enabled=False
```

Full command reference, including ground-truth/Brier-Score runs, lives in `docs/verification.md`.

## Experimental Results

Evaluated across six LLMs (Gemini-2.5-Flash, GPT-4o-mini, GPT-5.4-mini, MiniMax-M3, Mistral-Medium-3.5-128B, Mistral-Small-4-119B) against a `BasicForecaster` baseline:

| Model | AVS (Basic) | AVS (HybridACD) | Δ% |
|---|---|---|---|
| Gemini-2.5-Flash | 0.1116 | 0.0087 | −92.2% |
| GPT-4o-mini | 0.0307 | 0.0007 | −97.6% |
| GPT-5.4-mini | 0.4909 | 0.0107 | −97.8% |
| MiniMax-M3 | 0.1266 | 0.0005 | −99.6% |
| Mistral-Medium-3.5 | 0.0792 | 0.0087 | −89.1% |
| Mistral-Small-4 | 0.0740 | 0.0023 | −96.8% |
| **Average** | **0.1522** | **0.0053** | **−95.5%** |

On 242 real-world questions, HybridACD also **improved** Brier Score across all five fully-evaluated models (−2.4% to −17.3%), indicating the consistency gain is not bought at the cost of forecasting accuracy — no Goodhart collapse observed. An ablation on GPT-4o-mini shows **TCD is the dominant component**: removing it raises AVS from 0.0007 to 0.0147, while removing the adversarial rewriter only raises it to 0.0012.

Full tables, baseline comparisons (vs. `ArbitrageForecaster`, CoT forecasters), and cost-per-question figures are in `HybridACD_en.pdf` / `HybridACD_vi.pdf`.

## Limitations & Known Issues

From the report's own discussion, plus findings logged during development (see `DECISIONS.md` for full ADRs):

- **Calibration Error can increase** under hard constraints — observed on both Mistral models (flagged `†`, still under investigation) — likely when `[ℓ, u]` is narrower than the model's natural confidence distribution. A soft-penalty alternative to hard clipping is proposed future work.
- **Sequential processing** within a consistency set limits parallelization on large sets, since later bounds depend on earlier resolved predictions.
- **0.01 probability discretization** may be too coarse for applications needing finer-grained probability estimates.
- **Logit-bias reliability depends on the provider.** Reseller/proxy API endpoints may silently no-op logit-bias injection, in which case TCD's enforcement runs entirely through the prompt-rewriting + deterministic-clip fallback path rather than true logit-level constraint — this changes what "TCD" is actually doing in practice and should be checked/logged per endpoint.
- **A near-zero Aggregate Violation Score is not automatically good news** — it can also indicate degenerate/neutral output or a silent fallback path masking failure. Always cross-check against Brier Score before treating an AVS improvement as real progress.
- **Training-data leakage** is a risk for any post-2024-cutoff model evaluated on the `scraped` tuple set; results on that set should be reported with this caveat, and ideally paired with a leakage-clean dataset run for citation-grade claims.
- **Model-identity verification gap**: some API resellers/proxies cannot guarantee the model actually served matches the requested model ID, which is a reproducibility risk that should be disclosed alongside any reported numbers obtained through such an endpoint.

## Project Discipline & Contributing

This repository is developed under an explicit multi-agent working discipline (see `AGENTS.md`):

- **WIP = 1** — exactly one feature worked on at a time; no incidental refactors bundled in.
- **`passing` is irreversible** — a feature is only marked passing after a full-pipeline verification (tests + lint + smoke run), never on "looks correct."
- **Every architectural decision is recorded as an ADR** in `DECISIONS.md` before being reversed or built upon.
- **Small before large** — any new evaluation configuration gets a `--num_lines 5` smoke run before a full run; live API evaluation runs cost real money.
- **Baseline-first** — no treatment number (e.g. HybridACD's AVS) is reported without its baseline (e.g. BasicForecaster) on the same model/dataset/sample size.
- **Negative results are results** — a failed reproduction or an unexpected fallback path is logged with the same weight as a success, not silently retried until it "looks right."

Before contributing, read `AGENTS.md`, `docs/architecture.md`, `docs/conventions.md`, and the current `feature_list.json` / progress logs to understand what's already decided and what's actively in flight.

## Roadmap

Tracked as features in `feature_list.json`:

- **F001–F002** — bound-math correctness and config round-trip (passing).
- **F003** — full-pipeline smoke run on a live model (passing, with reseller-endpoint and AVS caveats).
- **F004** — full-scale reproduction of the paper's reported AVS reduction, run on both a paper-comparable dataset and a leakage-clean dataset.
- **F005** — reproduction of the ablation study (TCD-only vs. adversarial-only vs. full).
- **F006** — investigation of the Calibration Error regression on Mistral models, and a proposed soft-penalty alternative to hard clipping.
- **F007** — retrieval-augmented small-model forecasting (scoped, not yet started).

## Citation

If you use this work, please cite the foundational consistency-checks paper this project builds on:

```
Paleka, D., Sudhir, A.P., Alvarez, A., Bhat, V., Shen, A., Wang, E., Tramèr, F.
"Consistency Checks for Language Model Forecasters." (2025)
https://arxiv.org/abs/2412.18544
```

And the HybridACD report (`HybridACD_en.pdf` / `HybridACD_vi.pdf` in this repository) for the adversarial-rewriting + TCD architecture and reported results.

## References

1. Paleka et al. — Consistency Checks for Language Model Forecasters (2025)
2. Lowin — An Intuitive Guide to How LLMs Work (2024)
3. Love — Backwards Compatible: The Strange Math Behind Word Order in AI (2026)
4. Anonymous — Probability Consistency in Large Language Models (submitted, rejected, 2025)
5. Yang et al. — Probability-Consistent Preference Optimization for Enhanced LLM Reasoning, ACL Findings 2025
6. Zhou et al. — Bridging Internal Probability and Self-Consistency for Effective and Efficient LLM Reasoning (2025)
7. Koo et al. — Automata-Based Constraints for Language Model Decoding, COLM 2024
8. Yao et al. — Token Constraint Decoding Improves Robustness on Question Answering for LLMs (2025)
9. Huang et al. — DeAL: Decoding-Time Alignment for Large Language Models, ACL 2025
10. Schall & de Melo — The Hidden Cost of Structure: How Constrained Decoding Affects Language Model Performance, RANLP 2025
11. Mousavi & Termehchy — Towards Consistent Language Models Using Controlled Prompting and Decoding (2024)
