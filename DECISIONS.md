# Architecture Decision Records (ADRs)

Record deliberate choices, tradeoffs, and non-obvious constraints here. Do not reverse a prior decision without documenting the rationale.

## ADR-001: Separation of Harness, Source Paper, and Codebase
- **Date**: 2026-08-23
- **Context**: Need a clean, structured repository layout that isolates harness orchestration files (`AGENTS.md`, `docs/`, `feature_list.json`), reference papers/materials (`paper/`), and the actual evaluation codebase (`consistency-forecasting/`).
- **Decision**: Root holds harness files (`AGENTS.md`, `DECISIONS.md`, `feature_list.json`, `init.sh`, progress logs) and `docs/`; academic papers/slides reside in `paper/`; the implementation stays inside `consistency-forecasting/`.
- **Status**: Accepted.

## ADR-002: Decoding-Time Constraint Intervention (TCD) with Adversarial Rewriting
- **Date**: 2026-08-23
- **Context**: LLM forecasters frequently violate elementary probability rules (negation, conjunction, conditional bounds). Iterative post-hoc arbitrage (e.g. `ArbitrageForecaster`) is prohibitively expensive and overfits, while direct fine-tuning triggers Goodhart's law (collapsing predictions toward 0.5).
- **Decision**: Use `HybridACDForecaster` combining (1) adversarial question rewriting (varying syntax without altering semantics/resolution criteria) and (2) Token-Constrained Decoding (TCD) imposing algebraic bounds $[\\ell, u]$ via logit-bias with deterministic clipping fallback.
- **Status**: Accepted (implemented in `consistency-forecasting/src/forecasters/hybrid_acd_forecaster.py`).

## ADR-003: Retrieval Runs Once Per Question, Before Rewrite/TCD, With a Hard Date Cutoff
- **Date**: 2026-08-23
- **Context**: F007 proposes adding a pre-elicitation search/retrieval step so small models can close the accuracy gap to large models without retraining (small models forecast poorly mainly due to missing current-world context, not reasoning capacity — see F007 rationale). Two failure modes must be designed against from the start: (1) re-running search per adversarial-rewrite variant would multiply API cost for no benefit, since all variants share the same underlying question; (2) unrestricted search against already-resolved tuple sets (e.g. `src/data/tuples/scraped`, which resolved May–Aug 2024) could return articles reporting the actual outcome, since "today" is Aug 2026 relative to those questions — this would silently inflate accuracy by leaking the answer rather than improving forecasting.
- **Decision**: Retrieval executes exactly once per base question, before both adversarial rewriting and TCD (`retrieve(question) → context → adversarial_rewrite(question, context) → base_forecast(rewrite, context) → TCD_clip(...)`). Every retrieval call enforces a hard date-cutoff filter using the question's `created_at` / effective forecast date — never its resolution date — so no search result can postdate the point at which the forecast is nominally being made. This applies uniformly across tuple sets; it is not skipped for the `2028` set even though leakage is structurally impossible there, to keep the retrieval code path single and auditable.
- **Status**: Accepted (design decision only — not yet implemented; tracked as F007, blocked on F003).