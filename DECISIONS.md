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
