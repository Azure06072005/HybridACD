# Progress Log (Gemini sessions)

Update this at the end of every session. This is what the next session reads to avoid starting from zero.

## Session 3 — 2026-08-23
- Completed:
  1. Inspected raw prediction outputs in `src/data/forecasts/_smoke_test_gemini/NegChecker.jsonl` (Karpathy Section 5 discipline). Verified that the 0.000 violation was NOT degenerate constant output: base questions produced grounded historical analysis ($p=0.0$), while negated questions correctly had dynamic bounds $[1.0, 1.0]$ enforced by `get_consistency_bounds`, satisfying algebraic consistency by construction.
  2. Re-verified the entire test suite (`tests/test_hybrid_acd_forecaster.py` and `tests/test_config.py`) after all recent code edits — 16/16 unit tests PASSED (100% clean, zero regression on F001/F002).
  3. Reaffirmed standard evaluation model policy: `gpt-4o-mini` remains the official benchmark model (to match paper Table 2 and all 193 on-disk baselines for F004 and F007); non-standard endpoints/models serve only for mechanical pipeline debugging.
- In progress: F003 (smoke run on full 10 checkers with benchmark model).
- Blocked: Live provider quota on benchmark endpoint for full multi-checker run.
- Next session should: Execute full 10-checker F003 smoke run on `gpt-4o-mini` once provider credits are refreshed.

## Session 2 — 2026-08-23
- Completed:
  1. Implemented isolated typed configuration surface in `consistency-forecasting/src/hybrid_acd_config.py` (avoided namespace collisions with `llm_forecasting/config/constants.py`).
  2. Created `tests/test_config.py` validating all guardrails (TCDMode, ADR-002, ADR-003 anti-leakage rules).
  3. Fixed kwarg sanitization in `adversarial_rewrite_async` and `_sync` in `hybrid_acd_forecaster.py` to prevent passing checker metadata (`rule='NegChecker'`) to API endpoints.
  4. Made exception fallback routing in `src/common/llm_utils.py` configurable via `FALLBACK_MODEL`.
- In progress: F003 preparation.
- Blocked: F003 full 10-checker run on live API.
- Next session should: Run live evaluation smoke run for F003.

## Session 1 — 2026-08-23
- Completed:
  1. Surveyed all 193 pre-computed run directories in `src/data/forecasts/` and confirmed complete on-disk baselines for small and large models without retrieval (Brier score, calibration error, and 10-checker AVS).
  2. Audited `ResolverBasedForecaster` (`src/forecasters/various.py`, `src/perplexity_resolver/resolver.py`, `src/common/perplexity_client.py`) and confirmed outcome leakage (dates only inserted as prompt text, no API-level date cutoff, Perplexity recency filter is relative-only).
  3. Integrated `docs/retrieval_step_spec.md` with verified call order (`retrieve` → `adversarial_rewrite` → `base_forecast` → `TCD_clip`), cutoff filter rules, and leakage smoke-test requirements.
- In progress: F003 / F007 preparation.
- Blocked: F007 is blocked on F003 smoke run and on search-provider selection with absolute historical date cutoff.
- Next session should: Configure `.env` with API keys and run F003 smoke run (`num_lines=5`).

## Session 0 — 2026-08-23
- Completed: Reorganized repository structure into the standard HybridACD harness layout (`docs/`, `paper/`, root harness state files). Completed deep reading and analysis of `Consistency checks for LLM forecasters.pdf`, `HybridACD_en.pdf`, and the harness configuration.
- In progress: Initial environment verification and test confirmation.
- Blocked: —
- Next session should: Run `./init.sh` to confirm environment dependencies and test suite health, configure `.env` if live API runs (F003+) are needed.
