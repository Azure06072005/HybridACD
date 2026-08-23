# Progress Log (Gemini sessions)

Update this at the end of every session. This is what the next session reads to avoid starting from zero.

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
