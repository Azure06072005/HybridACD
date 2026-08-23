# Conventions

Project-specific style and patterns that aren't obvious from reading the code cold.
Only add rules here that have a source (why), applicability (when), and expiry
(when to remove). Audit regularly — delete anything stale.

- **Naming:** Forecaster classes live in `src/forecasters/` and are referenced on
  the CLI as `path/to/file.py::ClassName` (e.g.
  `src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster`). Output run
  directories follow `<ForecasterName>_<model>_<dataset-or-purpose>`
  (e.g. `HybridACD_gpt-4o-mini_tuples_scraped`, `HybridACD_groundtruth_run`).
  Source: matches existing `src/data/forecasts/*` directory names — keep new runs
  consistent so scripts that glob by prefix keep working. Applicability: any new
  evaluation run. Expiry: none known.

- **File/folder layout rules:** New forecaster implementations go in
  `src/forecasters/`, new consistency checkers in `src/static_checks/`. Do not put
  evaluation output (`stats_*.json`, calibration plots) anywhere except under
  `src/data/forecasts/<run_name>/` — the `--load` re-scoring step expects that
  layout. Source: `evaluation.py` load path assumptions. Applicability: always.

- **Error handling pattern:** `HybridACDForecaster`'s TCD step degrades gracefully
  — if the model API doesn't support logit bias, or the biased response fails to
  parse, it falls back to `clip(p_raw, ℓ, u)` rather than raising. Any new
  provider integration must preserve this fallback rather than letting a missing
  feature hard-fail the whole run. Source: `paper/HybridACD_en.pdf` §3.5,
  confirmed in `hybrid_acd_forecaster.py`. Applicability: all decoding-time
  constraint code.

- **Testing pattern:** Tests live in `consistency-forecasting/tests/`, one file
  per forecaster (`test_<forecaster_name>.py`). Bound-math tests (e.g.
  `test_bounds_neg_checker`, `test_bounds_and_checker`) don't call any live API —
  keep new rule-bound tests in this style so `pytest tests/test_hybrid_acd_forecaster.py`
  stays runnable without API keys, per `init.sh` step 2. Source:
  `tests/test_hybrid_acd_forecaster.py`. Applicability: any new consistency rule
  or bound formula added to `get_consistency_bounds()`.
