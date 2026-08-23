# Verification Commands

Only full-pipeline verification counts (Principle 10). All commands assume you are
in `consistency-forecasting/` (the actual codebase; the repo root only holds the
paper and harness files).

| Check       | Command                                                                 |
|-------------|--------------------------------------------------------------------------|
| Install     | `pip install -r requirements.txt --break-system-packages`               |
| Tests       | `pytest tests/ -v`                                                       |
| Lint        | `ruff check src/ tests/`                                                 |
| Type-check  | none configured (no mypy/pyright in repo) — treat lint as the substitute |
| Build       | none (pure Python, no compile step)                                     |
| Smoke run   | see below                                                                |

## Feature-specific verification (HybridACD forecaster)

Unit tests (no API key required — pure bound-math and config round-trip):
```bash
pytest tests/test_hybrid_acd_forecaster.py -v
```

Smoke run (requires a valid API key in `.env`, real model call, tiny sample size):
```bash
python src/evaluation.py \
  --tuple_dir src/data/tuples/scraped \
  --num_lines 5 --run --async -k all \
  -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster \
  -o model=gpt-4o-mini \
  --output_dir src/data/forecasts/_smoke_run

python src/evaluation.py --load src/data/forecasts/_smoke_run
cat src/data/forecasts/_smoke_run/stats_summary.json
```
Pass criterion for smoke run: command exits 0 and `stats_summary.json` contains a
non-null `AVS` (Aggregate Violation Score) field.

Full evaluation (matches paper's reported numbers, larger sample, real cost):
```bash
python src/evaluation.py \
  --tuple_dir src/data/tuples/scraped \
  --num_lines 200 --run --async -k all \
  -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster \
  -o model=<model-id> \
  --output_dir src/data/forecasts/HybridACD_<model>_run
```

Ground-truth / Brier Score run:
```bash
python src/ground_truth_run.py \
  -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster \
  -o model=<model-id> \
  --output_dir src/data/forecasts/HybridACD_<model>_groundtruth
```

Ablation (toggle components via forecaster options, matches paper Table 5):
```bash
# TCD only
-o adversarial_enabled=False -o tcd_enabled=True
# Adversarial only
-o adversarial_enabled=True -o tcd_enabled=False
```

A feature only moves to `passing` in `feature_list.json` when its own verification
command succeeds AND the checks above are clean.
