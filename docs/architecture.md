# Architecture

## What is this system?

HybridACD is a decoding-time intervention that makes LLM probabilistic forecasters
logically consistent across sets of related questions (negation, paraphrase,
conjunction, disjunction, conditional probability, etc.) without retraining the
model. It's implemented as one `Forecaster` subclass (`HybridACDForecaster`) added
to a fork of `dpaleka/consistency-forecasting`, which supplies the question sets,
consistency-rule checkers, and evaluation harness. HybridACD combines two
mechanisms: an adversarial question-rewriting agent, and token-constrained
decoding (TCD) that clips the final probability into a bound derived algebraically
from prior answers in the same consistency set.

## How is it organized?

```
consistency-forecasting/
  src/
    forecasters/
      hybrid_acd_forecaster.py   # HybridACDForecaster: adversarial rewrite + TCD
      basic_forecaster.py        # baseline: single-shot prompted forecast
      cot_forecaster.py          # baseline: chain-of-thought forecast
      consistent_forecaster.py   # baseline: post-hoc arbitrage calibration
      forecaster.py              # abstract base class all forecasters implement
    static_checks/
      Checker.py                 # base class for consistency-rule checkers
      MiniInstantiator.py        # generates question variants per rule
    common/
      datatypes.py                # ForecastingQuestion, Forecast, Prob
      llm_utils.py                 # query_api_chat / query_api_chat_native wrappers
    evaluation.py                 # runs a forecaster over tuple sets, computes AVS
    ground_truth_run.py           # runs a forecaster on resolved questions, computes Brier
    data/
      tuples/                     # consistency question sets (scraped, newsapi, 2028, synthetic)
      forecasts/                  # output dirs per (forecaster, model) run — stats_*.json
  tests/
    test_hybrid_acd_forecaster.py # bound-math + config tests for HybridACD
  experiments/
    commands*.sh                  # example CLI invocations per forecaster/model
paper/
  HybridACD_en.pdf                # source report (method + results)
```

## How do I run it locally?

```bash
cd consistency-forecasting
pip install -r requirements.txt --break-system-packages
cp .env.example .env   # fill in API keys
pytest tests/test_hybrid_acd_forecaster.py -v
```
See `docs/verification.md` for full evaluation / smoke-run / ablation commands.

## Key conventions that aren't obvious from the code itself

- A "consistency set" is a dict of `{key: ForecastingQuestion}` (e.g. `{"P": ..,
  "not_P": ..}` for negation). `HybridACDForecaster.elicit()` processes keys
  sequentially, feeding each prediction into `previous_predictions` so later
  bounds can depend on earlier answers — order within a set matters.
- Consistency rules and their algebraic bound formulas live in Table 1 of the
  paper (`paper/HybridACD_en.pdf`, §3.2) — `get_consistency_bounds()` in
  `hybrid_acd_forecaster.py` is the code implementation of that table. Any change
  to bound math must be checked against both the paper and
  `tests/test_hybrid_acd_forecaster.py`.
- The probability grid is discretized to steps of 0.01 (`V_num` in the paper) —
  this is a hard design constraint, not a tunable default.
- TCD's primary mechanism is a logit bias of `-100` on out-of-bound tokens; if the
  provider API doesn't expose logit bias (or parsing the biased response fails),
  it silently falls back to `clip(p_raw, ℓ, u)` — both paths must stay correct.
- Forecast/violation results are dumped as `stats_{CheckerType}.json` plus
  `stats_aggregated.json` / `stats_summary.json` inside each run's output dir.
  `python src/evaluation.py --load <dir>` is idempotent — safe to re-run.
