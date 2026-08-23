# F007 Retrieval Step — Implementation Spec

Status: design only, not implemented. Blocked on F003 and on provider selection (see "Provider requirement" below). This spec exists so implementation can start immediately once both blockers clear, without re-deriving the design.

## Why this file exists

`ResolverBasedForecaster`'s existing retrieval path was audited (2026-08-23, see `DECISIONS.md` ADR-003) and confirmed to leak resolution outcomes: it interpolates dates into prompt text only, never constrains the actual search API call, so live web search on already-resolved questions returns articles reporting the answer. This spec is written specifically to not repeat that mistake — the cutoff is enforced at the API call level, not merely in the prompt text.

## Provider requirement (must resolve before writing code)

The search provider MUST expose a verified, absolute (not relative) end-date parameter at the API level. Perplexity is disqualified — its `search_recency_filter` only accepts `"month"`/`"week"`/`"day"`/`"hour"`, which cannot express "before 2024-05-01" at all, regardless of how it's wired up.

Candidates to verify (check current docs directly before implementing):
- **Tavily**: `end_date` parameter on search requests (e.g. `YYYY-MM-DD`)
- **Exa (formerly Metaphor)**: `end_published_date` parameter (e.g. ISO-8601 string)
- **Bing News Search**: date-range query parameters (e.g. `freshness` or custom date filters)
- **Google Custom Search / Serper**: `tbs=cdr:1,cd_max:...` parameter

Whichever provider is chosen, write a one-line note in `DECISIONS.md` recording which provider and which exact parameter name was verified, with a link/reference to the docs version checked.

## Call order (per ADR-003)

```text
retrieve(question, cutoff_date)
    → context                                    # executed once per base question
    → adversarial_rewrite(question, context)      # existing HybridACD step
    → base_forecast(rewrite, context)              # existing HybridACD step
    → TCD_clip(forecast, bounds)                    # existing HybridACD step
```

Retrieval happens **exactly once per base question**, before the adversarial rewriter generates variants. Do NOT re-run retrieval per rewrite variant — all variants of the same base question share the same underlying facts, and re-searching per variant multiplies API cost for zero benefit.

## Cutoff date selection

Use the question's `created_date` (or whatever field represents "the point at which the forecast is nominally being made"), **never `resolution_date`**. Pass it to the search provider's absolute end-date parameter directly — do not rely solely on prompt-text mentions of the date as a backstop, as prompt-text-only dating produces silent outcome leakage.

## Pseudocode sketch

```python
async def retrieve_context(fq: ForecastingQuestion, provider_client) -> str:
    """
    One retrieval call per base question. Cutoff enforced at the API level.
    Raises if the provider does not accept/honor the cutoff parameter —
    do not silently fall back to unfiltered search.
    """
    cutoff = fq.created_date.strftime("%Y-%m-%d") if fq.created_date else None
    if cutoff is None:
        raise ValueError(f"Question {fq.id} missing created_date for historical cutoff filter.")

    results = await provider_client.search(
        query=fq.title,
        end_date=cutoff,          # exact param name depends on chosen provider
    )
    return format_context(results)

async def call_async(self, fq: ForecastingQuestion, **kwargs) -> Forecast:
    context = await retrieve_context(fq, self.search_client) if self.retrieval_enabled else ""
    rewritten_fq = await self.adversarial_rewrite_async(fq, context=context) if self.adversarial_enabled else fq
    raw_forecast = await self.call_with_tcd_async(rewritten_fq, lower_bound=0.0, upper_bound=1.0, context=context, **kwargs)
    return raw_forecast
```

## Verification requirement before this counts as "done" (per AGENTS.md Principle 9)

Before F007 can move to `passing`:
1. **Leakage smoke test**: run retrieval against a known post-cutoff-only news event and confirm the provider returns zero/irrelevant results — verify that passing the date parameter actually truncates the result set at the search index level.
2. **3-way comparison table**: report (a) small model, no retrieval [existing baseline on disk], (b) large model, no retrieval [existing baseline on disk], (c) small model + retrieval [new run] per `feature_list.json` F007's verification field, at smoke scale (`num_lines=5`) before any full run.
3. **Dual Metric Reporting**: Both AVS (consistency) and Brier Score (accuracy) must be reported together — TCD's consistency guarantee must not regress when retrieval context is injected into the reasoning prompt.
