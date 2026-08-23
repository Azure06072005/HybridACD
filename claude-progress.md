# Progress Log (Claude sessions)

Update this at the end of every session (Principle 5 & 12). This is what the next
session reads to avoid starting from zero.

## Session 4 — 2026-08-23
- Completed: F007 prep continued (not a scoped feature of its own — logged
  here as F007 prep work, not a new tracked item, per WIP=1). A session with
  repo access created `consistency-forecasting/src/hybrid_acd_config.py`
  (renamed from an initial `src/config.py` after discovering a namespace
  collision with the existing `forecasters/llm_forecasting/config/constants.py`
  subpackage — good catch, avoided a confusing import shadow) plus
  `tests/test_config.py`. Reconciled against the real `HybridACDForecaster`
  fields on disk (`model`, `adversarial_model`, `preface`, `examples`,
  `adversarial_enabled`, `tcd_enabled`, `research_enabled`) — confirms no
  retrieval field exists yet, so F007 is genuinely greenfield there, not an
  audit-and-extend of something already present. 16/16 tests passed
  (test_config.py + test_hybrid_acd_forecaster.py combined).
- IMPORTANT CAVEAT, not yet resolved: `hybrid_acd_config.py` is tested and
  correct in isolation but NOT wired into `hybrid_acd_forecaster.py` —
  nothing in the real forecaster reads `TCDConfig.logit_bias_value`,
  `probability_grid_step`, or `AdversarialConfig.max_rewrites_per_question`
  yet. It's scaffolding, not yet load-bearing. Do not describe TCD/adversarial
  behavior as "configurable" in any future write-up until this wiring exists
  and has its own test proving the forecaster actually reads these values.
- In progress: —
- Blocked: F007 still blocked_by F003, unchanged. Three sessions of F007
  design/prep work (spec, ADR updates, config scaffolding) have accumulated
  with zero sessions actually running F003.
- Next session should: STOP adding further F007 prep and run F003 for real —
  `python src/evaluation.py --tuple_dir src/data/tuples/scraped --num_lines 5
  --run --async -k all -p src/forecasters/hybrid_acd_forecaster.py::HybridACDForecaster
  -o model=gpt-4o-mini --output_dir src/data/forecasts/_smoke_run` — confirm
  a real API key is set in `.env` first (small live cost). F007 cannot move
  past `not_started` until this passes; further config/spec work without it
  is scope accumulating ahead of its own prerequisite.

## Session 3 — 2026-08-23
- Completed: Design-only, no code changes. Received and reviewed a direct source
  audit (from a session with real repo access) of ResolverBasedForecaster's
  search pipeline. CONFIRMED (not just suspected): its Brier 0.096/0.088 on the
  scraped set is outcome leakage, not forecasting — created_date/resolution_date
  are only interpolated into prompt text, never passed to the actual Perplexity
  API call, and Perplexity's own recency filter is relative-only anyway (cannot
  express an absolute historical cutoff even if wired up). Updated F007 and
  ADR-003 to record this as confirmed evidence and to add a hard requirement:
  F007's search provider must support a verified absolute end-date API
  parameter (not prompt text, not a relative filter) — named Tavily/Exa/Bing
  News as provider candidates to verify against current docs. Drafted a
  standalone retrieval-wrapper design spec (retrieval_step_spec.md) implementing
  ADR-003's call order and the cutoff-enforcement requirement, ready to hand to
  a session with repo write access.
- In progress: —
- Blocked: F007 still blocked on F003 (unchanged), and now also on picking one
  concrete search provider and verifying its absolute-date-cutoff parameter
  against that provider's current API docs before any code is written against it.
- Next session should: still run `./init.sh` / start F003 first (unchanged
  since Session 0). If continuing F007 prep: pick a provider from the spec's
  candidate list, verify its date-cutoff parameter against live docs, then
  implement retrieval_step_spec.md's design into hybrid_acd_forecaster.py.

## Session 2 — 2026-08-23
- Completed: Design-only, no code changes. Confirmed (via a separate local
  session's directory survey, not independently re-verified in this session)
  that non-retrieval baselines for F007 already exist on disk — 193 forecast
  run directories, covering BasicForecaster/CoT across gpt-4o-mini, gpt-4o,
  claude-3.5-sonnet, llama-3.1-8B/70B/405B, o1-preview, on both the scraped
  (242Q) and newsapi (1000Q) ground-truth sets plus tuples_scraped AVS. Updated
  F007's verification field accordingly — no fresh non-retrieval runs needed.
- In progress: —
- Blocked: New blocker discovered — a ResolverBasedForecaster (Perplexity
  sonar-large) run already exists with Brier 0.096 on the scraped set, well
  below every other baseline including large models. This is suspicious per
  Section 6 ("a metric that improved is not automatically good news") because
  the scraped set resolved May-Aug 2024 and is being evaluated ~2 years later
  — if that run's search calls don't enforce a pre-resolution date cutoff, the
  0.096 may reflect outcome leakage (the resolver finding articles reporting
  the actual result) rather than a genuine retrieval win. F007 is now blocked
  on auditing that run's search code for a date filter before its number can
  be trusted as an upper-bound reference.
- Next session should: still run `./init.sh` / start F003 first (unchanged).
  If picking up F007 prep work instead, the very first step is grep'ing
  ResolverBasedForecaster's implementation for how it calls its search API and
  whether a date cutoff is passed — do this before writing any new retrieval
  code, since it determines whether 0.096 is a real target or a leaked one.

## Session 1 — 2026-08-23
- Completed: Design-only session, no code changes. Verified GitHub push actually
  succeeded (earlier apparent mismatch was a stale/partial root-tree render on
  fetch, not a real push failure — confirmed via direct blob URLs to AGENTS.md,
  feature_list.json, DECISIONS.md, docs/architecture.md, all byte-identical to
  local copies). Did literature research on alternatives to TCD (RL-trained
  consistency, per-checker regression parallels to F006's Mistral problem,
  post-hoc arbitrage's known non-generalization). Scoped and added F007
  (retrieval-augmented small-model forecasting) to feature_list.json in
  not_started state, and ADR-003 to DECISIONS.md documenting the retrieval
  design (once-per-question, before rewrite/TCD, hard date-cutoff filter using
  created_at not resolution date, to prevent outcome leakage on already-resolved
  tuple sets like `scraped`).
- In progress: —
- Blocked: F007 is explicitly blocked on F003 (smoke run harness must be proven
  working before adding a retrieval step on top of it) — do not start F007
  implementation before F003 passes.
- Next session should: still run `./init.sh` and start F003 first, per Session 0's
  note — F007 does not change that priority, it's queued behind it.

## Session 0 — 2026-08-23
- Completed: Repo research only — read `paper/HybridACD_en.pdf` in full, mapped
  the `consistency-forecasting/` codebase (forecasters, static_checks,
  evaluation.py, existing test file), confirmed `hybrid_acd_forecaster.py` and
  `tests/test_hybrid_acd_forecaster.py` already exist and the two config/bounds
  tests already pass. Filled in harness docs (`docs/architecture.md`,
  `docs/verification.md`, `docs/conventions.md`) and seeded `feature_list.json`
  with F001-F006 reflecting what's already proven vs. what needs a live run.
  No code changes made this session.
- In progress: —
- Blocked: —
- Next session should: run `./init.sh` to confirm the environment actually
  installs and the existing tests pass locally, then start F003 (smoke run) —
  needs a real API key in `consistency-forecasting/.env` first. Do NOT jump to
  F004 (full 200-line reproduction) until F003 passes cheaply.

<!--
Template for future entries:

## Session N — YYYY-MM-DD
- Completed: F0xx (name) — all tests passing, evidence: commit <hash>
- In progress: F0yy (name) — what's done, what's left
- Blocked: (dependency / decision needed, or "none")
- Next session should: <one concrete next action>
-->