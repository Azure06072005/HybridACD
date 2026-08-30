# Progress Log (Claude sessions)

Update this at the end of every session (Principle 5 & 12). This is what the next
session reads to avoid starting from zero.

## Session 9 — 2026-08-29
- Completed: All three Session 8 decision points resolved.
  1. Endpoint settled: `wokushop` (https://llm.wokushop.com/v1, gpt-5.6-luna)
     is now the single project-wide default, superseding xah.io. Logged as an
     addendum in DECISIONS.md — should be folded into ADR-004's original text
     by whichever session has it, so the ADR reads as one settled endpoint
     rather than the two-endpoint drift across Sessions 7-8.
  2. F004 scope locked: BOTH scraped (paper-comparable, ADR-005 caveat
     required alongside any citation) AND 2028 (leakage-clean, no paper
     equivalent to compare against) runs are required, not optional, not
     sequential-with-a-pick. feature_list.json F004 updated accordingly and
     unblocked (blocked_by set to null).
  3. Official-provider logit-bias check explicitly deprioritized this round
     ("optimize for decision") — ADR-006's open question (proxy-specific vs.
     general logit-bias inefficacy) remains unresolved but is not gating
     F004. Worth returning to opportunistically if real OpenAI credits ever
     become available, not worth spending a session on right now.
- In progress: —
- Blocked: — (F004 is now genuinely unblocked)
- Next session should: run F004 exactly as scoped — smoke-scale (num_lines=5)
  on both scraped and 2028 first per small-before-large, THEN the full
  200-line run on both once smoke-scale looks sane on both. Do not run only
  scraped and call it done; do not run 2028 alone and skip the paper
  comparison either — both numbers are required in the same write-up.

## Session 8 — 2026-08-29
- Completed: Formally closed F003 as `passing` in feature_list.json — it had
  been mechanically proven for several sessions (scraped sweep, 2028 spot-
  check, TCD intervention logging, logit-bias inefficacy investigation) but
  never actually flipped from `not_started`. Evidence field now consolidates
  all of it in one place, including the ADR-005 leakage caveat and ADR-006
  logit-bias finding, so no future session has to re-derive the trail.
- Decision point identified, NOT resolved this session: F004's own spec
  requires model=gpt-4o-mini for direct paper-table comparability. Three
  things are in tension: (1) ADR-004 declared gpt-5.6-luna the permanent
  project model, (2) that model has itself drifted between two unverified
  reseller endpoints (xah.io, wokushop) without either being settled as THE
  endpoint, (3) even if a model/endpoint is picked, ADR-005 means scraped
  alone won't give a clean accuracy measurement for any post-2024-cutoff
  model — F004 likely needs to run on both scraped (nominal paper
  comparability, leakage caveat attached) and 2028 (clean signal, but not
  comparable to the paper's number at all, since the paper never tested on
  2028) to say anything meaningful.
- In progress: —
- Blocked: F004 blocked on the above decision, not on any technical issue.
- Next session should NOT start F004 code/runs until this is explicitly
  decided: (a) settle ONE endpoint for gpt-5.6-luna (or get real gpt-4o-mini
  access) and update ADR-004 to say which, (b) decide whether F004's
  behavior/verification text itself should be amended to require both
  scraped AND 2028 runs, given ADR-005 — if so, edit F004's spec explicitly
  before running anything, don't quietly run both and decide after which one
  "counts." Optionally: run the official (non-proxy) OpenAI logit-bias check
  flagged in ADR-006 first, since that's cheap, independent of this decision,
  and would settle whether logit-bias inefficacy is proxy-specific or general.

## Session 7 — 2026-08-29
- Completed: Model policy resolved explicitly — ADR-004 formally adopts
  `levuphong2909/gpt-5.6-luna` (via reseller) as the permanent benchmark model,
  superseding the earlier gpt-4o-mini-default policy from Session 6. F003's
  full `-k all` smoke run (num_lines=5, 10 checkers, 50 samples) completed:
  AVS=0.000 across every checker and every metric (default/frequentist/
  default_scaled), 0/50 violations. Verified via ruff (clean) and pytest
  (16/16). Committed and pushed (9dbcc77, then 85f06c7 for ADR-004).
- NEW FINDING (ADR-005): inspecting raw traces across NegChecker, AndChecker,
  ButChecker, ExpectedEvidenceChecker shows every base prediction in all 50
  samples was exactly 0.0 or 1.0, with CoT citing already-resolved 2024
  outcomes as settled fact. Since `scraped` resolved May-Aug 2024 and
  gpt-5.6-luna is a much more recent model, this is very likely training-data
  memorization, not genuine forecasting under uncertainty — a parallel
  leakage vector to ADR-003's search-based one, just via parametric memory
  instead of live retrieval. AVS=0.000 under these conditions is expected
  from memorization alone and is NOT strong evidence that HybridACD's
  consistency mechanism (TCD/adversarial rewrite) is doing meaningful work —
  though it does still validly confirm the pipeline mechanics (bound math,
  clipping, elicit sequencing) function correctly, which is what F003 itself
  checks.
- F003 status: mechanical criterion satisfied (exits 0, non-null AVS) — can
  be marked `passing` on that narrow basis. The AVS=0.000 number itself must
  carry the ADR-005 caveat in any future citation and must not be presented
  as evidence of forecaster quality.
- In progress: —
- Blocked: —
- Next session should: before starting F004 or F007 proper, run a spot-check
  smoke test on `src/data/tuples/2028` (not yet resolved, so no training-data
  leakage is possible) and confirm predictions show genuine intermediate
  probabilities rather than another 0/1 sweep. If 2028 also comes back all-
  extreme, that's a different finding (e.g. model refusing to express
  uncertainty) worth its own investigation before F004/F007 proceed.

## Session 6 — 2026-08-23
- Completed: Resolved both caveats from Session 5.
  (1) Inspected raw predictions behind the NegChecker AVS=0.000 result
  directly (`_smoke_test_gemini/NegChecker.jsonl`) rather than trusting the
  aggregate. Confirmed NOT a degenerate/constant collapse — base predictions
  varied per question (e.g. p(P)=0.0 for two different real questions, for
  question-specific reasons in the CoT). NOTE for interpretation, not a
  problem but worth remembering: NegChecker's bound is a single POINT
  ([1-p(P), 1-p(P)]), so AVS=0 here is close to guaranteed by construction
  once TCD clipping works at all — it mainly confirms the clipping mechanism
  functions, not that the forecaster's judgment is good. The real test of
  forecaster quality is checkers with genuine INTERVAL bounds (And, Or, Cond,
  CondCond) where TCD constrains a range but doesn't fully determine the
  answer. Don't let a clean NegChecker result set expectations for those.
  (2) Model policy confirmed explicitly: gpt-4o-mini remains the standard
  benchmark model for F003/F004/F007 comparison purposes. The
  gemini-3-flash-preview run was solely a mechanical pipeline-integration
  check and is not a baseline substitute — logged as policy, not just intent.
  (3) Re-ran test_hybrid_acd_forecaster.py + test_config.py after the
  kwargs-sanitization and llm_utils.py routing fixes: 16/16 still pass,
  F001/F002 confirmed not regressed by the pipeline fixes.
- In progress: F003 (pipeline mechanically proven; blocked only on
  gpt-4o-mini provider quota/credit refresh for the real full 10-checker run)
- Blocked: provider balance/quota for gpt-4o-mini, unchanged from Session 5.
- Next session should: once credits available, run the FULL `-k all` smoke
  command on gpt-4o-mini per docs/verification.md exactly. When results land,
  check the interval-bound checkers (And/Or/Cond/CondCond) with the same raw-
  prediction-inspection discipline used here for NegChecker before marking
  F003 passing — they're the harder, more informative test.

## Session 5 — 2026-08-23
- Completed: F003 prep/debugging (still not_started — no session marks this
  passing). Fixed a real bug blocking the smoke run: `adversarial_rewrite_async`
  /`_sync` in `hybrid_acd_forecaster.py` were passing non-API kwargs (e.g.
  `rule='NegChecker'`) straight through to the completions endpoint; sanitized.
  Also adjusted `llm_utils.py`/`.env` fallback routing to fix a hardcoded model
  mismatch. A partial live run (NegChecker only, not the full `-k all` F003
  requires) reported AVS 0.000 / 0 violations.
- CAVEATS, NOT YET RESOLVED — do not treat as evidence until addressed:
  1. AVS 0.000 on the very first live run after a pipeline bugfix is exactly
     the Goodhart pattern Section 6 warns about (metric improved ≠ good news).
     Before this counts as anything: inspect the actual raw predictions from
     that run (raw p, computed [l,u] bound, post-TCD result) for a handful of
     real NegChecker questions — confirm the forecaster is producing real,
     varied predictions that happen to be consistent, not a degenerate
     constant output that trivially satisfies the checker.
  2. The proposed unblocking command switches model from `gpt-4o-mini` (F003's
     spec) to `gemini-3-flash-preview`, due to an API balance shortfall on the
     original provider. This is a reasonable practical workaround for proving
     the pipeline mechanically works, but it must NOT quietly become the
     model F003 is "passing" with for baseline-comparison purposes — F004
     still requires gpt-4o-mini specifically (to match paper Table 2), and
     F007's comparison table is keyed to the existing on-disk gpt-4o-mini/
     gpt-4o baselines. If gemini-3-flash-preview is used to unblock F003
     mechanically, log that explicitly as "F003 passing evidence is on a
     substitute model" and still get a real gpt-4o-mini smoke run before
     F004 starts.
  3. Re-run `test_hybrid_acd_forecaster.py`'s bound-math (F001) and config
     (F002) tests specifically AFTER these latest kwargs/routing fixes — the
     16/16 pass count on record predates these changes and doesn't cover them.
     Per docs/conventions.md, this file is tested and paper-cited; a fix to
     its adversarial-rewrite path should not silently go unverified against
     F001/F002's existing tests in the same change.
- In progress: F003 (blocked on API credit top-up, mechanically otherwise close)
- Blocked: provider balance ($0.0469 available vs $0.0500 required per call,
  per this session's report) — needs a top-up before the full `-k all` run.
- Next session should: (a) top up API credits, (b) re-run F001/F002 tests to
  confirm the pipeline fixes didn't regress them, (c) run the FULL `-k all`
  smoke command (not just NegChecker) — decide explicitly whether on
  gpt-4o-mini or gemini-3-flash-preview and log which, (d) inspect raw
  predictions behind any AVS 0.000 before treating it as real, (e) only then
  update F003 to `passing` with the actual stats_summary.json evidence.

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