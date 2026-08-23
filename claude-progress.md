# Progress Log (Claude sessions)

Update this at the end of every session (Principle 5 & 12). This is what the next
session reads to avoid starting from zero.

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