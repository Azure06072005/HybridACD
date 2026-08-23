# Progress Log (Claude sessions)

Update this at the end of every session (Principle 5 & 12). This is what the next
session reads to avoid starting from zero.

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
