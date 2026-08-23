# Agent Instructions

This file is the routing manual. Keep it under 200 lines (Principle 4).
Detailed rules live in `docs/*.md` — read those only when the task needs them.

Project: HybridACD (Adversarial Constraint Decoding for LLM forecasting). The
actual codebase is in `consistency-forecasting/` (a fork of
`dpaleka/consistency-forecasting`); the paper is in `paper/`. Almost all agent
work happens inside `consistency-forecasting/src/forecasters/hybrid_acd_forecaster.py`
and its test file — see `docs/architecture.md` before touching anything else.

## Before Starting Any Work (Session Lifecycle — Principle 6)

1. Run `./init.sh` — install deps, run tests, confirm environment is healthy.
   Do NOT write feature code until init.sh passes clean.
2. Read `claude-progress.md` (or `gemini-progress.md`, whichever matches you) —
   what happened last session, what's blocked, what's next.
3. Read `feature_list.json` — what's done, active, blocked, not started.
4. Read `DECISIONS.md` — don't reverse a deliberate prior choice without a new reason.
5. Run `git log --oneline -10` — see recent changes.

If you cannot answer "what is this system / how do I run it / how do I verify it /
what's the current progress" after steps 1-5, STOP and say so. Don't guess (Principle 3).

## Hard Constraints (non-negotiable)

- Work on exactly ONE feature at a time (WIP=1, Principle 7). Do not "also fix" or
  "also refactor" anything outside the current feature's scope.
- Never mark a feature `passing` without a passing verification command. "Looks correct"
  is not evidence (Principle 9).
- Only full-pipeline verification counts: tests + lint + type-check + build + smoke run.
  A single passing unit test is not "done" (Principle 10).
- `passing` state is irreversible — once verified, don't silently downgrade it; if it
  breaks later, that's a regression, log it as one.
- Every session must end in a clean, resumable state (Principle 12) — see checklist below.
- If something is ambiguous or you're inventing scope that wasn't asked for, stop and
  say so in `claude-progress.md` rather than guessing.
- Live-API evaluation runs (F003+) cost real money — never bump `--num_lines` above
  what a feature's verification command specifies without asking first.

## Session Workflow

```
SELECT   → pick exactly one feature from feature_list.json (state: not_started or active)
EXECUTE  → implement → run full verification → fix → re-run until it passes
RECORD   → update feature_list.json state + evidence
WRAP UP  → update progress file, run Session Exit Checklist, commit, stop
```

## Session Exit Checklist (must pass before ending a session)

- [ ] Build passes (n/a for this repo — no compile step, see docs/verification.md)
- [ ] All tests pass
- [ ] Lint clean (`ruff check`)
- [ ] `feature_list.json` and progress file updated
- [ ] No debug code left behind (console.log/print/debugger/TODO markers you added)
- [ ] Standard startup command still works (`./init.sh`)
- [ ] Working tree is committed or explicitly left dirty with a note why

## Where to look for more detail

- `docs/architecture.md` — system structure, module boundaries (read once per session)
- `docs/verification.md` — exact test/lint/build/smoke-run commands for this repo
- `docs/conventions.md` — code style, naming, patterns specific to this project
- `paper/HybridACD_en.pdf` — the source report; ground truth for what the method
  and reported numbers are supposed to be
- Add topic docs here as the project grows. Don't grow this file past ~200 lines —
  split into `docs/` instead.

## When You're Stuck

Attribute the failure to one of five layers before asking the human for help:
task specification, context provision, execution environment, verification feedback,
or state management. Say which layer, then ask your question.
