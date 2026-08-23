---
trigger: always_on
---

Behavioral guidelines to reduce common LLM coding *and* research mistakes.
Merge with project-specific instructions (`AGENTS.md`, `docs/*.md`) as needed.
This project (HybridACD) is empirical ML research on top of an existing
codebase (`consistency-forecasting/`), not greenfield software — sections 5-9
below exist because "looks correct" and "the code runs" are not the same as
"the claim is true." Numbers get cited in a paper; treat them accordingly.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## 5. Karpathy-Style Research Code Discipline

Source: Karpathy's recurring public advice on nanoGPT/llm.c-style work — code
you can hold entirely in your head beats code that's "properly engineered."

- **Legibility over abstraction.** A 300-line script you can read top-to-bottom
  in one sitting beats a "clean" 8-file framework with the same behavior. Prefer
  editing `hybrid_acd_forecaster.py` directly over introducing a new plugin
  system, strategy pattern, or config-of-configs to accommodate one new rule.
- **Delete the framework impulse.** If a change "would be cleaner with a new
  base class / registry / factory," that's a signal to stop, not a plan. Two
  call sites don't justify an abstraction. Three might. Ask first either way
  (Rule 2).
- **Bugs hide in the diff between what you think the code does and what it
  does.** Before trusting any new number, print/log the actual tensors/values
  at the boundary — for this repo: print the raw model probability, the
  computed `[ℓ,u]` bound, and the post-TCD clipped/logit-biased result for a
  handful of real examples. Don't trust `stats_summary.json` you haven't looked
  behind.
- **One change, one number moves.** When investigating why a metric changed
  (AVS, Brier, Calibration Error), change exactly one variable (model,
  `adversarial_enabled`, `tcd_enabled`, dataset) between two runs you compare.
  Never read across runs that differ in more than one axis and attribute the
  delta to the one you care about.
- **Overfit to one example first.** When implementing or debugging a new
  consistency rule's bound formula, hand-verify it on one concrete example
  (paper-style: pick real `p_P`, `p_Q` values, compute `[ℓ,u]` by hand,
  compare to code output) *before* running it across a whole tuple set. If it's
  wrong on one example, it's wrong on all of them — a 200-line async eval run
  will not tell you that faster than a 5-line manual check.
- **Reproducibility is a first-class deliverable, not an afterthought.** Every
  run that produces a number that might get cited (in `feature_list.json`
  evidence, in progress logs, in any writeup) must record: exact command, model
  ID + provider, dataset/tuple_dir + `num_lines`, git commit hash, and output
  directory path. If you can't answer "what exact command produced this
  number" from the run's own artifacts, the number doesn't count as evidence.

## 6. Baseline-First Research Workflow

- **Never report a treatment number without its baseline in the same table.**
  "HybridACD got AVS 0.03" is meaningless without "BasicForecaster on the same
  model/dataset/sample got AVS X." Pull the baseline from existing
  `src/data/forecasts/*` dumps before running a new one if one already exists
  for that model — check first, don't waste API budget re-deriving what's
  already on disk.
- **Small before large.** Any new evaluation configuration gets a `--num_lines
  5`-scale smoke run before a `--num_lines 200`+ full run. This mirrors F003 →
  F004 in `feature_list.json` — do not skip the smoke tier to save a session.
  Live-API runs cost real money; a broken config caught at 5 lines costs cents,
  caught at 200 costs dollars.
- **Ablate one component at a time.** To attribute an effect to "the
  adversarial rewriter" vs "TCD," you need the 2x2 (neither / adv-only /
  tcd-only / both) — not just "with vs. without everything." The paper's own
  Table 5 already does this; new ablations should follow the same pattern
  (isolate `adversarial_enabled`, isolate `tcd_enabled`).
- **A metric that improved is not automatically good news.** If AVS drops
  suspiciously close to zero, check for the Goodhart failure mode the paper
  explicitly worries about (§Discussion): is the forecaster just outputting a
  degenerate/neutral value that trivially satisfies constraints instead of
  actually forecasting? Cross-check against Brier Score before believing an AVS
  improvement is real progress. This applies to any future consistency-rule
  work, not just the two components in the paper.
- **Treat "the paper's number" as a hypothesis to reproduce, not ground
  truth to assume.** F004 in `feature_list.json` exists because the paper's
  reported 0.0007 AVS on GPT-4o-mini needs independent reproduction before any
  downstream work builds on it as fact. If your reproduction doesn't match
  within reasonable tolerance, that's a finding — log it in
  `claude-progress.md`, don't quietly adjust the run until it matches.

## 7. Statistical and Measurement Hygiene

- **Sample size is part of the claim.** "`num_lines=5`" and "`num_lines=200`"
  produce numbers that are not comparable and not equally trustworthy — always
  state sample size alongside any metric, in code comments, progress logs, and
  any summary you write.
- **Watch for silent fallback paths masking failure.** `hybrid_acd_forecaster.py`
  falls back from logit-bias TCD to deterministic clipping when the API doesn't
  support logit bias or parsing fails (see `docs/conventions.md`). A run can
  "succeed" while silently using the weaker fallback path for every question.
  When investigating a metric, check whether the fallback triggered (log/count
  it) before concluding the primary mechanism (logit-bias TCD) is what produced
  the number.
- **Don't average across checker types unless the paper does.** AVS is an
  aggregate; per-checker breakdowns (NegChecker vs CondCondChecker etc.) can
  hide that one rule is doing all the work while another is broken. When
  debugging, read `stats_{CheckerType}.json` files individually, not just
  `stats_aggregated.json`.
- **Calibration Error regressions (F006) are a known open problem — don't
  paper over them.** If a change improves AVS but worsens Calibration Error,
  report both. Do not mark a feature `passing` in `feature_list.json` if it
  regresses a metric the task didn't intend to trade off, without flagging it
  explicitly in the evidence field.

## 8. Research Session Discipline (extends Section 4's WIP=1)

- **One hypothesis per session, matching one `feature_list.json` entry.**
  "Let's also check if it's the temperature" mid-session is a new hypothesis —
  name it, decide if it's this session or the next, don't silently fold it in.
- **State the falsifiable prediction before running.** Before a comparison run,
  write down what result would confirm vs. disconfirm the hypothesis. E.g. for
  F005 (ablation): "predict AVS(full) < AVS(TCD-only) < AVS(adversarial-only),
  matching paper Table 5 ordering — if the order differs, that's the finding,
  not noise to explain away."
- **Negative results are results.** If a reproduction fails or an ablation
  doesn't match the paper's ordering, that goes in `claude-progress.md` with
  the same weight as a success — do not quietly retry with different seeds/
  models until something confirms the expected story.

## 9. When Research Meets the Coding Rules Above

- Rule 2 (Simplicity First) applies doubly to research code: a quick, throwaway
  analysis script for one question ("does interval width correlate with
  calibration error, F006") does not need tests, CLI args, or config files. It
  needs to answer the question and be discarded or promoted deliberately — say
  which, don't leave ambiguous scratch scripts in `src/`.
- Rule 3 (Surgical Changes) applies to `hybrid_acd_forecaster.py` especially:
  it's tested, cited in a paper, and has known-passing bound-math tests (F001,
  F002). Changing bound formulas without updating
  `tests/test_hybrid_acd_forecaster.py` in the same change is not allowed —
  the tests are the executable version of paper Table 1, and drift between
  them is exactly the kind of silent bug this document exists to prevent.
- Rule 4's "verify, don't assume done" applies to metrics as much as to tests:
  a feature that touches evaluation code is not `passing` on "the eval script
  ran without an exception." It's `passing` when the produced numbers were
  sanity-checked per Sections 6-7 above.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer
rewrites due to overcomplication, clarifying questions come before
implementation rather than after mistakes — **and, for the research-specific
sections:** every metric reported in `feature_list.json` evidence has a
reproducible command attached, every ablation isolates one variable, and no
"the number went down, great" claim ships without a Goodhart/fallback-path
sanity check.