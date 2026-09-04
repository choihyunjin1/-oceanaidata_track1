# Meaningful official-score goal v2

This is the append-only control plane for the P1/P2/P3 improvement campaign.
It does not upload files or call the competition platform.

## Canonical state

- Contract: `configs/goals/meaningful_score_maximization_v2.json`
- Ledger: `artifacts/meaningful_score_goal_v2/registry.jsonl`
- Evaluator: `src/ocean_goal/meaningful_score.py`
- CLI: `scripts/run_meaningful_score_goal.py`
- Tests: `tests/test_meaningful_score_goal.py`

Version 2 supersedes version 1 because the authenticated participant notice says
that the answer-upload limit is **three per team per day**, not three per problem.
The official scoring window opens on 2026-08-25 and the safe final-model deadline
is 2026-09-07. Final model designation prevents later answer uploads.

## Commands

```powershell
.venv-p1\Scripts\python.exe scripts\run_meaningful_score_goal.py initialize
.venv-p1\Scripts\python.exe scripts\run_meaningful_score_goal.py status
.venv-p1\Scripts\python.exe scripts\run_meaningful_score_goal.py curve --evidence <learning_curve_evidence.json>
.venv-p1\Scripts\python.exe scripts\run_meaningful_score_goal.py curve --evidence <learning_curve_evidence.json> --append
.venv-p1\Scripts\python.exe scripts\run_meaningful_score_goal.py prepare-upload --approval <approval.json> [--curve-decision <decision.json>]
.venv-p1\Scripts\python.exe scripts\run_meaningful_score_goal.py completion --evidence <final_completion_evidence.json>
```

Run every decision without `--append` first. Append only after read-only integrity
review. `prepare-upload` returns readiness evidence only; a separate platform
action still requires exact-file user approval at action time.

## Learning-curve evidence contract

Each generation must bind its pre-fit config path and SHA, contain at most three
structurally distinct hypotheses, and declare that no score-derived tuning was
used. Every one of the 40/55/70/85/100% points must contain:

- a fresh incumbent refit and fresh challenger refit on identical folds and keys;
- three fixed seed metrics for both models;
- the metric of the three-seed prediction-mean ensemble;
- a 5,000-replicate paired cluster-bootstrap interval;
- exact reference-seed reproduction of the frozen incumbent at 100%.

The central evaluator then applies direction, absolute-effect, late-curve,
confidence-interval, fold, critical-slice, leakage, and reproducibility gates.
Failure of any gate produces `RESEARCH_ONLY`.

## Official-score states

- `score_incumbent`: any strict point gain on the same scoring version and split,
  provided approval and reproducibility checks pass.
- `meaningful_incumbent`: additionally requires a curve-qualified candidate and
  an official raw-metric gain meeting the problem-specific absolute threshold.
- `COMPLETE`: requires meaningful promotion and final/private retention for all
  three problems, a strictly higher total portfolio score, byte reproduction,
  final model packaging, and organizer verification.

Waiting for the official window, user approval, a score, or the next daily reset
is nonterminal. Local point estimates and generated model files are nonterminal.
