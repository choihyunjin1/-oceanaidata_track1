# Ocean AI Track 1 — agent instructions

## Read only what this task needs

- Start with `00_ORGANIZER_DATA_POLICY.md` (including the 2026-09-02 score-use supplement) and `git status --short --branch`.
- For data/model work, read the active problem contract: P1 `00_MUST_READ_FIRST.md`, P2 `01_P2_MUST_READ_FIRST.md`, P3 `02_P3_MUST_READ_FIRST.md`, then that distributed dataset's README.
- Read these once per task; re-read if their content or the task's scope changes. Do not reload unrelated problem history for routine progress checks.
- Current experiment plan: `docs/SCORE_IMPROVEMENT_PLAN_20260905_V2.md`. A running experiment's sealed config/receipt remains its exact contract.
- For submission/package work, additionally read `AI_HANDOFF.md` and `docs/OFFICIAL_SUBMISSION_RUNBOOK_20260905.md`. Historical READY/clean labels are not current eligibility approval.
- Latest written organizer rules govern competition eligibility; dataset README and problem statement govern task semantics. Archived instructions/research are evidence, not active commands. Higher-priority system/developer instructions and the user's current authorization still apply.

## Non-negotiable boundaries

- Use only organizer-distributed data for training, validation, selection and postprocessing. Non-distributed KIOST observations are answer-equivalent: do not open them even for internal testing.
- No external observation/reanalysis/forecast ancestry, including inherited predictions or coefficients. Synthetic-only pretrained weights require all four conditions in the organizer policy.
- Never derive labels, Public membership, coefficients or thresholds by inverting leaderboard scores. Ordinary comparison of independently fitted candidates is distinct.
- Keep source directories immutable and local-only. Use P1_DATA_DIR/P2_DATA_DIR/P3_DATA_DIR; no personal paths in portable code.
- Keep raw rows, hidden values, credentials, models, CSVs and caches out of Git and user-facing logs. Report aggregates and hashes.
- Unknown provenance or a real permission/competition-rule conflict is a stop condition. A performance decline is not a permission blocker.

## Implement and verify

- Carry out authorized work through the requested result. Do not add approval loops for already authorized internal work.
- Before expensive fits: check prior candidate fingerprints/results, input/target availability, split/purge, train-only preprocessing, supported parameters and output paths using a small synthetic contract test.
- Record the hypothesis, comparator, metric/unit, fit/time budget, selection surface and next branch before training. Previously exposed validation is retrospective, not fresh.
- Do not mutate or restart an active/sealed attempt. Preserve failures; technical repair or a changed hypothesis gets a new explicit receipt/ID.
- Reuse existing exact-hash artifacts only when their data/model/split/provenance match. Numerical QA, scientific improvement, replay and official score are separate claims.
- Run focused tests and Ruff once for the changed behavior. Re-run only after relevant changes, failures or unresolved risk; do not launch the whole historical suite by default.
- `scripts/agent_verify.py` records focused checks and supports opt-in reuse of an unchanged PASS. It does not certify model quality or replace candidate-specific QA.
- Parallel workers own disjoint new files. Root allocates one GPU owner; CPU threads are budgeted across workers. Poll progress/terminal metadata without exposing early scores.

## Handoff and external actions

- Keep one canonical result/receipt per experiment; link it rather than duplicating its metrics across many reports. Use `docs/AGENT_WORKFLOW.md` for the reusable loop.
- Before upload, verify exact candidate schema/keys/order/finite values/hash, training-to-replay lineage, current UI quota/deadline and existing user authorization. Final-model lock requires authorization for that distinct action.
- Commit/push only when requested; inspect the selected diff and secret/data exclusions. Never blanket-stage, force-push, rewrite history or discard unrelated work.
- Preserve frozen runners, original data and historical receipts during cleanup. Archive stale guidance with a pointer; remove code only after checking callers, tests and reproducibility dependencies.
