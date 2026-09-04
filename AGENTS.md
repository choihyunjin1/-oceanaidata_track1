# Project operating instructions

These rules apply to every task in this repository.

## Authority and mandatory preflight

Use this precedence when instructions conflict:

1. latest written organizer rule or direct organizer answer; the active 2026-09-01 snapshot is `00_ORGANIZER_DATA_POLICY.md`
2. source dataset README.md
3. latest problem statement supplied by the user
4. 00_MUST_READ_FIRST.md and the problem-specific must-read file
5. implementation notes and experiment records

Before reading data, changing code, running a notebook, training, using Git, or preparing a submission:

1. Read `00_ORGANIZER_DATA_POLICY.md` completely.
2. Read 00_MUST_READ_FIRST.md completely.
3. For P2 work, also read 01_P2_MUST_READ_FIRST.md completely. For P3 work, read 02_P3_MUST_READ_FIRST.md completely.
4. Read the source README for the active problem: P1_qc_anomaly/README.md, P2_profile_restore/README.md, or P3_wave_forecast/README.md.
5. Check git status --short --branch.
6. Confirm the action does not mutate or redistribute source data.
7. Stop and ask if any rule, data right, pretrained-weight provenance, or submission action is ambiguous.

## Source-data boundary

- Treat folders supplied through P1_DATA_DIR, P2_DATA_DIR, or P3_DATA_DIR as immutable, local-only input.
- Never edit, move, rename, copy into tracked paths, commit, push, upload, or redistribute the source ZIP/CSV/README/score files.
- Never include raw observation rows or values in Markdown, notebook prose, logs, screenshots, issues, or commit messages. Aggregated statistics and cryptographic hashes are allowed.
- Generated models, predictions, submissions, caches, and large extracts belong only in ignored local directories.
- A public download link does not override the source license or competition rules.

## Leakage and validation

- Keep test labels unknown. Do not reconstruct them from KORS/KHOA raw or real-time observations, public mirrors, exact source matching, leaderboards, or hidden-answer artifacts.
- Do not recover hidden answers from exact-source observations: P1 clean 2026 temperature, P2 hidden 2025-09/10 target-layer temperature/salinity, or P3 anonymous-case timestamps/future waves.
- Do not use any non-distributed observation, reanalysis, forecast, derived prediction, or feature. Public availability and historical-only timestamps do not create an exception.
- Do not use random row splits as primary validation. Preserve station/layer groups, chronological blocks, anomaly runs, and real observation gaps.
- Purge or embargo validation boundaries by at least the maximum feature and post-processing dependency.
- Fit scalers, imputers, station/layer baselines, feature statistics, thresholds, and post-processing parameters on the training portion of each fold only.
- Bidirectional offline QC is the approved primary workflow because the task distributes complete time series and states no online-only restriction. Keep every centered feature inside a continuous segment and protect outer folds with a purge longer than its dependency window.
- Maintain a strictly causal mode as an operational ablation. If a later written organizer rule forbids future context, promote the causal result and retire the offline candidate.
- Do not force the test positive rate to match train prevalence or the baseline submission.

## Competition data and pretrained-weight prohibition

- The 2026-09-01 organizer notice supersedes the older FAQ receipt that allowed public external data.
- Train, validate, select, calibrate, ensemble, and post-process only with organizer-distributed data. A downstream model remains ineligible when it inherits predictions or parameters from an external-data model.
- Treat non-distributed KIOST source observations as answer-equivalent and never access or compare them for competition work.
- Do not use pretrained weights learned from real observation, weather, or ocean data.
- A synthetic-only pretrained model is allowed only when all four organizer conditions in `00_ORGANIZER_DATA_POLICY.md` are proven. Unclear provenance means forbidden.
- Preserve historical external-data artifacts and receipts as audit evidence, but never reuse them or include them in the reproducibility package. Never place external raw files in Git.

## Portable and reproducible code

- Do not hard-code a personal absolute path, drive letter, username, or Korean source-directory path in Python or notebooks.
- Resolve P1 inputs from P1_DATA_DIR, P2 inputs from P2_DATA_DIR, and P3 inputs from P3_DATA_DIR. A repository search fallback may be used for local convenience and must fail on zero or multiple matches.
- Use UTF-8 for source and documentation files.
- Keep runtime parameters, seeds, split definitions, feature windows, and source provenance visible.
- Reader-facing notebooks must execute top-to-bottom, display aggregate output only, and use the section order tl;dr, Context & Methods, Data, Results, Takeaways.
- The official notice warns that unreproducible code, missing environment information, and Korean filename/path errors can cause disqualification.

## Experiment and submission records

For each candidate, record:

- experiment ID and timestamp in KST
- Git commit and dirty-worktree state
- Python/package environment and seed
- input hashes and feature/model/post-processing version
- exact fold dates, groups, purge length, and metrics
- candidate path, byte size, schema result, and SHA-256

The current official interface allows up to three prediction uploads per problem per day. Never upload without the user's explicit approval for the exact file. Before approval, run:

~~~powershell
.venv-p1\Scripts\python.exe scripts\validate_submission.py <candidate.csv>
~~~

## Official milestone

The 2026-08-07 and 2026-08-12 participant notices set problem release to 2026-08-13 and the final-model deadline to 2026-09-07. The current interface allows three prediction uploads per problem per day and locks later prediction uploads after final-model submission. The initial KIMST PDF's 2026-08-10 through 2026-09-04 schedule is retained only as superseded evidence. Re-check the participant UI immediately before any final action.

## Git safety

- Intended remote: https://github.com/choihyunjin1/-oceanaidata_track1
- Back up code, notebooks, rules, and small aggregate reports only.
- Inspect git status and git diff --cached before every commit.
- Do not commit, push, merge, or submit unless the user or coordinating agent explicitly requests that exact action.
