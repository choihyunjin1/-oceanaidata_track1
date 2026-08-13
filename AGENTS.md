# Project operating instructions

These rules apply to every task in this repository.

## Authority and mandatory preflight

Use this precedence when instructions conflict:

1. latest written organizer rule or direct organizer answer
2. source dataset README.md
3. latest problem statement supplied by the user
4. 00_MUST_READ_FIRST.md
5. implementation notes and experiment records

Before reading data, changing code, running a notebook, training, using Git, or preparing a submission:

1. Read 00_MUST_READ_FIRST.md completely.
2. Read the source P1_qc_anomaly/README.md completely.
3. Check git status --short --branch.
4. Confirm the action does not mutate or redistribute source data.
5. Stop and ask if any rule, data right, or submission action is ambiguous.

## Source-data boundary

- Treat the folder supplied through P1_DATA_DIR as immutable, local-only input.
- Never edit, move, rename, copy into tracked paths, commit, push, upload, or redistribute the source ZIP/CSV/README/score files.
- Never include raw observation rows or values in Markdown, notebook prose, logs, screenshots, issues, or commit messages. Aggregated statistics and cryptographic hashes are allowed.
- Generated models, predictions, submissions, caches, and large extracts belong only in ignored local directories.
- A public download link does not override the source license or competition rules.

## Leakage and validation

- Keep test labels unknown. Do not reconstruct them from KORS/KHOA raw or real-time observations, public mirrors, exact source matching, leaderboards, or hidden-answer artifacts.
- Do not use 2024-2026 external station observations. They overlap the train/test periods and may reveal the pre-injection source signal.
- Do not use random row splits as primary validation. Preserve station/layer groups, chronological blocks, anomaly runs, and real observation gaps.
- Purge or embargo validation boundaries by at least the maximum feature and post-processing dependency.
- Fit scalers, imputers, station/layer baselines, feature statistics, thresholds, and post-processing parameters on the training portion of each fold only.
- Bidirectional offline QC is the approved primary workflow because the task distributes complete time series and states no online-only restriction. Keep every centered feature inside a continuous segment and protect outer folds with a purge longer than its dependency window.
- Maintain a strictly causal mode as an operational ablation. If a later written organizer rule forbids future context, promote the causal result and retire the offline candidate.
- Do not force the test positive rate to match train prevalence or the baseline submission.

## External-data quarantine

- Current policy: no external values until written organizer approval.
- Keep every external-data experiment in an explicitly named, disabled path with source DOI, version, license, retrieval date, checksum, and transformation log.
- Even after approval, use only sources whose license permits the intended use. Competition permission and copyright permission are separate.
- Do not use or redistribute the S-ORS ScienceWatch dataset until its rights holder clarifies the absent per-item open license.
- Never place external raw files in Git.

## Portable and reproducible code

- Do not hard-code a personal absolute path, drive letter, username, or Korean source-directory path in Python or notebooks.
- Resolve P1 inputs from P1_DATA_DIR first; a repository search fallback may be used for local convenience and must fail on zero or multiple matches.
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

One submission opportunity is available per day according to the user-provided platform condition. Never upload without the user's explicit approval for the exact file. Before approval, run:

~~~powershell
.venv-p1\Scripts\python.exe scripts\validate_submission.py <candidate.csv>
~~~

## Official milestone

As checked on 2026-08-13, the latest competition UI shows problem release on 2026-08-13 and submission deadline/final-model selection on 2026-09-07. These current UI dates take operational precedence. The initial KIMST PDF's 2026-08-10 through 2026-09-04 hackathon and 2026-09-04 10:00 ZIP deadline are retained only as superseded schedule evidence. Re-check the UI and obtain written clarification of the exact deadline time before any final action.

## Git safety

- Intended remote: https://github.com/choihyunjin1/-oceanaidata_track1
- Back up code, notebooks, rules, and small aggregate reports only.
- Inspect git status and git diff --cached before every commit.
- Do not commit, push, merge, or submit unless the user or coordinating agent explicitly requests that exact action.
