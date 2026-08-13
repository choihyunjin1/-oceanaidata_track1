# Project operating instructions

These instructions apply to every task in this repository.

## Mandatory preflight

Before reading data, changing code, training a model, creating a submission, or using Git:

1. Read `00_MUST_READ_FIRST.md` completely.
2. Read the source dataset `README.md` at the exact path recorded there.
3. Re-read the latest organizer problem statement and rules supplied by the user. If it conflicts with local notes, stop and resolve the conflict before proceeding.
4. Preserve the original ZIP and extracted source files byte-for-byte.

## Data handling

- Treat `데이터셋 원본/` as immutable, local-only input.
- Never commit, push, upload, copy into tracked folders, or redistribute the source data. The supplied README explicitly prohibits redistribution.
- Put code and documentation in tracked folders; put generated models, predictions, and submissions in ignored local folders unless the user explicitly selects a safe artifact to track.

## Validation and submission safety

- Use chronological or blocked/grouped validation. Never rely on a random row split for this 10-minute time-series anomaly task.
- Record the seed, split definition, feature version, score, output path, Git commit, and file SHA-256 for every candidate considered for submission.
- There is one submission opportunity per day. Never upload a competition submission without the user's explicit approval for the exact file.
- Before approval, run `python scripts/validate_submission.py <candidate.csv>` and record its exact schema/order, row count, key equality/uniqueness, finite binary labels, and prediction file hash results.
- Keep test labels unknown. Do not infer or use prohibited information, external labels, or leakage from future/adjacent validation rows.

## Git backup

- Intended remote: `https://github.com/choihyunjin1/-oceanaidata_track1`
- Back up code, notebooks, rules, and small reports only.
- Inspect `git status` before every commit and confirm no source data or generated submission is staged.
