# P1 original source-only reconstruction

## Decision and fixed contract

Recover the exact historical recipe, not a new score-improvement search. Previous saved-model replay matched the historical 28.909341-point answer; this does not prove fresh training reproduces it.

- Package: `C:/Users/cedis/Documents/OceanFinalCandidates_20260906/P1_original_source_v1`.
- Fresh empty `03_model`; original XGBoost O (1 fit), event/day-weighted LightGBM B (3 fits), MS-TCN original e150 (3 fits): 7 fits total.
- GPU0 exclusively allocated to this attempt; tree CPU8, MS CPU2. Whole execution cap 21,600 seconds. No automatic restart or parameter changes.
- Comparator: historical answer SHA256 `57844ef235f987059d17bddb035cc0a98b92bb0d53263cf878c68b31bbc53687`.
- Primary reconstruction metric: exact answer SHA; additionally fresh-process tree/MS numerical replay, schema/key/order/row-count validation and runtime. No new OOF quality or official score claim.
- If exact: report source-to-answer recovery, separately disclose portability/eligibility limitations. If different: preserve artifacts, locate component mismatch without changing parameters based on the historical answer.
- No archived weights, caches, predictions, OOF files, fixed row patches or hidden truth are training/inference inputs. Historical answer hash is only an after-generation comparator. No uploads, commits or pushes.
- Historical train-OOF-selected cell policy is preserved as a declared fixed recipe. The original selector program is not recovered or rerun. Original offline whole-series covariate features are preserved; learned encoder/model fitting is TRAIN-only. This is not organizer acceptance certification.

## Preflight evidence

- Original raw features independently rebuilt: train 776,706 x 80 and test 169,011 x 80, values and parquet hashes exact. Canonical evidence: `artifacts/p1_original_feature_check_20260906_v1/result.json`. These QA caches are not inputs to this cold run.
- New package/historical composition focused pytest: 14 PASS. Ruff PASS. The source manifest pins 31 files, with no supplied models/caches/data.
- Notebook schema validated; top-to-bottom execution remains pending. CLI and notebook use the same full-workflow entrypoint; never execute both concurrently.
- At 22:55 KST, no Python training processes were present; GPU0 was available. Prior MS runtime approximately 110 minutes, so same-day midnight completion is not promised.

## Evidence ownership

Package `ATTEMPT_LOCK.json`, `04_logs/`, `03_model/tree/training-result.json`, `03_model/mstcn/`, `06_docs/P1_submission.csv.json`, and `terminal.json` are authoritative runtime evidence. Preserve all failures. This report must not label a running attempt complete.
