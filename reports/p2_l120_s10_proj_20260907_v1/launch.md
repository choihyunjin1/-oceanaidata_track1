# P2 L120 ten-seed plus projection — active execution

2026-09-07 11:28:10 KST, worker PID 11152 (Windows venv launcher 2408) started actual new GPU training. This is not a plan-only update. First order is B3; remaining folds B1/B2/B4/B5/B6/B7/B8 and FULL follow the sealed config. No early performance-based pruning or coefficient changes.

- New production fits: 56 internal (7 seeds × 8 unchanged folds) + 7 full-data = 63. Exact-hash reuse: 24 internal + 3 full-data models. Seeds are 20260901–20260910 with equal mean; only seeds 4–10 need new fits for this first completion.
- Additional portable fresh cold: 10 full-data fits, not a second repeat of internal cross-validation. Total new production fits through packaging: 73. First completion is expressly not a ten-fit fresh cold.
- GPU0 is assigned to P2. CPU2 preserves the prior numerical recipe and reduces simultaneous CPU contention with P1 and fixed CPU4 P3. No wall-time kill switch. P3 is untouched.
- Comparator: existing L120 three-seed predictions, regenerated directly from their fitted fold models and compared on identical keys. The scored official CSV is not an ingredient in ten-seed mean predictions.
- Primary: B3 projected RMSE in degrees Celsius. Secondary: all8 pooled RMSE, per-block/per-layer, supported T5 outage whole-fold and interval. Previously exposed evaluation is retrospective. Seven-day block bootstrap seed20260907/resamples2000; adverse block risk is separate, not an automatic veto.
- Projection is unchanged: complete layers2/3/4 only; finite T1 and first finite T5→T6→T7→T8; clip then exact endpoint-direction PAVA. Missing endpoints or incomplete profiles unchanged. No data deletion/target leakage.
- Sealed config and immutable input pins: `seal.json`. Training counters: `artifacts/p2_l120_s10_proj_20260907_v1/progress.json`. Model and prediction artifacts remain local-only.

The separate finishing supervisor started 11:35:20 KST (child PID632, launcher5640). It waits for successful terminal training, then performs whole-model/OOF independent replay QA, builds a three-reused-plus-seven-new completion package, runs complete source-only ten-fit cold, compares candidate SHA, and actually extracts/replays the saved-model ZIP. It will stop on a technical failure without overwriting or restarting the consumed attempt. Candidate and final receipts are `candidate-ready.json`, `independent-qa.json`, `result.json`; failure is `finish-failure.json`.

Local output target: `C:\Users\cedis\Documents\OceanFinalDay_20260907\P2_L120_s10_proj_v1`. Empty source package has a real training path; saved models are separate from answers. Thin notebooks invoke reviewed code and must execute successfully before notebook/package completion can be claimed.

Preflight: 4 focused synthetic tests PASS. Ruff PASS for new QA/portable/finish/tests. Active training runner has only B023 excluded: synchronous progress callback is invoked before advancing its enclosing loop, so no asynchronously escaping late-binding callback. No training/model code is changed after sealing.

Official score, final eligibility, general-model six-hour applicability and organizer deadline remain unverified. This task does not upload, delete, finalize, commit or push. Hidden truth and external observations are prohibited.
