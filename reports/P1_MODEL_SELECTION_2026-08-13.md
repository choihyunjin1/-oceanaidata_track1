# P1 model-selection record — 2026-08-13 KST

## Decision

The first local candidate is the non-augmented, bidirectional offline XGBoost
model. It passed the predeclared rolling-origin, bootstrap, and station-level
promotion checks against the prior LightGBM candidate. This is a local estimate,
not an official or hidden-test score. No competition upload was made.

## Validation contract

- Metric: row-level binary F1; weighted F1 reweights station-layer rows to test support
- Folds: 2025 Q2, Q3, and 2025-10-01 through 2025-12-10
- Purge: 7 days; positive events and observed gaps remain intact
- Model and post-processing selection: past-only inner blocked validation
- Reported outer predictions: each outer holdout is evaluated once with its inner selection
- External observation values: none
- Official baseline reference `0.548255`: recorded only; it is not directly comparable
  because hidden leaderboard labels and split weights are unavailable locally

## Honest outer-fold comparison

| Candidate | Mode | Augmentation | Micro F1 | Test-share weighted F1 | Decision |
|---|---|---:|---:|---:|---|
| LightGBM | offline | no | 0.816737 | 0.768804 | superseded |
| LightGBM | causal | no | 0.757248 | 0.703250 | operational ablation |
| LightGBM | offline | yes | 0.609332 | 0.553548 | rejected |
| CatBoost | offline | no | 0.806831 | 0.757848 | rejected |
| XGBoost | offline | no | **0.860371** | **0.813316** | selected |

The synthetic augmentation raised some type recalls but caused many false positives,
especially in Q2. It is retained as an optional experiment and excluded from the
candidate.

### Selected XGBoost by fold

| Fold | Weighted F1 | Plateau-only weighted F1 | Improvement |
|---|---:|---:|---:|
| 2025 Q2 | 0.747823 | 0.193510 | +0.554313 |
| 2025 Q3 | 0.878580 | 0.509990 | +0.368590 |
| 2025 Q4 | 0.883290 | 0.193168 | +0.690121 |

The independent OOF recount was TP 12,946, FP 1,093, and FN 3,109, reproducing
micro F1 `0.8603708380`. The paired event/day-block bootstrap for XGBoost minus
LightGBM had a 90% interval of `[+0.019929, +0.070360]` with improvement
probability `0.9985`. Station-level F1 changes were positive for all stations:
G-ORS `+0.07225`, I-ORS `+0.03557`, and S-ORS `+0.04633`.

### Selected XGBoost type recall

| Type | Recall |
|---|---:|
| spike | 0.8197 |
| noise | 0.9355 |
| flatline | 0.9997 |
| offset | 0.6492 |
| drift | 0.6462 |

The weakest station-layer groups are I-ORS layer 1 (F1 0.4328) and S-ORS layer 2
(F1 0.5064); these are explicit targets for the next feature and sequence-model
iteration.

## Stress tests using the fixed XGBoost selection

| Holdout | Micro F1 | Weighted F1 | Important limitation |
|---|---:|---:|---|
| S-ORS 2024 to 2025 H1 | 0.633619 | 0.656219 | drift recall 0.1655 |
| all non-G stations to G-ORS | 0.762181 | 0.762181 | drift recall 0; all G categories unseen |

The G-ORS holdout produced zero false positives, but the absence of detected drift
means this result must not be read as comprehensive transfer robustness.

## Frozen candidate

- Backend: XGBoost CPU histogram
- Trees: 700
- Offline post-processing: high threshold 0.20, low threshold 0.10,
  gap close 0 rows, minimum continuous run 12 rows
- Hard rule: full plateau length at least 6 rows; singleton spike candidates are preserved
- Model SHA-256: `d4b60c543691f51526b3909101763309a6560cc274386a8fb3323773ad469d12`
- Candidate logical path: `submissions/20260813T155449+0900_predict_378a4e89.csv`
- Candidate rows: 169,011
- Predicted positives: 6,504 (3.84827%; diagnostic only, never forced)
- Candidate SHA-256: `28243fda9bc56e25a698366823dfab3198cda21bfaec04f30fda6a899eaf0cd3`
- Strict validator: passed
- Saved-model reproduction: row-identical and SHA-identical
- Uploaded: **no**

The CSV and model remain ignored local artifacts. They must not be committed, placed
in the public code archive, or uploaded until the user approves that exact file.

## Deep and SSL status

The fold-local TCN, patch Transformer, masked-reconstruction SSL, 12-setting search
spaces, three-seed finalist retraining, checkpoint provenance, and leakage tests are
implemented. CUDA forward/backward and bounded end-to-end smoke experiments pass.
Their full multi-hour screen is not used in this first candidate and cannot be claimed
as a measured improvement until the fixed outer evaluation completes.
