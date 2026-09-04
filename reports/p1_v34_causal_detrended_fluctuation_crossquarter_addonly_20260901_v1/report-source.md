# P1 v34 causal detrended fluctuation result

## Terminal decision

- Experiment: `p1_v34_causal_detrended_fluctuation_crossquarter_addonly_20260901_v1`
- Decision: `NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED`
- Failed stage: `PRE_Q2_CALIBRATION_GATE`
- Frozen family action: close without alpha, scales, threshold, probe, guard, or budget retuning.
- Fits/runtime: 3 exactly-once fits / 8.64100000000326 seconds.
- Q2/Q3 transport target windows read: 0; Q4 target reads/actions: 0/0.
- Anchor removals: 0. Official/test/sample/submission/hidden, CSV, and upload accesses: 0.

No outer performance surface was opened, so raw outer F1, CI, long-event interior/boundary recall, and station-layer-quarter slices are intentionally unavailable. The canonical decision deltas are 0 nominal points and 0 transport-adjusted points; these are closed-gate bookkeeping values, not measured performance.

## Pre-Q2 gate evidence

| quantile | threshold | proposals | TP | precision | Wilson 90% LCB | gate issue |
|---:|---:|---:|---:|---:|---:|---|
| 0.995 | 0.6668935418 | 281 | 20 | 0.071174 | 0.049824 | supported cells include zero-TP and low-precision environments |
| 0.9975 | 0.6926888227 | 141 | 4 | 0.028369 | one chronological half only; supported cells have zero TP |
| 0.999 | 0.7431868315 | 57 | 0 | 0 | approximately 0 | one station-layer/half only; zero TP |

All three candidates failed the unchanged v28 prospective multi-environment gate, so no threshold was chosen and no target transport window was opened.

## v33 auditability receipt

Before any Q2 target read, the runner sealed label-blind Q2 positions, incumbent predictions, ensemble scores, and fixed-budget action masks for every preregistered threshold. Each mask has 332 add-only actions among 133,170 Q2 rows (share 0.0024930539911391454). The bundle contains no target or metric values and is diagnostic-only; it cannot promote or rescue v34.

- Manifest SHA-256: `752e43942f7bc1e2cb9a1cca0f015f5f3b790913b961d5ba52d1ce132264b8c7`
- Bundle SHA-256: `4d60a778195377c6c6da1274ff79357881ef2816845089cf720c2905ac0a4c7a`
- Model-state hashes: `63c4abce...03ab1`, `3fce95f3...a06d7`, `9f9500c1...eb7865`
- Receipt counters: Q2 target reads before seal 0; Q3/Q4 target reads 0.

## Verification

- Real zero-operation preflight repeated twice: byte-identical, 4,559 bytes, SHA-256 `e19c0cd6a53a2e14eb4b66cd3dbb3a9c6746daa417de594b291c436f503f66fe`.
- Synthetic guards PASS: linear-profile zero fluctuation, injected-fluctuation positive support, cadence-gap reset, station-layer reset, nanosecond cutoff distinction, future invariance, finite 8-feature shape.
- Focused pytest: 6/6 PASS before and after execution.
- Ruff: PASS before and after execution.
- Lifecycle-aware post-terminal QA: PASS, including config/runner/lock/completion/result and v33 manifest/bundle rehashes.
- Result SHA-256: `efd57f065ed09703013655eec1c3c72b0faaa58eea0aecaba463988ca7aa409d`.

The first local verification command accidentally selected `.venv`, which lacks pytest/Ruff. It performed no science, created no lock/artifact, and read no target. All authoritative checks used `.venv-p1`.

## Immutable hashes

- Config: `3feca51dc74b0d8223e33a3156a22ea04bfc847412161227156923c363b67107`
- Runner: `a04bcee6504ee68e1cfc3fc55728ed9d0228e7a5f8589f9b04e8e5be3f30fe47`
- Focused test: `b8f1c24516bc40e0cb376c034b6345ae77d7a236c4a4e92286d424db0dbe60f5`
- Completion: `0515f5b8ec77409ea44fd9c2e168cec70bb95387d64577d4e81a804e49f486c1`
- Lock: `d32b1021bbb71c278a6c4a756ecdba74935ca214e617e20340e2b71fa145c41c`
