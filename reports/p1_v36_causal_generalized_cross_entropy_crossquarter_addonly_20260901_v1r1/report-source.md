# P1 v36r1 GCE provenance recovery result

## Terminal decision

- Experiment: `p1_v36_causal_generalized_cross_entropy_crossquarter_addonly_20260901_v1r1`.
- Decision: `NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED` at `PRE_Q2_CALIBRATION_GATE`.
- Frozen science: q=0.7 generalized cross entropy, the same causal 8-feature temperature state, 8-unit tanh head, three seeds, thresholds, budgets, and v28/v33 guards as v36.
- Fits/runtime: 3 exactly-once fits / 9.047000000005937 seconds.
- Q2/Q3 transport target windows read: 0; Q4 target reads/actions: 0/0; anchor removals: 0.
- Official/test/sample/submission/hidden, CSV, and upload accesses: 0.

No outer surface opened; raw outer F1, CI, long-event recalls, and slices are unavailable by contract. Nominal and transport-adjusted points are both zero closed-gate bookkeeping values.

## Gate evidence

| quantile | threshold | proposals | TP | precision | Wilson 90% LCB | failure |
|---:|---:|---:|---:|---:|---:|---|
| .995 | .8245659471 | 281 | 243 | .864769 | .827714 | one supported G-ORS/L1 cell has 0/25 TP |
| .9975 | .8754034638 | 141 | 135 | .957447 | .919830 | actions collapse to one station-layer identity |
| .999 | .8958478570 | 57 | 57 | 1.0 | .954685 | one station-layer identity and one half only |

No candidate satisfied all unchanged multi-station/cell/half conditions. The family closes without q, model, threshold, or gate retuning.

## Provenance recovery

The invalid v36 artifact, metrics, and bundle were never read. v36r1 freshly regenerated all state in a new namespace and records three separate code identities:

- Wrapper: `97669106f96cb164b1dc76fccfaf03b047efd796773eaf49e87d5084ad88c515`.
- Frozen GCE science module: `1f11885d944131761f0efe6a66449e31f39c0b6ba7a4b0eb90ecbef626b369ee`.
- Shared v34 execution engine: `a04bcee6504ee68e1cfc3fc55728ed9d0228e7a5f8589f9b04e8e5be3f30fe47`.

The wrapper hash equals both preflight `runner_sha256` and terminal result `runner`/`wrapper`. Post-terminal QA independently rehashed all three.

## v33 receipt and verification

- Fresh Q2 label-blind bundle: all three thresholds, 332 actions each among 133,170 rows, three model-state hashes, target reads before seal 0.
- Bundle SHA-256: `9f42ba3651015ec8c78852d01f6a4a2b3a64edd8599292d6c1607d14a27102dc`.
- Manifest SHA-256: `f4c442efed9b04f2dee01ba7ea68ae8b19ce5d45b3ea12930b2df70740f3bdb9`.
- Two preflights: byte-identical, 4,366 bytes, SHA-256 `280ab36f2c865bfb46f242884280d030f1c5953f557ce0969644b89e8bfffc47`.
- Focused pytest: 5/5 before and after; Ruff PASS; lifecycle/provenance QA PASS.
- Result SHA-256: `bbb5231c0354acbe0e852722ea34bb913e837ca99993597c3736e61f4b7087ca`.
