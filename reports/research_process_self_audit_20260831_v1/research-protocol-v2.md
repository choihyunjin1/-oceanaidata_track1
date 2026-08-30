# Research protocol v2 — compact operating contract

## Mandatory inputs

- problem and official metric/unit
- current team champion and immutable hash
- train-only observable inventory and timing contract
- forbidden data and actions
- closed fingerprint ledger
- exposed discovery/selection surfaces
- untouched confirmation blocks
- fit, wall-clock and official-probe budget

## Candidate fingerprint

```json
{
  "problem": "P1|P2|P3",
  "lane": "PERFORMANCE|INFORMATION|INFRASTRUCTURE",
  "mechanism": "",
  "data_axes": [],
  "target_or_residual": "",
  "split_ids": [],
  "feature_family": "",
  "model_family": "",
  "postprocess": "",
  "comparator": "",
  "metric": "",
  "gate": "",
  "confirmation_surface": ""
}
```

## Mandatory arbiter sequence

1. `NO_NEW_EXPERIMENT` option.
2. Exact and semantic duplicate check.
3. Identifiability check.
4. Synthetic/dry-run contract smoke.
5. Smallest falsification screen.
6. Untouched confirmation.
7. Deployment or information-only adjudication.
8. Result, hash and negative-evidence registration.

## Global stop rules

- Same proxy class has two selection-to-confirmation sign reversals: close proxy class for performance selection.
- No untouched confirmation surface: no PERFORMANCE candidate.
- Technical failure: INVALID, never scientific NO_GO.
- Outcome-dependent threshold/window/seed change: new experiment ID and new confirmation budget required.
- External research output overlaps a closed fingerprint or invents score/causal facts: reject before implementation.
