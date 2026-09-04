# P1 v24 prospective selection-contract audit

## Decision

v24 is justified as a distinct prospective selector experiment, not a reopening of v16. Exact v16 at fixed `p=.95` remains closed with pooled delta F1 `+0.000392906`, raw `+0.010442710` points, and no Q4 additions; it is not reclassified under calibration v3.

v24 preserves the GCE score model exactly: q=.7, L2=.001, the frozen 165-column projection, the same leverage weights, and two forward pipeline fits. Its only new question is whether an inner chronological tail can select a useful add-only threshold. The first 75% of each training prefix fits the GCE model; the last 25% selects among all finite probabilities by actual anchor-union F1, requiring positive inner delta, central marginal precision above anchor F1/2, positive additions, and changed share at most 0.5%. Ties choose the higher threshold then fewer additions.

This is not a fresh confirmation. Inner threshold search has selection variance, and Q3/Q4 are already exposed by prior experiments. Any future run is adaptive development evidence only. Nevertheless, nested separation makes the selector a coherent new contract rather than a post-hoc threshold scan on outer results.

Outer gates remain dependent bootstrap and slice safeguards with prospective calibration v3: raw points at least `0.015383691` and calibrated points at least `0.01`. Historical execution, official rows, hidden labels, locks, CSV, and uploads remain zero at preseal.
