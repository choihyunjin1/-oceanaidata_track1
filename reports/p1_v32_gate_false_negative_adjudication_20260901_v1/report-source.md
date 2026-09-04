# P1 v32 audit-only false-negative adjudication

## Outcome

`INVALID_TECHNICAL_PRETARGET_STATE_UNAVAILABLE`; zero fits, zero refits, and zero new target reads. Q2, Q3, and Q4 stayed unopened.

Before any Q2 target access, the adjudication candidates were fixed exactly as instructed:

- v29 maximum Wilson-90 LCB: quantile `0.999`, threshold `0.8519394397735596`, pre-Q2 count `57`, precision `0.947368`, LCB `0.875420`.
- v30 maximum Wilson-90 LCB: quantile `0.9975`, threshold `0.7439806461334229`, pre-Q2 count `141`, precision `1.0`, LCB `0.981173`.

Artifact inventory then established that neither terminal namespace contains a serialized ensemble, a label-blind Q2 score vector, nor a Q2 action mask. Each contains only `preflight.json`, `pre_q2_candidate_threshold_seal.json`, `predictions_complete.json`, and `result.json`. A threshold and model hashes are not sufficient to reconstruct row actions. Recreating the ensemble would be a refit, expressly forbidden by the audit contract.

Accordingly, Q2 labels were not opened and no confusion, Q2 precision, or F1 delta was fabricated. The original v29/v30 conclusions remain unchanged. The v31 confusion matrix also remains non-identifiable.

## Prospective recommendation

Keep v28 unchanged because this audit supplies no new outcome evidence. For future rejected candidates, seal either the full deterministic ensemble state or, preferably, label-blind transport-window score vectors and fixed-threshold action masks before any target access. Then a separately authorized audit can open one target window with zero fits/refits while remaining incapable of changing the original decision.

Official/test/sample/submission/hidden reads, CSVs, uploads, new labels, and Q2/Q3/Q4 target reads were all zero.
