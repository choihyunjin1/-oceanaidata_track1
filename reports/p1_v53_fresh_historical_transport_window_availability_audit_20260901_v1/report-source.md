# P1 v53 fresh historical transport-window availability audit

## Terminal decision

`NO_FRESH_HISTORICAL_TRANSPORT_WINDOW` (0 fit).

This was a metadata/read-only availability audit, not a candidate experiment. The distributed train chronology contains eight KST station-quarter blocks from 2024Q1 through the partial 2025Q4. None is both causally usable and genuinely unexposed. No candidate, feature, model, threshold, or gate was selected; no runner, preflight, lock, fit, action, CSV, or upload was created.

## Chronology and support

Only aggregated `station/layer/time/label` counts were read from `train.csv`; no raw row or value is reproduced here. The 2024 quarters have only one station, so they cannot support the prospective multi-station transport requirement. In addition, 2024Q1 has no earlier distributed training prefix, while the July-August and October-November parts of 2024Q3/Q4 were already inner structure-selection windows. 2025Q1 has three stations but its January-February labels were already an inner selection window and the quarter is the canonical pre-Q2 fit/calibration surface.

The only well-supported multi-station outer windows are 2025Q2, Q3, and Q4. They are not fresh: the canonical learning-curve artifacts contain metrics and sealed prediction parts for all three; the frozen MS-TCN confirmation contract explicitly calls Q3/Q4 globally exposed; and the research ledger calls all Q2/Q3/Q4 hypothesis-exposed. A station, layer, month, or residual-week subset of one of these quarters is not a new independent window.

## v33 registry result

The v33 registry contains 11 complete label-blind Q2 action manifests, each with 133,170 Q2 rows and zero Q2 target reads before sealing. This is useful audit state, not new truth. v35 and v52 opened two distinct fixed action masks, but both used the same already-exposed Q2 labels; they therefore cannot supply an independent transport window or validate the guard globally.

## Required new evidence

The minimum defensible next surface is a newly acquired, competition-allowed labeled chronological block after 2025-12-10 that covers at least two stations and two supported station-layer identities. Its labels must remain sealed until exactly one candidate, numeric threshold, and action budget are preregistered. An organizer-designated authenticated never-opened labeled holdout would also qualify. Without one of these, another local transport adjudication would reuse exposed truth rather than add independent evidence.

## Access and lifecycle

- train-label aggregate audit reads: 1
- candidate-evaluation target reads: 0
- new Q2/Q3/Q4 reads: 0/0/0
- fits / optimizer steps / actions / removals: 0/0/0/0
- official / test / sample-submission / submission / hidden / CSV / upload: all 0
- v52 remains immutable; v28 and v33 are neither relaxed nor reinterpreted
