# P1 v47r1 causal hidden MixStyle

Status: `NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED`, failed at the prospective pre-Q2 calibration gate. This is a valid, immutable one-shot negative; the MixStyle family is frozen with no retry or retuning.

v47r1 was a science-neutral recovery of v47. The representation, p=0.5, Beta(0.1,0.1), features, three seeds, 12 epochs, thresholds, anchor union, v28 transport veto, and v33 auditability contract were unchanged. The only repair was reading `(4096 rows, 2 layers)` from `representation_support_gate`, rather than nonexistent model keys. The old v47 namespace remained consumed and was never rerun.

Two target-free READY preflights were byte-identical at 6,176 bytes, SHA-256 `2bddc63d1d9efca88ab279883397b441c274c079e7557ccdeeb389491e2a3650`. Support comprised five donor environments across two stations and four quarters. Readiness target-column access was zero.

Exactly three fits completed in 9.375 seconds, 144 optimizer steps each. Mixed steps were 85, 61, and 67; mixed rows were 344,380, 245,320, and 272,164. All donor pairs crossed station-quarter environments, labels were never mixed, and empirical mean lambda stayed near 0.5 for all seeds.

No threshold passed the amended environment guard:

- q=.995: 245/281, precision 0.871886, Wilson-90 LCB 0.835510; failed because supported G-ORS/L1 half-0 had 0/25 true positives.
- q=.9975: 129/141, precision 0.914894, LCB 0.868006; only one station-layer identity and one station supported.
- q=.999: 55/57, precision 0.964912, LCB 0.899370; only half-1 and one station-layer identity supported.

Thus Q2/Q3/Q4 target windows remained unopened, promoted actions=0, anchor removals=0, performance F1 was not evaluated, and canonical nominal plus transport-adjusted point deltas are both 0. The v33 bundle preserved label-blind Q2 action counts 332/299/84 for q=.995/.9975/.999 without target access or promotion. Official, hidden, test, sample-submission, submission, CSV materialization, and upload accesses were all zero.

## Frozen hashes

- result: `7e00bc9d4847c39760eac51d0607c464da74e272d9e1b19d8cfbc2b7afed5263`
- config: `3f0fe08e9b360d9c90ae40f6ded3fc12d1b182bf998246c6ed4e1ec60d604444`
- runner: `0e98bd60d3e409e7d4d9bf82da6c1008a071ef24e3846ad73fca924ce80bdd17`
- lock: `a4e96c0ef7b3c58324f752d22954c0dff7abf8fef767ed285f297e55cc838279`
- completion: `6476df96d35a56c1eb6651c56a0c21f652d3f013f9f91f4a5f00542ecb085a67`
- label-blind Q2 bundle: `42b16a8fed237a258819c1822664dbb5df13ada0e5fc7a3ec7004a595b587673`
- bundle manifest: `8462684d8e9a6fe539e33afc414da96d8bb60d8e50cc1918e644a3582644233d`
