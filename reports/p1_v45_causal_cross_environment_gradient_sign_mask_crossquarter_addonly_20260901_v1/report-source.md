# P1 v45 causal cross-environment gradient-sign AND-mask

## Terminal decision

`NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED` at `PRE_Q2_CALIBRATION_GATE`. This is a valid exactly-once negative, not a technical failure. The fixed family is frozen without threshold, agreement, environment, seed, or optimizer retuning.

## Distinct science and target-free support

Repository-wide search found no executed AND-mask, hard-to-vary gradient-sign agreement, or environment-coordinate sign-consensus recipe in P1/P2/P3. P2 PCGrad projects pairwise conflicting task vectors; v45 instead computed all source-environment gradients at the same parameters and zeroed individual parameter coordinates unless all five signs agreed. It used no projection, domain adversary, gradient-variance matching, IRM/V-REx/Group-DRO penalty, loss reweighting, partial pooling, density ratio, router, or threshold change. Parascandolo et al., *Learning Explanations that are Hard to Vary* (ICLR 2021, https://openreview.net/forum?id=hb1sDDSLbV), motivates the mechanism only and supplies no P1 performance claim.

The v45 environment contract was independently sealed from label-free train occupancy rather than by weakening v44. A station x quarter environment was supported only with at least 4,096 rows and two distinct layers. This yielded five environments, two stations, all four quarters, minimum 4,536 rows, and minimum two layers. Each epoch used 16 nonoverlapping 256-row batches per environment. Complete sign unanimity across five environments has prospective symmetric-independent-null probability `2^(1-5)=0.0625`. Environment bits were training metadata only and never model inputs at inference.

Two zero-operation preflights were byte-identical: 5,705 bytes, SHA-256 `c3863e463a3a81a31e69c1558588ee828d2b77dc5b71937372b14811a801660c`; labels/targets read were zero and no lock/artifact existed. Focused pytest passed 6/6 and Ruff passed before execution.

## Exactly-once result

Three fixed seeds completed 192 optimizer steps each in 8.781 seconds. Mean retained parameter-coordinate shares were 0.211291, 0.184349, and 0.217207. The highest pooled precision was q=.999: 46/57 TP, precision 0.807018, Wilson90 LCB 0.707953, but supported action existed only in the second half of S-ORS/L5, so the unchanged v28 station/cell/half diversity gate rejected it. q=.9975 had 96/141 TP, precision 0.680851, LCB 0.613395 and three stations, but both supported G-ORS/L1 halves had zero TP. q=.995 had 162/281 TP, precision 0.576512 and LCB 0.527525, below the fixed 0.55 LCB floor and with the same G-ORS zero-TP failure.

No threshold was chosen. Q2/Q3 transport truth and Q4 truth remained unopened; raw F1 delta, nominal points, and transport-adjusted points are therefore exactly zero by no-action convention. Anchor removals and Q4 actions were zero. The v33 auditability bundle preserved all three label-blind Q2 action vectors before any Q2 target read: counts 332, 332, and 121; bundle SHA-256 `feb436b453395270ab735c2f71498a9652f77b2d7773725d64ee91dbe65f896a`.

Post-terminal lifecycle QA passed 20/20, focused pytest again passed 6/6, and Ruff passed. Official, test, sample submission, hidden, CSV, upload, and submission access remained zero.

## Next distinct axis

The next axis to fingerprint is fixed causal last-layer spectral decoupling: penalize squared logits to reduce domination by source-specific easy correlations while leaving station/quarter metadata entirely outside the model. It must first be distinguished from ordinary weight decay, focal/GCE/R-Drop, confidence calibration, VIB, and prior domain objectives, and must receive a label-free representation/support preflight before any seal or fit. This report does not authorize it.
