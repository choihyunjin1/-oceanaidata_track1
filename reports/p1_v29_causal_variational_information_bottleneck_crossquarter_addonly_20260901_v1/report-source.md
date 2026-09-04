# P1 v29 causal variational-information-bottleneck cross-quarter cycle

## Terminal decision

`NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED`, immutable and no retune. The cycle stopped at the pre-Q2 calibration gate: Q2 and Q3 transport targets were never read, and the Q4 performance surface was never opened.

[Alemi et al. (ICLR 2017)](https://research.google/pubs/deep-variational-information-bottleneck/) motivate a supervised stochastic latent trained with predictive loss plus KL compression. This supports the objective mechanism only; it is not evidence for P1 accuracy or transport.

Repository fingerprinting found no executed P1 information-bottleneck or stochastic Gaussian bottleneck classifier. v29 reconstructed no inputs, learned no prototype, used no contrastive views, domain adversary, pairwise ranking, Dirichlet evidence, uncertainty threshold, or score recalibration. Its exact objective and coefficients are now closed.

## Cross-quarter protocol

The separately sealed v28 guard, SHA-256 `a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6`, fixed one ensemble and one threshold before Q2. The same threshold had to pass Q2 and Q3 independently before Q4 could open; Q2/Q3 threshold selection and refits were zero.

- two real zero-operation preflights were byte-identical: 3,683 bytes, SHA-256 `e665abcf91d79147e45d0e08abed15b6721b8c74c77b56f64359715cf65a5f2c`.
- fixed causal 8-feature state, 12-unit hidden layer, 4-unit stochastic Gaussian bottleneck, KL coefficient `0.001`, 12 epochs, and three fixed seeds trained once before Q2; 3 fits total, below the maximum 9.
- fixed posterior-mean inference and quantiles `0.995/0.9975/0.999`; no sweep, retry, per-window selection, or outer tuning.

All pre-Q2 candidates failed the multi-environment gate despite high pooled precision:

| quantile | proposals | precision | Wilson-90 LCB | failure |
|---:|---:|---:|---:|---|
| `0.995` | `281` | `0.846975` | `0.808354` | G-ORS/L1 half 0 had 0/27 TP |
| `0.9975` | `141` | `0.914894` | `0.868006` | supported proposals collapsed to S-ORS/L5 only |
| `0.999` | `57` | `0.947368` | `0.875420` | only one station/cell and one half supported |

No numeric threshold was therefore eligible. The immutable closure recorded zero transport-window target reads, zero Q4 target reads, zero Q4 actions, zero anchor removals, and nominal/transport-adjusted points `0.0/0.0`. Q4 F1, CI, long-event recall, and slices were intentionally not computed because opening that target after a failed gate would violate the prospective contract.

## QA and hashes

- focused pytest `6/6 PASS`; Ruff `PASS`; post-terminal lifecycle/hash QA all checks `PASS`
- runtime `9.218999999997322` seconds; fits `3`
- config `4119bff05ebe644634ba1eb5af493e8764562b5b13366a8858643c4f3129b538`
- runner `e1fa64cc4e5f1345e2cea416cb466c60863c76b8bb1c57cbde0def724be042af`
- guard `a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6`
- completion `44acfe267333a4c506793c3657d28ba91151342acbbfe60063bac4b0c86d42af`
- lock `27bbadaf4d0372d5f30d71223126560da5598da908622b9c2018a65498cb673c`
- result `270fb5a19f21322a4c5877b26756a6cf95bb328a883809ca17111cdf6c60a8d2`

Official/test/sample/submission/hidden reads, CSVs, and uploads were zero.
