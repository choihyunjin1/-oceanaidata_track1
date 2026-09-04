# P1 v30 causal backward-Teager-energy cross-quarter cycle

## Terminal decision

`NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED`, immutable and no retune. The pre-Q2 calibration gate failed, so Q2/Q3 transport targets and the Q4 performance target remained unopened.

[Kaiser (ICASSP 1990)](https://doi.org/10.1109/ICASSP.1990.115702) motivates a local nonlinear energy operator. v30 used the causal one-sample-delayed form `x[t-1]^2 - x[t-2]x[t]`; the source does not establish oceanographic physical energy, anomaly validity, or P1 performance.

The representation was repository-novel versus linear wavelet modulus/scale energy, state-bin diffusion moments, extrema prominence/return time, recurrence/visibility, and cross-variable signed area. It used a fixed linear probe, not VIB, Barlow, prototype, domain, ranking, or threshold retuning.

## Protocol and gate

- v28 cross-quarter guard SHA-256 `a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6` remained unchanged.
- two real zero-operation preflights were byte-identical: 3,784 bytes, SHA-256 `683ef9a9543f6671f5c222eb4f5ce2f43a3a1e96706a6a505290747ee85de964`.
- prefix-only station-layer normalization, cadence/gap reset, energy sign split and causal 6/24/96-row summaries; fixed three-seed linear probe trained once pre-Q2, 12 epochs, exactly 3 fits.
- one threshold had to be selected pre-Q2 and remain unchanged through Q2 and Q3; no per-window threshold selection or refit.

All pre-Q2 candidates failed station/cell diversity:

| quantile | proposals | precision | Wilson-90 LCB | supported action geometry |
|---:|---:|---:|---:|---|
| `0.995` | `281` | `0.921708` | `0.891147` | S-ORS/L5 only, both halves |
| `0.9975` | `141` | `1.0` | `0.981173` | S-ORS/L5 only, both halves |
| `0.999` | `57` | `1.0` | `0.954685` | S-ORS/L5 only, second half |

This is a valid guard rejection: high pooled precision was not accepted as evidence of cross-station transport. No threshold was selected; Q2/Q3 target reads, Q4 target reads/actions, anchor removals, nominal points, and transport-adjusted points were all zero. Q4 F1, CI, long-event recall, and slices were intentionally not computed.

## QA and hashes

- focused pytest `5/5 PASS`; Ruff `PASS`; post-terminal lifecycle/hash QA all checks `PASS`
- runtime `8.625` seconds; fits `3`
- config `1f81f421382988be783ac26856d90487a7983e0b8a73b24b35e6f087052cc786`
- runner `f8e0879a56801cb279229597606e33ad70730548ea45f4cffdc8e0ad8faab118`
- guard `a051d4d0837b395f9c1d42d71d65572efb99d6c1aeae6aabdb52c8427539c8d6`
- completion `cd152e6e4d059242abe721f8a28f25dcf1c753350467cfde11c798504a400044`
- lock `b20793397ce31434f8e0b1ae55384b2992f2860d7434265f9b2894a05b7b4583`
- result `1c7b930dcd47dac3db605f62b637cc6da5a9e9daf706ceee39e8db2e657e97eb`

Official/test/sample/submission/hidden reads, CSVs, and uploads were zero.
