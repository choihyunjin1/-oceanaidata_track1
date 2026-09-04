# P1 v51 cross-station predictive-causality support audit

Audience: P1 experiment review. Date: 2026-09-01 KST. Scope: original README and train-only `station/layer/time/temp`; label, anomaly type, historical targets, official test/sample/submission, hidden values, and CSV outputs were not read.

## Decision

`NO_GO_ZERO_FIT_CROSS_STATION_PREDICTIVE_INFORMATION_UNSTABLE_AND_REPOSITORY_OVERLAP`. Layer-1 timestamp occupancy is ample, but no station receives stable daily predictive information from the other stations. No runner, READY preflight, lock, supervised fit, target read, action, or upload was created.

The fixed audit compared a station's own lagged layer-1 history with the same model augmented by the other two stations' lagged histories at 10 minutes, 1 hour, and 6 hours. Restricted and unrestricted Ridge models used alpha `10`, fit only through `2025-01-16 17:50 UTC`, and were checked label-free through `2025-03-24 14:50 UTC`. This is a predictive-causality diagnostic, not a structural-causality claim. Granger's original paper motivates restricted-versus-unrestricted prediction, while Diamant et al. motivate cross-sensor QA only when related datasets reflect the same phenomenon; neither paper claims P1 performance ([Granger 1969](https://doi.org/10.2307/1912791); [Diamant et al. 2020](https://doi.org/10.3390/rs12213470)).

## Target-free support

The three layer-1 series have `9,842` exact common timestamps before the cutoff. Pairwise exact overlaps over train are `24,125`, `26,470`, and `39,049`, so row occupancy is not the blocker. Each fixed increment model had roughly `1,830` fit rows and `7,567–7,578` validation rows.

Remote histories slightly reduced pooled increment MSE at I-ORS (`+0.000788` relative) and S-ORS (`+0.002482`) but harmed G-ORS (`-0.004503`). More importantly, positive daily MSE gain occurred on only `23/61`, `29/61`, and `30/61` days; the median daily gain was negative at all three stations. Remote level-history models were substantially worse at all stations (`-0.397039`, `-0.161301`, `-0.343883` relative MSE gain), showing that high seasonal level correlation is not transport-stable incremental information.

The prospective support rule required at least two stations, positive gain in both halves, and a strict majority of positive daily blocks. Zero stations passed the daily stability requirement. Relaxing it to accept the pooled I/S gains would be a result-driven support-gate change and is prohibited.

## Repository negative fingerprint

- Exact P1 Granger code was not found, but the repository already contains the same restricted/unrestricted lagged regression mechanism in P3 v73. Changing variables and decoder does not create a new mathematical information axis.
- An earlier P1 long-event implementation accidentally subtracted contemporaneous values from other stations; independent QA rejected that geographic peer residual because unrelated ocean regimes can contaminate the feature. Lagging the peers changes the estimator but does not establish that the stations are related sensors.
- P1 v9 already closes conditional-dependence-change retuning. A common-mode dynamic factor would repeat the robust-subspace/state-space lineages, and an arbitrary horizontal graph remains unauthorized.

## Terminal accounting and next gap

Fits, optimizer steps, preflights, locks, target reads, actions, removals, official/test/sample/submission/hidden/CSV/upload accesses are all `0`. v28, v33, add-only union, removal `0`, fixed thresholds, and max-nine-fit limits remain unchanged.

The remaining research gap is not another regression lag or threshold. It is a contemporaneous, non-reconstructible physical driver shared across at least two stations—or a genuinely fresh labeled transport window. v50 proved that no such local allowed external driver currently exists, so fabricating station relatedness from seasonal temperature levels is not justified.
