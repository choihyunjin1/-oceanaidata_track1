# P3 inner-crossfit LCB router cycle v6

## 결론

- governing internal PASS: **0/3**
- 제출 CSV: **0개**, upload 0
- v5 결과를 이용한 threshold 재조정 없이, lead-level expected MSE gain과 train-only inner-crossfit one-sided LCB로 구조를 변경했다.

| candidate | delta RMSE(m) | CI90 high | P(improve) | blocks | episodes | worst block | C/M/O points | PASS |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P3_1_EXTRATREES_LEAD_GAIN_LCB | -0.000376 | +0.007168 | 0.537 | 3/6 | 0.511 | +0.033349 | 24.089836/24.209560/24.328993 | FAIL |
| P3_2_CATBOOST_HUBER_LEAD_GAIN_LCB | -0.001993 | +0.005538 | 0.676 | 4/6 | 0.519 | +0.033349 | 24.115705/24.235236/24.353598 | FAIL |
| P3_3_CONSENSUS_LEAD_GAIN_LCB | -0.001186 | +0.006279 | 0.606 | 3/6 | 0.511 | +0.033349 | 24.103950/24.222417/24.341419 | FAIL |

## 고정 평가 계약

- outer 6 bimonth episode-blocked folds, station-local ±78h purge; every outer fold uses five inner block cross-fits.
- LCB is point gain minus the fixed 80th percentile of signed inner-OOF overprediction residuals; selection boundary is metric-aligned zero gain.
- governing gates: structure/finite/physical validity, pooled RMSE improvement, 133-episode bootstrap CI90 upper < 0.
- block win count, episode win share, and worst block are retained as diagnostics, not promotion gates; no official mixture identifies justified hard thresholds for them.
- 18/24h may select only base or frozen alpha=.425 physical expert; 3/6/9/12h are exact champion no-op.
- official inputs are opened only after internal PASS; hidden truth and upload remain zero.
- C/M/O score ranges conditionally map episode CI and internal delta to the current 0.575233m / 24.203599 champion with slope -15.870739 points/m.
- This 1:1 mapping is a planning range only. Historical P3 local-to-official sign reversals make transport uncertain.
