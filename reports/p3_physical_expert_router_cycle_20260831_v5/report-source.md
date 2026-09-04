# P3 physical expert router cycle v5

## 결론

- 엄격 내부 PASS: **0/3**
- 제출 CSV: **0개**, upload 0
- v4의 연속 alpha 회귀를 중단하고, 과거 591 past-only feature와 six-lead 궤적으로 champion-vs-base advantage를 분류했다.

| candidate | delta RMSE(m) | blocks | episodes | P(improve) | conditional points C/M/O | PASS |
|---|---:|---:|---:|---:|---:|---|
| P3_1_CATBOOST_SOFT_PHYSICAL_ROUTER | -0.001759 | 4/6 | 0.526 | 0.760 | 24.164908/24.231515/24.297407 | FAIL |
| P3_2_EXTRATREES_HARD_PHYSICAL_ROUTER | -0.004559 | 5/6 | 0.316 | 0.916 | 24.190978/24.275947/24.361092 | FAIL |
| P3_3_LOGISTIC_ABSTAIN_PHYSICAL_ROUTER | -0.004299 | 4/6 | 0.331 | 0.935 | 24.197298/24.271826/24.342800 | FAIL |

## 계약과 QA

- 6 bimonth holdouts, station-local ±78h purge; output target never enters features.
- 133 contiguous historical episodes, 5,000-replicate episode bootstrap.
- Router output is base↔alpha0.425 champion only; 3/6/9/12h exact no-op.
- PASS requires pooled improvement, episode majority, 4/6 blocks, P>=0.8, CI90 upper<0, worst block<=+0.01m.
- official inputs remain unopened unless a strict PASS exists; hidden truth and upload are always zero.
- Conditional score ranges use current Public 0.575233m / 24.203599 points and slope -15.870739 points/m.
- The range is a 1:1 internal-to-official planning translation, not evidence of transport; prior P3 direction reversals remain a required caveat.
