# P3 joint six-lead reduced-rank residual audit — v22

## 결론

`STOP_SEMANTIC_DUPLICATE`이다. 데이터 부족으로 멈춘 것이 아니다. historical validation surface에는 182개 완전 사례와 사례당 정확히 `3/6/9/12/18/24h` 여섯 리드가 있으며, 1,092행 모두 uniform KMA alpha `0.425` reference와 target이 유한하다. 그러나 제안된 rank-1/rank-2 robust reduced-rank regression은 새로운 정보축이 아니라, 이미 실행된 causal joint-six-output residual regression의 용량 제약 ablation이다.

따라서 config/runner/candidate를 만들지 않았고 fit, outer score, CSV, upload는 모두 0이다. 후보 RMSE와 예상 공식점수를 적는 것은 방어할 수 없다. 유일하게 확정된 숫자는 exact no-op인 현 Public champion RMSE `0.575233m`, `24.203599점`이며, 이는 실행하지 않은 reduced-rank 후보의 예상치가 아니다.

## Deep Research 판단

[Izenman의 고전 reduced-rank regression](https://doi.org/10.1016/0047-259X(75)90042-1)은 다변량 선형회귀 계수행렬의 rank를 제한하는 모델이다. [Velu, Reinsel, Wichern의 multiple-time-series reduced-rank model](https://doi.org/10.1093/biomet/73.1.105)도 여러 시계열의 coefficient structure에 같은 저차원 제약을 둔다. [Ben Taieb, Sorjamaa, Bontempi의 multiple-output forecasting](https://doi.org/10.1016/j.neucom.2009.11.030)은 여러 horizon을 공동 출력해 horizon 간 의존성을 보존하는 목적을 설명한다. 이 문헌들은 proposed mechanism이 통계적으로 정당함은 뒷받침하지만, 현재 저장소에서 새롭다는 증거는 아니다.

저장소에는 이미 다음 의미상 동일한 핵심이 있다.

1. `p3_nlinear_station_ridge_residual_20260828_v1`은 336개 past-only multiresolution features를 station별 train-fold 표준화한 뒤 6개 미래 Hs delta를 한 번에 예측하는 multi-output Ridge이다. 78시간 gap과 chronological outer validation을 사용했고, terminal delta는 incumbent 대비 `+0.005902259m`였다.
2. `src/p3_wave/causal_spectral_kernel.py`는 past-only causal features를 train-only median/IQR scaling한 뒤, closed-form multi-output Ridge coefficient matrix로 6-le드 residual을 공동 예측한다. proposed v22와 입력표현은 다르지만 핵심 statistical map `X -> Y_(six leads)`와 residual objective가 같다.
3. `src/p3_wave/timexer_direct_multilead.py` 역시 station-aware 48시간 causal encoder에서 joint six-output residual head를 사용했으며, corresponding execution은 incumbent 대비 `+0.098487624m` 악화로 종료됐다.

rank 1/2 truncation, fixed winsor/Huber, comparator를 uniform alpha `0.425`로 바꾸는 것은 exact implementation 차이는 만든다. 그러나 새 신호를 추가하거나 다른 deployment contract를 만들지 않는다. [Huber의 robust location framework](https://doi.org/10.1214/aoms/1177703732)은 contamination에 대한 bounded-influence 설계를 정당화하지만, 동일 모델 계열을 새로운 정보축으로 바꾸지는 않는다. [Argyriou, Evgeniou, Pontil의 multi-task feature learning](https://home.ttic.edu/~argyriou/papers/mtl_feat.pdf)도 related tasks의 shared low-dimensional representation을 설명하지만 동일한 novelty 한계가 있다.

## 데이터 계약 감사

공식 입력을 열지 않고 기존 historical loader만 read-only 실행했다.

| Check | Result |
|---|---:|
| historical rows | 1,092 |
| complete case blocks | 182 |
| exact lead tuple | `(3, 6, 9, 12, 18, 24)` for every case |
| duplicate pair keys | 0 |
| nonfinite target/reference | 0 |
| bimonth block cases | 40 / 31 / 22 / 26 / 23 / 40 |
| official/hidden access | 0 |

즉 `STOP_DUPLICATE_OR_NO_DATA`의 원인은 `DUPLICATE`, `NO_DATA`가 아니다.

## 다음 신규 정보축 제안

다음 감사 대상은 `causal continuous-path interaction encoder`이다. 48시간의 multivariate observations와 structural masks를 하나의 시간경로로 보고, order-2 path-signature 또는 Neural-CDE류의 bounded representation으로 순서가 있는 cross-channel interaction을 표현한다. 이 방향은 alpha micro-grid, coefficient-rank ablation, per-row tree/Ridge/Huber, analog, Fourier amplitude와는 다를 가능성이 있다.

다만 이 제안도 즉시 실행할 수 있는 승인은 아니다. TCN, TimeXer, masked SSL, state-space와 exact/semantic duplication을 먼저 감사하고, 통과할 때만 architecture 1개를 사전등록해야 한다. exposed 182-case surface의 결과는 계속 `EXPLORATORY_ONLY`이며 Public transport 보장은 없다.

## Access and execution receipt

- historical target rows read: `1,092`
- official test/sample/submission/hidden rows: `0`
- model fits / predictions / outer scores: `0 / 0 / 0`
- CSV materializations / uploads: `0 / 0`
- result-based tuning: `false`
