# P3 parallel physical-axis candidate cycle v4

## 결론

- 엄격 내부 PASS: **0/3**
- 제출 CSV 생성: **0개** (업로드 0)
- direct residual 및 단순 alpha sweep를 반복하지 않고, KMA 물리축 내부의 train-only 상태 게이트만 평가했다.

## 후보 결과

| candidate | delta RMSE(m) | improved blocks | improved episodes | P(improve) | CI90 high | worst block | PASS |
|---|---:|---:|---:|---:|---:|---:|---|
| P3_1_PHYSICAL_STATE_GBDT_GATE | +0.000984 | 3/6 | 0.481 | 0.355 | +0.005263 | +0.018010 | FAIL |
| P3_2_PHYSICAL_STATE_HUBER_AXIS | +0.002157 | 2/6 | 0.429 | 0.166 | +0.005845 | +0.019286 | FAIL |
| P3_3_PHYSICAL_STATE_RIDGE_AXIS | +0.000118 | 2/6 | 0.496 | 0.485 | +0.003593 | +0.016520 | FAIL |

## 검증 계약

- 6개 bimonth holdout, 동일 station ±78시간 purge
- 독립 단위: station별 78시간 초과 gap으로 분리한 historical episode
- PASS: pooled 개선, episode 과반 개선, 4/6 block 개선, bootstrap P>=0.8, CI90 upper<0, worst block degradation<=0.01m
- short leads 3/6/9/12h exact no-op, long leads는 alpha [0,0.65] 물리축 제약
- hidden truth 0행, upload 0회

## 데이터 품질

- historical rows/cases/episodes: 1092/182/133
- duplicate pair keys 0, non-finite 0, every case six leads intact
- official inputs were not opened until after all internal gates were finalized.
