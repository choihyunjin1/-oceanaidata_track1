# P2 supported-layer change-coherence exploratory cycle 20260901 v11r1

## 결론

상태: `EXPLORATORY_PASS_REQUIRES_FRESH_CONFIRMATION`. pooled ΔRMSE -0.140661374°C, 명목 +1.764953점, transport-adjusted +0.219987점이다.
fold: 2024_sep_oct -0.235940006, 2025_jul_aug -0.002672450, 2025_nov_dec -0.002095438. layer: L2 -0.049517700, L3 -0.056225226, L4 -0.232665485.
month: 2024-09 -0.116624112, 2024-10 -0.310652784, 2025-07 -0.003658091, 2025-08 -0.001334451, 2025-11 -0.001718358, 2025-12 -0.003837216.
active 23,533/69,850행, active-slice ΔRMSE -0.6896864204272495.
세 historical fold는 모두 exposed exploratory surface이며 fresh confirmation이 아니다.

## 구조와 중복 배제

훈련 support가 확인된 L1/L5/L6/L7만 사전 고정했다. 각 층의 exact-10-minute signed change를 training-only median/MAD로 표준화하고, 동시간 cross-layer median에서 한 층만 고립 이탈한 최대값을 coherence score로 썼다.
최소 3개 층이 없거나 score<=6이면 bit-exact champion이다. active 행도 endpoint baseline 대비 champion correction을 최소 50% 보존한다. 행 삭제, fit, learned gate, threshold search, v10 metric tuning은 모두 0이다.

## v10·v7 상태

v10은 L8 training support 0건으로 0-fit technical INVALID이며 같은 ID를 재실행하지 않았다.
v7 ready pack은 미접촉·미업로드 상태를 유지했다: `REPORT_ATTESTED_READY_UNSUBMITTED_HIGH_TRANSPORT_RISK`, report-calibrated +0.013765점. 챔피언 보존이 기본이다.

## 접근 경계

official/test/sample/baseline/score/query support/hidden/submission CSV/upload 접근은 모두 0이다.

## 기술 recovery

v11은 fit/prediction/metric 0에서 read-only ndarray masking으로 종료됐다. v11r1은 `.to_numpy(dtype=float, copy=True)`만 변경했고 candidate, support, fold, Huber, gate, operation counters는 predecessor config hash로 고정했다.
