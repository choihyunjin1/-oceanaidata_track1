# P2 OAS, rank-1, and low-rank residual models

- card id: `p2-oas-rank1-lowrank`
- problem: `P2`
- mechanism: OAS anchor calibration, seasonal/layer rank-1 residuals, and nested low-rank capacity.
- covered historical families: 5
- covered later key cases: 3
- primary status counts: `CLOSED_EXACT` 1, `DISCOVERY_ONLY` 3, `INFORMATION_POSITIVE` 4

## 재사용할 것

- Reuse the OAS champion lineage, cross-fit rank-1 factorization, and bin-level factor isolation.
- Reuse nested selection so capacity is chosen inside each outer split.
- Keep bin17 as positive official directional evidence; do not generalize to adjacent bins.

## 그대로 반복하지 않을 것

- Universal density penalty and already tested full nested-PLS grid.
- Pooling bin17 and bin18 merely because they are adjacent.

## 재개 조건

A new frozen rank/bin factor with positive dependent-block evidence or a new outer surface.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P2-F00` | `DISCOVERY_ONLY` | - | router 400 rounds 0.788890; 5000 rounds 0.866540 |
| historical_family | `P2-F03` | `DISCOVERY_ONLY` | - | pooled v2 benefit +0.006050 C; same-season CI crosses zero; original O hash matches v2 |
| historical_family | `P2-F09` | `INFORMATION_POSITIVE` | - | global predicted/observed 0.537237736/0.537238; official benefits global +0.003847, L2 -0.000832, L4 +0.004549 C |
| historical_family | `P2-F10` | `DISCOVERY_ONLY` | PROXY_EXPOSED | first screen benefit +0.002812 C; exposed confirmation -0.000678 C with legacy CI90 [-0.002019,+0.000747] |
| historical_family | `P2-F11` | `CLOSED_EXACT` | - | U predicted Public 0.535750480 C, predicted benefit +0.000785520 C vs current L4; upload 0 |
| key_case | `crossfit_rank1_v2` | `INFORMATION_POSITIVE` | - | {'benefit': 0.0024538337165882496, 'benefit_ci90': [0.0015699465453665516, 0.004330818825389526], 'prior_replay_state': 'HIGH_VALUE_CHALLENGER_RESEARCH_ONLY'} |
| key_case | `nested_pls` | `INFORMATION_POSITIVE` | - | {'benefit': 0.002041991611855032, 'benefit_ci90': [0.0010516304896743154, 0.0038742386414534645], 'prior_replay_state': 'HIGH_VALUE_CHALLENGER_RESEARCH_ONLY'} |
| key_case | `supervised_rank1` | `INFORMATION_POSITIVE` | - | {'benefit': 0.004798862410555937, 'benefit_ci90': [0.003107124366859581, 0.008505912410134975], 'prior_replay_state': 'HIGH_VALUE_CHALLENGER_RESEARCH_ONLY'} |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
