# P2 Gaussian/state-conditioned copula models

- card id: `p2-copula-probabilistic`
- problem: `P2`
- mechanism: Conditional residual transport using empirical/Gaussian copulas and state availability.
- covered historical families: 0
- covered later key cases: 3
- primary status counts: `CLOSED_EXACT` 2, `INFORMATION_POSITIVE` 1

## 재사용할 것

- Reuse train-only support audit, profile mapper preflight, and frozen deployment packaging.
- Retain the official sign reversal as strong proxy-failure evidence.

## 그대로 반복하지 않을 것

- Gaussian copula v2 exact frozen official recipe.
- Availability-aware v2 exact primary recipe and incomplete v1 mapper.

## 재개 조건

A materially different conditional target with external calibration; no same-proxy retuning.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| key_case | `availability_aware_copula_v2` | `CLOSED_EXACT` | - | {'benefit': -0.001990430450011793, 'benefit_ci90': [-0.004967253376855786, -0.0006617799475544305], 'prior_replay_state': 'PRIMARY_HARM_RESEARCH_ONLY'} |
| key_case | `gaussian_copula_v2` | `CLOSED_EXACT` | PROXY_EXPOSED | {'benefit': 0.010616065033425048, 'benefit_ci90': [0.007700262306877281, 0.01738439679930879], 'prior_replay_state': 'HIGH_VALUE_CHALLENGER_RESEARCH_ONLY'} |
| key_case | `state_conditioned_copula` | `INFORMATION_POSITIVE` | - | {'benefit': 0.0034591760934881144, 'benefit_ci90': [0.0019229331946138068, 0.006529881874038945], 'prior_replay_state': 'HIGH_VALUE_CHALLENGER_RESEARCH_ONLY'} |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
