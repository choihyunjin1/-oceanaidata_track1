# P3 CatBoost and tabular routing

- card id: `p3-catboost-tabular`
- problem: `P3`
- mechanism: Corrected repeated-forward CatBoost, routing, and successive-halving challengers.
- covered historical families: 1
- covered later key cases: 1
- primary status counts: `CLOSED_EXACT` 1, `DISCOVERY_ONLY` 1

## 재사용할 것

- Reuse synthetic compatibility smoke tests and confirmation-schema preflight.
- Keep the exact181 benchmark as a reproducible exposed-surface anchor.

## 그대로 반복하지 않을 것

- Frozen challenger21 confirmation and incompatible Ordered/non-symmetric grid.
- Selection score as confirmation evidence.

## 재개 조건

A new feature target or untouched episode surface after full parameter compatibility validation.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P3-F02` | `DISCOVERY_ONLY` | - | exact181 0.77910484 m; strong split and reproduction integrity but exposed labels |
| key_case | `catboost_repaired_confirmation` | `CLOSED_EXACT` | PROXY_EXPOSED | {'benefit': -0.007974130725359796, 'benefit_ci90': [-0.013854034848595, -0.0015388552106252453], 'prior_replay_state': 'PRIMARY_HARM_RESEARCH_ONLY'} |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
