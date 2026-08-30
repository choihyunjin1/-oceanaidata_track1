# P2 deep, GBM, and stack models

- card id: `p2-deep-gbm-stack`
- problem: `P2`
- mechanism: Deep ensembles, GBM addons, and selected stacks.
- covered historical families: 2
- covered later key cases: 0
- primary status counts: `CLOSED_EXACT` 2

## 재사용할 것

- Reuse LOBO/nested estimates as the admissible score, not fitted-stack confidence intervals.
- Keep the tuned zero blend weight as evidence that the addon was unnecessary.

## 그대로 반복하지 않을 것

- Exact deep finalist stack and CatBoost layerwise/top-3 HPO recipes.

## 재개 조건

A new representation with nested outer evaluation, not a wider search of the same addon space.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P2-F01` | `CLOSED_EXACT` | - | fitted benefit +0.043076 C; LOBO benefit +0.013229 C; optimism gap 0.029846 C |
| historical_family | `P2-F02` | `CLOSED_EXACT` | - | CatBoost-layerwise LOBO benefit +0.001083 C with CI crossing zero; tuned final blend weight 0 |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
