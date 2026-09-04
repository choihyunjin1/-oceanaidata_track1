# P2 profile, analog, RFF, and prequential residual models

- card id: `p2-profile-analog-prequential`
- problem: `P2`
- mechanism: Annual/profile transport, analog retrieval, RFF state profiles, and prequential residuals.
- covered historical families: 7
- covered later key cases: 0
- primary status counts: `CLOSED_EXACT` 7

## 재사용할 것

- Reuse fail-fast p100 screens and layer/fold instability diagnostics.
- Reuse prequential timing and supported-row gates.

## 그대로 반복하지 않을 것

- All seven listed exact profile/analog/RFF/prequential recipes.

## 재개 조건

A new representation or target; parameter-only variants of the same transport are closed.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P2-F12` | `CLOSED_EXACT` | - | p100 candidate-minus-reference +0.53550567 C on 46965 eligible rows; late/full/fold gates failed |
| historical_family | `P2-F13` | `CLOSED_EXACT` | - | p100 candidate-minus-reference +0.05303207 C; all three folds worse |
| historical_family | `P2-F14` | `CLOSED_EXACT` | - | p100 candidate-minus-reference +0.23215957 C; fold and layer instability |
| historical_family | `P2-F15` | `CLOSED_EXACT` | - | p100 candidate-minus-reference +0.46921949 C; all folds worse and slice guard failed |
| historical_family | `P2-F16` | `CLOSED_EXACT` | - | p100 candidate-minus-reference +2.16461277 C; every fold and layer materially worse |
| historical_family | `P2-F17` | `CLOSED_EXACT` | - | p100 candidate-minus-reference +0.57258931 C; every fold and layer worse |
| historical_family | `P2-F18` | `CLOSED_EXACT` | - | p100 candidate-minus-reference +0.01946428/+0.02357576/+0.01966471 C; late/full/fold/slice gates failed |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
