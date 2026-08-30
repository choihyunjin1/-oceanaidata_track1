# P1 external-profile and density transfer

- card id: `p1-external-density-transfer`
- problem: `P1`
- mechanism: External profile transfer, point residuals, and target-covariate density correction.
- covered historical families: 2
- covered later key cases: 0
- primary status counts: `CLOSED_EXACT` 2

## 재사용할 것

- Reuse domain-shift diagnostics and fallback identity checks.
- Keep unlabeled profiles as diagnostics only unless competition-compatible targets exist.

## 그대로 반복하지 않을 것

- Exact external point-residual and density-ratio recipes.

## 재개 조건

New labeled transport information or a new causal transfer hypothesis.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P1-F04` | `CLOSED_EXACT` | - | point-residual benefit -0.063301 F1; external profiles have no P1 anomaly target labels |
| historical_family | `P1-F13` | `CLOSED_EXACT` | - | no new target comparison; emitted artifact reproduces Round B fallback |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
