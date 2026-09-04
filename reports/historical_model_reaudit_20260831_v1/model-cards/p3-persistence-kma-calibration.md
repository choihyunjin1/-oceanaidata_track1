# P3 persistence, shrink, and KMA calibration

- card id: `p3-persistence-kma-calibration`
- problem: `P3`
- mechanism: Persistence baselines, lead-specific shrink, reverse official axis, and KMA blending.
- covered historical families: 5
- covered later key cases: 0
- primary status counts: `CLOSED_EXACT` 1, `DISCOVERY_ONLY` 1, `INFORMATION_POSITIVE` 2, `PROXY_EXPOSED` 1

## 재사용할 것

- Reuse lead-specific official-factor decomposition and frozen alpha contracts.
- Keep uniform KMA alpha 0.425 as public-best lineage evidence.
- Reuse station-ablation direction only at displayed-score resolution.

## 그대로 반복하지 않을 것

- Positive-shrink A/B and exact local cross-fit KMA strategy.
- Nearby-alpha sweeps after observing the official optimum.

## 재개 조건

A preregistered new lead/station factor or an untouched calibration surface.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P3-F01` | `DISCOVERY_ONLY` | - | local incumbent 0.78016092 m; benefit +0.00658137 m vs router; original Public 0.607071 m |
| historical_family | `P3-F03` | `PROXY_EXPOSED` | CLOSED_EXACT | local benefits A +0.00044396 and B +0.00027536 m; official benefits A -0.004609 and B -0.002275 m |
| historical_family | `P3-F05` | `INFORMATION_POSITIVE` | CLOSED_EXACT | domain AUC 0.996779; calibrated outer benefit +0.00252937 m with CI crossing zero; deployment -0.00062087 m |
| historical_family | `P3-F10` | `INFORMATION_POSITIVE` | - | official benefits global +0.007999, 12h +0.000390, 18/24h +0.007689 m; all local directions reversed |
| historical_family | `P3-F11` | `CLOSED_EXACT` | - | one reverse-axis curvature/subset query; official scores absent and upload 0 |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
