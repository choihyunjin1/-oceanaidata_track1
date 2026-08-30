# P2 physical, external, and surrogate transfer

- card id: `p2-physical-surrogate-transfer`
- problem: `P2`
- mechanism: TEOS/tide/NASA/ERA5 additives, physical projections, and forward surrogates.
- covered historical families: 5
- covered later key cases: 0
- primary status counts: `CLOSED_EXACT` 3, `DISCOVERY_ONLY` 1, `PROXY_EXPOSED` 1

## 재사용할 것

- Reuse exact reference reconstruction, supported-row accounting, and physical no-op guards.
- Retain local-to-official sign reversals as calibration evidence.

## 그대로 반복하지 않을 것

- Exact surrogate v5, matched-budget fallback A/B, and tested external addons.

## 재개 조건

A genuinely new physical variable with causal timing and supported-row coverage.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P2-F04` | `DISCOVERY_ONLY` | - | TEOS benefit -0.049191 C; tide NO_GO; NASA exact no-op; ERA5 negligible harm |
| historical_family | `P2-F05` | `PROXY_EXPOSED` | CLOSED_EXACT | surrogate benefit +0.073375 C; official benefit -0.172435 C |
| historical_family | `P2-F06` | `CLOSED_EXACT` | INVALID_TECHNICAL | Stage-B p100 benefit -0.138081 C; joint-L4 p100 -0.006663 C; exact refit recipe absent |
| historical_family | `P2-F07` | `CLOSED_EXACT` | - | exact A benefit -0.038748 C; B supported rows 0 and fallback benefit -0.022857 C; official A/B -0.172435/-0.058836 C |
| historical_family | `P2-F08` | `CLOSED_EXACT` | - | 900/900 jobs and 45/45 cells QA PASS; p100 fallback benefit -0.024753 C and W=.50 stack -0.095301 C |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
