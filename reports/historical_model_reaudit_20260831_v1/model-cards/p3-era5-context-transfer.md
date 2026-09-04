# P3 ERA5/context-transfer models

- card id: `p3-era5-context-transfer`
- problem: `P3`
- mechanism: Fixed ERA5 source pretraining, source-quality gate, and local continuation.
- covered historical families: 1
- covered later key cases: 0
- primary status counts: `INVALID_TECHNICAL` 1

## 재사용할 것

- Reuse the 363-file manifest/checksum/time-continuity preflight and environment separation.
- Reuse the source-gate result from later valid ERA5 solution tests.

## 그대로 반복하지 않을 것

- The consumed dependency-failed one-shot lock.
- The exact later ERA5 Hs-squared/source-gate solution shown worse than incumbent.

## 재개 조건

A new preregistered attempt with ML dependency preflight and a materially new transfer target.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P3-F12` | `INVALID_TECHNICAL` | - | 2026-08-27 03:29 KST metadata: raw 305/363, partial 0, final recovery process running; no efficacy score |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
