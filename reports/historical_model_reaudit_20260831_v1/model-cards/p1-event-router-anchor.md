# P1 event router and official anchor lineage

- card id: `p1-event-router-anchor`
- problem: `P1`
- mechanism: Tabular/event-day routing, disagreement logic, and frozen row additions.
- covered historical families: 5
- covered later key cases: 0
- primary status counts: `DISCOVERY_ONLY` 1, `INFORMATION_POSITIVE` 3, `PROXY_EXPOSED` 1

## 재사용할 것

- Keep exact public-anchor hashes and row-set identities as regression baselines.
- Reuse the factorized G/I/S row-addition decomposition and additive-only safety contract.
- Use official probes only as directional mechanism evidence, never as rowwise truth.

## 그대로 반복하지 않을 것

- Round A exact rescue recipe and union disagreement rule.
- Any unfrozen recombination of G/I/S after observing an official score.

## 재개 조건

A preregistered, hash-frozen factor or a genuinely new router mechanism on a fresh surface.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P1-F00` | `DISCOVERY_ONLY` | - | local micro-F1 0.816737/0.757248/0.609332/0.806831/0.860371; XGB-original Public 0.790709 |
| historical_family | `P1-F05` | `PROXY_EXPOSED` | CLOSED_EXACT | local benefit +0.000571 to +0.002087; Public benefit -0.004564 |
| historical_family | `P1-F06` | `INFORMATION_POSITIVE` | - | event-day local benefit +0.004186; Public benefit +0.003001 |
| historical_family | `P1-F10` | `INFORMATION_POSITIVE` | - | Public vs B: Router +0.024163, Intersection +0.009218, Union -0.011404 F1 |
| historical_family | `P1-F12` | `INFORMATION_POSITIVE` | - | local vs B +0.000848/+0.000606/+0.001453 F1; test/local support shifts 9.17x and 16.13x |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
