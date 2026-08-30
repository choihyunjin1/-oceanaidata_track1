# P3 neural sequence, RevIN, and masked-SSL models

- card id: `p3-neural-sequence-revin-ssl`
- problem: `P3`
- mechanism: RevIN patching, NLinear/DLinear/state-space/TimeXer-style models, and masked SSL.
- covered historical families: 2
- covered later key cases: 1
- primary status counts: `CLOSED_EXACT` 2, `DISCOVERY_ONLY` 1

## 재사용할 것

- Reuse selection-matched confirmation and exact no-op detection.
- Keep reference-mismatch subruns as QA lessons only.

## 그대로 반복하지 않을 것

- All listed exact valid variants and masked-SSL confirmation.

## 재개 조건

A materially new architecture on a G0-clean, untouched episode surface.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P3-F04` | `CLOSED_EXACT` | - | candidate benefit -0.00431369 m; legacy case and episode CIs both below zero |
| historical_family | `P3-F08` | `DISCOVERY_ONLY` | INVALID_TECHNICAL | valid matched variants materially worse; Gen6 full exact no-op; mismatch-only subvariants inadmissible |
| key_case | `selection_matched_masked_ssl` | `CLOSED_EXACT` | - | {'benefit': -0.31415523848029037, 'benefit_ci90': [-0.38878588237499895, -0.2448493754044661], 'prior_replay_state': 'PRIMARY_HARM_RESEARCH_ONLY'} |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
