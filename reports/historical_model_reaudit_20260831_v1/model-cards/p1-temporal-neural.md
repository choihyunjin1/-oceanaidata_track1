# P1 temporal neural, MS-TCN, and representation learning

- card id: `p1-temporal-neural`
- problem: `P1`
- mechanism: TCN/MS-TCN sequence backbones, robust objectives, representation learning, and event decoders.
- covered historical families: 2
- covered later key cases: 7
- primary status counts: `CLOSED_EXACT` 5, `DISCOVERY_ONLY` 1, `OLD_GATE_REJECTED` 3

## 재사용할 것

- Preserve per-epoch best checkpoints and their validation surface instead of final epoch only.
- Reuse sealed Sobol candidate manifests, dependent block evaluation, and anchor-preserving decoders.
- Reuse the distinction between selection, confirmation, and official surfaces.

## 그대로 반복하지 않을 것

- The exact 32-point Sobol search space or frozen trial18/threshold 0.8 confirmation.
- Fixed Group-DRO and event-balanced SupCon objectives already shown harmful.

## 재개 조건

A materially new representation or objective with an untouched block surface and frozen checkpoint rule.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P1-F01` | `CLOSED_EXACT` | - | local micro-F1 0.767582 and 0.799755 |
| historical_family | `P1-F07` | `CLOSED_EXACT` | - | full-fraction benefits -0.310074/-0.338562/-0.508065/0.0 |
| key_case | `environment_balanced_replay` | `OLD_GATE_REJECTED` | - | {'benefit': 4.26e-05, 'benefit_ci90': None, 'prior_replay_state': 'REOPEN_FROZEN_CONFIRMATION_ONLY_LOW_PRIORITY'} |
| key_case | `event_balanced_supcon` | `CLOSED_EXACT` | - | {'benefit': -0.16487411020776765, 'benefit_ci90': None, 'prior_replay_state': 'PRIMARY_HARM_RESEARCH_ONLY'} |
| key_case | `group_dro_fixed_objective` | `CLOSED_EXACT` | - | {'benefit': -0.013480538245885798, 'benefit_ci90': None, 'prior_replay_state': 'PRIMARY_HARM_RESEARCH_ONLY'} |
| key_case | `hierarchical_event_precision_addonly` | `DISCOVERY_ONLY` | - | {'benefit': -0.0023805798050094973, 'benefit_ci90': [-0.017810464928264918, 0.013951393933980668], 'prior_replay_state': 'INCONCLUSIVE_RESEARCH_ONLY'} |
| key_case | `segment_precision_router_core` | `OLD_GATE_REJECTED` | - | {'benefit': 0.0009688594039944931, 'benefit_ci90': None, 'prior_replay_state': 'REOPEN_FROZEN_CONFIRMATION_ONLY'} |
| key_case | `sobol_trial18_threshold08` | `CLOSED_EXACT` | CHECKPOINT_PEAK, PROXY_EXPOSED | {'benefit': 0.0005656370384116149, 'benefit_ci90': None, 'prior_replay_state': 'REOPEN_FROZEN_CONFIRMATION_ONLY'} |
| key_case | `window_phase_consistency` | `OLD_GATE_REJECTED` | - | {'benefit': 0.00024178220395010275, 'benefit_ci90': None, 'prior_replay_state': 'REOPEN_FROZEN_CONFIRMATION_ONLY'} |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
