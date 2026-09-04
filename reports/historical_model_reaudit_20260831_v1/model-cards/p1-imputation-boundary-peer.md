# P1 imputation, boundary, topology, and peer reliability

- card id: `p1-imputation-boundary-peer`
- problem: `P1`
- mechanism: Postprocessing and event reconstruction from blocks, boundaries, topology, or peers.
- covered historical families: 8
- covered later key cases: 3
- primary status counts: `CLOSED_EXACT` 5, `DISCOVERY_ONLY` 2, `OLD_GATE_REJECTED` 4

## 재사용할 것

- Reuse paired day/block deltas, exact changed-row sets, and critical-cell diagnostics.
- Keep block-inpaint and peer signals as frozen-confirmation hypotheses, not promoted models.

## 그대로 반복하지 않을 것

- Exact target-masked quantile, topology bridge, seeded boundary v2, and fixed24h peer recipes.
- Worst-slice-only vetoes that were not calibrated to official value.

## 재개 조건

New endpoint/peer mechanism plus fresh block confirmation; old recipes require exact frozen replay only.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P1-F02` | `OLD_GATE_REJECTED` | - | benefit +0.002591 F1; legacy CI90 [-0.015669,+0.051219]; worst station-layer -0.058874 |
| historical_family | `P1-F03` | `CLOSED_EXACT` | - | benefit -0.630823 F1; legacy CI90 [-0.704246,-0.599631] |
| historical_family | `P1-F08` | `CLOSED_EXACT` | - | honest-forward threshold benefit 0; synthetic pooled benefit -0.005692 with heterogeneous folds |
| historical_family | `P1-F09` | `CLOSED_EXACT` | - | semimarkov inner-only +0.002434 with catastrophic cell; residual pooled benefit 0 and rescued rows 0 |
| historical_family | `P1-F11` | `CLOSED_EXACT` | - | benefit -0.00145376 F1; 1/3 folds improved |
| historical_family | `P1-F14` | `CLOSED_EXACT` | INVALID_TECHNICAL | valid v2 micro benefit -0.00541901 F1; weighted benefit -0.00421524; legacy CI90 [-0.0134226,+0.0016918]; two earlier subruns invalid |
| historical_family | `P1-F15` | `OLD_GATE_REJECTED` | - | micro benefit +0.00463978 F1; weighted benefit +0.000120772; legacy CI90 [-0.001677,+0.011611]; worst group -0.0477463 |
| historical_family | `P1-F16` | `DISCOVERY_ONLY` | - | B-A test-share weighted benefit +0.00268799 F1, G-ORS benefit -0.00794118, legacy CI90 [-0.00921025,+0.00355313], normal FP-day +14.38% |
| key_case | `block_inpaint` | `OLD_GATE_REJECTED` | - | {'benefit': 0.002591206095757803, 'benefit_ci90': [-0.01566918281468185, 0.051219], 'prior_replay_state': 'REOPEN_FROZEN_CONFIRMATION_ONLY'} |
| key_case | `dynamic_peer_reliability` | `OLD_GATE_REJECTED` | - | {'benefit': 0.004639779101666464, 'benefit_ci90': [-0.001677, 0.011611], 'prior_replay_state': 'REOPEN_FROZEN_CONFIRMATION_ONLY'} |
| key_case | `gors_depth_invariance` | `DISCOVERY_ONLY` | - | {'benefit': 0.00268799, 'benefit_ci90': [-0.00921025, 0.00355313], 'prior_replay_state': 'INCONCLUSIVE_RESEARCH_ONLY'} |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
