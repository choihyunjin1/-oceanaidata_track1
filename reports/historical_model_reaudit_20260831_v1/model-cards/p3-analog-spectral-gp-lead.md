# P3 analog, spectral, sparse-GP, and lead-continuous models

- card id: `p3-analog-spectral-gp-lead`
- problem: `P3`
- mechanism: Episode analogs, spectral kernels, GP abstention, and smooth lead/regime corrections.
- covered historical families: 3
- covered later key cases: 2
- primary status counts: `CLOSED_EXACT` 1, `DISCOVERY_ONLY` 4

## 재사용할 것

- Reuse globally 78h episode-disjoint splits and episode-block bootstrap.
- Keep lead-continuous as discovery evidence with its fresh one-case reversal attached.
- Reuse abstention coverage accounting independently from RMSE effect.

## 그대로 반복하지 않을 것

- Exact analog chain and matched spectral RFF recipe.

## 재개 조건

Multiple fresh independent episodes with a frozen mechanism; one episode is insufficient.

## 근거 레코드

| grain | id | primary status | tags | evidence |
|---|---|---|---|---|
| historical_family | `P3-F06` | `DISCOVERY_ONLY` | - | adaptive inner benefit +0.00592550 m; reused outer benefit -0.00327201 m and all folds worse |
| historical_family | `P3-F07` | `CLOSED_EXACT` | INVALID_TECHNICAL | valid matched rerun benefit -0.04278254 m; earlier mismatch result superseded |
| historical_family | `P3-F09` | `DISCOVERY_ONLY` | PROXY_EXPOSED | active benefit +0.00418782 m; legacy IID anchor-day CI90 [-0.00158453,+0.01012909]; delta_gain 0.005 m |
| key_case | `lead_continuous` | `DISCOVERY_ONLY` | PROXY_EXPOSED | {'benefit': 0.004187821972187478, 'benefit_ci90': [-0.00158453, 0.01012909], 'prior_replay_state': 'EXPLORATORY_CHALLENGER_RESEARCH_ONLY'} |
| key_case | `sparse_gp_abstention` | `DISCOVERY_ONLY` | - | {'benefit': -0.0034750705164049434, 'benefit_ci90': [-0.009929381222944561, 0.003149363373718584], 'prior_replay_state': 'INCONCLUSIVE_RESEARCH_ONLY'} |

## 해석 경계

`CLOSED_EXACT`는 이 카드의 모델 계열 전체가 아니라 원장에 적힌 정확한 레시피만 닫는다. `INFORMATION_POSITIVE`와 `CHECKPOINT_PEAK`도 자동 제출 승인을 뜻하지 않는다.
