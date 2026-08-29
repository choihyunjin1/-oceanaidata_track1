# P1·P2·P3 음성 증거 통합 원장

기준일: 2026-08-30 KST
목적: 이미 시험해 실패가 확인된 **정확한 구조·규칙·검증계약**을 재실행하지 않고, 기술 오류나 미검증 가설을 과학적 실패로 오분류하지 않기 위함.

## 결론

지금까지의 결과는 세 가지로 나뉜다.

1. **닫힌 exact family**: 동일 구조·동일 정보축·동일 gate로 다시 실행할 이유가 없다.
2. **기술적 INVALID 또는 계약상 식별 불가**: 성능이 나쁘다고 입증된 것이 아니다. 결함을 먼저 고친 새 experiment ID가 필요하거나, 현 데이터 계약에서는 실행 자체가 불가능하다.
3. **미검증 가설**: support나 selection 신호만 존재하며 승격·제출 근거가 아니다.

따라서 “TCN 전체”, “CatBoost 전체”, “외부자료 전체”, “copula 전체”처럼 넓게 닫지 않는다. 이 원장이 금지하는 것은 아래에 적힌 exact recipe의 반복이다.

## P1 — 닫힌 exact family

| 묶음 | 입증된 음성 결과 | 처분 | 근거 |
|---|---|---|---|
| 초기 tabular/sequence 후속 | TCN·Patch Transformer가 당시 XGB보다 열위; sequence v2/v3/v4 악화, v5 no-op | 해당 exact architecture 종료 | `promotion_retroaudit_20260827_v1` P1-F01/F07 |
| 규칙·구간 복원 | block inpaint CI가 0을 지나고 worst slice 악화; target-masked quantile 큰 악화; semi-Markov/long-event residual outer 0 또는 rescue 0 | 해당 exact rule/decoder 종료 | P1-F02/F03/F09 |
| 외부·합성 전이 | IORS point residual 악화; Round A causal rescue는 local 미세 양수이나 Public 악화; synthetic event injection pooled 악화 | 해당 전이·generator·threshold 종료 | P1-F04/F05/F08 |
| incumbent postprocess | Union이 Round B보다 Public 악화; endpoint-unanimity bridge 악화; density correction은 fallback만 재현 | exact Union/bridge/density correction 종료 | P1-F10/F11/F13 |
| boundary·peer·depth 규칙 | seeded boundary completion, fixed-24h peer reliability, symmetric depth mask가 fold/slice/FP gate 실패 | 해당 exact rule 종료 | P1-F14/F15/F16 |
| TS2Vec prototype | coverage를 100%로 복구한 뒤에도 calibration·qualification TP=0, F1=0 | representation-plus-prototype exact family 종료 | `execution_followup_20260828_v7` |
| exact degradation-mask Transformer | noise는 통과했으나 offset/drift boundary와 F1 gate 실패 | 현 mask/architecture 명세 종료 | `approved_parallel_execution_20260828_v9` |
| environment-balanced replay | 10-epoch shadow에서 Q3 음수, 23개 추가행 TP=0 | balanced-replay proxy 종료 | `p1_value_preflight_robust_screen_20260829_v1` |
| MS-TCN++/ASRF add-only global threshold | Q2 +0.09816의 고립 peak가 Q4 -0.03148, pooled -0.00514로 붕괴 | 단일 Q2 최대점·전역 threshold·즉시 OR recipe 종료 | `p1_incumbent_preserving_mstcn_asrf_v2` |
| sparse bootstrap veto | historical Q3/Q4는 강했지만 Public F1 0.820339로 당시 팀 최고보다 -0.013209 | 강한 sparse removal 운송 규칙 종료 | `p1_mstcn_lower_bound_veto_20260829_v2` |
| window-phase consistency | q99 0.003728, XOR 29, 기본 e150 대비 약 +0.000242로 preflight 실패 | phase-consistency 학습축 종료 | `parallel_local_preflight_cycle_20260829_v1` |
| Sobol-selected MS-TCN space | 32 discovery + 4 reseed fits; all-month 양수였지만 pooled ΔF1 +0.000566 < +0.003 | 봉인된 32-point 공간 추가 튜닝 금지 | `parallel_hpo_cycle_20260829_v1` |
| Group-DRO MS-TCN | pooled ΔF1 -0.01348, 최소 월 -0.02355, station share 0.802817 | 고정 group loss/objective 종료 | `parallel_robust_repair_cycle_20260829_v2` |

현재 P1 팀 최고 계보를 만든 e150+GI2 자체는 닫힌 family가 아니다. 다만 GI 전체 6행, 강한 sparse veto, 동일 전역 threshold 재탐색은 공식 또는 확인창 증거로 이미 분해됐다.

## P2 — 닫힌 exact family

| 묶음 | 입증된 음성 결과 | 처분 | 근거 |
|---|---|---|---|
| 초기 deep/GBM addon | fitted optimism, CatBoost LOBO 미세·불확실 이득, tuned blend weight 0 | 해당 stack/addon 종료 | `promotion_retroaudit_20260827_v1` P2-F01/F02 |
| 외부·물리 addon | TEOS 큰 악화, tide no-go, NASA no-op, ERA5 극미세 harm | 시험한 exact addon 종료 | P2-F04 |
| surrogate/architecture matched | forward surrogate는 local +0.0734 뒤 Public -0.1724; curve/L4 및 matched A/B가 full-prefix/Public에서 역전 | 해당 exact surrogate/refit/fallback 종료 | P2-F05/F06/F07/F08 |
| density/annual/offset/profile analog | universal density weight .10, annual transfer, terminal offset, median consensus, OAS conditional, day-sequence analog, RFF state profile가 gate 또는 전 fold 악화 | 해당 exact recipe 종료 | P2-F10/F12–F17 |
| prequential residual | H1/H2/H3 모두 late/full/slice gate 실패 | 세 exact hypothesis 종료 | P2-F18 |
| dynamic low-rank/GP 계열 | uncertainty guard로 active 0 또는 사실상 no-op; unsafe cap 완화 금지 | 봉인된 state-space/GP recipe 종료 | `parallel_deep_research_breakthrough_20260828_v5`, `parallel_deep_research_execution_20260828_v3` |
| supervised rank-1 및 heave residual | pooled 미세 개선이 있었지만 fold 회귀·proxy lineage·활성 0.0687% 또는 effect gate 미달 | 해당 exact residual recipe 종료 | `independent_information_axis_deep_research_20260828_v15`, `parallel_deep_research_execution_20260828_v4` |
| two-sided boundary bridge | 필요한 flank 2개가 official hidden target 기간과 충돌 | 현 outside-flank 계약 종료 | `parallel_local_preflight_cycle_20260829_v1` |
| nested PLS capacity grid | 243×3 평가, 84 fits; pooled -0.00204°C이나 fold 개선/회귀/무변화 1개씩, inner eligibility 실패 | 봉인된 close-family 종료 | `parallel_hpo_cycle_20260829_v1` |

P2 OAS/rank-1 **전체**를 닫지 않는다. OAS40과 후속 rank-1 강도축은 실제 Public 개선을 만들었다. 다만 같은 exposed OOF에서 local-only 순위를 절대 gate로 쓰거나, 위 exact residual/PLS 조합을 다시 탐색하지 않는다.

## P3 — 닫힌 exact family

| 묶음 | 입증된 음성 결과 | 처분 | 근거 |
|---|---|---|---|
| positive shrink/patch/analog/spectral | positive shrink official 방향 역전, RevIN patch 악화, causal analog outer 악화, spectral RFF 큰 악화 | 해당 exact axis/patch/chain 종료 | `promotion_retroaudit_20260827_v1` P3-F03–F07 |
| 기타 구조 | NLinear/sequence/dense72 악화, Gen6 no-op; 일부 mismatch 결과는 증거 제외 | 유효 exact variants만 종료 | P3-F08 |
| TimeXer-style direct | incumbent 0.77995m 대비 0.87844m, 0/3 fold 개선 | exact direct six-lead 구조 종료 | `parallel_deep_research_breakthrough_20260828_v5` |
| ERA5 transfer solution | source/transfer 신호는 있었지만 incumbent 대비 +0.00233m, S-ORS +0.02170m | 현 solution gate recipe 종료; ERA5 전체는 미반증 | `parallel_deep_research_execution_20260828_v3` |
| future-wind MOS | perfect-future-wind oracle도 pooled +0.00134m 악화, 7/7 gate 실패 | predicted-future-wind + frozen-KMA MOS 계약 종료 | `parallel_local_preflight_cycle_20260829_v1` |
| KMA local alpha surface | cross-fit 최선도 +0.00618m 악화, 1/3 fold 개선 | local 세분화 α를 제출값으로 쓰는 전략 종료 | `p3_kma_alpha_surface_sweep_20260829_v1` |
| frozen CatBoost challenger_21 | selection ΔRMSE -0.02286m가 confirmation에서 +0.00797m로 역전; 3/3 fold, 3/3 station, 6/6 lead 악화, bootstrap CI90 전부 양수 | frozen candidate·iteration·router·KMA 조합 종료; 138-fit search 재실행 금지 | `p3_catboost_confirmation_contract_repair_20260830_v3` |

P3의 18/24h KMA 공식 축은 0→20%→40%에서 개선했으나, 이후 lead 분리·강도 외삽은 공식 성능이 다시 나빠지는 관측이 있었다. 따라서 동일 축을 더 세분화해 Public을 반복 질의하는 접근은 종료하고, 새 정보축 또는 기술적으로 완결된 local confirmation을 요구한다.

## 실패로 세면 안 되는 항목

| 항목 | 현재 상태 | 이유 | 다음 조건 |
|---|---|---|---|
| P2 copula support audit | `TRAIN_ONLY_SUPPORT_PASS_QUERY_AUDIT_NOT_AUTHORIZED` | 47,216 complete timestamps와 최소 unique 2,353은 support 증거일 뿐 복원 성능이 아님 | query-independent 모델을 새 ID로 평가하거나 별도 승인된 query audit |
| P2 copula conditional mean v1 | `INVALID_TERMINAL_TECHNICAL_FAILURE_RESOLVED_BY_V2` | incomplete historical profile mapper가 첫 metric 전 종료; prediction 0개, 과학적 점수 0개 | 같은 ID 재실행 금지; v2 exact-zero fallback으로 계약 수리 |
| P3 CatBoost ordered HPO v1 | `INVALID_TERMINAL_TECHNICAL_FAILURE` | `Ordered + Depthwise` 비호환으로 75번째 시도에서 중단; ranking 0 | valid-combination smoke를 선행한 새 ID |
| P3 CatBoost valid HPO v2 | `INVALID_TERMINAL_TECHNICAL_FAILURE_RESOLVED_BY_V3` | 138-fit selection은 ΔRMSE -0.02286m로 통과했지만 첫 confirmation 뒤 컬럼 계약 KeyError. v3에서 contract를 수리해 과학적 NO_GO를 판정함 | 기존 v2 lock 재사용 금지; v3 frozen candidate는 닫힌 exact family로 이동 |
| P2 boundary residual bridge | `NO_GO_CONTRACT_LEAKAGE` | 성능 실패가 아니라 필요한 입력이 금지 기간과 충돌 | 같은 이름으로 내부 flank로 바꾸지 말 것 |

## 2026-08-30 P2 copula 판정

`p2_gaussian_copula_conditional_mean_20260830_v2`는 pooled `-0.010616°C`, 2/3 fold, 3/3 layer 개선, bootstrap CI90 upper `-0.007700°C`로 사전등록 핵심 신호를 모두 통과했다. 그러나 `2025_nov_dec +0.034267°C`와 training-only inner worst-group instability 때문에 더 엄격한 실행 gate는 실패했다. 따라서 copula 전체를 닫지 않고, 동일 seasonal empirical margins + Kendall latent correlation + `[0.1,0.3,0.5]` shrinkage + 동일 split recipe만 재실행 금지한다. 결과 분류는 `PRIMARY_SIGNAL_PASS_STRICT_STABILITY_NO_GO_RESEARCH_ONLY`다.

## 공식 점수로 확인된 exact 축

- P1 e150+GI2는 Public F1 0.833548로 팀 최고가 됐고, GI 전체 6행은 0.833333으로 더 나빴다. “추가행이 많을수록 좋다”는 가설은 기각됐다.
- P1 sparse lower-bound veto는 Public F1 0.820339로 팀 최고보다 -0.013209였다. historical Q4의 강한 제거 규칙은 Public으로 운반되지 않았다.
- P2 OAS/rank-1은 여러 차례 Public 개선을 만들었다. 따라서 이 축은 family 전체를 닫지 않고, 실패한 강도·후처리만 닫는다.
- P3 KMA long-lead 20%와 40%는 당시 기준 개선했으나 local cross-fit은 반대였고, 후속 세분화는 불안정했다. local↔official 보정식을 전 문제 또는 전 family에 공통 적용하지 않는다.

2026-08-29 마감 직전 관측된 추가 공식 점수(P2 정점/층별, P3 공식기하)는 이 커밋 전 저장소에 독립 receipt가 없었다. 수치는 별도 원장에 receipt와 함께 보존되기 전까지 본 원장의 “입증된 exact 축”에 합치지 않는다.

## 재실행 금지 규칙

1. 표에 적힌 exact family는 이름만 바꿔 같은 feature·split·postprocess·threshold를 다시 실행하지 않는다.
2. `INVALID`는 NO_GO로 바꾸지 않는다. 새 ID, 새 lock, 실패 원인에 대한 contract test가 있어야 다시 시작한다.
3. 이미 노출된 Q2/Q3/Q4와 동일 OOF는 discovery 자료다. fresh confirmation 또는 일회성 official mechanism probe 없이 `LOCAL_CONFIRMED`라고 부르지 않는다.
4. 작은 local 개선은 기록하되 fold 방향, critical slice, effect size와 함께 본다. 공식 점수와 물리 단위의 delta를 동일 숫자로 비교하지 않는다.
5. 다음 딥리서치는 이 원장의 닫힌 exact recipe와 구조적으로 다른 후보만 추천해야 한다.

## 원장 범위와 주요 근거

- 전체 2026-08-27 family 재분류: `reports/promotion_retroaudit_20260827_v1/report-source.md`
- 2026-08-28 구조 실행 계보: `reports/parallel_deep_research_execution_20260828_v2`~`v4`, `execution_followup_20260828_v7`, `approved_parallel_execution_20260828_v9`
- 2026-08-29~30 preflight/HPO/repair: `reports/parallel_local_preflight_cycle_20260829_v1`, `parallel_hpo_cycle_20260829_v1`, `parallel_robust_repair_cycle_20260829_v2`, `p3_catboost_confirmation_contract_repair_20260830_v3`, `p2_gaussian_copula_conditional_mean_20260830_v2`
- 공식 점수 계보: `reports/deadline_submission_results_20260828_v1/official-results.md`, `reports/p1_mstcn_lower_bound_veto_20260829_v2/report-source.md`

이 문서는 과거 보고서를 대체하지 않는다. 연구 자원 배분을 위한 상위 색인이고, 상세 수치·해시·한계는 각 원 보고서가 권위 원장이다.
