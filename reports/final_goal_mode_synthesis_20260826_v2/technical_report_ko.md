# Ocean AI P1–P3 최종 종합 v2 — P2 권위 평가 반영

## 기술 요약

**실행 결함은 해결됐고, 그 결과 P2의 결론도 명확해졌습니다. 신규 구조는 full-budget 공정 평가에서 incumbent보다 열등하므로 정답은 신규 구조를 더 튜닝하는 것이 아니라 incumbent를 유지하는 것입니다.** P2 v5는 사전 봉인된 계약으로 900/900 jobs와 45/45 cells를 attempt 1회, resume 0회로 끝냈고 독립 사후 QA도 통과했습니다. 따라서 이번 판정은 실행 실패 보고가 아니라, 정상 실행으로 얻은 확정적인 no-promotion 결정입니다.

full-budget 78,156행에서 incumbent RMSE는 0.990022659℃였습니다. 활성 후보 중 가장 손실이 작은 fallback 계열도 1.014670867℃로 산술 ΔRMSE가 +0.024648208℃였고, paired-day bootstrap Δ는 +0.024752639℃, 90% CI [+0.017022633, +0.032302690]로 명확히 악화했습니다. Stack의 bootstrap Δ도 incumbent 비중을 높일수록 피해가 줄었지만 W0750은 +0.036064373℃였고 W0500은 +0.095300665℃였습니다. 모든 활성 후보는 3개 fold와 3개 layer에서 각각 0승이었습니다.

“구 모델은 완전히 최적화했고 새 모델은 덜 최적화해 역전하지 못한 것 아닌가”라는 우려도 등록 grid 안에서는 지지되지 않습니다. incumbent 비중을 0.50→0.625→0.75로 높일수록 악화가 +0.095301→+0.062994→+0.036064℃로 단조 감소하고 no-op에서 0이 됩니다. 최적 방향이 새 성분 강화가 아니라 incumbent 복귀이기 때문입니다. 다만 이는 등록 grid에 한정된 결론이며 모든 가능한 미래 구조를 배제하지는 않습니다.

P1은 776,706행, positive 32,126행, 80개 feature, 3개 seed의 full fit을 완료해 모델 번들을 동결했습니다. P3 ERA5는 metadata 관찰 시점에 225/363개(61.98%), partial 0, prepare Python process 2개로 진행 중이며 아직 성능 미판정입니다. 이 보고서에서는 CSV·submission을 만들거나 업로드하지 않았고 공식 test/sample/submission 경로를 열지 않았습니다.

## 핵심 결과: P2 full-budget에서는 모든 활성 후보가 악화

P2의 primary metric은 같은 78,156행에서 3개 outer fold와 layer 2·3·4를 동일 가중한 RMSE입니다. 산술 Δ는 후보 primary RMSE에서 incumbent primary RMSE를 뺀 값입니다. bootstrap Δ는 동일 KST calendar day를 함께 재표집해 계산한 값이라 비선형 집계 때문에 산술 Δ와 미세하게 다를 수 있습니다. 둘 다 음수가 개선, 양수가 악화입니다.

| 설정 | RMSE ℃ | 산술 ΔRMSE ℃ | bootstrap ΔRMSE ℃ | bootstrap 90% CI | fold 승 | layer 승 | 판정 |
|---|---:|---:|---:|---|---:|---:|---|
| INCUMBENT_NOOP | 0.990022659 | 0 | 0 | [0, 0] | 기준 | 기준 | 유지 |
| CAUSAL_RESIDUAL_SCALE025 | 0.990022659 | 0 | 0 | [0, 0] | 0/3 | 0/3 | correction 지원행 0, 개선 후보 아님 |
| CAUSAL_ON_FALLBACK | 1.014670867 | +0.024648208 | +0.024752639 | [+0.017022633, +0.032302690] | 0/3 | 0/3 | 기각 |
| FALLBACK_BLEND50_A0625 | 1.014670867 | +0.024648208 | +0.024752639 | [+0.017022633, +0.032302690] | 0/3 | 0/3 | 기각 |
| STACK_W0750 | 1.025954556 | +0.035931897 | +0.036064373 | [+0.025866383, +0.046020266] | 0/3 | 0/3 | 기각 |
| STACK_W0625 | 1.052837568 | +0.062814909 | +0.062994065 | [+0.048046210, +0.077469729] | 0/3 | 0/3 | 기각 |
| STACK_W0500 | 1.085108890 | +0.095086231 | +0.095300665 | [+0.076047286, +0.114234377] | 0/3 | 0/3 | 기각 |

CAUSAL_ON_FALLBACK과 FALLBACK_BLEND50_A0625는 이 평가에서 동일한 출력과 지표를 냈습니다. CAUSAL_RESIDUAL_SCALE025는 incumbent와 수치가 같지만 supported correction row와 nonzero correction row가 모두 0이므로 작동한 개선 모델로 셀 수 없습니다.

## 학습량을 늘리면 p040의 이득이 사라진다

p040에서 STACK_W0500은 Δ -0.115903632℃, 90% CI [-0.142705904, -0.089421416], fallback은 Δ -0.049506302℃, CI [-0.059749744, -0.039250619]로 robust improvement였습니다. 그러나 p055에서 fallback CI가 0을 포함하며 이득이 사라졌고 stack은 악화했습니다. p070, p085, p100에서는 두 계열 모두 악화했습니다.

| 학습 fraction | incumbent RMSE ℃ | fallback bootstrap ΔRMSE ℃ | STACK_W0500 bootstrap ΔRMSE ℃ | 해석 |
|---:|---:|---:|---:|---|
| 0.40 | 1.432998560 | -0.049506302 | -0.115903632 | 저자료에서만 robust 개선 |
| 0.55 | 1.116903623 | +0.001878824 | +0.031598868 | fallback 이득 소멸, stack 악화 |
| 0.70 | 1.148420057 | +0.005941554 | +0.031638702 | 모두 악화 |
| 0.85 | 1.025529373 | +0.026755814 | +0.090444372 | 모두 명확히 악화 |
| 1.00 | 0.990022659 | +0.024752639 | +0.095300665 | full-budget에서 모두 명확히 악화 |

이 패턴은 새 구조가 incumbent보다 우월하다는 증거가 아니라, 데이터가 적을 때 생기는 regularization 효과로 해석하는 것이 타당합니다. p040 신호는 연구 가설로 보존할 수 있지만 full-budget 배포 또는 제출 근거로 운반할 수 없습니다.

## 비교 범위와 지표 정의

- P2 v5의 locked population은 fraction마다 동일한 78,156행입니다. fold 행 수는 26,167, 25,338, 26,651이고 seed는 20260823·20260824·20260825입니다.
- primary score는 fold-equal/layer-equal RMSE ℃이며, paired-day bootstrap은 동일 KST calendar day draw를 공유하는 5,000회 재표집입니다.
- 서로 다른 validation surface의 절대 RMSE는 직접 비교하지 않습니다. 각 surface 안에서 candidate-minus-incumbent 방향만 해석합니다.
- 공식 점수는 사용자가 제공한 original/A/B 3개 관측이며 local selection·tuning이 끝난 뒤 transport 감사에만 사용했습니다. 로컬 제출 패키지로 후보 매핑을 독립 검증할 수는 없습니다.
- P1 full fit은 운영 산출물 동결이며 새 local/official 성능 비교가 아닙니다. P3는 다운로드 metadata만 있어 아직 모델 결과가 아닙니다.

## 방법과 봉인 계약

P2 v5는 3개 outer fold, 5개 prefix fraction, 3개 seed의 45개 seeded cell을 사용했습니다. top-level component job은 900개이며 모든 job과 cell을 새 namespace에서 완료했습니다. 첫 실행 한 번으로 끝났고 resume은 없었습니다. 결과를 본 뒤 weight·grid·모델 정의를 바꾸거나 재실행하지 않았습니다.

사후 verifier는 terminal/result/control/auth/config/module/runner/seal 연결, 900 jobs, 45 cells, 5개 aggregate parquet의 score 재계산, partial 부재, 공식 경로·submission·upload·P3 mutation 0을 독립 확인했습니다. model/checkpoint 값은 verifier가 열지 않았습니다.

## 구 최적화 모델과 새 구조의 비교

기존 matched-budget v1은 두 개의 서로 다른 진단면을 분리했습니다. exact frozen lineage에서는 incumbent 0.749530162℃보다 sealed-grid stack 최선 0.788278436℃가 나빴습니다. old forward causal surrogate의 5개 fraction 합산 headline에서는 incumbent 1.222532062℃보다 STACK_W0500 1.149959725℃가 좋아 보였지만, 이 값은 v5 p100과 비교 단위가 다릅니다. protocol-aligned한 v1 p100만 보면 incumbent 1.012448338℃, STACK_W0500 1.077449852℃, 산술 Δ +0.065001514℃로 이미 악화였습니다. 이번 v5 p100에서도 incumbent 0.990022659℃보다 같은 STACK_W0500이 1.085108890℃로 악화했습니다.

서로 다른 표본·lineage·fit semantics의 절대 RMSE는 직접 뺄 수 없습니다. 비교 가능한 p100 방향은 v1과 v5가 모두 non-noop 후보를 기각하며, v5가 그 방향을 확인·강화했습니다. old exact panel도 같은 방향입니다. 따라서 “새 모델을 충분히 최적화하면 뒤집힐 것”이라는 가설로 결과 기반 추가 탐색을 정당화할 수 없습니다.

## 로컬과 공식 점수의 transport 감사

사용자가 제공한 공식 P2 RMSE는 original 0.541085, Round A 0.713520, Round B 0.599921입니다. 즉 original 대비 A는 +0.172435℃, B는 +0.058836℃ 악화했습니다. 이 값들은 local selection·tuning이 끝난 뒤 기록했으며, 공식 사이트나 로컬 제출 패키지에서 후보 매핑을 독립 검증한 값은 아닙니다.

| 문제·로컬 평가면 | Round A 로컬/공식 방향 | Round B 로컬/공식 방향 | 방향 일치 |
|---|---|---|---:|
| P1 common local | 개선 / 악화 | 개선 / 개선 | 1/2 |
| P2 exact frozen lineage | 악화 / 악화 | 악화 / 악화 | 2/2 |
| P2 old 5-fraction pooled surrogate | 개선 / 악화 | 개선 / 악화 | 0/2 |
| P2 old protocol-aligned p100 | 악화 / 악화 | 악화 / 악화 | 2/2 |
| P3 shrink local | 개선 / 악화 | 개선 / 악화 | 0/2 |

따라서 로컬 점수 전체가 공식 점수를 그대로 대변한다고 볼 수는 없습니다. 그러나 P2에서는 exact 및 protocol-aligned p100의 두 후보 방향이 공식과 모두 일치했고, pooled surrogate는 두 후보 모두 반대로 예측했습니다. P2 의사결정에는 exact/full-budget surface를 우선하고 pooled surrogate를 배제하는 것이 맞습니다. v5는 새 공식 제출을 하지 않았으므로 공식 성능을 주장할 수 없지만, 그 no-promotion 방향은 과거 A/B transport 증거와 일관됩니다.

문제별 3점은 두 후보의 방향 감사에는 유용하지만 절대 보정식·순위 상관·일반적 rank equivalence를 추정하기에는 너무 작고 validation population과 lineage도 다릅니다. 추가 제출은 제출 전에 고정한 local score와 공식 score의 쌍을 계속 누적해 별도 평가해야 합니다.

## P1과 P3 상태

**P1은 해결 완료입니다.** production parser와 full preprocessing을 통과한 776,706행, positive 32,126행, 80 features를 사용해 세 seed를 실제 fit했습니다. 동결 bundle은 6,879,614 bytes이며 SHA-256은 fdcf0d96aa88603841778a82546e404d15bf5c479443db453fa3751cee5c0052입니다. 예측·점수·submission·upload는 모두 0입니다.

**P3는 미판정이 맞습니다.** 2026-08-26 00:45 KST metadata 기준 raw/yearly 225/363(61.98%), partial 0, prepare Python process 2개였습니다. 데이터 값은 열지 않았고 고정 protocol도 바꾸지 않았습니다. 다운로드·preflight·봉인 one-shot 실행이 끝나기 전에는 성공이나 실패로 분류하지 않습니다.

## 한계·불확실성·강건성

- P2 결론은 등록된 모델 family와 grid에 대한 결론이지 모든 가능한 미래 구조의 불가능성 증명이 아닙니다.
- p040의 개선은 실재하지만 p055 이상으로 운반되지 않으므로 full-budget 우월성 증거가 아닙니다.
- paired-day bootstrap은 historical pseudo-test 불확실성 요약이며 official hidden evaluation을 대체하지 않습니다.
- CAUSAL_RESIDUAL_SCALE025의 동률은 correction이 작동해 얻은 성능이 아니라 지원행 0으로 인한 no-op입니다.
- 서로 다른 과거 비교면의 절대 RMSE를 직접 비교하지 않았습니다.
- original/A/B 3개 공식 관측으로 두 후보 방향은 감사할 수 있지만 calibration 또는 일반적 rank equivalence를 추정할 수는 없습니다.
- P3 ERA5 결과는 아직 없으므로 이번 P2 결론과 분리했습니다.

## 권고되는 다음 단계

1. P2 incumbent를 유지하고 v5 active candidate는 승격하지 않습니다.
2. 결과를 본 뒤 새 stack weight, fallback alpha 또는 grid를 추가해 재실행하지 않습니다.
3. p040 결과는 저자료 regularization 연구 가설로만 기록합니다.
4. P1 frozen model bundle은 hash로 보존하되, 별도 승인 없이는 예측·제출을 만들지 않습니다.
5. P3 ERA5는 기존 봉인 protocol을 그대로 완료하고 그 결과만 별도 판정합니다.
6. 추가 제출마다 사전에 고정한 local score와 공식 score를 쌍으로 보존하고, 표본이 더 쌓인 뒤 calibration 가능성을 다시 평가합니다.

## 추가로 답해야 할 질문

1. P2에서 full-budget으로 갈수록 새 성분이 악화하는 원인은 학습 target shift, base error correlation, 또는 meta-selection 중 어디에 집중되는가?
2. p040의 regularization 이득을 새 family 없이 incumbent 자체의 학습 규제로 흡수할 수 있는가?
3. official 관측을 최소 몇 개 확보해야 local rank 방향 일치율을 실용적으로 평가할 수 있는가?
4. P3 ERA5 source gate가 실패하면 외생 transfer family를 종료할 것인가?

## 재현성 상태

P2 terminal status는 COMPLETE_LOCAL_AUTHORITATIVE_SURROGATE_V5_NO_PROMOTION입니다. result SHA-256은 049d2a8279b4567787dcad562972440a2d135d4f4215f4ebd59478b072131cdc, terminal receipt는 dbde6e3da46c0e8e8b179f0a818295106ff2f84296b6a564158c0704b6fa22d9, postexecution QA는 1c61704a53c95bbc2093324019fb5f9ade79acb51fd276e5647282f1027eb94a, QA manifest는 099a15480b33c039571a2bac41059b7e07dac5afe1c350c6b5bc8217e733d633입니다. 공식 test/sample/submission 접근, CSV/submission 생성, upload는 모두 수행하지 않았습니다.
