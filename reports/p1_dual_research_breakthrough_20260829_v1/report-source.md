# P1 이중 리서치·구조 돌파구 검증 보고서

## 결론

이번 사이클에서 **공식 제출로 승격할 새 후보는 없다.** 문헌과 Gemini가 제안한 방향을 현재 P1 계보에 맞게 세 가지 독립 구조로 구현해 봉인된 역사 미래창에서 확인했지만 모두 현 e150 계보를 넘지 못했다.

가장 중요한 새 증거는 `partial pooling`의 강한 선택창 착시다. Q2에서는 124개 추가 중 122개가 정답이어서 row-F1이 `+0.012522557` 상승했지만, Q3에서는 111개 추가 중 정답이 8개뿐이어서 `-0.006952610`, Q4에서는 exact no-op이었다. Q3+Q4 pooled는 `-0.004121280`이다. 따라서 관측소·층별 점수 보정이나 threshold 미세조정은 현재 도메인 이동을 견디지 못한다.

이번 결과는 “MS-TCN 계열의 절대 최대치”를 증명하지 않는다. 다만 현재의 e150 출력 뒤에 붙이는 저용량 보정, raw/type/boundary 재참조, 외부 I-ORS 점 잔차라는 세 축은 재튜닝 우선순위에서 내려야 한다. 다음 유효 축은 **학습 단계에서 환경별 최악 손실을 직접 다루는 station×layer×time-regime robust training**이며, 새 미래창 또는 공식 probe가 없는 한 성능 향상을 단정하면 안 된다.

## 기준선과 공식 증거의 해석 한계

- 팀 내부 최고는 `e150 + GI 2행`, Public F1 `0.833548`, 환산 점수 `28.909341`이다. 여기서 champion은 대회 전체 1위가 아니라 **우리 팀의 현 최고 제출**을 뜻한다.
- visible leader와의 점수 차이는 약 `3.096`점이므로 개선 여지는 분명하다. 그러나 그 차이가 특정 로컬 F1 차이와 일대일로 대응한다는 증거는 없다.
- e150 all과 GS-only의 공식 차이는 I-ORS 80행을 포함한 **파일 전체 방향의 조건부 효과**만 식별한다. 80행 각각이 TP이거나 recall 기여라는 인과 주장은 불가능하다.
- spike/noise/flatline/offset/drift 라벨은 행 단위로 중첩될 수 있다. 상호배타적 다중분류로 다루면 계약 위반이다.
- 공식 지표는 row-level F1이다. action-segmentation 문헌의 edit score나 F1@IoU는 보조 진단일 뿐 승격 목표를 대체하지 않는다.

## 자체 딥리서치에서 채택한 근거

1. 한국해양과학기술원의 KORS 수온 QC 연구는 고정 규칙 하나보다 관측소·수심·시간 문맥을 함께 다루는 필요성을 뒷받침한다: https://e-opr.org/articles/xml/rQWX/
2. MS-TCN++는 다단계 refinement로 시간적 과분할을 줄이지만, 원래 목적함수와 P1의 row-F1·중첩 anomaly-type 계약은 동일하지 않다. 구조 아이디어만 전이했다: https://github.com/sj-li/MS-TCN2
3. ASRF의 boundary branch는 구간 경계 표현의 근거지만, boundary 개선이 곧 row-F1 향상이라는 주장은 하지 않았다: https://arxiv.org/abs/2007.06866
4. RAINCOAT은 시계열 도메인 적응에서 주파수·시간 표현과 target alignment를 함께 다룬다. 우리 결과의 Q2→Q3 붕괴는 단순 score calibration보다 representation/domain robustness가 다음 축이라는 판단을 지지한다: https://proceedings.mlr.press/v202/he23b.html
5. GroupDRO류 접근은 평균 위험이 아니라 최악 환경 위험을 낮추는 구조적 대안이다. 다만 P1에서는 환경 정의와 fresh confirmation이 필요하므로 아직 성능 주장이 아니라 다음 실험 가설이다.

## 이번 사이클에서 실제 실행한 세 구조

### 1. type/boundary/raw-context refinement cascade

산출물: `artifacts/p1_mstcn_type_boundary_cascade_shadow_20260829_v1/result.json`

| 창 | baseline F1 | candidate F1 | delta | 핵심 원인 |
|---|---:|---:|---:|---|
| Q3 | 0.906858 | 0.898499 | -0.008359 | 207개 추가 중 37 TP; S-ORS에 198개 추가 집중 |
| Q4 | 0.887220 | 0.889447 | +0.002227 | FP 21개 순감소, TP 변화 없음 |
| pooled | - | - | -0.003992 | 창별 효과 반전 |

판정: `NO_GO_SINGLE_SEED_STRUCTURAL_SCREEN`. Q4의 작은 FP 개선은 Q3의 S-ORS false additions를 상쇄하지 못했다.

### 2. frozen bounded trust adapter

산출물: `artifacts/p1_mstcn_frozen_trust_adapter_shadow_20260829_v2/result.json`

| 창 | delta F1 | 변화 |
|---|---:|---|
| Q3 | -0.000812 | 9개 제거가 모두 TP |
| Q4 | -0.000106 | 1개 추가가 FP |
| pooled | 약 -0.00053 | baseline 보존에는 성공, 효용 없음 |

판정: `NO_GO_SINGLE_SEED_STRUCTURAL_SCREEN`. 강한 변형 폭을 막아도 안정적 이득은 생기지 않았다.

### 3. L2-shrunk station-layer partial pooling calibrator

산출물: `artifacts/p1_mstcn_partial_pooling_calibrator_shadow_20260829_v1/result.json`

Q2에서만 `C=0.1`, threshold `0.3`을 선택했다. Q3·Q4 blind payload를 모두 봉인한 뒤 truth를 열었다.

| 창 | baseline F1 | candidate F1 | delta | additions TP/rows |
|---|---:|---:|---:|---:|
| Q2 selection | 0.867676 | 0.880198 | +0.012523 | 122/124 |
| Q3 confirmation | 0.912188 | 0.905236 | -0.006953 | 8/111 |
| Q4 confirmation | 0.898901 | 0.898901 | 0.000000 | 0/0 |
| Q3+Q4 pooled | 0.906804 | 0.902682 | -0.004121 | 8/111 |

Q3의 I-ORS 추가 57개는 TP가 0개였고, S-ORS 추가 54개 중 TP는 8개였다. 선택창의 station-layer calibration은 미래창으로 transport되지 않았다.

## 과거 미실행처럼 보였던 외부 I-ORS 축 재감사

외부 I-ORS 2014–2023 점 잔차 실험은 미실행이 아니었다. `2026-08-21T17:47:44+09:00`에 one-shot outer validation을 완료했고 재실행 금지 lock이 있다.

- result: `artifacts/p1_iors_external_point_residual_oof_v1/20260821T174744+0900/result.json`
- decision: `NO_GO_POINT_RESIDUAL`
- overall weighted F1 delta: `-0.048997532`
- I-ORS micro F1 delta: `-0.193552793`
- improved folds: `0`
- paired block bootstrap 90% CI: `[-0.101087257, -0.029788056]`
- worst I-ORS layer F1 delta: `-0.986486486`

따라서 같은 외부 q50 residual을 다시 튜닝하는 것은 독립 돌파구가 아니다.

## 가설 공간 판정

| 축 | 현재 판정 | 근거 |
|---|---|---|
| 단일 checkpoint 마지막값 대신 중간값 | 닫힘 | e120/125/130/145/150 및 union/majority/intersection 재감사에서 e150 pooled 최고 |
| e150 뒤 station/layer threshold 보정 | 닫힘 | partial pooling Q2 강한 양성 → Q3 붕괴 |
| raw/type/boundary refinement 재참조 | 현재 recipe 닫힘 | cascade pooled 음수 |
| frozen 저용량 residual adapter | 현재 recipe 닫힘 | Q3/Q4 모두 음수 |
| 외부 I-ORS point residual | 닫힘 | one-shot outer에서 큰 음수 |
| long-event CP/LightGBM/RPCA/inpaint | 현재 구현 닫힘 | 기존 실행에서 no-op 또는 gate 실패 |
| 환경 강건 학습 자체 | 열림 | 아직 station×layer×time-regime worst-risk objective로 e150을 재학습하지 않음 |
| 진정한 source→2026 representation alignment | 조건부 열림 | target 무라벨 사용 범위와 새 검증 계약 필요 |

## 다음 단일 실험 제안

다음에는 여러 후처리를 동시에 만들지 않는다. `station × layer × season/quarter`를 environment로 두고, 현재 165개 past-only feature와 MS-TCN topology·anchor-preserving decoder는 고정한 채 학습 손실만 robust objective로 바꾼다.

1. Q2 이전 학습 prefix에서 환경별 row loss를 계산한다.
2. 평균 ERM과 worst-environment loss의 convex combination을 사전등록한다.
3. Q2에서 계수 하나와 checkpoint만 선택한다.
4. Q3·Q4는 다시 결과 기반 튜닝 없이 방향 일치 여부만 확인한다.
5. anchor positive는 제거하지 않고, 추가 행의 station별 precision과 pooled row-F1을 함께 본다.
6. Q3/Q4가 이미 연구 과정에서 여러 번 노출됐으므로 이는 retrospective confirmation으로 명시한다. 가능하면 다음 미노출 날짜창 또는 제한된 공식 probe를 최종 판단에 사용한다.

중단 조건은 두 미래창 delta 방향 불일치, 한 관측소에 추가의 80% 이상 집중, 또는 pooled delta ≤ 0이다. 이 조건이면 공식 CSV를 만들지 않는다.

## 최종 판단

이번 사이클의 값어치는 “새 제출물 생성”이 아니라 **잘못된 돌파구 세 개를 실제 수치로 닫고, 병목을 학습단 domain robustness로 좁힌 것**이다. 현재 결과를 공식 제출하면 제한된 기회를 소비할 근거가 없다. 다음 제출 후보는 robust-training 실험이 최소 두 창에서 같은 방향으로 이길 때만 만든다.
