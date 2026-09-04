# P1 parallel candidate cycle 2026-08-31 v4

## 결론

P1은 기존 내부 PASS `P1_2_HIST_GBDT_OOF_STACK_UNION` 1개를 유지한다. 이번
사이클에서 구조적으로 독립적인 causal boundary-extension 및 causal run-state
후보를 실제 4 fits로 Q2→Q3, Q2+Q3→Q4 순방향 검증했지만 새 PASS는 없었다.
따라서 P1의 내부 통과 제출본 수는 1개이며, 새 CSV는 만들지 않았다.

## 고정 설계

- 비교자: raw E150 OOF proxy
- 검증: Q2 학습→Q3 평가, Q2+Q3 학습→Q4 평가
- PASS gate: pooled F1 개선, 두 fold 모두 delta F1 0 이상, anchor removal 0,
  실제 addition 1개 이상
- 후보 A: 이전 앵커 뒤 3개 10분 스텝만 다루는 RandomForest boundary-extension
- 후보 B: 현재 및 과거 6스텝 신호만 쓰는 ExtraTrees causal run-state
- 두 후보 모두 미래 행 feature를 사용하지 않으며 add-only이다.

## 실측 결과

| 후보 | Q3 delta F1 | Q4 delta F1 | pooled delta F1 | 추가 TP/FP | 판정 |
|---|---:|---:|---:|---:|---|
| `P1_4_CAUSAL_BOUNDARY_EXTENSION_RF` | -0.001648578 | 0.000000000 | -0.000975439 | 0/22 | FAIL |
| `P1_5_CAUSAL_RUN_STATE_EXTRA_TREES` | 0.000000000 | 0.000000000 | 0.000000000 | 0/0 | FAIL |

Boundary 후보는 Q2에서 학습한 경계 패턴이 Q3으로 수송되지 않아 22개 추가가
모두 오탐이었다. Run-state 후보는 고정 신뢰도 기준에서 전부 abstain하여 기존
앵커와 동률이었다. 결과 이후 cutoff, 구조, fold 또는 feature를 바꾸지 않았다.

## 제출 상태

- 기존 PASS: `P1_2_HIST_GBDT_OOF_STACK_UNION`, 내부 delta F1
  `+0.001380986`, SHA-256
  `8c0069b9b73f196b22ed624fe20e8434158483aed642c34d8a6f2bdf0d48afba`
- 신규 materialized CSV: 0
- 공식 covariate read: 0
- hidden truth read: 0
- upload: 0

## QA

행 grain은 `station, year, layer, time`으로 고정했다. 두 후보 모두 Q3/Q4
287,862행을 평가했고 anchor removal 0을 독립 확인했다. PASS 후보만 CSV를 만들게
한 계약도 충족했다. `independent-qa.json`은 PASS이며, 모델 재학습 없이 원시
historical label/anchor confusion count로 모든 fold 및 pooled F1과 gate를 다시 계산한
`independent-recompute.json`도 PASS다.

실행 전후 `py_compile`, Ruff, focused pytest를 수행했다. 회귀 테스트는 미래 시점의
확률을 바꾸어도 이전 시점의 causal feature가 변하지 않는지, 두 candidate mask가
anchor positive를 선택하지 않는지를 확인한다.
