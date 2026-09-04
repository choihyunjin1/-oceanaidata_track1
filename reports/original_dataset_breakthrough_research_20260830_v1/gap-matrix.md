# 원본 데이터 기반 돌파구 gap matrix

## 결론

원본 training 구조와 기존 실패 계보를 다시 맞춘 뒤 문제별 **0-fit support
preflight**를 먼저 수행했다. P1은 support 단계에서 종료됐고, P2와 P3만 봉인된
Stage-1 one-shot으로 넘어갔다. 최종적으로 이번 사이클에서 승격된 후보는 없다.
P2는 pooled 신호가 있으나 시간 안정성 gate를 아슬아슬하게 실패한 열린 근접
실패이고, P3의 exact masked-SSL + shared-Huber 조합은 전 slice 악화로 닫힌다.

| 문제 | 원본 데이터에서 다시 확인한 간극 | 이미 닫힌 exact lane | 아직 열린 메커니즘 | 가장 싼 반증 | stage-0 승격/종료 |
|---|---|---|---|---|---|
| P1 | 776,706행이지만 실질 독립 표본은 이진 연속 사건 263개다. typed 경계를 추가하면 289개이며 두 정의를 섞으면 안 된다. 최신 SupCon은 사건 support가 있었어도 proposal precision이 F1 개선 필요선보다 크게 낮았다. | row-level SupCon/soft-F1/top-k, 고정 Group-DRO, 32점 Sobol 공간, 단일 long-event generator, frozen-83 ranker exact recipe | 동결된 여러 OOF proposal bank를 사건으로 합친 뒤, incumbent 음성 영역에서 실제 add-only ΔF1 utility를 선택하는 작은 residual head | prefix별 utility-positive event와 OOF provenance만 집계 | fit≥10, calibration≥4, ≥3 station-layer cell, max-cell≤70%, provenance 100%. 하나라도 실패하면 0 fit 종료 |
| P2 | complete T/S 전층 시각 47,216개, depth까지 완전한 시각 47,215개다. 공개층 6/24/72h state는 99,756시각에서 유한하다. 고정 seasonal copula는 pooled 개선이 있었지만 Nov-Dec와 inner stability가 무너졌다. | seasonal empirical margin + 단일 Kendall Gaussian dependence + shrinkage 0.1/0.3/0.5, 243점 nested PLS exact family | 공개 T/S 성층·혼합 state에 따라 dependence 자체가 변하는 저자유도 conditional copula; OOD에서는 exact no-op | chronological block × state tertile의 Kendall τ 이질성/support 집계 | 재현되는 |Δτ|≥0.10 edge가 ≥2개이고 cell≥500, KST day≥30, training block≥2. 실패하면 0 fit 종료 |
| P3 | 공식 리드는 3/6/9/12/18/24h다. shared canonical builder의 48h elapsed high-state anchor는 8,121개다. 1.5≤Hs<2.2 및 12h 상승>0.2인 dense anchor는 2,131개, station-global 78h 독립 anchor는 243개(G/I/S=86/68/89)다. 별도의 `t-48h Hs finite` 엄격 진단은 2,117/242개다. | dense72 N-HiTS-style 45-fit hierarchical basis, causal sequence/spectral/state-space, forcing analog, ERA5 transfer, confirmed CatBoost challenger | selection-matched cohort가 각 forward window에도 충분하다면 fold-local masked SSL + frozen robust head 또는 sparse posterior-abstaining residual | canonical anchor/fold support·hash·센서오류 flag만 재계산 | station global≥30, complete applicable window≥2, 각 applicable window≥20, cadence/footprint/hash 모두 일치. 부족하면 selection-matched lane 종료 |

## Stage-0/Stage-1 실측 결론

| 문제 | Stage-0 | Stage-1 | 현재 판단 |
|---|---|---|---|
| P1 | Q2→Q3는 통과했지만 Q4 utility-positive event가 3개로 최소 4개에 못 미쳤고 proposal precision 0.1025도 F1/2 필요선 0.4572에 못 미침 | 0 fit; 실행하지 않음 | `NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT`. heterogeneous residual-head 가설을 현 데이터 support로 반증 |
| P2 | state cell 6/6, 이질성 edge 7개로 통과 | pooled ΔRMSE -0.003459°C, CI90 전부 음수이나 개선 window 1/3, JJA +0.003174°C로 0.003°C cap을 0.000174°C 초과 | exact recipe는 `NO_GO`; conditional-dependence 메커니즘은 후속 독립 사이클 대상으로 열어 둠 |
| P3 | canonical 157 validation cases, 창별 41/65/51로 통과 | candidate 1.125374m vs paired incumbent 0.811219m, Δ +0.314155m; 모든 window/station/lead 악화 | `NO_GO_CLOSE_THIS_EXACT_RECIPE`; masked-history + shared-Huber 조합 폐쇄 |

## 이상점 처리에 대한 공통 결정

- P1의 `label=1` 사건과 raw temperature 극값은 탐지 대상이므로 제거하지 않는다.
- P3의 고파고·급상승은 공식 선택조건과 맞닿은 신호이므로 제거하지 않는다.
- P2의 수괴 경계·약층 극값도 인접 층과 시간적으로 일관되면 보존한다.
- 삭제 후보는 중복 시각, 비물리 범위, 단일점 jump-and-return, stuck run처럼
  독립적인 센서 오류 징후가 겹친 경우로 한정한다.
- 기본 분석은 전량 보존 + robust scaling/rank/Huber이고, hard deletion은 민감도
  진단 arm일 뿐이다. 삭제 arm만 좋아지거나 rare-state support가 20% 이상 줄면
  이상점 제거 가설을 종료한다.

## 중요한 정정

과거 `audit_dataset_mechanisms_20260828.py`의 P3 결과는 공식 리드 대신
12/24/36/48/60/72h를 썼고, 78h count도 보고서에 적힌 rising subset이 아니라
더 넓은 high-state 집합에서 계산했다. 본 사이클의 `dataset-audit.json`은 공식
리드와 shared canonical 60분 anchor를 사용한다. 과거 254 count와 장기 리드
지표는 현재 승격 근거에서 제외한다. 또한 본 사이클 초안의 242 count는
`t-48h Hs finite`를 추가 요구한 더 엄격한 진단값이고, canonical masked-history
support 값 243과 구분해 보존한다.
