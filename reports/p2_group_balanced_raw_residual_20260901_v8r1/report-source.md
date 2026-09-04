# P2 group-balanced raw-residual exploratory cycle 20260901 v8r1

## 결론

상태: `EXPLORATORY_NO_GO_BOTH_PREDECLARED_OBJECTIVES`. 이 결과는 exposed historical surface의 탐색 결과이며 fresh confirmation이 아니다.

명목 점수 환산은 공식 기대값이 아니다. 기존 환산식이 보정된 구간보다 이번 same-season ΔRMSE가 수십 배 커서 극단 외삽이며, 아래 `raw 예상점수`와 `transport 보정점수`는 방향성 진단용 숫자로만 보존한다.

| 후보 | pooled ΔRMSE | Sep-Oct | Jul-Aug | Nov-Dec | raw 예상점수 | transport 보정점수 | gate |
|---|---:|---:|---:|---:|---:|---:|---|
| P2_V8_GROUP_BALANCED_L2_RAW_RESIDUAL | -1.645609261 | -3.955038622 | +0.411513852 | +1.510579689 | +24.920033 | +24.798351 | NO_GO |
| P2_V8_GROUP_BALANCED_L1_RAW_RESIDUAL | -1.770951382 | -4.092515637 | +0.325289755 | +1.322777484 | +26.447607 | +26.325925 | NO_GO |

## 구조와 중복 배제

공개층만으로 만든 고정 55개 특징에서 raw-Celsius 선형보간 잔차를 학습했고, layer×등록창×KST-day가 동일 위험을 갖도록 sample weight를 고정했다. DINEOF/GP/CatBoost, soft benefit gate, PAVA/isotonic, rank·season·bin search는 사용하지 않았다.

행 제거는 하지 않았다. primary L2가 strict gate에 실패한 경우에만 결과 전에 등록한 L1 robust-loss 후보를 한 번 실행했다.

## 검증 경계

2024 Sep-Oct 61일과 2025 Jul-Aug/Nov-Dec는 과거 연구에서 이미 노출됐다. 따라서 fold, layer, paired KST-day bootstrap, 점수 환산은 후보 폐쇄와 우선순위 판단용 탐색 증거일 뿐 독립 확인이 아니다.

fits=6; official/test/sample/query/hidden rows=0; submission CSV=0; upload=0.
