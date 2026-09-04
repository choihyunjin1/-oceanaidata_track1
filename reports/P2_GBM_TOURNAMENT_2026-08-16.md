# P2 GBM 계열 구조 비교

## 결론

동일한 69,850개 target-proxy OOF 행과 public-only phase 81개 특징에서 여섯 GBM 구조를 400 boosting iteration으로 비교했다. 단독 모델은 LightGBM ExtraTrees형이 RMSE `0.816096°C`로 가장 낮았지만, 기존 deep stack을 보완하는 모델로는 layerwise CatBoost가 가장 적합했다.

Layerwise CatBoost를 deep stack과 층별로 결합하면 fitted OOF RMSE는 `0.745814→0.743860°C`, 다른 두 블록에서만 가중치를 정하는 LOBO RMSE는 `0.775660→0.774577°C`로 감소했다. 다만 LOBO 개선의 KST-day bootstrap 90% CI는 `[-0.005133,+0.003215]°C`로 0을 포함한다. 따라서 현재 제출 1순위는 기존 `P2_DEEP_STACK_V1.csv`를 유지하고, 다음 파라미터 최적화 대상은 layerwise CatBoost로 정한다.

## 고정 비교 계약

- 평가 지표: layer 2·3·4 통합 row-level RMSE(°C)
- 검증 행: 2024년 9~10월, 2025년 7~8월, 2025년 11~12월의 69,850개 키
- 입력: 공개층 temp·psal·depth, 수심 선형보간 baseline, 시간·조석·public-layer 동역학 특징
- 금지: target layer 2·3·4의 가림 temp·psal 입력, 외부 관측값, hidden 정답
- 공통 예산: 400 boosting iteration; 이 단계에서는 backend별 하이퍼파라미터 최적화 없음
- 비교 계열: LightGBM GBDT, LightGBM ExtraTrees, LightGBM DART, XGBoost hist, CatBoost pooled, CatBoost layerwise

## 결과

| LOBO pair 순위 | 모델 | 단독 RMSE | fitted pair RMSE | LOBO pair RMSE | LOBO deep 대비 |
|---:|---|---:|---:|---:|---:|
| 1 | CatBoost layerwise | 0.833549 | 0.743860 | 0.774577 | -0.001083 |
| 2 | CatBoost pooled | 0.843419 | 0.744670 | 0.782503 | +0.006843 |
| 3 | LightGBM DART | 0.829720 | 0.745814 | 0.784244 | +0.008583 |
| 4 | LightGBM GBDT | 0.840923 | 0.742085 | 0.790544 | +0.014884 |
| 5 | LightGBM ExtraTrees | 0.816096 | 0.745579 | 0.793065 | +0.017405 |
| 6 | XGBoost hist | 0.965441 | 0.745796 | 0.806669 | +0.031009 |

LightGBM GBDT의 fitted pair가 가장 낮지만 LOBO에서 크게 역전된다. 이는 동일 OOF에 맞춘 blend weight의 과적합 신호다. 반면 CatBoost layerwise는 fitted와 LOBO에서 모두 deep 기준을 소폭 개선한 유일한 계열이다.

## CatBoost 층별 보완 구조

| 목표층 | Deep weight | CatBoost weight |
|---:|---:|---:|
| 2 | 0.719160 | 0.280840 |
| 3 | 0.720740 | 0.279260 |
| 4 | 1.000000 | 0.000000 |

CatBoost의 이득은 layer 2와 3에 집중되며 layer 4에서는 deep stack을 그대로 유지한다. 이 구조는 하나의 pooled CatBoost보다 계층별 수온 곡률 차이를 더 잘 분리한다.

## 독립 검증

- OOF 행 69,850개, OOF 키 69,850개로 중복 없음
- 여섯 모델 모두 동일 키·동일 truth와 정렬됨
- 저장 RMSE와 독립 재계산 RMSE가 허용오차 `1e-12` 이내 일치
- 층별 pair 투영 최대 절대오차 0
- 여섯 단독 제출과 deep+CatBoost 연구 challenger 모두 26,061행·키·순서·유한 범위 검증 통과
- 원자료 행과 hidden 정답을 보고서 또는 Git에 기록하지 않음

## 다음 파라미터 탐색

다음 세대는 layerwise CatBoost를 주 대상으로 한다. 탐색 축은 층별 boosting horizon과 learning rate, depth, `l2_leaf_reg`, `random_strength`, bootstrap 설정, `rsm`이다. 모든 선택은 각 outer block 밖의 blocked inner validation에서 수행하고, 최종 비교는 같은 69,850행 standalone·deep pair·LOBO pair로 고정한다.

LightGBM ExtraTrees는 단독 성능 comparator로 유지한다. 이번 결과만으로 ExtraTrees를 최종 앙상블에 추가하면 LOBO가 악화하므로 바로 채택하지 않는다.

## 산출물 상태

- 결과·OOF·모델: 로컬 ignored `artifacts/p2_gbm_family_tournament_v1`
- 연구 challenger: `submissions/p2/P2_DEEP_GBM_RESEARCH_V1.csv`
- 업로드: 실행하지 않음
- 현재 공식 제출 1순위: 기존 `submissions/p2/P2_DEEP_STACK_V1.csv`
