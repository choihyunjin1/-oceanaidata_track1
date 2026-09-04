# P1 v32a 독립 후보 최종 결론

## 결론

**제출하지 않습니다.** `v32a Ordered CatBoost + event/day-balanced weights`는 현 챔피언이나 v30/v31 예측을 입력으로 쓰지 않은 독립 학습 후보였지만, 사전등록된 Q2/Q3/Q4 historical gate를 명확히 통과하지 못했습니다. exactly-once 3-fit은 284.673초에 종료됐고 공식 test/sample/submission/hidden 접근과 CSV 생성·업로드는 모두 0입니다.

Q3+Q4에서 후보 F1은 **0.731268**, frozen E150 raw reference는 **0.906804**로 차이는 **-0.175536 F1**입니다. event/day bootstrap 90% CI도 **[-0.245140, -0.110258]**로 전부 불리합니다. empirical Public score slope를 단순 적용한 정보성 환산은 **24.2439점**, 90% 구간 **[22.3940, 25.9789]점**으로 현 best 28.909341점보다 낮습니다. 이 환산은 공식 점수 보장이 아니라 historical delta의 크기를 점수 단위로 표현한 것입니다.

## 사전등록 모델과 실행

- 특징: immutable `train.csv`와 label-free offline feature cache 80열만 사용
- 모델: CPU `CatBoostClassifier`, `Ordered` boosting, depth 7, 240 iterations
- 학습 가중치: positive event와 normal station-layer-day의 inverse-square-root balance
- decoder: 고정 확률 threshold 0.80, incumbent union·veto·후처리 없음
- 검증: 7일 purge가 있는 2025 Q2/Q3/Q4 rolling-origin outer folds
- 불확실성: positive event + normal station-layer-day paired bootstrap 1,000회
- fit 수: 정확히 3회; Q2 64.583초, Q3 91.116초, Q4 99.502초

## Historical 결과

| Fold | 후보 F1 | tabular reference F1 | ΔF1 | 후보 양성률 |
|---|---:|---:|---:|---:|
| Q2 | 0.746161 | 0.775850 | -0.029689 | 2.4225% |
| Q3 | 0.715330 | 0.901888 | -0.186558 | 2.0313% |
| Q4 | 0.754350 | 0.903772 | -0.149422 | 2.3469% |
| pooled | 0.736293 | 0.860248 | -0.123955 | — |

pooled candidate는 precision **0.995331**, recall **0.584242**였습니다. 즉 모델은 오탐 44행으로 정밀도는 극단적으로 높았지만 양성 6,675행을 놓쳐 공식 binary F1 목적에 부적합했습니다. 실패 원인은 분산이나 한 fold 우연이 아니라 Q2/Q3/Q4 모두 음의 delta이고 pooled bootstrap 90% CI도 **[-0.168093, -0.080596]**인 구조적 recall 부족입니다.

## Gate 판정

- Q2/Q3/Q4 모두 tabular reference 이상: **FAIL**
- pooled tabular delta > 0: **FAIL**
- pooled bootstrap CI90 lower > 0: **FAIL**
- Q3+Q4 E150 delta > 0: **FAIL**
- Q3+Q4 E150 bootstrap CI90 lower > 0: **FAIL**
- runtime ≤ 1,200초: **PASS**
- official/hidden/test/sample/submission/upload 접근 0: **PASS**

결과를 본 뒤 threshold를 낮추거나 모델을 재실행하는 것은 exactly-once 계약을 위반하므로 수행하지 않았습니다. 이 exact `Ordered CatBoost + event/day weight + threshold 0.80` recipe는 terminal negative evidence로 닫습니다.

## 재현·QA

- config: `configs/experiments/p1_ordered_catboost_eventday_20260831_v32a.json`
- runner: `scripts/run_p1_ordered_catboost_eventday_20260831_v32a.py`
- focused tests: 4 PASS
- Ruff: runner/test/QA PASS
- independent QA: 25/25 PASS (`independent-qa.json`)
- historical OOF: 421,032행, SHA-256 `7f3db0fbe9cb8348a15526764b9c62a220add38a0064132b08f30e7ba8f9fe19`

공식 제출 파일은 만들지 않았으며 이 결과로 제출 기회를 소비할 근거가 없습니다.
