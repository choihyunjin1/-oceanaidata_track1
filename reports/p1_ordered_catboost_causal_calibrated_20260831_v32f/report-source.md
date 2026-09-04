# P1 v32f prefix-calibrated 독립 후보 결론

## 결론

**terminal `NO_GO_INTERNAL_GATE`; 공식 materializer를 준비하지 않습니다.** v32a의 고정 threshold 0.80 저재현율을 outer fold 직전 45일 calibration으로 교정했지만, Q2/Q3/Q4 모두 frozen tabular reference보다 낮았고 Q3+Q4 E150 비교의 bootstrap 구간도 전부 음수였습니다. outer 결과를 본 threshold·모델 재조정은 하지 않았습니다.

- exactly-once fit: 3회, 총 249.402초
- 공식 test/sample/submission/hidden/upload 접근: 모두 0
- Q3+Q4 후보 F1: **0.789402**
- Q3+Q4 E150 F1: **0.906804**
- ΔF1: **-0.117402**
- paired event/day bootstrap CI90: **[-0.168205, -0.071284]**, P(improve)=0
- empirical score 환산: **25.7890점**, CI90 **[24.4388, 27.0147]점**; 현재 best 28.909341점보다 낮음

점수 환산은 historical F1 delta에 과거 empirical slope를 적용한 정보성 수치이며 공식 점수 보장이 아닙니다.

## 봉인된 교정 절차

각 outer fold의 eligible training prefix 끝 45일을 calibration으로 떼고, 그보다 앞선 행만 CatBoost 학습에 사용했습니다. offline feature의 장기 의존성을 보호하기 위해 fit과 calibration 사이에 14일 purge를 뒀습니다. calibration probability/label에서만 `0.05..0.95, step 0.01` grid의 binary F1을 최대로 하는 threshold를 선택했고 동률은 0.5에 가장 가까운 값으로 결정했습니다.

| Fold | fit rows | calibration rows | threshold | calibration F1 | outer candidate F1 | tabular F1 | ΔF1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 | 300,620 | 37,777 | 0.06 | 0.402835 | 0.681611 | 0.775850 | -0.094239 |
| Q3 | 389,335 | 63,654 | 0.68 | 0.781813 | 0.727719 | 0.901888 | -0.174168 |
| Q4 | 543,819 | 80,674 | 0.40 | 0.839152 | 0.867362 | 0.903772 | -0.036410 |

Q2 threshold 0.06은 recall 0.8069까지 복원했지만 precision이 0.5900으로 무너졌습니다. Q3 threshold 0.68은 precision 0.9796이나 recall 0.5789로 v32a와 같은 저재현율이 남았습니다. Q4는 precision 0.9277, recall 0.8144, F1 0.8674로 가장 근접했으나 tabular와 E150 모두 이기지 못했습니다. fold별 최적 threshold의 0.06/0.68/0.40 분산은 45일 calibration의 regime transport 불안정성 자체를 보여줍니다.

pooled 후보 F1은 **0.745112**, tabular reference는 **0.860248**, ΔF1은 **-0.115136**입니다. pooled bootstrap CI90 **[-0.165454, -0.072504]**와 P(improve)=0으로 우연한 한 fold 실패가 아닙니다.

## Gate

- 각 Q2/Q3/Q4 tabular 대비 nonnegative: FAIL
- pooled delta positive: FAIL
- pooled CI90 lower positive: FAIL
- Q3+Q4 E150 delta positive: FAIL
- Q3+Q4 E150 CI90 lower positive: FAIL
- runtime ≤ 1,200초: PASS
- official access 0: PASS

따라서 exact `Ordered CatBoost + event/day weights + trailing-45-day threshold calibration` recipe는 닫습니다. 이 결과로 제출 기회를 소비하지 않습니다.

## QA·재현

- config: `configs/experiments/p1_ordered_catboost_causal_calibrated_20260831_v32f.json`
- runner: `scripts/run_p1_ordered_catboost_causal_calibrated_20260831_v32f.py`
- focused pytest: 4 PASS
- py_compile/Ruff: PASS
- independent QA: 35/35 PASS
- historical OOF: 421,032행, SHA-256 `220ddf4289dd0554d0031c616040c84f76667835d511bf45d5f44a26ede15661`
