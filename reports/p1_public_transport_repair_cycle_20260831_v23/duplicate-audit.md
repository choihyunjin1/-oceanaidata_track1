# P1 v23 stable-top-k lineage audit

## 결론

`P1_2_HIST_GBDT_OOF_STACK_UNION`을 안정 순위 기반 top-k add-only 규칙으로 바꾸는 v23은 현재 **실행 불가(NO_GO)** 이다. 과거 실행은 내부 전체 ΔF1 `+0.0013809855753390554`였지만 Q3는 `0`, Q4만 `+0.003432580085375281`였고, 공식 Public 결과는 기존 최고점과 동률이었다. 더 중요한 blocker는 prefix별 연속 점수, full-train 연속 점수, 직렬화된 모델 checkpoint, 정확한 feature hash가 저장되지 않았다는 점이다.

과거 exact HGB 모델과 threshold를 다시 실행하는 것은 금지되어 있다. 공식 출력에서 추가된 4행을 보고 지금 `k=4`를 택하면 outer/official 결과를 본 뒤의 사후 선택이므로 prospective top-k 규칙이 아니다. 따라서 v23은 fit 0, candidate 0, official/hidden/CSV/upload/lock 0으로 닫는다.

## 중복 및 비중복 판단

- 과거 exact HGB model/threshold 재실행: **금지된 중복**. 이미 노출된 `HistGradientBoostingClassifier`와 inner threshold grid를 그대로 다시 계산한다.
- 연속 score의 환경별 순위 안정성을 요구하는 top-k 구조: 개념적으로는 과거 threshold union과 다르지만, 필요한 score lineage가 없으므로 현재 구현 가능한 후보가 아니다.
- `k=4`: **사후적 규칙**. binary deployment 결과의 4 additions와 공식 tie를 이미 관측한 뒤 정해지는 값이므로 봉인할 수 없다.

## 필요한 prospective evidence

새로운 독립 cycle에서 아래를 결과 노출 전에 저장·봉인해야만 이 구조를 평가할 수 있다.

1. Q2-prefix→Q3 및 Q2+Q3-prefix→Q4의 keyed continuous outer scores.
2. station, layer, chronological time block별 score rank-stability table.
3. exact feature/schema hashes와 bit-reproducible model checkpoint 또는 keyed full-train deployment scores.
4. outer label과 공식 결과를 보기 전에 정한 top-k 및 tie-break.
5. P1 prospective transport calibration에 따른 raw gate와 calibrated expected point delta `>=0.01`.

현재 자료에는 1–4가 없다. 따라서 환경 안정성, block bootstrap, worst-slice, addition precision을 독립 재계산할 수 없으며, 통계 gate를 통과했다고 주장할 수 없다.

## 출처 및 해석

- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11:2079–2107. 후보·hyperparameter 선택과 성능평가를 같은 정보로 수행할 때 selection bias가 생긴다는 근거다. https://www.jmlr.org/papers/v11/cawley10a.html
- scikit-learn `TimeSeriesSplit` 공식 문서. 시간순 데이터에서 미래 관측으로 과거를 학습하거나 평가하지 않도록 prefix train/test 순서를 유지해야 한다. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Politis & Romano (1994), *The Stationary Bootstrap*, JASA. 시간 의존성을 보존하는 block-resampling 근거지만, v23은 keyed score가 없어 bootstrap 자체를 수행하지 않았다. https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870

이 문서의 연구 문헌은 설계 원칙만 뒷받침한다. v23의 NO_GO 결론은 저장소의 source runner, aggregate internal result, artifact inventory를 직접 대조한 재현성 감사에 근거한다.
