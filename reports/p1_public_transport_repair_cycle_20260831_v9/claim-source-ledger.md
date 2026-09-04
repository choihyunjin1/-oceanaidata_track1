# Claim–source ledger

| 주장 | 근거 | 적용 | 한계 |
|---|---|---|---|
| 시간자료는 미래를 학습하고 과거를 평가하는 일반 CV를 피해야 한다 | scikit-learn, [TimeSeriesSplit 공식 문서](https://scikit-learn.org/1.0/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Q2→Q3, Q2+Q3→Q4 prequential contract | 균등 간격/대표성 자체를 보장하지 않음 |
| calibrated score의 F1-optimal threshold는 최적 F1/2와 연결된다 | Lipton et al., [Thresholding Classifiers to Maximize F1 Score](https://pmc.ncbi.nlm.nih.gov/articles/PMC4442797/) | train anchor F1/2를 proposal precision 기준으로 사용 | row probability가 아닌 proposal posterior에 적용하므로 보수 LCB를 추가한 응용 |
| covariate shift 아래 모델 선택은 분포 차이를 명시해야 한다 | Sugiyama et al., [Covariate Shift Adaptation by Importance Weighted Cross Validation](https://jmlr.org/papers/v8/sugiyama07a.html) | local→Public 수송 오차를 별도 family-aware penalty로 취급 | 본 실험은 density-ratio IWCV를 직접 구현하지 않음 |
| dependent data에는 block resampling이 iid resampling보다 적합하다 | Politis & Romano, [The Stationary Bootstrap](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870) | KST calendar-day paired block bootstrap | stationary bootstrap 자체가 아니라 고정 day blocks의 실용 변형 |
| P1 현재 Public 기준은 F1 0.833548, 28.909341점이고 직전 add-only 후보는 동률 | `reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json` | comparator 및 prior tie 제외 | Public leaderboard만 관측됨 |
| hard conditional router의 경험적 penalty/raw gate는 0.321905690/0.331905690점 | `reports/public_transport_calibration_20260831_v2/calibration.json` | v9 family/tier gate | confidence interval이 아닌 관측 worst-residual guardrail |
| v9은 6 fits, additions 0, PASS 0 | `artifacts/p1_public_transport_repair_cycle_20260831_v9/result.json` | 최종 과학 결론 | 공식 분포 성능을 직접 측정하지 않음 |
| v9 config/runner family registration이 calibration v2와 일치 | `reports/p1_public_transport_repair_cycle_20260831_v9/independent-qa.json` | 재현성 및 access audit | 소프트웨어 QA는 과학적 효용을 대신하지 않음 |
