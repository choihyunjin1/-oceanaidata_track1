# P3 Public transport repair cycle v8

## 결론

- Public-risk calibrated PASS: **0/3**
- 제출 CSV: **0개**, upload 0
- 직전 v5 ExtraTrees의 내부 개선이 Public에서 반전된 실측값을 0.321905690점 페널티로 직접 차감했다.

| candidate | delta RMSE(m) | episode CI90 upper | group CI90 upper | worst station-lead | raw conservative pts | calibrated pts | PASS |
|---|---:|---:|---:|---:|---:|---:|---|
| P3_1_SELECTION_MATCHED_HUBER_CALIBRATOR | 0.024268 | 0.045781 | 0.039670 | 0.052365 | 0.000000 | -0.321906 | False |
| P3_2_SELECTION_MATCHED_RIDGE_CALIBRATOR | 0.007232 | 0.016801 | 0.018164 | 0.028732 | 0.000000 | -0.321906 | False |
| P3_3_SELECTION_MATCHED_SHALLOW_ET_CALIBRATOR | 0.004666 | 0.013212 | 0.016100 | 0.026098 | 0.000000 | -0.321906 | False |

## 판정 해석

- raw conservative gain은 78시간 독립 episode bootstrap의 90% CI 상단으로 계산했다.
- station×forward-window group bootstrap과 station×lead 최악 회귀는 별도 hard gate다.
- Public 결과를 학습 label로 사용하지 않았고, 과거 후보의 예측-실측 점수 잔차는 승격 페널티로만 사용했다.

## 1차 출처

- Sugiyama, Krauledat, Müller (JMLR 2007), Importance Weighted Cross Validation: https://jmlr.org/papers/v8/sugiyama07a.html
- Tibshirani et al. (NeurIPS 2019), Conformal Prediction Under Covariate Shift: https://papers.nips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html
- Shah et al. (ICML 2022), Selective Regression under Fairness Criteria: https://proceedings.mlr.press/v162/shah22a.html
- Sagawa et al. (ICLR 2020), Distributionally Robust Neural Networks for Group Shifts: https://arxiv.org/abs/1911.08731

## 경계

- hidden truth 0, upload 0. PASS가 없으면 official test/index/champion 값도 0행 읽는다.
- v4-v7 artifact와 다른 문제 lane은 변경하지 않았다.
