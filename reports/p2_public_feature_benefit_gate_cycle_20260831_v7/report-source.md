# P2 public-feature benefit gate cycle 20260831 v7

## 결론

`P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE` 한 개가 새 보수적 Public 수송 gate를 통과했고 제출 CSV가 준비됐다. Oct nested ΔRMSE는 -0.013071643°C, 90% block-bootstrap 상단은 -0.010794735°C, raw 기대점수 +0.135447점, 최악 관측 수송 penalty 0.121682점 차감 후 +0.013765점이다. 기준은 inclusive `>= +0.01점`이다.

RandomForest 후보는 raw +0.055417점, calibrated -0.066265점으로 탈락했다. PASS 하나만 full fit/materialize했고 공식 test_index 26,061행은 내부 PASS 뒤에만 읽었다. hidden truth와 upload는 0건이다.

## 내부 검증

| 후보 | Oct ΔRMSE | Oct L2/L3/L4 | active share | raw 점수 | calibrated 점수 | gate |
|---|---:|---|---:|---:|---:|---|
| ExtraTrees benefit gate | -0.013071643 | -0.040405 / -0.030541 / -0.014580 | 0.517655 | +0.135447 | +0.013765 | PASS |
| RandomForest benefit gate | -0.007837443 | -0.040905 / -0.022529 / -0.006993 | 0.578304 | +0.055417 | -0.066265 | FAIL |

Sep 2024에서 sealed HGB correction이 reference보다 제곱오차를 줄였는지를 public hydrographic lag/state features로 학습하고 Oct 2024를 untouched nested test로 사용했다. Jul-Aug/Nov-Dec에는 official 계절과 불일치하므로 champion으로 exact abstain한다. threshold 0.5, tree 수, leaf 크기는 결과 전에 고정했다.

## 제출본

- CSV: `artifacts/p2_public_feature_benefit_gate_cycle_20260831_v7/submission/P2_V7_EXTRATREES_PUBLIC_BENEFIT_GATE/P2_submission.csv`
- SHA-256: `c6f2a7e02ff3e5064ec653af0a52b117cbf8ae49d80e651a2a96276190f4f620`
- rows: 26,061
- pre-projection active rows: 11,286
- 내부 candidate fits 2 + PASS full fit 1 = 총 3 fits

## 근거

- Cawley & Talbot (2010), nested evaluation/selection bias: https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf
- Sugiyama et al. (2007), covariate shift and importance-aware validation: https://jmlr.csail.mit.edu/papers/volume8/sugiyama07a/sugiyama07a.pdf
- Huber (1964), robust M-estimation: https://doi.org/10.1214/aoms/1177703732
