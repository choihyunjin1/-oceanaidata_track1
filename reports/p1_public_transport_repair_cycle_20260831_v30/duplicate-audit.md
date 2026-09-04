# P1 v30 duplicate and leakage audit

## 결론

`P1_1_LABEL_FREE_RELIABILITY_GUARDED_LABEL_SHIFT_EM`은 v28의 동결된 세 logit calibrator, inner threshold, Saerens EM을 그대로 유지하면서 **label-free score reliability**로만 station×layer mask와 KST-day cap을 정하는 신규 보호 계층이다. Historical execution, attempt lock, official/hidden read, CSV, upload는 모두 0으로 봉인한다.

## v29와의 비중복 경계

- v29는 inner group truth로 proposed-addition precision과 within-group ΔF1을 계산해 group eligibility를 정한다.
- v30은 group truth, group F1, group precision을 함수 인자로 받지 않는다. Prefix inner rows에서 calibrator posterior와 frozen three-source mean 사이의 score discrepancy만 사용한다.
- v30은 support `>=256`, group discrepancy mean의 one-sided 90% bound가 prefix global absolute-discrepancy q90 이내인지, 그리고 row corrected-score margin lower bound가 0 이상인지만 본다.
- 일별 0.5% cap도 margin lower bound와 원행순만 사용한다. Outer label, v28 실패 station-layer, Q3/Q4 결과는 rule, support, threshold에 들어가지 않는다.

## 기존 family와의 경계

- v28은 label-shift correction 뒤 group/day safety를 적용하지 않았다.
- 과거 dynamic peer reliability는 고정 24h peer coherence 규칙과 label-scored promotion이었다. v30은 windowed peer rule을 만들지 않고 이미 frozen된 세 source probability의 동시 score disagreement만 측정한다.
- HGB top-k, event bank, quarter selector, outer-result threshold update, station/layer별 label tuning은 모두 금지한다.

## 봉인된 계약

한 outer prefix당 v28 logistic calibrator 1회, 총 2 fits만 허용한다. Inner threshold와 EM hyperparameters는 v28과 bit-exact다. Reliability constants는 minimum group rows `256`, one-sided z `1.2815515655446004`, global discrepancy quantile `.90`, daily cap `.005`다. Unknown/low-support/unreliable group은 fail-closed하고, margin tie는 원래 행 순서로 푼다.

Prospective calibration v3 SHA `0f448207...21a10`, penalty `0.005383691점`, inclusive raw gate `0.015383691점`, calibrated gate `+0.01점`, Q3/Q4 비악화, dependent bootstrap CI90 low>0와 P(improve)>=.8, add-only 및 모든 slice safety gate를 유지한다. Q3/Q4는 reused development surface이므로 향후 실행되더라도 independent confirmation으로 부르지 않는다.
