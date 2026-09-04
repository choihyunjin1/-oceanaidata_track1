# P2 v52 score-priority third moment + input gradient

## 결론

상태: `SCORE_PRIORITY_PASS_EXPLICIT_STABILITY_RISK`. pooled RMSE `3.033063006 C`, ΔRMSE `-0.052651613 C`, nominal `+0.660648`점, transport `+0.538966`점.

Score gate `True`, stability diagnostic `False`. Worst fold-layer: `2025_nov_dec/L4` `+0.019484292 C`.

v50의 masked signed third-central-moment pooling과 v23의 observed-public-temperature input-gradient L2(lambda=0.01)를 사전 고정 결합했다. 배포 observations.csv와 truth-free 파생 scoring frame만 사용하며 scratch 9 fits다. 외부/사전학습/official/test/sample/baseline/query/hidden/CSV/upload=0.
