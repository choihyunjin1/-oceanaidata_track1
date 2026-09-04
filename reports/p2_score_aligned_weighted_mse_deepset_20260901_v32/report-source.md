# P2 v32 score-aligned weighted-MSE DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_SCORE_ALIGNED_WEIGHTED_MSE`. pooled ΔRMSE `-0.050116921 C`, canonical nominal `+0.628844`점, transport `+0.507162`점.

fold ΔRMSE: Sep-Oct `-0.079823547`, Jul-Aug `-0.017169040`, Nov-Dec `-0.004189122`.

prospective fold×layer gate: `False`, non-harm `7/9`, max cell `+0.023443899 C`.

v13 task pipeline에서 SmoothL1을 uncapped weighted MSE로만 교체했다. Gneiting and Raftery (JASA 2007)는 scoring-rule 동기만 제공하며 P2 성능 근거가 아니다. extra coefficient/clipping/winsor/downweight/delete/sweep/router/ensemble/official/hidden/CSV/upload=0.
