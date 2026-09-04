# P2 v44 target-layer GradNorm balanced DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_TARGET_LAYER_GRADNORM`. pooled delta RMSE `-0.049641308 C`, canonical nominal `+0.622876` points, transport `+0.501194` points.

prospective fold x layer gate: `False`, non-harm `6/9`, max cell `+0.043823672 C`.

Exact v13 architecture에서 target-layer task weight만 fixed GradNorm alpha 1.5로 학습했다. sweep/projection/sign mask/router/ensemble/row deletion/Public selection/official/hidden/CSV/upload=0.
