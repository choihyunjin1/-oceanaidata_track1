# P2 v46 hidden-activation LayerNorm DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_HIDDEN_ACTIVATION_LAYERNORM`. pooled delta RMSE `-0.046730641 C`, canonical nominal `+0.586354` points, transport `+0.464672` points.

prospective fold x layer gate `False`, non-harm `8/9`, max cell `+0.006441084 C`.

Exact v13 five-Linear DeepSets의 네 hidden Linear 뒤, ReLU 전에 affine LayerNorm(32, eps=1e-5)만 추가했다. v45/v45c 비교는 terminal 후 ledger 진단만 수행했고 selection/retune/router/ensemble은 없다. official/test/sample/hidden/query/CSV/upload=0.
