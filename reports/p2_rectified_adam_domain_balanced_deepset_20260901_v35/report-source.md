# P2 v35 fixed-RAdam DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_FIXED_RADAM`. pooled delta RMSE `-0.045479955 C`, canonical nominal `+0.570661` points, transport `+0.448979` points.

fold delta RMSE: Sep-Oct `-0.073544878`, Jul-Aug `-0.009898921`, Nov-Dec `-0.012061629`.

prospective fold x layer gate: `False`, non-harm `7/9`, max cell `+0.023676814 C`.

Exact v13 AdamW만 fixed RAdam(lr=.001, betas=.9/.999, eps=1e-8, decoupled WD=.0001)으로 교체했다. Liu et al. (ICLR 2020)은 optimization 동기만 제공하며 P2 성능 근거가 아니다. warmup/scheduler/sweep/router/ensemble/official-feedback selection/official/hidden/CSV/upload=0.
