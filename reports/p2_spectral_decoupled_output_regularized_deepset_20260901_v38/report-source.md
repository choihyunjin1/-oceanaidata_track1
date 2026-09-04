# P2 v38 normalized-residual spectral decoupling

## 결론

상태: `EXPLORATORY_NO_GO_SPECTRAL_DECOUPLED_OUTPUT`. pooled delta RMSE `-0.050566638 C`, canonical nominal `+0.634486` points, transport `+0.512804` points.

fold delta RMSE: Sep-Oct `-0.080197120`, Jul-Aug `-0.018130068`, Nov-Dec `-0.008126075`.

prospective fold x layer gate: `False`, non-harm `6/9`, max cell `+0.027090475 C`.

Exact v13 weighted SmoothL1에 `0.5*0.01*weighted_mean(prediction^2)` normalized-residual output penalty만 추가했다. Pezeshki et al. (NeurIPS 2021)은 동기만 제공하며 P2 regression 성능 근거가 아니다. P1-v46 인접성은 공개했고 P1 결과 기반 selection은 0이다. sweep/router/ensemble/official/hidden/CSV/upload=0.
