# P2 v34 gradient-centralized DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_GRADIENT_CENTRALIZATION`. pooled delta RMSE `-0.051833901 C`, canonical nominal `+0.650387` points, transport `+0.528705` points.

fold delta RMSE: Sep-Oct `-0.080957244`, Jul-Aug `-0.023523969`, Nov-Dec `-0.009212516`.

prospective fold x layer gate: `False`, non-harm `6/9`, max cell `+0.020429340 C`.

Exact v13 backward 이후 Linear weight gradient의 입력축 평균만 0으로 만들고 AdamW를 한 번 적용했다. Yong et al. (ECCV 2020)은 optimization 동기만 제공하며 P2 성능 근거가 아니다. coefficient/second loss/task split/sweep/router/ensemble/official-feedback selection/official/hidden/CSV/upload=0.
