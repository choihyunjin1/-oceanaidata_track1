# P2 v41 all-Linear WeightNorm DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_WEIGHT_NORMALIZATION`. pooled delta RMSE `-0.049634805 C`, canonical nominal `+0.622794` points, transport `+0.501112` points.

fold delta RMSE: Sep-Oct `-0.080188546`, Jul-Aug `-0.011360068`, Nov-Dec `-0.011430515`.

prospective fold x layer gate: `False`, non-harm `6/9`, max cell `+0.032314734 C`.

Exact v13의 다섯 Linear에 learned per-output magnitude/direction WeightNorm 재매개화만 적용했다. Salimans and Kingma (NIPS 2016)는 동기만 제공하며 P2 성능 근거가 아니다. v27 spectral code/tolerance/power iteration을 재사용하지 않았다. sweep/router/ensemble/row deletion/official/hidden/CSV/upload=0.
