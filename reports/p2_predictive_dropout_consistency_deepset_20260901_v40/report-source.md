# P2 v40 predictive-dropout consistency DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_PREDICTIVE_DROPOUT_CONSISTENCY`. pooled delta RMSE `-0.048630479 C`, canonical nominal `+0.610192` points, transport `+0.488510` points.

fold delta RMSE: Sep-Oct `-0.074094325`, Jul-Aug `-0.029877873`, Nov-Dec `-0.007151311`.

prospective fold x layer gate: `False`, non-harm `8/9`, max cell `+0.019217156 C`.

Exact v13에 fixed dropout 0.1과 two-pass fixed-variance Gaussian predictive-consistency만 inseparable intervention으로 추가했다. Liang et al. (NeurIPS 2021)은 동기만 제공하며 P2 regression 성능 근거가 아니다. P1-v39 adjacency는 공개했고 code/result/gate transfer는 0이다. sweep/router/ensemble/row deletion/official/hidden/CSV/upload=0.
