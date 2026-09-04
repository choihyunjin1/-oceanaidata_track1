# P2 v31 within-layer month-adversarial DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_WITHIN_LAYER_MONTH_ADVERSARY`. pooled ΔRMSE `-0.040325950 C`, canonical nominal `+0.505991`점, transport `+0.384309`점.

fold ΔRMSE: Sep-Oct `-0.056291888`, Jul-Aug `-0.048588071`, Nov-Dec `+0.010417393`.

prospective fold×layer gate: `False`, non-harm `7/9`, max cell `+0.020189326 C`.

v13 task path를 그대로 두고 pooled token latent에만 layer별 12-way month classifier와 fixed GRL(0.1)을 추가했다. Ganin et al. (JMLR 2016)은 representation 동기만 제공하며 P2 성능 근거가 아니다. P1 v24 code/output 재사용0, schedule/sweep/router/ensemble/official/hidden/CSV/upload=0.
