# P2 v30 fixed IRMv1 layer-month invariant DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_IRMV1_LAYER_MONTH_INVARIANCE`. pooled ΔRMSE `-0.049380924 C`, canonical nominal `+0.619609`점, transport `+0.497927`점.

fold ΔRMSE: Sep-Oct `-0.079247120`, Jul-Aug `-0.013811506`, Nov-Dec `-0.008841474`.

prospective fold×layer gate: `False`, non-harm `6/9`, max cell `+0.029760029 C`.

v13 science를 고정하고 layer×calendar-month group risk의 fixed classifier-scale gradient-square IRMv1 penalty(coefficient=1.0)만 추가했다. Arjovsky et al. (2019)은 representation 동기만 제공하며 P2 성능 근거가 아니다. anneal/sweep/router/ensemble/official/hidden/CSV/upload=0.
