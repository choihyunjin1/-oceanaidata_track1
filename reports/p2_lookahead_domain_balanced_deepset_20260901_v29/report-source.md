# P2 v29 fixed Lookahead-AdamW DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_LOOKAHEAD_OPTIMIZER`. pooled ΔRMSE `-0.048278095 C`, canonical nominal `+0.605771`점, transport `+0.484089`점.

fold ΔRMSE: Sep-Oct `-0.076382591`, Jul-Aug `-0.018288200`, Nov-Dec `-0.006208503`.

prospective fold×layer gate: `False`, non-harm `6/9`, max cell `+0.022354348 C`.

v13 science를 고정하고 AdamW trajectory에 fixed Lookahead(k=5, alpha=0.5)만 적용했다. Zhang et al. (NeurIPS 2019)은 optimizer 동기만 제공하며 P2 성능 근거가 아니다. sweep/scheduler/router/posthoc ensemble/official/hidden/CSV/upload=0.
