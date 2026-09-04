# P2 v33 virtual-domain MLDG DeepSets

## 결론

상태: `EXPLORATORY_NO_GO_VIRTUAL_DOMAIN_MLDG`. pooled delta RMSE `-0.050293901 C`, canonical nominal `+0.631064` points, transport `+0.509382` points.

fold delta RMSE: Sep-Oct `-0.076386982`, Jul-Aug `-0.030886082`, Nov-Dec `-0.014201201`.

prospective fold x layer gate: `False`, non-harm `6/9`, max cell `+0.011366274 C`.

Exact v13 SmoothL1 pipeline에 deterministic layer x month virtual meta-test와 one differentiable base-LR inner step만 추가했다. Li et al. (AAAI 2018)은 domain-shift simulation 동기만 제공하며 P2 성능 근거가 아니다. sweep/router/ensemble/row deletion/official-feedback selection/official/hidden/CSV/upload=0.
