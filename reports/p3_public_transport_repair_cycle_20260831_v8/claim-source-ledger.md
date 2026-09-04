# P3 v8 claim-source ledger

| Claim | Primary or authoritative source | Use in v8 | Limitation |
|---|---|---|---|
| Ordinary CV is biased under covariate shift; validation should reflect deployment covariates. | Sugiyama, Krauledat, Müller, JMLR 2007, https://jmlr.org/papers/v8/sugiyama07a.html | Uses the frozen high-wave/rising selection-matched historical cohort. | Exact density ratios were not identifiable from one Public score. |
| Covariate-shift uncertainty can be handled with weighted calibration when unlabeled deployment covariates are available. | Tibshirani et al., NeurIPS 2019, https://papers.nips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html | Motivates deployment-support abstention and explicit transport uncertainty. | v8 uses a conservative empirical Public penalty, not formal conformal coverage. |
| Selective regression can improve aggregate error while harming subgroups. | Shah et al., ICML 2022, https://proceedings.mlr.press/v162/shah22a.html | station×lead worst-case is a hard gate. | Stations are operational groups, not protected groups. |
| Average performance can hide failures on atypical groups. | Sagawa et al., ICLR 2020, https://arxiv.org/abs/1911.08731 | station×window group bootstrap accompanies episode bootstrap. | v8 is an audit, not group-DRO training. |
| v5 ExtraTrees reversed from -0.0045586m internally to +0.015723m on Public. | `reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json` | Supplies the 0.321905690-point Public reversal penalty. | One adverse P3 submission is an empirical guardrail, not a confidence bound. |
