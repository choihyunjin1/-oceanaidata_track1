# Claim-to-source ledger

## Internal evidence claims

| Claim | Evidence | Status |
|---|---|---|
| P1 v30 official score is F1 `0.798819`, `27.986329점`, a `-0.923012점` regression from `28.909341점` | live OCN-01 problem card and submission-management receipt at 2026-08-31 21:22 KST; `official-submission-receipt.json` | Verified |
| P3 v19 official score is RMSE `0.589840`, `23.971758점`, `+0.017717점` over the immediate comparator but below personal best | live OCN-03 problem card and submission-management receipt at 2026-08-31 21:23 KST; `official-submission-receipt.json` | Verified |
| P2 v7 could not be uploaded because the site displayed remaining submissions `0/3` and disabled both controls | live OCN-02 problem card; `official-submission-receipt.json` | Verified blocker |
| P1 v30 passes every frozen strict internal gate | `artifacts/p1_public_transport_repair_cycle_20260831_v30/result.json` SHA `5f478359…826`; pooled ΔF1 `+0.001820930`, Q3 positive, Q4 no-op, bootstrap CI90 low positive | Verified |
| P1 raw/penalty/calibrated values are `0.048396908/0.005383691/0.043013217` | internal result, public-transport calibration v3, and `reports/p1_public_transport_repair_cycle_20260831_v30/postrun-qa.json` | Independently recomputed |
| P1 CSV has 169,011 binary finite rows in exact official key order | materialization receipt, validator, postrun QA, and root key-order check; CSV SHA `639c26cd…efa3` | Verified |
| P1 official EM target prevalence reached `0.999999` and creates transport risk | `reports/p1_public_transport_repair_cycle_20260831_v30/materialization-result.json` | Verified warning; not a hidden-label result |
| P2 v7 has exactly one passing candidate | `artifacts/p2_public_feature_benefit_gate_cycle_20260831_v7/result.json` SHA `db35e824…a10` | Verified |
| P2 raw/penalty/calibrated values are `0.135447268/0.121682092/0.013765176` | result plus `reports/p2_public_feature_benefit_gate_cycle_20260831_v7/independent-root-ready-qa.json` | Independently recomputed |
| P2 CSV has 26,061 valid rows in official order | materialization receipt, root-ready QA, final cross-QA; CSV SHA `c6f2a7e0…620` | Verified |
| P3 v19 passes all sealed gates | `artifacts/p3_kma_continuous_wave_power_factor_cycle_20260831_v19/result.json` SHA `fba999a9…e5c` | Verified |
| P3 raw/penalty/calibrated values are `0.068090634/0.049586054/0.018504580` | result plus independent QA 15/15 and final cross-QA | Independently recomputed |
| P3 CSV has 1,200 valid rows in official order | delivery QA and final cross-QA; CSV SHA `b1b72f90…d0c4` | Verified |
| Hidden truth and uploads are zero | P1/P2/P3 result, materialization/delivery receipts, postrun/final cross-QA | Verified |

## Primary research sources

| Supported methodological claim | Primary source | Scope note |
|---|---|---|
| Finite-sample model-selection criteria can themselves be overfit and produce selection bias | Gavin C. Cawley and Nicola L. C. Talbot, “On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation,” JMLR 11, 2010. [JMLR](https://www.jmlr.org/papers/v11/cawley10a.html) | Supports adaptive-surface caveat; does not set the numeric gate |
| Covariate-shift-aware conformal inference requires explicit assumptions and weighting/density-ratio information | Ryan J. Tibshirani, Rina Foygel Barber, Emmanuel Candès, Aaditya Ramdas, “Conformal Prediction Under Covariate Shift,” NeurIPS 2019. [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html) | Supports transport-risk caution; current penalty is not a conformal guarantee |
| Domain-generalization comparisons depend on consistent model selection; strong baselines remain important | Ishaan Gulrajani and David Lopez-Paz, “In Search of Lost Domain Generalization,” ICLR 2021. [OpenReview](https://openreview.net/forum?id=lQdXeXDoWtI) | Supports explicit selection protocol; not evidence that these candidates generalize |
| Average loss can hide atypical-group failures; regularization matters for worst-group generalization | Shiori Sagawa, Pang Wei Koh, Tatsunori B. Hashimoto, Percy Liang, “Distributionally Robust Neural Networks for Group Shifts,” ICLR 2020. [OpenReview](https://openreview.net/forum?id=ryxGuJrFvS) | Supports group diagnostics; no group-DRO guarantee claimed here |
| Standard K-fold CV for autoregressive series is valid only under stated conditions such as uncorrelated errors | Christoph Bergmeir, Rob J. Hyndman, Bonsoo Koo, “A Note on the Validity of Cross-Validation for Evaluating Autoregressive Time Series Prediction,” CSDA 120, 2018. [DOI](https://doi.org/10.1016/j.csda.2017.11.003) | Supports explicit time-series assumptions and conservative blocking |
| Ignoring temporal/spatial/hierarchical dependence can underestimate predictive error | David R. Roberts et al., “Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure,” Ecography 40, 2017. [DOI](https://doi.org/10.1111/ecog.02881) | Supports blocked/group-aware validation |
| CV estimates a nuanced prediction target and standard CV intervals may under-cover because fold errors are dependent | Stephen Bates, Trevor Hastie, Robert Tibshirani, “Cross-Validation: What Does It Estimate and How Well Does It Do It?”, JASA 119, 2024. [DOI](https://doi.org/10.1080/01621459.2023.2197686) | Supports caution in interpreting bootstrap/CV uncertainty |

Search stopped after the named primary sources directly covered selection bias, covariate shift, domain/group shift, and structured/CV uncertainty. Additional sources would not change the pending integration decision.
