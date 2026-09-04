# Claim-source ledger — P3 continuous-energy KMA factor v14b

| Claim | Evidence | Status |
|---|---|---|
| The 2D alpha grid was not rerun because a 90,601-pair exhaustive surface already exists. | `reports/p3_kma_alpha2d_nested_cycle_20260831_v14/cancellation-receipt.json` and `p3_kma_alpha_surface_sweep_20260829_v1` provenance | Verified; 0 fits in cancelled v14 |
| The candidate is continuous, low-DOF, and has no hard support router. | Sealed config and runner formula `alpha24=0.20+0.40*ECDF` | Verified |
| Each outer block uses only feature cases ending at least 78 hours before block start. | `candidate.ecdf_calibration_receipts` plus independent QA boundary checks | Verified |
| The candidate improved pooled historical RMSE but did not pass the family-aware public-transport gate. | `artifacts/p3_kma_continuous_energy_factor_cycle_20260831_v14b/result.json` | Verified |
| Official inputs, hidden truth, CSV materialization, and upload remained zero. | Result `data_access`, outputs, and independent QA | Verified |

Primary methodological context remains the preregistered transport/selection evidence used by the preceding P3 cycle: Sugiyama et al. (JMLR 2007), Tibshirani et al. (NeurIPS 2019), Shah et al. (ICML 2022), and Sagawa et al. (2019). No external claim changes the sealed gate or the observed result.
