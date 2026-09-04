# Claim-to-source ledger

| Claim | Source | Relevance | Limitation |
|---|---|---|---|
| Signature features are ordered sample cross-moments for sequential data and converge to path-level features | Franz J. Király and Harald Oberhauser, *Kernels for Sequentially Ordered Data*, JMLR 20(31), 2019, [paper](https://www.jmlr.org/papers/v20/16-314.html) | Supports deterministic ordered level-2 cross-interactions | Does not imply P3 accuracy or Public transport |
| Continuous-path models directly address partially observed multivariate time series | Patrick Kidger, James Morrill, James Foster, Terry Lyons, *Neural Controlled Differential Equations for Irregular Time Series*, NeurIPS 2020, [paper](https://papers.neurips.cc/paper_files/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html) | Supports the path representation and explicit mask channels | v23 is deterministic and linear, not a Neural CDE |
| P3 already tried fixed-lag multi-output Ridge | `artifacts/p3_nlinear_station_ridge_residual_20260828_v1/result.json` | Distinguishes ordered iterated integrals from lag/summary features | Same historical task and linear residual head remain shared |
| P3 already tried causal Fourier random-feature multi-output regression | `src/p3_wave/causal_spectral_kernel.py`; closed spectral registry | Distinguishes order-sensitive cross-integrals from Fourier-amplitude features | Both ultimately use a multi-output coefficient matrix |
| Deep six-output sequence heads are already closed locally | `src/p3_wave/timexer_direct_multilead.py`; `reports/parallel_breakthrough_execution_20260828_v6/summary.md` | Justifies prohibiting a Neural CDE on 182 cases | Different architectures do not provide independent validation data |
| v23 was beneficial in pooled RMSE but failed the complete stability/transport gate | `artifacts/p3_path_signature_residual_cycle_20260901_v23/result.json` | Primary local result | Repeatedly exposed 182-case surface; no Public guarantee |
