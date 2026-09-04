# P3 detrended cross-correlation residual cycle v33

## 결론

- overall decision: **NO_GO_ALL_DCCA_CANDIDATES**.
- Univariate multifractal DFA was rejected before fit because of semantic proximity to closed wavelet/scattering and spectral lineages.
- v33 uses fixed scale-dependent detrended cross-covariance; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_DCCA66_RIDGE512_ADD10: NO_GO; RMSE 0.780089927m; delta -0.001101598m; raw +0.017483 points; transport-adjusted -0.032103; blocks 3/6; worst block +0.003083876m; worst lead +0.001074137m; worst station-lead +0.005860823m; worst reference-tail block +0.011524362m; episode CI90 [-0.0027430068137036523, 0.0006049451087880301]; block-station CI90 [-0.002470973435291923, 0.0004905505333194178].
- P3_2_DCCA66_RIDGE2048_ADD10: NO_GO; RMSE 0.779937311m; delta -0.001254214m; raw +0.019905 points; transport-adjusted -0.029681; blocks 5/6; worst block +0.005169571m; worst lead +0.001001380m; worst station-lead +0.002345618m; worst reference-tail block +0.009452950m; episode CI90 [-0.0026092586949718, 0.00016083514779334255]; block-station CI90 [-0.0024992517686289904, 0.00020321286966394348].

Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed the scales, pairs, Ridge strengths, or blend.
