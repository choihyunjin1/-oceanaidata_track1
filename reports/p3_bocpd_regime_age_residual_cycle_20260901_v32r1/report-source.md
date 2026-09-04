# P3 BOCPD regime-age residual recovery cycle v32r1

## 결론

- overall decision: **NO_GO_ALL_BOCPD_CANDIDATES**.
- Lead-increment and shape-projection ideas were rejected before fit as closed basis/smoothing semantics.
- v32 uses fixed causal run-length posterior summaries; the 182-case surface remains EXPLORATORY_ONLY.
- P3_1_BOCPD96_RIDGE512_ADD10: NO_GO; RMSE 0.779557786m; delta -0.001633739m; raw +0.025929 points; transport-adjusted -0.023657; blocks 5/6; worst block +0.013680983m; worst lead +0.003962271m; worst station-lead +0.007481785m; episode CI90 [-0.004291902323943941, 0.0010810798377888346]; block-station CI90 [-0.004333414988431155, 0.0014201252836399175].
- P3_2_BOCPD96_RIDGE2048_ADD10: NO_GO; RMSE 0.779240927m; delta -0.001950598m; raw +0.030957 points; transport-adjusted -0.018629; blocks 5/6; worst block +0.009843187m; worst lead +0.001119909m; worst station-lead +0.003073313m; episode CI90 [-0.003739244509130174, 1.3433817137514955e-05]; block-station CI90 [-0.00399123647401341, 0.0002754025198152282].

Official test/sample/submission/hidden access, CSV materialization, and upload were all zero. No row was deleted and no outer result changed the encoder, Ridge strengths, or blend.

Science changes: 0; numerical log-domain adapter changes: 1.
