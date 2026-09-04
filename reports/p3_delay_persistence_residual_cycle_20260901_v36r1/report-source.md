# P3 delay-persistence science-neutral recovery v36r1

## 결론

- overall decision: **PASS_CANDIDATE_AVAILABLE**.
- v36r1 changes only the numerical MST adapter so exact-zero edges remain valid. Science changes are zero; the failed v36 exposed zero fits and zero outer scores.
- The 182-case surface remains EXPLORATORY_ONLY and no v35 prediction, ensemble, or router was reused.
- P3_1_DPH72_RIDGE512_ADD10: NO_GO; RMSE 0.779626905m; delta -0.001564620m; raw +0.024832 points; transport-adjusted -0.024754; blocks 5/6; worst block +0.005589977m; worst lead +0.000906026m; worst station-lead +0.001821311m; worst reference-tail block +0.013428257m; episode CI90 [-0.0031465143965336783, 9.081860848827512e-05]; block-station CI90 [-0.0030302584505169927, 0.00018491743227925803].
- P3_2_DPH72_RIDGE2048_ADD10: PASS_STABLE; RMSE 0.779468385m; delta -0.001723140m; raw +0.027348 points; transport-adjusted -0.022239; blocks 5/6; worst block +0.006086707m; worst lead +0.000867813m; worst station-lead +0.001652809m; worst reference-tail block +0.011348688m; episode CI90 [-0.0031159036143104057, -0.00023684142967390613]; block-station CI90 [-0.0031017337229757966, -0.00013843154159955386].
Official test/sample/submission/hidden access, CSV materialization, and upload were all zero.
