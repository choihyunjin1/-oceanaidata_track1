# P3 fixed simplex stacking cycle v25

## 결론

- overall decision: **NO_GO_ALL_SIMPLEX_CANDIDATES**.
- RevIN was stopped at 0 fits as an exact semantic duplicate of the already executed and harmful robust RevIN PatchTST family.
- The executed stacker fits non-negative sum-to-one weights only on each purged outer-training fold, then applies a presealed small residual from the exact uniform champion.
- This is EXPLORATORY_ONLY on the repeatedly exposed 182-case surface, not a Public transport guarantee.
- P3_1_FIXED_SIMPLEX_SHRINK10: NO_GO; RMSE 0.781392328m; delta +0.000200803m; raw -0.003187 points; transport-adjusted -0.052773; blocks 3/6; worst block +0.001779018m; worst station-lead +0.003577762m.
  - episode CI90 [-0.00044756569531489987, 0.000882419788224359]; block-station CI90 [-0.00041580178223720176, 0.000920794809828457].
- P3_2_FIXED_SIMPLEX_SHRINK20: NO_GO; RMSE 0.781721475m; delta +0.000529951m; raw -0.008411 points; transport-adjusted -0.057997; blocks 3/6; worst block +0.003718806m; worst station-lead +0.007474757m.
  - episode CI90 [-0.0007989545103980199, 0.0019009653108505223]; block-station CI90 [-0.0007357574643188813, 0.0020054299670962117].

No official test/sample/submission/hidden value was read. No CSV was materialized and no upload occurred. No row was deleted and no outer result changed a base model, weight rule, or shrink strength.
