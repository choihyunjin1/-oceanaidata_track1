# P3 station-graph Laplacian residual cycle v24

## 결론

- overall decision: **NO_GO_ALL_STATION_GRAPH_CANDIDATES**.
- Concurrent cross-station propagation was stopped at 0 fits because anonymous cases lack deployment-time UTC/simultaneity linkage.
- Directional wind-sea/wave-age was stopped at 0 fits as a semantic duplicate of existing sea-state and forcing-conditioned implementations.
- The executed axis shares station-specific residual coefficients through a fixed geographic Laplacian and never reads another station at query time.
- This is EXPLORATORY_ONLY on the repeatedly exposed 182-case surface, not a Public transport guarantee.
- P3_1_STATION_GRAPH_LAP16_ADD10: NO_GO; RMSE 0.781589947m; delta +0.000398422m; raw -0.006323 points; transport-adjusted -0.055909; blocks 4/6; worst block +0.015888007m; worst station-lead +0.008579797m.
  - episode CI90 [-0.0025496762807795613, 0.0033469933029496093]; block-station CI90 [-0.0030505082290756213, 0.004312129880498583].
- P3_2_STATION_GRAPH_LAP64_ADD10: NO_GO; RMSE 0.781383411m; delta +0.000191886m; raw -0.003045 points; transport-adjusted -0.052631; blocks 5/6; worst block +0.015824253m; worst station-lead +0.007964762m.
  - episode CI90 [-0.0027146963737028516, 0.0031115465040998]; block-station CI90 [-0.003090586988144922, 0.003792796552291589].

No official test/sample/submission/hidden value was read. No CSV was materialized and no upload occurred. Target winsorization was fit on each outer-training fold, with zero row deletion.
