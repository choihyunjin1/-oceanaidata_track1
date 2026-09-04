# P3 deterministic path-signature residual cycle v23

## 결론

- overall decision: **NO_GO_ALL_PATH_SIGNATURES**.
- This is EXPLORATORY_ONLY on the repeatedly exposed 182-case surface; it is not a Public transport guarantee.
- P3_1_PATHSIG_L2_RIDGE256_ADD15: NO_GO; RMSE 0.778398117m; delta -0.002793408m; raw +0.044333 points; transport-adjusted -0.005253; blocks 5/6; worst block +0.018105663m.
  - episode CI90 [-0.00627989880727034, 0.0009432512621910149]; block-station CI90 [-0.006020679080417213, 0.0008924558847098697]; worst station-lead +0.006528202m.
- P3_2_PATHSIG_L2_RIDGE1024_ADD15: NO_GO; RMSE 0.778400719m; delta -0.002790806m; raw +0.044292 points; transport-adjusted -0.005294; blocks 5/6; worst block +0.013763536m.
  - episode CI90 [-0.005465343913465065, -1.0957132112780861e-06]; block-station CI90 [-0.0052758370098423156, 7.738339606304724e-05]; worst station-lead +0.004024401m.

## Method and research basis

The new representation is a deterministic level-2 path signature: ordered iterated integrals over time, five past-only physical channels, and their observation masks. Kiraly and Oberhauser describe signature features as ordered sample cross-moments for sequential data ([JMLR 2019](https://www.jmlr.org/papers/v20/16-314.html)). Neural CDE work confirms the broader continuous-path view for partially observed multivariate series, but this bounded cycle deliberately uses no deep network ([Kidger et al., NeurIPS 2020](https://papers.neurips.cc/paper_files/paper/2020/hash/4a5876b450b45371f6cfe5047ac8cd45-Abstract.html)).

No official test/sample/submission/hidden value was read. No CSV was materialized and no upload occurred. Target winsorization was fixed on each outer-training fold, and no row was deleted.
