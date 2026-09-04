# P1 v25 Deep-SVDD hypersphere semantic gate

`CLOSED_ZERO_FIT_SEMANTIC_DUPLICATE`. No train/README row, target, fit, lock, artifact, official/test/sample/submission/hidden file, CSV, or upload was touched.

[Ruff et al. (ICML 2018)](https://proceedings.mlr.press/v80/ruff18a.html) motivate mapping normal data toward a hypersphere center and using distance from that center as the anomaly score. In P1, however, the learned-normal-embedding plus normal-center-distance plus thresholded-addition fingerprint is already occupied by the TS2Vec conditional normal-prototype lane. That lane learned timestamp embeddings, formed conditional normal prototypes, and proposed anomalies by prototype distance; it closed on embedding coverage before label-based recovery.

Replacing hierarchical contrastive training with hypersphere compactness changes the loss but preserves the same normal-distance representation/action. The repository also records one-class contrastive normal modeling as the same broad mechanism. Therefore v25 is not independent enough for another historical target opening. Hypercenter, radius, architecture, or threshold choices were never made from data or outcomes.

Immutable counters: source reads `0`, targets `0`, fits `0`, locks/artifacts `0`, official/test/sample/submission/hidden/CSV/uploads `0`.
