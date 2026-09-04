# P1 v41 co-teaching support gate

## Decision

`NO_GO_TARGET_FREE_LABEL_NOISE_SUPPORT_UNIDENTIFIED`; immutable zero-fit closure.

Han et al.'s [NeurIPS 2018 Co-teaching paper](https://proceedings.neurips.cc/paper_files/paper/2018/hash/a19744e268754fb0148b017647355b7b-Abstract.html) motivates two networks that exchange small-loss examples for training with noisy labels. It does not establish label corruption, a forget rate, or performance for P1.

The P1 README calls `label` a ground-truth label and documents no label-noise rate, corruption process, independent annotators, trusted-clean subset, or label-free noise proxy. Estimating a forget rate from per-row losses would require reading targets before the support decision and would make the sealed schedule outcome-dependent. Importing a rate from the paper's synthetic image-label experiments would have no P1 data-contract justification. The required target-free support gate therefore fails.

Repository-wide negative search found no prior co-teaching, cross-peer small-loss selection, peer-loss, or learned noise-transition execution. Its mechanism is distinct from v36 GCE's single-model bounded scalar loss, v38 focal loss's single-model easy-row downweighting, and v39 R-Drop's two stochastic passes of one model. Novelty alone does not cure the missing scientific support.

Fit accounting itself is feasible: two peers are two optimizer fits per seed, so three seeds would consume exactly `2 × 3 = 6` of the maximum nine fits. Because the independent label-noise support gate fails, the allowed counts remain optimizer fits `0`, target reads `0`, preflights `0`, locks/artifacts `0`, and official/test/sample/submission/hidden/CSV/uploads/removals `0`. No forget schedule, threshold, or peer model was instantiated.

This family must not be retried by deriving a noise rate from the labels or borrowing a CIFAR/MNIST rate. The next P1 axis should not presume label corruption; a representation or objective whose support can be established solely from README/train covariates is required.
