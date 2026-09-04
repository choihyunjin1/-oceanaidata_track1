# P1 v31 prospective duplicate audit

v31 does not reclassify v28 or relax v29r1. It preserves v28's frozen three-logit classifier and EM constants, but creates new predictions through a train-prefix-only continuous logit shrinkage factor. The last 25% of each prefix is split in chronological order: its first half estimates the shrink factor against observed prevalence; its second half selects the add-only threshold. The outer fold supplies covariates only. This differs from v28 full EM, v29 hard group/day guards, v27 ECDF ranks, and fixed threshold/model candidates.

Prospective gate v4 is applied: pooled/Q3/Q4/bootstrap/+0.01 transport/anchor/precision/overall change/concentration remain hard; per-day change and every station-layer point delta are reported warnings only. Historical execution, attempt lock, official/hidden reads, CSV, and upload remain zero pending authorization.
