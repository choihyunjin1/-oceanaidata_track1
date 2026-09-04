# Claim-source ledger

| Claim | Source | Use | Limitation |
|---|---|---|---|
| P1 has one observed Public pair with residual -0.005383691 points | `official-submission-results-20260831.json`; calibration v2 | Computes the v3 P1 empirical penalty | n=1, not an interval |
| P2 smooth candidates have residuals -0.043888327 and -0.121682092 | calibration v2 | Shows the origin of the v2 smooth-tier worst case | P2 regression, not P1 classification |
| Finite-sample model-selection criteria can be overfit | Cawley & Talbot, JMLR 2010, https://www.jmlr.org/papers/v11/cawley10a.html | Freezing and no retroactive use | General methodological result |
| Nested evaluation reduces parameter-selection bias | Varma & Simon, BMC Bioinformatics 2006, https://doi.org/10.1186/1471-2105-7-91 | Requires nested/frozen validation | Different application domain |
| Ordinary CV may be biased under covariate shift | Sugiyama et al., JMLR 2007, https://jmlr.org/papers/v8/sugiyama07a.html | Supports target-relevant transport calibration | Assumes covariate shift; does not estimate this penalty |
