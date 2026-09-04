# Public transport calibration v3: P1 problem-specific audit

## Executive conclusion

The v2 cross-problem `SMOOTH_LEARNED_PROFILE` penalty of `0.121682092` points is not statistically exchangeable with P1 merely because both candidates were labeled “smooth.” That value is the worst residual of two P2 temperature-regression submissions, whereas P1 is F1 classification with an anchor-preserving add-only intervention. The problems have different metrics, score slopes, label prevalence, candidate action, and shift mechanism. No empirical evidence identifies their transport residuals as draws from one distribution.

For newly registered P1 experiments only, v3 therefore uses a same-problem-first hierarchy. P1 has one measured pair: expected central `+0.005383691` points and actual `0`, hence an adverse residual and provisional penalty of `0.005383691`. The minimum desired improvement remains `+0.01`, producing a prospective raw threshold of `0.015383691` points. Existing v5-v22 outcomes are not reclassified.

## Evidence and reasoning

The official ledger contains one P1 pair, two P2 pairs, and one P3 hard-router pair; calibration v2 additionally records a P2 fixed rule and P3 KMA pair. P2’s worst smooth residual `-0.121682092` reflects regression errors transported into P2 competition points. Applying it to a P1 F1 intervention assumes cross-problem exchangeability without replication or a shared generative mechanism.

[Cawley and Talbot (JMLR, 2010)](https://www.jmlr.org/papers/v11/cawley10a.html) show that optimizing noisy model-selection criteria can itself overfit and bias evaluation. This supports freezing the penalty hierarchy before a candidate’s internal result and forbids using v3 to rescue earlier failures. [Varma and Simon (BMC Bioinformatics, 2006)](https://doi.org/10.1186/1471-2105-7-91) show why selection and evaluation must be nested; v3 retains that requirement. [Sugiyama, Krauledat, and Müller (JMLR, 2007)](https://jmlr.org/papers/v8/sugiyama07a.html) demonstrate that ordinary validation can lose unbiasedness under covariate shift. Their result does not justify borrowing a residual across unrelated tasks; it instead underscores that transport must match the target distribution and loss.

## Prospective rule

1. Use the worst adverse official residual from the same P1 exact family.
2. Otherwise use the worst same-problem, same-tier residual when available.
3. Otherwise use the worst observed P1 residual across its registered families.
4. Cross-problem v2 fallback is permitted only when a problem has zero official pairs.
5. Add `0.01` points to the selected penalty using an inclusive gate.

The current P1 n=1 penalty is an empirical floor, not a confidence interval or an upper bound. No penalty reduction may be reviewed before three same-problem pairs. New P1 adverse observations update the problem penalty by the maximum residual. Model selection must remain nested or frozen, and Public scores cannot become row or event labels.

## Limitations

One P1 pair cannot estimate variance, family effects, season dependence, or a tail quantile. The proposed `0.005383691` penalty may understate future transport loss. Conversely, the P2-derived `0.121682092` is demonstrably conservative but not calibrated for P1. v3 chooses relevance over an unsupported cross-task worst case and labels the resulting uncertainty explicitly.

No model was trained, no official row-level file or hidden label was read, and no CSV or upload was created.
