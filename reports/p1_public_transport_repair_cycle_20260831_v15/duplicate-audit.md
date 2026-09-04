# P1 v15 duplicate and leakage audit

## Conclusion

`P1_1_CAUSAL_ADDITIVE_SPLINE_RESIDUAL` is not a duplicate of prior P1 executions. Repository search found no P1 additive cubic-spline logistic execution. Prior Huber IRLS work was a seasonal baseline rather than this incumbent-negative add-only classifier.

The candidate is fixed at ten causal residual-magnitude features, per-prefix median/MAD scaling, independent univariate cubic spline bases, L2 logistic regression, and probability 0.90. It performs exactly two prequential fits: Q2 to Q3, then Q2+Q3 to Q4. Outer-fold labels do not select features, knots, threshold, weights, or caps. The 0.5 singleton weight is computed only inside each training prefix. Caps are pass/fail gates and never trim proposals.

Q3/Q4 have been used by prior development cycles, so any result is adaptive-surface development evidence rather than an independent confirmation.
