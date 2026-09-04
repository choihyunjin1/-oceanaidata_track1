# P1 v22 selection-contract audit

## Decision

v22 is worth pre-registering only as a new prospective development experiment, not as a reinterpretation of v20r1. The Wilson-90 lower bound was an inner selection guard layered on top of outer dependent bootstrap, slice gates, and Public-transport penalty. Removing that inner confidence bound avoids duplicated uncertainty penalties while retaining a genuinely nested separation: threshold selection sees only the last 25% of the training prefix; Q3/Q4 remain outer scoring blocks.

The relaxation is not free of bias. It searches all observed inner scores and therefore optimizes a high-variance rare-event F1 surface. The outer blocks have also been repeatedly exposed by earlier families. Any v22 result is adaptive development evidence, not fresh confirmation. v20r1 remains `NO_PASS` and is not rescored.

## Duplicate boundary

v6 used an inner-calibrated discriminative ExtraTrees probability and an F1 marginal rule. v22 uses the v20r1 analytic two-class Student-t likelihood ratio and selects the exact add-only union F1. Thus the score model is not v6, while the broad nested-threshold principle is known. The only new scientific question is whether v20r1's all-abstain was caused by duplicated inner conservatism.

The outer contract is unchanged: Q3/Q4 nonnegative, dependent day bootstrap CI90 lower above zero, bootstrap improvement probability at least 0.8, unchanged slice and intervention gates, raw points at least 0.131682092 and calibrated points at least 0.01.
