# P1 v6 metric-consistency preflight

The exact add-only algebra is valid. For anchor counts `T, P, N`, adding `a` true positives and `b` false positives changes F1 from `2T/(2T+P+N)` to `2(T+a)/(2T+P+N+a+b)`. Strict improvement is therefore equivalent to `a/(a+b) > F1/2`. Exhaustive integer-grid tests passed.

Lipton, Elkan, and Narayanaswamy independently show that when scores are calibrated probabilities, the F1-optimal decision threshold is half the optimal F1: https://arxiv.org/abs/1402.1892. This supports the metric relationship but does not create the missing P1 calibration surface.

The proposed ROCKET continuation is an exact semantic duplicate of P1 v17 causal MiniRocket-lite and the v32 closure ledger, so it closed at zero fits. Recalibrating v5r1 also closed at zero fits: its Q2 inner interval ends on 2025-03-24, while the authenticated champion OOF begins on 2025-04-01. The v5r1 seals contain outer scores, not Q2 inner incumbent predictions. Using exposed outer results or inventing an anchor score would violate the requested inner-only rule.

Decision: `CLOSE_ZERO_FIT_DUPLICATE_AND_MISSING_INNER_ANCHOR`. Fits, outer targets, official access, CSV, and uploads were all zero. Result SHA-256 is `54b50dc9d034c09e58255285c6d5e12cd42df4ce29dc12dcf59213b160500988`. A separate lifecycle-safe verifier rechecked the consumed namespace without modifying its immutable runner and passed.
