# P1 v16 duplicate and leakage audit

## Conclusion

Repository-wide search found no prior P1 generalized-cross-entropy or positive-unlabeled/class-noise correction execution. The candidate is therefore non-duplicate. It is deliberately not called PU learning because label propensity and the latent class prior are unidentified.

The mature frozen 74-numeric safe projection is encoded to exactly 165 columns with train-prefix median/IQR statistics, missing channels, row-valid/gap channels, and train-prefix category vocabularies. The only learned prediction surface is an affine sigmoid optimized with fixed GCE q=0.7 and L2=0.001. Extreme rows are downweighted but never removed. The two fits and threshold 0.95 are fixed before outer scoring.

Q3/Q4 are reused development surfaces, so a result is not an independent confirmation.
