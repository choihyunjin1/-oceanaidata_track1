# P1 v29r1 metric-only recovery conclusion

The recovery completed with zero additional fits and bit-exact reuse of v29's sealed prediction NPZ. The candidate is a close but strict **NO_GO**: 10 Q3 additions were all true positives and Q4 abstained, but raw expected improvement was `+0.014682715` points, below the prospective `+0.015383691` threshold; calibrated improvement was `+0.009299024`, below `+0.01`. The dependent bootstrap CI90 lower bound was exactly zero, not strictly positive, and all additions concentrated in one station-layer-quarter cell.

- Q3: 10 additions, precision 1.0, delta F1 +0.000942689.
- Q4: abstain, delta F1 0.
- pooled delta F1 +0.000552436; P(improvement) 0.865; CI90 [0, +0.001520986].
- anchor removals 0; changed fraction 0.000034739; maximum KST-day fraction 0.004251299.
- additional fits 0; prediction writes 0; sealed NPZ remained SHA-256 `5d0f4915acc95abd26c59bbcf2890738178fc2b9fc737ac9004254388e5f91e9`.
- official/hidden/CSV/upload 0/0/0/0. No candidate was materialized.

This result must not be relaxed or retrospectively passed: it misses both the raw and calibrated prospective thresholds and the strict CI/slice guards.
