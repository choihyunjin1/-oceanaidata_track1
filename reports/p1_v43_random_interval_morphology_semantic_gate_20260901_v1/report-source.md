# P1 v43 random-interval morphology semantic gate

## Decision

`NO_GO_EXACT_FAMILY_DUPLICATE_CAUSAL_CIF_V32B_V32D`; immutable zero-fit closure.

The proposed v43 fingerprint was a causal backward window, sealed random subintervals, median/IQR/mean/std/slope/min/max/last-minus-first summaries, and a small tree or linear add-only scorer. Existing P1 v32b already executed this exact family: a 36-row causal window, 32 deterministic Sobol intervals, those exact eight summaries over five channels, 1,280 features, and a 192-tree ExtraTrees scorer in two historical fits.

The overlap is not merely generic rolling-feature similarity. `run_p1_causal_cif_lite32_20260831_v32b.py` constructs the interval endpoints and computes the same mean, standard deviation, slope, minimum, maximum, median, IQR, and endpoint difference. The standard P1 feature bank separately supplies trailing median/residual/std/difference-std/MAD/slope proxies, while causal MiniRocket supplies fixed dilated local-morphology convolutions and PPV. Direct interval-set code already owns interval-level action geometry.

The sealed v32d follow-up also made the negative evidence material: pooled delta F1 was `-0.068800366`; all 901 additions were false positives, and 531 incumbent true positives were removed. v43's current add-only contract would prevent removals, but changing only the decoder does not create a new representation family and the user explicitly prohibited reconstructible execution.

No interval count, window, summary subset, forest depth, seed, threshold, or decoder was selected. Preflights/fits/targets/locks/artifacts and official/test/sample/submission/hidden/CSV/uploads/removals are all `0`. Reparameterizing this family would be outcome-informed retuning of the frozen v32b/v32d lineage.
