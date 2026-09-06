# Train-fit postpolicy comparison — preregistered before execution

Two separate, zero-backbone-fit changes are compared with the frozen bracket baseline on the existing nested two-sided historical evaluation. Source artifacts and the 18 saved fit/prediction pairs are pinned in the experiment config. The primary historical block is H1 2025; all three outer blocks and risk slices remain reported. This is repeated historical evaluation, not a fresh confirmation.

1. Range: fit finite normal-labelled training temperature minimum/maximum separately for each outer training partition. Apply the inclusive bounds automatically, without official inputs. Training-normal FP=0 is a construction, not a guarantee for unseen observations.
2. Cell policy: retain the existing inner-calibrated O/B thresholds and choose B, O, intersection or union by inner cell F1, in that tie order. Unseen or single-class inner cells use the global inner policy. Outer labels never choose a policy.

The combined version is a preregistered secondary comparison, eligible only when both separate changes have positive primary mean gain. Risk is reported separately; no leaderboard score enters selection. No normal observations are removed or downweighted. No official or hidden values, CSV generation, uploads, or new backbone fits are authorized by this runner. Budget: CPU 2, GPU 0, 1,800 seconds. Six synthetic checks and Ruff passed before execution.

An exact fresh-process replay and independent arithmetic audit are required before interpreting the result or preparing a deployment candidate. All old artifacts and attempt locks remain unchanged.
