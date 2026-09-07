# P3 uncapped execution — started, not yet a candidate

2026-09-07 10:30:34 KST: new execution launched with the user's explicit authorization to remove the execution time cap and continue. Old failed attempt is untouched.

- Root: `C:\Users\cedis\Documents\OceanFinalDay_20260907\P3_numeric_cpudet_unbounded_v2`.
- Launcher 18188, supervisor 39904, observed train child 41712.
- At 10:32 KST: `CPU_TRAINING`, 29/36 completed backbones inherited and verified, next fit running, no terminal artifact. Reuse preparation PASS; official rows 0. The only stderr message observed is a local Jupyter TCP transport warning, not a training failure.
- Focused synthetic pytest: 3 PASS; Ruff PASS. Source package synthetic CatBoost smoke: 4 fits PASS, production fits 0. Full TRAIN/PREDICT notebooks remain running/unvalidated until their actual execution receipts exist.
- Config SHA256: `f18b59a24036368b05cb15c9fe0dd967df14da9b64c7716553f7b99e3a3e5bdb`.
- Runner SHA256: `1f8318edc89c8df6be5210669d74edcc353666757f490c77279df0954a009bc2`.
- Supervisor SHA256: `ee812c5b7d3c7e8d0a9b8d68b4400554941eb84192c50439231cb2bb98c2aedc`.

## Fixed sequence and evidence claims

1. `completion_1`: copy only 29 exact-hash completed same-recipe backbones plus verified prepared training features into this new directory. Fit remaining 7 backbones and 5 routers. Run independent internal QA, inference and separate-PID exact answer replay. This is verified-prefix completion, **not a fresh cold**.
2. `fresh_cold_2`: start with empty model/cache directories; independently prepare data and fit 36 backbones + 5 routers. Run the same QA/inference/replay; compare answer SHA with completion_1.
3. Only after actual validation, prepare final source/saved-model packages and run extracted-package replay. The old two-cold-only finalizer does not implement this evidence contract and must not be used unchanged.

New production fits planned: 12 + 41 = 53. Reused complete fits: 29. One independent fresh cold, not two. Synthetic smoke fits are separate. CPU4/GPU0; model parameters, seeds, features, split, thresholds and shrink stay unchanged. No runtime-driven early stop, iteration reduction or automatic retry on technical failure.

The `seconds` progress field retains the original preparation timestamp and includes the stopped interval; it is **not current active training duration**. Use the new execution lock/current invocation receipts for the new elapsed duration and per-fit receipts for completed fit cost.

## Timing and external-action boundaries

Runtime forecast, not measurement of a complete cold: the old 29-fit progress was 13,967.968s since prepare and same-fold multi fits took 1,068.969/1,075.297s. Fable review 2's rough proportional estimate is about 17,300s plus five routers and QA/inference (around five hours on this PC). This is a coarse extrapolation, not a guarantee; later full-data fits may cost more. At 10:57 KST, the actual first 30 historical fits summed to 14,712.832s excluding preparation, pauses and QA, and the first full single took 120.687s. Read fresh_cold_2's actual wall time from its completion receipt to supersede estimates after completion. Six-hour compliance remains unproved regardless of whether that general contest rule applies.

Before completion, retain the scored fallback ff42a6a0 (RMSE 0.604351m / 23.741446 points). Final report must distinguish answer SHA equality between prefix-completion and one independent fresh cold, actual wall time, and user-confirmed deadline (currently unknown). These are decision evidence, not authorization to designate or upload a model. No active execution setting or thread count is changed by this documentation update.

No 4h/6h/15:00 kill switch. General P3 applicability of an official six-hour reproduction rule has not been established from the primary notice; the supplied synthetic-pretraining exception does contain six hours. Runtime will be reported separately from eligibility. This execution does not waive any verified organizer rule.

No uploads, deletions, final designation, commits or pushes. No hidden truth or external data. Existing P1/P2 candidates remain unchanged. Heartbeat `p3-cpu-cold-qa` now monitors this new root, remains quiet without actionable change, and reports completion/failure before pausing.
