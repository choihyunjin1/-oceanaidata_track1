# P2 C60/L120: eight-fold three-seed comparison completed

**Conclusion: the existing Sep–Oct B3 primary improvement is unchanged, but extending the same three-seed comparison to all eight folds shows a worse pooled mean for L120. This is mixed evidence, not a universal improvement.** Preserve both the existing C60 fallback and the already prepared L120 candidate; do not select a new recipe, replace B3 with a favorable slice, or impose a new retrospective veto. Official performance is not estimated from these internal metrics.

The fixed new28fits completed in767.375seconds. Twenty exact-hash models were reused without fitting; the active comparison now contains48models: C60/L120 ×8folds ×3seeds. A separate process replayed every natural and supported-outage validation prediction for all48models in43.547seconds with maximum absolute error0. Independent arithmetic/lineage QA passed425checks; focused synthetic tests6/6 and RuffPASS. This stage read no official index/sample/hidden input and generated no CSV or upload.

## Primary and secondary evidence

All values are RMSE in °C; Δ=L120−C60, so negative is better. Ensemble metrics are calculated from the mean of three predictions before computing pooled SSE/RMSE, not the average of per-seed or per-fold RMSEs.

| Evaluation population | n | C60 mean3 | L120 mean3 | Δ |
|---|---:|---:|---:|---:|
| **B3 primary, natural** |26273|0.488284326|0.483505057|−0.004779270|
| All8 secondary, natural |166268|1.229592481|1.251771599|+0.022179118|
| B3, injected outage evaluated over whole fold |26273|0.538760112|0.563601402|+0.024841290|
| B3, actual injected-outage interval only |7322|0.445059242|0.561965765|+0.116906524|
| All8, injected outage evaluated over whole folds |166268|1.295999270|1.316770441|+0.020771171|
| All8, actual injected-outage intervals only |42292|1.634028093|1.616091565|−0.017936528|
| B3, naturally missing T5 |180|0.535881156|0.765854952|+0.229973796|
| All8, naturally missing T5 |17657|0.342135743|0.324093780|−0.018041963|

B3 predictions are **element-for-element identical to the earlier three-seed primary arrays**; no new B3 fit was performed. Its paired7day/2000-resample90%CI remains [−0.020369156,+0.010871728]°C, empirical bootstrap improvement fraction0.667,10blocks. This is not a probability of official score improvement.

All8 natural pooled SSE is251380.201638677 for C60 versus260530.672332467 for L120 over the same166268unique keys. Its58-block90%CI is[−0.003886561,+0.048565600]°C, empirical improvement fraction0.083;33/58blocks worsen, worst7day Δ+0.298559225°C. For all8 actual-outage intervals, the18-block90%CI is[−0.050883882,+0.018661024]°C, empirical improvement fraction0.7625. Intervals do not establish an unambiguous benefit.

## Same-key fold comparison

| Fold | n | C60 | L120 |
|---|---:|---:|---:|
| B1 |22940|1.985715582|1.985490075|
| B2 |26018|1.337971870|1.325582977|
| B3 primary |26273|0.488284326|0.483505057|
| B4 |18093|0.042008929|0.042661178|
| B5 |16417|0.723103699|0.713066243|
| B6 |26308|1.800797639|1.920018946|
| B7 |13335|0.987673598|0.951496967|
| B8 |16884|0.260733138|0.217714846|

Six of eight fold means improve, but this does not overturn the worse pooled SSE: the B6 deterioration is larger. Natural T5-missing all8 improvement also does not overturn B3's missingness risk:16884 of17657missing rows belong to B8. No fold or evaluation row was removed to change a denominator.

B4/B8 retain `NOT_ESTIMABLE_NO_ROWS` for their pre-fixed last17day outage intervals (0eligible rows), rather than being called successful outage tests. Their whole-fold values are still reported. No dates were moved, unsupported rows dropped, or replacement artificial episodes selected. The earlier source-support receipt preserves the reason: available target observations end before those intervals.

## Exact reuse, fit accounting and safeguards

- Newfits:7non-B3folds ×2additional seeds20260902/03 ×2arms =28. Reused:16first-seed models plus4additional B3 seed models =20. Active C60/L120 comparison =48.
- The earlier28-fit study additionally contained8D60screen models; these were not reloaded or retrained. Thus cumulative historical training across the original study and this completion is56fits, not48newfits. The separately completed L120 full3 adds3, giving59actual fits before any later cold reproduction. Synthetic tests in this completion perform0fits.
- Each new fit's training-array SHA equals its same-fold/arm original first-seed input before fitting. Every model retains train keys/truth/arrays, recipe/device/thread provenance, prediction SHA, and new model state. Training blocks torch.load, so this is not warm starting old weights.
- No recipe, epoch, weight decay, feature, target masking, split/purge, augmentation, domain weighting, gradient penalty, postprocessing, selection rule or seed list changed. C60=60epochs/wd0.0001; L120=120epochs/wd0.0001; both CUDA/CPU2. The existing recipe has no PAVA/envelope projection.
- Training workerPID40456 and replayPID37664 differ; the exact original20model hashes and old parent terminal/QA/replay remain unchanged. The one-shot new attempt consumed28fits once and stayed within its3600-second cap. GPU was released after replay/QA.
- These are previously exposed historical folds and seed-sensitivity evidence, not new independent holdouts or unbiased post-selection estimates. The earlier B3 first seed selected L120, while its additional seeds examined sensitivity; adding otherfold seeds does not erase that selection history.

## Reproducible evidence and next action

- [Prefit contract](preregistration.md), [immutable seal](preregistration-seal.json), [independent QA and detailed risks](independent-qa.json).
- [Terminal result](../../artifacts/p2_c3_multiseed_completion_20260906_v1/terminal_result.json): SHA `71184ee9b791ab80970ebd64ad540fb588698cc65ad9558f667b356c48c096d1`.
- [Evaluation arrays](../../artifacts/p2_c3_multiseed_completion_20260906_v1/evaluation.npz), local-only: SHA `4ce3a80171e40cdb3ee760eec43e86b48441de05db9b2c587ef3e12ac2f695a5`.
- [Fresh replay](../../artifacts/p2_c3_multiseed_completion_20260906_v1/fresh-replay.json): SHA `17d22d55f0cfe7e073eb50af1c7a9403e6f662d317c5ee0cf698335ae2c9a4e3`.
- QA SHA `ae93c7105863d5bf2d9901f74ffcad9c4b6ee78b69f3d3393c8fa3595c1427bd`; source observations SHA `cc5d6fd9ea398a613e485b34fd530c7dc064fa040fa675127a12318856ab178a`.
- [Prior first-seed/primary3seed study](../p2_c3_training_comparison_20260906_v1/report-source.md), [already prepared exact L120 full3 candidate](../p2_c3_training_comparison_full_20260906_v1/report-source.md).

The validation assessment is **share with caveats**: same-key arithmetic, lineage and replay are verified, while evidence is retrospective and benefits vary by time and missingness. No automatic next tuning branch is started. The separately authorized source-only L120 cold/notebook reproduction is packaging verification, not a reaction to these scores, and is tracked in [its own pre-fit record](../p2_L120_portable_cold_20260906_v1/preregistration.md). Existing C60 and L120 answer files are unchanged by this study.
