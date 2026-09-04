# Metric-aligned gate gap matrix

| Requirement | P1 | P2 | P3 | Consequence |
|---|---|---|---|---|
| Official-form primary metric | pooled row micro-F1 available | pooled all-row temperature RMSE available | pooled six-lead Hs RMSE available | Use these as the sole efficacy primary. |
| Numeric practical margin backed by official score/cost | absent | absent | absent | Do not invent fixed raw-unit cutoffs; report direction and uncertainty tiers. |
| Fresh unexposed local surface | none in Q2/Q3/Q4 | none in three historical windows | none in exposed historical cohorts | All new results are `RESEARCH_ONLY`, even when intervals are favorable. |
| Dependence-preserving interval | must preserve day and event structure | saved CI is paired by KST day; contiguous-block provenance is incomplete | saved CIs keep six leads per anchor but generally do not cluster storm episodes | Recompute with predeclared blocks when row-level predictions exist; otherwise disclose the limitation. |
| Valid hard safety objective | anchor positives must never be removed | correction physical/action bound can be structural | keys/leads/schema/finite physical range are structural | These remain hard validity gates. |
| Slice threshold tied to official mixture or explicit cost | not established | not established | not established | Month/window/station/layer/lead thresholds become diagnostics, not automatic vetoes. |
| Candidate with favorable primary evidence | none among latest SupCon / support-bank outcomes | Gaussian copula and state-conditioned copula | none among latest CatBoost confirmation / masked SSL | Reclassify two P2 candidates; P1/P3 harm conclusions remain. |
| Local-to-official transport calibration | unstable and family-specific | unstable and layer/regime dependent | reverse-axis transport has changed signs | No global local-to-official multiplier. Preserve mechanism-specific ledgers. |
| Outlier ground truth | unavailable | unavailable | unavailable | No hard row deletion; robust loss/flags only. |
| Official action in this cycle | not authorized | not authorized | not authorized | CSV generation and upload remain zero. |

## New one-shot evidence under the recalibrated gate

| Problem | Candidate | Pooled primary | Dependence-aware uncertainty | State | Gate lesson |
|---|---|---:|---|---|---|
| P1 | add-only hierarchical event precision LCB | `-0.002380580` F1 benefit | one-sided bounds `[-0.017810465, +0.013951394]` | `INCONCLUSIVE_RESEARCH_ONLY` | Removing arbitrary event/window vetoes prevents a false harm claim, but does not create evidence of benefit. |
| P2 | availability-aware continuous sparse copula v2 | `+0.001990430C` candidate-minus-reference RMSE | CI90 `[+0.000661780, +0.004967253]C` | `PRIMARY_HARM_RESEARCH_ONLY` | The exact recipe loses on the sole primary even after a valid guard-only repair. |
| P3 | selection-matched sparse Bayesian abstention | `-0.003475071m` incumbent-minus-candidate benefit | CI90 `[-0.009929381, +0.003149363]m` | `INCONCLUSIVE_RESEARCH_ONLY` | Legacy magnitude/window/lead vetoes are unnecessary; the primary interval itself is unresolved. |

## Outlier and domain-guard finding

P2 v1 exposed a guard-design defect rather than a model-domain defect. Eighteen validation rows already exceeded the absolute `45C` limit in the frozen reference (layer 3: 10, layer 4: 8); the candidate was finite and exactly unchanged on all 18, with zero new or active violations. V2 changed only the guard contract and preserved all prediction values. No row was deleted. This is direct evidence for distinguishing inherited comparator extremes from candidate-created violations and against unvalidated blanket outlier removal.

## Legacy gate sensitivity replay

The zero-fit replay in `gate-replay.json` changes only two decisions:

- P2 Gaussian-copula conditional mean: legacy `NO_GO` becomes `HIGH_VALUE_CHALLENGER_RESEARCH_ONLY` because pooled benefit is `0.010616065C` and benefit CI90 is `[0.007700262, 0.017384397]C`. The `2025_nov_dec` regression remains a high transport-risk flag.
- P2 state-conditioned copula: legacy `NO_GO` becomes `HIGH_VALUE_CHALLENGER_RESEARCH_ONLY` because pooled benefit is `0.003459176C` and benefit CI90 is `[0.001922933, 0.006529882]C`. The JJA regression remains a transport diagnostic.

P1 SupCon (`-0.164874110` F1 benefit), P3 CatBoost confirmation (`-0.007974131m` benefit, CI wholly unfavorable), and P3 masked SSL (`-0.314155238m` benefit, CI wholly unfavorable) remain closed exact recipes under any metric-aligned gate.
