# Final gap matrix

| Problem | Candidate/axis | Local or sealed evidence | Official evidence | Final state | Next permitted direction |
|---|---|---|---|---|---|
| P1 | MS-TCN Sobol trial18, threshold 0.8 | `delta F1=-0.01188912`; CI90 entirely below 0 | Not submitted | `PRIMARY_HARM / NOT_READY` | New structural axis only; do not deploy or tune trial18 |
| P1 | GI single-row fallbacks | QA-ready local packs | Both mapped to 8/29 submissions | `SEMANTIC_DUPLICATE` | None from this pair |
| P2 | Gaussian copula v2 conditional residual | Historical pooled proxy gain; one adverse fold | `0.442259`, worse than `0.430209` | `PUBLIC_HARM / DO_NOT_REPEAT` | Rebuild validation/comparator alignment before another copula attempt |
| P3 | KMA 18/24h uniform `alpha=0.425` | Frozen official-curve optimum | `0.575233`, `+0.000473` points | `PUBLIC_BEST_ONLY` | Hold for Private; no nearby-alpha micro-sweep |
| P3 | Remaining split-alpha packs | QA files exist but neighboring official and cross-fit evidence is adverse | Not submitted | `DOMINATED / NO_UPLOAD` | New information axis only |

The largest unresolved issue is validation-to-Public transfer, not insufficient parameter density. P1 and P2 both show that favorable selection-stage or proxy evidence can reverse under broader or official evaluation. Future promotion should require comparator fidelity, blind multi-period confirmation, and an explicit duplicate/platform-history check before consuming a submission.
