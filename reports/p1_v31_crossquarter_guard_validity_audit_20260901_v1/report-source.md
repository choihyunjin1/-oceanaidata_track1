# P1 v31 cross-quarter guard validity audit

## Scope

This is a zero-fit, receipt-only audit. It read only already terminal `result.json` and published report receipts for v24, v27, v29, and v30. It did not read train labels, a new Q4 surface, official/test/sample/submission/hidden files, or any CSV, and it does not re-evaluate or change any prior decision.

## Observed evidence

| cycle | prospective gate state | outer outcome available | outcome |
|---|---|---|---|
| v24 | pre-v26 legacy guard; same S-ORS/L5 appeared in both halves | yes, historical motivation only | 332 additions, 0 TP, F1 delta `-0.0093598243` |
| v27 | v26 multi-station/cell guard passed | yes | 332 additions, 4 TP, precision `0.012048`, F1 delta `-0.0090977183` |
| v29 | v28 pre-Q2 guard rejected all candidates | no; Q2/Q3/Q4 unopened | zero action |
| v30 | v28 pre-Q2 guard rejected all candidates | no; Q2/Q3/Q4 unopened | zero action |

v24 is excluded from the prospective confusion matrix because the amendment is nonretroactive. Among post-v26 decisions with an observable independent outer result, the multi-station/cell pass count is 1 and the beneficial-outer count is 0: pass positive predictive value `0/1`. There is no prospectively rejected candidate with an observed outer outcome, because the safety contract deliberately leaves those targets unopened; sensitivity, specificity, negative predictive value, and false-negative rate are therefore undefined.

For the stronger v28 cross-quarter gate, acceptance coverage is `0/2` completed axes (v29/v30), and independent-outcome coverage is also `0/2`. The gate is thus not validated as a discriminator. Current receipts establish one false-positive pass for the v26 diversity gate and no labeled evidence about rejected candidates. They do not establish that the v28 gate is too strict; they establish that its discriminative validity is presently non-identifiable.

## Prospective alternative only

Do not relax or retune v28 on these results. A future guard-validation protocol should pre-register a small sequence of candidates and, for audit purposes only, obtain outcome labels for a fixed random subset of rejects after their zero-action decisions are irreversibly sealed. This creates both pass and reject outcome cells without permitting promotion, threshold changes, or retroactive rescue. Until such audit-only outcomes accumulate, report v28 as a conservative safety veto with unknown sensitivity—not as a validated transport predictor.

No existing result changes. v24/v27/v29/v30 remain closed and immutable.
