# P1 v35 v34-reject Q2 adjudication

## Outcome

- Terminal diagnosis: `GUARD_TRUE_NEGATIVE_ON_THIS_FIXED_REJECT`.
- Fixed candidate: v34 q=0.995, threshold 0.6668935418128967, selected before Q2 target access because it had the maximum pre-Q2 Wilson 90% LCB among the sealed candidates.
- Reused state: v34 label-blind Q2 `actions_candidate_0` only; 0 fits, 0 refits, 0 score recomputations, and 0 threshold reselections.
- Q2: 133,170 rows, 332 additions, 0 TP, 332 FP, precision 0.
- F1: anchor 0.7784135753749013, candidate 0.7537256400458541, delta -0.024687935329047228.
- Q3/Q4 target reads: 0/0. Official/test/sample/submission/hidden, CSV, uploads: 0.

This is not a v34 rescue or promotion. The parent v34 `NO_GO_CROSS_QUARTER_TRANSPORT_VETO_Q4_UNOPENED` conclusion remains unchanged.

## Guard diagnosis

For this fixed rejected candidate, the guard did not create a false negative: opening Q2 showed an unequivocally harmful add-only action surface. Combining only prospectively compatible observed receipts gives one harmful guard pass (v27) and one harmful guard reject (v35). Thus the observed negative predictive value is 1/1, but sensitivity remains undefined because no beneficial observed candidate exists; specificity is 1/2. This tiny adjudication sample supports keeping v28 unchanged but cannot establish overall discriminative validity.

Prospective recommendation: retain the current v28 guard. Continue only predeclared audit-only sampling of future immutable rejects whose label-blind scores/actions and state hashes were preserved before target access. Do not relax the guard, retune v34, or use v35 to promote a candidate.

## Verification

- Two zero-fit preflights: byte-identical, 1,579 bytes, SHA-256 `19002e163cd6c51f4d538da11ec1e7364fb70619640fcbbed6c2c5ade5020b3e`.
- Focused pytest: 4/4 PASS before and after execution.
- Ruff: PASS before and after execution.
- Lifecycle-aware immutable QA: PASS.
- Parent bundle revalidated: `4d60a778195377c6c6da1274ff79357881ef2816845089cf720c2905ac0a4c7a`.
- Selection seal: `ca9409c37118757f0933a50c4a8c6b18f60dfe07b752283ac0247f614087d736`.
- Result: `6a62eaf24f3b00fde202fd0d0f5a9a413df5f345d73dbaa602154b468be358c4`.
