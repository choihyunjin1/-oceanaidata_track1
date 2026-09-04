# P1 v52 prospective gate information audit

## Outcome

Terminal diagnosis: `GUARD_TRUE_NEGATIVE_ON_THIS_FIXED_REJECT`.

The parent was fixed without target outcomes by the rule “earliest direct, non-recovery, post-v35 P1 science PRE_Q2 NO_GO with PASS QA and a complete target-free v33 bundle.” Technical-invalid v36/v37 and their recovery IDs were excluded prospectively, selecting v38 focal-loss. Within v38, maximum pre-Q2 Wilson-90 LCB fixed q=`0.9975`, threshold `0.7167971730232239`, and `actions_candidate_1` before Q2 target access. Wilson's original paper supports the score-bound method only; it does not validate this guard or imply P1 performance ([Wilson 1927](https://doi.org/10.1080/01621459.1927.10502953)).

The authenticated Q2 bundle SHA was `c77115bb5b7d1ef4788afa639f299c4f2cea6959d1d713650d76366796af78e2`. No model was refit, no score was recomputed, and no threshold was reselected. The fixed action added `77` rows: `4` TP and `73` FP, precision `0.051948052`. Q2 anchor F1 was `0.778413575`; the immutable action reduced it to `0.773328111`, delta `-0.005085464`.

This is evidence for one additional true-negative reject, not a v38 rescue and not validation of v28. v35 evaluated a different immutable v34 mask; v52 evaluated v38. Both use the same historical Q2 label surface, so they are separate candidate outcomes but not statistically independent target windows. Across the currently compatible candidate-level receipts, rejected/nonbeneficial is `2`, rejected/beneficial is `0`, accepted/nonbeneficial is `1`, and accepted/beneficial is `0`. Observed reject NPV is `2/2` and candidate-level specificity is `2/3`, but sensitivity remains undefined because no beneficial observed candidate exists. The tiny, shared-surface sample cannot justify relaxation or claim discriminator validity.

The v38 decision remains unchanged. Promotion, rescue, retuning, score reconstruction, CSV creation, and Q3/Q4 access were all zero. Fits/refits were `0/0`; only one Q2 target window was read. Official, test, sample-submission, submission, hidden, CSV, and upload accesses were `0`.

## Verification and next evidence

- Focused pytest: `6/6` PASS; Ruff: PASS.
- Two preflights: byte-identical, `2,014` bytes, SHA-256 `90e67aa88e66d703d684319084a40ceeada84b4018b43b1bb09a413df2412002`.
- Post-terminal immutable QA: PASS.
- Result SHA-256: `5fa54dc58dc6193d0469127ab8f560f29f6f31704890668fc8f5ef8ed36b231f`.

Minimum new evidence needed next is a future preregistered candidate whose v33 bundle is sealed before a genuinely unexposed labeled transport window, ideally producing at least one beneficial outcome. Another Q2 rejected-mask audit would add correlated candidate evidence but still could not identify sensitivity.
