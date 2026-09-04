# P1 v20 duplicate and design audit

## Conclusion

`P1_1_ROBUST_STUDENT_T_CLASS_LLR_ADDONLY` is not an exact or semantic duplicate of v5-v19. It is presealed for synthetic validation only; historical, official, hidden, lock, CSV, and upload counts remain zero.

## Separation from prior families

- v5-v8 used discriminative tree scores, calibration, consensus, or drift routing.
- v9-v14 used CAPA benefit routing or deterministic anchor morphology/peer rules.
- v15-v17 used additive splines, GCE affine discrimination, or MiniRocket PPV features.
- v18 uses soft symbolic transition features with a discriminative linear head.
- Earlier TS2Vec prototypes model distance from the normal embedding distribution; v20 instead fits both labeled classes on ten fixed scalar causal features and forms an analytic Student-t log-likelihood ratio.
- No repository hit implemented a two-class diagonal Student-t likelihood ratio with the exact chronological inner threshold rule below.

## Frozen threshold discipline

For each outer fit, the last 25% of its training prefix is the sole calibration set. The first 75% fits class medians, MAD scales, and Jeffreys-smoothed prior. Every finite calibration score is considered deterministically; the chosen threshold maximizes calibration add-only F1 subject to positive additions, Wilson-90 precision lower bound above inner incumbent F1/2, and changed fraction at most 0.5%. Ties choose the higher threshold then fewer additions. Outer Q3/Q4 labels never select the threshold or score.

The transport family is `SMOOTH_LEARNED_PROFILE`; penalty is 0.121682092 points and the inclusive raw minimum is 0.131682092 points. Expected historical cost is two pipeline fits, under 180 seconds, 1 GiB RSS, and no GPU.
