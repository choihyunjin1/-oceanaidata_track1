# 2026-08-30 frozen-candidate confirmation — final result

## Conclusion

Only P3 improved the official Public score. The frozen P3 KMA long-lead correction at `alpha=0.425` reduced RMSE from `0.575262` to `0.575233` and added `0.000473` points. P2 Gaussian-copula v2 was a useful but negative official probe: RMSE rose from the incumbent `0.430209` to `0.442259`, so that transfer axis is `DO_NOT_REPEAT`. P1 trial18 is not submission-ready: the sealed Q3+Q4 confirmation produced `delta F1=-0.0118891200` with a wholly negative 90% paired block-bootstrap interval.

The live leaderboard was independently reread at 2026-08-30 22:08 KST. Team `분당독고다이` remains rank 5 with P1 `28.909341`, P2 `27.935277`, P3 `24.203599`, and total `81.048217`. The P2 harm probe did not replace the best-of-problem score.

## P1 — trial18 frozen confirmation

- Decision: `PRIMARY_HARM_RESEARCH_ONLY / NOT_READY`.
- Pooled Q3+Q4 rows: `287,862`.
- Incumbent F1: `0.9068037200`; trial18 F1: `0.8949146000`.
- Delta F1: `-0.0118891200`.
- Paired 21-day block-bootstrap, 10,000 replicates, 90% CI: `[-0.0260995305, -0.0019713775]`.
- Original work: 6 fits and 17,550 optimizer steps. Sealed-evaluation recovery added 0 fits and 0 predictions.
- The original runner and first evaluation harness failed before truth metrics because of evaluator-only key/receipt bugs. A pinned 0-fit sealed-artifact evaluator repaired only those two mappings, passed focused pytest/Ruff, and opened historical truth once.
- Official/test/sample/submission/hidden rows read: 0. Official CSVs/uploads: 0.
- The two apparent fallback files (`63156b…2675`, `b02f08…8598`) reconcile to 2026-08-29 platform submissions and are semantic duplicates; eligible fallback count is 0.

## P2 — Gaussian copula v2 official probe

- Candidate SHA-256: `f498c6e1d7e22d11d5571b971454f0e375247fc2ff5ae3387bfcb4186460c4a3`.
- Candidate contract: 26,061 rows, exact `station,layer,time,temp`, one fit, 0 HPO, 0 result-based retry.
- Official submission: 2026-08-30 22:05 KST.
- Public RMSE/points: `0.442259 / 27.784078`.
- Previous champion: `0.430209 / 27.935277`.
- Delta: `+0.012050°C / -0.151199 points`.
- Decision: `PUBLIC_HARM / DO_NOT_REPEAT`.
- Interpretation: local historical gain against a proxy comparator did not transfer to the official Public set. This is evidence against the axis, not a reason for post-hoc retuning on the same score.

## P3 — KMA uniform long-lead correction

- Candidate SHA-256: `144f5e1740a338df881b5076b8d0a8764630c5836982a6ff4326e93c2e24219e`.
- Frozen change: apply KMA correction `alpha=0.425` only to 18h and 24h; 3–12h are exact no-ops.
- Official submission: 2026-08-30 21:19 KST.
- Public RMSE/points: `0.575233 / 24.203599`.
- Previous champion: `0.575262 / 24.203126`.
- Delta: `-0.000029m / +0.000473 points`.
- Decision: `PUBLIC_BEST_ONLY / PRIVATE_UNCONFIRMED`.
- The observed RMSE matched the preregistered curve estimate (`0.575232`) to displayed precision. This supports the frozen micro-exploit but not another nearby-alpha sweep.

## Submission accounting and deadline

- P1 today: `3/3` remain; no eligible candidate.
- P2 today: `2/3` remain after one negative official probe.
- P3 today: `2/3` remain after one positive official probe.
- The official schedule notice sets the overall problem-submission deadline to 2026-09-07. KST midnight only resets/expires the daily quota; it is not the overall competition deadline.
- No additional candidate met all three requirements: nonduplicate, QA-ready, and not already dominated by local or official evidence.

## Primary evidence

- `p1-trial18-sealed-evaluation-qa.json` — SHA-256 `4c13a2b50a81c987b05bad9f889a4b60bb9f0b51405be2ad9e8fbeef2b1b333f`.
- `p2-gaussian-copula-v2-official-result-integration.json` — SHA-256 `62c68b773cd37105693ed9ded80300374cb83a5564b354dde33785efee81efde`.
- `../p2_gaussian_copula_v2_exact_frozen_submission_pack_20260830_v3/official-submission-receipt.json` — SHA-256 `198e2162dfe44d65a06468530ea8aa5ecdb71dc7be498e60bb954d4b73fbc248`.
- `../p3_kma_uniform_0425_official_submission_20260830_v1/official-submission-receipt.json` — SHA-256 `3277c684751c4ae88b29d33a573229f780c8b0af3ad5f558bda88c30803c894d`.
