# Claim-source ledger

| Claim | Source | Scope / caveat |
|---|---|---|
| 문제 최고 P1/P2/P3는 32.110453 / 28.674902 / 24.784043 | `reports/parallel_dataset_breakthrough_cycle_20260831_v2/leaderboard-readonly-recheck.json` | 2026-08-31 공식 리더보드 read-only snapshot |
| P1 공식 최고는 F1 0.833548 / 28.909341 | `reports/p1_official_component_score_ledger_20260901_v1/score-ledger.json` | aggregate official receipts; hidden row truth 0 |
| P1 historical promotable 후보 누락 0 | `reports/p1_v54_historical_promotable_candidate_registry_audit_20260901_v1/report-source.md` | 470 metadata documents, 149 results; candidate CSV values 0 |
| P1 fresh historical multi-station window 0 | `reports/p1_v53_fresh_historical_transport_window_availability_audit_20260901_v1/report-source.md` | train chronology aggregate only |
| P2 v52가 RMSE 0.424019 / 28.012945로 새 최고 | `reports/p2_v52_official_submission_20260901_v1/official-submission-receipt.json` | official aggregate receipt |
| P3 clean 최고는 RMSE 0.583892 / 24.066168 | `configs/compliance/p3_clean_incumbent_20260901.json` | external lineage cutoff enforced |
| P3 fractional candidate는 clean OOF보다 0.000901185m 악화 | `artifacts/p3_clean_fractional_change_residual_20260901_c1r1/result.json` and independent recomputation | 181 train-only cases; official inference 0 |
| TabPFN-2.6 is synthetic-only and recommended below 100k rows, 2000 features | [official PriorLabs repository](https://github.com/PriorLabs/TabPFN) | Must explicitly pin 2.6 because default version can change |
| TabPFN-2.5/2.6 weights require license acceptance and support local cache/path | [official PriorLabs docs](https://docs.priorlabs.ai/how-to-access-gated-models) | User must accept; no token committed |
| changeforest supplies nonparametric multivariate change-point detection | [JMLR 2023](https://www.jmlr.org/papers/v24/22-0512.html) | Research basis only; not evidence of P1 gain |
| Multivariate change-point consistency can exploit changes beyond means | [AISTATS 2024](https://proceedings.mlr.press/v238/wu24g.html) | Research basis only; not evidence of P1 gain |
