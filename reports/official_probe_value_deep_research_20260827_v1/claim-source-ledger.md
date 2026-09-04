# Claim-to-source ledger

이 파일은 내부 검증용이다. 사용자용 결론의 근거 추적을 위해 유지한다.

| Claim | Evidence | Source | Confidence / caveat |
|---|---|---|---|
| local–official sign 6/14, high-comparability 3/5 | machine-generated calibration rows and summary | `reports/next_day_breakthrough_deep_research_20260827_v1/local_official_calibration.json` | High for recorded rows; small correlated sample |
| P1 Router 10.84×, P2 L4 12.76×, P3 reverse sign reversals | exact local/official contrasts | same calibration JSON; `OFFICIAL_RESULTS_20260826.json` | High for arithmetic; lineage caveats recorded per row |
| Current official team score/rank and problem points | authenticated DOM snapshot on 2026-08-27 | https://oceanaidata.org/app/leaderboard | High at observation time; leaderboard is live |
| No 2026-08-27 team submissions yet | authenticated submission-management DOM snapshot | https://oceanaidata.org/app/submissions | High at observation time |
| P1 remaining 3/3 | authenticated P1 problem DOM snapshot | https://oceanaidata.org/app/problems/5 | High at observation time; P2/P3 per-problem quota uses user-confirmed rule |
| Model-selection criterion can itself be overfit | original JMLR paper | Gavin C. Cawley, Nicola L. C. Talbot, 2010, https://www.jmlr.org/papers/v11/cawley10a.html | High |
| Repeated leaderboard feedback can overfit holdout | original ICML/PMLR paper | Avrim Blum, Moritz Hardt, 2015, https://proceedings.mlr.press/v37/blum15.html | High |
| Limited query allocation is a fixed-budget exploration problem | original COLT/PMLR note | Chao Qin, 2022, https://proceedings.mlr.press/v178/open-problem-qin22a.html | Medium for analogy; our operational policy is an inference, not theorem application |
| P1 e150 local metrics | sealed diagnostic artifact | `artifacts/p1_mstcn_checkpoint_diagnostic_20260827_v2/fixed_epoch_150_metrics.json` | High; already exposed historical folds, no official CSV |
| P2 checkpoint curve | recovered sealed artifact | `artifacts/p2_joint_hydrographic_multitask_layer4_checkpoint_v1/metrics.json` | High for retrospective numbers; prefix chosen post hoc is not deployable confirmation |
| P3 checkpoint retrospective | generated retrospective report and sealed artifacts | `reports/generated/checkpoint_retrospective_20260827_ko.md` | High for recorded metrics; no official transport evidence for this family |
| Round E candidates and hashes | frozen manifest + independent QA | `C:/Users/cedis/Downloads/해양 해커톤 제출용/20260827_round_E_preregistered_P1x3_P2x3_P3x3/SET_MANIFEST.json`; `reports/next_day_breakthrough_deep_research_20260827_v1/independent_bundle_qa.json` | High; no upload performed as of observation |
