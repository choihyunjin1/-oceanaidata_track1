# Claim-source ledger

| claim | local evidence | external primary source | limitation |
|---|---|---|---|
| P1 virgin local tail은 0행이고 exposed outer score는 독립 holdout이 아니다. | `artifacts/validation_system_audit_20260822/p1.json` | [Cawley & Talbot 2010](https://www.jmlr.org/papers/v11/cawley10a.html) | exposure count는 보수적 하한 |
| P1 range anomaly는 point-only shortcut으로 평가하면 안 된다. | P1 event/anchor metric contract | [Tatbul et al. 2018](https://papers.nips.cc/paper_files/paper/2018/hash/8f468c873a32bb0619eaeb2050ba45d1-Abstract.html) | 공식 score와 동일한 metric이라는 뜻은 아님 |
| P2 bin17은 alpha50보다 0.001058°C 개선, bin18은 0.000015°C 악화다. | `reports/official_information_probe_cycle_20260830_v1/p2-official-result.json` | [Beckers & Rixen 2003](https://doi.org/10.1175/1520-0426(2003)020%3C1839:ECADFF%3E2.0.CO;2) | 같은 Public split; Private 미상 |
| P3 uniform KMA 0.425는 alpha=0보다 0.008659m 개선됐다. | `reports/official_information_probe_cycle_20260830_v1/p3-official-result.json` | [Ellenson et al. 2020](https://doi.org/10.1016/j.coastaleng.2019.103595) | 문헌은 exact KMA weight를 지지하지 않음 |
| P3 station 제거와 ERA5 Hs² residual은 contemporary champion보다 악화됐다. | P3 official result와 `p3_official_submission_receipt_20260828.json` | [Ellenson et al. 2020](https://doi.org/10.1016/j.coastaleng.2019.103595) | 서로 다른 contemporary baseline |
| 시간순서를 보존한 새 block이 없으면 재검증이 아니다. | `artifacts/promotion_retroaudit_20260827_v1/gap_matrix.json` | [Bergmeir & Benítez 2012](https://doi.org/10.1016/j.ins.2011.12.028) | 새 label이 생기면 재평가 필요 |
| 자동 target/extreme 제거는 허용하지 않는다. | preflight config의 outlier policy | - | immutable sensor-QC는 별도 허용 |

## Artifact source notes

- Audience: technical.
- Comparison basis: P1 exposure history, P2 pre-frozen Public factor decomposition, P3 pre-frozen official deltas versus each candidate's declared contemporary baseline.
- Quantitative visual: P3 official RMSE deltas, horizontal bar with zero reference; four categories are the complete reviewed official axis set used in this preflight.
- Omitted P1/P2 charts: P1 counts mix outer results and candidate evaluations; P2 has only two disjoint factor probes. Tables and narrative are more honest than cross-unit or underpowered charts.
- Snapshot status: ready; no required source is missing.
- Data safety: no official row, CSV, hidden truth, credential, or raw dataset is included.
