# Prior P3 result and metric JSON audit

- files read: 86
- numeric delta-RMSE claims: 3551
- excluded official/hidden/submission paths: 3

## Largest reported internal improvements

| delta RMSE(m) | path | key |
|---:|---|---|
| -0.202807914 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[0].diagnostics.episodes.records.S-ORS|42.delta_rmse` |
| -0.202807914 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[1].diagnostics.episodes.records.S-ORS|42.delta_rmse` |
| -0.202807914 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[2].diagnostics.episodes.records.S-ORS|42.delta_rmse` |
| -0.170456252 | `artifacts/p3_parallel_candidate_cycle_20260831_v4/result.json` | `candidates[0].episodes.records.S-ORS|42.delta_rmse` |
| -0.150448763 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[0].diagnostics.episodes.records.S-ORS|24.delta_rmse` |
| -0.150448763 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[1].diagnostics.episodes.records.S-ORS|24.delta_rmse` |
| -0.150448763 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[2].diagnostics.episodes.records.S-ORS|24.delta_rmse` |
| -0.133037926 | `artifacts/p3_parallel_candidate_cycle_20260831_v4/result.json` | `candidates[2].episodes.records.S-ORS|42.delta_rmse` |
| -0.119653858 | `artifacts/p3_physical_expert_router_cycle_20260831_v5/result.json` | `candidates[1].episodes.records.S-ORS|24.delta_rmse` |
| -0.119653858 | `artifacts/p3_v5_extratrees_competition_adjudication_20260831_v7/result.json` | `adjudication.v5_record.episodes.records.S-ORS|24.delta_rmse` |
| -0.119653858 | `artifacts/p3_v5_extratrees_competition_adjudication_20260831_v7/result.json` | `output.adjudication.v5_record.episodes.records.S-ORS|24.delta_rmse` |
| -0.113036186 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[0].diagnostics.episodes.records.S-ORS|29.delta_rmse` |
| -0.113036186 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[1].diagnostics.episodes.records.S-ORS|29.delta_rmse` |
| -0.113036186 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[2].diagnostics.episodes.records.S-ORS|29.delta_rmse` |
| -0.113036186 | `artifacts/p3_physical_expert_router_cycle_20260831_v5/result.json` | `candidates[1].episodes.records.S-ORS|29.delta_rmse` |
| -0.113036186 | `artifacts/p3_physical_expert_router_cycle_20260831_v5/result.json` | `candidates[2].episodes.records.S-ORS|29.delta_rmse` |
| -0.113036186 | `artifacts/p3_v5_extratrees_competition_adjudication_20260831_v7/result.json` | `adjudication.v5_record.episodes.records.S-ORS|29.delta_rmse` |
| -0.113036186 | `artifacts/p3_v5_extratrees_competition_adjudication_20260831_v7/result.json` | `output.adjudication.v5_record.episodes.records.S-ORS|29.delta_rmse` |
| -0.109290393 | `artifacts/p3_parallel_candidate_cycle_20260831_v4/result.json` | `candidates[1].episodes.records.S-ORS|42.delta_rmse` |
| -0.094894161 | `artifacts/p3_physical_expert_router_cycle_20260831_v5/result.json` | `candidates[0].episodes.records.S-ORS|42.delta_rmse` |
| -0.088364642 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[0].diagnostics.episodes.records.S-ORS|47.delta_rmse` |
| -0.088364642 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[1].diagnostics.episodes.records.S-ORS|47.delta_rmse` |
| -0.088364642 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[2].diagnostics.episodes.records.S-ORS|47.delta_rmse` |
| -0.083475754 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[0].diagnostics.episodes.records.S-ORS|38.delta_rmse` |
| -0.083475754 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[1].diagnostics.episodes.records.S-ORS|38.delta_rmse` |

## Largest reported regressions

| delta RMSE(m) | path | key |
|---:|---|---|
| 0.294526919 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.persistence.propensity_weighted.by_station.I-ORS.delta_rmse` |
| 0.283811449 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.chronos2_nested.propensity_weighted.by_station.I-ORS.delta_rmse` |
| 0.217426832 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.wave_state_all_181.candidate_metrics.persistence.propensity_weighted.by_station.S-ORS.delta_rmse` |
| 0.191140901 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.event_nlinear.unweighted.by_station.S-ORS.delta_rmse` |
| 0.185975413 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.persistence.propensity_weighted.delta_rmse` |
| 0.185424532 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.wave_state_all_181.candidate_metrics.chronos2_nested.nearest_neighbor_weighted.by_station.I-ORS.delta_rmse` |
| 0.176636184 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.chronos2_nested.unweighted.by_station.I-ORS.delta_rmse` |
| 0.167270311 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[2].diagnostics.episodes.records.S-ORS|40.delta_rmse` |
| 0.167270311 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[1].diagnostics.episodes.records.S-ORS|40.delta_rmse` |
| 0.167270311 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[0].diagnostics.episodes.records.S-ORS|40.delta_rmse` |
| 0.164819314 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.wave_state_all_181.candidate_metrics.event_nlinear.propensity_weighted.by_station.S-ORS.delta_rmse` |
| 0.162449101 | `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json` | `candidate_metrics.chronos2_nested.propensity_weighted.by_station.I-ORS.delta_rmse` |
| 0.160094950 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.chronos2_nested.propensity_weighted.delta_rmse` |
| 0.159554554 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.chronos2_nested.nearest_neighbor_weighted.by_station.I-ORS.delta_rmse` |
| 0.157170432 | `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json` | `candidate_metrics.chronos2_nested.nearest_neighbor_weighted.by_station.I-ORS.delta_rmse` |
| 0.156624069 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.wave_state_all_181.candidate_metrics.event_nlinear.unweighted.by_station.S-ORS.delta_rmse` |
| 0.156624069 | `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json` | `candidate_metrics.event_nlinear.unweighted.by_station.S-ORS.delta_rmse` |
| 0.156105721 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.complete_wind_source.candidate_metrics.chronos2_nested.unweighted.by_station.S-ORS.delta_rmse` |
| 0.153029399 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.wave_state_all_181.candidate_metrics.chronos2_nested.propensity_weighted.by_station.I-ORS.delta_rmse` |
| 0.151576570 | `artifacts/p3_target_shift_retroaudit_20260828_v2/result.json` | `surfaces.wave_state_all_181.candidate_metrics.chronos2_nested.unweighted.by_station.I-ORS.delta_rmse` |
| 0.151576570 | `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json` | `candidate_metrics.chronos2_nested.unweighted.by_station.I-ORS.delta_rmse` |
| 0.148278038 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[2].diagnostics.episodes.records.S-ORS|39.delta_rmse` |
| 0.148278038 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[1].diagnostics.episodes.records.S-ORS|39.delta_rmse` |
| 0.148278038 | `artifacts/p3_inner_lcb_router_cycle_20260831_v6/result.json` | `candidates[0].diagnostics.episodes.records.S-ORS|39.delta_rmse` |
| 0.147807446 | `artifacts/p3_target_shift_retroaudit_20260828_v1/result.json` | `candidate_metrics.persistence.propensity_weighted.by_station.S-ORS.delta_rmse` |
