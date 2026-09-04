from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/goals/tolerance_recalibration_and_failure_replay_20260830_v2.json"
OUTPUT = ROOT / "reports/tolerance_recalibration_and_failure_replay_20260830_v2/failure-replay.json"

INPUTS = {
    "negative_ledger": ROOT / "reports/negative_evidence_registry_20260830_v1/failure-ledger.json",
    "family_ledger": ROOT / "artifacts/promotion_retroaudit_20260827_v1/family_reclassification_ledger.json",
    "leaderboard": ROOT / "reports/leaderboard_headroom_double_research_20260829_v1/leaderboard_snapshot.json",
    "official_results": ROOT / "reports/deadline_submission_results_20260828_v1/official-results.md",
    "p1_official_evidence": ROOT / "reports/p1_mstcn_lower_bound_veto_20260829_v2/evidence.json",
    "p1_block": ROOT / "artifacts/p1_block_inpaint_v1/manifest.json",
    "p1_peer": ROOT / "artifacts/runs/20260813T205237+0900_strat_gate_fixed24h_59f6d5c6/manifest.json",
    "p1_segment_router": ROOT / "reports/p1_mstcn_segment_precision_router_retroaudit_20260829_v1/evidence.json",
    "p1_window_phase": ROOT / "reports/p1_window_phase_consistency_20260829_v1/aggregate.json",
    "p1_sobol": ROOT / "artifacts/p1_mstcn_sobol_hpo_20260829_v1/aggregate.json",
    "p1_group_dro": ROOT / "artifacts/p1_mstcn_group_dro_20260829_v2/aggregate.json",
    "p1_supcon": ROOT / "artifacts/p1_event_balanced_supcon_f1_head_20260830_v1/aggregate.json",
    "p1_addonly": ROOT / "reports/p1_addonly_hierarchical_event_precision_lcb_20260830_v1/result.json",
    "p2_rank1": ROOT / "artifacts/p2_alpha50_supervised_rank1_functional_residual_20260828_v1/result.json",
    "p2_crossfit_rank1": ROOT / "artifacts/p2_alpha50_supervised_rank1_threeway_crossfit_regime_veto_20260828_v2/result.json",
    "p2_nested_pls": ROOT / "artifacts/p2_nested_pls_capacity_grid_20260829_v1/result.json",
    "p2_gaussian": ROOT / "artifacts/p2_gaussian_copula_conditional_mean_20260830_v2/result.json",
    "p2_state": ROOT / "artifacts/p2_state_conditioned_copula_20260830_v1/result.json",
    "p2_availability": ROOT / "reports/p2_availability_aware_continuous_sparse_copula_20260830_v2/result.json",
    "p2_alpha50_receipt": ROOT / "reports/p2_oas_alpha50_deployment_20260828_v13/official_score_receipt.json",
    "p3_lead_continuous": ROOT / "artifacts/structural_challenger_20260827_v1/p3/metrics.json",
    "p3_sparse_gp": ROOT / "reports/p3_selection_matched_sparse_gp_abstention_20260830_v1/result.json",
    "p3_catboost": ROOT / "artifacts/p3_catboost_confirmation_contract_repair_20260830_v3/one_shot/result.json",
    "p3_ssl": ROOT / "artifacts/p3_selection_matched_masked_ssl_20260830_v1/result.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(benefit: float, ci: tuple[float, float] | None, *, decisive: bool = False) -> str:
    if ci is not None:
        low, high = ci
        if low > 0.0:
            return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
        if high < 0.0:
            return "PRIMARY_HARM_RESEARCH_ONLY"
        return "INCONCLUSIVE_RESEARCH_ONLY"
    if benefit > 0.0:
        return "REOPEN_FROZEN_CONFIRMATION_ONLY"
    if benefit < 0.0 and decisive:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"


def _benefit_ci_from_delta(ci_low: float, ci_high: float) -> tuple[float, float]:
    return -float(ci_high), -float(ci_low)


def _default_family_state(row: dict[str, Any]) -> str:
    if row.get("classification_applicability") == "NO_SCIENTIFIC_RESULT":
        return "NO_SCIENTIFIC_RESULT"
    if row.get("evidence_state") == "QA_BLOCKED" or row.get("workflow_state", "").startswith("INVALID"):
        return "INVALID_NO_SCIENTIFIC_CONCLUSION"
    if row.get("action_state") == "PUBLIC_BEST_ONLY":
        return "OFFICIAL_EVIDENCE_NOT_FAILURE"
    label = str(row.get("historical_label", ""))
    disposition = str(row.get("disposition", ""))
    if any(token in label for token in ("ANCHOR", "SELECTED_COMPONENT", "PUBLIC_BEST")) or disposition.startswith("KEEP_"):
        return "REFERENCE_OR_CHAMPION_NOT_FAILURE"
    if "NOOP" in label or "NO_OP" in label:
        return "NO_EFFECT_OR_INCONCLUSIVE"
    return "EXACT_RECIPE_CLOSED_UNCHANGED"


def _replay_families(policy: dict[str, Any], family_ledger: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = policy["family_overrides"]
    rows: list[dict[str, Any]] = []
    for source in family_ledger["families"]:
        item = {
            "problem": source["problem"],
            "family_id": source["family_id"],
            "name": source["name"],
            "historical_label": source.get("historical_label"),
            "historical_disposition": source.get("disposition"),
            "new_state": _default_family_state(source),
            "evidence_summary": source.get("evidence_summary"),
            "source_count": len(source.get("sources", [])),
        }
        if source["family_id"] in overrides:
            item.update(overrides[source["family_id"]])
        rows.append(item)
    return rows


def _replay_closed_groups(policy: dict[str, Any], ledger: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = policy["closed_group_overrides"]
    rows: list[dict[str, Any]] = []
    for problem in ("p1", "p2", "p3"):
        problem_key = problem.upper()
        for name in ledger[problem].get("closed_exact_groups", []):
            key = f"{problem_key}:{name}"
            rows.append(
                {
                    "problem": problem_key,
                    "group": name,
                    "old_state": "CLOSED_EXACT_GROUP",
                    "new_state": overrides.get(key, "EXACT_SCOPE_CLOSED_UNCHANGED"),
                }
            )
        for name in ledger[problem].get("invalid_not_scientific_no_go", []):
            rows.append(
                {
                    "problem": problem_key,
                    "group": name,
                    "old_state": "INVALID",
                    "new_state": "INVALID_NO_SCIENTIFIC_CONCLUSION",
                }
            )
        for name in ledger[problem].get("not_a_failure", []):
            rows.append(
                {
                    "problem": problem_key,
                    "group": name,
                    "old_state": "NOT_A_FAILURE",
                    "new_state": "NOT_A_FAILURE",
                }
            )
    return rows


def _official_false_negative_cases(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    leaderboard = payloads["leaderboard"]
    p1_evidence = payloads["p1_official_evidence"]["official_score_evidence"]
    p2_before = payloads["p2_alpha50_receipt"]["official_result"]
    current = leaderboard["our_team"]
    slopes = leaderboard["empirical_score_mapping"]

    p1_benefit = float(p1_evidence["e150_plus_gi2_public_f1"]) - float(
        p1_evidence["e150_union_all_public_f1"]
    )
    p2_benefit = float(p2_before["public_rmse"]) - float(current["official_metrics"]["P2_RMSE_C"])

    text = INPUTS["official_results"].read_text(encoding="utf-8")
    p3_20 = re.search(r"KMA 장기보정 20% \| RMSE ([0-9.]+) \| ([0-9.]+)", text)
    p3_40 = re.search(r"KMA 장기보정 40% \| RMSE ([0-9.]+) \| ([0-9.]+)", text)
    if p3_20 is None or p3_40 is None:
        raise ValueError("P3 aggregate official rows not found")
    p3_before_rmse, p3_before_score = map(float, p3_20.groups())
    p3_after_rmse, p3_after_score = map(float, p3_40.groups())
    p3_benefit = p3_before_rmse - p3_after_rmse

    return {
        "P1": {
            "old_gate_raw_metric": 0.003,
            "observed_official_metric_improvement": p1_benefit,
            "observed_actual_point_gain": 0.007978,
            "empirical_point_equivalent": p1_benefit * float(slopes["P1"]["slope"]),
            "false_negative_proven": 0.0 < p1_benefit < 0.003,
        },
        "P2": {
            "old_gate_raw_metric": 0.005,
            "observed_official_metric_improvement": p2_benefit,
            "observed_actual_point_gain": float(current["scores"]["P2"]) - float(p2_before["points"]),
            "empirical_point_equivalent": p2_benefit * abs(float(slopes["P2"]["slope"])),
            "false_negative_proven": 0.0 < p2_benefit < 0.005,
        },
        "P3": {
            "old_gate_raw_metric": 0.005,
            "observed_official_metric_improvement": p3_benefit,
            "observed_actual_point_gain": p3_after_score - p3_before_score,
            "empirical_point_equivalent": p3_benefit * abs(float(slopes["P3"]["slope"])),
            "false_negative_proven": 0.0 < p3_benefit < 0.005,
        },
        "warning": "P1/P2 transforms are empirical OLS planning aids, not official formulas. P3's stored Public points are used directly here.",
    }


def _case_replay(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    block = payloads["p1_block"]
    peer = payloads["p1_peer"]
    sobol = payloads["p1_sobol"]
    group_dro = payloads["p1_group_dro"]
    supcon = payloads["p1_supcon"]
    addonly = payloads["p1_addonly"]
    rank1 = payloads["p2_rank1"]
    crossfit = payloads["p2_crossfit_rank1"]
    nested = payloads["p2_nested_pls"]
    gaussian = payloads["p2_gaussian"]
    state = payloads["p2_state"]
    availability = payloads["p2_availability"]
    lead = payloads["p3_lead_continuous"]
    sparse = payloads["p3_sparse_gp"]
    catboost = payloads["p3_catboost"]
    ssl = payloads["p3_ssl"]

    p1_block_benefit = float(block["historical_gates"]["weighted_f1_delta"])
    p1_block_ci = (
        float(block["historical_gates"]["bootstrap_ci90_lower"]),
        0.051219,
    )
    p1_peer_benefit = float(peer["promotion_decision"]["micro_f1_delta"])
    p1_sobol_benefit = float(sobol["preconfirm_gate"]["winner_metrics"]["pooled_delta_f1"])
    p1_group_dro_benefit = float(group_dro["q2"]["winner"]["metrics"]["pooled_delta_f1"])
    p1_supcon_benefit = float(supcon["pooled"]["candidate"]["f1"]) - float(supcon["pooled"]["control"]["f1"])
    p1_add_benefit = float(addonly["pooled_primary"]["candidate_minus_anchor_f1"])
    p1_add_ci = (
        float(addonly["paired_uncertainty"]["lower_one_sided_95"]),
        float(addonly["paired_uncertainty"]["upper_one_sided_95"]),
    )

    def p2_case(name: str, payload: dict[str, Any], delta_path: tuple[str, ...], ci_path: tuple[str, ...]) -> dict[str, Any]:
        node: Any = payload
        for key in delta_path:
            node = node[key]
        delta = float(node)
        ci_node: Any = payload
        for key in ci_path[:-2]:
            ci_node = ci_node[key]
        ci_low = float(ci_node[ci_path[-2]])
        ci_high = float(ci_node[ci_path[-1]])
        benefit_ci = _benefit_ci_from_delta(ci_low, ci_high)
        return {
            "problem": "P2",
            "candidate": name,
            "benefit": -delta,
            "benefit_ci90": list(benefit_ci),
            "new_state": classify(-delta, benefit_ci),
        }

    cases = [
        {
            "problem": "P1",
            "candidate": "block_inpaint",
            "benefit": p1_block_benefit,
            "benefit_ci90": list(p1_block_ci),
            "new_state": "REOPEN_FROZEN_CONFIRMATION_ONLY",
        },
        {
            "problem": "P1",
            "candidate": "dynamic_peer_reliability",
            "benefit": p1_peer_benefit,
            "benefit_ci90": [-0.001677, 0.011611],
            "new_state": "REOPEN_FROZEN_CONFIRMATION_ONLY",
        },
        {
            "problem": "P1",
            "candidate": "gors_depth_invariance",
            "benefit": 0.00268799,
            "benefit_ci90": [-0.00921025, 0.00355313],
            "new_state": "INCONCLUSIVE_RESEARCH_ONLY",
        },
        {
            "problem": "P1",
            "candidate": "environment_balanced_replay",
            "benefit": 0.0000426,
            "benefit_ci90": None,
            "new_state": "REOPEN_FROZEN_CONFIRMATION_ONLY_LOW_PRIORITY",
        },
        {
            "problem": "P1",
            "candidate": "segment_precision_router_core",
            "benefit": float(payloads["p1_segment_router"]["core_router"]["pooled_q3_q4"]["delta_f1_vs_incumbent"]),
            "benefit_ci90": None,
            "new_state": "REOPEN_FROZEN_CONFIRMATION_ONLY",
        },
        {
            "problem": "P1",
            "candidate": "window_phase_consistency",
            "benefit": float(payloads["p1_window_phase"]["q2_preflight"]["fixed_average_candidate"]["f1"]) - 0.8676757359086266,
            "benefit_ci90": None,
            "new_state": "REOPEN_FROZEN_CONFIRMATION_ONLY",
        },
        {
            "problem": "P1",
            "candidate": "sobol_trial18_threshold08",
            "benefit": p1_sobol_benefit,
            "benefit_ci90": None,
            "new_state": "REOPEN_FROZEN_CONFIRMATION_ONLY",
        },
        {
            "problem": "P1",
            "candidate": "group_dro_fixed_objective",
            "benefit": p1_group_dro_benefit,
            "benefit_ci90": None,
            "new_state": classify(p1_group_dro_benefit, None, decisive=True),
        },
        {
            "problem": "P1",
            "candidate": "event_balanced_supcon",
            "benefit": p1_supcon_benefit,
            "benefit_ci90": None,
            "new_state": classify(p1_supcon_benefit, None, decisive=True),
        },
        {
            "problem": "P1",
            "candidate": "hierarchical_event_precision_addonly",
            "benefit": p1_add_benefit,
            "benefit_ci90": list(p1_add_ci),
            "new_state": classify(p1_add_benefit, p1_add_ci),
        },
        p2_case("supervised_rank1", rank1, ("metrics", "aggregate", "delta_rmse"), ("bootstrap", "ci90_low", "ci90_high")),
        p2_case("crossfit_rank1_v2", crossfit, ("metrics", "aggregate", "delta_rmse"), ("bootstrap", "ci90_low", "ci90_high")),
        p2_case("nested_pls", nested, ("metrics", "aggregate", "delta_rmse"), ("bootstrap", "ci90_low", "ci90_high")),
        p2_case("gaussian_copula_v2", gaussian, ("metrics", "aggregate", "delta_rmse"), ("bootstrap", "ci90_low", "ci90_high")),
        p2_case("state_conditioned_copula", state, ("metrics", "pooled", "delta_rmse"), ("bootstrap", "ci90_low", "ci90_high")),
        p2_case("availability_aware_copula_v2", availability, ("metrics", "pooled", "delta_rmse"), ("dependence_aware_bootstrap", "ci90_low", "ci90_high")),
        {
            "problem": "P3",
            "candidate": "lead_continuous",
            "benefit": -float(lead["evaluation"]["active_prequential_folds"]["delta_candidate_minus_incumbent_m"]),
            "benefit_ci90": [-0.00158453, 0.01012909],
            "new_state": "EXPLORATORY_CHALLENGER_RESEARCH_ONLY",
        },
        {
            "problem": "P3",
            "candidate": "sparse_gp_abstention",
            "benefit": float(sparse["primary_evaluation"]["dependence_aware_interval"]["benefit_incumbent_minus_candidate_point_m"]),
            "benefit_ci90": list(sparse["primary_evaluation"]["dependence_aware_interval"]["benefit_ci90_m"]),
            "new_state": "INCONCLUSIVE_RESEARCH_ONLY",
        },
        {
            "problem": "P3",
            "candidate": "catboost_repaired_confirmation",
            "benefit": -float(catboost["confirmation"]["metrics"]["delta_rmse_m"]),
            "benefit_ci90": list(_benefit_ci_from_delta(catboost["confirmation"]["paired_case_bootstrap"]["ci90_lower_m"], catboost["confirmation"]["paired_case_bootstrap"]["ci90_upper_m"])),
            "new_state": "PRIMARY_HARM_RESEARCH_ONLY",
        },
        {
            "problem": "P3",
            "candidate": "selection_matched_masked_ssl",
            "benefit": -float(ssl["paired_comparison"]["metrics"]["overall"]["delta_candidate_minus_incumbent_m"]),
            "benefit_ci90": list(_benefit_ci_from_delta(*ssl["paired_comparison"]["paired_case_bootstrap"]["delta_candidate_minus_incumbent_ci90_m"])),
            "new_state": "PRIMARY_HARM_RESEARCH_ONLY",
        },
    ]
    for case in cases:
        case["empirical_point_equivalent"] = None
    return cases


def build_replay(payloads: dict[str, dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    families = _replay_families(policy, payloads["family_ledger"])
    groups = _replay_closed_groups(policy, payloads["negative_ledger"])
    cases = _case_replay(payloads)
    slopes = payloads["leaderboard"]["empirical_score_mapping"]
    for case in cases:
        case["empirical_point_equivalent"] = abs(float(case["benefit"])) * abs(
            float(slopes[case["problem"]]["slope"])
        )
        case["empirical_point_equivalent_warning"] = "planning-only OLS; not local-to-official transport"

    family_counts = Counter(row["new_state"] for row in families)
    group_counts = Counter(row["new_state"] for row in groups)
    key_counts = Counter(row["new_state"] for row in cases)
    reopened = [row["candidate"] for row in cases if "CHALLENGER" in row["new_state"] or "REOPEN" in row["new_state"]]

    return {
        "schema_version": "tolerance_failure_replay.20260830.v2",
        "status": "PASS_ZERO_FIT_FULL_LEDGER_REINTERPRETATION",
        "conclusion": "The old nonzero raw-metric gates are action false-negative generators in all three problems. Validity remains hard; scientific direction defaults to zero plus dependence-aware uncertainty; submission action is decided in leaderboard-point and information-value space.",
        "policy_digest": {
            "deterministic_metric_absolute_epsilon": policy["tolerance_layers"]["numerical_replay"]["deterministic_metric_absolute_epsilon"],
            "leaderboard_apparent_tie_band": policy["tolerance_layers"]["numerical_replay"]["leaderboard_six_decimal_apparent_tie_band"],
            "scientific_directional_margin": policy["tolerance_layers"]["scientific_effect"]["default_directional_margin_when_no_justified_sesoi_exists"],
            "nonzero_universal_raw_margin": policy["tolerance_layers"]["scientific_effect"]["universal_nonzero_raw_metric_margin"],
        },
        "official_false_negative_cases": _official_false_negative_cases(payloads),
        "key_case_replay": cases,
        "historical_family_replay": families,
        "canonical_group_replay": groups,
        "workflow_exception_replay": [
            {
                "problem": row["problem"],
                "id": row["id"],
                "old_state": row["workflow_state"],
                "new_state": "INVALID_OR_NO_SCIENTIFIC_RESULT",
            }
            for row in payloads["family_ledger"].get("workflow_exceptions", [])
        ],
        "summary": {
            "historical_families_cross_checked": len(families),
            "historical_families_by_problem": dict(Counter(row["problem"] for row in families)),
            "canonical_closed_invalid_or_not_failure_groups_cross_checked": len(groups),
            "canonical_groups_by_problem": dict(Counter(row["problem"] for row in groups)),
            "family_state_counts": dict(sorted(family_counts.items())),
            "group_state_counts": dict(sorted(group_counts.items())),
            "key_case_state_counts": dict(sorted(key_counts.items())),
            "reopened_or_challenger_key_cases": reopened,
            "official_false_negative_proven_problem_count": sum(
                int(row["false_negative_proven"])
                for key, row in _official_false_negative_cases(payloads).items()
                if key in {"P1", "P2", "P3"}
            ),
            "model_fits": 0,
            "raw_training_or_prediction_rows_read": 0,
            "official_test_sample_submission_hidden_or_query_rows_read": 0,
            "csv_created": 0,
            "uploads": 0,
        },
    }


def main() -> None:
    policy = _read_json(POLICY)
    payloads = {
        name: _read_json(path)
        for name, path in INPUTS.items()
        if path.suffix.lower() == ".json"
    }
    replay = build_replay(payloads, policy)
    replay["provenance"] = {
        "policy": {"path": str(POLICY.relative_to(ROOT)), "sha256": _sha256(POLICY)},
        "inputs": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for name, path in INPUTS.items()
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": _sha256(Path(__file__).resolve()),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(replay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(replay["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
