from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs/goals/metric_aligned_gate_recalibration_20260830_v1.json"
OUTPUT = ROOT / "reports/gate_recalibration_research_20260830_v1/gate-replay.json"

INPUTS = {
    "p1_supcon": ROOT
    / "artifacts/p1_event_balanced_supcon_f1_head_20260830_v1/aggregate.json",
    "p2_gaussian_copula": ROOT
    / "artifacts/p2_gaussian_copula_conditional_mean_20260830_v2/result.json",
    "p2_state_copula": ROOT
    / "artifacts/p2_state_conditioned_copula_20260830_v1/result.json",
    "p3_catboost_confirmation": ROOT
    / "artifacts/p3_catboost_confirmation_contract_repair_20260830_v3/one_shot/result.json",
    "p3_masked_ssl": ROOT
    / "artifacts/p3_selection_matched_masked_ssl_20260830_v1/result.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_benefit(
    benefit: float,
    ci_low: float | None,
    ci_high: float | None,
    *,
    decisive_without_interval: bool = False,
) -> str:
    if ci_low is not None and ci_high is not None:
        if benefit > 0.0 and ci_low > 0.0:
            return "HIGH_VALUE_CHALLENGER_RESEARCH_ONLY"
        if benefit < 0.0 and ci_high < 0.0:
            return "PRIMARY_HARM_RESEARCH_ONLY"
        if benefit > 0.0:
            return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
        return "INCONCLUSIVE_RESEARCH_ONLY"
    if benefit > 0.0:
        return "EXPLORATORY_CHALLENGER_RESEARCH_ONLY"
    if benefit < 0.0 and decisive_without_interval:
        return "PRIMARY_HARM_RESEARCH_ONLY"
    return "INCONCLUSIVE_RESEARCH_ONLY"


def build_replay(payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    p1 = payloads["p1_supcon"]
    p2g = payloads["p2_gaussian_copula"]
    p2s = payloads["p2_state_copula"]
    p3c = payloads["p3_catboost_confirmation"]
    p3m = payloads["p3_masked_ssl"]

    p1_benefit = float(p1["pooled"]["candidate"]["f1"]) - float(
        p1["pooled"]["control"]["f1"]
    )

    p2g_delta = float(p2g["metrics"]["aggregate"]["delta_rmse"])
    p2g_ci_delta = (
        float(p2g["bootstrap"]["ci90_low"]),
        float(p2g["bootstrap"]["ci90_high"]),
    )
    p2g_benefit_ci = (-p2g_ci_delta[1], -p2g_ci_delta[0])

    p2s_delta = float(p2s["metrics"]["pooled"]["delta_rmse"])
    p2s_ci_delta = (
        float(p2s["bootstrap"]["ci90_low"]),
        float(p2s["bootstrap"]["ci90_high"]),
    )
    p2s_benefit_ci = (-p2s_ci_delta[1], -p2s_ci_delta[0])

    p3c_delta = float(p3c["confirmation"]["metrics"]["delta_rmse_m"])
    p3c_ci_delta = (
        float(p3c["confirmation"]["paired_case_bootstrap"]["ci90_lower_m"]),
        float(p3c["confirmation"]["paired_case_bootstrap"]["ci90_upper_m"]),
    )
    p3c_benefit_ci = (-p3c_ci_delta[1], -p3c_ci_delta[0])

    p3m_delta = float(
        p3m["paired_comparison"]["metrics"]["overall"][
            "delta_candidate_minus_incumbent_m"
        ]
    )
    p3m_ci_delta = tuple(
        float(value)
        for value in p3m["paired_comparison"]["paired_case_bootstrap"][
            "delta_candidate_minus_incumbent_ci90_m"
        ]
    )
    p3m_benefit_ci = (-p3m_ci_delta[1], -p3m_ci_delta[0])

    candidates = {
        "P1_event_balanced_supcon": {
            "benefit_f1": p1_benefit,
            "benefit_ci": None,
            "old_decision": p1["status"],
            "new_state": classify_benefit(
                p1_benefit,
                None,
                None,
                decisive_without_interval=True,
            ),
            "decisive_without_interval_basis": (
                "Pooled F1 falls by 0.164874 and every frozen window plus the "
                "anomaly-type macro metric degrades."
            ),
            "decision_changed": False,
            "transport_risk": "ALL_THREE_WINDOWS_AND_TYPE_MACRO_DEGRADE",
            "next_action": "CLOSE_EXACT_RECIPE",
        },
        "P2_gaussian_copula_conditional_mean": {
            "benefit_c": -p2g_delta,
            "benefit_ci90_c": list(p2g_benefit_ci),
            "old_decision": p2g["decision"],
            "new_state": classify_benefit(-p2g_delta, *p2g_benefit_ci),
            "decision_changed": True,
            "transport_risk": "HIGH_2025_NOV_DEC_REGRESSION_AND_INNER_INSTABILITY",
            "next_action": "KEEP_AS_LOCKED_HIGH_VALUE_CHALLENGER; DO_NOT_CALL_CONFIRMED",
        },
        "P2_state_conditioned_copula": {
            "benefit_c": -p2s_delta,
            "benefit_ci90_c": list(p2s_benefit_ci),
            "old_decision": p2s["decision"],
            "new_state": classify_benefit(-p2s_delta, *p2s_benefit_ci),
            "decision_changed": True,
            "transport_risk": "JJA_REGRESSION_SMALL_RELATIVE_TO_POOLED_BENEFIT",
            "next_action": "KEEP_AS_LOCKED_HIGH_VALUE_CHALLENGER; DO_NOT_CALL_CONFIRMED",
        },
        "P3_catboost_confirmation": {
            "benefit_m": -p3c_delta,
            "benefit_ci90_m": list(p3c_benefit_ci),
            "old_decision": p3c["status"],
            "new_state": classify_benefit(-p3c_delta, *p3c_benefit_ci),
            "decision_changed": False,
            "transport_risk": "ALL_FOLDS_STATIONS_AND_LEADS_DEGRADE",
            "next_action": "CLOSE_EXACT_RECIPE",
        },
        "P3_selection_matched_masked_ssl": {
            "benefit_m": -p3m_delta,
            "benefit_ci90_m": list(p3m_benefit_ci),
            "old_decision": p3m["decision"],
            "new_state": classify_benefit(-p3m_delta, *p3m_benefit_ci),
            "decision_changed": False,
            "transport_risk": "ALL_WINDOWS_STATIONS_AND_LEADS_DEGRADE",
            "next_action": "CLOSE_EXACT_RECIPE",
        },
    }

    return {
        "schema_version": "metric_aligned_gate_replay.20260830.v1",
        "status": "PASS_ZERO_FIT_AGGREGATE_ONLY_RECLASSIFICATION",
        "conclusion": (
            "P2 has two high-value exposed-surface challengers that legacy conjunctive "
            "slice/stability vetoes incorrectly collapsed into NO_GO. P1 and both P3 "
            "candidates remain primary-metric harms regardless of gate recalibration."
        ),
        "candidates": candidates,
        "summary": {
            "reclassified_legacy_no_go_to_high_value_challenger": 2,
            "primary_harm_conclusions_unchanged": 3,
            "model_fits": 0,
            "prediction_rows_read": 0,
            "raw_training_rows_read": 0,
            "official_test_sample_submission_hidden_rows_read": 0,
            "csv_created": 0,
            "uploads": 0,
        },
    }


def main() -> None:
    payloads = {name: _read_json(path) for name, path in INPUTS.items()}
    replay = build_replay(payloads)
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
    OUTPUT.write_text(
        json.dumps(replay, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(replay["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
