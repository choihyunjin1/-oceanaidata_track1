"""Read-only methodology audit for the prospective P1 transport gate v4.

This script fits no model, reads no official rows, and never reclassifies v28.
It checks the official-metric contract preserved in the repository, summarizes
only aggregate train-side support, and verifies the future-only gate partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
POLICY_ID = "p1_prospective_transport_gate_20260831_v4"
POLICY_PATH = ROOT / "configs/goals" / f"{POLICY_ID}.json"
OUTPUT_DIR = ROOT / "reports" / POLICY_ID
TRAIN_DISTRIBUTION_PATH = OUTPUT_DIR / "train-only-distribution.json"
QA_PATH = OUTPUT_DIR / "independent-qa.json"
V28_RESULT_PATH = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v28/result.json"
ANCHOR_PATH = ROOT / "artifacts/p1_current_router_oof_anchor_v1/anchor.parquet"
OFFICIAL_LEDGER_PATH = (
    ROOT
    / "reports/parallel_internal_pass_registry_20260831_v1/official-submission-results-20260831.json"
)
DEADLINE_RESULTS_PATH = ROOT / "reports/deadline_submission_results_20260828_v1/official-results.md"
CALIBRATION_PATH = ROOT / "reports/public_transport_calibration_20260831_v3/calibration.json"
METRIC_POLICY_PATH = ROOT / "configs/goals/metric_aligned_gate_recalibration_20260830_v1.json"
TOLERANCE_POLICY_PATH = ROOT / "configs/goals/tolerance_recalibration_and_failure_replay_20260830_v2.json"
EVALUATION_FOLDS = ("2025_q3", "2025_q4")
FORMER_DAILY_CAP = 0.005


class AuditError(RuntimeError):
    """Raised when the prospective policy or evidence contract is inconsistent."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(native(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise AuditError("quantile input must be finite and non-empty")
    probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    return {
        f"p{int(round(probability * 100)):02d}": float(value)
        for probability, value in zip(probabilities, np.quantile(array, probabilities), strict=True)
    }


def prospective_decision_state(
    *,
    validity_pass: bool,
    hard_gate_results: dict[str, bool],
    diagnostic_results: dict[str, bool],
) -> str:
    """Return a v4 state without allowing diagnostics to reverse the hard decision."""

    if not validity_pass:
        return "QA_BLOCKED"
    if not hard_gate_results or not all(hard_gate_results.values()):
        return "NO_PASS_PRIMARY_GATE"
    if diagnostic_results and not all(diagnostic_results.values()):
        return "PASS_PRIMARY_WITH_TRANSPORT_WARNING"
    return "PASS_PRIMARY_NO_WARNING"


def load_policy() -> dict[str, Any]:
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("policy_id") != POLICY_ID:
        raise AuditError("prospective gate identity mismatch")
    frozen = policy["frozen_input_hashes"]
    mismatches: dict[str, dict[str, str]] = {}
    for relative, expected in frozen.items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            mismatches[relative] = {"expected": expected, "actual": actual}
    if mismatches:
        raise AuditError(f"frozen evidence hash mismatch: {mismatches}")
    return policy


def load_train_only_surface() -> tuple[pd.DataFrame, pd.DataFrame]:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import run_full_internal_submission_cycle_20260831_v2 as source_cycle

    historical, _ = source_cycle.p1_frame()
    historical = historical.loc[:, ["station", "year", "layer", "time", "fold", "label_base"]].copy()
    historical["time"] = pd.to_datetime(historical["time"], utc=True)
    anchor = pd.read_parquet(
        ANCHOR_PATH,
        columns=["station", "year", "layer", "time", "fold", "current_router_prediction"],
    )
    anchor["time"] = pd.to_datetime(anchor["time"], utc=True)
    keys = ["station", "year", "layer", "time", "fold"]
    if historical.duplicated(keys).any() or anchor.duplicated(keys).any():
        raise AuditError("historical or anchor key duplication")
    merged = historical.merge(anchor, on=keys, how="inner", validate="one_to_one")
    if len(merged) != len(historical) or len(merged) != len(anchor):
        raise AuditError("historical/anchor alignment changed")
    evaluated = merged.loc[merged["fold"].isin(EVALUATION_FOLDS)].copy()
    return merged, evaluated


def distribution_summary(full: pd.DataFrame, evaluated: pd.DataFrame) -> dict[str, Any]:
    evaluated["kst_day"] = (
        evaluated["time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    )
    by_day = evaluated.groupby("kst_day", observed=True).agg(
        rows=("label_base", "size"),
        truth_positive_rows=("label_base", "sum"),
        anchor_positive_rows=("current_router_prediction", "sum"),
    )
    by_day["truth_positive_rate"] = by_day["truth_positive_rows"] / by_day["rows"]
    by_day["anchor_positive_rate"] = by_day["anchor_positive_rows"] / by_day["rows"]
    by_day["former_cap_integer_actions"] = np.floor(FORMER_DAILY_CAP * by_day["rows"]).astype(int)

    by_cell = evaluated.groupby(["station", "layer"], observed=True).agg(
        rows=("label_base", "size"),
        truth_positive_rows=("label_base", "sum"),
        anchor_positive_rows=("current_router_prediction", "sum"),
    )
    by_cell["truth_positive_rate"] = by_cell["truth_positive_rows"] / by_cell["rows"]
    by_cell["anchor_positive_rate"] = by_cell["anchor_positive_rows"] / by_cell["rows"]
    minimum_day_rows = int(by_day["rows"].min())
    minimum_nonzero_fraction = 1.0 / minimum_day_rows
    cell_rate_minimum = float(by_cell["truth_positive_rate"].min())
    cell_rate_maximum = float(by_cell["truth_positive_rate"].max())

    return {
        "schema_version": "p1.prospective_transport_gate.v4.train_distribution.1",
        "scope": "historical train-only Q2-Q4 surface; gate diagnostics use Q3-Q4 evaluation folds",
        "full_surface": {
            "rows": int(len(full)),
            "fold_rows": {
                str(key): int(value)
                for key, value in full["fold"].value_counts().sort_index().items()
            },
            "truth_positive_rate": float(full["label_base"].mean()),
            "anchor_positive_rate": float(full["current_router_prediction"].mean()),
        },
        "q3_q4_evaluation_surface": {
            "rows": int(len(evaluated)),
            "unique_kst_days": int(len(by_day)),
            "station_layer_cells": int(len(by_cell)),
            "truth_positive_rate": float(evaluated["label_base"].mean()),
            "anchor_positive_rate": float(evaluated["current_router_prediction"].mean()),
            "day_row_count_quantiles": quantiles(by_day["rows"]),
            "day_truth_positive_rate_quantiles": quantiles(by_day["truth_positive_rate"]),
            "day_anchor_positive_rate_quantiles": quantiles(by_day["anchor_positive_rate"]),
            "station_layer_row_count_quantiles": quantiles(by_cell["rows"]),
            "station_layer_truth_positive_count_quantiles": quantiles(
                by_cell["truth_positive_rows"]
            ),
            "station_layer_truth_positive_rate_quantiles": quantiles(
                by_cell["truth_positive_rate"]
            ),
            "station_layer_positive_rate_max_to_min_ratio": (
                cell_rate_maximum / cell_rate_minimum
            ),
        },
        "former_daily_0_005_cap_integer_audit": {
            "threshold": FORMER_DAILY_CAP,
            "minimum_day_rows": minimum_day_rows,
            "maximum_day_rows": int(by_day["rows"].max()),
            "minimum_nonzero_changed_fraction": minimum_nonzero_fraction,
            "days_forcing_zero_integer_actions": int(
                (by_day["former_cap_integer_actions"] == 0).sum()
            ),
            "integer_action_cap_quantiles": quantiles(by_day["former_cap_integer_actions"]),
            "conclusion": "A fixed per-day percentage cap is denominator-sensitive and discontinuous; it is not a calibrated proxy for pooled F1 or transport loss.",
        },
        "privacy": {
            "row_values_persisted": 0,
            "aggregate_only": True,
            "official_rows_read": 0,
            "hidden_truth_rows_read": 0,
        },
    }


def official_probe_summary() -> dict[str, Any]:
    ledger = json.loads(OFFICIAL_LEDGER_PATH.read_text(encoding="utf-8"))
    p1 = [item for item in ledger["submissions"] if item["problem"] == "P1"]
    deadline = DEADLINE_RESULTS_PATH.read_text(encoding="utf-8")
    deadline_rows = re.findall(
        r"\|\s*\d+\s*\|\s*P1\s*\|.*?\|\s*F1\s+([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*\+([0-9.]+)\s*\|",
        deadline,
    )
    if len(p1) != 1 or len(deadline_rows) != 3:
        raise AuditError("P1 aggregate official probe ledger shape changed")
    return {
        "20260831_p1_probe": {
            "candidate": p1[0]["candidate"],
            "internal_delta_f1": p1[0]["internal_delta_f1"],
            "expected_central_points_delta": p1[0]["expected_points_delta"]["central"],
            "official_points_delta_vs_best": p1[0]["official_points_delta_vs_best"],
            "outcome": p1[0]["outcome"],
        },
        "20260828_p1_probe_count": len(deadline_rows),
        "20260828_public_f1_values": [float(item[0]) for item in deadline_rows],
        "20260828_public_point_values": [float(item[1]) for item in deadline_rows],
        "methodological_use": "Aggregate probes calibrate overall historical-to-Public transport risk. They contain no evidence that a universal per-day cap or every-slice nonregression veto predicts Public F1.",
        "official_rows_read": 0,
    }


def audit_workspace() -> dict[str, Any]:
    policy = load_policy()
    full, evaluated = load_train_only_surface()
    distribution = distribution_summary(full, evaluated)
    probes = official_probe_summary()
    v28 = json.loads(V28_RESULT_PATH.read_text(encoding="utf-8"))
    metric_policy = json.loads(METRIC_POLICY_PATH.read_text(encoding="utf-8"))
    tolerance_policy = json.loads(TOLERANCE_POLICY_PATH.read_text(encoding="utf-8"))
    must_read = (ROOT / "00_MUST_READ_FIRST.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    hard = policy["hard_scientific_and_transport_gates"]
    diagnostics = policy["diagnostic_only_transport_checks"]
    frozen_text = json.dumps(policy, sort_keys=True)

    checks = {
        "official_contract_binary_f1_preserved": "binary F1" in must_read
        and "binary F1" in readme,
        "anomaly_type_not_ranked_preserved": "anomaly_type은 순위에 반영되지 않고" in must_read,
        "governing_policy_primary_metric_pooled": metric_policy["problem_units"]["P1"][
            "paired_unit"
        ].startswith("joint KST-day")
        and metric_policy["decision_hierarchy"]["level_1_primary_utility"]["P1"].startswith(
            "pooled row-level binary micro-F1"
        ),
        "governing_policy_slices_diagnostic_by_default": metric_policy["decision_hierarchy"][
            "level_3_transport_diagnostics"
        ].startswith("Window, season, station, layer"),
        "tolerance_policy_transport_hard_only_when_justified": "official mixture"
        in tolerance_policy["tolerance_layers"]["transport"]["hard_only_if"],
        "future_only_and_v28_retro_pass_forbidden": policy["effective_scope"][
            "future_preregistrations_only"
        ]
        and policy["effective_scope"]["p1_v28_retroactive_pass_forbidden"],
        "v28_original_state_unchanged": v28["candidate"]["strict_internal_pass"] is False
        and set(v28["candidate"]["gates"])
        >= {
            "each_kst_day_changed_fraction_at_most_0_005",
            "each_supported_station_layer_nonnegative",
        },
        "only_two_gate_roles_changed": set(policy["audit_isolation"]["only_role_changes_in_v4"])
        == {
            "maximum_changed_fraction_any_kst_day",
            "minimum_each_supported_station_layer_delta_f1",
        },
        "daily_and_slice_checks_not_hard": "maximum_changed_fraction_any_kst_day" not in hard
        and "minimum_each_supported_station_layer_delta_f1" not in hard
        and set(diagnostics)
        == {
            "maximum_changed_fraction_any_kst_day",
            "minimum_each_supported_station_layer_delta_f1",
        },
        "minimum_plus_0_01_calibrated_retained": np.isclose(
            hard["minimum_calibrated_expected_point_delta_inclusive"], 0.01
        ),
        "v3_transport_raw_gate_retained": np.isclose(
            hard["minimum_raw_expected_point_delta_inclusive"],
            calibration["p1"]["prospective_minimum_raw_expected_points_delta"],
        ),
        "pooled_dependent_bootstrap_retained": hard["dependent_resampling_unit"].startswith(
            "joint KST-day"
        )
        and hard["dependent_bootstrap_ci90_low_strictly_above_f1"] == 0.0
        and hard["dependent_bootstrap_probability_improved_minimum_inclusive"] == 0.8,
        "anchor_removal_zero_retained": hard["anchor_removals_required"] == 0,
        "other_v28_gates_held_constant": set(
            policy["audit_isolation"]["unchanged_from_v28_pending_separate_audit"]
        )
        == {
            "minimum_each_q3_q4_delta_f1_inclusive",
            "maximum_overall_changed_fraction_inclusive",
            "maximum_addition_concentration_any_station_layer_quarter_inclusive",
        },
        "no_v28_outcome_specific_thresholds_in_policy": all(
            token not in frozen_text
            for token in (
                "0.03306565895134624",
                "-0.011480277632946234",
                "0.008746570712812818",
                "0.22708372220708112",
            )
        ),
        "train_only_rows_match": distribution["full_surface"]["rows"] == 421_032,
        "former_daily_cap_has_integer_discontinuity": distribution[
            "former_daily_0_005_cap_integer_audit"
        ]["days_forcing_zero_integer_actions"]
        > 0
        and distribution["former_daily_0_005_cap_integer_audit"][
            "minimum_nonzero_changed_fraction"
        ]
        > FORMER_DAILY_CAP,
        "station_layer_support_is_heterogeneous": distribution["q3_q4_evaluation_surface"][
            "station_layer_positive_rate_max_to_min_ratio"
        ]
        > 1.0,
        "aggregate_public_probes_present_without_row_access": probes["20260828_p1_probe_count"]
        == 3
        and probes["official_rows_read"] == 0,
        "model_official_hidden_csv_upload_zero": all(
            policy["boundaries"][key] == 0
            for key in (
                "model_fits",
                "official_rows_read",
                "hidden_truth_rows_read",
                "submission_csv_created",
                "uploads",
            )
        ),
        "diagnostics_cannot_reverse_hard_pass": prospective_decision_state(
            validity_pass=True,
            hard_gate_results={"pooled": True, "bootstrap": True, "transport": True},
            diagnostic_results={"daily": False, "slice": False},
        )
        == "PASS_PRIMARY_WITH_TRANSPORT_WARNING",
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    payload = {
        "schema_version": "p1.prospective_transport_gate.v4.independent_qa.1",
        "policy_id": POLICY_ID,
        "status": status,
        "conclusion": (
            "BOTH_EXTRA_V28_GATES_OVERSTRICT_AS_HARD_VETOES_PROSPECTIVELY_DIAGNOSTIC_ONLY"
            if status == "PASS"
            else "QA_BLOCKED"
        ),
        "checks": checks,
        "prospective_decision": {
            "maximum_changed_fraction_any_kst_day": "DIAGNOSTIC_WARNING_ONLY",
            "minimum_each_supported_station_layer_delta_f1": "DIAGNOSTIC_WARNING_ONLY",
            "v28_original_decision": "NO_GO_SAFETY_GATES",
            "v28_retroactive_reclassification": False,
        },
        "train_only_distribution": distribution,
        "aggregate_official_probe_evidence": probes,
        "hashes": {
            "policy_sha256": sha256_file(POLICY_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "v28_result_sha256": sha256_file(V28_RESULT_PATH),
            "anchor_sha256": sha256_file(ANCHOR_PATH),
            "calibration_v3_sha256": sha256_file(CALIBRATION_PATH),
            "official_aggregate_ledger_sha256": sha256_file(OFFICIAL_LEDGER_PATH),
        },
        "operations": {
            "model_fits": 0,
            "historical_train_rows_read": int(len(full)),
            "aggregate_official_probe_ledgers_read": 2,
            "official_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit:
        raise SystemExit("pass --audit; this runner has no model or official execution mode")
    payload = audit_workspace()
    write_json(TRAIN_DISTRIBUTION_PATH, payload["train_only_distribution"])
    write_json(QA_PATH, payload)
    print(json.dumps(native(payload), indent=2, sort_keys=True))
    if payload["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
