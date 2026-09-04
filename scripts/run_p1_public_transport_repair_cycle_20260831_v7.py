"""Exactly-once layer-2 drift-only P1 transport adapter.

The v6 attempt proved before fitting that its L1 inner calibration block was
one-class.  This preregistered successor learns from all anchor-negative source
layers, calibrates only the high-probability drift phenotype, and changes only
layer 2 at deployment.  The threshold is the exact marginal add-only F1 rule;
all other rows are a bit-for-bit anchor no-op.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p1_public_transport_repair_cycle_20260831_v6 as prior  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v7"
CONFIG_PATH = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
DELIVERY = Path(
    r"C:\Users\cedis\Downloads\해양 해커톤 제출용"
    r"\20260831_P1_PUBLIC_TRANSPORT_REPAIR_CYCLE_V7"
)
NAMES = {
    "row": "P1_1_L2_DRIFT_CALIBRATED_ET_ROW",
    "group_shrunk": "P1_2_L2_DRIFT_CALIBRATED_ET_GROUP_SHRUNK",
    "consensus": "P1_3_L2_DRIFT_CALIBRATED_ET_CONSENSUS",
}


def native(value: Any) -> Any:
    return prior.native(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any, *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as handle:
        json.dump(native(payload), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def source_fit_scope(frame: pd.DataFrame) -> np.ndarray:
    return frame["e150_prediction"].eq(0).to_numpy()


def drift_calibration_scope(frame: pd.DataFrame) -> np.ndarray:
    return source_fit_scope(frame) & frame["pmax"].ge(0.01).to_numpy()


def deployment_scope(frame: pd.DataFrame) -> np.ndarray:
    return drift_calibration_scope(frame) & frame["layer"].eq(2).to_numpy()


def install_frozen_scope() -> None:
    prior.base.source_training_mask = source_fit_scope
    prior.base.calibration_eligibility = drift_calibration_scope
    prior.base.deployment_eligibility = deployment_scope
    prior.NAMES = NAMES
    prior.DELIVERY = DELIVERY


def load_contract() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    root_gate = json.loads(prior.base.CALIBRATION_PATH.read_text(encoding="utf-8"))[
        "gates"
    ]["P1"]
    policy = config["decision_policy"]
    checks = [
        np.isclose(
            policy["worst_observed_public_transport_residual_points"],
            root_gate["worst_observed_transport_residual_points"],
            atol=1e-15,
        ),
        np.isclose(
            policy["minimum_raw_expected_point_delta_inclusive"],
            root_gate["minimum_uncalibrated_expected_points_delta"],
            atol=1e-15,
        ),
        np.isclose(
            policy["bootstrap_ci90_low_minimum"],
            root_gate["full_transport_metric_improvement_equivalent"],
            atol=1e-15,
        ),
    ]
    if not all(checks):
        raise RuntimeError("root transport calibration mismatch")
    return config


def precision_lower_bound(tp: int, fp: int, confidence: float) -> float:
    if tp <= 0:
        return 0.0
    return float(beta.ppf(1.0 - confidence, tp, fp + 1))


def component_precision_receipts(
    threshold_payload: dict[str, Any], confidence: float
) -> list[dict[str, Any]]:
    if "component_receipts" in threshold_payload:
        sources = [
            receipt
            for family in threshold_payload["component_receipts"].values()
            for receipt in family.values()
        ]
    else:
        sources = list(threshold_payload.values())
    records = []
    for receipt in sources:
        tp = int(receipt["calibration_true_positive_additions"])
        fp = int(receipt["calibration_false_positive_additions"])
        threshold = float(receipt["theoretical_add_only_threshold_strict"])
        lower = precision_lower_bound(tp, fp, confidence)
        records.append(
            {
                "outer_test_fold": receipt.get("outer_test_fold"),
                "true_positive_additions": tp,
                "false_positive_additions": fp,
                "confidence": confidence,
                "exact_one_sided_precision_lower_bound": lower,
                "required_marginal_precision": threshold,
                "pass": lower > threshold,
            }
        )
    return records


def apply_v7_gates(
    records: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    confidence = float(config["decision_policy"]["inner_precision_lcb_confidence"])
    for record in records:
        precision_receipts = component_precision_receipts(
            record["thresholds"], confidence
        )
        record["inner_precision_lower_bounds"] = precision_receipts
        record["gates"]["inner_station_pooled_precision_lcb_above_f1_half"] = all(
            item["pass"] for item in precision_receipts
        )
        record["gates"]["all_forward_blocks_nonnegative"] = all(
            item["delta_f1"] >= 0.0 for item in record["by_fold"].values()
        )
        record["gates"]["bootstrap_lcb_meets_full_transport_f1_equivalent"] = (
            record["day_block_bootstrap"]["ci90_low"]
            >= float(config["decision_policy"]["bootstrap_ci90_low_minimum"])
        )
        record["strict_internal_pass"] = bool(all(record["gates"].values()))
    return native(records)


def independent_qa(result: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "candidate_count_1_to_3": 1 <= len(result["candidates"]) <= 3,
        "historical_fit_budget": result["historical_fit_count"] <= 24,
        "total_fit_budget": result["fit_count"] <= 38,
        "all_add_only": all(item["add_only"] for item in result["candidates"]),
        "all_anchor_removals_zero": all(
            item["anchor_removals"] == 0 for item in result["candidates"]
        ),
        "all_forward_blocks_nonnegative_for_passes": all(
            (not item["strict_internal_pass"])
            or item["gates"]["all_forward_blocks_nonnegative"]
            for item in result["candidates"]
        ),
        "all_passes_clear_precision_lcb": all(
            (not item["strict_internal_pass"])
            or item["gates"]["inner_station_pooled_precision_lcb_above_f1_half"]
            for item in result["candidates"]
        ),
        "all_passes_clear_transport_f1_lcb": all(
            (not item["strict_internal_pass"])
            or item["gates"]["bootstrap_lcb_meets_full_transport_f1_equivalent"]
            for item in result["candidates"]
        ),
        "only_passes_materialized": {item["name"] for item in result["outputs"]}
        == {
            item["name"]
            for item in result["candidates"]
            if item["strict_internal_pass"]
        },
        "other_layers_no_op_contract": result["transport_scope"][
            "all_other_rows_exact_no_op"
        ],
        "official_read_only_after_pass": bool(result["outputs"])
        == bool(result["operations"]["official_covariate_reads"]),
        "hidden_truth_reads_zero": result["operations"]["hidden_truth_reads"] == 0,
        "uploads_zero": result["operations"]["uploads"] == 0,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def validate_only() -> dict[str, Any]:
    install_frozen_scope()
    config = load_contract()
    return {
        "status": "VALID",
        "config_sha256": sha256_file(CONFIG_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "candidate_count": len(config["candidates"]),
        "historical_fit_budget": 24,
        "total_fit_budget": 38,
    }


def execute() -> dict[str, Any]:
    if ARTIFACT.exists() or REPORT.exists():
        raise FileExistsError("exactly-once v7 path already exists")
    install_frozen_scope()
    config = load_contract()
    ARTIFACT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    started = time.perf_counter()
    write_json(
        ARTIFACT / "attempt_lock.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "pid": os.getpid(),
            "started_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "v6_terminal_failure_sha256": sha256_file(
                ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v6/terminal_failure.json"
            ),
        },
    )
    write_json(
        ARTIFACT / "progress.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "loading_historical_oof_only",
            "fit_count": 0,
            "performance_withheld_until_terminal": True,
        },
        exclusive=False,
    )
    historical, _ = prior.base.prior_cycle.p1_frame()
    historical, actual_features = prior.base.prior_p1.add_causal_features(historical)
    if sorted(set(prior.base.MODEL_FEATURES) - set(actual_features)):
        raise RuntimeError("frozen feature mismatch")
    records, historical_fits = prior.evaluate(historical, config)
    records = apply_v7_gates(records, config)
    write_json(
        ARTIFACT / "progress.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "internal_gate_complete",
            "fit_count": historical_fits,
            "pass_count": sum(item["strict_internal_pass"] for item in records),
            "performance_withheld_until_terminal": True,
        },
        exclusive=False,
    )
    outputs, official_access, deployment_fits = prior.materialize(
        historical, records, config
    )
    result = {
        "schema_version": "p1.public_transport_repair_cycle.20260831.v7",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "decision_policy": config["decision_policy"],
        "design_change_from_v6": config["design_change_from_v6"],
        "validation_contract": config["validation"],
        "transport_scope": config["transport_scope"],
        "candidates": records,
        "pass_count": sum(item["strict_internal_pass"] for item in records),
        "outputs": outputs,
        "historical_fit_count": historical_fits,
        "deployment_fit_count": deployment_fits,
        "fit_count": historical_fits + deployment_fits,
        "operations": {
            **official_access,
            "hidden_truth_reads": 0,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": sha256_file(CONFIG_PATH),
            "runner_sha256": sha256_file(Path(__file__)),
            "root_calibration_sha256": sha256_file(prior.base.CALIBRATION_PATH),
            "v6_failure_sha256": sha256_file(
                ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v6/terminal_failure.json"
            ),
        },
    }
    result["independent_qa"] = independent_qa(result)
    write_json(ARTIFACT / "result.json", result)
    write_json(REPORT / "independent-qa.json", result["independent_qa"])
    if outputs:
        write_json(DELIVERY / "SET_MANIFEST.json", result)
    write_json(
        ARTIFACT / "progress.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "phase": "terminal",
            "status": result["status"],
            "fit_count": result["fit_count"],
            "pass_count": result["pass_count"],
            "outputs": len(outputs),
        },
        exclusive=False,
    )
    return native(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--validate-only", action="store_true")
    group.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_only(), ensure_ascii=False, indent=2))
        return 0
    try:
        result = execute()
    except Exception as exc:
        if ARTIFACT.exists():
            write_json(
                ARTIFACT / "terminal_failure.json",
                {
                    "status": "TERMINAL_TECHNICAL_FAILURE",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "automatic_restart_allowed": False,
                    "hidden_truth_reads": 0,
                    "uploads": 0,
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
