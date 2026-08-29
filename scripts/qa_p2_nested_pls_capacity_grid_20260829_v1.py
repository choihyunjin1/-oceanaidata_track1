"""Independent aggregate-only QA for the sealed P2 nested PLS capacity grid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for directory in (ROOT, SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts import (  # noqa: E402
    run_p2_alpha50_supervised_rank1_functional_residual_20260828_v1 as base,
)

EXPERIMENT_ID = "p2_nested_pls_capacity_grid_20260829_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def qa(artifact: Path) -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    commitment_path = artifact / "prediction_commitment.json"
    grid_path = artifact / "inner_grid_summary.json"
    result_path = artifact / "result.json"
    for path in (commitment_path, grid_path, result_path):
        require(path.is_file(), f"missing artifact: {path}")
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    require(commitment["experiment_id"] == EXPERIMENT_ID, "commitment ID drifted")
    require(result["experiment_id"] == EXPERIMENT_ID, "result ID drifted")
    require(commitment["config_sha256"] == base.sha256_file(CONFIG), "config hash drifted")
    require(
        commitment["grid_summary_sha256"] == base.sha256_file(grid_path),
        "grid summary hash drifted",
    )
    require(grid["grid_points_per_outer"] == 243, "grid point count drifted")
    require(len(grid["outer"]) == 3, "outer grid count drifted")
    require(all(len(records) == 243 for records in grid["outer"].values()), "outer grid incomplete")
    fit_counts = result["fit_counts"]
    require(fit_counts["inner_pls_fits"] == 81, "inner fit count drifted")
    require(fit_counts["outer_pls_fits"] == 3, "outer fit count drifted")
    require(fit_counts["total_pls_fits"] == 84, "total fit count drifted")
    require(
        result["evaluation_counts"]["outer_grid_points_evaluated"] == 729,
        "candidate evaluation count drifted",
    )
    require(
        result["evaluation_counts"]["rotation_point_evaluations"] == 2187,
        "rotation-point evaluation count drifted",
    )
    for fold, record in commitment["outputs"].items():
        path = ROOT / record["path"]
        require(path.is_file(), f"missing prediction: {fold}")
        require(path.stat().st_size == record["bytes"], f"prediction bytes drifted: {fold}")
        require(base.sha256_file(path) == record["sha256"], f"prediction hash drifted: {fold}")
        with np.load(path, allow_pickle=False) as payload:
            require("truth" not in payload.files, f"truth leaked into prediction: {fold}")
            require(len(payload["candidate"]) == record["rows"], f"row count drifted: {fold}")
            require(np.isfinite(payload["candidate"]).all(), f"non-finite prediction: {fold}")
    require(not list(artifact.rglob("*.csv")), "CSV created in artifact")
    require(commitment["outer_truth_rows_read_before_commitment"] == 0, "outer truth read early")
    require(commitment["official_hidden_gap_rows_read"] == 0, "hidden gap read")
    require(commitment["official_test_sample_submission_rows_read"] == 0, "official rows read")
    require(result["official_hidden_gap_rows_read"] == 0, "result hidden gap read")
    require(result["official_test_sample_submission_rows_read"] == 0, "result official rows read")
    require(result["submission_generated_or_uploaded"] is False, "submission action occurred")
    require(
        all(int(info.get("num_threads", 0)) <= 4 for info in result["runtime"]["threadpool_info"]),
        "runtime threadpool exceeded four",
    )
    expected_decision = (
        "GO_LOCAL_ONLY_NO_UPLOAD"
        if all(result["gate_checks"].values())
        else "NO_GO_CLOSE_FAMILY"
    )
    require(result["decision"] == expected_decision, "decision disagrees with exact gates")
    report = ROOT / config["report_directory"] / "report-source.md"
    require(report.is_file(), "report source missing")
    return {
        "schema_version": "p2.nested_pls_capacity_grid.independent_qa.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS",
        "decision": result["decision"],
        "grid_points_per_outer": 243,
        "candidate_evaluations": 729,
        "rotation_point_evaluations": 2187,
        "total_pls_fits": 84,
        "official_hidden_test_sample_submission_rows_read": 0,
        "csv_or_upload_performed": False,
        "config_sha256": base.sha256_file(CONFIG),
        "commitment_sha256": base.sha256_file(commitment_path),
        "grid_summary_sha256": base.sha256_file(grid_path),
        "result_sha256": base.sha256_file(result_path),
        "report_sha256": base.sha256_file(report),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        type=Path,
        default=ROOT / "artifacts" / EXPERIMENT_ID,
    )
    arguments = parser.parse_args()
    artifact = arguments.artifact.expanduser().resolve()
    result = qa(artifact)
    output = artifact / "independent_qa.json"
    base.atomic_json(output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
