"""Independently verify the terminal P2 boundary-bridge contract audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_p2_boundary_residual_bridge_20260829_v1 as experiment  # noqa: E402


def verify() -> dict[str, Any]:
    config = experiment.load_config()
    output = ROOT / config["output"]["directory"]
    result_path = output / config["output"]["result"]
    qa_path = output / config["output"]["independent_qa"]
    if qa_path.exists():
        raise FileExistsError(qa_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    collisions = {
        (item["block"], item["side"], item["overlap_start"], item["overlap_stop"])
        for item in result["hidden_target_collisions"]
    }
    expected = {
        (
            "2025_jul_aug",
            "right",
            "2025-09-01T00:00:00+09:00",
            "2025-09-04T00:00:00+09:00",
        ),
        (
            "2025_nov_dec",
            "left",
            "2025-10-29T00:00:00+09:00",
            "2025-11-01T00:00:00+09:00",
        ),
    }
    zero_read_fields = (
        "official_hidden_target_rows_read",
        "official_test_index_rows_read",
        "official_sample_submission_rows_read",
        "official_submission_rows_read",
        "source_observation_rows_read",
    )
    zero_output_fields = (
        "bridge_fit_count",
        "prediction_rows_generated",
        "prediction_files_generated",
        "model_files_generated",
        "csv_files_generated",
    )
    checks = {
        "terminal_decision": result["decision"] == "NO_GO_CONTRACT_LEAKAGE",
        "family_closed": result["family_status"] == "CLOSED_NO_RETRY",
        "one_shot": int(result["execution_count"]) == 1,
        "exact_hidden_collisions": collisions == expected,
        "six_required_two_blocked": int(result["required_flanks"]) == 6
        and int(result["blocked_flanks"]) == 2,
        "no_data_rows_read": all(int(result[key]) == 0 for key in zero_read_fields),
        "no_data_paths_opened": result["data_paths_opened"] == [],
        "no_prediction_model_csv": all(int(result[key]) == 0 for key in zero_output_fields),
        "metric_gate_not_evaluated": result["metric_gate_evaluated"] is False
        and all(value is None for value in result["gate_checks"].values()),
        "contract_not_reinterpreted": result["contract_reinterpreted"] is False
        and result["internal_first_or_last_72h_used"] is False,
        "no_bridge_or_projector_application": result["smoothstep_applied"] is False
        and result["projector_applied"] is False,
        "no_submission_action": result["submission_generated_or_uploaded"] is False,
    }
    qa = {
        "schema_version": "p2.boundary_residual_bridge.independent_qa.v1",
        "experiment_id": experiment.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": result["decision"],
        "checks": checks,
        "result_sha256": experiment.sha256_file(result_path),
        "official_or_source_rows_read": 0,
        "csv_generated_or_uploaded": False,
        "model_or_prediction_generated": False,
        "model_or_prediction_rerun": False,
    }
    if not all(checks.values()):
        raise experiment.ContractError(f"independent QA failed: {checks}")
    experiment.atomic_json(qa_path, qa)
    return qa


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(verify(), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
