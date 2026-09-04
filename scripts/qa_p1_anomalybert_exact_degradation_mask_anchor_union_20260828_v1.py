"""Independent aggregate-only QA for the sealed P1 degradation-mask pilot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "scripts/run_p1_anomalybert_exact_degradation_mask_anchor_union_20260828_v1.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_anomalybert_runner_qa", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import sealed runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".partial"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _smoke_qa(runner: Any, config: dict[str, Any]) -> dict[str, Any]:
    contract = runner.validate_contract(config)
    smoke_path = ROOT / config["artifacts"]["smoke_directory"] / "smoke.json"
    smoke = _read_json(smoke_path)
    checks = {
        "contract_pass": contract["status"] == "PASS",
        "smoke_pass": smoke["status"] == "PASS",
        "cpu_finite": smoke["cpu"]["finite"] is True,
        "cpu_exact_shape": smoke["cpu"]["output_shape"] == [2, 1024],
        "gpu_finite_or_skipped": smoke["gpu"].get("finite") is True
        or smoke["gpu"].get("skipped") is True,
        "model_fit_zero": smoke["model_fit_count"] == 0,
        "scientific_rows_zero": smoke["scientific_rows_read"] == 0,
        "q2_deferred": contract["inputs"]["q2_truth_and_keys"]["deferred"] is True,
        "q2_rows_zero": smoke["q2_truth_rows_read"] == 0,
        "q3_q4_rows_zero": smoke["q3_q4_rows_read"] == 0,
        "official_rows_zero": smoke["official_test_sample_submission_rows_read"] == 0,
        "no_submission_or_upload": smoke["submission_generated_or_uploaded"] is False,
    }
    return {
        "schema_version": "p1.anomalybert_exact_degradation_mask.smoke_qa.v1",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "smoke_sha256": runner.sha256_file(smoke_path),
        "config_sha256": runner.sha256_file(runner.CONFIG),
        "module_sha256": runner.sha256_file(runner.MODULE),
        "runner_sha256": runner.sha256_file(runner.RUNNER),
        "model_fit_count": 0,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }


def _final_qa(runner: Any, config: dict[str, Any]) -> dict[str, Any]:
    contract = runner.validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    result_path = output / "result.json"
    manifest_path = output / "manifest.json"
    checkpoint_path = output / "best_checkpoint.pt"
    preflight = _read_json(output / "preflight.json")
    result = _read_json(result_path)
    manifest = _read_json(manifest_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    history = result["training"]["history"]
    best = max(
        history,
        key=lambda row: (
            float(row["synthetic_validation_macro_raw_f1"]),
            -int(row["epoch"]),
        ),
    )
    manifest_hashes_match = True
    for relative, record in manifest["files"].items():
        path = output / relative if "/" not in relative and "\\" not in relative else ROOT / relative
        manifest_hashes_match &= (
            path.is_file()
            and path.stat().st_size == int(record["bytes"])
            and runner.sha256_file(path) == record["sha256"]
        )
    artifacts = [path for path in output.rglob("*") if path.is_file()]
    forbidden_artifacts = [
        str(path.relative_to(output))
        for path in artifacts
        if path.suffix.lower() == ".csv" or "submission" in path.name.lower()
    ]
    access = result["access"]
    fidelity_pass = result["synthetic_fidelity"]["gate_pass"] is True
    calibration_opened = "calibration" in result
    qualification_opened = "qualification" in result
    q2_opened = "q2_outer" in result
    sequential_access = (
        (fidelity_pass or not calibration_opened)
        and (calibration_opened or access["calibration_truth_rows_read"] == 0)
        and (
            qualification_opened
            or access["qualification_truth_rows_read"] == 0
        )
        and (q2_opened or access["q2_truth_rows_read"] == 0)
    )
    if calibration_opened and not result["calibration"]["gate_pass"]:
        sequential_access &= not qualification_opened and not q2_opened
    if qualification_opened and not result["qualification"]["gate_pass"]:
        sequential_access &= not q2_opened
    checks = {
        "independent_contract_pass": contract["status"] == "PASS",
        "preflight_pass_before_fit": preflight["status"] == "PASS"
        and preflight["model_fit_count"] == 0,
        "model_fit_count_exactly_one": result["training"]["model_fit_count"] == 1,
        "best_epoch_is_earliest_global_maximum": int(result["training"]["best_epoch"])
        == int(best["epoch"]),
        "best_metric_matches_history": bool(
            np.isclose(
                float(result["training"]["best_synthetic_validation_macro_raw_f1"]),
                float(best["synthetic_validation_macro_raw_f1"]),
            )
        ),
        "checkpoint_matches_best": bool(
            int(checkpoint["epoch"]) == int(result["training"]["best_epoch"])
            and np.isclose(
                float(checkpoint["synthetic_validation_macro_raw_f1"]),
                float(result["training"]["best_synthetic_validation_macro_raw_f1"]),
            )
        ),
        "sequential_gate_access": bool(sequential_access),
        "q3_q4_rows_zero": access["q3_q4_rows_read"] == 0,
        "official_rows_zero": access["official_test_sample_submission_rows_read"] == 0,
        "no_csv_or_submission_artifact": not forbidden_artifacts,
        "no_submission_or_upload": access["submission_generated_or_uploaded"] is False,
        "anchor_deletions_zero": result["anchor_deletions"] == 0,
        "no_point_adjustment_smoothing_or_truth_fill": result["point_adjustment"] is False
        and result["score_smoothing"] is False
        and result["truth_fill"] is False,
        "no_result_based_rerun": result["result_based_rerun"] is False
        and config["training"]["result_based_retry"] is False,
        "manifest_config_hash": manifest["config_sha256"]
        == runner.sha256_file(runner.CONFIG),
        "manifest_module_hash": manifest["module_sha256"]
        == runner.sha256_file(runner.MODULE),
        "manifest_runner_hash": manifest["runner_sha256"]
        == runner.sha256_file(runner.RUNNER),
        "manifest_artifact_hashes": bool(manifest_hashes_match),
        "manifest_no_raw_keys_csv_upload": manifest["raw_values_persisted"] is False
        and manifest["keys_persisted"] is False
        and manifest["submission_csv_generated"] is False
        and manifest["upload_performed"] is False,
        "runner_refuses_existing_output": "one-shot output already exists"
        in runner.RUNNER.read_text(encoding="utf-8"),
    }
    return {
        "schema_version": "p1.anomalybert_exact_degradation_mask.final_qa.v1",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_decision": result["status"],
        "checks": checks,
        "access": access,
        "best_epoch": result["training"]["best_epoch"],
        "model_fit_count": result["training"]["model_fit_count"],
        "artifact_hashes": {
            "result.json": runner.sha256_file(result_path),
            "manifest.json": runner.sha256_file(manifest_path),
            "best_checkpoint.pt": runner.sha256_file(checkpoint_path),
        },
        "config_sha256": runner.sha256_file(runner.CONFIG),
        "module_sha256": runner.sha256_file(runner.MODULE),
        "runner_sha256": runner.sha256_file(runner.RUNNER),
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke", action="store_true")
    modes.add_argument("--final", action="store_true")
    arguments = parser.parse_args()
    runner = _load_runner()
    config = runner._json(runner.CONFIG)
    if arguments.smoke:
        receipt = _smoke_qa(runner, config)
        output = ROOT / config["artifacts"]["smoke_directory"] / "qa.json"
    else:
        receipt = _final_qa(runner, config)
        output = ROOT / config["artifacts"]["directory"] / "final_qa.json"
    _atomic_json(output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
