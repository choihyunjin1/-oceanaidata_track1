"""Independent aggregate-only QA for the P1 TS2Vec-style smoke receipt."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_p1_ts2vec_conditional_normal_prototype_rescue_20260828_v1.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("p1_ts2vec_runner_qa", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to import sealed runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _smoke_qa(runner: Any, config: dict[str, Any]) -> dict[str, Any]:
    independent_contract = runner.validate_contract(config)
    smoke_path = ROOT / config["artifacts"]["smoke_directory"] / "smoke.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    checks = {
        "independent_contract_pass": independent_contract["status"] == "PASS",
        "smoke_pass": str(smoke["status"]).startswith("PASS_"),
        "smoke_contract_matches_current_config": smoke["contract"]["config_sha256"]
        == independent_contract["config_sha256"],
        "cpu_finite": smoke["cpu"]["finite"] is True,
        "gpu_finite_or_explicitly_skipped": smoke["gpu"].get("finite") is True
        or smoke["gpu"].get("skipped") is True,
        "scientific_rows_zero": smoke["scientific_source_rows_read"] == 0,
        "label_rows_zero": smoke["labels_read"] == 0,
        "full_fit_zero": smoke["full_model_fit_count"] == 0,
        "q2_deferred": independent_contract["inputs"]["q2_e150_anchor"]["deferred"] is True,
        "official_rows_zero": smoke["official_rows_read"] == 0,
        "official_upload_false": config["execution_policy"]["official_upload_authorized"] is False,
        "full_gpu_waits_for_coordinator": config["execution_policy"]["full_gpu_run_requires_coordinator_release_after_smoke"] is True,
    }
    return {
        "schema_version": "p1.ts2vec_conditional_normal_prototype.smoke_qa.v1",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "smoke_sha256": runner.sha256_file(smoke_path),
        "config_sha256": runner.sha256_file(runner.CONFIG),
        "module_sha256": runner.sha256_file(runner.MODULE),
        "runner_sha256": runner.sha256_file(runner.RUNNER),
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }


def _final_qa(runner: Any, config: dict[str, Any]) -> dict[str, Any]:
    runner = _load_runner()
    independent_contract = runner.validate_contract(config)
    output = ROOT / config["artifacts"]["directory"]
    result_path = output / "result.json"
    checkpoint_path = output / "encoder.pt"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    training = result["training"]
    history = training["history"]
    eligible = [row for row in history if float(row["embedding_variance"]) >= 1e-4]
    best = min(eligible, key=lambda row: (float(row["validation_loss"]), int(row["epoch"])))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    artifact_files = [path for path in output.rglob("*") if path.is_file()]
    forbidden_artifacts = [
        str(path.relative_to(output))
        for path in artifact_files
        if path.suffix.lower() == ".csv" or "submission" in path.name.lower()
    ]
    immutable_paths = [str(spec["path"]).lower() for spec in config["immutable_inputs"].values()]
    runner_source = runner.RUNNER.read_text(encoding="utf-8")
    checks = {
        "independent_contract_pass": independent_contract["status"] == "PASS",
        "terminal_status_no_go_coverage": result["status"] == "NO_GO_COVERAGE",
        "coverage_below_frozen_gate": float(result["historical_embedding_coverage"]) < 0.95,
        "model_fit_count_exactly_one": training["model_fit_count"] == 1,
        "best_checkpoint_rule_frozen": config["representation"]["checkpoint_rule"]
        == "minimum_heldout_label_free_hierarchical_contrastive_loss_subject_to_embedding_variance_gte_1e-4",
        "best_epoch_matches_eligible_minimum": int(training["best_epoch"]) == int(best["epoch"]),
        "best_loss_matches_eligible_minimum": bool(
            np.isclose(
                float(training["best_validation_loss"]),
                float(best["validation_loss"]),
            )
        ),
        "checkpoint_epoch_matches_result": int(checkpoint["epoch"]) == int(training["best_epoch"]),
        "q2_truth_rows_zero": result["q2_truth_rows_read"] == 0,
        "q2_anchor_hash_deferred": independent_contract["inputs"]["q2_e150_anchor"]["deferred"] is True,
        "no_q3_q4_or_official_scientific_path_opened": "frozen_truth_and_folds"
        not in runner_source
        and not any(
            token in path
            for path in immutable_paths
            for token in ("q3", "q4", "test.csv", "sample_submission", "submission.csv")
        ),
        "coverage_stop_precedes_q2_outer_code": "if historical_coverage < 0.95:" in runner_source
        and "return result" in runner_source,
        "no_csv_or_submission_artifact": not forbidden_artifacts,
        "official_upload_false": config["execution_policy"]["official_upload_authorized"] is False,
        "result_based_rerun_false": config["execution_policy"]["result_based_window_epoch_threshold_rerun"] is False,
        "runner_refuses_existing_output": "full experiment artifact already exists; retries are prohibited" in runner_source,
    }
    return {
        "schema_version": "p1.ts2vec_conditional_normal_prototype.final_qa.v1",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "terminal_decision": "NO_GO_COVERAGE_NO_RERUN",
        "checks": checks,
        "artifacts": {
            "encoder.pt": {"bytes": checkpoint_path.stat().st_size, "sha256": runner.sha256_file(checkpoint_path)},
            "result.json": {"bytes": result_path.stat().st_size, "sha256": runner.sha256_file(result_path)},
        },
        "config_sha256": runner.sha256_file(runner.CONFIG),
        "module_sha256": runner.sha256_file(runner.MODULE),
        "runner_sha256": runner.sha256_file(runner.RUNNER),
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_rows_read": 0,
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
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
