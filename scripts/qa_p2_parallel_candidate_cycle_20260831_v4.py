"""Independent, read-only QA for the completed P2 v4 cycle."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_p2_parallel_candidate_cycle_20260831_v4 as run  # noqa: E402


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


def main() -> None:
    result = json.loads((run.ARTIFACT / "result.json").read_text(encoding="utf-8"))
    lock = json.loads((run.ARTIFACT / "attempt_lock.json").read_text(encoding="utf-8"))
    recovery = json.loads(
        (run.ARTIFACT / "technical_recovery.json").read_text(encoding="utf-8")
    )
    observations = pd.read_csv(run.OBSERVATIONS, usecols=["time", "layer", "temp"])
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    truth = observations.loc[observations["layer"].isin([2, 3, 4])]

    metric_checks: dict[str, Any] = {}
    hash_checks: dict[str, bool] = {}
    gate_checks: dict[str, bool] = {}
    result_by_name = {record["name"]: record for record in result["candidates"]}
    for name, record in result["prediction_commitment"]["outputs"].items():
        path = Path(record["path"])
        hash_checks[name] = run.sha256_file(path) == record["sha256"]
        with np.load(path, allow_pickle=False) as payload:
            scored = pd.DataFrame(
                {
                    "time": pd.to_datetime(payload["time_ns"], unit="ns", utc=True),
                    "layer": payload["layer"].astype(int),
                    "reference": payload["reference"],
                    "candidate": payload["candidate"],
                    "fold": payload["fold"].astype(str),
                }
            )
        scored = scored.merge(truth, on=["time", "layer"], how="left", validate="one_to_one")
        if scored["temp"].isna().any():
            raise RuntimeError(f"truth join failed for {name}")
        reference_rmse = run.rmse(scored["temp"].to_numpy(), scored["reference"].to_numpy())
        candidate_rmse = run.rmse(scored["temp"].to_numpy(), scored["candidate"].to_numpy())
        by_fold = {}
        for fold in run.FOLD_ORDER:
            part = scored.loc[scored["fold"].eq(fold)]
            reference_fold = run.rmse(part["temp"].to_numpy(), part["reference"].to_numpy())
            candidate_fold = run.rmse(part["temp"].to_numpy(), part["candidate"].to_numpy())
            by_fold[fold] = {"delta_rmse": candidate_fold - reference_fold}
        expected = result_by_name[name]
        metric_checks[name] = {
            "reference_rmse_match": close(reference_rmse, expected["reference_rmse"]),
            "candidate_rmse_match": close(candidate_rmse, expected["candidate_rmse"]),
            "pooled_delta_match": close(
                candidate_rmse - reference_rmse, expected["delta_rmse"]
            ),
            "fold_deltas_match": all(
                close(by_fold[fold]["delta_rmse"], expected["by_fold"][fold]["delta_rmse"])
                for fold in run.FOLD_ORDER
            ),
        }
        recomputed_gate = run.strict_gate(by_fold, candidate_rmse - reference_rmse)
        gate_checks[name] = (
            recomputed_gate == expected["gate_checks"]
            and all(recomputed_gate.values()) == expected["strict_internal_pass"]
        )

    submission_checks: dict[str, Any] = {}
    test_index = pd.read_csv(run.TEST_INDEX, dtype={"station": "string", "time": "string"})
    observations_full = pd.read_csv(run.OBSERVATIONS)
    endpoints = run.public_endpoint_frame(observations_full)
    for record in result["submissions"]:
        path = Path(record["path"])
        submission = pd.read_csv(path, dtype={"station": "string", "time": "string"})
        qa = run.validate_submission(submission, test_index, endpoints)
        submission_checks[record["name"]] = {
            "sha256_match": run.sha256_file(path) == record["sha256"],
            "structural_qa_pass": qa["status"] == "PASS",
            "recorded_qa_match": qa == record["qa"],
        }

    registry = json.loads((run.REPORT / "pass-registry.json").read_text(encoding="utf-8"))
    registry_checks = {
        record["name"]: Path(record["path"]).is_file()
        and run.sha256_file(Path(record["path"])) == record["sha256"]
        for record in registry["candidates"]
    }

    checks = {
        "attempt_lock_preserved": lock["runner_sha256"] != recovery["new_runner_sha256"],
        "current_runner_matches_recovery": run.sha256_file(run.Path(run.__file__))
        == recovery["new_runner_sha256"],
        "technical_failure_before_metric": recovery["performance_metric_seen_before_repair"]
        is False,
        "technical_failure_before_prediction": recovery["prediction_written_before_repair"]
        is False,
        "technical_failure_before_official_read": recovery["official_rows_read_before_repair"]
        == 0,
        "prediction_hashes_match": all(hash_checks.values()),
        "metrics_recomputed_exactly": all(
            all(record.values()) for record in metric_checks.values()
        ),
        "gates_recomputed_exactly": all(gate_checks.values()),
        "only_pass_materialized": {record["name"] for record in result["submissions"]}
        == {
            record["name"]
            for record in result["candidates"]
            if record["strict_internal_pass"]
        },
        "submission_checks_pass": all(
            all(record.values()) for record in submission_checks.values()
        ),
        "two_pass_registry_hashes_match": registry["pass_count"] == 2
        and all(registry_checks.values()),
        "hidden_truth_rows_zero": result["hidden_truth_rows_read"] == 0,
        "upload_count_zero": result["upload_count"] == 0,
    }
    qa = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recovery_chain": {
            "initial_runner_sha256": lock["runner_sha256"],
            "repaired_runner_sha256": recovery["new_runner_sha256"],
            "reason": recovery["reason"],
            "repair": recovery["repair"],
            "performance_adaptive_change": False,
        },
        "prediction_hash_checks": hash_checks,
        "metric_checks": metric_checks,
        "gate_checks": gate_checks,
        "submission_checks": submission_checks,
        "pass_registry_checks": registry_checks,
    }
    run.atomic_json(run.REPORT / "independent-qa-recomputed.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
