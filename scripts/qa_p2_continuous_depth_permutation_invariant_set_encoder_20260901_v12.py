"""Independent metric/hash/access QA for the terminal P2 v12 artifact."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for item in (SRC, SCRIPTS):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import run_p2_continuous_depth_permutation_invariant_set_encoder_20260901_v12 as runner  # noqa: E402

from p2_restore.features import build_training_features  # noqa: E402
from p2_restore.normalized_curvature_residual import build_normalized_curvature_design  # noqa: E402


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(abs(left - right) <= tolerance)


def main() -> None:
    result_path = runner.REPORT / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(runner.CONFIG.read_text(encoding="utf-8"))
    prediction_path = runner.ARTIFACT / "P2_V12_COMPACT_DEEPSET_BLEND020.npz"
    payload = np.load(prediction_path, allow_pickle=False)
    reference = payload["reference"].astype(float)
    candidate = payload["candidate"].astype(float)
    time_ns = payload["time_ns"].astype(np.int64)
    layer = payload["layer"].astype(int)

    raw = os.environ.get("P2_DATA_DIR")
    if not raw:
        raise RuntimeError("P2_DATA_DIR is required")
    observations_path = Path(raw).resolve() / "observations.csv"
    observations = pd.read_csv(observations_path, dtype={"station": "string", "time": "string"})
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    feature_table = build_training_features(observations)
    design = build_normalized_curvature_design(feature_table.frame)
    index = pd.MultiIndex.from_arrays(
        [runner.metric_engine.canonical_time_ns(design.keys["time"]), design.keys["layer"]]
    )
    positions = index.get_indexer(pd.MultiIndex.from_arrays([time_ns, layer]))
    if np.any(positions < 0):
        raise RuntimeError("independent truth alignment failed")
    truth = design.truth[positions]
    pooled_reference = runner.metric_engine.rmse(truth, reference)
    pooled_candidate = runner.metric_engine.rmse(truth, candidate)
    record = result["candidate"]
    counters = result["operation_counters"]
    action = np.abs(candidate - reference)
    checks: dict[str, bool] = {
        "result_terminal": result["status"].startswith("EXPLORATORY_"),
        "fit_count_within_budget": 0 < int(result["fit_count"]) <= 9,
        "prediction_rows_match": len(candidate) == int(record["prediction_commitment"]["rows"]),
        "prediction_hash_match": runner.sha256_file(prediction_path)
        == record["prediction_commitment"]["sha256"],
        "config_hash_match": runner.sha256_file(runner.CONFIG) == result["hashes"]["config"],
        "runner_hash_match": runner.sha256_file(runner.RUNNER) == result["hashes"]["runner"],
        "observations_hash_match": runner.sha256_file(observations_path)
        == config["source_contract"]["observations_sha256"],
        "reference_rmse_recomputed": close(pooled_reference, record["reference_rmse"]),
        "candidate_rmse_recomputed": close(pooled_candidate, record["candidate_rmse"]),
        "delta_rmse_recomputed": close(
            pooled_candidate - pooled_reference, record["delta_rmse"]
        ),
        "blend_action_bound_half_degree": float(action.max()) <= 0.5 + 1e-12,
        "prediction_finite": bool(np.isfinite(candidate).all()),
        "permutation_invariance_tested": result["permutation_invariance"]["maximum_abs_error"]
        <= 1e-6,
        "row_deletion_zero": result["training"]["row_deletion"] == 0,
        "official_access_zero": all(
            int(counters[name]) == 0
            for name in (
                "official_test_index_rows_read",
                "sample_rows_read",
                "baseline_file_rows_read",
                "score_file_rows_read",
                "query_support_rows_read",
                "hidden_truth_rows_read",
                "submission_csv_created",
                "uploads",
            )
        ),
    }
    qa: dict[str, Any] = {
        "schema_version": "p2.continuous_depth_permutation_invariant_set_encoder.independent_qa.20260901.v12",
        "experiment_id": runner.EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": {
            "rows": int(len(candidate)),
            "reference_rmse": pooled_reference,
            "candidate_rmse": pooled_candidate,
            "delta_rmse": pooled_candidate - pooled_reference,
            "abs_action_p99_C": float(np.quantile(action, 0.99)),
            "abs_action_max_C": float(action.max()),
        },
        "access": {
            "observations_rows_read": int(len(observations)),
            "official_test_index_rows_read": 0,
            "sample_rows_read": 0,
            "baseline_file_rows_read": 0,
            "score_file_rows_read": 0,
            "query_support_rows_read": 0,
            "hidden_truth_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "hashes": {
            "result": runner.sha256_file(result_path),
            "prediction_npz": runner.sha256_file(prediction_path),
            "config": runner.sha256_file(runner.CONFIG),
            "runner": runner.sha256_file(runner.RUNNER),
        },
    }
    runner.atomic_json(runner.REPORT / "independent-qa.json", qa)
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
