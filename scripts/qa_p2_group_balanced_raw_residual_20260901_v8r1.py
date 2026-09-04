"""Independent aggregate QA for P2 group-balanced raw residual v8r1."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = "p2_group_balanced_raw_residual_20260901_v8r1"
ARTIFACT = ROOT / "artifacts" / EXP
REPORT = ROOT / "reports" / EXP
RESULT = REPORT / "result.json"
FOLDS = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def close(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1e-12))


def main() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    data_dir = os.environ.get("P2_DATA_DIR")
    if not data_dir:
        raise RuntimeError("P2_DATA_DIR is required")
    observations_path = Path(data_dir).resolve() / "observations.csv"
    if observations_path.name != "observations.csv":
        raise RuntimeError("unexpected observations filename")
    observations = pd.read_csv(
        observations_path,
        usecols=["time", "layer", "temp"],
        dtype={"time": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    truth_lookup = observations.loc[
        observations["layer"].isin([2, 3, 4]), ["time", "layer", "temp"]
    ]
    checks: dict[str, bool] = {
        "terminal_no_go": result["status"]
        == "EXPLORATORY_NO_GO_BOTH_PREDECLARED_OBJECTIVES",
        "exploratory_not_fresh": result["claim_level"]
        == "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "fit_count_six": result["fit_count"] == 6,
        "fallback_executed": result["fallback_executed"] is True,
        "two_candidates": len(result["candidates"]) == 2,
        "training_rows_45935": result["training"]["rows"] == 45935,
        "feature_count_55": result["training"]["feature_count"] == 55,
        "six_weight_groups": len(result["training"]["weight_receipt"]["groups"]) == 6,
        "weights_sum_one": close(result["training"]["weight_receipt"]["raw_weight_sum"], 1.0),
        "weights_mean_one": close(
            result["training"]["weight_receipt"]["normalized_mean"], 1.0
        ),
        "v8_invalid_artifact_preserved": (
            ROOT / "artifacts/p2_group_balanced_raw_residual_20260901_v8/attempt_lock.json"
        ).is_file(),
    }
    recomputed: list[dict[str, Any]] = []
    for record in result["candidates"]:
        path = Path(record["prediction_commitment"]["path"])
        sealed = np.load(path)
        time = pd.to_datetime(sealed["time_ns"], unit="ns", utc=True)
        layer = sealed["layer"].astype(int)
        reference = sealed["reference"].astype(float)
        candidate = sealed["candidate"].astype(float)
        fold = sealed["fold"].astype(str)
        keys = pd.DataFrame({"time": time, "layer": layer})
        truth = keys.merge(truth_lookup, on=["time", "layer"], how="left", validate="one_to_one")
        y = truth["temp"].to_numpy(float)
        name = record["name"]
        checks[f"{name}:hash"] = sha256_file(path) == record["prediction_commitment"]["sha256"]
        checks[f"{name}:rows"] = len(candidate) == 69850
        checks[f"{name}:finite"] = bool(
            np.isfinite(reference).all() and np.isfinite(candidate).all() and np.isfinite(y).all()
        )
        ref_rmse = rmse(y, reference)
        cand_rmse = rmse(y, candidate)
        checks[f"{name}:pooled_reference"] = close(ref_rmse, record["reference_rmse"])
        checks[f"{name}:pooled_candidate"] = close(cand_rmse, record["candidate_rmse"])
        fold_metrics: dict[str, float] = {}
        for fold_name in FOLDS:
            mask = fold == fold_name
            delta = rmse(y[mask], candidate[mask]) - rmse(y[mask], reference[mask])
            fold_metrics[fold_name] = delta
            checks[f"{name}:fold:{fold_name}"] = close(
                delta, record["by_fold"][fold_name]["delta_rmse"]
            )
        for layer_value in (2, 3, 4):
            mask = layer == layer_value
            delta = rmse(y[mask], candidate[mask]) - rmse(y[mask], reference[mask])
            checks[f"{name}:layer:{layer_value}"] = close(
                delta, record["by_layer"][str(layer_value)]["delta_rmse"]
            )
        checks[f"{name}:one_of_three_folds_only"] = sum(
            value < 0.0 for value in fold_metrics.values()
        ) == 1
        checks[f"{name}:strict_no_go"] = record["strict_exploratory_pass"] is False
        recomputed.append(
            {
                "name": name,
                "reference_rmse": ref_rmse,
                "candidate_rmse": cand_rmse,
                "delta_rmse": cand_rmse - ref_rmse,
                "by_fold_delta_rmse": fold_metrics,
            }
        )
    counters = result["operation_counters"]
    for field in (
        "official_test_index_rows_read",
        "sample_rows_read",
        "baseline_file_rows_read",
        "score_file_rows_read",
        "query_support_rows_read",
        "hidden_truth_rows_read",
        "submission_csv_created",
        "uploads",
    ):
        checks[f"operation_zero:{field}"] = counters[field] == 0
    qa = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": recomputed,
        "interpretation": {
            "official_like_points_are_nominal_extrapolation_only": True,
            "reason": "The local RMSE deltas are far outside the small-delta calibration range and the comparator surface is exposed.",
            "submission_ready": False,
        },
        "operation_counters": {
            "observations_rows_read": int(len(observations)),
            "official_rows_read": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "independent-qa.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
