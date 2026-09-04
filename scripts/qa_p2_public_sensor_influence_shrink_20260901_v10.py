"""Independent arithmetic/hash QA for the terminal P2 v10 sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_public_sensor_influence_shrink_20260901_v10"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        raise SystemExit("Use --execute")
    result = json.loads((ARTIFACT / "result.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source_config = config
    if "source_contract" not in source_config:
        predecessor = source_config.get("predecessor", {})
        predecessor_path = predecessor.get("config_path")
        if not predecessor_path:
            raise RuntimeError("source contract and predecessor config are both absent")
        source_config = json.loads(
            (ROOT / predecessor_path).read_text(encoding="utf-8")
        )
    action_path = Path(result["candidate"]["action_commitment"]["path"])
    if not action_path.is_absolute():
        action_path = ROOT / action_path
    values = np.load(action_path, allow_pickle=False)
    reference = values["reference"].astype(float)
    candidate = values["candidate"].astype(float)
    weight = values["influence_weight"].astype(float)
    endpoint = values["endpoint_baseline"].astype(float)
    time_ns = values["time_ns"].astype(np.int64)
    layer = values["layer"].astype(int)

    data_dir = os.environ.get("P2_DATA_DIR")
    if not data_dir:
        raise RuntimeError("P2_DATA_DIR is required")
    observations = pd.read_csv(
        Path(data_dir) / "observations.csv",
        dtype={"time": "string", "station": "string"},
    )
    observations["time"] = pd.to_datetime(observations["time"], utc=True)
    truth_frame = observations.loc[observations["layer"].isin([2, 3, 4]), ["time", "layer", "temp"]].copy()
    truth_index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(truth_frame["time"]).as_unit("ns").asi8, truth_frame["layer"]]
    )
    action_index = pd.MultiIndex.from_arrays([time_ns, layer])
    position = truth_index.get_indexer(action_index)
    if np.any(position < 0):
        raise RuntimeError("truth alignment failed")
    truth = truth_frame["temp"].to_numpy(float)[position]
    delta = rmse(truth, candidate) - rmse(truth, reference)
    active = weight < 1.0
    expected = endpoint + weight * (reference - endpoint)
    checks = {
        "experiment_id": result["experiment_id"] == EXPERIMENT_ID,
        "action_hash": sha256_file(action_path) == result["hashes"]["action_npz"],
        "rows": len(candidate) == result["rows"],
        "formula_exact": np.array_equal(expected, candidate),
        "inactive_bit_exact": np.array_equal(candidate[~active], reference[~active]),
        "weight_bounds": bool(np.all((weight >= 0.5) & (weight <= 1.0))),
        "row_deletion_zero": result["influence_receipt"]["rows_deleted"] == 0,
        "fit_count_zero": result["fit_count"] == 0,
        "pooled_delta_recomputed": abs(delta - result["candidate"]["delta_rmse"]) < 1e-12,
        "active_rows_recomputed": int(active.sum()) == result["candidate"]["active_rows"],
        "all_finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
        "access_zero": all(
            result["operation_counters"][key] == 0
            for key in (
                "official_test_index_rows_read",
                "sample_rows_read",
                "baseline_file_rows_read",
                "score_file_rows_read",
                "query_support_rows_read",
                "hidden_truth_rows_read",
                "existing_v7_submission_csv_value_reads",
                "submission_csv_created",
                "uploads",
            )
        ),
        "config_hash": sha256_file(CONFIG) == result["hashes"]["config"],
        "observation_hash": sha256_file(Path(data_dir) / "observations.csv")
        == source_config["source_contract"]["observations_sha256"],
    }
    absolute_change = np.abs(candidate - reference)
    squared_error_gain = np.square(reference - truth) - np.square(candidate - truth)
    ordered_gain = np.sort(squared_error_gain)[::-1]
    total_gain = float(squared_error_gain.sum())
    concentration = {}
    for fraction in (0.001, 0.01, 0.05):
        count = max(1, int(np.ceil(len(ordered_gain) * fraction)))
        concentration[str(fraction)] = {
            "rows": count,
            "share_of_total_sse_gain": float(ordered_gain[:count].sum() / total_gain)
            if total_gain != 0.0
            else None,
        }
    qa: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recomputed": {
            "reference_rmse": rmse(truth, reference),
            "candidate_rmse": rmse(truth, candidate),
            "delta_rmse": delta,
            "active_rows": int(active.sum()),
            "inactive_rows": int((~active).sum()),
        },
        "transport_risk_diagnostics": {
            "absolute_change_C_quantiles": {
                str(quantile): float(np.quantile(absolute_change, quantile))
                for quantile in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
            },
            "active_absolute_change_C_quantiles": {
                str(quantile): float(np.quantile(absolute_change[active], quantile))
                for quantile in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
            },
            "sse_gain_C2": total_gain,
            "sse_gain_concentration": concentration,
            "truth_range_C": [float(np.min(truth)), float(np.max(truth))],
            "reference_range_C": [float(np.min(reference)), float(np.max(reference))],
            "candidate_range_C": [float(np.min(candidate)), float(np.max(candidate))],
        },
        "official_test_sample_baseline_score_query_hidden_submission_upload_access": 0,
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "independent-qa.json"
    path.write_text(json.dumps(qa, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(qa, indent=2, ensure_ascii=False))
    if qa["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
