"""Independent recomputation for the completed P3 v23 path-signature cycle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "scripts", ROOT / "src"):
    if str(entry) not in os.sys.path:
        os.sys.path.insert(0, str(entry))

import run_p3_path_signature_residual_cycle_20260901_v23 as runner  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import bootstrap  # noqa: E402

RESULT = ROOT / "artifacts/p3_path_signature_residual_cycle_20260901_v23/result.json"
ARRAYS = ROOT / "artifacts/p3_path_signature_residual_cycle_20260901_v23/evaluation-arrays.npz"
OUTPUT = ROOT / "reports/p3_path_signature_residual_cycle_20260901_v23/independent-qa.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(float(left) - float(right)) <= tolerance)


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def reconstruct_frame(arrays: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    leads = arrays["lead_h"].astype(int)
    for case in range(len(arrays["anchor_id"])):
        for position, lead in enumerate(leads):
            rows.append(
                {
                    "target_hs": float(arrays["truth"][case, position]),
                    "reference": float(arrays["uniform"][case, position]),
                    "block": str(arrays["block"][case]),
                    "station": str(arrays["station"][case]),
                    "episode_id": str(arrays["episode"][case]),
                    "lead_h": int(lead),
                }
            )
    return pd.DataFrame(rows)


def group_delta(
    frame: pd.DataFrame, prediction: np.ndarray, group: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, local in frame.groupby(group, observed=True, sort=True):
        indices = local.index.to_numpy(dtype=np.int64)
        truth = local["target_hs"].to_numpy(dtype=np.float64)
        reference = local["reference"].to_numpy(dtype=np.float64)
        before = float(np.sqrt(np.mean(np.square(reference - truth))))
        after = float(np.sqrt(np.mean(np.square(prediction[indices] - truth))))
        result[str(key)] = after - before
    return result


def main() -> int:
    if OUTPUT.exists():
        raise RuntimeError("independent QA output already exists")
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        frame = reconstruct_frame(arrays)
        truth = arrays["truth"].reshape(-1).astype(np.float64)
        uniform = arrays["uniform"].reshape(-1).astype(np.float64)
        uniform_rmse = float(np.sqrt(np.mean(np.square(uniform - truth))))
        checks: dict[str, bool] = {
            "array_shape_182_by_6": arrays["truth"].shape == (182, 6),
            "finite_arrays": all(
                np.isfinite(arrays[name]).all()
                for name in ("truth", "uniform", "candidate_1", "candidate_2")
            ),
            "fit_count_12": result["fit_count"] == 12
            and len(result["fit_receipts"]) == 12
            and sum(item["fit_count"] for item in result["fit_receipts"]) == 12,
            "six_fits_each": all(
                sum(item["candidate"] == spec.name for item in result["fit_receipts"]) == 6
                for spec in runner.SPECS
            ),
            "row_deletion_zero": all(
                item["row_deletion"] == 0 for item in result["fit_receipts"]
            ),
            "arrays_hash": sha256(ARRAYS)
            == result["provenance"]["evaluation_arrays_sha256"],
            "config_hash": sha256(runner.CONFIG) == result["provenance"]["config_sha256"],
            "runner_hash": sha256(runner.Path(runner.__file__))
            == result["provenance"]["runner_sha256"],
            "official_access_zero": all(
                value == 0
                for key, value in result["data_access"].items()
                if key != "historical_target_rows"
            ),
            "historical_rows_1092": result["data_access"]["historical_target_rows"] == 1092,
            "result_based_tuning_false": result["execution"]["result_based_tuning"] is False,
            "candidate_count_two": len(result["candidates"]) == 2,
        }
        candidate_receipts: list[dict[str, Any]] = []
        for index, item in enumerate(result["candidates"], start=1):
            prediction = arrays[f"candidate_{index}"].reshape(-1).astype(np.float64)
            candidate_rmse = float(np.sqrt(np.mean(np.square(prediction - truth))))
            delta = candidate_rmse - uniform_rmse
            raw_points = -delta * runner.POINTS_PER_RMSE_M
            block = group_delta(frame, prediction, "block")
            station = group_delta(frame, prediction, "station")
            lead = group_delta(frame, prediction, "lead_h")
            offset = (index - 1) * 100
            episode = bootstrap(frame, prediction, ("episode_id",), 20260931 + offset)
            grouped = bootstrap(frame, prediction, ("block", "station"), 20260932 + offset)
            local_checks = {
                "uniform_rmse": close(uniform_rmse, item["rmse_m"]["uniform_0p425"]),
                "candidate_rmse": close(candidate_rmse, item["rmse_m"]["candidate"]),
                "delta": close(delta, item["rmse_m"]["delta_candidate_minus_uniform"]),
                "raw_points": close(raw_points, item["expected_points"]["raw_gain"]),
                "block_deltas": all(
                    close(value, item["by_block"][key]["delta_rmse_m"])
                    for key, value in block.items()
                ),
                "station_deltas": all(
                    close(value, item["station"][key]["delta_rmse_m"])
                    for key, value in station.items()
                ),
                "lead_deltas": all(
                    close(value, item["lead"][key]["delta_rmse_m"])
                    for key, value in lead.items()
                ),
                "episode_ci": all(
                    close(left, right)
                    for left, right in zip(
                        episode["ci90_m"], item["episode_bootstrap"]["ci90_m"], strict=True
                    )
                ),
                "block_station_ci": all(
                    close(left, right)
                    for left, right in zip(
                        grouped["ci90_m"],
                        item["block_station_bootstrap"]["ci90_m"],
                        strict=True,
                    )
                ),
                "decision_no_go": item["decision"] == "NO_GO",
            }
            checks[f"candidate_{index}_recomputed"] = all(local_checks.values())
            candidate_receipts.append(
                {
                    "name": item["name"],
                    "rmse_m": candidate_rmse,
                    "delta_m": delta,
                    "raw_points": raw_points,
                    "checks": local_checks,
                }
            )
    qa = {
        "schema_version": "p3.path_signature_residual.independent_qa.v23",
        "experiment_id": runner.EXPERIMENT_ID,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "checks": len(checks),
            "passed": sum(checks.values()),
            "failed": sum(not value for value in checks.values()),
            "model_fits": 12,
            "official_rows": 0,
            "csv_materializations": 0,
            "uploads": 0,
        },
        "candidates": candidate_receipts,
        "hashes": {
            "result_sha256": sha256(RESULT),
            "arrays_sha256": sha256(ARRAYS),
            "qa_runner_sha256": sha256(Path(__file__)),
        },
    }
    OUTPUT.write_bytes(canonical(qa))
    print(json.dumps({"decision": qa["decision"], **qa["counts"]}, ensure_ascii=False))
    return 0 if qa["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
