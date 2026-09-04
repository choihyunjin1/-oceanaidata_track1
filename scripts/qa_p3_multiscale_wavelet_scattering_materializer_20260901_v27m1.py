"""Independent post-materialization QA for immutable P3 v27m1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_multiscale_wavelet_scattering_materializer_20260901_v27m1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = ROOT / "scripts/materialize_p3_multiscale_wavelet_scattering_20260901_v27m1.py"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
RESULT = ARTIFACT / "result.json"
ARRAYS = ARTIFACT / "official-action-geometry.npz"
CSV = ARTIFACT / "submission/P3_V27M1_SCATTER336_RIDGE1024_ADD10/P3_submission.csv"
KEYS = ["case_id", "station", "lead_h"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS, allow_pickle=False)
    frame = pd.read_csv(CSV, dtype={"case_id": "string", "station": "string"})

    candidate = arrays["candidate"].astype(np.float64)
    champion = arrays["champion"].astype(np.float64)
    action = arrays["action"].astype(np.float64)
    case_id = arrays["case_id"].astype(str)
    station = arrays["station"].astype(str)
    lead_h = arrays["lead_h"].astype(np.int64)
    guard = config["deployment_guards"]

    station_p99 = {
        name: float(np.quantile(np.abs(action[station == name]), 0.99))
        for name in sorted(set(station))
    }
    lead_p99 = {
        str(int(value)): float(np.quantile(np.abs(action[lead_h == value]), 0.99))
        for value in sorted(set(lead_h))
    }
    csv_roundtrip_max_abs_m = float(
        np.max(np.abs(frame["hs_pred"].to_numpy(np.float64) - candidate))
    )
    geometry = {
        "rows": int(len(candidate)),
        "action_p99_m": float(np.quantile(np.abs(action), 0.99)),
        "action_max_m": float(np.max(np.abs(action))),
        "action_rms_m": float(np.sqrt(np.mean(np.square(action)))),
        "prediction_min_m": float(candidate.min()),
        "prediction_max_m": float(candidate.max()),
        "station_p99_m": station_p99,
        "lead_p99_m": lead_p99,
        "changed_rows": int(np.count_nonzero(action)),
    }
    csv_keys_match_arrays = bool(
        np.array_equal(frame["case_id"].astype(str).to_numpy(), case_id)
        and np.array_equal(frame["station"].astype(str).to_numpy(), station)
        and np.array_equal(frame["lead_h"].to_numpy(np.int64), lead_h)
    )
    checks = {
        "status_ready_not_uploaded": result["status"] == "READY_NOT_UPLOADED",
        "guard_passed": result["guard_result"]["passed"] is True,
        "all_recorded_guard_checks_pass": all(result["guard_result"]["checks"].values()),
        "config_hash": sha256(CONFIG) == result["provenance"]["config_sha256"],
        "runner_hash": sha256(RUNNER) == result["provenance"]["runner_sha256"],
        "array_hash": sha256(ARRAYS) == result["action_artifact"]["sha256"],
        "csv_hash": sha256(CSV) == result["submission"]["sha256"],
        "csv_bytes": CSV.stat().st_size == result["submission"]["bytes"],
        "csv_schema": list(frame.columns) == KEYS + ["hs_pred"],
        "rows_exact": len(frame) == len(candidate) == 1200,
        "keys_unique": not frame.duplicated(KEYS).any(),
        "keys_match_arrays": csv_keys_match_arrays,
        "candidate_matches_csv_within_ieee754_roundtrip": bool(
            csv_roundtrip_max_abs_m <= 8.0 * np.finfo(np.float64).eps
        ),
        "action_identity": np.allclose(candidate - champion, action, atol=0.0, rtol=0.0),
        "finite": bool(np.isfinite(candidate).all() and np.isfinite(action).all()),
        "station_set": set(station) == set(guard["required_station_set"]),
        "lead_set": set(lead_h) == set(guard["required_lead_set"]),
        "prediction_range": geometry["prediction_min_m"] >= guard["prediction_min_m_gte"]
        and geometry["prediction_max_m"] <= guard["prediction_max_m_lte"],
        "action_p99": geometry["action_p99_m"] <= guard["absolute_action_p99_m_lte"],
        "action_max": geometry["action_max_m"] <= guard["absolute_action_max_m_lte"],
        "action_rms": geometry["action_rms_m"] <= guard["action_rms_m_lte"],
        "station_p99": max(station_p99.values())
        <= guard["maximum_station_action_p99_m_lte"],
        "lead_p99": max(lead_p99.values()) <= guard["maximum_lead_action_p99_m_lte"],
        "row_deletion_zero": config["candidate"]["row_deletion"] == 0,
        "hidden_score_upload_zero": result["data_access"]["hidden_truth_rows_read"] == 0
        and result["data_access"]["score_file_rows_read"] == 0
        and result["data_access"]["uploads"] == 0,
        "no_posthoc_or_retuning": not result["execution"]["posthoc_routing"]
        and not result["execution"]["result_based_tuning"],
    }
    payload = {
        "schema_version": "p3.multiscale_wavelet_scattering.materializer.independent_qa.v27m1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "geometry_recomputed": geometry,
        "csv_numeric_roundtrip_max_abs_m": csv_roundtrip_max_abs_m,
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "runner_sha256": sha256(RUNNER),
            "result_sha256": sha256(RESULT),
            "array_sha256": sha256(ARRAYS),
            "submission_sha256": sha256(CSV),
        },
        "data_access": {
            "official_source_rows_read_by_qa": 0,
            "hidden_truth_rows_read": 0,
            "score_file_rows_read": 0,
            "uploads": 0,
        },
        "caveat": "Internal v27 evidence is exploratory; this QA validates materialization integrity, not Public transport.",
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "independent-qa.json").write_bytes(canonical(payload))
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    if payload["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
