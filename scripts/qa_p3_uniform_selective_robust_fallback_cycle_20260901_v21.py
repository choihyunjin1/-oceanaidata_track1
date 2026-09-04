"""Independent recalculation QA for P3 selective robust fallback v21."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_uniform_selective_robust_fallback_cycle_20260901_v21"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
RESULT = ARTIFACT_DIR / "result.json"
ARRAYS = ARTIFACT_DIR / "evaluation-arrays.npz"
OUTPUT = REPORT_DIR / "independent-qa.json"


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def group_delta(
    truth: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *keys: np.ndarray,
) -> dict[str, float]:
    labels = np.array(
        ["|".join(parts) for parts in zip(*(key.astype(str) for key in keys), strict=True)]
    )
    return {
        label: rmse(truth[labels == label], candidate[labels == label])
        - rmse(truth[labels == label], reference[labels == label])
        for label in sorted(set(labels.tolist()))
    }


def close(left: float, right: float) -> bool:
    return bool(abs(left - right) <= 1e-12)


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    with np.load(ARRAYS, allow_pickle=False) as data:
        truth = data["truth"]
        uniform = data["uniform"]
        candidates = [data["candidate_1"], data["candidate_2"]]
        lead = data["lead_h"]
        block = data["block"]
        station = data["station"]
    per_candidate: dict[str, Any] = {}
    metric_checks = []
    block_checks = []
    station_lead_checks = []
    short_checks = []
    for reported, candidate in zip(result["candidates"], candidates, strict=True):
        candidate_rmse = rmse(truth, candidate)
        uniform_rmse = rmse(truth, uniform)
        block_delta = group_delta(truth, candidate, uniform, block)
        station_lead = group_delta(truth, candidate, uniform, station, lead)
        metric_checks.append(
            close(candidate_rmse, reported["rmse_m"]["candidate"])
            and close(uniform_rmse, reported["rmse_m"]["uniform_0p425"])
            and close(
                candidate_rmse - uniform_rmse,
                reported["rmse_m"]["delta_candidate_minus_uniform"],
            )
        )
        block_checks.append(
            all(
                close(value, reported["by_block"][key]["delta_rmse_m"])
                for key, value in block_delta.items()
            )
            and sum(value < 0 for value in block_delta.values()) == reported["improved_blocks"]
        )
        station_lead_checks.append(
            close(max(station_lead.values()), reported["worst_station_lead_delta_m"])
        )
        short = np.isin(lead, [3, 6, 9, 12])
        short_checks.append(bool(np.array_equal(candidate[short], uniform[short])))
        per_candidate[reported["name"]] = {
            "candidate_rmse_m": candidate_rmse,
            "uniform_rmse_m": uniform_rmse,
            "delta_rmse_m": candidate_rmse - uniform_rmse,
            "by_block_delta_m": block_delta,
            "worst_station_lead_delta_m": max(station_lead.values()),
        }
    checks = {
        "complete": result["status"] == "COMPLETE",
        "two_candidates": len(result["candidates"]) == 2,
        "all_metrics_recomputed": all(metric_checks),
        "all_block_metrics_recomputed": all(block_checks),
        "all_station_lead_worst_recomputed": all(station_lead_checks),
        "all_short_leads_bit_exact_uniform": all(short_checks),
        "all_arrays_finite": bool(
            all(np.isfinite(item).all() for item in [truth, uniform, *candidates])
        ),
        "zero_target_fits": result["fit_count"]["target_fits"] == 0,
        "six_shared_feature_only_fits": result["fit_count"]["shared_feature_only_quantile_fits"]
        == 6,
        "gate_receipts_target_free": all(
            item["target_rows_read_before_gate_fixed"] == 0 for item in result["gate_receipts"]
        ),
        "no_result_based_tuning": result["execution"]["result_based_tuning"] is False
        and result["execution"]["threshold_searches"] == 0,
        "zero_official_hidden_csv_upload": all(
            result["data_access"][key] == 0
            for key in (
                "official_test_rows",
                "official_sample_rows",
                "official_submission_rows",
                "hidden_truth_rows",
                "csv_materializations",
                "uploads",
            )
        ),
        "arrays_hash_matches": result["provenance"]["evaluation_arrays_sha256"]
        == hashlib.sha256(ARRAYS.read_bytes()).hexdigest(),
    }
    receipt = {
        "schema_version": "p3.uniform_selective_robust_fallback.independent_qa.v21",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recalculated": per_candidate,
        "interpretation_boundary": "EXPLORATORY_ONLY; exposed 182-case surface, no Public transport guarantee.",
    }
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.write_bytes(canonical(receipt))
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
