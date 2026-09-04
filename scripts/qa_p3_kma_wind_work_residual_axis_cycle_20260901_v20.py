"""Independent recalculation QA for the sealed P3 wind-work residual-axis run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_kma_wind_work_residual_axis_cycle_20260901_v20"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
RESULT = ARTIFACT_DIR / "result.json"
ARRAYS = ARTIFACT_DIR / "evaluation-arrays.npz"
OUTPUT = REPORT_DIR / "independent-qa.json"


def canonical(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def metric(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(truth - prediction))))


def grouped_delta(
    truth: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    *keys: np.ndarray,
) -> dict[str, float]:
    labels = np.array(
        ["|".join(parts) for parts in zip(*(key.astype(str) for key in keys), strict=True)]
    )
    output: dict[str, float] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        output[label] = metric(truth[mask], candidate[mask]) - metric(truth[mask], comparator[mask])
    return output


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return bool(abs(left - right) <= tolerance)


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    with np.load(ARRAYS, allow_pickle=False) as arrays:
        truth = arrays["truth"]
        persistence = arrays["persistence"]
        uniform = arrays["uniform"]
        v19 = arrays["v19"]
        candidate = arrays["candidate"]
        lead = arrays["lead_h"]
        block = arrays["block"]
        station = arrays["station"]

    reported = result["candidate"]["rmse_m"]
    recalculated = {
        "persistence": metric(truth, persistence),
        "uniform_kma_0p425": metric(truth, uniform),
        "v19_wave_power": metric(truth, v19),
        "candidate": metric(truth, candidate),
    }
    recalculated["delta_candidate_minus_persistence"] = (
        recalculated["candidate"] - recalculated["persistence"]
    )
    recalculated["delta_candidate_minus_uniform"] = (
        recalculated["candidate"] - recalculated["uniform_kma_0p425"]
    )
    recalculated["delta_candidate_minus_v19"] = (
        recalculated["candidate"] - recalculated["v19_wave_power"]
    )
    by_block = grouped_delta(truth, candidate, v19, block)
    station_lead = grouped_delta(truth, candidate, v19, station, lead)
    checks = {
        "result_complete": result["status"] == "COMPLETE",
        "all_reported_metrics_recomputed": all(
            close(recalculated[key], reported[key]) for key in recalculated
        ),
        "by_block_recomputed": all(
            close(value, result["candidate"]["by_block_vs_v19"][key]["delta_rmse_m"])
            for key, value in by_block.items()
        ),
        "improved_blocks_recomputed": sum(value < 0 for value in by_block.values())
        == result["candidate"]["improved_blocks_vs_v19"],
        "worst_station_lead_recomputed": close(
            max(station_lead.values()), result["candidate"]["worst_station_lead_delta_vs_v19_m"]
        ),
        "short_leads_bit_exact_uniform": bool(
            np.array_equal(
                candidate[np.isin(lead, [3, 6, 9, 12])], uniform[np.isin(lead, [3, 6, 9, 12])]
            )
        ),
        "finite_arrays": bool(
            all(np.isfinite(item).all() for item in (truth, persistence, uniform, v19, candidate))
        ),
        "six_ridge_outer_fits": result["fit_count"]["ridge_outer_fits"] == 6,
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
        "no_result_based_tuning": result["execution"]["result_based_tuning"] is False,
        "arrays_hash_matches": result["provenance"]["evaluation_arrays_sha256"]
        == __import__("hashlib").sha256(ARRAYS.read_bytes()).hexdigest(),
    }
    receipt = {
        "schema_version": "p3.kma_wind_work_residual_axis.independent_qa.v20",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recalculated_rmse_m": recalculated,
        "recalculated_by_block_delta_vs_v19_m": by_block,
        "recalculated_worst_station_lead_delta_vs_v19_m": max(station_lead.values()),
        "interpretation_boundary": (
            "Exploratory on an exposed historical development surface; this QA does not establish Public transport."
        ),
    }
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.write_bytes(canonical(receipt))
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
