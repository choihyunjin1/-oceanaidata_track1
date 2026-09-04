"""Independent recalculation QA for P3 joint wave-wind support v20a."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_kma_joint_wave_wind_support_cycle_20260901_v20a"
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
    comparator: np.ndarray,
    *keys: np.ndarray,
) -> dict[str, float]:
    labels = np.array(
        ["|".join(parts) for parts in zip(*(key.astype(str) for key in keys), strict=True)]
    )
    return {
        label: rmse(truth[labels == label], candidate[labels == label])
        - rmse(truth[labels == label], comparator[labels == label])
        for label in sorted(set(labels.tolist()))
    }


def close(left: float, right: float) -> bool:
    return bool(abs(left - right) <= 1e-12)


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    with np.load(ARRAYS, allow_pickle=False) as data:
        truth = data["truth"]
        persistence = data["persistence"]
        uniform = data["uniform"]
        v19 = data["v19"]
        candidate = data["candidate"]
        lead = data["lead_h"]
        block = data["block"]
        station = data["station"]
    recalculated = {
        "persistence": rmse(truth, persistence),
        "uniform_kma_0p425": rmse(truth, uniform),
        "v19_wave_power": rmse(truth, v19),
        "candidate": rmse(truth, candidate),
    }
    recalculated.update(
        {
            "delta_candidate_minus_persistence": recalculated["candidate"]
            - recalculated["persistence"],
            "delta_candidate_minus_uniform": recalculated["candidate"]
            - recalculated["uniform_kma_0p425"],
            "delta_candidate_minus_v19": recalculated["candidate"] - recalculated["v19_wave_power"],
        }
    )
    block_delta = group_delta(truth, candidate, v19, block)
    station_lead = group_delta(truth, candidate, v19, station, lead)
    reported = result["candidate"]
    non24 = lead != 24
    checks = {
        "complete": result["status"] == "COMPLETE",
        "rmse_recomputed": all(
            close(value, reported["rmse_m"][key]) for key, value in recalculated.items()
        ),
        "block_deltas_recomputed": all(
            close(value, reported["by_block_vs_v19"][key]["delta_rmse_m"])
            for key, value in block_delta.items()
        ),
        "improved_blocks_recomputed": sum(value < 0 for value in block_delta.values())
        == reported["improved_blocks_vs_v19"],
        "worst_station_lead_recomputed": close(
            max(station_lead.values()), reported["worst_station_lead_delta_vs_v19_m"]
        ),
        "non24_bit_exact_v19": bool(np.array_equal(candidate[non24], v19[non24])),
        "finite_arrays": bool(
            all(np.isfinite(item).all() for item in (truth, persistence, uniform, v19, candidate))
        ),
        "zero_target_fits": result["fit_count"]["target_fits"] == 0,
        "six_feature_only_fits": result["fit_count"]["wind_support_ecdf_feature_only_fits"] == 6,
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
        == hashlib.sha256(ARRAYS.read_bytes()).hexdigest(),
    }
    receipt = {
        "schema_version": "p3.kma_joint_wave_wind_support.independent_qa.v20a",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "recalculated_rmse_m": recalculated,
        "recalculated_by_block_delta_vs_v19_m": block_delta,
        "recalculated_worst_station_lead_delta_vs_v19_m": max(station_lead.values()),
        "interpretation_boundary": "EXPLORATORY_ONLY; exposed historical surface and no Public transport guarantee.",
    }
    if OUTPUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUTPUT}")
    OUTPUT.write_bytes(canonical(receipt))
    print(json.dumps({"status": receipt["status"], "checks": len(checks)}))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
