"""Independent read-only QA for terminal P1 v32f."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_p1_ordered_catboost_eventday_20260831_v32a as reference_loader  # noqa: E402

EXPERIMENT_ID = "p1_ordered_catboost_causal_calibrated_20260831_v32f"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
KEYS = ["station", "year", "layer", "time"]


def metric(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "f1": float(f1_score(truth, prediction)),
        "precision": float(precision_score(truth, prediction, zero_division=1)),
        "recall": float(recall_score(truth, prediction, zero_division=1)),
        "positives": int(np.asarray(prediction, dtype=np.int8).sum()),
    }


def close(left: float, right: float) -> bool:
    return bool(abs(float(left) - float(right)) <= 1e-12)


def main() -> int:
    result_path = ARTIFACT / "result.json"
    oof_path = ARTIFACT / "historical_oof.parquet"
    config_path = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
    runner_path = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    frame = pd.read_parquet(oof_path)
    truth = frame["label"].to_numpy(np.int8)
    candidate = frame["candidate_prediction"].to_numpy(np.int8)
    tabular = frame["deployment_prediction"].to_numpy(np.int8)
    checks: dict[str, bool] = {
        "terminal_no_go": result["status"] == "NO_GO_INTERNAL_GATE",
        "rows_421032": len(frame) == 421_032,
        "keys_unique": not frame.duplicated(KEYS).any(),
        "candidate_binary": bool(np.isin(candidate, [0, 1]).all()),
        "probabilities_finite": bool(np.isfinite(frame["candidate_probability"]).all()),
        "fit_count_three": result["fit_count"] == 3,
        "runtime_within_cap": result["runtime_seconds"] <= config["maximum_runtime_seconds"],
        "official_access_zero": set(result["official_access"].values()) == {0},
        "oof_hash_matches": reference_loader.sha256_file(oof_path)
        == result["historical_oof"]["sha256"],
        "config_hash_matches": reference_loader.sha256_file(config_path)
        == result["input_hashes"]["config"],
        "no_submission_artifact": not any(
            "submission" in path.name.lower() for path in ARTIFACT.glob("**/*") if path.is_file()
        ),
    }
    fold_recomputed: dict[str, dict[str, object]] = {}
    for fold in ("2025_q2", "2025_q3", "2025_q4"):
        mask = frame["fold"].eq(fold).to_numpy()
        candidate_metric = metric(truth[mask], candidate[mask])
        tabular_metric = metric(truth[mask], tabular[mask])
        thresholds = frame.loc[mask, "calibrated_threshold"].unique()
        fold_recomputed[fold] = {
            "candidate": candidate_metric,
            "tabular": tabular_metric,
            "delta_f1": candidate_metric["f1"] - tabular_metric["f1"],
            "threshold": float(thresholds[0]),
        }
        checks[f"{fold}_one_threshold"] = len(thresholds) == 1
        checks[f"{fold}_threshold_matches"] = len(thresholds) == 1 and close(
            thresholds[0], result["thresholds"][fold]
        )
        checks[f"{fold}_threshold_on_grid"] = bool(
            np.any(np.isclose(np.arange(0.05, 0.951, 0.01), thresholds[0]))
        )
        checks[f"{fold}_candidate_f1_matches"] = close(
            candidate_metric["f1"], result["by_fold"][fold]["candidate"]["f1"]
        )
        checks[f"{fold}_tabular_f1_matches"] = close(
            tabular_metric["f1"], result["by_fold"][fold]["tabular_reference"]["f1"]
        )
    tabular_loaded, e150 = reference_loader.aligned_references(frame)
    checks["tabular_reference_reloaded_matches"] = bool(np.array_equal(tabular, tabular_loaded))
    q34 = frame["fold"].isin(["2025_q3", "2025_q4"]).to_numpy()
    q34_candidate = metric(truth[q34], candidate[q34])
    q34_e150 = metric(truth[q34], e150[q34])
    checks["q34_candidate_f1_matches"] = close(
        q34_candidate["f1"], result["q3_q4_vs_e150"]["candidate"]["f1"]
    )
    checks["q34_e150_f1_matches"] = close(
        q34_e150["f1"], result["q3_q4_vs_e150"]["reference"]["f1"]
    )
    pooled_candidate = metric(truth, candidate)
    pooled_tabular = metric(truth, tabular)
    checks["pooled_candidate_f1_matches"] = close(
        pooled_candidate["f1"], result["pooled_vs_tabular"]["candidate"]["f1"]
    )
    checks["pooled_tabular_f1_matches"] = close(
        pooled_tabular["f1"], result["pooled_vs_tabular"]["reference"]["f1"]
    )
    gates = {
        "all_q2_q3_q4_delta_f1_vs_tabular_nonnegative": all(
            float(record["delta_f1"]) >= 0.0 for record in fold_recomputed.values()
        ),
        "pooled_delta_f1_vs_tabular_positive": pooled_candidate["f1"] > pooled_tabular["f1"],
        "pooled_bootstrap_ci90_low_vs_tabular_positive": (
            result["pooled_vs_tabular"]["bootstrap"]["difference_ci90"][0] > 0.0
        ),
        "q3_q4_delta_f1_vs_e150_positive": q34_candidate["f1"] > q34_e150["f1"],
        "q3_q4_bootstrap_ci90_low_vs_e150_positive": (
            result["q3_q4_vs_e150"]["bootstrap"]["difference_ci90"][0] > 0.0
        ),
        "runtime_at_most_seconds": result["runtime_seconds"] <= config["maximum_runtime_seconds"],
        "official_accesses_equal_zero": set(result["official_access"].values()) == {0},
    }
    checks["gates_recomputed_match"] = gates == result["gates"]
    checks["calibration_periods_are_45_days"] = all(
        pd.Timestamp(record["calibration_max_time_utc"])
        - pd.Timestamp(record["calibration_min_time_utc"])
        < pd.Timedelta(days=45)
        and pd.Timestamp(record["calibration_max_time_utc"])
        - pd.Timestamp(record["calibration_min_time_utc"])
        > pd.Timedelta(days=44)
        for record in result["fit_records"]
    )
    checks["internal_purge_at_least_14_days"] = all(
        pd.Timestamp(record["calibration_min_time_utc"])
        - pd.Timestamp(record["fit_max_time_utc"])
        >= pd.Timedelta(days=14)
        for record in result["fit_records"]
    )
    checks["runner_no_materializer"] = all(
        token not in runner_path.read_text(encoding="utf-8")
        for token in ("predict_submission", "write_submission", "validate_submission")
    )
    qa = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "recomputed": {
            "folds": fold_recomputed,
            "pooled_candidate": pooled_candidate,
            "pooled_tabular": pooled_tabular,
            "q3_q4_candidate": q34_candidate,
            "q3_q4_e150": q34_e150,
            "gates": gates,
        },
        "hashes": {
            "config": reference_loader.sha256_file(config_path),
            "runner": reference_loader.sha256_file(runner_path),
            "result": reference_loader.sha256_file(result_path),
            "historical_oof": reference_loader.sha256_file(oof_path),
        },
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    output = REPORT / "independent-qa.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": qa["status"], "checks": f"{qa['checks_passed']}/{qa['checks_total']}"}))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
