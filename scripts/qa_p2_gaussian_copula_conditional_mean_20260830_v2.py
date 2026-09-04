"""Independent result and lineage QA for the completed P2 copula pilot."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for _directory in (ROOT, SRC):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

from p2_restore.p2_alpha40_quasiperiodic_gp_residual_20260828_v1 import (  # noqa: E402
    paired_kst_day_bootstrap,
)
from scripts import (  # noqa: E402
    run_p2_gaussian_copula_conditional_mean_20260830_v1 as engine,
)

EXPERIMENT_ID = "p2_gaussian_copula_conditional_mean_20260830_v2"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
OVERLAY = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
BASE_CONFIG = ROOT / "configs/experiments/p2_gaussian_copula_conditional_mean_20260830_v1.json"
ANCHOR = ROOT / "artifacts/p2_state_conditional_lean_v1/oof.parquet"


def metric(frame: pd.DataFrame) -> dict[str, float | int]:
    reference = float(
        np.sqrt(np.mean(np.square(frame["reference"].to_numpy() - frame["truth"].to_numpy())))
    )
    candidate = float(
        np.sqrt(np.mean(np.square(frame["candidate"].to_numpy() - frame["truth"].to_numpy())))
    )
    return {
        "rows": int(len(frame)),
        "reference_rmse": reference,
        "candidate_rmse": candidate,
        "delta_rmse": candidate - reference,
    }


def collect_model_receipts(value: Any) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {
            "minimum_eigenvalue",
            "condition_number",
            "finite_monotone_empirical_margins",
            "kendall_tau_latent_correlation",
        }.issubset(value):
            receipts.append(value)
        for child in value.values():
            receipts.extend(collect_model_receipts(child))
    elif isinstance(value, list):
        for child in value:
            receipts.extend(collect_model_receipts(child))
    return receipts


def run() -> dict[str, Any]:
    commitment_path = ARTIFACT / "prediction_commitment.json"
    result_path = ARTIFACT / "result.json"
    lock_path = ARTIFACT / "attempt.lock.json"
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    overlay = json.loads(OVERLAY.read_text(encoding="utf-8"))
    base_config = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    frames: list[pd.DataFrame] = []
    prediction_hashes: dict[str, str] = {}
    incomplete_receipts: dict[str, Any] = {}
    prediction_schema_pass = True
    for fold, record in commitment["outputs"].items():
        path = ROOT / record["path"]
        digest = engine.base.sha256_file(path)
        prediction_hashes[fold] = digest
        prediction_schema_pass &= digest == record["sha256"]
        prediction = engine.read_prediction(path)
        prediction_schema_pass &= list(prediction.columns) == [
            "time",
            "layer",
            "current_blend50",
            "reference",
            "candidate",
            "correction",
        ]
        prediction_schema_pass &= len(prediction) == int(record["rows"])
        counts = prediction.groupby("time", sort=False)["layer"].nunique()
        incomplete_times = counts.index[counts < 3]
        incomplete = prediction["time"].isin(incomplete_times)
        incomplete_correction = prediction.loc[incomplete, "correction"].to_numpy(
            dtype=np.float64
        )
        max_incomplete_correction = float(
            np.max(np.abs(incomplete_correction), initial=0.0)
        )
        incomplete_receipts[fold] = {
            "incomplete_profile_times": int(len(incomplete_times)),
            "incomplete_rows": int(incomplete.sum()),
            "maximum_absolute_correction": max_incomplete_correction,
        }
        truth = pd.read_parquet(
            ANCHOR,
            columns=["time", "layer", "truth", "block"],
            filters=[("block", "==", fold)],
        ).drop(columns="block")
        truth["time"] = pd.to_datetime(truth["time"], utc=True)
        scored = prediction.merge(truth, on=["time", "layer"], validate="one_to_one")
        scored["fold"] = fold
        frames.append(scored)
    scored = pd.concat(frames, ignore_index=True)
    metrics = {
        "aggregate": metric(scored),
        "by_fold": {
            str(key): metric(group) for key, group in scored.groupby("fold", sort=True)
        },
        "by_layer": {
            str(int(key)): metric(group)
            for key, group in scored.groupby("layer", sort=True)
        },
    }
    bootstrap = paired_kst_day_bootstrap(
        scored,
        replicates=int(base_config["gate"]["bootstrap_replicates"]),
        seed=int(base_config["gate"]["bootstrap_seed"]),
    )
    fold_deltas = [float(value["delta_rmse"]) for value in metrics["by_fold"].values()]
    layer_deltas = [float(value["delta_rmse"]) for value in metrics["by_layer"].values()]
    primary_checks = {
        "pooled_delta_rmse_lt_0": metrics["aggregate"]["delta_rmse"] < 0.0,
        "at_least_two_of_three_folds_improve": sum(value < 0.0 for value in fold_deltas) >= 2,
        "no_target_layer_worse_by_more_than_0_001_c": max(layer_deltas) <= 0.001,
        "paired_bootstrap_upper_lt_0": bootstrap["ci90_high"] < 0.0,
    }
    model_receipts = collect_model_receipts(commitment)
    maximum_condition = max(float(value["condition_number"]) for value in model_receipts)
    minimum_eigenvalue = min(float(value["minimum_eigenvalue"]) for value in model_receipts)
    structural_checks = {
        "prediction_schema_and_hashes": bool(prediction_schema_pass),
        "finite_monotone_empirical_margins": all(
            bool(value["finite_monotone_empirical_margins"]) for value in model_receipts
        ),
        "kendall_tau_latent_correlation": all(
            bool(value["kendall_tau_latent_correlation"]) for value in model_receipts
        ),
        "covariance_psd": minimum_eigenvalue >= -1e-10,
        "condition_number_guard": maximum_condition
        <= float(base_config["copula"]["maximum_condition_number"]),
        "incomplete_profile_exact_noop": all(
            value["maximum_absolute_correction"] <= 1e-12
            for value in incomplete_receipts.values()
        ),
        "fit_budget_exact": int(commitment["fit_counts"]["total_copula_fits"]) == 30,
        "truth_late_commitment": (
            commitment["truth_metric_computed"] is False
            and int(commitment["outer_validation_truth_rows_read"]) == 0
        ),
        "official_access_csv_upload_zero": (
            int(commitment["official_rows_read"]) == 0
            and int(result["official_hidden_test_sample_submission_rows_read"]) == 0
            and int(result["submission_csv_count"]) == 0
            and int(result["upload_count"]) == 0
        ),
    }
    metric_tolerance = 1e-12
    metric_match = all(
        abs(float(metrics["aggregate"][key]) - float(result["metrics"]["aggregate"][key]))
        <= metric_tolerance
        for key in ("reference_rmse", "candidate_rmse", "delta_rmse")
    ) and abs(float(bootstrap["ci90_high"]) - float(result["bootstrap"]["ci90_high"])) <= metric_tolerance
    strict_checks = {key: bool(value) for key, value in result["gate_checks"].items()}
    checks = {
        **structural_checks,
        "independent_metrics_match": metric_match,
        "primary_preregistered_promotion_checks_all_pass": all(primary_checks.values()),
        "strict_execution_gate_fails": not all(strict_checks.values()),
        "v1_is_invalid_not_scientific_no_go": overlay["failed_attempt"]["classification"]
        == "INVALID_TERMINAL_TECHNICAL_FAILURE",
        "base_config_hash_match": engine.base.sha256_file(BASE_CONFIG)
        == overlay["base_config"]["sha256"],
    }
    return {
        "schema_version": "p2.gaussian_copula.independent_qa.v2",
        "experiment_id": EXPERIMENT_ID,
        "qa_status": "PASS" if all(checks.values()) else "FAIL",
        "scientific_classification": "PRIMARY_SIGNAL_PASS_STRICT_STABILITY_NO_GO_RESEARCH_ONLY",
        "checks": checks,
        "primary_preregistered_promotion_checks": primary_checks,
        "strict_execution_checks": strict_checks,
        "independent_metrics": metrics,
        "independent_bootstrap": bootstrap,
        "incomplete_profile_receipts": incomplete_receipts,
        "covariance_receipt": {
            "model_receipts": len(model_receipts),
            "minimum_eigenvalue": minimum_eigenvalue,
            "maximum_condition_number": maximum_condition,
        },
        "hashes": {
            "overlay_config": engine.base.sha256_file(OVERLAY),
            "base_config": engine.base.sha256_file(BASE_CONFIG),
            "attempt_lock": engine.base.sha256_file(lock_path),
            "prediction_commitment": engine.base.sha256_file(commitment_path),
            "result": engine.base.sha256_file(result_path),
            "predictions": prediction_hashes,
        },
        "fit_counts": commitment["fit_counts"],
        "official_rows_read": 0,
        "csv_count": 0,
        "upload_count": 0,
    }


def main() -> None:
    qa = run()
    path = REPORT / "independent-qa.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
