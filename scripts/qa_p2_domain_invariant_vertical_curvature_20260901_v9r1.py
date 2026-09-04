"""Independent metric/hash QA for the terminal P2 v9r1 exploratory run."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_domain_invariant_vertical_curvature_20260901_v9r1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
TARGET_LAYERS = (2, 3, 4)
FOLD_ORDER = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def close(left: float, right: float, tolerance: float = 1e-11) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)


def independent_bootstrap(
    fold: np.ndarray,
    day: np.ndarray,
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "fold": fold,
            "day": day,
            "truth": truth,
            "reference": reference,
            "candidate": candidate,
        }
    )
    pieces = list(frame.groupby(["fold", "day"], sort=True, observed=True))
    counts = np.asarray([len(part) for _, part in pieces], dtype=float)
    reference_sse = np.asarray(
        [np.square(part.reference - part.truth).sum() for _, part in pieces]
    )
    candidate_sse = np.asarray(
        [np.square(part.candidate - part.truth).sum() for _, part in pieces]
    )
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=float)
    for index in range(replicates):
        draw = rng.integers(0, len(pieces), len(pieces))
        rows = counts[draw].sum()
        delta[index] = math.sqrt(candidate_sse[draw].sum() / rows) - math.sqrt(
            reference_sse[draw].sum() / rows
        )
    return {
        "ci90_low": float(np.quantile(delta, 0.05)),
        "ci90_high": float(np.quantile(delta, 0.95)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }


def main() -> None:
    result = json.loads((ARTIFACT / "result.json").read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    data_dir = os.environ.get("P2_DATA_DIR")
    if not data_dir:
        raise RuntimeError("P2_DATA_DIR is required")
    observations_path = Path(data_dir).resolve() / "observations.csv"
    if sha256_file(observations_path) != config["source_contract"]["observations_sha256"]:
        raise RuntimeError("observations hash drift")
    observations = pd.read_csv(
        observations_path, usecols=["time", "layer", "temp"], dtype={"time": "string"}
    )
    observations["time_ns"] = (
        pd.DatetimeIndex(pd.to_datetime(observations["time"], utc=True))
        .as_unit("ns")
        .asi8
    )
    truth_index = observations.set_index(["time_ns", "layer"])["temp"]
    if truth_index.index.has_duplicates:
        raise RuntimeError("truth keys duplicated")

    checks: dict[str, bool] = {
        "terminal_status_no_go": result["status"]
        == "EXPLORATORY_NO_GO_BOTH_SEALED_CANDIDATES",
        "claim_level_exploratory": result["claim_level"]
        == "EXPLORATORY_ONLY_NO_FRESH_CONFIRMATION",
        "fit_count_exact_6": result["fit_count"] == 6,
        "candidate_count_exact_2": len(result["candidates"]) == 2,
        "config_hash": result["hashes"]["config"] == sha256_file(CONFIG),
        "runner_hash": result["hashes"]["runner"] == sha256_file(RUNNER),
        "observations_hash": result["hashes"]["observations"]
        == sha256_file(observations_path),
        "scoring_hash": result["hashes"]["scoring_frame"]
        == sha256_file(ROOT / config["source_contract"]["scoring_frame"]),
        "v7_report_attested_unsubmitted": result["v7_readiness_audit"]["status"]
        == "REPORT_ATTESTED_READY_UNSUBMITTED_HIGH_TRANSPORT_RISK",
        "official_access_zero": all(
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
        "technical_recovery_receipt_exists": (ARTIFACT / "technical_recovery.json").is_file(),
    }
    candidate_qa: dict[str, Any] = {}
    bootstrap_seed = int(config["evaluation"]["bootstrap_seed"])
    bootstrap_replicates = int(config["evaluation"]["bootstrap_replicates"])
    for item in result["candidates"]:
        commitment = item["prediction_commitment"]
        path = Path(commitment["path"])
        arrays = np.load(path, allow_pickle=False)
        time_ns = arrays["time_ns"].astype(np.int64)
        layer = arrays["layer"].astype(int)
        fold = arrays["fold"].astype(str)
        reference = arrays["reference"].astype(float)
        candidate = arrays["candidate"].astype(float)
        index = pd.MultiIndex.from_arrays([time_ns, layer])
        truth = truth_index.reindex(index).to_numpy(float)
        if not np.isfinite(truth).all():
            raise RuntimeError("independent truth alignment failed")
        local_day = (
            pd.DatetimeIndex(pd.to_datetime(time_ns, unit="ns", utc=True))
            .tz_convert("Asia/Seoul")
            .date
        )
        local_month = np.asarray(
            pd.DatetimeIndex(pd.to_datetime(time_ns, unit="ns", utc=True))
            .tz_convert("Asia/Seoul")
            .strftime("%Y-%m"),
            dtype=str,
        )
        recomputed_delta = rmse(truth, candidate) - rmse(truth, reference)
        by_fold = {}
        by_layer = {}
        by_month = {}
        for value in FOLD_ORDER:
            selected = fold == value
            by_fold[value] = rmse(truth[selected], candidate[selected]) - rmse(
                truth[selected], reference[selected]
            )
        for value in TARGET_LAYERS:
            selected = layer == value
            by_layer[str(value)] = rmse(truth[selected], candidate[selected]) - rmse(
                truth[selected], reference[selected]
            )
        for value in sorted(np.unique(local_month)):
            selected = local_month == value
            by_month[value] = {
                "rows": int(selected.sum()),
                "delta_rmse_C": rmse(truth[selected], candidate[selected])
                - rmse(truth[selected], reference[selected]),
            }
        bootstrap = independent_bootstrap(
            fold,
            np.asarray(local_day, dtype=object),
            truth,
            reference,
            candidate,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        )
        record_checks = {
            "prediction_hash": commitment["sha256"] == sha256_file(path),
            "rows": len(candidate) == commitment["rows"] == 69850,
            "finite": bool(np.isfinite(reference).all() and np.isfinite(candidate).all()),
            "pooled_delta": close(recomputed_delta, item["delta_rmse"]),
            "fold_deltas": all(
                close(by_fold[key], item["by_fold"][key]["delta_rmse"])
                for key in FOLD_ORDER
            ),
            "layer_deltas": all(
                close(by_layer[key], item["by_layer"][key]["delta_rmse"])
                for key in by_layer
            ),
            "bootstrap_ci90": close(
                bootstrap["ci90_low"], item["bootstrap"]["ci90_low"]
            )
            and close(bootstrap["ci90_high"], item["bootstrap"]["ci90_high"]),
            "bootstrap_probability": close(
                bootstrap["probability_improved"],
                item["bootstrap"]["probability_improved"],
            ),
            "metrics_after_commitment": commitment["metric_computed_at_commitment"]
            is False,
            "truth_not_used_for_action": commitment["truth_used_to_choose_action"] is False,
            "three_layer_fits": len(commitment["fit_receipts"]) == 3,
            "row_deletion_zero": all(
                receipt["winsor"]["rows_deleted"] == 0
                for receipt in commitment["fit_receipts"]
            ),
        }
        checks[f"candidate_{item['name']}_all_checks"] = all(record_checks.values())
        candidate_qa[item["name"]] = {
            "checks": record_checks,
            "recomputed_pooled_delta_rmse_C": recomputed_delta,
            "recomputed_by_fold_delta_rmse_C": by_fold,
            "recomputed_by_layer_delta_rmse_C": by_layer,
            "recomputed_by_month": by_month,
            "recomputed_bootstrap": bootstrap,
            "prediction_sha256": sha256_file(path),
        }

    payload = {
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "candidate_qa": candidate_qa,
        "operation_counters": {
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
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "independent-qa.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    if payload["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
