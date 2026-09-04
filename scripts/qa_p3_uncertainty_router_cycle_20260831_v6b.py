"""Independent recomputation QA for P3 uncertainty router v6b."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_uncertainty_router_cycle_20260831_v6b"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
RESULT_PATH = ARTIFACT_DIR / "result.json"
PREDICTION_PATH = ARTIFACT_DIR / "outer_predictions.parquet"
QA_PATH = REPORT_DIR / "independent-qa.json"
ATTEMPT_LOCK = ARTIFACT_DIR.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
RUNNER_PATH = ROOT / "scripts" / "run_p3_uncertainty_router_cycle_20260831_v6b.py"
KEYS = ["case_id", "station", "lead_h"]
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_REPLICATES = 5_000


def rmse(actual: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(actual - prediction))))


def independent_bootstrap(frame: pd.DataFrame) -> dict[str, float]:
    grouped = list(frame.groupby("episode_id", observed=True, sort=True))
    ref_sse = np.asarray(
        [np.square(group["reference"] - group["target_hs"]).sum() for _, group in grouped]
    )
    cand_sse = np.asarray(
        [np.square(group["candidate"] - group["target_hs"]).sum() for _, group in grouped]
    )
    counts = np.asarray([len(group) for _, group in grouped], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(grouped), size=len(grouped))
        denominator = counts[draw].sum()
        deltas[index] = np.sqrt(cand_sse[draw].sum() / denominator) - np.sqrt(
            ref_sse[draw].sum() / denominator
        )
    return {
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0.0)),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    lock = json.loads(ATTEMPT_LOCK.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(PREDICTION_PATH)
    checks: list[dict[str, Any]] = []
    records = {record["policy"]["name"]: record for record in result["candidates"]}
    for name, group in predictions.groupby("candidate_name", observed=True, sort=True):
        record = records[str(name)]
        actual = group["target_hs"].to_numpy(dtype=np.float64)
        reference = group["reference"].to_numpy(dtype=np.float64)
        candidate = group["candidate"].to_numpy(dtype=np.float64)
        recomputed_reference = rmse(actual, reference)
        recomputed_candidate = rmse(actual, candidate)
        bootstrap = independent_bootstrap(group)
        inactive = ~group["lead_h"].isin([18, 24]).to_numpy()
        item = {
            "candidate_name": str(name),
            "rows": int(len(group)),
            "finite": bool(np.isfinite(candidate).all()),
            "short_leads_exact_champion_noop": bool(
                np.array_equal(candidate[inactive], reference[inactive])
            ),
            "reference_rmse_abs_error": abs(recomputed_reference - record["reference_rmse"]),
            "candidate_rmse_abs_error": abs(recomputed_candidate - record["candidate_rmse"]),
            "delta_rmse_abs_error": abs(
                (recomputed_candidate - recomputed_reference) - record["delta_rmse"]
            ),
            "bootstrap_ci90_low_abs_error": abs(
                bootstrap["ci90_low"] - record["bootstrap"]["ci90_low"]
            ),
            "bootstrap_ci90_high_abs_error": abs(
                bootstrap["ci90_high"] - record["bootstrap"]["ci90_high"]
            ),
            "bootstrap_probability_abs_error": abs(
                bootstrap["probability_improved"]
                - record["bootstrap"]["probability_improved"]
            ),
        }
        item["pass"] = bool(
            item["rows"] == 1_092
            and item["finite"]
            and item["short_leads_exact_champion_noop"]
            and max(
                item["reference_rmse_abs_error"],
                item["candidate_rmse_abs_error"],
                item["delta_rmse_abs_error"],
                item["bootstrap_ci90_low_abs_error"],
                item["bootstrap_ci90_high_abs_error"],
                item["bootstrap_probability_abs_error"],
            )
            < 1e-12
        )
        checks.append(item)
    output_checks: list[dict[str, Any]] = []
    for output in result["outputs"]:
        path = Path(output["path"])
        frame = pd.read_csv(path, dtype={"case_id": "string", "station": "string"})
        item = {
            "candidate_name": output["candidate_name"],
            "exists": path.exists(),
            "rows": int(len(frame)),
            "columns_exact": list(frame.columns) == KEYS + ["hs_pred"],
            "duplicate_keys": int(frame.duplicated(KEYS).sum()),
            "finite": bool(np.isfinite(frame["hs_pred"].to_numpy(dtype=np.float64)).all()),
            "range_valid": bool(frame["hs_pred"].between(0.0, 30.0).all()),
            "sha256_match": sha256(path) == output["sha256"],
        }
        item["pass"] = bool(
            item["exists"]
            and item["rows"] == 1_200
            and item["columns_exact"]
            and item["duplicate_keys"] == 0
            and item["finite"]
            and item["range_valid"]
            and item["sha256_match"]
        )
        output_checks.append(item)
    encoded_policies = json.dumps(
        result["design"]["policies"], sort_keys=True, separators=(",", ":")
    ).encode()
    recomputed_policy_sha256 = hashlib.sha256(encoded_policies).hexdigest()
    contract_checks = {
        "attempt_lock_exists": ATTEMPT_LOCK.exists(),
        "runner_sha256_matches_result": sha256(RUNNER_PATH) == result["runner_sha256"],
        "runner_sha256_matches_attempt_lock": sha256(RUNNER_PATH) == lock["runner_sha256"],
        "policy_sha256_matches_result": (
            recomputed_policy_sha256 == result["policy_contract_sha256"]
        ),
        "policy_sha256_matches_attempt_lock": (
            recomputed_policy_sha256 == lock["policy_contract_sha256"]
        ),
        "three_frozen_policies": len(result["design"]["policies"]) == 3,
        "fit_count_is_252": result["fit_count_total"] == 252,
        "no_result_based_retry_or_threshold_tuning": (
            not result["result_based_retry_or_threshold_tuning"]
            and not lock["result_based_retry_or_threshold_tuning"]
        ),
    }
    qa = {
        "schema_version": "p3.uncertainty_advantage_router.independent_qa.20260831.v6b",
        "experiment_id": EXPERIMENT_ID,
        "status": "PASS"
        if len(checks) == 3
        and all(item["pass"] for item in checks)
        and all(item["pass"] for item in output_checks)
        and all(contract_checks.values())
        and result["hidden_truth_rows_read"] == 0
        and result["uploads"] == 0
        else "FAIL",
        "candidate_checks": checks,
        "output_checks": output_checks,
        "contract_checks": contract_checks,
        "official_access": result["official_access"],
        "hidden_truth_rows_read": result["hidden_truth_rows_read"],
        "uploads": result["uploads"],
        "result_sha256": sha256(RESULT_PATH),
        "outer_predictions_sha256": sha256(PREDICTION_PATH),
    }
    QA_PATH.write_text(
        json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if qa["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
