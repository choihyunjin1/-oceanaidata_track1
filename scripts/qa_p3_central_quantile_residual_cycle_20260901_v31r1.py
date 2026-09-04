"""Independent post-run QA for P3 v31r1 quantile recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "scripts"))

import qa_p3_nonlinear_recurrence_ordinal_residual_cycle_20260901_v29 as qa  # noqa: E402
from run_p3_sors_longlead_episode_selector_cycle_20260831_v11 import (  # noqa: E402
    POINTS_PER_RMSE_M,
    bootstrap,
)

EXPERIMENT_ID = "p3_central_quantile_residual_cycle_20260901_v31r1"
SOURCE_ID = "p3_central_quantile_residual_cycle_20260901_v31"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
SOURCE_CONFIG = ROOT / "configs/experiments" / f"{SOURCE_ID}.json"
SOURCE_RUNNER = ROOT / "scripts" / f"run_{SOURCE_ID}.py"
SOURCE_LOCK = ROOT / "artifacts" / f"{SOURCE_ID}.ATTEMPT_LOCK.json"
SOURCE_FAILURE = ROOT / "reports" / SOURCE_ID / "failure-receipt.json"
SOURCE_RESULT = ROOT / "artifacts" / SOURCE_ID / "result.json"
RESULT = ARTIFACT / "result.json"
ARRAYS = ARTIFACT / "evaluation-arrays.npz"
LOCK = ARTIFACT.parent / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
OUTPUT = REPORT / "independent-qa.json"


def main() -> int:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    arrays = np.load(ARRAYS, allow_pickle=False)
    truth = arrays["truth"].astype(np.float64)
    reference = arrays["uniform"].astype(np.float64)
    prediction = arrays["candidate_1"].astype(np.float64)
    leads = arrays["lead_h"].astype(np.int64)
    frame = pd.DataFrame(
        {
            "target_hs": truth.reshape(-1),
            "reference": reference.reshape(-1),
            "block": np.repeat(arrays["block"].astype(str), len(leads)),
            "station": np.repeat(arrays["station"].astype(str), len(leads)),
            "episode_id": np.repeat(arrays["episode"].astype(str), len(leads)),
            "lead_h": np.tile(leads, len(truth)),
        }
    )
    candidate = result["candidate"]
    checks: dict[str, bool] = {
        "terminal_complete": result["status"] == "COMPLETE",
        "terminal_no_go": result["decision"] == "NO_GO_CENTRAL_QUANTILE_CANDIDATE",
        "shape_182_by_6": truth.shape == reference.shape == prediction.shape == (182, 6),
        "finite_arrays": all(
            np.isfinite(item).all() for item in (truth, reference, prediction)
        ),
        "fit_count_12": result["fit_count"] == 12
        and len(result["fit_receipts"]) == 12,
        "two_quantiles_per_block": all(
            sum(receipt["block"] == block for receipt in result["fit_receipts"]) == 2
            for block in ("01_02", "03_04", "05_06", "07_08", "09_10", "11_12")
        ),
        "sealed_case_features": result["feature_receipt"]["columns"] == 108,
        "zero_row_deletion": all(
            receipt["row_deletion"] == 0 for receipt in result["fit_receipts"]
        ),
        "official_access_zero": all(
            value == 0
            for key, value in result["data_access"].items()
            if key != "historical_target_rows"
        ),
        "config_official_zero": all(
            value == 0 for value in config["official_policy"].values()
        ),
        "source_result_absent": not SOURCE_RESULT.exists(),
        "source_evidence_hashes": result["provenance"]["source_config_sha256"]
        == qa.sha256(SOURCE_CONFIG)
        and result["provenance"]["source_runner_sha256"] == qa.sha256(SOURCE_RUNNER)
        and result["provenance"]["source_lock_sha256"] == qa.sha256(SOURCE_LOCK)
        and result["provenance"]["source_failure_receipt_sha256"]
        == qa.sha256(SOURCE_FAILURE),
        "science_neutral_recovery": result["execution"]["science_changes"] == 0
        and result["execution"]["scorer_adapter_changes"] == 1,
        "lock_consumed_once": lock["status"] == "ATTEMPT_CONSUMED_ONE_SHOT",
        "runner_hash": result["provenance"]["runner_sha256"] == qa.sha256(RUNNER),
        "config_hash": result["provenance"]["config_sha256"] == qa.sha256(CONFIG),
        "arrays_hash": result["provenance"]["evaluation_arrays_sha256"]
        == qa.sha256(ARRAYS),
        "no_result_tuning": not result["execution"]["result_based_tuning"]
        and result["execution"]["outer_result_parameter_changes"] == 0,
    }
    baseline = qa.rmse(truth, reference)
    after = qa.rmse(truth, prediction)
    delta = after - baseline
    stored = candidate["rmse_m"]
    checks["independent_metric_recalculation"] = (
        abs(stored["uniform_0p425"] - baseline) < 1e-12
        and abs(stored["candidate"] - after) < 1e-12
        and abs(stored["delta_candidate_minus_uniform"] - delta) < 1e-12
    )
    flat = prediction.reshape(-1)
    slice_match = True
    for columns, key in (
        (["block"], "by_block"),
        (["station"], "station"),
        (["lead_h"], "lead"),
        (["station", "lead_h"], "station_lead"),
    ):
        slice_match &= qa.close_dict(qa.grouped_delta(frame, flat, columns), candidate[key])
    work = frame.assign(candidate=flat)
    for block, group in work.groupby("block", observed=True):
        threshold = float(np.quantile(group["reference"].to_numpy(float), 0.80))
        mask = group["reference"].to_numpy(float) >= threshold
        tail = qa.rmse(
            group.loc[mask, "target_hs"].to_numpy(float),
            group.loc[mask, "candidate"].to_numpy(float),
        ) - qa.rmse(
            group.loc[mask, "target_hs"].to_numpy(float),
            group.loc[mask, "reference"].to_numpy(float),
        )
        slice_match &= abs(
            tail - candidate["reference_tail_by_block"][str(block)]["delta_rmse_m"]
        ) < 1e-12
    checks["block_station_lead_tail_recalculation"] = slice_match
    episode_ci = bootstrap(frame, flat, ("episode_id",), 20261301)
    group_ci = bootstrap(frame, flat, ("block", "station"), 20261302)
    checks["bootstrap_ci_recalculation"] = np.allclose(
        episode_ci["ci90_m"], candidate["episode_bootstrap"]["ci90_m"], atol=1e-12
    ) and np.allclose(
        group_ci["ci90_m"],
        candidate["block_station_bootstrap"]["ci90_m"],
        atol=1e-12,
    )
    raw_points = -delta * POINTS_PER_RMSE_M
    checks["canonical_points_recalculation"] = (
        abs(candidate["expected_points"]["raw_gain"] - raw_points) < 1e-12
        and abs(
            candidate["expected_points"]["transport_adjusted_gain"]
            - (raw_points - qa.TRANSPORT_PENALTY)
        )
        < 1e-12
        and abs(
            candidate["expected_points"]["nominal_official_score"]
            - (qa.OFFICIAL_POINTS + raw_points)
        )
        < 1e-12
    )
    stable = all(candidate["stable_checks"].values())
    high_risk = (not stable) and all(candidate["high_risk_checks"].values())
    expected = (
        "PASS_STABLE" if stable else "PRESERVE_HIGH_RISK" if high_risk else "NO_GO"
    )
    checks["decision_recalculation"] = candidate["decision"] == expected
    payload = {
        "schema_version": "p3.central_quantile_residual.recovery_independent_qa.v31r1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "check_count": len(checks),
        "passed": sum(checks.values()),
        "failed": sum(not value for value in checks.values()),
        "model_fits": result["fit_count"],
        "official_rows": 0,
        "csv_materializations": 0,
        "uploads": 0,
        "hashes": {
            "result": qa.sha256(RESULT),
            "arrays": qa.sha256(ARRAYS),
            "runner": qa.sha256(RUNNER),
            "config": qa.sha256(CONFIG),
            "source_failure_receipt": qa.sha256(SOURCE_FAILURE),
        },
        "caveat": "The 182-case surface is repeatedly exposed and EXPLORATORY_ONLY; no Public transport is guaranteed.",
    }
    OUTPUT.write_bytes(qa.canonical(payload))
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": payload["check_count"],
                "passed": payload["passed"],
                "failed": payload["failed"],
                "model_fits": payload["model_fits"],
                "official_rows": 0,
                "csv_materializations": 0,
                "uploads": 0,
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["decision"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
