"""Independent post-run QA for P3 v30 directed transfer entropy residuals."""

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

EXPERIMENT_ID = "p3_directed_transfer_entropy_residual_cycle_20260901_v30"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports" / EXPERIMENT_ID
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = ROOT / "scripts" / f"run_{EXPERIMENT_ID}.py"
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
    predictions = [
        arrays["candidate_1"].astype(np.float64),
        arrays["candidate_2"].astype(np.float64),
    ]
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
    checks: dict[str, bool] = {
        "terminal_complete": result["status"] == "COMPLETE",
        "terminal_no_go": result["decision"]
        == "NO_GO_ALL_DIRECTED_INFORMATION_CANDIDATES",
        "shape_182_by_6": truth.shape
        == reference.shape
        == predictions[0].shape
        == predictions[1].shape
        == (182, 6),
        "finite_arrays": all(
            np.isfinite(item).all() for item in (truth, reference, *predictions)
        ),
        "fit_count_12": result["fit_count"] == 12
        and len(result["fit_receipts"]) == 12,
        "six_outer_blocks_per_candidate": all(
            sum(
                receipt["candidate"] == candidate["name"]
                for receipt in result["fit_receipts"]
            )
            == 6
            for candidate in result["candidates"]
        ),
        "sealed_directed_information_features": result["feature_receipt"]["columns"]
        == 180,
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
        "lock_consumed_once": lock["status"] == "ATTEMPT_CONSUMED_ONE_SHOT",
        "runner_hash": result["provenance"]["runner_sha256"] == qa.sha256(RUNNER),
        "config_hash": result["provenance"]["config_sha256"] == qa.sha256(CONFIG),
        "arrays_hash": result["provenance"]["evaluation_arrays_sha256"]
        == qa.sha256(ARRAYS),
        "no_result_tuning": not result["execution"]["result_based_tuning"]
        and result["execution"]["outer_result_parameter_changes"] == 0,
    }
    baseline = qa.rmse(truth, reference)
    metric_match = slice_match = ci_match = point_match = decision_match = True
    for index, (candidate, prediction) in enumerate(
        zip(result["candidates"], predictions, strict=True)
    ):
        flat = prediction.reshape(-1)
        after = qa.rmse(truth, prediction)
        delta = after - baseline
        stored = candidate["rmse_m"]
        metric_match &= abs(stored["uniform_0p425"] - baseline) < 1e-12
        metric_match &= abs(stored["candidate"] - after) < 1e-12
        metric_match &= abs(stored["delta_candidate_minus_uniform"] - delta) < 1e-12
        for columns, key in (
            (["block"], "by_block"),
            (["station"], "station"),
            (["lead_h"], "lead"),
            (["station", "lead_h"], "station_lead"),
        ):
            slice_match &= qa.close_dict(
                qa.grouped_delta(frame, flat, columns), candidate[key]
            )
        work = frame.assign(candidate=flat)
        for block, group in work.groupby("block", observed=True):
            threshold = float(np.quantile(group["reference"].to_numpy(float), 0.80))
            mask = group["reference"].to_numpy(float) >= threshold
            delta_tail = qa.rmse(
                group.loc[mask, "target_hs"].to_numpy(float),
                group.loc[mask, "candidate"].to_numpy(float),
            ) - qa.rmse(
                group.loc[mask, "target_hs"].to_numpy(float),
                group.loc[mask, "reference"].to_numpy(float),
            )
            slice_match &= abs(
                delta_tail
                - candidate["reference_tail_by_block"][str(block)]["delta_rmse_m"]
            ) < 1e-12
        episode_ci = bootstrap(frame, flat, ("episode_id",), 20261301 + index * 100)
        group_ci = bootstrap(frame, flat, ("block", "station"), 20261302 + index * 100)
        ci_match &= np.allclose(
            episode_ci["ci90_m"], candidate["episode_bootstrap"]["ci90_m"], atol=1e-12
        ) and np.allclose(
            group_ci["ci90_m"],
            candidate["block_station_bootstrap"]["ci90_m"],
            atol=1e-12,
        )
        raw_points = -delta * POINTS_PER_RMSE_M
        point_match &= abs(candidate["expected_points"]["raw_gain"] - raw_points) < 1e-12
        point_match &= abs(
            candidate["expected_points"]["transport_adjusted_gain"]
            - (raw_points - qa.TRANSPORT_PENALTY)
        ) < 1e-12
        point_match &= abs(
            candidate["expected_points"]["nominal_official_score"]
            - (qa.OFFICIAL_POINTS + raw_points)
        ) < 1e-12
        stable = all(candidate["stable_checks"].values())
        high_risk = (not stable) and all(candidate["high_risk_checks"].values())
        expected = (
            "PASS_STABLE"
            if stable
            else "PRESERVE_HIGH_RISK"
            if high_risk
            else "NO_GO"
        )
        decision_match &= candidate["decision"] == expected
    checks.update(
        {
            "independent_metric_recalculation": metric_match,
            "block_station_lead_tail_recalculation": slice_match,
            "bootstrap_ci_recalculation": ci_match,
            "canonical_points_recalculation": point_match,
            "decision_recalculation": decision_match,
        }
    )
    payload = {
        "schema_version": "p3.directed_transfer_entropy_residual.independent_qa.v30",
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
