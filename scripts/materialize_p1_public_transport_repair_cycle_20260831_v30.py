"""Exactly-once full-history materializer for the frozen P1 v30 PASS."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for directory in (ROOT, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_full_internal_submission_cycle_20260831_v2 as source_cycle  # noqa: E402
import run_p1_parallel_candidate_cycle_20260831_v4 as official_source  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v16 as surface  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v28 as v28  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v30 as v30  # noqa: E402

from src.p1_qc.label_free_reliability_cap import (  # noqa: E402
    apply_label_free_day_cap,
    fit_label_free_group_reliability,
    reliability_margin_lower_bound,
)
from src.p1_qc.prequential_label_shift_em import label_shift_em  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v30_materializer"
CONFIG = ROOT / "configs/experiments/p1_public_transport_repair_cycle_20260831_v30.json"
INTERNAL_RESULT = ROOT / "artifacts/p1_public_transport_repair_cycle_20260831_v30/result.json"
EXPECTED_INTERNAL_RESULT_SHA256 = (
    "5f47835934bdfababb2a86c7674596a39ddc64da677609c7992d83c359b04826"
)
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT = ROOT / "reports/p1_public_transport_repair_cycle_20260831_v30/materialization-result.json"
DELIVERY = ROOT / "submissions/p1_public_transport_repair_cycle_20260831_v30"
OUTPUT = DELIVERY / "P1_submission.csv"
P1_KEYS = ["station", "year", "layer", "time"]


class ContractError(RuntimeError):
    """Frozen v30 deployment or output contract violation."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract() -> tuple[dict, dict]:
    config = v30.load_contract()
    result = json.loads(INTERNAL_RESULT.read_text(encoding="utf-8"))
    checks = {
        "internal_result_sha": sha256(INTERNAL_RESULT)
        == EXPECTED_INTERNAL_RESULT_SHA256,
        "internal_terminal": result["status"] == "COMPLETE_INTERNAL_ONLY",
        "internal_pass": result["pass_count"] == 1
        and result["candidate"]["strict_internal_pass"] is True,
        "exact_candidate": result["candidate"]["name"] == config["candidate"],
        "two_historical_fits": result["fit_count"] == 2,
        "all_internal_gates_pass": all(result["candidate"]["gates"].values()),
        "official_pre_materialization_zero": result["operations"]["official_reads"] == 0,
        "hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "csv_zero": result["operations"]["submission_csv_created"] == 0,
        "upload_zero": result["operations"]["uploads"] == 0,
    }
    if not all(checks.values()):
        raise ContractError(f"v30 internal PASS contract mismatch: {checks}")
    return config, result


def validate_output_frame(submission: pd.DataFrame, official_keys: pd.DataFrame) -> dict:
    checks = {
        "rows_169011": len(submission) == len(official_keys) == 169_011,
        "schema_exact": list(submission.columns) == [*P1_KEYS, "label"],
        "keys_unique": not submission.duplicated(P1_KEYS).any(),
        "key_order_exact": all(
            np.array_equal(
                submission[key].astype(str).to_numpy(),
                official_keys[key].astype(str).to_numpy(),
            )
            for key in P1_KEYS
        ),
        "label_binary": set(submission["label"].unique()).issubset({0, 1}),
        "label_finite": bool(np.isfinite(submission["label"].to_numpy(float)).all()),
        "label_integer": pd.api.types.is_integer_dtype(submission["label"].dtype),
        "duplicate_rows_zero": not submission.duplicated().any(),
    }
    if not all(checks.values()):
        raise ContractError(f"v30 official output contract failed: {checks}")
    return checks


def preflight() -> dict:
    config, _ = load_contract()
    seed_paths = sorted(
        official_source.P1_E150_DEPLOY.glob("full_width_512_seed_*_test_prediction.npz")
    )
    checks = {
        "materializer_artifact_absent": not ARTIFACT.exists(),
        "delivery_absent": not DELIVERY.exists(),
        "output_absent": not OUTPUT.exists(),
        "official_test_exists": (official_source.P1_DATA / "test.csv").is_file(),
        "champion_exists": official_source.P1_CHAMPION.is_file(),
        "three_e150_seed_arrays_exist": len(seed_paths) == 3,
        "historical_pass_frozen": sha256(INTERNAL_RESULT)
        == EXPECTED_INTERNAL_RESULT_SHA256,
        "candidate_frozen": config["candidate"]
        == "P1_1_LABEL_FREE_RELIABILITY_GUARDED_LABEL_SHIFT_EM",
        "upload_not_implemented": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "internal_result_sha256": sha256(INTERNAL_RESULT),
            "materializer_sha256": sha256(Path(__file__)),
        },
        "official_values_read": 0,
        "hidden_truth_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }


def execute() -> dict:
    if ARTIFACT.exists() or DELIVERY.exists() or OUTPUT.exists():
        raise FileExistsError("v30 materializer exactly-once path already exists")
    config, internal_result = load_contract()
    ARTIFACT.mkdir(parents=True)
    started = time.perf_counter()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "config_sha256": sha256(CONFIG),
        "internal_result_sha256": sha256(INTERNAL_RESULT),
        "materializer_sha256": sha256(Path(__file__)),
        "official_reads_before_lock": 0,
        "hidden_truth_reads": 0,
        "uploads": 0,
    }
    lock_path = ARTIFACT / "attempt_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    historical, anchor, _, dependency = surface.load_feature_surface()
    truth = historical["label_base"].to_numpy(np.int8)
    source_probability = np.column_stack(
        [
            historical["probability_base"].to_numpy(np.float64),
            historical["probability_peer"].to_numpy(np.float64),
            historical["e150_probability"].to_numpy(np.float64),
        ]
    )
    design = v28.frozen_logit_matrix(
        source_probability[:, 0],
        source_probability[:, 1],
        source_probability[:, 2],
    )
    times_ns = (
        pd.to_datetime(historical["time"], utc=True)
        .astype("int64")
        .to_numpy(np.int64)
    )
    full_prefix = np.ones(len(historical), dtype=bool)
    split = v28.chronological_inner_split(
        times_ns,
        full_prefix,
        fit_fraction=float(config["model"]["fit_fraction"]),
    )
    fit_negative = split.fit_mask & (anchor == 0)
    calibration = split.calibration_mask
    calibration_negative = calibration & (anchor == 0)
    model = v28._calibrator(config)
    model.fit(design[fit_negative], truth[fit_negative])
    inner_negative_probability = model.predict_proba(design[calibration_negative])[:, 1]
    inner_probability = np.zeros(int(calibration.sum()), dtype=np.float64)
    inner_anchor = anchor[calibration]
    inner_probability[inner_anchor == 0] = inner_negative_probability
    source_prevalence = float(
        np.clip(truth[calibration_negative].mean(), 1e-6, 1.0 - 1e-6)
    )
    threshold = v28.select_inner_threshold(
        inner_probability,
        truth[calibration],
        inner_anchor,
        maximum_changed_fraction=float(config["safety"]["maximum_changed_fraction"]),
    )
    calibration_score = model.predict_proba(design[calibration])[:, 1]
    reliability_config = config["label_free_reliability"]
    group_receipts = fit_label_free_group_reliability(
        calibration_score,
        source_probability[calibration],
        historical.loc[calibration, "station"].astype(str).to_numpy(),
        historical.loc[calibration, "layer"].to_numpy(),
        minimum_group_rows=int(reliability_config["minimum_group_rows"]),
        one_sided_z=float(reliability_config["one_sided_z"]),
        global_absolute_discrepancy_quantile=float(
            reliability_config["global_absolute_discrepancy_quantile"]
        ),
    )

    historical_meta, _ = source_cycle.p1_frame()
    _, official_columns = official_source.add_causal_features(historical_meta)
    raw_test, official = official_source.official_frame(official_columns)
    official_source_probability = np.column_stack(
        [
            official["probability_base"].to_numpy(np.float64),
            official["probability_peer"].to_numpy(np.float64),
            official["e150_probability"].to_numpy(np.float64),
        ]
    )
    official_anchor = official["e150_prediction"].to_numpy(np.int8)
    official_design = v28.frozen_logit_matrix(
        official_source_probability[:, 0],
        official_source_probability[:, 1],
        official_source_probability[:, 2],
    )
    official_negative = official_anchor == 0
    official_source_score = model.predict_proba(official_design[official_negative])[:, 1]
    corrected, em = label_shift_em(
        official_source_score,
        source_prevalence,
        maximum_iterations=int(config["em"]["maximum_iterations"]),
        tolerance=float(config["em"]["tolerance"]),
        epsilon=float(config["em"]["epsilon"]),
    )
    if not em.converged:
        raise ContractError("official v30 label-shift EM did not converge")
    margin = reliability_margin_lower_bound(
        corrected,
        threshold.threshold,
        official_source_probability[official_negative],
        official.loc[official_negative, "station"].astype(str).to_numpy(),
        official.loc[official_negative, "layer"].to_numpy(),
        group_receipts,
        one_sided_z=float(reliability_config["one_sided_z"]),
    )
    full_proposed = np.zeros(len(official), dtype=bool)
    full_margin = np.full(len(official), -np.inf, dtype=np.float64)
    full_proposed[official_negative] = margin >= 0.0
    full_margin[official_negative] = margin
    local_day = (
        pd.to_datetime(raw_test["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    additions = apply_label_free_day_cap(
        full_proposed,
        full_margin,
        local_day,
        maximum_fraction=float(
            config["outer_day_guard"]["maximum_changed_fraction_per_day"]
        ),
    )
    label = np.maximum(official_anchor, additions.astype(np.int8)).astype(np.int8)
    submission = raw_test[P1_KEYS].copy()
    submission["label"] = label
    output_checks = validate_output_frame(submission, raw_test[P1_KEYS])
    removals = int(((label == 0) & (official_anchor == 1)).sum())
    if removals:
        raise ContractError("v30 materialization removed a frozen anchor")
    DELIVERY.mkdir(parents=True, exist_ok=False)
    submission.to_csv(OUTPUT, index=False, lineterminator="\n")
    maximum_day_share = float(
        pd.DataFrame({"day": local_day, "addition": additions})
        .groupby("day", observed=True)["addition"]
        .agg(["sum", "size"])
        .eval("sum / size")
        .max()
    )
    receipt = {
        "schema_version": "p1.v30.materialization.1",
        "experiment_id": EXPERIMENT_ID,
        "status": "MATERIALIZED_NOT_UPLOADED",
        "runtime_seconds": time.perf_counter() - started,
        "internal_pass": {
            "result_sha256": sha256(INTERNAL_RESULT),
            "delta_f1": internal_result["candidate"]["delta_f1"],
            "raw_expected_points_delta": internal_result["candidate"][
                "raw_expected_points_delta"
            ],
            "calibrated_expected_points_delta": internal_result["candidate"][
                "calibrated_conservative_expected_points_delta"
            ],
        },
        "deployment_fit": {
            "v30_calibrator_fits": 1,
            "official_source_base_peer_fits": 2,
            "fit_rows": int(fit_negative.sum()),
            "inner_rows": int(calibration.sum()),
            "inner_threshold": threshold.threshold,
            "inner_additions": threshold.additions,
            "source_prevalence": source_prevalence,
            "eligible_groups": sorted(
                key for key, value in group_receipts.items() if value.eligible
            ),
            "outer_official_labels_read": 0,
            "em_target_prevalence": em.target_prevalence,
            "em_iterations": em.iterations,
            "em_converged": em.converged,
        },
        "output": {
            "path": str(OUTPUT.resolve()),
            "rows": len(submission),
            "bytes": OUTPUT.stat().st_size,
            "sha256": sha256(OUTPUT),
            "positive_rows": int(label.sum()),
            "additions_vs_anchor": int((additions & (official_anchor == 0)).sum()),
            "anchor_removals": removals,
            "margin_eligible_before_day_cap": int(full_proposed.sum()),
            "maximum_kst_day_addition_fraction": maximum_day_share,
            "checks": output_checks,
        },
        "source_feature_dependency_receipt": dependency,
        "operations": {
            "historical_surface_reads": 2,
            "official_test_covariate_reads": 2,
            "official_champion_prediction_reads": 1,
            "official_e150_prediction_array_reads": 3,
            "hidden_truth_reads": 0,
            "submission_csv_created": 1,
            "uploads": 0,
        },
        "hashes": {
            "config_sha256": sha256(CONFIG),
            "internal_result_sha256": sha256(INTERNAL_RESULT),
            "materializer_sha256": sha256(Path(__file__)),
            "lock_sha256": sha256(lock_path),
            "output_sha256": sha256(OUTPUT),
        },
    }
    (ARTIFACT / "result.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    REPORT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        print(json.dumps(execute(), indent=2, sort_keys=True))
        return
    if not args.preflight:
        raise SystemExit("use --preflight or --execute")
    result = preflight()
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
