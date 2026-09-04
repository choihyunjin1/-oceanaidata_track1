"""Exactly-once organizer-data-only materializer for the frozen P1 v28 score probe."""

from __future__ import annotations

import argparse
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

import materialize_p1_public_transport_repair_cycle_20260831_v30 as v30m  # noqa: E402
import run_p1_public_transport_repair_cycle_20260831_v28 as v28  # noqa: E402

from src.p1_qc.prequential_label_shift_em import label_shift_em  # noqa: E402

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v28m1"
DEPLOYMENT_CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
REPORT = REPORT_DIR / "result.json"
P1_KEYS = ["station", "year", "layer", "time"]


class ContractError(RuntimeError):
    """Frozen v28 score-priority deployment contract violation."""


def sha256(path: Path) -> str:
    return v30m.sha256(path)


def load_contract() -> tuple[dict, dict, dict]:
    deployment = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    frozen = deployment["frozen_candidate"]
    source_config_path = ROOT / frozen["source_config"]
    source_runner_path = ROOT / frozen["source_runner"]
    source_result_path = ROOT / frozen["source_result"]
    source_qa_path = ROOT / frozen["source_qa"]
    source_config = v28.load_contract()
    result = json.loads(source_result_path.read_text(encoding="utf-8"))
    qa = json.loads(source_qa_path.read_text(encoding="utf-8"))
    gates = result["candidate"]["gates"]
    failed = sorted(name for name, passed in gates.items() if not passed)
    policy = deployment["data_policy"]
    checks = {
        "deployment_identity": deployment["experiment_id"] == EXPERIMENT_ID,
        "source_identity": frozen["source_experiment_id"] == v28.EXPERIMENT_ID,
        "source_config_hash": sha256(source_config_path)
        == frozen["source_config_sha256"],
        "source_runner_hash": sha256(source_runner_path)
        == frozen["source_runner_sha256"],
        "source_result_hash": sha256(source_result_path)
        == frozen["source_result_sha256"],
        "source_qa_hash": sha256(source_qa_path) == frozen["source_qa_sha256"],
        "source_terminal": result["status"] == frozen["required_source_status"],
        "candidate_exact": result["candidate"]["name"]
        == source_config["candidate"]
        == frozen["candidate_name"],
        "historical_fit_count": result["fit_count"]
        == frozen["historical_fit_count"]
        == 2,
        "historical_delta": np.isclose(
            result["candidate"]["delta_f1"], frozen["historical_delta_f1"]
        ),
        "historical_points": np.isclose(
            result["candidate"]["raw_expected_points_delta"],
            frozen["raw_expected_points_delta"],
        )
        and np.isclose(
            result["candidate"]["calibrated_conservative_expected_points_delta"],
            frozen["transport_adjusted_expected_points_delta"],
        ),
        "only_preregistered_safety_failures": failed
        == sorted(frozen["known_failed_safety_gates"]),
        "source_official_zero": result["operations"]["official_reads"] == 0,
        "source_hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "source_csv_zero": result["operations"]["submission_csv_created"] == 0,
        "source_upload_zero": result["operations"]["uploads"] == 0,
        "source_qa_pass": qa["status"] == "PASS"
        and qa["scientific_decision"] == "NO_GO_SAFETY_GATES",
        "no_tuning_retry": frozen["tuning"] == 0
        and frozen["automatic_retries"] == 0,
        "organizer_only": policy["organizer_distributed_data_only"] is True,
        "scratch_only": policy["scratch_training_only"] is True,
        "forbidden_sources_false": all(
            policy[key] is False
            for key in (
                "internet_data_allowed",
                "kiost_original_data_allowed",
                "external_observation_allowed",
                "external_reanalysis_allowed",
                "external_forecast_allowed",
                "real_observation_pretrained_weights_allowed",
                "hidden_truth_allowed",
                "official_aggregate_score_for_model_selection",
                "upload_allowed",
            )
        ),
    }
    if not all(checks.values()):
        raise ContractError(f"v28m1 deployment contract mismatch: {checks}")
    return deployment, source_config, result


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
        raise ContractError(f"v28m1 official output contract failed: {checks}")
    return checks


def output_paths(deployment: dict) -> tuple[Path, Path, Path]:
    delivery = Path(deployment["output"]["directory"])
    output = delivery / deployment["output"]["filename"]
    manifest = delivery / "manifest.json"
    return delivery, output, manifest


def preflight() -> dict:
    deployment, _, _ = load_contract()
    delivery, output, manifest = output_paths(deployment)
    seed_paths = sorted(
        v30m.official_source.P1_E150_DEPLOY.glob(
            "full_width_512_seed_*_test_prediction.npz"
        )
    )
    checks = {
        "artifact_absent": not ARTIFACT.exists(),
        "report_absent": not REPORT.exists(),
        "delivery_absent": not delivery.exists(),
        "output_absent": not output.exists(),
        "manifest_absent": not manifest.exists(),
        "official_test_exists": (v30m.official_source.P1_DATA / "test.csv").is_file(),
        "champion_exists": v30m.official_source.P1_CHAMPION.is_file(),
        "three_scratch_e150_arrays_exist": len(seed_paths) == 3,
        "upload_not_implemented": deployment["data_policy"]["upload_allowed"] is False,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "hashes": {
            "deployment_config_sha256": sha256(DEPLOYMENT_CONFIG),
            "materializer_sha256": sha256(Path(__file__)),
        },
        "official_values_read": 0,
        "hidden_truth_reads": 0,
        "external_rows_read": 0,
        "pretrained_weight_files_loaded": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }


def execute() -> dict:
    deployment, config, internal_result = load_contract()
    delivery, output, manifest = output_paths(deployment)
    if ARTIFACT.exists() or REPORT.exists() or delivery.exists() or output.exists():
        raise FileExistsError("v28m1 exactly-once path already exists")
    ARTIFACT.mkdir(parents=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    lock = {
        "experiment_id": EXPERIMENT_ID,
        "pid": os.getpid(),
        "deployment_config_sha256": sha256(DEPLOYMENT_CONFIG),
        "materializer_sha256": sha256(Path(__file__)),
        "official_reads_before_lock": 0,
        "hidden_truth_reads": 0,
        "external_rows_read": 0,
        "pretrained_weight_files_loaded": 0,
        "uploads": 0,
    }
    lock_path = ARTIFACT / "attempt_lock.json"
    lock_path.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    historical, anchor, _, dependency = v30m.surface.load_feature_surface()
    truth = historical["label_base"].to_numpy(np.int8)
    source_probability = np.column_stack(
        [
            historical["probability_base"].to_numpy(np.float64),
            historical["probability_peer"].to_numpy(np.float64),
            historical["e150_probability"].to_numpy(np.float64),
        ]
    )
    design = v28.frozen_logit_matrix(*source_probability.T)
    times_ns = pd.to_datetime(historical["time"], utc=True).astype("int64").to_numpy(np.int64)
    split = v28.chronological_inner_split(
        times_ns,
        np.ones(len(historical), dtype=bool),
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

    historical_meta, _ = v30m.source_cycle.p1_frame()
    _, official_columns = v30m.official_source.add_causal_features(historical_meta)
    raw_test, official = v30m.official_source.official_frame(official_columns)
    official_source_probability = np.column_stack(
        [
            official["probability_base"].to_numpy(np.float64),
            official["probability_peer"].to_numpy(np.float64),
            official["e150_probability"].to_numpy(np.float64),
        ]
    )
    official_anchor = official["e150_prediction"].to_numpy(np.int8)
    official_design = v28.frozen_logit_matrix(*official_source_probability.T)
    official_negative = official_anchor == 0
    official_source_score = model.predict_proba(official_design[official_negative])[:, 1]
    corrected, em = label_shift_em(
        official_source_score,
        source_prevalence,
        maximum_iterations=int(config["outer_label_shift"]["maximum_iterations"]),
        tolerance=float(config["outer_label_shift"]["tolerance"]),
        epsilon=float(config["outer_label_shift"]["epsilon"]),
    )
    if not em.converged:
        raise ContractError("official v28m1 label-shift EM did not converge")
    additions = np.zeros(len(official), dtype=bool)
    additions[official_negative] = corrected >= threshold.threshold
    label = np.maximum(official_anchor, additions.astype(np.int8)).astype(np.int8)
    submission = raw_test[P1_KEYS].copy()
    submission["label"] = label
    output_checks = validate_output_frame(submission, raw_test[P1_KEYS])
    removals = int(((label == 0) & (official_anchor == 1)).sum())
    if removals:
        raise ContractError("v28m1 materialization removed a frozen anchor")

    delivery.mkdir(parents=True, exist_ok=False)
    submission.to_csv(output, index=False, lineterminator="\n")
    local_day = (
        pd.to_datetime(raw_test["time"], utc=True)
        .dt.tz_convert("Asia/Seoul")
        .dt.strftime("%Y-%m-%d")
        .to_numpy()
    )
    maximum_day_share = float(
        pd.DataFrame({"day": local_day, "addition": additions})
        .groupby("day", observed=True)["addition"]
        .agg(["sum", "size"])
        .eval("sum / size")
        .max()
    )
    failed_gates = sorted(
        name
        for name, passed in internal_result["candidate"]["gates"].items()
        if not passed
    )
    receipt = {
        "schema_version": "p1.v28_score_priority_deployment.result.20260901.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "MATERIALIZED_READY_NOT_UPLOADED_EXPLICIT_STABILITY_RISK",
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 1,
        "source_model_fit_count": 2,
        "internal_evidence": {
            "delta_f1": internal_result["candidate"]["delta_f1"],
            "raw_expected_points_delta": internal_result["candidate"]["raw_expected_points_delta"],
            "transport_adjusted_expected_points_delta": internal_result["candidate"]["calibrated_conservative_expected_points_delta"],
            "failed_safety_gates": failed_gates,
            "historical_maximum_kst_day_changed_fraction": internal_result["candidate"]["maximum_kst_day_changed_fraction"],
            "negative_station_layer_count": sum(
                value < 0
                for value in internal_result["candidate"]["station_layer_delta_f1"].values()
            ),
        },
        "deployment_fit": {
            "calibrator_fits": 1,
            "official_source_base_peer_fits": 2,
            "fit_rows": int(fit_negative.sum()),
            "inner_rows": int(calibration.sum()),
            "inner_threshold": threshold.threshold,
            "inner_additions": threshold.additions,
            "source_prevalence": source_prevalence,
            "outer_official_labels_read": 0,
            "em_target_prevalence": em.target_prevalence,
            "em_iterations": em.iterations,
            "em_converged": em.converged,
        },
        "output": {
            "path": str(output.resolve()),
            "rows": len(submission),
            "bytes": output.stat().st_size,
            "sha256": sha256(output),
            "positive_rows": int(label.sum()),
            "additions_vs_anchor": int((additions & (official_anchor == 0)).sum()),
            "anchor_removals": removals,
            "maximum_kst_day_addition_fraction_unlabeled": maximum_day_share,
            "checks": output_checks,
        },
        "submission_metadata": {
            "title": deployment["output"]["title"],
            "summary": deployment["output"]["summary"],
        },
        "data_policy": deployment["data_policy"],
        "source_feature_dependency_receipt": dependency,
        "operations": {
            "historical_surface_reads": 2,
            "official_test_covariate_reads": 2,
            "official_champion_prediction_reads": 1,
            "official_e150_prediction_array_reads": 3,
            "official_aggregate_score_reads_for_selection": 0,
            "internet_rows_read": 0,
            "kiost_original_rows_read": 0,
            "external_observation_rows_read": 0,
            "external_reanalysis_rows_read": 0,
            "external_forecast_rows_read": 0,
            "real_observation_pretrained_weight_files_loaded": 0,
            "hidden_truth_reads": 0,
            "submission_csv_created": 1,
            "automatic_retries": 0,
            "tuning": 0,
            "uploads": 0,
        },
        "hashes": {
            "deployment_config_sha256": sha256(DEPLOYMENT_CONFIG),
            "source_config_sha256": sha256(ROOT / deployment["frozen_candidate"]["source_config"]),
            "source_runner_sha256": sha256(ROOT / deployment["frozen_candidate"]["source_runner"]),
            "source_result_sha256": sha256(ROOT / deployment["frozen_candidate"]["source_result"]),
            "source_qa_sha256": sha256(ROOT / deployment["frozen_candidate"]["source_qa"]),
            "materializer_sha256": sha256(Path(__file__)),
            "lock_sha256": sha256(lock_path),
            "output_sha256": sha256(output),
        },
    }
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    (ARTIFACT / "result.json").write_text(payload, encoding="utf-8")
    REPORT.write_text(payload, encoding="utf-8")
    manifest.write_text(payload, encoding="utf-8")
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
