"""Exactly-once organizer-data-only materializer for frozen P1 v31r1."""

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
import run_p1_public_transport_repair_cycle_20260831_v31r1 as v31r1  # noqa: E402

from src.p1_qc.logit_shrunk_label_shift import (  # noqa: E402
    correct_to_prior,
    shrink_lambda,
    shrunk_target_prevalence,
)

EXPERIMENT_ID = "p1_public_transport_repair_cycle_20260831_v31m1"
DEPLOYMENT_CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
REPORT = REPORT_DIR / "result.json"
P1_KEYS = ["station", "year", "layer", "time"]


class ContractError(RuntimeError):
    """Frozen v31 score-priority deployment contract violation."""


def sha256(path: Path) -> str:
    return v30m.sha256(path)


def load_contract() -> tuple[dict, dict, dict]:
    deployment = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    frozen = deployment["frozen_candidate"]
    base_path = ROOT / frozen["base_config"]
    auth_path = ROOT / frozen["authorization_config"]
    runner_path = ROOT / frozen["source_runner"]
    result_path = ROOT / frozen["source_result"]
    prediction_path = ROOT / frozen["source_prediction"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    policy = deployment["data_policy"]
    checks = {
        "identity": deployment["experiment_id"] == EXPERIMENT_ID,
        "base_hash": sha256(base_path) == frozen["base_config_sha256"],
        "authorization_hash": sha256(auth_path)
        == frozen["authorization_config_sha256"],
        "runner_hash": sha256(runner_path) == frozen["source_runner_sha256"],
        "result_hash": sha256(result_path) == frozen["source_result_sha256"],
        "prediction_hash": sha256(prediction_path)
        == frozen["source_prediction_sha256"],
        "source_identity": result["experiment_id"]
        == frozen["source_experiment_id"],
        "source_terminal": result["status"] == frozen["required_source_status"],
        "candidate_exact": result["candidate"]["name"]
        == base["candidate"]
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
        "source_qa_pass": result["independent_qa"]["status"] == "PASS",
        "source_official_zero": result["operations"]["official_reads"] == 0,
        "source_hidden_zero": result["operations"]["hidden_truth_reads"] == 0,
        "source_csv_zero": result["operations"]["submission_csv_created"] == 0,
        "source_upload_zero": result["operations"]["uploads"] == 0,
        "no_changes": all(
            frozen[key] == 0
            for key in (
                "model_parameter_changes",
                "threshold_rule_changes",
                "automatic_retries",
                "tuning",
            )
        ),
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
        raise ContractError(f"v31m1 deployment contract mismatch: {checks}")
    return deployment, base, result


def validate_output_frame(
    submission: pd.DataFrame, official_keys: pd.DataFrame
) -> dict[str, bool]:
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
        "label_finite": bool(
            np.isfinite(submission["label"].to_numpy(float)).all()
        ),
        "label_integer": pd.api.types.is_integer_dtype(submission["label"].dtype),
        "duplicate_rows_zero": not submission.duplicated().any(),
    }
    if not all(checks.values()):
        raise ContractError(f"v31m1 official output contract failed: {checks}")
    return checks


def deployability_checks(
    label: np.ndarray,
    champion: np.ndarray,
    additions: np.ndarray,
    historical_positive_fraction: float,
) -> dict[str, bool]:
    positive_fraction = float(np.mean(label))
    return {
        "binary_nonconstant": bool(np.unique(label).size == 2),
        "positive_additions": bool(np.any(additions & (champion == 0))),
        "anchor_removals_zero": bool(
            not np.any((label == 0) & (champion == 1))
        ),
        "positive_fraction_within_historical_multiplier": bool(
            positive_fraction <= 2.0 * historical_positive_fraction
        ),
    }


def output_paths(deployment: dict) -> tuple[Path, Path, Path]:
    delivery = Path(deployment["output"]["directory"])
    output = delivery / deployment["output"]["filename"]
    manifest = delivery / "manifest.json"
    return delivery, output, manifest


def preflight() -> dict:
    deployment, _, _ = load_contract()
    delivery, output, manifest = output_paths(deployment)
    immutable = deployment["immutable_inputs"]
    data_dir = v30m.official_source.P1_DATA
    champion = v30m.official_source.P1_CHAMPION
    checks = {
        "artifact_absent": not ARTIFACT.exists(),
        "report_absent": not REPORT.exists(),
        "delivery_absent": not delivery.exists(),
        "output_absent": not output.exists(),
        "manifest_absent": not manifest.exists(),
        "train_hash": sha256(data_dir / "train.csv")
        == immutable["train_csv_sha256"],
        "test_hash": sha256(data_dir / "test.csv")
        == immutable["test_csv_sha256"],
        "champion_hash": sha256(champion) == immutable["champion_csv_sha256"],
        "upload_not_implemented": deployment["data_policy"]["upload_allowed"]
        is False,
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
        raise FileExistsError("v31m1 exactly-once path already exists")
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
    times_ns = (
        pd.to_datetime(historical["time"], utc=True)
        .astype("int64")
        .to_numpy(np.int64)
    )
    split = v28.chronological_inner_split(
        times_ns,
        np.ones(len(historical), dtype=bool),
        fit_fraction=float(config["model"]["outer_prefix_fit_fraction"]),
    )
    shrink_mask, selection_mask, inner_cutoff = v31r1.split_remainder(
        times_ns, split.calibration_mask
    )
    fit_negative = split.fit_mask & (anchor == 0)
    shrink_negative = shrink_mask & (anchor == 0)
    selection_negative = selection_mask & (anchor == 0)
    model = v28._calibrator({"model": config["model"]})
    model.fit(design[fit_negative], truth[fit_negative])
    epsilon = float(config["em"]["epsilon"])
    source_prevalence = float(
        np.clip(truth[fit_negative].mean(), epsilon, 1.0 - epsilon)
    )
    shrink_source_probability = model.predict_proba(design[shrink_negative])[:, 1]
    _, shrink_em = v28.label_shift_em(
        shrink_source_probability,
        source_prevalence,
        maximum_iterations=int(config["em"]["maximum_iterations"]),
        tolerance=float(config["em"]["tolerance"]),
        epsilon=epsilon,
    )
    observed_prevalence = float(
        np.clip(truth[shrink_negative].mean(), epsilon, 1.0 - epsilon)
    )
    shrink = shrink_lambda(
        source_prevalence,
        shrink_em.target_prevalence,
        observed_prevalence,
        epsilon=epsilon,
    )
    selection_source_probability = model.predict_proba(
        design[selection_negative]
    )[:, 1]
    _, selection_em = v28.label_shift_em(
        selection_source_probability,
        source_prevalence,
        maximum_iterations=int(config["em"]["maximum_iterations"]),
        tolerance=float(config["em"]["tolerance"]),
        epsilon=epsilon,
    )
    selection_target = shrunk_target_prevalence(
        source_prevalence,
        selection_em.target_prevalence,
        shrink,
        epsilon=epsilon,
    )
    selection_corrected = correct_to_prior(
        selection_source_probability,
        source_prevalence,
        selection_target,
        epsilon=epsilon,
    )
    selection_probability = np.zeros(int(selection_mask.sum()), dtype=np.float64)
    selection_probability[anchor[selection_mask] == 0] = selection_corrected
    threshold = v28.select_inner_threshold(
        selection_probability,
        truth[selection_mask],
        anchor[selection_mask],
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
    _, official_em = v28.label_shift_em(
        official_source_score,
        source_prevalence,
        maximum_iterations=int(config["em"]["maximum_iterations"]),
        tolerance=float(config["em"]["tolerance"]),
        epsilon=epsilon,
    )
    official_target = shrunk_target_prevalence(
        source_prevalence,
        official_em.target_prevalence,
        shrink,
        epsilon=epsilon,
    )
    official_corrected = correct_to_prior(
        official_source_score,
        source_prevalence,
        official_target,
        epsilon=epsilon,
    )
    if not np.isfinite(official_corrected).all():
        raise ContractError("official v31m1 corrected probabilities are nonfinite")
    additions = np.zeros(len(official), dtype=bool)
    additions[official_negative] = official_corrected >= threshold.threshold
    label = np.maximum(official_anchor, additions.astype(np.int8)).astype(np.int8)
    submission = raw_test[P1_KEYS].copy()
    submission["label"] = label
    output_checks = validate_output_frame(submission, raw_test[P1_KEYS])
    deploy_checks = deployability_checks(
        label,
        official_anchor,
        additions,
        float(
            deployment["frozen_candidate"][
                "historical_candidate_positive_fraction"
            ]
        ),
    )

    delivery.mkdir(parents=True, exist_ok=False)
    submission.to_csv(output, index=False, lineterminator="\n")
    champion_hash = sha256(v30m.official_source.P1_CHAMPION)
    output_hash = sha256(output)
    deploy_checks["output_hash_differs_from_current_champion"] = (
        output_hash != champion_hash
    )
    deployable = all(deploy_checks.values())
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
    failed_hard_gates = sorted(
        name
        for name, passed in internal_result["candidate"]["hard_gates"].items()
        if not passed
    )
    receipt = {
        "schema_version": "p1.v31_score_priority_deployment.result.20260901.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "MATERIALIZED_READY_NOT_UPLOADED_EXPLICIT_STABILITY_RISK"
            if deployable
            else "BLOCK_UPLOAD_DEPLOYABILITY_GATE"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 3,
        "source_candidate_full_history_fit_count": 1,
        "official_feature_source_fit_count": 2,
        "internal_evidence": {
            "delta_f1": internal_result["candidate"]["delta_f1"],
            "raw_expected_points_delta": internal_result["candidate"][
                "raw_expected_points_delta"
            ],
            "transport_adjusted_expected_points_delta": internal_result[
                "candidate"
            ]["calibrated_conservative_expected_points_delta"],
            "failed_hard_gates": failed_hard_gates,
            "q4_delta_f1": internal_result["candidate"]["by_fold"]["2025_q4"][
                "delta_f1"
            ],
            "bootstrap_ci90": [
                internal_result["candidate"]["day_block_bootstrap"]["ci90_low"],
                internal_result["candidate"]["day_block_bootstrap"]["ci90_high"],
            ],
        },
        "deployment_fit": {
            "fit_rows": int(fit_negative.sum()),
            "shrink_rows": int(shrink_mask.sum()),
            "selection_rows": int(selection_mask.sum()),
            "inner_cutoff_ns": inner_cutoff,
            "source_prevalence": source_prevalence,
            "shrink_em_target_prevalence": shrink_em.target_prevalence,
            "observed_shrink_prevalence": observed_prevalence,
            "shrink_lambda": shrink,
            "selection_em_target_prevalence": selection_em.target_prevalence,
            "selection_shrunk_target_prevalence": selection_target,
            "inner_threshold": threshold.threshold,
            "inner_additions": threshold.additions,
            "outer_official_labels_read": 0,
            "official_em_target_prevalence": official_em.target_prevalence,
            "official_shrunk_target_prevalence": official_target,
            "coefficient_sha256": v28.stable_hash(
                model.coef_.astype(np.float64), model.intercept_.astype(np.float64)
            ),
        },
        "output": {
            "path": str(output.resolve()),
            "rows": len(submission),
            "bytes": output.stat().st_size,
            "sha256": output_hash,
            "current_champion_sha256": champion_hash,
            "positive_rows": int(label.sum()),
            "negative_rows": int((label == 0).sum()),
            "positive_fraction": float(label.mean()),
            "additions_vs_champion": int(
                (additions & (official_anchor == 0)).sum()
            ),
            "anchor_removals": int(
                ((label == 0) & (official_anchor == 1)).sum()
            ),
            "maximum_kst_day_addition_fraction_unlabeled": maximum_day_share,
            "checks": output_checks,
            "deployability_checks": deploy_checks,
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
            "source_base_config_sha256": sha256(
                ROOT / deployment["frozen_candidate"]["base_config"]
            ),
            "source_authorization_config_sha256": sha256(
                ROOT / deployment["frozen_candidate"]["authorization_config"]
            ),
            "source_runner_sha256": sha256(
                ROOT / deployment["frozen_candidate"]["source_runner"]
            ),
            "source_result_sha256": sha256(
                ROOT / deployment["frozen_candidate"]["source_result"]
            ),
            "source_prediction_sha256": sha256(
                ROOT / deployment["frozen_candidate"]["source_prediction"]
            ),
            "materializer_sha256": sha256(Path(__file__)),
            "lock_sha256": sha256(lock_path),
            "output_sha256": output_hash,
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
    if args.preflight == args.execute:
        raise SystemExit("use exactly one of --preflight or --execute")
    result = execute() if args.execute else preflight()
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("status") == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
