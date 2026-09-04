"""Exactly-once deployment materializer for frozen P1 v32g CatBoost union."""

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

import run_full_internal_submission_cycle_20260831_v2 as paths  # noqa: E402
import run_p1_ordered_catboost_eventday_20260831_v32a as v32a  # noqa: E402

EXPERIMENT_ID = "p1_e150_catboost_precision_union_deployment_20260901_v32g1"
DEPLOYMENT_CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT = ROOT / "artifacts" / EXPERIMENT_ID
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
REPORT = REPORT_DIR / "result.json"
P1_KEYS = ["station", "year", "layer", "time"]


class ContractError(RuntimeError):
    """Frozen v32g score-priority deployment contract violation."""


def sha256(path: Path) -> str:
    return v32a.sha256_file(path)


def load_contract() -> tuple[dict, dict, dict]:
    deployment = json.loads(DEPLOYMENT_CONFIG.read_text(encoding="utf-8"))
    frozen = deployment["frozen_candidate"]
    source_config_path = ROOT / frozen["source_config"]
    source_runner_path = ROOT / frozen["source_runner"]
    source_result_path = ROOT / frozen["source_result"]
    source_qa_path = ROOT / frozen["source_qa"]
    base_config_path = ROOT / frozen["base_model_config"]
    base_runner_path = ROOT / frozen["base_model_runner"]
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    result = json.loads(source_result_path.read_text(encoding="utf-8"))
    qa = json.loads(source_qa_path.read_text(encoding="utf-8"))
    policy = deployment["data_policy"]
    checks = {
        "identity": deployment["experiment_id"] == EXPERIMENT_ID,
        "source_config_hash": sha256(source_config_path)
        == frozen["source_config_sha256"],
        "source_runner_hash": sha256(source_runner_path)
        == frozen["source_runner_sha256"],
        "source_result_hash": sha256(source_result_path)
        == frozen["source_result_sha256"],
        "source_qa_hash": sha256(source_qa_path) == frozen["source_qa_sha256"],
        "base_config_hash": sha256(base_config_path)
        == frozen["base_model_config_sha256"],
        "base_runner_hash": sha256(base_runner_path)
        == frozen["base_model_runner_sha256"],
        "source_identity": result["experiment_id"]
        == frozen["source_experiment_id"],
        "source_terminal": result["status"] == frozen["required_source_status"],
        "threshold_exact": result["candidate"]["threshold"]
        == source_config["candidate"]["catboost_probability_threshold"]
        == base_config["model"]["probability_threshold"]
        == frozen["probability_threshold"],
        "internal_delta": np.isclose(
            result["pooled"]["delta_f1"], frozen["historical_delta_f1"]
        ),
        "q34_delta": np.isclose(
            result["q3_q4"]["delta_f1"], frozen["q3_q4_delta_f1"]
        ),
        "points_exact": np.isclose(
            result["public_score_translation"]["expected_points_center"],
            frozen["expected_points_center"],
        ),
        "historical_additions": result["candidate"]["additions"]
        == frozen["historical_additions"],
        "source_qa_pass": qa["status"] == "PASS_NO_GO_VERIFIED",
        "source_official_zero": all(
            value == 0 for value in result["official_access"].values()
        ),
        "source_upload_zero": qa["operations"]["uploads"] == 0,
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
        raise ContractError(f"v32g1 deployment contract mismatch: {checks}")
    return deployment, base_config, result


def load_features(path: Path, expected_hash: str, expected_rows: int) -> tuple[pd.DataFrame, list[int]]:
    if sha256(path) != expected_hash:
        raise ContractError(f"feature cache hash mismatch: {path}")
    frame = pd.read_parquet(path)
    if len(frame) != expected_rows or frame.shape[1] != 80:
        raise ContractError(f"feature cache shape mismatch: {path}")
    categorical = [
        frame.columns.get_loc(name)
        for name in ("station", "layer_category", "depth_regime")
    ]
    for position, column in enumerate(frame.columns):
        if position in categorical:
            frame[column] = frame[column].astype("string").fillna("<NA>").astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    return frame, categorical


def validate_output_frame(submission: pd.DataFrame, official_keys: pd.DataFrame) -> dict[str, bool]:
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
        raise ContractError(f"v32g1 official output contract failed: {checks}")
    return checks


def deployability_checks(
    label: np.ndarray,
    champion: np.ndarray,
    additions: np.ndarray,
    historical_positive_fraction: float,
    maximum_changed_fraction: float,
) -> dict[str, bool]:
    return {
        "binary_nonconstant": bool(np.unique(label).size == 2),
        "positive_additions": bool(np.any(additions & (champion == 0))),
        "anchor_removals_zero": bool(not np.any((label == 0) & (champion == 1))),
        "changed_fraction_within_frozen_limit": bool(
            np.mean(additions & (champion == 0)) <= maximum_changed_fraction
        ),
        "positive_fraction_within_historical_multiplier": bool(
            np.mean(label) <= 2.0 * historical_positive_fraction
        ),
    }


def output_paths(deployment: dict) -> tuple[Path, Path, Path]:
    delivery = Path(deployment["output"]["directory"])
    output = delivery / deployment["output"]["filename"]
    return delivery, output, delivery / "manifest.json"


def preflight() -> dict:
    deployment, _, _ = load_contract()
    immutable = deployment["immutable_inputs"]
    delivery, output, manifest = output_paths(deployment)
    checks = {
        "artifact_absent": not ARTIFACT.exists(),
        "report_absent": not REPORT.exists(),
        "delivery_absent": not delivery.exists(),
        "output_absent": not output.exists(),
        "manifest_absent": not manifest.exists(),
        "train_hash": sha256(paths.P1_DATA / "train.csv")
        == immutable["train_csv_sha256"],
        "test_hash": sha256(paths.P1_DATA / "test.csv")
        == immutable["test_csv_sha256"],
        "champion_hash": sha256(paths.P1_CHAMPION)
        == immutable["champion_csv_sha256"],
        "train_cache_hash": sha256(ROOT / immutable["train_feature_cache"])
        == immutable["train_feature_cache_sha256"],
        "test_cache_hash": sha256(ROOT / immutable["test_feature_cache"])
        == immutable["test_feature_cache_sha256"],
        "test_metadata_hash": sha256(ROOT / immutable["test_feature_metadata"])
        == immutable["test_feature_metadata_sha256"],
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
    deployment, base_config, internal_result = load_contract()
    immutable = deployment["immutable_inputs"]
    delivery, output, manifest = output_paths(deployment)
    if ARTIFACT.exists() or REPORT.exists() or delivery.exists() or output.exists():
        raise FileExistsError("v32g1 exactly-once path already exists")
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
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    train = v32a.load_train(paths.P1_DATA, immutable["train_csv_sha256"])
    train_features, categorical = load_features(
        ROOT / immutable["train_feature_cache"],
        immutable["train_feature_cache_sha256"],
        len(train),
    )
    test_features, test_categorical = load_features(
        ROOT / immutable["test_feature_cache"],
        immutable["test_feature_cache_sha256"],
        169_011,
    )
    if list(train_features.columns) != list(test_features.columns) or categorical != test_categorical:
        raise ContractError("train/test feature schema mismatch")
    truth = train["label"].to_numpy(np.int8)
    weights = v32a.event_day_weight(train, truth)
    model = v32a.model_from_config(base_config)
    model.fit(
        v32a.Pool(
            train_features,
            label=truth,
            weight=weights,
            cat_features=categorical,
        )
    )
    probability = model.predict_proba(
        v32a.Pool(test_features, cat_features=categorical)
    )[:, 1]
    if not np.isfinite(probability).all():
        raise ContractError("v32g1 official probabilities are nonfinite")

    raw_test = pd.read_csv(
        paths.P1_DATA / "test.csv", dtype={"station": "string", "time": "string"}
    )
    champion = pd.read_csv(
        paths.P1_CHAMPION,
        dtype={"station": "string", "time": "string", "label": "int8"},
    )
    if len(raw_test) != 169_011 or not champion[P1_KEYS].equals(raw_test[P1_KEYS]):
        raise ContractError("official champion key/order mismatch")
    champion_label = champion["label"].to_numpy(np.int8)
    threshold = float(deployment["frozen_candidate"]["probability_threshold"])
    independent_positive = probability >= threshold
    additions = (champion_label == 0) & independent_positive
    label = np.maximum(champion_label, additions.astype(np.int8)).astype(np.int8)
    submission = raw_test[P1_KEYS].copy()
    submission["label"] = label
    output_checks = validate_output_frame(submission, raw_test[P1_KEYS])
    guard = deployment["deployability_guard"]
    deploy_checks = deployability_checks(
        label,
        champion_label,
        additions,
        float(deployment["frozen_candidate"]["historical_candidate_positive_fraction"]),
        float(guard["maximum_changed_fraction"]),
    )

    delivery.mkdir(parents=True, exist_ok=False)
    submission.to_csv(output, index=False, lineterminator="\n")
    champion_hash = sha256(paths.P1_CHAMPION)
    output_hash = sha256(output)
    deploy_checks["output_hash_differs_from_current_champion"] = output_hash != champion_hash
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
    receipt = {
        "schema_version": "p1.v32g_score_priority_deployment.result.20260901.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "MATERIALIZED_READY_NOT_UPLOADED_EXPLICIT_INTERNAL_NO_GO"
            if deployable
            else "BLOCK_UPLOAD_DEPLOYABILITY_GATE"
        ),
        "runtime_seconds": time.perf_counter() - started,
        "fit_count": 1,
        "internal_evidence": {
            "status": internal_result["status"],
            "delta_f1": internal_result["pooled"]["delta_f1"],
            "q3_q4_delta_f1": internal_result["q3_q4"]["delta_f1"],
            "expected_points_center": internal_result["public_score_translation"]["expected_points_center"],
            "expected_points_ci90": internal_result["public_score_translation"]["expected_points_ci90"],
            "historical_additions": internal_result["candidate"]["additions"],
            "historical_true_positive_additions": internal_result["candidate"]["true_positive_additions"],
            "historical_false_positive_additions": internal_result["candidate"]["false_positive_additions"],
        },
        "deployment_fit": {
            "family": "CatBoostClassifier",
            "scratch_full_history_fit": 1,
            "fit_rows": len(train),
            "iterations": int(model.tree_count_),
            "threshold": threshold,
            "independent_positive_rows": int(independent_positive.sum()),
            "independent_positive_fraction": float(independent_positive.mean()),
            "official_labels_read": 0,
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
            "additions_vs_champion": int(additions.sum()),
            "anchor_removals": int(((label == 0) & (champion_label == 1)).sum()),
            "maximum_kst_day_addition_fraction_unlabeled": maximum_day_share,
            "checks": output_checks,
            "deployability_checks": deploy_checks,
        },
        "submission_metadata": {
            "title": deployment["output"]["title"],
            "summary": deployment["output"]["summary"],
        },
        "data_policy": deployment["data_policy"],
        "operations": {
            "organizer_train_rows_read": len(train),
            "official_test_covariate_rows_read": len(raw_test),
            "official_champion_prediction_rows_read": len(champion),
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
            "base_model_config_sha256": sha256(ROOT / deployment["frozen_candidate"]["base_model_config"]),
            "base_model_runner_sha256": sha256(ROOT / deployment["frozen_candidate"]["base_model_runner"]),
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
