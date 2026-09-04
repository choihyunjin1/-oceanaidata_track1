"""Run the append-only P1 Gen5 incumbent-distillation residual curve once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ocean_goal.meaningful_score_ledger_v5 import validate_ledger
from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p1_qc.causal_raw_features_v4r2 import (
    CAUSAL_FEATURE_COLUMNS,
    assert_future_value_invariance,
    build_causal_raw_features,
    build_exact_prefix_causal_matrix,
)
from p1_qc.features import build_features
from p1_qc.incumbent_residual_tcn import (
    ResidualModelConfig,
    ResidualTrainingConfig,
    build_three_block_inner_splits,
    exact_identity_or_residual,
    fit_incumbent_residual_model,
    load_fitted_incumbent_residual_model,
    predict_incumbent_residual_probability,
    save_fitted_incumbent_residual_model,
)
from p1_qc.pipeline import TabularEncoder, _fit_model

_R4_RUNNER = PROJECT_ROOT / "scripts/run_p1_masked_pretrain_binary_event_v4r4.py"
_R4_SPEC = importlib.util.spec_from_file_location("p1_gen5_r4_shared_adapter", _R4_RUNNER)
if _R4_SPEC is None or _R4_SPEC.loader is None:
    raise ImportError("failed to load pinned P1 Gen4r4 adapter")
r4 = importlib.util.module_from_spec(_R4_SPEC)
sys.modules[_R4_SPEC.name] = r4
_R4_SPEC.loader.exec_module(r4)
shared = r4.shared

EXPECTED_CONFIG_SHA256 = "da7427dcfa58daff7d9825653c34296aeb6c4d0648d0d2295715c5e8c0179396"
EXPECTED_CONFIG_DEEP_SHA256 = "e2c72aaa0fadf45e640f90989d3e1bc2630545cb5e85d9fc6257d7555e8280af"
CANONICAL_CONFIG = "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5.json"
CANONICAL_ARTIFACT = "artifacts/p1_incumbent_rule_distillation_neural_residual_v5"
CANONICAL_LOCK = "artifacts/p1_incumbent_rule_distillation_neural_residual_v5.ATTEMPT_LOCK.json"
HYPOTHESIS = "incumbent_rule_distillation_with_out_of_fold_neural_residual"
CANONICAL_ROOT_PATH = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
CANONICAL_DATA_DIR_PATH = (
    CANONICAL_ROOT_PATH / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"
)
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
_SHARED_JSON_NEW = r4._SHARED_JSON_NEW
EXECUTION_TOMBSTONE = (
    "artifacts/p1_incumbent_rule_distillation_neural_residual_v5/EXECUTION_TOMBSTONE.json"
)
EXECUTION_TOMBSTONE_SHA256 = (
    "00f990794a1ef1d9ebab9c1fc27a1d6d41c495944da34da9f4e2cda4e8bd7fc9"
)


def _enforce_execution_tombstone(root: Path) -> None:
    path = (root.resolve(strict=True) / EXECUTION_TOMBSTONE).resolve(strict=True)
    if shared._sha(path) != EXECUTION_TOMBSTONE_SHA256:
        raise PermissionError("invalidated Gen5 execution tombstone SHA differs")
    value = shared._json(path)
    if not (
        value.get("generation") == "p1_incumbent_rule_distillation_neural_residual_v5"
        and value.get("successor_generation")
        == "p1_incumbent_rule_distillation_neural_residual_v5r2"
        and value.get("execution_prohibited") is True
        and value.get("authorization_must_fail_before_attempt_lock") is True
        and value.get("attempt_lock_created") is False
        and value.get("curve_model_fits") == 0
        and value.get("test_value_reads") == 0
        and value.get("uploads") == 0
    ):
        raise PermissionError("invalidated Gen5 execution tombstone semantics differ")
    raise PermissionError(
        "p1_incumbent_rule_distillation_neural_residual_v5 is superseded and non-executable"
    )


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / CANONICAL_CONFIG,
        "design": root
        / "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5_design.json",
        "artifact": root / CANONICAL_ARTIFACT,
        "lock": root / CANONICAL_LOCK,
        "base_config": root / "configs/p1.toml",
        "goal": root / "configs/goals/meaningful_score_maximization_v3.json",
        "offline_feature_cache": root
        / "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "offline_feature_metadata": root
        / "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "feature_cache": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.parquet",
        "feature_metadata": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.json",
        "gen1": root / "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen4r4": root / "artifacts/p1_masked_pretrain_binary_event_v4r4",
        "frozen_oof": root / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "ledger": root / "artifacts/meaningful_score_goal_v5/registry.jsonl",
    }


def _assert_single_link(path: Path, *, role: str) -> None:
    if os.name == "nt" and os.stat(path).st_nlink != 1:
        raise PermissionError(f"hardlinked canonical identity anchor is forbidden: {role}")


def _assert_canonical_identity(root: Path, data_dir: Path) -> tuple[Path, Path]:
    lexical_root = Path(os.path.abspath(root))
    lexical_data = Path(os.path.abspath(data_dir))
    if lexical_root != CANONICAL_ROOT_PATH:
        raise PermissionError("workspace clone or non-canonical root is forbidden")
    if lexical_data != CANONICAL_DATA_DIR_PATH:
        raise PermissionError("non-canonical P1 data directory is forbidden")
    if root.resolve(strict=True) != CANONICAL_ROOT_PATH:
        raise PermissionError("canonical workspace resolves through an alias or reparse point")
    if data_dir.resolve(strict=True) != CANONICAL_DATA_DIR_PATH:
        raise PermissionError("canonical data directory resolves through an alias or reparse point")
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    anchors = {
        "workspace": CANONICAL_ROOT_PATH,
        "data_dir": CANONICAL_DATA_DIR_PATH,
        "config": CANONICAL_ROOT_PATH / CANONICAL_CONFIG,
        "runner": Path(__file__).resolve(strict=True),
        "train": CANONICAL_DATA_DIR_PATH / "train.csv",
        "test": CANONICAL_DATA_DIR_PATH / "test.csv",
        "sample_submission": CANONICAL_DATA_DIR_PATH / "sample_submission.csv",
    }
    for role, path in anchors.items():
        if getattr(os.lstat(path), "st_file_attributes", 0) & reparse:
            raise PermissionError(f"reparse-point canonical identity anchor is forbidden: {role}")
        _assert_single_link(path, role=role)
    return CANONICAL_ROOT_PATH, CANONICAL_DATA_DIR_PATH


def _verify_v5_ledger_binding(
    root: Path, config: dict[str, Any], ledger_path: Path
) -> dict[str, Any]:
    binding = config["v5_ledger_binding"]
    observed = {
        "path": ledger_path.relative_to(root).as_posix(),
        "sha256": shared._sha(ledger_path),
        "bytes": int(ledger_path.stat().st_size),
    }
    if observed != {key: binding[key] for key in ("path", "sha256", "bytes")}:
        raise PermissionError("canonical v5 ledger path, SHA, or byte count differs")
    records = validate_ledger(root, ledger_path)
    if (
        len(records) != binding["event_count"]
        or records[-1]["seq"] != binding["head_seq"]
        or records[-1]["event_sha256"] != binding["head_event_sha256"]
    ):
        raise PermissionError("canonical v5 ledger latest head differs")
    uploads = sum(record["payload"].get("upload_performed") is True for record in records)
    if (
        binding["all_event_upload_performed_false"] is not True
        or binding["semantic_upload_count"] != 0
        or uploads != 0
        or not all(record["payload"].get("upload_performed") is False for record in records)
    ):
        raise PermissionError("canonical v5 ledger upload semantics differ from zero")
    return {**observed, "event_count": len(records), "head_seq": records[-1]["seq"]}


def _verify_failed_generation(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    result = shared._json(paths["gen4r4"] / "result.json")
    evidence_path = paths["gen4r4"] / "learning_curve_evidence.json"
    manifest_path = paths["gen4r4"] / "manifest.json"
    if not (
        result["experiment_id"] == "p1_masked_pretrain_binary_event_v4r4"
        and result["status"] == "RESEARCH_ONLY_NO_PASS"
        and result["passed"] is False
        and result["exactly_one_next_structural_diagnosis"] == HYPOTHESIS
        and shared._sha(evidence_path) == config["diagnosis_binding"]["failed_evidence_sha256"]
        and shared._sha(manifest_path)
        == config["immutable_inputs"][
            "artifacts/p1_masked_pretrain_binary_event_v4r4/manifest.json"
        ]
    ):
        raise PermissionError("failed Gen4r4 diagnosis binding differs")
    return {
        "generation": result["experiment_id"],
        "status": result["status"],
        "passed": False,
        "next_structural_diagnosis": HYPOTHESIS,
        "evidence_sha256": shared._sha(evidence_path),
    }


def _model_config(config: dict[str, Any], feature_count: int, group_count: int) -> ResidualModelConfig:
    model = config["model"]
    result = ResidualModelConfig(
        input_feature_count=feature_count,
        group_count=group_count,
        width=int(model["width"]),
        group_embedding_width=int(model["group_embedding_width"]),
        dilations=tuple(int(value) for value in model["dilations"]),
        kernel_size=int(model["kernel_size"]),
        dropout=float(model["dropout"]),
        norm_groups=int(model["norm_groups"]),
        maximum_absolute_logit_correction=float(model["maximum_absolute_logit_correction"]),
    )
    result.validate()
    if result.receptive_field_rows != int(model["receptive_field_rows"]):
        raise ValueError("registered Gen5 receptive field differs")
    return result


def _training_config(config: dict[str, Any]) -> ResidualTrainingConfig:
    training = config["training"]
    weights = training["loss_weights"]
    result = ResidualTrainingConfig(
        optimizer_steps=int(training["optimizer_steps_per_residual_fit"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        main_loss_weight=float(weights["main"]),
        distillation_loss_weight=float(weights["distillation"]),
        identity_regularizer_weight=float(weights["identity_regularizer"]),
    )
    result.validate()
    return result


def authorize_entry(
    *, root: Path, data_dir: Path, requested_config: Path, requested_artifact: Path
) -> tuple[dict[str, Any], dict[str, Path], dict[str, dict[str, Any]]]:
    _enforce_execution_tombstone(root)
    root, data_dir = _assert_canonical_identity(root, data_dir)
    paths = _paths(root)
    if requested_config.resolve(strict=True) != paths["config"].resolve(strict=True):
        raise PermissionError("non-canonical config path is forbidden")
    if requested_artifact.resolve(strict=False) != paths["artifact"].resolve(strict=False):
        raise PermissionError("non-canonical artifact path is forbidden")
    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical config byte SHA differs")
    config = json.loads(content)
    if shared._deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical config deep JSON differs")
    if config["experiment_id"] != "p1_incumbent_rule_distillation_neural_residual_v5":
        raise PermissionError("experiment identity differs")
    if config.get("comparison_mode") != "EXACT_OFFICIAL_PREFIX_REFIT":
        raise PermissionError("P1 Gen5 comparison mode must remain exact")
    expected_paths = {
        "config": CANONICAL_CONFIG,
        "design": "configs/experiments/p1_incumbent_rule_distillation_neural_residual_v5_design.json",
        "base_config": "configs/p1.toml",
        "goal_contract": "configs/goals/meaningful_score_maximization_v3.json",
        "offline_feature_cache": "artifacts/cache/train_offline_e9fe1eb46cb7431f.parquet",
        "offline_feature_metadata": "artifacts/cache/train_offline_e9fe1eb46cb7431f.json",
        "causal_feature_cache": "artifacts/cache/train_causal_raw_prefix_safe_v4r2.parquet",
        "causal_feature_metadata": "artifacts/cache/train_causal_raw_prefix_safe_v4r2.json",
        "gen1_artifact": "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen4r4_artifact": "artifacts/p1_masked_pretrain_binary_event_v4r4",
        "frozen_oof": "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "v5_ledger": "artifacts/meaningful_score_goal_v5/registry.jsonl",
        "artifact": CANONICAL_ARTIFACT,
        "attempt_lock": CANONICAL_LOCK,
    }
    if config["canonical_paths"] != expected_paths:
        raise PermissionError("canonical path contract differs")
    if [item["id"] for item in config["hypotheses"]] != [HYPOTHESIS]:
        raise PermissionError("single registered hypothesis differs")
    if tuple(config["prefix_fractions"]) != FRACTIONS or tuple(config["seeds"]) != SEEDS:
        raise PermissionError("prefix or seed contract differs")
    inner = config["inner_cross_fit"]
    training = config["training"]
    gate = config["train_only_no_op_gate"]
    if not (
        inner["split_count"] == 3
        and inner["timestamp_boundaries"] == [0.25, 0.5, 0.75, 1.0]
        and inner["purge_days"] == 7
        and inner["cutoff_target_reads"] == 0
        and inner["teacher_fits_per_outer_cell"] == 9
        and inner["expected_curve_teacher_fits"] == 135
        and training["optimizer_steps_per_residual_fit"] == 120
        and training["residual_gate_fits"] == 45
        and training["residual_refits"] == 45
        and training["expected_curve_fit_cells"] == 225
        and training["expected_curve_optimizer_steps"] == 10800
        and gate["fit_blocks"] == [1, 2]
        and gate["held_out_gate_block"] == 3
        and gate["failed_gate_probability_identity"] == "np.array_equal"
        and gate["missing_required_station_in_gate_block"]
        == "NO_OP_EXACT_INCUMBENT"
        and gate["missing_binary_class_in_gate_block"] == "NO_OP_EXACT_INCUMBENT"
        and gate["outer_validation_metrics_available_to_gate"] is False
        and gate["full_fit_identity_decision_reference"]
        == "2025_q4_fixed_postprocess"
        and gate["full_fit_gate_application_rule"]
        == "residual_must_pass_under_all_three_fixed_fold_postprocess_mappings"
    ):
        raise PermissionError("Gen5 cross-fit, fit-count, or no-op contract differs")
    if tuple(config["features"]["selected_numeric_columns"]) != CAUSAL_FEATURE_COLUMNS:
        raise PermissionError("Gen5 causal residual feature allowlist differs")
    if not all(value is True for value in config["prohibitions"].values()):
        raise PermissionError("all Gen5 prohibitions must remain enabled")
    implementations = {
        "residual_module": root / "src/p1_qc/incumbent_residual_tcn.py",
        "causal_feature_module": root / "src/p1_qc/causal_raw_features_v4r2.py",
        "feature_builder": root / "src/p1_qc/features.py",
        "pipeline": root / "src/p1_qc/pipeline.py",
        "temporal_layout": root / "src/p1_qc/temporal_event_tcn.py",
        "gen4r4_runner": _R4_RUNNER,
        "shared_gen3_adapter": root / "scripts/run_p1_binary_event_tcn_dense_natural_v3.py",
        "shared_gen2_runner": root
        / "scripts/run_p1_station_layer_temporal_convolution_event_v2.py",
        "gen1_runner": root / "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
        "base_config": paths["base_config"],
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score_v3.py",
        "v5_ledger_contract": root / "configs/goals/meaningful_score_ledger_v5.json",
        "v5_ledger_evaluator": root / "src/ocean_goal/meaningful_score_ledger_v5.py",
        "module_test": root / "tests/test_p1_incumbent_residual_tcn.py",
        "runner_test": root
        / "tests/test_run_p1_incumbent_rule_distillation_neural_residual_v5.py",
    }
    for name, path in implementations.items():
        if shared._sha(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    design = shared._json(paths["design"])
    if not (
        design["status"] == "STATIC_DESIGN_ONLY_NOT_PREREGISTERED_NOT_EXECUTABLE"
        and design["v5_ledger_binding"] == config["v5_ledger_binding"]
        and design["single_hypothesis"]["id"] == HYPOTHESIS
        and design["static_design_counters"]["model_fits"] == 0
        and design["static_design_counters"]["attempt_locks"] == 0
        and design["static_design_counters"]["uploads"] == 0
    ):
        raise PermissionError("append-only Gen5 static design binding differs")
    pins = shared._verify_input_pins(root, data_dir, config)
    _verify_v5_ledger_binding(root, config, paths["ledger"])
    _verify_failed_generation(config, paths)
    metadata = shared._json(paths["feature_metadata"])
    if not (
        metadata["feature_columns"] == list(CAUSAL_FEATURE_COLUMNS)
        and metadata["future_value_perturbation_invariant"] is True
        and metadata["target_columns_read"] == 0
    ):
        raise PermissionError("causal feature-cache audit differs")
    replayed = shared._verify_input_pins(root, data_dir, config)
    if pins != replayed or set(pins) != set(config["immutable_inputs"]):
        raise PermissionError("Gen5 start/end immutable-pin surface differs")
    return config, paths, pins


def _json_new(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name == "preregistration.json":
        config = shared._json(PROJECT_ROOT / CANONICAL_CONFIG)
        value = {
            **value,
            "static_design": {
                "path": config["canonical_paths"]["design"],
                "sha256": config["immutable_inputs"][config["canonical_paths"]["design"]],
            },
            "canonical_v5_ledger_binding": config["v5_ledger_binding"],
            "failed_generation_binding": config["diagnosis_binding"],
            "historical_prefix_cutoff_provenance": {
                "literal_disclosure": "Four sealed adjusted cutoff timestamps were historically produced by Gen1 reading training labels to retreat complete positive-event boundaries.",
                "current_run_prefix_selector_target_reads": 0,
            },
            "fit_budget": {
                "curve_teacher_fits": 135,
                "curve_residual_gate_fits": 45,
                "curve_residual_refits": 45,
                "curve_total_fits": 225,
                "curve_total_residual_optimizer_steps": 10800,
                "maximum_full_fit_fits_on_pass": 18,
            },
            "canonical_identity_verified_before_seal": True,
            "outer_validation_target_reads_at_seal": 0,
        }
    elif isinstance(value, dict) and path.name == "learning_curve_evidence.json":
        value = {
            **value,
            "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT",
            "leakage_checks": {
                **value["leakage_checks"],
                "inner_split_cutoffs_use_timestamps_only": True,
                "inner_teacher_fit_and_prediction_rows_disjoint": True,
                "inner_teacher_seven_day_purge_exact": True,
                "teacher_raw_rows_restricted_to_exact_outer_prefix": True,
                "residual_raw_context_strictly_causal_future_half_zero_masked": True,
                "gate_predictions_sealed_before_gate_block_labels_read": True,
                "outer_validation_targets_unread_until_all_outer_predictions_sealed": True,
                "failed_gate_returns_exact_incumbent_seed_probability_and_prediction": True,
                "gate_blocks_missing_a_required_station_fail_closed_to_exact_incumbent": True,
                "gate_blocks_missing_a_binary_class_fail_closed_to_exact_incumbent": True,
                "canonical_v5_seq9_head_and_zero_upload_semantics_verified": True,
            },
            "reproducibility_checks": {
                **value["reproducibility_checks"],
                "fixed_135_teacher_45_gate_45_refit_curve_fits": True,
                "fixed_10800_total_residual_optimizer_steps": True,
                "all_teacher_and_residual_saved_models_reload_exact": True,
                "all_gate_decisions_use_preregistered_train_only_rule": True,
                "canonical_v5_seq9_binding_replayed": True,
            },
        }
    elif isinstance(value, dict) and path.name == "result.json":
        value = {
            **value,
            "operation_counters": {
                **value["operation_counters"],
                "curve_teacher_model_fits": 135,
                "curve_residual_gate_model_fits": 45,
                "curve_residual_refit_model_fits": 45,
                "curve_total_model_fits": 225,
                "curve_gate_optimizer_steps": 5400,
                "curve_refit_optimizer_steps": 5400,
                "curve_total_residual_optimizer_steps": 10800,
                "test_value_reads": 0,
                "uploads": 0,
            },
        }
    _SHARED_JSON_NEW(path, value)


def _binary_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    tp = int(np.count_nonzero((truth == 1) & (prediction == 1)))
    fp = int(np.count_nonzero((truth == 0) & (prediction == 1)))
    fn = int(np.count_nonzero((truth == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else float(2 * tp / denominator)


def _postprocess_ids(
    train: Any,
    ids: np.ndarray,
    probability: np.ndarray,
    postprocess: dict[str, Any],
) -> np.ndarray:
    frame = train.iloc[ids][["station", "year", "layer", "time", "temp", "psal", "depth"]].copy()
    plateau = shared.gen1.detect_plateaus(frame).to_numpy(bool)
    spike = shared.gen1.detect_singleton_spikes(frame).to_numpy(bool)
    return shared.apply_postprocess(frame, probability, plateau, spike, postprocess).astype(
        np.int8, copy=False
    )


def _save_joblib_new(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        joblib.dump(value, handle, compress=3)
    return shared._sha(path)


def _teacher_oof(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: Any,
    p1_config: Any,
    outer_prefix_ids: np.ndarray,
    outer_forbidden_ids: np.ndarray | None,
    fold_name: str,
    fold_ordinal: int,
    fraction_tag: str,
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits = build_three_block_inner_splits(
        train.loc[:, ["time"]],
        outer_prefix_ids,
        purge_days=int(config["inner_cross_fit"]["purge_days"]),
    )
    prefix_frame = train.iloc[outer_prefix_ids][
        ["station", "year", "layer", "time", "temp", "psal", "depth"]
    ].reset_index(drop=True)
    bundle = build_features(
        prefix_frame,
        config=p1_config,
        mode="offline",
        cadence_minutes=p1_config.data.cadence_minutes,
        group_columns=p1_config.data.group_columns,
    )
    global_to_local = np.full(len(train), -1, dtype=np.int64)
    global_to_local[outer_prefix_ids] = np.arange(len(outer_prefix_ids), dtype=np.int64)
    n_rows = len(train)
    seed_probability = {seed: np.full(n_rows, 0.5, dtype=np.float32) for seed in SEEDS}
    seed_decision = {seed: np.zeros(n_rows, dtype=np.int8) for seed in SEEDS}
    receipts: list[dict[str, Any]] = []
    exact_parameters = dict(p1_config.raw["models"]["xgboost"])
    threads = int(p1_config.raw["project"]["threads"])
    for split in splits:
        train_ids = split.teacher_train_ids
        prediction_ids = split.teacher_prediction_ids
        if outer_forbidden_ids is not None and (
            np.intersect1d(train_ids, outer_forbidden_ids).size
            or np.intersect1d(prediction_ids, outer_forbidden_ids).size
        ):
            raise PermissionError("inner teacher touched forbidden outer validation rows")
        local_train = global_to_local[train_ids]
        local_prediction = global_to_local[prediction_ids]
        if (local_train < 0).any() or (local_prediction < 0).any():
            raise PermissionError("inner teacher IDs escape exact outer prefix")
        labels = shared.pd.to_numeric(
            train.iloc[train_ids]["label"], errors="raise"
        ).to_numpy(np.int8)
        if len(np.unique(labels)) != 2:
            raise ValueError("inner teacher train block must contain both labels")
        encoder = TabularEncoder().fit(bundle, local_train)
        train_features = encoder.transform(bundle, local_train)
        prediction_features = encoder.transform(bundle, local_prediction)
        for seed in SEEDS:
            started = time.perf_counter()
            model = _fit_model(
                "xgboost",
                exact_parameters,
                int(seed) + int(fold_ordinal),
                threads,
                train_features,
                labels,
            )
            probability = model.predict_proba(prediction_features)[:, 1].astype(np.float32)
            seed_probability[seed][prediction_ids] = probability
            seed_decision[seed][prediction_ids] = _postprocess_ids(
                train,
                prediction_ids,
                probability,
                config["fixed_fold_postprocess"][fold_name],
            )
            relative = (
                f"teacher_models/{scope}/{fraction_tag}/{fold_name}/"
                f"block_{split.block}/seed_{seed}.joblib"
            )
            model_path = shared._safe_path(paths["artifact"], relative)
            model_sha = _save_joblib_new(model_path, {"encoder": encoder, "model": model})
            loaded = joblib.load(model_path)
            reproduced = loaded["model"].predict_proba(
                loaded["encoder"].transform(bundle, local_prediction)
            )[:, 1].astype(np.float32)
            reload_exact = bool(np.array_equal(probability, reproduced))
            if not reload_exact:
                raise RuntimeError("saved inner teacher did not reproduce OOF probability")
            blind_relative = (
                f"teacher_blind_predictions/{scope}/{fraction_tag}/{fold_name}/"
                f"block_{split.block}/seed_{seed}.npy"
            )
            blind_path = shared._safe_path(paths["artifact"], blind_relative)
            blind_sha = shared._npy_new(blind_path, probability)
            receipts.append(
                {
                    "role": "inner_teacher",
                    "scope": scope,
                    "fraction_tag": fraction_tag,
                    "fold": fold_name,
                    "block": split.block,
                    "seed": seed,
                    "train_rows": int(len(train_ids)),
                    "prediction_rows": int(len(prediction_ids)),
                    "train_ids_sha256": split.train_ids_sha256,
                    "prediction_ids_sha256": split.prediction_ids_sha256,
                    "purge_days": split.purge_days,
                    "train_end_utc": split.train_end_utc,
                    "prediction_start_utc": split.prediction_start_utc,
                    "teacher_fit_and_prediction_rows_disjoint": True,
                    "outer_validation_rows_touched": 0,
                    "raw_rows_outside_outer_prefix_read_for_features": 0,
                    "model_relative_path": relative,
                    "model_sha256": model_sha,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "saved_model_reload_prediction_exact": reload_exact,
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "test_value_reads": 0,
                }
            )
            shared._emit(
                "gen5_teacher_fit_complete",
                scope=scope,
                fraction_tag=fraction_tag,
                fold=fold_name,
                block=split.block,
                seed=seed,
                completed_in_cell=len(receipts),
                total_in_cell=9,
                elapsed_seconds=receipts[-1]["elapsed_seconds"],
            )
    oof_ids = np.concatenate([split.teacher_prediction_ids for split in splits])
    if len(np.unique(oof_ids)) != len(oof_ids):
        raise AssertionError("inner teacher OOF blocks overlap")
    matrix = np.column_stack([seed_probability[seed] for seed in SEEDS])
    mean_probability = matrix.mean(axis=1).astype(np.float32)
    std_probability = matrix.std(axis=1).astype(np.float32)
    return (
        {
            "splits": splits,
            "oof_ids": oof_ids,
            "seed_probability": seed_probability,
            "seed_decision": seed_decision,
            "mean_probability": mean_probability,
            "std_probability": std_probability,
        },
        receipts,
    )


def _gate_decision(
    *,
    train: Any,
    gate_ids: np.ndarray,
    truth: np.ndarray,
    base_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    micro_delta = _binary_f1(truth, residual_prediction) - _binary_f1(truth, base_prediction)
    stations = train.iloc[gate_ids]["station"].astype(str).to_numpy()
    station_deltas: dict[str, float] = {}
    missing_stations: list[str] = []
    for station in STATIONS:
        mask = stations == station
        if not mask.any():
            missing_stations.append(station)
            continue
        station_deltas[station] = _binary_f1(
            truth[mask], residual_prediction[mask]
        ) - _binary_f1(truth[mask], base_prediction[mask])
    thresholds = config["train_only_no_op_gate"]["apply_residual_if_all"]
    improved = sum(value > 0.0 for value in station_deltas.values())
    missing_binary_class = len(np.unique(truth)) != 2
    passed = bool(
        not missing_stations
        and not missing_binary_class
        and micro_delta >= float(thresholds["micro_f1_delta_at_least"])
        and improved >= int(thresholds["improved_station_count_at_least"])
        and min(station_deltas.values())
        >= float(thresholds["worst_station_f1_delta_at_least"])
    )
    return {
        "passed": passed,
        "micro_f1_delta": float(micro_delta),
        "station_deltas": station_deltas,
        "missing_required_stations": missing_stations,
        "missing_required_station_fail_closed": bool(missing_stations),
        "missing_binary_class_fail_closed": bool(missing_binary_class),
        "improved_station_count": int(improved),
        "thresholds": thresholds,
    }


def _run_curve(
    *,
    root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: Any,
    features: np.ndarray,
    feature_columns: list[str],
    layout: Any,
    folds: list[dict[str, Any]],
    prefix_ids: dict[tuple[str, float], np.ndarray],
    comparator_parts: dict[tuple[str, float], Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p1_config = shared.load_config(paths["base_config"])
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    all_receipts: list[dict[str, Any]] = []
    primary_receipts: list[dict[str, Any]] = []
    teacher_receipts: list[dict[str, Any]] = []
    gate_receipts: list[dict[str, Any]] = []
    part_receipts: list[dict[str, Any]] = []
    completed_primary = 0
    for fraction in FRACTIONS:
        fraction_tag = shared._tag(fraction)
        for fold in folds:
            fold_name = str(fold["name"])
            train_ids = prefix_ids[(fold_name, fraction)]
            validation_ids = np.asarray(fold["val_idx"], dtype=np.int64)
            comparator = shared._comparator_frame(
                comparator_parts[(fold_name, fraction)], fold, fraction
            )
            prefix_features, prefix_feature_sha = build_exact_prefix_causal_matrix(
                train,
                train_ids,
                full_reference=features,
            )
            teacher, cell_teacher_receipts = _teacher_oof(
                config=config,
                paths=paths,
                train=train,
                p1_config=p1_config,
                outer_prefix_ids=train_ids,
                outer_forbidden_ids=validation_ids,
                fold_name=fold_name,
                fold_ordinal=int(fold["ordinal"]),
                fraction_tag=fraction_tag,
                scope="curve",
            )
            teacher_receipts.extend(cell_teacher_receipts)
            all_receipts.extend(cell_teacher_receipts)
            splits = teacher["splits"]
            residual_fit_ids = np.concatenate(
                [splits[0].teacher_prediction_ids, splits[1].teacher_prediction_ids]
            )
            gate_ids = splits[2].teacher_prediction_ids
            refit_ids = teacher["oof_ids"]
            outer_context = np.unique(np.concatenate([train_ids, validation_ids]))
            prediction_features = prefix_features.copy()
            prediction_features[validation_ids] = features[validation_ids]
            seed_final_probabilities: list[np.ndarray] = []
            seed_final_predictions: list[np.ndarray] = []
            cell_gate_passes: list[bool] = []
            for seed in SEEDS:
                gate_started = time.perf_counter()
                gate_forbidden = np.unique(np.concatenate([validation_ids, gate_ids]))
                gate_labels = shared.pd.to_numeric(
                    train.iloc[residual_fit_ids]["label"], errors="raise"
                ).to_numpy(np.int8)
                gate_model = fit_incumbent_residual_model(
                    prefix_features,
                    layout,
                    residual_fit_ids,
                    gate_labels,
                    teacher["seed_probability"][seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][seed],
                    context_ids=train_ids,
                    forbidden_ids=gate_forbidden,
                    seed=int(seed) + int(fold["ordinal"]),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                )
                gate_probability = predict_incumbent_residual_probability(
                    gate_model,
                    prefix_features,
                    layout,
                    gate_ids,
                    teacher["seed_probability"][seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][seed],
                    context_ids=train_ids,
                    device="cuda",
                )
                gate_model_relative = (
                    f"gate_models/{fraction_tag}/{fold_name}/seed_{seed}.pt"
                )
                gate_model_path = shared._safe_path(paths["artifact"], gate_model_relative)
                save_fitted_incumbent_residual_model(gate_model, gate_model_path)
                loaded_gate = load_fitted_incumbent_residual_model(gate_model_path)
                reproduced_gate = predict_incumbent_residual_probability(
                    loaded_gate,
                    prefix_features,
                    layout,
                    gate_ids,
                    teacher["seed_probability"][seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][seed],
                    context_ids=train_ids,
                    device="cuda",
                )
                if not np.array_equal(gate_probability, reproduced_gate):
                    raise RuntimeError("saved gate residual did not reproduce probability")
                gate_blind_relative = (
                    f"gate_blind_predictions/{fraction_tag}/{fold_name}/seed_{seed}.npy"
                )
                gate_blind_path = shared._safe_path(paths["artifact"], gate_blind_relative)
                gate_blind_sha = shared._npy_new(gate_blind_path, gate_probability)
                gate_residual_prediction = _postprocess_ids(
                    train,
                    gate_ids,
                    gate_probability,
                    config["fixed_fold_postprocess"][fold_name],
                )
                gate_truth = shared.pd.to_numeric(
                    train.iloc[gate_ids]["label"], errors="raise"
                ).to_numpy(np.int8)
                gate_result = _gate_decision(
                    train=train,
                    gate_ids=gate_ids,
                    truth=gate_truth,
                    base_prediction=teacher["seed_decision"][seed][gate_ids],
                    residual_prediction=gate_residual_prediction,
                    config=config,
                )
                cell_gate_passes.append(bool(gate_result["passed"]))
                gate_receipt = {
                    "role": "residual_gate",
                    "fraction": fraction,
                    "fold": fold_name,
                    "seed": seed,
                    "train_ids_sha256": gate_model.train_ids_sha256,
                    "gate_ids_sha256": shared.ids_sha256(gate_ids),
                    "optimizer_steps": training_config.optimizer_steps,
                    "gate_prediction_sealed_before_gate_label_read": True,
                    "gate_blind_prediction_relative_path": gate_blind_relative,
                    "gate_blind_prediction_sha256": gate_blind_sha,
                    "model_relative_path": gate_model_relative,
                    "model_sha256": shared._sha(gate_model_path),
                    "model_state_sha256": gate_model.model_state_sha256,
                    "saved_model_reload_prediction_exact": True,
                    "gate": gate_result,
                    "outer_validation_target_reads": 0,
                    "elapsed_seconds": float(time.perf_counter() - gate_started),
                    "test_value_reads": 0,
                }
                gate_receipts.append(gate_receipt)
                all_receipts.append(gate_receipt)

                refit_started = time.perf_counter()
                refit_labels = shared.pd.to_numeric(
                    train.iloc[refit_ids]["label"], errors="raise"
                ).to_numpy(np.int8)
                refit_model = fit_incumbent_residual_model(
                    prefix_features,
                    layout,
                    refit_ids,
                    refit_labels,
                    teacher["seed_probability"][seed],
                    teacher["mean_probability"],
                    teacher["std_probability"],
                    teacher["seed_decision"][seed],
                    context_ids=train_ids,
                    forbidden_ids=validation_ids,
                    seed=int(seed) + int(fold["ordinal"]),
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                )
                n_rows = len(train)
                outer_seed = np.full(n_rows, 0.5, dtype=np.float32)
                outer_mean = np.full(n_rows, 0.5, dtype=np.float32)
                outer_std = np.zeros(n_rows, dtype=np.float32)
                outer_decision = np.zeros(n_rows, dtype=np.int8)
                seed_column = f"baseline__seed_{seed}__probability"
                decision_column = f"baseline__seed_{seed}__prediction"
                outer_seed[validation_ids] = comparator[seed_column].to_numpy(np.float32)
                outer_mean[validation_ids] = comparator["baseline_probability"].to_numpy(
                    np.float32
                )
                comparator_seed_matrix = np.column_stack(
                    [
                        comparator[f"baseline__seed_{registered}__probability"].to_numpy(
                            np.float32
                        )
                        for registered in SEEDS
                    ]
                )
                outer_std[validation_ids] = comparator_seed_matrix.std(axis=1).astype(np.float32)
                outer_decision[validation_ids] = comparator[decision_column].to_numpy(np.int8)
                residual_probability = predict_incumbent_residual_probability(
                    refit_model,
                    prediction_features,
                    layout,
                    validation_ids,
                    outer_seed,
                    outer_mean,
                    outer_std,
                    outer_decision,
                    context_ids=outer_context,
                    device="cuda",
                )
                model_relative = f"models/{fraction_tag}/{fold_name}/seed_{seed}.pt"
                model_path = shared._safe_path(paths["artifact"], model_relative)
                save_fitted_incumbent_residual_model(refit_model, model_path)
                loaded = load_fitted_incumbent_residual_model(model_path)
                reproduced = predict_incumbent_residual_probability(
                    loaded,
                    prediction_features,
                    layout,
                    validation_ids,
                    outer_seed,
                    outer_mean,
                    outer_std,
                    outer_decision,
                    context_ids=outer_context,
                    device="cuda",
                )
                reload_exact = bool(np.array_equal(residual_probability, reproduced))
                if not reload_exact:
                    raise RuntimeError("saved refit residual did not reproduce blind probability")
                final_probability = exact_identity_or_residual(
                    comparator[seed_column].to_numpy(np.float32),
                    residual_probability,
                    gate_passed=bool(gate_result["passed"]),
                )
                if gate_result["passed"]:
                    final_prediction = shared.apply_postprocess(
                        train.iloc[validation_ids],
                        final_probability,
                        comparator["plateau"].to_numpy(bool),
                        comparator["spike_candidate"].to_numpy(bool),
                        config["fixed_fold_postprocess"][fold_name],
                    ).astype(np.int8)
                else:
                    final_prediction = comparator[decision_column].to_numpy(np.int8).copy()
                    if not (
                        np.array_equal(
                            final_probability,
                            comparator[seed_column].to_numpy(np.float32),
                        )
                        and np.array_equal(
                            final_prediction,
                            comparator[decision_column].to_numpy(np.int8),
                        )
                    ):
                        raise AssertionError("failed gate did not preserve exact incumbent seed")
                blind_relative = (
                    f"blind_predictions/{fraction_tag}/{fold_name}/seed_{seed}.npy"
                )
                blind_path = shared._safe_path(paths["artifact"], blind_relative)
                blind_sha = shared._npy_new(blind_path, final_probability)
                seed_final_probabilities.append(final_probability)
                seed_final_predictions.append(final_prediction)
                completed_primary += 1
                primary_receipt = {
                    "role": "residual_refit",
                    "fraction": fraction,
                    "fold": fold_name,
                    "seed": seed,
                    "train_rows": int(len(refit_ids)),
                    "validation_rows": int(len(validation_ids)),
                    "optimizer_steps": training_config.optimizer_steps,
                    "raw_prefix_causal_feature_sha256": prefix_feature_sha,
                    "gate_passed": bool(gate_result["passed"]),
                    "failed_gate_exact_incumbent_identity": not gate_result["passed"],
                    "train_ids_sha256": refit_model.train_ids_sha256,
                    "validation_ids_sha256": shared.ids_sha256(validation_ids),
                    "model_relative_path": model_relative,
                    "model_sha256": shared._sha(model_path),
                    "model_state_sha256": refit_model.model_state_sha256,
                    "scaler_sha256": refit_model.scaler.state_sha256,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "blind_prediction_sealed_before_validation_target_read": True,
                    "saved_model_reload_prediction_exact": reload_exact,
                    "elapsed_seconds": float(time.perf_counter() - refit_started),
                    "validation_target_reads": 0,
                    "test_value_reads": 0,
                }
                primary_receipts.append(primary_receipt)
                all_receipts.append(primary_receipt)
                shared._emit(
                    "gen5_primary_fit_complete",
                    completed=completed_primary,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    gate_passed=bool(gate_result["passed"]),
                )
            part = comparator.copy()
            for seed, probability, prediction in zip(
                SEEDS, seed_final_probabilities, seed_final_predictions, strict=True
            ):
                part[f"challenger__seed_{seed}__probability"] = probability
                part[f"challenger__seed_{seed}__prediction"] = prediction
            if not any(cell_gate_passes):
                mean_probability = comparator["baseline_probability"].to_numpy(np.float32).copy()
                mean_prediction = comparator["baseline_prediction"].to_numpy(np.int8).copy()
                if not np.array_equal(mean_probability, comparator["baseline_probability"].to_numpy(np.float32)):
                    raise AssertionError("all-failed gates did not preserve incumbent ensemble")
            else:
                mean_probability = np.mean(
                    np.column_stack(seed_final_probabilities), axis=1
                ).astype(np.float32)
                mean_prediction = shared.apply_postprocess(
                    train.iloc[validation_ids],
                    mean_probability,
                    comparator["plateau"].to_numpy(bool),
                    comparator["spike_candidate"].to_numpy(bool),
                    config["fixed_fold_postprocess"][fold_name],
                ).astype(np.int8)
            part["challenger_probability"] = mean_probability
            part["challenger_prediction"] = mean_prediction
            part_relative = f"prediction_parts/{fold_name}_{fraction_tag}.parquet"
            part_path = shared._safe_path(paths["artifact"], part_relative)
            part_sha = shared._parquet_new(part_path, part)
            part_receipts.append(
                {
                    "fraction": fraction,
                    "fold": fold_name,
                    "rows": int(len(part)),
                    "path": part_relative,
                    "sha256": part_sha,
                    "gate_pass_count": int(sum(cell_gate_passes)),
                    "key_order_sha256": hashlib.sha256(
                        shared.pd.util.hash_pandas_object(
                            part.loc[:, [*shared.KEY_COLUMNS, "fold"]], index=False
                        ).to_numpy("<u8").tobytes()
                    ).hexdigest(),
                }
            )
    if not (
        len(teacher_receipts) == 135
        and len(gate_receipts) == 45
        and len(primary_receipts) == 45
        and len(all_receipts) == 225
        and sum(row["optimizer_steps"] for row in gate_receipts) == 5400
        and sum(row["optimizer_steps"] for row in primary_receipts) == 5400
    ):
        raise AssertionError("Gen5 curve fit or optimizer-step count differs")
    completion = {
        "schema_version": "p1_incumbent_residual_predictions_complete.v5",
        "created_at": shared._now(),
        "fit_cells": 225,
        "teacher_fit_count": 135,
        "residual_gate_fit_count": 45,
        "residual_refit_count": 45,
        "optimizer_steps": 5400,
        "gate_optimizer_steps": 5400,
        "total_residual_optimizer_steps": 10800,
        "teacher_model_receipts": teacher_receipts,
        "gate_model_receipts": gate_receipts,
        "model_receipts": primary_receipts,
        "prediction_parts": part_receipts,
        "all_inner_and_outer_blind_predictions_sealed_before_their_target_reads": True,
        "aggregate_scores_computed_before_completion": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    shared._json_new(paths["artifact"] / "predictions_complete.json", completion)
    return all_receipts, completion


def _full_fit_models(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: Any,
    features: np.ndarray,
    feature_columns: list[str],
    layout: Any,
) -> dict[str, Any]:
    full_ids = np.arange(len(train), dtype=np.int64)
    p1_config = shared.load_config(paths["base_config"])
    teacher, teacher_receipts = _teacher_oof(
        config=config,
        paths=paths,
        train=train,
        p1_config=p1_config,
        outer_prefix_ids=full_ids,
        outer_forbidden_ids=None,
        fold_name="2025_q4",
        fold_ordinal=0,
        fraction_tag="p100",
        scope="full_fit",
    )
    splits = teacher["splits"]
    gate_train_ids = np.concatenate(
        [splits[0].teacher_prediction_ids, splits[1].teacher_prediction_ids]
    )
    gate_ids = splits[2].teacher_prediction_ids
    refit_ids = teacher["oof_ids"]
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    gate_models: list[Any] = []
    refit_models: list[Any] = []
    gate_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        gate_model = fit_incumbent_residual_model(
            features,
            layout,
            gate_train_ids,
            shared.pd.to_numeric(train.iloc[gate_train_ids]["label"], errors="raise").to_numpy(
                np.int8
            ),
            teacher["seed_probability"][seed],
            teacher["mean_probability"],
            teacher["std_probability"],
            teacher["seed_decision"][seed],
            context_ids=full_ids,
            forbidden_ids=gate_ids,
            seed=seed,
            device="cuda",
            model_config=model_config,
            training_config=training_config,
        )
        gate_probability = predict_incumbent_residual_probability(
            gate_model,
            features,
            layout,
            gate_ids,
            teacher["seed_probability"][seed],
            teacher["mean_probability"],
            teacher["std_probability"],
            teacher["seed_decision"][seed],
            context_ids=full_ids,
            device="cuda",
        )
        gate_model_path = shared._safe_path(
            paths["artifact"], f"full_fit/gate_model_seed_{seed}.pt"
        )
        save_fitted_incumbent_residual_model(gate_model, gate_model_path)
        loaded_gate = load_fitted_incumbent_residual_model(gate_model_path)
        reproduced_gate = predict_incumbent_residual_probability(
            loaded_gate,
            features,
            layout,
            gate_ids,
            teacher["seed_probability"][seed],
            teacher["mean_probability"],
            teacher["std_probability"],
            teacher["seed_decision"][seed],
            context_ids=full_ids,
            device="cuda",
        )
        if not np.array_equal(gate_probability, reproduced_gate):
            raise RuntimeError("saved full-fit gate model did not reproduce")
        gate_blind_path = shared._safe_path(
            paths["artifact"], f"full_fit/gate_blind_seed_{seed}.npy"
        )
        shared._npy_new(gate_blind_path, gate_probability)
        gate_truth = shared.pd.to_numeric(
            train.iloc[gate_ids]["label"], errors="raise"
        ).to_numpy(np.int8)
        per_postprocess: dict[str, Any] = {}
        for fold_name, postprocess in config["fixed_fold_postprocess"].items():
            per_postprocess[fold_name] = _gate_decision(
                train=train,
                gate_ids=gate_ids,
                truth=gate_truth,
                base_prediction=_postprocess_ids(
                    train,
                    gate_ids,
                    teacher["seed_probability"][seed][gate_ids],
                    postprocess,
                ),
                residual_prediction=_postprocess_ids(
                    train, gate_ids, gate_probability, postprocess
                ),
                config=config,
            )
        gate_results.append(
            {
                "seed": seed,
                "passed_all_three_fixed_fold_postprocesses": all(
                    row["passed"] for row in per_postprocess.values()
                ),
                "per_postprocess": per_postprocess,
                "prediction_sealed_before_gate_label_read": True,
                "gate_model_sha256": shared._sha(gate_model_path),
                "gate_model_reload_prediction_exact": True,
            }
        )
        gate_models.append(gate_model)
        refit_models.append(
            fit_incumbent_residual_model(
                features,
                layout,
                refit_ids,
                shared.pd.to_numeric(
                    train.iloc[refit_ids]["label"], errors="raise"
                ).to_numpy(np.int8),
                teacher["seed_probability"][seed],
                teacher["mean_probability"],
                teacher["std_probability"],
                teacher["seed_decision"][seed],
                context_ids=full_ids,
                forbidden_ids=None,
                seed=seed,
                device="cuda",
                model_config=model_config,
                training_config=training_config,
            )
        )
    full_bundle = build_features(
        train.loc[:, ["station", "year", "layer", "time", "temp", "psal", "depth"]],
        config=p1_config,
        mode="offline",
        cadence_minutes=p1_config.data.cadence_minutes,
        group_columns=p1_config.data.group_columns,
    )
    encoder = TabularEncoder().fit(full_bundle, full_ids)
    full_offline_features = encoder.transform(full_bundle, full_ids)
    labels = shared.pd.to_numeric(train["label"], errors="raise").to_numpy(np.int8)
    exact_parameters = dict(p1_config.raw["models"]["xgboost"])
    threads = int(p1_config.raw["project"]["threads"])
    base_models: list[Any] = []
    base_paths: list[tuple[str, Path, str]] = []
    base_probabilities: list[np.ndarray] = []
    for seed in SEEDS:
        base_model = _fit_model(
            "xgboost", exact_parameters, seed, threads, full_offline_features, labels
        )
        base_relative = f"full_fit/base_seed_{seed}.joblib"
        base_path = shared._safe_path(paths["artifact"], base_relative)
        base_sha = _save_joblib_new(base_path, {"encoder": encoder, "model": base_model})
        loaded_base = joblib.load(base_path)
        full_base = base_model.predict_proba(full_offline_features)[:, 1].astype(np.float32)
        original_base = full_base
        reloaded_base = loaded_base["model"].predict_proba(
            loaded_base["encoder"].transform(full_bundle, full_ids)
        )[:, 1].astype(np.float32)
        if not np.array_equal(original_base, reloaded_base):
            raise RuntimeError("saved full-fit incumbent base did not reproduce")
        base_models.append(base_model)
        base_paths.append((base_relative, base_path, base_sha))
        base_probabilities.append(full_base)
    base_matrix = np.column_stack(base_probabilities)
    base_mean = base_matrix.mean(axis=1).astype(np.float32)
    base_std = base_matrix.std(axis=1).astype(np.float32)
    base_decisions = [
        _postprocess_ids(
            train,
            full_ids,
            probability,
            config["fixed_fold_postprocess"]["2025_q4"],
        )
        for probability in base_probabilities
    ]
    reference_ids = gate_ids[: min(4096, len(gate_ids))]
    packages: list[dict[str, Any]] = []
    for index, seed in enumerate(SEEDS):
        base_relative, base_path, base_sha = base_paths[index]
        residual_relative = f"full_fit/residual_seed_{seed}.pt"
        residual_path = shared._safe_path(paths["artifact"], residual_relative)
        save_fitted_incumbent_residual_model(refit_models[index], residual_path)
        loaded_residual = load_fitted_incumbent_residual_model(residual_path)
        if loaded_residual.model_state_sha256 != refit_models[index].model_state_sha256:
            raise RuntimeError("saved full-fit residual state differs")
        original_residual = predict_incumbent_residual_probability(
            refit_models[index],
            features,
            layout,
            reference_ids,
            base_probabilities[index],
            base_mean,
            base_std,
            base_decisions[index],
            context_ids=full_ids,
            device="cuda",
        )
        reloaded_residual = predict_incumbent_residual_probability(
            loaded_residual,
            features,
            layout,
            reference_ids,
            base_probabilities[index],
            base_mean,
            base_std,
            base_decisions[index],
            context_ids=full_ids,
            device="cuda",
        )
        if not np.array_equal(original_residual, reloaded_residual):
            raise RuntimeError("saved full-fit residual did not reproduce inference")
        gate_passed = bool(
            gate_results[index]["passed_all_three_fixed_fold_postprocesses"]
        )
        package_probability = exact_identity_or_residual(
            base_probabilities[index][reference_ids],
            reloaded_residual,
            gate_passed=gate_passed,
        )
        if not gate_passed and not np.array_equal(
            package_probability, base_probabilities[index][reference_ids]
        ):
            raise AssertionError("failed full-fit gate did not preserve incumbent identity")
        packages.append(
            {
                "seed": seed,
                "base_model_path": base_relative,
                "base_model_sha256": base_sha,
                "residual_model_path": residual_relative,
                "residual_model_sha256": shared._sha(residual_path),
                "residual_model_state_sha256": refit_models[index].model_state_sha256,
                "gate": gate_results[index],
                "saved_base_reload_prediction_exact": True,
                "saved_residual_reload_state_exact": True,
                "saved_residual_reload_inference_exact": True,
                "failed_gate_reference_probability_identity_exact": not gate_passed,
                "serialization_reference_ids_sha256": shared.ids_sha256(reference_ids),
            }
        )
    if len(teacher_receipts) != 9 or len(gate_models) != 3 or len(refit_models) != 3:
        raise AssertionError("full-fit teacher/gate/refit count differs")
    receipt = {
        "performed": True,
        "model_count": 18,
        "inner_teacher_fits": 9,
        "residual_gate_fits": 3,
        "residual_refits": 3,
        "incumbent_inference_base_fits": 3,
        "saved_inference_package_count": 3,
        "optimizer_steps": 720,
        "teacher_model_receipts": teacher_receipts,
        "models": packages,
        "feature_columns": feature_columns,
        "test_value_reads": 0,
        "test_prediction_generations": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    shared._json_new(paths["artifact"] / "full_fit_models.json", receipt)
    return receipt


def _patch_shared_engine() -> None:
    shared.__file__ = str(Path(__file__).resolve())
    shared.EXPECTED_CONFIG_SHA256 = EXPECTED_CONFIG_SHA256
    shared.EXPECTED_CONFIG_DEEP_SHA256 = EXPECTED_CONFIG_DEEP_SHA256
    shared.CANONICAL_CONFIG = CANONICAL_CONFIG
    shared.CANONICAL_ARTIFACT = CANONICAL_ARTIFACT
    shared.CANONICAL_LOCK = CANONICAL_LOCK
    shared.HYPOTHESIS = HYPOTHESIS
    shared.FRACTIONS = FRACTIONS
    shared.SEEDS = SEEDS
    shared._paths = _paths
    shared._json_new = _json_new
    shared.authorize_entry = authorize_entry
    shared._model_config = _model_config
    shared._training_config = _training_config
    shared._prefixes = r4._pinned_label_free_prefixes
    shared._run_curve = _run_curve
    shared._full_fit_models = _full_fit_models
    shared.evaluate_learning_curve = evaluate_learning_curve
    shared.load_contract = load_contract


_patch_shared_engine()


def _causal_feature_audit(*, data_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    raw = shared.pd.read_csv(
        data_dir / "train.csv",
        usecols=["station", "layer", "time", "temp", "psal", "depth"],
        low_memory=False,
    )
    rebuilt = build_causal_raw_features(raw)
    cached = shared.pd.read_parquet(paths["feature_cache"], columns=list(CAUSAL_FEATURE_COLUMNS))
    if not np.array_equal(rebuilt.to_numpy(np.float32), cached.to_numpy(np.float32), equal_nan=True):
        raise PermissionError("pinned causal cache differs from raw-only rebuild")
    parsed = shared.pd.to_datetime(raw["time"], errors="raise", utc=True, format="mixed")
    groups = raw["station"].astype(str) + "|" + raw["layer"].astype(str)
    order = shared.pd.DataFrame(
        {"group": groups, "time": parsed, "row": np.arange(len(raw), dtype=np.int64)}
    )
    order.sort_values(["group", "time", "row"], kind="mergesort", inplace=True)
    prefix_ids: list[int] = []
    for _, rows in order.groupby("group", sort=False, observed=True):
        keep = max(1, int(len(rows) * 0.7))
        prefix_ids.extend(rows.iloc[:keep]["row"].astype(int).tolist())
    invariance_sha = assert_future_value_invariance(raw, sorted(prefix_ids))
    return {
        "target_columns_read": 0,
        "feature_count": len(CAUSAL_FEATURE_COLUMNS),
        "cache_exact_to_raw_rebuild": True,
        "future_value_perturbation_invariant": True,
        "future_value_perturbation_prefix_sha256": invariance_sha,
    }


def _inner_split_static_audit(
    *, data_dir: Path, paths: dict[str, Path], config: dict[str, Any]
) -> dict[str, Any]:
    metadata = shared.pd.read_csv(
        data_dir / "train.csv",
        usecols=["station", "year", "layer", "time"],
        low_memory=False,
    )
    frozen = shared.pd.read_parquet(paths["frozen_oof"], columns=[*shared.KEY_COLUMNS, "fold"])
    folds, _ = shared.gen1._fold_runtime(
        metadata,
        shared.load_config(paths["base_config"]),
        frozen,
    )
    prefix_ids, prefix_audit = r4._pinned_label_free_prefixes(
        object(), folds, int(config["features"]["cadence_minutes"])
    )
    cells: list[dict[str, Any]] = []
    split_refs: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
    for fraction in FRACTIONS:
        for fold in folds:
            fold_name = str(fold["name"])
            outer_ids = prefix_ids[(fold_name, fraction)]
            splits = build_three_block_inner_splits(metadata, outer_ids, purge_days=7)
            gate_stations = set(
                metadata.iloc[splits[2].teacher_prediction_ids]["station"].astype(str)
            )
            if (
                len(splits) != 3
                or any(
                    np.intersect1d(split.teacher_train_ids, split.teacher_prediction_ids).size
                    for split in splits
                )
                or any(
                    np.intersect1d(split.teacher_prediction_ids, fold["val_idx"]).size
                    for split in splits
                )
                or not all(
                    np.isin(split.teacher_train_ids, outer_ids).all()
                    and np.isin(split.teacher_prediction_ids, outer_ids).all()
                    for split in splits
                )
            ):
                raise PermissionError(f"label-free inner split static audit differs: {fraction}:{fold_name}")
            cell = {
                    "fraction": fraction,
                    "fold": fold_name,
                    "outer_prefix_rows": int(len(outer_ids)),
                    "outer_prefix_ids_sha256": shared.ids_sha256(outer_ids),
                    "outer_validation_ids_sha256": shared.ids_sha256(fold["val_idx"]),
                    "gate_required_stations": sorted(gate_stations),
                    "gate_missing_required_stations": sorted(set(STATIONS) - gate_stations),
                    "missing_required_station_will_fail_closed": gate_stations
                    != set(STATIONS),
                    "blocks": [
                        {
                            "block": split.block,
                            "train_rows": int(len(split.teacher_train_ids)),
                            "prediction_rows": int(len(split.teacher_prediction_ids)),
                            "train_ids_sha256": split.train_ids_sha256,
                            "prediction_ids_sha256": split.prediction_ids_sha256,
                        }
                        for split in splits
                    ],
                }
            cells.append(cell)
            split_refs.append((cell, splits))
    prefix_rows = [row for fraction in prefix_audit.values() for row in fraction.values()]
    if len(cells) != 15 or len(prefix_rows) != 15:
        raise PermissionError("exact outer-prefix inner-split cell count differs")
    labels = shared.pd.to_numeric(
        shared.pd.read_csv(data_dir / "train.csv", usecols=["label"], low_memory=False)[
            "label"
        ],
        errors="raise",
    ).to_numpy(np.int8)

    def counts(ids: np.ndarray) -> dict[str, int]:
        selected = labels[ids]
        return {
            "negative": int(np.count_nonzero(selected == 0)),
            "positive": int(np.count_nonzero(selected == 1)),
        }

    for cell, splits in split_refs:
        teacher_counts = [counts(split.teacher_train_ids) for split in splits]
        gate_fit_ids = np.concatenate(
            [splits[0].teacher_prediction_ids, splits[1].teacher_prediction_ids]
        )
        refit_ids = np.concatenate([split.teacher_prediction_ids for split in splits])
        gate_counts = counts(splits[2].teacher_prediction_ids)
        gate_fit_counts = counts(gate_fit_ids)
        refit_counts = counts(refit_ids)
        if any(min(row.values()) <= 0 for row in teacher_counts) or min(
            gate_fit_counts.values()
        ) <= 0 or min(refit_counts.values()) <= 0:
            raise PermissionError(
                f"post-split training-label viability differs: {cell['fraction']}:{cell['fold']}"
            )
        cell["post_split_training_label_viability"] = {
            "teacher_train_binary_counts": teacher_counts,
            "residual_gate_fit_binary_counts": gate_fit_counts,
            "residual_refit_binary_counts": refit_counts,
            "gate_block_binary_counts": gate_counts,
            "gate_missing_binary_class_will_fail_closed": min(gate_counts.values()) <= 0,
        }
    return {
        "source_columns_read": ["station", "year", "layer", "time"],
        "split_construction_target_columns_read": 0,
        "post_split_training_label_columns_read": ["label"],
        "post_split_training_label_read_after_all_split_ids_frozen": True,
        "target_fold_scores": 0,
        "test_value_reads": 0,
        "outer_cell_count": len(cells),
        "split_count_per_outer_cell": 3,
        "purge_days": 7,
        "train_prediction_disjoint": True,
        "outer_validation_disjoint": True,
        "all_ids_within_exact_outer_prefix": True,
        "gate_cells_with_all_required_stations": int(
            sum(not row["gate_missing_required_stations"] for row in cells)
        ),
        "gate_cells_missing_a_required_station": int(
            sum(bool(row["gate_missing_required_stations"]) for row in cells)
        ),
        "missing_required_station_gate_fails_closed_to_exact_incumbent": True,
        "all_teacher_and_residual_fit_sets_contain_both_binary_classes": True,
        "missing_binary_class_gate_fails_closed_to_exact_incumbent": True,
        "historical_label_derived_cutoff_cells": int(
            sum(row["boundary_split_risk_if_nominal_cutoff_used"] for row in prefix_rows)
        ),
        "historical_cutoff_label_derivation_disclosed_literally": True,
        "cells": cells,
    }


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _paths(root)
    config, paths, pins = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_artifact=paths["artifact"],
    )
    comparator = shared._verify_gen1_parts(root, paths)
    causal_audit = _causal_feature_audit(data_dir=data_dir, paths=paths)
    split_audit = _inner_split_static_audit(
        data_dir=data_dir, paths=paths, config=config
    )
    return {
        "status": "CANONICAL_CHECK_ONLY_PASS",
        "experiment_id": config["experiment_id"],
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "immutable_pin_count": len(pins),
        "gen1_comparator_parts": len(comparator),
        "curve_fit_cells": 225,
        "curve_teacher_fits": 135,
        "curve_residual_gate_fits": 45,
        "curve_residual_refits": 45,
        "curve_total_residual_optimizer_steps": 10800,
        "maximum_full_fit_fits_on_pass": 18,
        "causal_feature_audit": causal_audit,
        "inner_split_audit": split_audit,
        "v5_ledger_binding": _verify_v5_ledger_binding(root, config, paths["ledger"]),
        "failed_generation_binding": _verify_failed_generation(config, paths),
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
        "artifact_absent": not paths["artifact"].exists(),
        "attempt_lock_absent": not paths["lock"].exists(),
    }


def seal(*, root: Path, data_dir: Path) -> dict[str, Any]:
    static = check_only(root=root, data_dir=data_dir)
    result = shared.seal(root=root, data_dir=data_dir)
    receipt_path = _paths(root)["artifact"] / "preseal_static_qa.json"
    receipt = {
        "schema_version": "p1_incumbent_residual_preseal_static_qa.v5",
        "created_before_first_fit": True,
        "canonical_root": str(CANONICAL_ROOT_PATH),
        "canonical_data_dir": str(CANONICAL_DATA_DIR_PATH),
        "clone_reparse_hardlink_rejected": True,
        "static_check": static,
        "start_end_pin_mapping": {
            "authorize_return_key_count": static["immutable_pin_count"],
            "inherited_recompute_key_count": static["immutable_pin_count"],
            "semantic_audit_records_returned_as_pins": False,
            "exact_mapping_if_inputs_unchanged": True,
        },
        "historical_prefix_cutoff_provenance_literal": "Four cutoffs were historically label-derived in Gen1; the current selector reads timestamps and row positions only.",
        "operation_counters": {
            "curve_model_fits": 0,
            "target_fold_scores": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
            "attempt_locks": 0,
        },
    }
    shared._json_new(receipt_path, receipt)
    return {**result, "preseal_static_qa_sha256": shared._sha(receipt_path)}


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    return shared.run_experiment(root=root, data_dir=data_dir)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--seal-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(os.path.abspath(args.root))
    data_dir = Path(os.path.abspath(args.data_dir))
    if args.check_only:
        result = check_only(root=root, data_dir=data_dir)
    elif args.seal_only:
        result = seal(root=root, data_dir=data_dir)
    else:
        result = run_experiment(root=root, data_dir=data_dir)
    if not args.run:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
