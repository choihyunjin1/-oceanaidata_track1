"""Run the append-only P1 causal masked-pretrain binary-event Gen4r3 curve once."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import ast
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
from p1_qc.masked_pretrain_binary_tcn import (
    MaskedPretrainModelConfig,
    MaskedPretrainTrainingConfig,
    fit_masked_pretrain_binary_event_model,
    load_fitted_masked_pretrain_model,
    predict_masked_pretrain_binary_probability,
    save_fitted_masked_pretrain_model,
)

_GEN3_ADAPTER = PROJECT_ROOT / "scripts/run_p1_binary_event_tcn_dense_natural_v3.py"
_GEN3_SPEC = importlib.util.spec_from_file_location("p1_gen4_shared_curve_adapter", _GEN3_ADAPTER)
if _GEN3_SPEC is None or _GEN3_SPEC.loader is None:
    raise ImportError("failed to load pinned P1 Gen3 curve adapter")
gen3 = importlib.util.module_from_spec(_GEN3_SPEC)
sys.modules[_GEN3_SPEC.name] = gen3
_GEN3_SPEC.loader.exec_module(gen3)
shared = gen3.shared

EXPECTED_CONFIG_SHA256 = "9cb6bbfa535c51a5c58a04251f9ab68a8a5f6511cdcabc5277dfcb73f9d19ff8"
EXPECTED_CONFIG_DEEP_SHA256 = "1fe173b815c8131c3ed33c74a1453da662a7aadb8e1be7823486428deb4e4e5f"
CANONICAL_CONFIG = "configs/experiments/p1_masked_pretrain_binary_event_v4r3.json"
CANONICAL_ARTIFACT = "artifacts/p1_masked_pretrain_binary_event_v4r3"
CANONICAL_LOCK = "artifacts/p1_masked_pretrain_binary_event_v4r3.ATTEMPT_LOCK.json"
HYPOTHESIS = (
    "masked_sequence_pretraining_then_binary_event_finetune_causal_prefix_safe_"
    "sealed_incumbent_cutoffs"
)
CANONICAL_ROOT_PATH = Path(r"C:\Users\cedis\PycharmProjects\PythonProject")
CANONICAL_DATA_DIR_PATH = (
    CANONICAL_ROOT_PATH / "데이터셋 원본/데이터셋_P1/P1_qc_anomaly"
)
FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
_SHARED_JSON_NEW = shared._json_new
EXECUTION_TOMBSTONE = (
    "artifacts/p1_masked_pretrain_binary_event_v4r3/EXECUTION_TOMBSTONE.json"
)
EXECUTION_TOMBSTONE_SHA256 = (
    "3c8b83b68f2aa009beeaf0f3873473ac382de1500e0c29dfffdc94813acf55c9"
)


def _enforce_execution_tombstone(root: Path) -> None:
    path = (root.resolve(strict=True) / EXECUTION_TOMBSTONE).resolve(strict=True)
    if shared._sha(path) != EXECUTION_TOMBSTONE_SHA256:
        raise PermissionError("invalidated Gen4r3 execution tombstone SHA differs")
    value = shared._json(path)
    if not (
        value.get("generation") == "p1_masked_pretrain_binary_event_v4r3"
        and value.get("execution_prohibited") is True
        and value.get("authorization_must_fail_before_attempt_lock") is True
        and value.get("attempt_lock_created") is False
        and value.get("curve_model_fits") == 0
        and value.get("uploads") == 0
    ):
        raise PermissionError("invalidated Gen4r3 execution tombstone semantics differ")
    raise PermissionError("p1_masked_pretrain_binary_event_v4r3 is superseded and non-executable")


def _paths(root: Path) -> dict[str, Path]:
    return {
        "config": root / CANONICAL_CONFIG,
        "artifact": root / CANONICAL_ARTIFACT,
        "lock": root / CANONICAL_LOCK,
        "base_config": root / "configs/p1.toml",
        "goal": root / "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.parquet",
        "feature_metadata": root / "artifacts/cache/train_causal_raw_prefix_safe_v4r2.json",
        "gen1": root / "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen2": root / "artifacts/p1_station_layer_temporal_convolution_event_v2",
        "gen3": root / "artifacts/p1_binary_event_tcn_dense_natural_v3",
        "frozen_oof": root / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "superseded_preregistration": root
        / "artifacts/p1_masked_pretrain_binary_event_v4r2/preregistration.json",
        "v4_tombstone": root
        / "artifacts/p1_masked_pretrain_binary_event_v4/EXECUTION_TOMBSTONE.json",
        "v4r2_tombstone": root
        / "artifacts/p1_masked_pretrain_binary_event_v4r2/EXECUTION_TOMBSTONE.json",
        "v4_runner": root / "scripts/run_p1_masked_pretrain_binary_event_v4.py",
        "v4r2_runner": root / "scripts/run_p1_masked_pretrain_binary_event_v4r2.py",
        "v4_lock": root
        / "artifacts/p1_masked_pretrain_binary_event_v4.ATTEMPT_LOCK.json",
        "v4r2_lock": root
        / "artifacts/p1_masked_pretrain_binary_event_v4r2.ATTEMPT_LOCK.json",
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
    if not records:
        raise PermissionError("canonical v5 ledger is empty")
    head = records[-1]
    if (
        len(records) != binding["event_count"]
        or head["seq"] != binding["head_seq"]
        or head["event_sha256"] != binding["head_event_sha256"]
    ):
        raise PermissionError("canonical v5 ledger latest head differs")
    uploads = sum(record["payload"].get("upload_performed") is True for record in records)
    if (
        binding["all_event_upload_performed_false"] is not True
        or binding["semantic_upload_count"] != 0
        or uploads != 0
        or not all(record["payload"].get("upload_performed") is False for record in records)
    ):
        raise PermissionError("canonical v5 ledger semantic upload count differs from zero")
    return {**observed, "event_count": len(records), "head_event_sha256": head["event_sha256"]}


def _runner_tombstone_constants(path: Path) -> tuple[str, str, bool]:
    """Read the predecessor refusal contract without importing or mutating shared state."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, Any] = {}
    authorize_first_is_tombstone = False
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            if value is not None:
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in {
                        "EXECUTION_TOMBSTONE",
                        "EXECUTION_TOMBSTONE_SHA256",
                    }:
                        constants[target.id] = ast.literal_eval(value)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "authorize_entry":
            first = node.body[0] if node.body else None
            authorize_first_is_tombstone = bool(
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Call)
                and isinstance(first.value.func, ast.Name)
                and first.value.func.id == "_enforce_execution_tombstone"
            )
    return (
        str(constants.get("EXECUTION_TOMBSTONE", "")),
        str(constants.get("EXECUTION_TOMBSTONE_SHA256", "")),
        authorize_first_is_tombstone,
    )


def _verify_superseded_execution_guards(
    root: Path, config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    specs = (
        (
            "v4",
            "p1_masked_pretrain_binary_event_v4",
            "artifacts/p1_masked_pretrain_binary_event_v4/preregistration.json",
            "e989c48470e0c65c80cfd24162d39e4e747c6eb1f3eb104b49b5db03ad8418ba",
            "artifacts/p1_masked_pretrain_binary_event_v4/STATIC_QA_NO_GO_SUPERSEDED_20260823.json",
            "c31ceb39d96be73a0efcb8817fa19f70b30fa477d3b62df5a11b95ef1e79cee1",
        ),
        (
            "v4r2",
            "p1_masked_pretrain_binary_event_v4r2",
            "artifacts/p1_masked_pretrain_binary_event_v4r2/preregistration.json",
            "ad9e654efcd520578f66d23816b1c6e1a35e200e391b812341dd2bdc665d21f2",
            "artifacts/p1_masked_pretrain_binary_event_v4r2/STATIC_QA_NO_GO_SUPERSEDED_20260823.json",
            "28b89a1d15d1ee923672dcc26663db85e69a5f02ebd5fe7a3f6a64cad9c79eb4",
        ),
    )
    result: dict[str, Any] = {}
    for key, generation, prereg_path, prereg_sha, receipt_path, receipt_sha in specs:
        tombstone_path = paths[f"{key}_tombstone"]
        runner_path = paths[f"{key}_runner"]
        lock_path = paths[f"{key}_lock"]
        tombstone = shared._json(tombstone_path)
        expected_tombstone_relative = tombstone_path.relative_to(root).as_posix()
        runner_relative_tombstone, runner_tombstone_sha, first_guard = (
            _runner_tombstone_constants(runner_path)
        )
        if not (
            tombstone["schema_version"] == "p1_execution_tombstone.v1"
            and tombstone["problem"] == "P1"
            and tombstone["generation"] == generation
            and tombstone["preregistration_path"] == prereg_path
            and tombstone["preregistration_sha256"] == prereg_sha
            and tombstone["supersession_receipt_path"] == receipt_path
            and tombstone["supersession_receipt_sha256"] == receipt_sha
            and tombstone["successor_generation"]
            == "p1_masked_pretrain_binary_event_v4r3"
            and tombstone["execution_prohibited"] is True
            and tombstone["authorization_must_fail_before_attempt_lock"] is True
            and tombstone["attempt_lock_created"] is False
            and tombstone["append_only"] is True
            and all(
                tombstone[name] == 0
                for name in (
                    "curve_model_fits",
                    "target_fold_scores",
                    "test_value_reads",
                    "candidate_files",
                    "uploads",
                )
            )
            and not lock_path.exists()
            and runner_relative_tombstone == expected_tombstone_relative
            and runner_tombstone_sha == shared._sha(tombstone_path)
            and first_guard
            and shared._sha(runner_path)
            == config["implementation_sha256"][f"superseded_{key}_runner"]
        ):
            raise PermissionError(f"superseded {key} execution refusal differs")
        result[key] = {
            "generation": generation,
            "runner_sha256": shared._sha(runner_path),
            "tombstone_path": expected_tombstone_relative,
            "tombstone_sha256": shared._sha(tombstone_path),
            "authorization_guard_is_first_statement": True,
            "attempt_lock_absent": True,
            "curve_model_fits": 0,
            "test_value_reads": 0,
            "uploads": 0,
        }
    return result


def _pinned_label_free_prefixes(
    train: Any, folds: list[dict[str, Any]], cadence_minutes: int
) -> tuple[dict[tuple[str, float], np.ndarray], dict[str, Any]]:
    """Select exact incumbent prefix IDs using only sealed timestamps and row keys.

    Gen1's immutable audit files record the adjusted cutoffs and exact ID hashes.
    Their historical event-safe derivation is disclosed, but this current selector
    never accesses a label value or column.
    """

    del train, cadence_minutes
    completion_path = (
        PROJECT_ROOT
        / "artifacts/p1_meaningful_learning_curve_generation_v1/predictions_complete.json"
    )
    completion = shared._json(completion_path)
    receipts = {
        (str(row["fold"]), float(row["fraction"])): row for row in completion["parts"]
    }
    expected = {(fold["name"], fraction) for fold in folds for fraction in FRACTIONS}
    if set(receipts) != expected:
        raise PermissionError("sealed Gen1 prefix-audit receipt cells differ")
    result: dict[tuple[str, float], np.ndarray] = {}
    audit: dict[str, Any] = {}
    for fraction in FRACTIONS:
        audit[shared._tag(fraction)] = {}
        for fold in folds:
            cell = (str(fold["name"]), fraction)
            receipt = receipts[cell]
            parquet = (
                PROJECT_ROOT / str(receipt["parquet"]).replace("\\", "/")
            ).resolve(strict=True)
            if not parquet.is_relative_to(
                (PROJECT_ROOT / "artifacts/p1_meaningful_learning_curve_generation_v1").resolve(
                    strict=True
                )
            ):
                raise PermissionError(f"sealed Gen1 prefix audit escapes artifact: {cell}")
            part_audit_path = parquet.with_suffix(".json")
            if shared._sha(part_audit_path) != receipt["audit_sha256"]:
                raise PermissionError(f"sealed Gen1 prefix audit SHA differs: {cell}")
            part_audit = shared._json(part_audit_path)
            if part_audit["fold"] != cell[0] or float(part_audit["fraction"]) != fraction:
                raise PermissionError(f"sealed Gen1 prefix audit identity differs: {cell}")
            adjusted = int(shared.pd.Timestamp(part_audit["adjusted_cutoff_utc"]).value)
            nominal = int(shared.pd.Timestamp(part_audit["nominal_cutoff_utc"]).value)
            fold_train = np.asarray(fold["train_idx"], dtype=np.int64)
            time_ns = np.asarray(fold["time_ns"], dtype=np.int64)
            ids = fold_train[time_ns[fold_train] <= adjusted]
            if (
                len(ids) != int(part_audit["prefix_rows"])
                or shared.ids_sha256(ids) != part_audit["prefix_positions_sha256"]
                or np.intersect1d(ids, fold["val_idx"]).size
            ):
                raise PermissionError(f"sealed Gen1 prefix IDs differ: {cell}")
            result[cell] = ids
            audit[shared._tag(fraction)][cell[0]] = {
                "fraction": fraction,
                "eligible_rows": int(len(fold_train)),
                "prefix_rows": int(len(ids)),
                "nominal_cutoff_utc": part_audit["nominal_cutoff_utc"],
                "adjusted_cutoff_utc": part_audit["adjusted_cutoff_utc"],
                "current_run_prefix_selector_target_reads": 0,
                "current_run_prefix_selector_columns": ["time", "row_position"],
                "historical_event_safe_cutoff_lineage_disclosed": True,
                "historical_event_boundary_retreat_iterations": int(
                    part_audit["event_boundary_retreat_iterations"]
                ),
                "boundary_split_risk_if_nominal_cutoff_used": bool(adjusted < nominal),
                "id_sha256_little_endian_int64": shared.ids_sha256(ids),
                "validation_id_sha256_little_endian_int64": shared.ids_sha256(
                    fold["val_idx"]
                ),
                "exact_to_immutable_incumbent_fold_train_ids": True,
            }
    return result, audit


def _model_config(
    config: dict[str, Any], feature_count: int, group_count: int
) -> MaskedPretrainModelConfig:
    model = config["model"]
    result = MaskedPretrainModelConfig(
        input_feature_count=feature_count,
        group_count=group_count,
        width=int(model["width"]),
        group_embedding_width=int(model["group_embedding_width"]),
        dilations=tuple(int(value) for value in model["dilations"]),
        kernel_size=int(model["kernel_size"]),
        dropout=float(model["dropout"]),
        norm_groups=int(model["norm_groups"]),
    )
    result.validate()
    if result.receptive_field_rows != model["receptive_field_rows"]:
        raise ValueError("registered receptive field differs from implementation")
    return result


def _training_config(config: dict[str, Any]) -> MaskedPretrainTrainingConfig:
    training = config["training"]
    result = MaskedPretrainTrainingConfig(
        optimizer_steps=int(training["optimizer_steps_per_cell"]),
        pretrain_steps=int(training["pretrain_steps_per_cell"]),
        finetune_steps=int(training["finetune_steps_per_cell"]),
        batch_size=int(training["batch_size"]),
        pretrain_learning_rate=float(training["pretrain_learning_rate"]),
        finetune_learning_rate=float(training["finetune_learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        mask_probability=float(training["mask_probability"]),
        auxiliary_loss_weight=float(training["auxiliary_loss_weight"]),
        boundary_band_rows=int(training["boundary_band_rows"]),
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
    if config["experiment_id"] != "p1_masked_pretrain_binary_event_v4r3":
        raise PermissionError("experiment identity differs")
    if config.get("comparison_mode") != "EXACT_OFFICIAL_PREFIX_REFIT":
        raise PermissionError("P1 Gen4r3 comparison mode must remain exact")
    if config["canonical_paths"] != {
        "config": CANONICAL_CONFIG,
        "base_config": "configs/p1.toml",
        "goal_contract": "configs/goals/meaningful_score_maximization_v3.json",
        "feature_cache": "artifacts/cache/train_causal_raw_prefix_safe_v4r2.parquet",
        "feature_metadata": "artifacts/cache/train_causal_raw_prefix_safe_v4r2.json",
        "gen1_artifact": "artifacts/p1_meaningful_learning_curve_generation_v1",
        "gen2_artifact": "artifacts/p1_station_layer_temporal_convolution_event_v2",
        "gen3_artifact": "artifacts/p1_binary_event_tcn_dense_natural_v3",
        "frozen_oof": "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet",
        "superseded_preregistration": "artifacts/p1_masked_pretrain_binary_event_v4r2/preregistration.json",
        "v4_tombstone": "artifacts/p1_masked_pretrain_binary_event_v4/EXECUTION_TOMBSTONE.json",
        "v4r2_tombstone": "artifacts/p1_masked_pretrain_binary_event_v4r2/EXECUTION_TOMBSTONE.json",
        "superseded_v4_runner": "scripts/run_p1_masked_pretrain_binary_event_v4.py",
        "superseded_v4r2_runner": "scripts/run_p1_masked_pretrain_binary_event_v4r2.py",
        "v4_attempt_lock": "artifacts/p1_masked_pretrain_binary_event_v4.ATTEMPT_LOCK.json",
        "v4r2_attempt_lock": "artifacts/p1_masked_pretrain_binary_event_v4r2.ATTEMPT_LOCK.json",
        "v5_ledger": "artifacts/meaningful_score_goal_v5/registry.jsonl",
        "artifact": CANONICAL_ARTIFACT,
        "attempt_lock": CANONICAL_LOCK,
    }:
        raise PermissionError("canonical path contract differs")
    if [item["id"] for item in config["hypotheses"]] != [HYPOTHESIS]:
        raise PermissionError("single registered hypothesis differs")
    if tuple(config["prefix_fractions"]) != FRACTIONS or tuple(config["seeds"]) != SEEDS:
        raise PermissionError("prefix or seed contract differs")
    training = config["training"]
    if not (
        training["optimizer_steps_per_cell"] == 120
        and training["pretrain_steps_per_cell"] == 30
        and training["finetune_steps_per_cell"] == 90
        and training["batch_size"] == 8192
        and training["pretraining_label_reads"] == 0
        and training["runtime_prefix_selector_target_reads"] == 0
        and training["expected_curve_fit_cells"] == 45
        and training["expected_curve_optimizer_steps"] == 5400
        and training["expected_curve_pretrain_steps"] == 1350
        and training["expected_curve_finetune_steps"] == 4050
        and training["main_event_loss"]
        == "unweighted_BCEWithLogits_on_dense_natural_prefix_rows"
        and training["phase_balanced_resampling"] is False
        and training["natural_prior_probability_correction"] is False
        and training["hyperparameter_search"] is False
    ):
        raise PermissionError("fixed masked-pretrain/fine-tune contract differs")
    model = config["model"]
    if not (
        model["pretraining_head"] == "masked_center_feature_reconstruction"
        and model["probability_rule"] == "sigmoid(binary_event_logit)_only"
        and model["reconstruction_or_auxiliary_probability_use_forbidden"] is True
        and model["inference_head"] == "binary_event"
        and model["auxiliary_heads"] == ["onset", "offset"]
    ):
        raise PermissionError("Gen4r3 head/inference contract differs")
    if not all(value is True for value in config["prohibitions"].values()):
        raise PermissionError("all prohibitions must remain enabled")
    features = config["features"]
    if not (
        features["within_prefix_unlabeled_centered_context_allowed"] is True
        and features["out_of_prefix_context_zero_masked"] is True
        and features["raw_prefix_rebuilt_before_every_fit_cell"] is True
        and features["future_raw_value_perturbation_invariant"] is True
        and features["forward_centered_or_terminal_run_features"] == []
        and tuple(features["selected_numeric_columns"]) == CAUSAL_FEATURE_COLUMNS
        and features["scaler"]
        == "exact_prefix_train_only_componentwise_median_iqr_plus_finite_mask"
    ):
        raise PermissionError("Gen4r3 exact-prefix feature context contract differs")
    protocol = config["prefix_protocol"]
    if not (
        protocol["selector"]
        == "sealed_gen1_adjusted_cutoff_timestamp_and_row_position_only"
        and protocol["current_run_selector_columns"] == ["time", "row_position"]
        and protocol["current_run_selector_target_reads"] == 0
        and protocol["current_run_pretraining_target_tensor_materializations_before_pretraining"]
        == 0
        and protocol["exact_incumbent_prefix_row_count_and_sha_required"] is True
        and protocol["historical_event_safe_cutoff_lineage_disclosed"] is True
        and protocol["historical_event_protected_fold_key_lineage_disclosed"] is True
        and protocol["current_run_fold_selector_target_reads"] == 0
        and protocol["nominal_event_boundary_split_risk_documented"] is True
        and protocol["expected_historical_boundary_retreat_cells"] == 4
    ):
        raise PermissionError("Gen4r3 label-free exact-prefix selector contract differs")
    implementations = {
        "masked_pretrain_module": root / "src/p1_qc/masked_pretrain_binary_tcn.py",
        "causal_feature_module": root / "src/p1_qc/causal_raw_features_v4r2.py",
        "causal_cache_builder": root / "scripts/build_p1_causal_raw_feature_cache_v4r2.py",
        "binary_event_module": root / "src/p1_qc/binary_event_tcn.py",
        "shared_temporal_layout_module": root / "src/p1_qc/temporal_event_tcn.py",
        "shared_gen3_adapter": _GEN3_ADAPTER,
        "shared_gen2_runner": root
        / "scripts/run_p1_station_layer_temporal_convolution_event_v2.py",
        "gen1_runner": root / "scripts/run_p1_meaningful_learning_curve_generation_v1.py",
        "base_config": paths["base_config"],
        "pipeline": root / "src/p1_qc/pipeline.py",
        "validation": root / "src/p1_qc/validation.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score_v3.py",
        "v5_ledger_contract": root / "configs/goals/meaningful_score_ledger_v5.json",
        "v5_ledger_evaluator": root / "src/ocean_goal/meaningful_score_ledger_v5.py",
        "superseded_v4_runner": paths["v4_runner"],
        "superseded_v4r2_runner": paths["v4r2_runner"],
    }
    for name, path in implementations.items():
        if shared._sha(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    pins = shared._verify_input_pins(root, data_dir, config)
    metadata = shared._json(paths["feature_metadata"])
    if not (
        metadata["feature_columns"] == list(CAUSAL_FEATURE_COLUMNS)
        and metadata["builder_module_sha256"]
        == config["implementation_sha256"]["causal_feature_module"]
        and metadata["future_value_perturbation_invariant"] is True
        and metadata["target_columns_read"] == 0
        and metadata["forward_or_centered_operations"] == 0
    ):
        raise PermissionError("causal feature-cache audit metadata differs")
    ledger = _verify_v5_ledger_binding(root, config, paths["ledger"])
    pins["canonical_v5_ledger_semantic_binding"] = ledger
    pins["superseded_execution_guards"] = _verify_superseded_execution_guards(
        root, config, paths
    )
    return config, paths, pins


def _json_new(path: Path, value: Any) -> None:
    if isinstance(value, dict) and path.name == "preregistration.json":
        binding = shared._json(PROJECT_ROOT / CANONICAL_CONFIG)["v5_ledger_binding"]
        value = {
            **value,
            "supersedes_invalid_static_preregistration": {
                "path": "artifacts/p1_masked_pretrain_binary_event_v4r2/preregistration.json",
                "sha256": "ad9e654efcd520578f66d23816b1c6e1a35e200e391b812341dd2bdc665d21f2",
                "reason": "independent_static_QA_NO_GO_literal_runtime_prefix_target_read",
            },
            "superseded_execution_guards_verified": {
                "v4_tombstone_sha256": (
                    "28563cd076df5cc617545ccbc079c50df02cfc05bde746324127f8513bc6cda1"
                ),
                "v4r2_tombstone_sha256": (
                    "865edcb74421839aa1bad5871356dcb8f3412ef3abb9af12581bd27df955bd0d"
                ),
                "authorization_guards_are_first_statement": True,
                "predecessor_attempt_locks_absent": True,
            },
            "canonical_identity_verified_before_seal": True,
            "canonical_v5_ledger_binding": {
                key: binding[key]
                for key in (
                    "path",
                    "sha256",
                    "bytes",
                    "event_count",
                    "head_seq",
                    "head_event_sha256",
                    "semantic_upload_count",
                )
            },
            "causal_raw_feature_contract_verified_before_seal": True,
            "current_run_prefix_selector_contract": {
                "target_reads": 0,
                "columns": ["time", "row_position"],
                "exact_incumbent_row_count_and_sha_required": True,
                "historical_event_safe_lineage_disclosed": True,
                "nominal_boundary_split_risk_documented": True,
            },
        }
    elif isinstance(value, dict) and path.name == "predictions_complete.json":
        value = {
            **value,
            "pretrain_optimizer_steps": 1350,
            "finetune_optimizer_steps": 4050,
            "pretraining_label_reads": 0,
            "runtime_prefix_selector_target_reads": 0,
            "pretraining_target_tensor_materializations_before_pretraining": 0,
            "raw_prefix_causal_rebuilds": 15,
            "future_derived_feature_reads": 0,
        }
    elif isinstance(value, dict) and path.name == "learning_curve_evidence.json":
        leakage = {
            **value["leakage_checks"],
            "masked_pretraining_completed_before_prefix_label_target_construction": True,
            "current_run_prefix_selector_does_not_access_target_values": True,
            "current_run_fold_selector_does_not_access_target_values": True,
            "sealed_incumbent_prefix_row_counts_and_hashes_reproduced": True,
            "historical_event_safe_prefix_lineage_is_explicitly_disclosed": True,
            "nominal_cutoff_event_boundary_split_risk_is_documented": True,
            "masked_pretraining_uses_exact_prefix_features_only": True,
            "each_fit_uses_raw_prefix_only_causal_feature_rebuild": True,
            "future_value_perturbation_cannot_change_prefix_features": True,
            "forward_centered_and_terminal_run_features_excluded": True,
            "canonical_v5_ledger_head_and_zero_upload_semantics_verified": True,
            "exact_canonical_root_data_and_single_link_identity_verified": True,
            "reconstruction_and_auxiliary_heads_excluded_from_inference": True,
        }
        reproducibility = {
            **value["reproducibility_checks"],
            "fixed_1350_pretrain_and_4050_finetune_optimizer_steps": True,
            "all_15_raw_prefix_feature_rebuilds_match_pinned_causal_cache": True,
            "all_15_label_free_prefix_selections_match_pinned_incumbent_hashes": True,
            "canonical_v5_ledger_head_replayed_and_bound": True,
        }
        value = {
            **value,
            "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT",
            "leakage_checks": leakage,
            "reproducibility_checks": reproducibility,
        }
    elif isinstance(value, dict) and path.name == "result.json":
        value = {
            **value,
            "operation_counters": {
                **value["operation_counters"],
                "curve_pretrain_optimizer_steps": 1350,
                "curve_finetune_optimizer_steps": 4050,
                "pretraining_label_reads": 0,
                "runtime_prefix_selector_target_reads": 0,
                "pretraining_target_tensor_materializations_before_pretraining": 0,
                "raw_prefix_causal_rebuilds": 15,
                "future_derived_feature_reads": 0,
                "semantic_upload_count": 0,
            },
        }
    _SHARED_JSON_NEW(path, value)


def _deferred_prefix_labels(train: Any, train_ids: np.ndarray) -> tuple[Any, dict[str, bool]]:
    state = {"materialized": False}

    def load() -> np.ndarray:
        if state["materialized"]:
            raise RuntimeError("prefix labels were requested more than once")
        labels = np.full(len(train), -1, dtype=np.int8)
        labels[train_ids] = shared.pd.to_numeric(
            train.iloc[train_ids]["label"], errors="raise"
        ).to_numpy(np.int8)
        state["materialized"] = True
        return labels

    return load, state


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
    del root
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    receipts: list[dict[str, Any]] = []
    part_receipts: list[dict[str, Any]] = []
    completed = 0
    for fraction in FRACTIONS:
        for fold in folds:
            fold_name = str(fold["name"])
            train_ids = prefix_ids[(fold_name, fraction)]
            validation_ids = fold["val_idx"]
            comparator = shared._comparator_frame(
                comparator_parts[(fold_name, fraction)], fold, fraction
            )
            prefix_features, prefix_feature_sha = build_exact_prefix_causal_matrix(
                train,
                train_ids,
                full_reference=features,
            )
            scaler = shared.PrefixRobustScaler.fit(
                prefix_features, train_ids, forbidden_ids=validation_ids
            )
            seed_probabilities: list[np.ndarray] = []
            for seed in SEEDS:
                shared._emit(
                    "fit_cell_start",
                    completed_before=completed,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    train_rows=len(train_ids),
                    validation_rows=len(validation_ids),
                    pretrain_steps=training_config.pretrain_steps,
                    finetune_steps=training_config.finetune_steps,
                    prefix_labels_materialized=False,
                    raw_prefix_causal_feature_sha256=prefix_feature_sha,
                )
                started = time.perf_counter()
                deferred_labels, label_state = _deferred_prefix_labels(train, train_ids)
                fitted = fit_masked_pretrain_binary_event_model(
                    prefix_features,
                    train.loc[:, ["station", "layer", "time"]],
                    deferred_labels,
                    layout,
                    train_ids,
                    forbidden_ids=validation_ids,
                    seed=seed,
                    device="cuda",
                    model_config=model_config,
                    training_config=training_config,
                    scaler=scaler,
                )
                if not label_state["materialized"] or not fitted.labels_materialized_after_pretraining:
                    raise RuntimeError("prefix labels were not materialized after pretraining")
                probability = predict_masked_pretrain_binary_probability(
                    fitted,
                    features,
                    layout,
                    validation_ids,
                    device="cuda",
                    batch_size=4096,
                )
                model_relative = f"models/{shared._tag(fraction)}/{fold_name}/seed_{seed}.pt"
                model_path = shared._safe_path(paths["artifact"], model_relative)
                save_fitted_masked_pretrain_model(fitted, model_path)
                blind_relative = (
                    f"blind_predictions/{shared._tag(fraction)}/{fold_name}/seed_{seed}.npy"
                )
                blind_path = shared._safe_path(paths["artifact"], blind_relative)
                blind_sha = shared._npy_new(blind_path, probability)
                reloaded = load_fitted_masked_pretrain_model(model_path)
                reproduced = predict_masked_pretrain_binary_probability(
                    reloaded,
                    features,
                    layout,
                    validation_ids,
                    device="cuda",
                    batch_size=4096,
                )
                reload_exact = bool(np.array_equal(probability, reproduced))
                if not reload_exact:
                    raise RuntimeError("saved Gen4r3 model did not reproduce blind probabilities")
                seed_probabilities.append(probability)
                completed += 1
                receipts.append(
                    {
                        "fraction": fraction,
                        "fold": fold_name,
                        "seed": seed,
                        "train_rows": int(len(train_ids)),
                        "validation_rows": int(len(validation_ids)),
                        "optimizer_steps": training_config.optimizer_steps,
                        "pretrain_optimizer_steps": training_config.pretrain_steps,
                        "finetune_optimizer_steps": training_config.finetune_steps,
                        "pretraining_label_reads": 0,
                        "runtime_prefix_selector_target_reads": 0,
                        "pretraining_target_tensor_materializations_before_pretraining": 0,
                        "raw_prefix_causal_feature_sha256": prefix_feature_sha,
                        "raw_prefix_rebuild_exact_to_pinned_causal_cache": True,
                        "future_derived_feature_reads": 0,
                        "labels_materialized_after_pretraining": True,
                        "train_ids_sha256": fitted.train_ids_sha256,
                        "validation_ids_sha256": shared.ids_sha256(validation_ids),
                        "phase_counts": list(fitted.phase_counts),
                        "natural_priors": fitted.natural_priors.tolist(),
                        "sampling_priors": fitted.sampling_priors.tolist(),
                        "mean_training_loss": fitted.mean_training_loss,
                        "mean_pretrain_loss": fitted.mean_pretrain_loss,
                        "mean_finetune_loss": fitted.mean_finetune_loss,
                        "mean_event_loss": fitted.mean_event_loss,
                        "mean_auxiliary_loss": fitted.mean_auxiliary_loss,
                        "model_relative_path": model_relative,
                        "model_sha256": shared._sha(model_path),
                        "model_state_sha256": fitted.model_state_sha256,
                        "scaler_sha256": fitted.scaler.state_sha256,
                        "blind_prediction_relative_path": blind_relative,
                        "blind_prediction_sha256": blind_sha,
                        "blind_prediction_sealed_before_validation_target_read": True,
                        "saved_model_reload_prediction_exact": reload_exact,
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "validation_target_reads": 0,
                        "test_value_reads": 0,
                    }
                )
                shared._emit(
                    "fit_cell_complete",
                    completed=completed,
                    total=45,
                    fraction=fraction,
                    fold=fold_name,
                    seed=seed,
                    elapsed_seconds=receipts[-1]["elapsed_seconds"],
                )
            part = comparator.copy()
            for seed, probability in zip(SEEDS, seed_probabilities, strict=True):
                part[f"challenger__seed_{seed}__probability"] = probability
                part[f"challenger__seed_{seed}__prediction"] = shared.apply_postprocess(
                    train.iloc[validation_ids],
                    probability,
                    comparator["plateau"].to_numpy(bool),
                    comparator["spike_candidate"].to_numpy(bool),
                    config["fixed_fold_postprocess"][fold_name],
                )
            mean_probability = np.mean(np.column_stack(seed_probabilities), axis=1)
            part["challenger_probability"] = mean_probability.astype(np.float32)
            part["challenger_prediction"] = shared.apply_postprocess(
                train.iloc[validation_ids],
                mean_probability,
                comparator["plateau"].to_numpy(bool),
                comparator["spike_candidate"].to_numpy(bool),
                config["fixed_fold_postprocess"][fold_name],
            )
            part_relative = f"prediction_parts/{fold_name}_{shared._tag(fraction)}.parquet"
            part_path = shared._safe_path(paths["artifact"], part_relative)
            part_sha = shared._parquet_new(part_path, part)
            part_receipts.append(
                {
                    "fraction": fraction,
                    "fold": fold_name,
                    "rows": int(len(part)),
                    "path": part_relative,
                    "sha256": part_sha,
                    "key_order_sha256": hashlib.sha256(
                        shared.pd.util.hash_pandas_object(
                            part.loc[:, [*shared.KEY_COLUMNS, "fold"]], index=False
                        ).to_numpy("<u8").tobytes()
                    ).hexdigest(),
                }
            )
    total_steps = sum(row["optimizer_steps"] for row in receipts)
    total_pretrain = sum(row["pretrain_optimizer_steps"] for row in receipts)
    total_finetune = sum(row["finetune_optimizer_steps"] for row in receipts)
    if (completed, total_steps, total_pretrain, total_finetune) != (45, 5400, 1350, 4050):
        raise AssertionError("Gen4r3 fit-cell or stage optimizer-step count differs")
    completion = {
        "schema_version": "p1_masked_pretrain_predictions_complete.v4r3",
        "created_at": shared._now(),
        "fit_cells": completed,
        "optimizer_steps": total_steps,
        "pretrain_optimizer_steps": total_pretrain,
        "finetune_optimizer_steps": total_finetune,
        "pretraining_label_reads": 0,
        "runtime_prefix_selector_target_reads": 0,
        "pretraining_target_tensor_materializations_before_pretraining": 0,
        "raw_prefix_causal_rebuilds": 15,
        "all_raw_prefix_rebuilds_exact_to_pinned_causal_cache": True,
        "future_derived_feature_reads": 0,
        "prediction_parts": part_receipts,
        "model_receipts": receipts,
        "all_blind_predictions_sealed_before_validation_target_read": True,
        "aggregate_scores_computed_before_completion": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    shared._json_new(paths["artifact"] / "predictions_complete.json", completion)
    return receipts, completion


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
    rebuilt = build_causal_raw_features(train).to_numpy(np.float32)
    if not np.array_equal(rebuilt, features, equal_nan=True):
        raise PermissionError("full-fit pinned causal cache differs from raw rebuild")
    scaler = shared.PrefixRobustScaler.fit(features, full_ids)
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    models: list[dict[str, Any]] = []
    for seed in SEEDS:
        deferred_labels, label_state = _deferred_prefix_labels(train, full_ids)
        fitted = fit_masked_pretrain_binary_event_model(
            features,
            train.loc[:, ["station", "layer", "time"]],
            deferred_labels,
            layout,
            full_ids,
            forbidden_ids=None,
            seed=seed,
            device="cuda",
            model_config=model_config,
            training_config=training_config,
            scaler=scaler,
        )
        if not label_state["materialized"] or not fitted.labels_materialized_after_pretraining:
            raise RuntimeError("full-fit labels were not deferred until after pretraining")
        relative = f"full_fit/seed_{seed}.pt"
        path = shared._safe_path(paths["artifact"], relative)
        save_fitted_masked_pretrain_model(fitted, path)
        loaded = load_fitted_masked_pretrain_model(path)
        if loaded.model_state_sha256 != fitted.model_state_sha256:
            raise RuntimeError("full-fit Gen4r3 model state differs after reload")
        models.append(
            {
                "seed": seed,
                "path": relative,
                "sha256": shared._sha(path),
                "model_state_sha256": fitted.model_state_sha256,
                "scaler_sha256": fitted.scaler.state_sha256,
                "train_ids_sha256": fitted.train_ids_sha256,
                "optimizer_steps": training_config.optimizer_steps,
                "pretrain_optimizer_steps": training_config.pretrain_steps,
                "finetune_optimizer_steps": training_config.finetune_steps,
                "pretraining_label_reads": 0,
                "labels_materialized_after_pretraining": True,
            }
        )
    full_steps = sum(row["optimizer_steps"] for row in models)
    full_pretrain = sum(row["pretrain_optimizer_steps"] for row in models)
    full_finetune = sum(row["finetune_optimizer_steps"] for row in models)
    if (len(models), full_steps, full_pretrain, full_finetune) != (3, 360, 90, 270):
        raise AssertionError("Gen4r3 full-fit model or stage optimizer-step count differs")
    receipt = {
        "performed": True,
        "model_count": len(models),
        "optimizer_steps": full_steps,
        "pretrain_optimizer_steps": full_pretrain,
        "finetune_optimizer_steps": full_finetune,
        "pretraining_label_reads": 0,
        "models": models,
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
    shared._paths = _paths
    shared._json_new = _json_new
    shared.authorize_entry = authorize_entry
    shared._model_config = _model_config
    shared._training_config = _training_config
    shared._prefixes = _pinned_label_free_prefixes
    shared._run_curve = _run_curve
    shared._full_fit_models = _full_fit_models
    shared.TemporalEventModelConfig = MaskedPretrainModelConfig
    shared.FixedStepTrainingConfig = MaskedPretrainTrainingConfig
    shared.fit_fixed_step_temporal_event_model = fit_masked_pretrain_binary_event_model
    shared.predict_temporal_event_probability = predict_masked_pretrain_binary_probability
    shared.save_fitted_temporal_event_model = save_fitted_masked_pretrain_model
    shared.load_fitted_temporal_event_model = load_fitted_masked_pretrain_model
    shared.evaluate_learning_curve = evaluate_learning_curve
    shared.load_contract = load_contract


_patch_shared_engine()


def _causal_cache_audit(*, data_dir: Path, paths: dict[str, Path]) -> dict[str, Any]:
    raw = shared.pd.read_csv(
        data_dir / "train.csv",
        usecols=["station", "layer", "time", "temp", "psal", "depth"],
        low_memory=False,
    )
    rebuilt = build_causal_raw_features(raw)
    cached = shared.pd.read_parquet(paths["feature_cache"], columns=list(CAUSAL_FEATURE_COLUMNS))
    if not rebuilt.equals(cached):
        left = rebuilt.to_numpy(np.float32)
        right = cached.to_numpy(np.float32)
        if not np.array_equal(left, right, equal_nan=True):
            raise PermissionError("pinned causal cache differs from current raw-only rebuild")
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
        "raw_feature_columns_read": ["station", "layer", "time", "temp", "psal", "depth"],
        "target_columns_read": 0,
        "feature_count": len(CAUSAL_FEATURE_COLUMNS),
        "cache_exact_to_raw_rebuild": True,
        "future_value_perturbation_invariant": True,
        "future_value_perturbation_prefix_sha256": invariance_sha,
    }


def _prefix_selector_audit(
    *, data_dir: Path, paths: dict[str, Path], config: dict[str, Any]
) -> dict[str, Any]:
    """Reproduce all exact comparator prefixes without loading a target column."""

    train_keys = shared.pd.read_csv(
        data_dir / "train.csv",
        usecols=["station", "year", "layer", "time"],
        low_memory=False,
    )
    if {"label", "anomaly_type"}.intersection(train_keys.columns):
        raise PermissionError("static prefix selector loaded a target column")
    frozen = shared.pd.read_parquet(
        paths["frozen_oof"], columns=[*shared.KEY_COLUMNS, "fold"]
    )
    folds, _ = shared.gen1._fold_runtime(
        train_keys,
        shared.load_config(paths["base_config"]),
        frozen,
    )
    prefix_ids, audit = _pinned_label_free_prefixes(
        object(), folds, int(config["features"]["cadence_minutes"])
    )
    cells = [row for fraction in audit.values() for row in fraction.values()]
    retreat_cells = sum(row["boundary_split_risk_if_nominal_cutoff_used"] for row in cells)
    if (
        len(prefix_ids) != 15
        or len(cells) != 15
        or retreat_cells
        != int(config["prefix_protocol"]["expected_historical_boundary_retreat_cells"])
        or any(row["current_run_prefix_selector_target_reads"] != 0 for row in cells)
        or not all(row["exact_to_immutable_incumbent_fold_train_ids"] for row in cells)
    ):
        raise PermissionError("label-free exact-prefix selector audit differs")
    return {
        "selector": config["prefix_protocol"]["selector"],
        "source_columns_read": ["station", "year", "layer", "time"],
        "target_columns_read": 0,
        "fold_selector_target_reads": 0,
        "cell_count": len(cells),
        "exact_incumbent_prefix_row_count_and_sha_cells": len(cells),
        "historical_event_boundary_retreat_cells": int(retreat_cells),
        "historical_event_safe_cutoff_lineage_disclosed": True,
        "nominal_event_boundary_split_risk_documented": True,
        "selector_ignores_train_argument": True,
        "cell_id_hashes": {
            f"{row['fraction']:.2f}:{fold}": row["id_sha256_little_endian_int64"]
            for fraction_rows in audit.values()
            for fold, row in fraction_rows.items()
        },
    }


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    result = shared.check_only(root=root, data_dir=data_dir)
    config = json.loads((root / CANONICAL_CONFIG).read_text(encoding="utf-8"))
    paths = _paths(root)
    audit = _causal_cache_audit(data_dir=data_dir, paths=paths)
    prefix_audit = _prefix_selector_audit(
        data_dir=data_dir, paths=paths, config=config
    )
    return {
        **result,
        "experiment_id": config["experiment_id"],
        "comparison_mode": config["comparison_mode"],
        "pretrain_steps": config["training"]["pretrain_steps_per_cell"],
        "finetune_steps": config["training"]["finetune_steps_per_cell"],
        "pretraining_label_reads": 0,
        "runtime_prefix_selector_target_reads": 0,
        "pretraining_target_tensor_materializations_before_pretraining": 0,
        "event_probability_only": True,
        "reconstruction_or_auxiliary_probability_use": False,
        "batch_size": config["training"]["batch_size"],
        "canonical_root_exact": True,
        "canonical_data_dir_exact": True,
        "clone_reparse_hardlink_rejected": True,
        "v5_ledger_event_count": config["v5_ledger_binding"]["event_count"],
        "v5_ledger_head_event_sha256": config["v5_ledger_binding"][
            "head_event_sha256"
        ],
        "v5_ledger_semantic_upload_count": 0,
        "superseded_v4_v4r2_execution_refusals_verified": True,
        "causal_feature_audit": audit,
        "prefix_selector_audit": prefix_audit,
    }


def seal(*, root: Path, data_dir: Path) -> dict[str, Any]:
    root, data_dir = _assert_canonical_identity(root, data_dir)
    paths = _paths(root)
    config = json.loads((root / CANONICAL_CONFIG).read_text(encoding="utf-8"))
    audit = _causal_cache_audit(data_dir=data_dir, paths=paths)
    prefix_audit = _prefix_selector_audit(
        data_dir=data_dir, paths=paths, config=config
    )
    result = shared.seal(root=root, data_dir=data_dir)
    receipt_path = paths["artifact"] / "preseal_static_qa.json"
    receipt = {
        "schema_version": "p1_masked_pretrain_preseal_static_qa.v4r3",
        "created_before_first_fit": True,
        "canonical_root": str(CANONICAL_ROOT_PATH),
        "canonical_data_dir": str(CANONICAL_DATA_DIR_PATH),
        "clone_reparse_hardlink_rejected": True,
        "causal_feature_audit": audit,
        "prefix_selector_audit": prefix_audit,
        "superseded_execution_guards": _verify_superseded_execution_guards(
            root, config, paths
        ),
        "v5_ledger_binding": json.loads(
            (root / CANONICAL_CONFIG).read_text(encoding="utf-8")
        )["v5_ledger_binding"],
        "operation_counters": {
            "curve_model_fits": 0,
            "target_fold_scores": 0,
            "runtime_prefix_selector_target_reads": 0,
            "pretraining_target_tensor_materializations": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
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
