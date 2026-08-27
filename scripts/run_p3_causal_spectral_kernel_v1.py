"""Run the append-only P3 causal spectral-kernel learning curve exactly once.

The runner is train/OOF-only. It never opens anonymous-test values and cannot
create a submission because the inherited historical-frozen OOF equality check
is known false and therefore forces the central gate to fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ocean_goal.meaningful_score_ledger_v5 import validate_ledger
from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p3_wave.causal_spectral_kernel import (
    CausalSpectralKernelConfig,
    TrainOnlyRobustScaler,
    build_causal_spectral_features,
    feature_names_sha256,
    fit_and_predict_causal_spectral_kernel,
    load_fitted_causal_spectral_kernel,
    predict_causal_spectral_kernel,
    save_fitted_causal_spectral_kernel,
)
from p3_wave.corrected_repeated_forward import (
    OFFICIAL_LEADS,
    CorrectedFold,
    build_corrected_repeated_forward_folds,
)
from p3_wave.meaningful_learning_curve import (
    PREFIX_FRACTIONS,
    central_evidence,
    chronological_prefix_ids,
    evaluate_hypothesis_gate,
    evaluate_point,
)
from p3_wave.models import threshold_case_weights
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock, safe_new_stage_path
from p3_wave.revin_patch import assign_storm_episodes_from_wave, validate_raw_context
from p3_wave.validation import rmse

_GEN2_RUNNER_PATH = Path(__file__).with_name("run_p3_causal_forcing_sequence_residual_v1.py")
_GEN2_SPEC = importlib.util.spec_from_file_location("p3_gen3_gen2_helpers", _GEN2_RUNNER_PATH)
if _GEN2_SPEC is None or _GEN2_SPEC.loader is None:
    raise ImportError("failed to load pinned Gen2 runner helpers")
gen2 = importlib.util.module_from_spec(_GEN2_SPEC)
sys.modules[_GEN2_SPEC.name] = gen2
_GEN2_SPEC.loader.exec_module(gen2)

EXPECTED_CONFIG_SHA256 = "3bf5aaa053a702f46f34ab696b7876bdf954845dbea275279c58dc0c204e3f1b"
EXPECTED_CONFIG_DEEP_SHA256 = "3b92f8dfca516dd9cb350e60e08b846d7c5deb7399f97f9628fc7aae7f8bc851"
GEN2_CONFIG_SHA256 = "f9b7b0eb76ca0d152a0e87f4eb4fc30b3d2a1cc929e6697cd1324cd9a59e84ca"
GEN2_CONFIG_DEEP_SHA256 = "c71d8743d66bea605017cd88fcdaef769b7bd5e0d5d19ae8c49c4f7fb8b9168e"
CANONICAL_CONFIG_RELATIVE = "configs/experiments/p3_causal_spectral_kernel_v1.json"
CANONICAL_GEN2_CONFIG_RELATIVE = "configs/experiments/p3_causal_forcing_sequence_residual_v1.json"
CANONICAL_GOAL_RELATIVE = "configs/goals/meaningful_score_maximization_v3.json"
CANONICAL_COMPACT_CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
CANONICAL_SEQUENCE_CACHE_RELATIVE = "artifacts/p3/sequences_all20_v1"
CANONICAL_GEN1_RELATIVE = "artifacts/p3_meaningful_learning_curve_20260823_v1"
CANONICAL_GEN2_RELATIVE = "artifacts/p3_causal_forcing_sequence_residual_20260823_v1"
CANONICAL_V5_LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v5/registry.jsonl"
CANONICAL_OUTPUT_RELATIVE = "artifacts/p3_causal_spectral_kernel_20260823_v1"
CANONICAL_LOCK_RELATIVE = "artifacts/p3_causal_spectral_kernel_20260823_v1.ATTEMPT_LOCK.json"
CANONICAL_EXECUTION_CLAIM_RELATIVE = (
    "artifacts/p3_causal_spectral_kernel_20260823_v1.EXECUTION_CLAIM.json"
)
CANONICAL_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATA_DIR = Path(r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast")
HYPOTHESIS = "causal_multiresolution_spectral_rff_kernel"
TARGET_COLUMNS = tuple(f"target_{lead}" for lead in OFFICIAL_LEADS)


def _now() -> str:
    return gen2._now()


def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_paths(root: Path) -> dict[str, Path]:
    workspace = root.resolve(strict=True)
    return {
        "config": workspace / CANONICAL_CONFIG_RELATIVE,
        "gen2_config": workspace / CANONICAL_GEN2_CONFIG_RELATIVE,
        "goal": workspace / CANONICAL_GOAL_RELATIVE,
        "compact_cache": workspace / CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": workspace / CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "gen1": workspace / CANONICAL_GEN1_RELATIVE,
        "gen2": workspace / CANONICAL_GEN2_RELATIVE,
        "v5_ledger": workspace / CANONICAL_V5_LEDGER_RELATIVE,
        "output": workspace / CANONICAL_OUTPUT_RELATIVE,
        "lock": workspace / CANONICAL_LOCK_RELATIVE,
        "claim": workspace / CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }


def _implementation_paths(root: Path, paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "gen2_runner": _GEN2_RUNNER_PATH,
        "gen2_config": paths["gen2_config"],
        "spectral_kernel_module": root / "src/p3_wave/causal_spectral_kernel.py",
        "corrected_split_module": root / "src/p3_wave/corrected_repeated_forward.py",
        "learning_curve_module": root / "src/p3_wave/meaningful_learning_curve.py",
        "persistence_shrink_module": root / "src/p3_wave/persistence_shrink.py",
        "one_shot_guard_module": root / "src/p3_wave/one_shot_guard.py",
        "goal_contract": paths["goal"],
        "goal_evaluator": root / "src/ocean_goal/meaningful_score_v3.py",
        "v5_ledger_contract": root / "configs/goals/meaningful_score_ledger_v5.json",
        "v5_ledger_evaluator": root / "src/ocean_goal/meaningful_score_ledger_v5.py",
    }


def _reference_paths(root: Path, paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "gen1_metrics": paths["gen1"] / "metrics.json",
        "gen1_learning_curve_oof": paths["gen1"] / "oof/learning_curve_oof.parquet",
        "gen1_manifest": paths["gen1"] / "manifest.json",
        "gen1_independent_qa": root
        / "artifacts/p3_meaningful_learning_curve_20260823_v1_QA/independent_aggregate_audit.json",
        "gen2_metrics": paths["gen2"] / "metrics.json",
        "gen2_learning_curve_evidence": paths["gen2"] / "learning_curve_evidence.json",
        "gen2_manifest": paths["gen2"] / "manifest.json",
        "gen2_validation_keys": paths["gen2"] / "validation_keys.parquet",
    }


def _verify_v5_anchor(root: Path, paths: dict[str, Path], config: dict[str, Any]) -> None:
    records = validate_ledger(root, paths["v5_ledger"])
    expected = config["central_ledger_anchor"]
    matching = [record for record in records if record.get("seq") == expected["gen2_event_seq"]]
    if len(matching) != 1:
        raise PermissionError("canonical v5 Gen2 event is absent or duplicated")
    event = matching[0]
    if event.get("event_sha256") != expected["gen2_event_sha256"]:
        raise PermissionError("canonical v5 Gen2 event SHA differs")
    payload = event.get("payload", {})
    if (
        payload.get("evidence", {}).get("sha256") != expected["gen2_evidence_sha256"]
        or payload.get("decision", {}).get("decision") != expected["gen2_decision"]
        or payload.get("upload_performed") is not False
    ):
        raise PermissionError("canonical v5 Gen2 event semantics differ")


def authorize_entry(
    *,
    root: Path,
    data_dir: Path,
    requested_config: Path,
    requested_output: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    """First action: bind canonical paths, bytes, deep JSON, and immutable pins."""

    if root.resolve(strict=True) != CANONICAL_WORKSPACE_ROOT:
        raise PermissionError("non-canonical workspace root is forbidden")
    paths = _canonical_paths(root)
    if data_dir.resolve(strict=True) != CANONICAL_DATA_DIR.resolve(strict=True):
        raise PermissionError("non-canonical P3 data directory is forbidden")
    if requested_config.resolve(strict=True) != paths["config"].resolve(strict=True):
        raise PermissionError("non-canonical config path is forbidden")
    if requested_output.resolve(strict=False) != paths["output"].resolve(strict=False):
        raise PermissionError("non-canonical output path is forbidden")
    if paths["output"].exists():
        raise FileExistsError("canonical append-only output already exists")

    content = paths["config"].read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_CONFIG_SHA256:
        raise PermissionError("canonical config byte SHA differs")
    config = json.loads(content)
    if _deep_sha(config) != EXPECTED_CONFIG_DEEP_SHA256:
        raise PermissionError("canonical config fails compiled deep-JSON equality")
    expected_paths = {
        "config": CANONICAL_CONFIG_RELATIVE,
        "goal_contract": CANONICAL_GOAL_RELATIVE,
        "compact_cache": CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "gen1_artifact": CANONICAL_GEN1_RELATIVE,
        "gen2_artifact": CANONICAL_GEN2_RELATIVE,
        "v5_ledger": CANONICAL_V5_LEDGER_RELATIVE,
        "output": CANONICAL_OUTPUT_RELATIVE,
        "attempt_lock": CANONICAL_LOCK_RELATIVE,
        "execution_claim": CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }
    if config.get("canonical_paths") != expected_paths:
        raise PermissionError("canonical path fields differ")
    if config.get("experiment_id") != "p3_causal_spectral_kernel_v1":
        raise PermissionError("experiment identity differs")
    if config.get("created_before_first_fit") is not True:
        raise PermissionError("preregistration timing declaration differs")
    if tuple(item["id"] for item in config["hypotheses"]) != (HYPOTHESIS,):
        raise PermissionError("single structural hypothesis differs")
    if config["validation"]["training_prefix_fractions"] != list(PREFIX_FRACTIONS):
        raise PermissionError("prefix curve differs")
    if config["model"] != {
        "class": "CausalSpectralRandomFourierKernelRidge",
        "random_feature_count": 128,
        "frequency_pair_count": 64,
        "frequency_distribution": "Normal(0,1/sqrt(train_only_median_squared_distance))",
        "kernel_bandwidth_rule": "median positive pairwise squared distance on 256 deterministic evenly spaced prefix-train rows only",
        "paired_basis": "cosine_and_sine_for_each_frequency_no_random_phase",
        "ridge_penalty": 1.0,
        "intercept_penalty": 1e-10,
        "solver": "float64_closed_form_linear_solve",
        "multi_output_leads_h": list(OFFICIAL_LEADS),
        "seed_replicates": [20260816, 20260817, 20260818],
        "seed_reducer": "arithmetic_mean_prediction_within_fold_prefix",
        "residual_target": "target_hs_minus_current_hs",
        "case_weight": "same_fixed_current_hs_threshold_weight_as_comparator_generation",
        "hyperparameter_search": False,
        "expected_fit_cells": 45,
        "expected_closed_form_solves": 45,
    }:
        raise PermissionError("frozen spectral-kernel model contract differs")
    if config["postprocess"] != {
        "clip_m": [0.0, 30.0],
        "fixed_persistence_shrink_active_leads_h": [12, 18, 24],
        "fixed_persistence_weight": 0.2,
        "identical_to_gen1_comparator": True,
        "new_router_or_blend": False,
    }:
        raise PermissionError("fixed postprocess differs")
    if not all(config["prohibitions"].values()):
        raise PermissionError("all prohibitions must remain enabled")

    for name, path in _implementation_paths(root, paths).items():
        if gen2.gen1.base.sha256_file(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    for name, path in _reference_paths(root, paths).items():
        if gen2.gen1.base.sha256_file(path) != config["reference_evidence_sha256"][name]:
            raise PermissionError(f"reference evidence SHA differs: {name}")
    _verify_v5_anchor(root, paths, config)
    return config, paths


def _input_paths(root: Path, data_dir: Path, paths: dict[str, Path]) -> dict[str, Path]:
    gen2_paths = gen2._canonical_paths(root)
    return gen2._resolved_input_paths(root=root, data_dir=data_dir, paths=gen2_paths)


def _prefix_id_sha(ids: np.ndarray) -> str:
    return gen2._prefix_id_sha(ids)


def _preflight(
    *, root: Path, data_dir: Path, config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    inputs = _input_paths(root, data_dir, paths)
    snapshot = gen2._verify_input_hashes(inputs, config["expected_sha256"])

    gen2_config_bytes = paths["gen2_config"].read_bytes()
    if hashlib.sha256(gen2_config_bytes).hexdigest() != GEN2_CONFIG_SHA256:
        raise PermissionError("pinned Gen2 config byte SHA differs")
    gen2_config = json.loads(gen2_config_bytes)
    if _deep_sha(gen2_config) != GEN2_CONFIG_DEEP_SHA256:
        raise PermissionError("pinned Gen2 config deep JSON differs")

    anchor_path = paths["compact_cache"] / "train_anchors.parquet"
    anchors = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", "station", "anchor_time", "current_hs"],
    )
    if len(anchors) != 24_360:
        raise ValueError("training anchor row contract differs")
    expected_ids = np.arange(len(anchors), dtype=np.int64)
    if not np.array_equal(anchors["anchor_id"].to_numpy(np.int64), expected_ids):
        raise ValueError("anchor_id must be exact sequence-cache row index")
    if not np.isfinite(anchors["current_hs"].to_numpy(float)).all():
        raise ValueError("current observed hs is non-finite")

    raw = np.load(paths["sequence_cache"] / "train_values.npy", mmap_mode="r")
    station = np.load(paths["sequence_cache"] / "train_station.npy", mmap_mode="r")
    if raw.shape != (24_360, 289, 10) or raw.dtype != np.float32:
        raise ValueError("train raw-sequence contract differs")
    if station.shape != (24_360,) or station.dtype != np.int64:
        raise ValueError("train station-code contract differs")
    validate_raw_context(torch.from_numpy(np.array(raw[:8], copy=True)))
    encoded_station = anchors["station"].map({"G-ORS": 0, "I-ORS": 1, "S-ORS": 2})
    if not np.array_equal(station, encoded_station.to_numpy(np.int64)):
        raise ValueError("sequence station codes differ from anchor keys")

    wave = pd.read_csv(data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["validation"]["windows"],
        gap_hours=config["validation"]["gap_hours"],
        footprint_hours=config["validation"]["footprint_hours"],
    )
    if len(selected) != 181 or split_audit["validation_row_count"] != 1_086:
        raise ValueError("corrected validation surface differs")

    prefix_ids: dict[float, dict[str, np.ndarray]] = {}
    prefix_audit: dict[str, Any] = {}
    anchor_lookup = anchors.set_index("anchor_id")
    for fraction in PREFIX_FRACTIONS:
        prefix_ids[fraction] = {}
        tag = f"{int(round(fraction * 100)):03d}"
        prefix_audit[tag] = {}
        for fold in folds:
            ids = chronological_prefix_ids(anchors, fold.train_ids, fraction)
            prefix_ids[fraction][fold.name] = ids
            times = pd.to_datetime(anchor_lookup.loc[ids, "anchor_time"], utc=True)
            gap = float(
                (pd.Timestamp(fold.validation_start) - times.max()).total_seconds() / 3600.0
            )
            prefix_audit[tag][fold.name] = {
                "fraction": float(fraction),
                "count": int(len(ids)),
                "full_count": int(len(fold.train_ids)),
                "id_sha256_little_endian_int64": _prefix_id_sha(ids),
                "nested_subset_of_safe_outer_train": bool(np.isin(ids, fold.train_ids).all()),
                "maximum_anchor_before_validation_start_hours": gap,
            }
    leakage_checks = {
        "station_global_validation_gap_at_least_78h": all(
            value >= 78.0 for value in split_audit["station_global_minimum_gap_hours"].values()
        ),
        "validation_station_episode_reuse_zero": split_audit["repeated_station_episode_count"] == 0,
        "validation_72h_footprint_overlap_zero": split_audit[
            "context48_plus_target24_footprint_overlap_pairs"
        ]
        == 0,
        "outer_train_validation_episode_overlap_zero": all(
            row["shared_train_validation_station_episode_count"] == 0
            for row in split_audit["folds"].values()
        ),
        "outer_train_validation_gap_at_least_78h": all(
            row["minimum_train_validation_anchor_gap_hours"] >= 78.0
            for row in split_audit["folds"].values()
        ),
        "all_prefixes_nested_in_safe_outer_train": all(
            row["nested_subset_of_safe_outer_train"]
            for current in prefix_audit.values()
            for row in current.values()
        ),
    }
    if not all(leakage_checks.values()):
        raise AssertionError("preflight leakage checks failed")

    comparator_keys = pd.read_parquet(
        paths["gen1"] / "oof/learning_curve_oof.parquet",
        columns=[
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "current_hs",
            "persistence",
            "incumbent_prediction",
            "prefix_fraction",
        ],
    )
    keys = ["prefix_fraction", "fold", "anchor_id", "station", "lead_h"]
    if len(comparator_keys) != 5 * 1_086 or comparator_keys.duplicated(keys).any():
        raise ValueError("sealed Gen1 comparator key contract differs")
    for fraction in PREFIX_FRACTIONS:
        current = comparator_keys.loc[comparator_keys["prefix_fraction"].eq(fraction)]
        if len(current) != 1_086 or current["anchor_id"].nunique() != 181:
            raise ValueError("sealed Gen1 prefix surface differs")

    gen1_metrics = json.loads((paths["gen1"] / "metrics.json").read_text(encoding="utf-8"))
    gen1_protocol = json.loads(
        (paths["gen1"] / "learning_curve_evidence.json").read_text(encoding="utf-8")
    )["curve_protocol"]
    if gen1_protocol["incumbent_reference_seed_full_prediction_exact_to_frozen_oof"] is not False:
        raise ValueError("Gen1 inherited fail-close reference fact unexpectedly changed")

    gen2_metrics = json.loads((paths["gen2"] / "metrics.json").read_text(encoding="utf-8"))
    gen2_full = gen2_metrics["points"]["1.0"]
    diagnosis = config["gen2_failure_diagnosis"]
    if (
        gen2_metrics["central_goal_evaluator"]["decision"] != "RESEARCH_ONLY"
        or float(gen2_full["delta_candidate_minus_incumbent_m"])
        != diagnosis["full_delta_candidate_minus_incumbent_m"]
        or list(gen2_full["delta_ci90_m"]) != diagnosis["full_ci90_m"]
    ):
        raise ValueError("sealed Gen2 aggregate diagnosis differs")

    spectral_started = time.perf_counter()
    spectral_features, feature_names = build_causal_spectral_features(raw, station)
    spectral_elapsed = time.perf_counter() - spectral_started
    return {
        "inputs": inputs,
        "snapshot": snapshot,
        "anchor_path": anchor_path,
        "anchors": anchors,
        "raw": raw,
        "station": station,
        "spectral_features": spectral_features,
        "feature_names": feature_names,
        "feature_names_sha256": feature_names_sha256(feature_names),
        "spectral_feature_elapsed_seconds": float(spectral_elapsed),
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
        "prefix_ids": prefix_ids,
        "prefix_audit": prefix_audit,
        "leakage_checks": leakage_checks,
        "gen1_metrics": gen1_metrics,
        "gen1_protocol": gen1_protocol,
    }


def _load_train_targets(
    anchor_path: Path,
    anchors: pd.DataFrame,
    train_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return only the current cell's allowed training labels and label-free weights."""

    ids = np.asarray(train_ids, dtype=np.int64)
    frame = pd.read_parquet(
        anchor_path,
        columns=["anchor_id", *TARGET_COLUMNS],
        filters=[("anchor_id", "in", [int(value) for value in ids])],
    )
    if len(frame) != len(ids) or frame["anchor_id"].duplicated().any():
        raise ValueError("train-only target filter returned wrong rows")
    ordered = frame.set_index("anchor_id").loc[ids]
    target = ordered.loc[:, TARGET_COLUMNS].to_numpy(np.float64)
    current = anchors.set_index("anchor_id").loc[ids, "current_hs"].to_numpy(np.float64)
    delta = target - current[:, None]
    weights = threshold_case_weights(current).astype(np.float64)
    if delta.shape != (len(ids), 6) or not np.isfinite(delta).all():
        raise ValueError("train-only target payload differs")
    return delta, weights


def _load_comparator_truth_after_blind(paths: dict[str, Path]) -> pd.DataFrame:
    frame = pd.read_parquet(paths["gen1"] / "oof/learning_curve_oof.parquet")
    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "current_hs",
        "persistence",
        "incumbent_prediction",
        "prefix_fraction",
    }
    if not required.issubset(frame.columns) or len(frame) != 5 * 1_086:
        raise ValueError("sealed Gen1 comparator truth contract differs")
    return frame


def _protected_roots(root: Path, data_dir: Path, paths: dict[str, Path]) -> tuple[Path, ...]:
    return (
        data_dir,
        paths["compact_cache"],
        paths["sequence_cache"],
        paths["gen1"],
        paths["gen2"],
        root / "submissions",
        root / "output",
        root / "데이터셋 원본",
    )


def _write_npy_exclusive(path: Path, values: np.ndarray) -> str:
    return gen2._write_npy_exclusive(path, values)


def _postprocess(delta: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    return gen2._postprocess(delta, current_hs)


def _run_curve(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    preflight: dict[str, Any],
    stage: Path,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], list[dict[str, Any]]]:
    features = preflight["spectral_features"]
    names = preflight["feature_names"]
    anchors = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    protected = _protected_roots(root, data_dir, paths)
    model_config = CausalSpectralKernelConfig()
    anchor_lookup = anchors.set_index("anchor_id")
    blind_predictions: dict[tuple[float, int, str], np.ndarray] = {}
    receipts: list[dict[str, Any]] = []
    completed = 0

    for fraction in PREFIX_FRACTIONS:
        prefix_tag = f"p{int(round(fraction * 100)):03d}"
        prefix_scalers: dict[str, TrainOnlyRobustScaler] = {}
        train_payloads: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for fold in folds:
            train_ids = preflight["prefix_ids"][fraction][fold.name]
            prefix_scalers[fold.name] = TrainOnlyRobustScaler.fit(
                features, train_ids, forbidden_ids=fold.validation_ids
            )
            train_payloads[fold.name] = _load_train_targets(
                preflight["anchor_path"], anchors, train_ids
            )
        for seed in config["model"]["seed_replicates"]:
            for fold in folds:
                train_ids = preflight["prefix_ids"][fraction][fold.name]
                validation_ids = fold.validation_ids
                if np.intersect1d(train_ids, validation_ids).size:
                    raise AssertionError("train/validation IDs overlap before fit")
                train_target, train_weight = train_payloads[fold.name]
                current_hs = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(np.float64)
                print(
                    json.dumps(
                        {
                            "phase": "fit_spectral_kernel_cell",
                            "completed_before": completed,
                            "total": 45,
                            "prefix": fraction,
                            "seed": seed,
                            "fold": fold.name,
                            "train_cases": len(train_ids),
                            "validation_target_values_read_by_model": 0,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                started = time.perf_counter()
                prediction_delta, fitted = fit_and_predict_causal_spectral_kernel(
                    features,
                    train_target,
                    train_weight,
                    train_ids,
                    validation_ids,
                    seed=int(seed),
                    config=model_config,
                    scaler=prefix_scalers[fold.name],
                    forbidden_ids=validation_ids,
                    names=names,
                )
                prediction = _postprocess(prediction_delta, current_hs)
                model_relative = f"models/{prefix_tag}/seed_{seed}/folds/{fold.name}/model.npz"
                model_path = safe_new_stage_path(stage, model_relative, protected_roots=protected)
                save_fitted_causal_spectral_kernel(fitted, model_path)
                model_sha = gen2.gen1.base.sha256_file(model_path)
                blind_relative = f"blind_predictions/{prefix_tag}/seed_{seed}/{fold.name}.npy"
                blind_path = safe_new_stage_path(stage, blind_relative, protected_roots=protected)
                blind_sha = _write_npy_exclusive(blind_path, prediction.astype(np.float64))

                reloaded = load_fitted_causal_spectral_kernel(model_path)
                reload_delta = predict_causal_spectral_kernel(
                    reloaded, features, validation_ids, names=names
                )
                reload_prediction = _postprocess(reload_delta, current_hs)
                reload_exact = bool(np.array_equal(reload_prediction, prediction))
                maximum_reload_difference = float(np.max(np.abs(reload_prediction - prediction)))
                if not reload_exact:
                    raise RuntimeError("saved kernel reload failed exact reproduction")
                blind_predictions[(fraction, int(seed), fold.name)] = prediction
                completed += 1
                receipts.append(
                    {
                        "prefix_fraction": float(fraction),
                        "seed": int(seed),
                        "fold": fold.name,
                        "train_cases": int(len(train_ids)),
                        "validation_cases": int(len(validation_ids)),
                        "closed_form_solves": 1,
                        "train_id_sha256": _prefix_id_sha(train_ids),
                        "validation_id_sha256": _prefix_id_sha(validation_ids),
                        "scaler_fit_id_sha256": fitted.scaler.fit_ids_sha256,
                        "feature_names_sha256": fitted.feature_names_sha256,
                        "train_only_median_squared_distance": float(fitted.median_squared_distance),
                        "model_relative_path": model_relative,
                        "model_sha256": model_sha,
                        "blind_prediction_relative_path": blind_relative,
                        "blind_prediction_sha256": blind_sha,
                        "blind_prediction_sealed_before_validation_truth_attachment": True,
                        "saved_model_reload_prediction_exact": reload_exact,
                        "saved_model_reload_max_abs_difference_m": maximum_reload_difference,
                        "elapsed_seconds": float(time.perf_counter() - started),
                        "test_or_hidden_value_reads": 0,
                    }
                )

    if completed != 45 or sum(row["closed_form_solves"] for row in receipts) != 45:
        raise AssertionError("fit-cell or closed-form-solve count differs")
    comparator_truth = _load_comparator_truth_after_blind(paths)
    points: dict[float, dict[str, Any]] = {}
    all_frames: list[pd.DataFrame] = []
    for fraction in PREFIX_FRACTIONS:
        seed_frames: list[pd.DataFrame] = []
        for seed in config["model"]["seed_replicates"]:
            fold_frames: list[pd.DataFrame] = []
            for fold in folds:
                comparator = gen2._cell_comparator(comparator_truth, fraction=fraction, fold=fold)
                comparator["challenger_prediction"] = blind_predictions[
                    (fraction, int(seed), fold.name)
                ].reshape(-1)
                fold_frames.append(comparator)
            seed_frames.append(pd.concat(fold_frames, ignore_index=True))
        keys = ["fold", "anchor_id", "station", "lead_h"]
        ordered = [frame.sort_values(keys).reset_index(drop=True) for frame in seed_frames]
        invariant = ["target_hs", "current_hs", "persistence", "incumbent_prediction"]
        for frame in ordered[1:]:
            if not frame[keys].equals(ordered[0][keys]):
                raise ValueError("seed OOF keys differ")
            if not np.array_equal(
                frame[invariant].to_numpy(float), ordered[0][invariant].to_numpy(float)
            ):
                raise ValueError("seed OOF invariant values differ")
        mean_frame = ordered[0][keys + invariant].copy()
        mean_frame["challenger_prediction"] = np.mean(
            np.column_stack([frame["challenger_prediction"].to_numpy(float) for frame in ordered]),
            axis=1,
        )
        mean_frame["prefix_fraction"] = float(fraction)
        all_frames.append(mean_frame)
        point = evaluate_point(
            mean_frame,
            candidate_column="challenger_prediction",
            bootstrap_replicates=config["validation"]["bootstrap_replicates"],
            bootstrap_seed=int(config["validation"]["bootstrap_seed"]) + int(round(fraction * 100)),
        )
        gen1_point = preflight["gen1_metrics"]["points_by_hypothesis"]["fixed_horizon_splice"][
            str(fraction)
        ]
        point["incumbent_seed_metrics"] = [
            float(value) for value in gen1_point["incumbent_seed_metrics"]
        ]
        point["challenger_seed_metrics"] = [
            float(rmse(frame["target_hs"], frame["challenger_prediction"])) for frame in ordered
        ]
        points[fraction] = point
        print(
            json.dumps(
                {
                    "phase": "prefix_scored_after_all_45_blind_predictions_sealed",
                    "prefix": fraction,
                    "completed_cells": completed,
                    "delta_candidate_minus_incumbent_m": point["delta_candidate_minus_incumbent_m"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return pd.concat(all_frames, ignore_index=True), points, receipts


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    return gen2._artifact_hashes(stage)


def _verify_persisted_attempt(
    *, paths: dict[str, Path], config: dict[str, Any], attempt: dict[str, Any]
) -> None:
    expected_keys = {
        "created_at",
        "status",
        "experiment_id",
        "canonical_config_sha256",
        "o_excl",
        "rerun_forbidden",
        "sha256",
    }
    if set(attempt) != expected_keys:
        raise PermissionError("attempt receipt fields differ")
    if not paths["lock"].is_file():
        raise FileNotFoundError("canonical persistent attempt lock is absent")
    if gen2.gen1.base.sha256_file(paths["lock"]) != attempt["sha256"]:
        raise PermissionError("canonical persistent attempt lock SHA differs")
    persisted = json.loads(paths["lock"].read_text(encoding="utf-8"))
    if persisted != {key: value for key, value in attempt.items() if key != "sha256"}:
        raise PermissionError("in-memory attempt differs from canonical persisted lock")
    if (
        attempt["status"] != "ATTEMPT_CONSUMED_ONE_SHOT"
        or attempt["experiment_id"] != config["experiment_id"]
        or attempt["canonical_config_sha256"] != EXPECTED_CONFIG_SHA256
        or attempt["o_excl"] is not True
        or attempt["rerun_forbidden"] is not True
    ):
        raise PermissionError("persistent attempt lock semantics differ")


def _run_after_lock(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    authorized_config, authorized_paths = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_output=paths["output"],
    )
    if authorized_config != config or authorized_paths != paths:
        raise PermissionError("direct-call config or canonical path context differs")
    _verify_persisted_attempt(paths=paths, config=config, attempt=attempt)
    execution_claim = acquire_persistent_attempt_lock(
        paths["claim"],
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    started = time.perf_counter()
    preflight = _preflight(root=root, data_dir=data_dir, config=config, paths=paths)
    print(
        json.dumps(
            {
                "phase": "preflight_pass",
                "validation_cases": 181,
                "fit_cells": 45,
                "closed_form_solves": 45,
                "test_value_reads": 0,
                "upload_count": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_causal_spectral_kernel_v1_", dir=tmp_root))
    curve_oof, points, receipts = _run_curve(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        preflight=preflight,
        stage=stage,
    )
    gen2.gen1.base._atomic_parquet(stage / "oof/learning_curve_oof.parquet", curve_oof)
    gen2.gen1.base._atomic_parquet(
        stage / "validation_keys.parquet",
        preflight["selected"][["fold", "anchor_id", "station", "episode_id"]],
    )
    gen2.gen1.base._atomic_json(
        stage / "feature_contract.json",
        {
            "feature_count": len(preflight["feature_names"]),
            "feature_names": list(preflight["feature_names"]),
            "feature_names_sha256": preflight["feature_names_sha256"],
            "source": "train_only_raw_48h_context",
        },
    )

    reproducibility_checks = {
        "canonical_config_path_sha_and_deep_json_equal": True,
        "sealed_gen1_comparator_fresh_refit_each_prefix_and_seed": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "same_prefix_ids_for_comparator_and_challenger": True,
        "challenger_fresh_refit_each_prefix_fold_seed": True,
        "fixed_three_seed_prediction_mean_reducer": True,
        "same_metric_clip_and_fixed_0p20_shrink": True,
        "hyperparameter_alpha_shrink_and_weight_search_zero": True,
        "all_45_models_and_blind_predictions_saved_and_hashed": len(receipts) == 45,
        "all_saved_models_reload_prediction_exact": all(
            row["saved_model_reload_prediction_exact"] for row in receipts
        ),
        "blind_predictions_sealed_before_validation_truth_attachment": all(
            row["blind_prediction_sealed_before_validation_truth_attachment"] for row in receipts
        ),
        "train_target_api_contains_only_current_cell_train_ids": True,
        "fixed_closed_form_solver_no_validation_selection": True,
    }
    gate = evaluate_hypothesis_gate(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility_checks,
    )
    evidence = central_evidence(
        points,
        leakage_checks=preflight["leakage_checks"],
        reproducibility_checks=reproducibility_checks,
    )
    evidence["comparison_mode"] = "EXACT_OFFICIAL_PREFIX_REFIT"
    evidence["preregistration"] = {
        "generation_id": config["experiment_id"],
        "config_path": CANONICAL_CONFIG_RELATIVE,
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "created_before_first_fit": True,
        "hypothesis_count": 1,
        "hypothesis_count_at_most_3": True,
        "score_derived_tuning": False,
    }
    evidence["curve_protocol"] = {
        "comparison_mode": "EXACT_OFFICIAL_PREFIX_REFIT",
        "prefix_fractions": list(PREFIX_FRACTIONS),
        "seed_ids": [int(value) for value in config["model"]["seed_replicates"]],
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "bootstrap_replicates": int(config["validation"]["bootstrap_replicates"]),
        "bootstrap_cluster": "whole_case",
        "incumbent_fresh_refit_each_prefix": True,
        "challenger_fresh_refit_each_prefix": True,
        "same_fold_keys_metric_postprocess": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "inherited_from_gen1_fail_closed": True,
    }
    for point in evidence["points"]:
        fraction = float(point["fraction"])
        point["incumbent_seed_metrics"] = list(points[fraction]["incumbent_seed_metrics"])
        point["challenger_seed_metrics"] = list(points[fraction]["challenger_seed_metrics"])
    central = evaluate_learning_curve(load_contract(root, CANONICAL_GOAL_RELATIVE), evidence)
    if central["passed"] or gate["passed"]:
        raise AssertionError("known false exact-reference check must fail closed")
    gen2.gen1.base._atomic_json(stage / "learning_curve_evidence.json", evidence)

    access = {
        "test_sequence_cache_value_reads": 0,
        "test_feature_cache_value_reads": 0,
        "test_index_value_reads": 0,
        "test_context_value_reads": 0,
        "test_target_or_hidden_label_reads": 0,
        "absolute_test_timestamp_recovery_attempts": 0,
        "current_or_frozen_submission_value_reads": 0,
        "current_or_frozen_submission_writes": 0,
        "upload_attempts": 0,
    }
    input_after = gen2._verify_input_hashes(preflight["inputs"], config["expected_sha256"])
    if input_after != preflight["snapshot"]:
        raise RuntimeError("source/cache/current/frozen inputs changed during run")
    status = "NO_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST_READS"
    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "interpretation": (
            "Corrected same-surface Gen3 research evidence for a genuinely distinct "
            "spectral random-feature kernel. It is not an official hidden score or upload "
            "authorization; inherited Gen1 exact-reference mismatch forces fail-close."
        ),
        "one_shot_attempt": attempt,
        "one_shot_execution_claim": execution_claim,
        "hypothesis": HYPOTHESIS,
        "gen2_failure_diagnosis": config["gen2_failure_diagnosis"],
        "points": {str(fraction): points[fraction] for fraction in PREFIX_FRACTIONS},
        "local_gate": gate,
        "central_goal_evaluator": central,
        "split_audit": preflight["split_audit"],
        "prefix_audit": preflight["prefix_audit"],
        "leakage_checks": preflight["leakage_checks"],
        "reproducibility_checks": reproducibility_checks,
        "feature_build": {
            "shape": [24_360, 435],
            "feature_names_sha256": preflight["feature_names_sha256"],
            "elapsed_seconds": preflight["spectral_feature_elapsed_seconds"],
            "validation_target_values_used": 0,
        },
        "training_receipts": receipts,
        "access_counters": access,
        "candidate_validation": None,
        "invariants": {
            "append_only": True,
            "model_or_hyperparameter_search_run": False,
            "shrink_alpha_or_weight_micro_tuning_run": False,
            "test_target_or_hidden_label_reads": 0,
            "absolute_test_timestamp_recovered": False,
            "current_or_frozen_submission_mutated": False,
            "official_submission_uploads": 0,
            "team_wide_daily_upload_limit_assumed": True,
            "source_cache_current_frozen_sha_unchanged": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    gen2.gen1.base._atomic_json(stage / "metrics.json", metrics)
    registry = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "hypotheses": [
            {
                "id": HYPOTHESIS,
                "curve_qualified": False,
                "central_decision": central["decision"],
                "full_delta_m": float(points[1.0]["delta_candidate_minus_incumbent_m"]),
                "promotion_status": "research_only_do_not_promote",
            }
        ],
        "candidate_created": False,
        "candidate_uploaded": False,
        "current_frozen_sha256": config["expected_sha256"]["current/ready_submission.csv"],
        "current_frozen_unchanged": True,
        "official_upload_count": 0,
    }
    gen2.gen1.base._atomic_json(stage / "registry.json", registry)

    implementation_paths = {
        "config": paths["config"],
        "runner": Path(__file__).resolve(),
        "runner_tests": root / "tests/test_p3_causal_spectral_kernel_runner.py",
        "module_tests": root / "tests/test_p3_causal_spectral_kernel.py",
        **_implementation_paths(root, paths),
    }
    manifest = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "append_only_generation": True,
        "canonical_contract": {
            "config_path": CANONICAL_CONFIG_RELATIVE,
            "config_sha256": EXPECTED_CONFIG_SHA256,
            "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
            "config_full_deep_equality": True,
            "gen1_path": CANONICAL_GEN1_RELATIVE,
            "gen2_path": CANONICAL_GEN2_RELATIVE,
            "v5_gen2_event_sha256": config["central_ledger_anchor"]["gen2_event_sha256"],
            "output_path": CANONICAL_OUTPUT_RELATIVE,
            "attempt_lock_path": CANONICAL_LOCK_RELATIVE,
            "attempt_lock_sha256": attempt["sha256"],
            "execution_claim_path": CANONICAL_EXECUTION_CLAIM_RELATIVE,
            "execution_claim_sha256": execution_claim["sha256"],
        },
        "implementation_sha256": {
            name: gen2.gen1.base.sha256_file(path) for name, path in implementation_paths.items()
        },
        "git": gen2.gen1.base._git_state(root),
        "input_sha256_before": preflight["snapshot"],
        "input_sha256_after": input_after,
        "source_cache_current_frozen_unchanged": True,
        "output_files": _artifact_hashes(stage),
        "curve_qualified": False,
        "candidate_created": False,
        "candidate_uploaded": False,
        "official_upload_count": 0,
        "access_counters": access,
    }
    gen2.gen1.base._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = gen2.gen1.base.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    if paths["output"].exists():
        raise FileExistsError("canonical output appeared before atomic move")
    stage.replace(paths["output"])
    result = {
        "status": status,
        "artifact_dir": CANONICAL_OUTPUT_RELATIVE,
        "metrics_sha256": gen2.gen1.base.sha256_file(paths["output"] / "metrics.json"),
        "oof_sha256": gen2.gen1.base.sha256_file(
            paths["output"] / "oof/learning_curve_oof.parquet"
        ),
        "learning_curve_evidence_sha256": gen2.gen1.base.sha256_file(
            paths["output"] / "learning_curve_evidence.json"
        ),
        "registry_sha256": gen2.gen1.base.sha256_file(paths["output"] / "registry.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": None,
        "central_decision": central["decision"],
        "elapsed_seconds": float(time.perf_counter() - started),
        "official_upload_count": 0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def check_only(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _canonical_paths(root)
    config, paths = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_output=paths["output"],
    )
    preflight = _preflight(root=root, data_dir=data_dir, config=config, paths=paths)
    return {
        "status": "CANONICAL_CHECK_ONLY_PASS",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "config_deep_json_sha256": EXPECTED_CONFIG_DEEP_SHA256,
        "validation_cases": int(len(preflight["selected"])),
        "validation_rows": int(preflight["split_audit"]["validation_row_count"]),
        "fit_cells": 45,
        "closed_form_solves": 45,
        "feature_shape": list(preflight["spectral_features"].shape),
        "feature_names_sha256": preflight["feature_names_sha256"],
        "hypotheses": [HYPOTHESIS],
        "leakage_checks": preflight["leakage_checks"],
        "inherited_exact_reference_check": False,
        "output_absent": not paths["output"].exists(),
        "attempt_lock_absent": not paths["lock"].exists(),
        "execution_claim_absent": not paths["claim"].exists(),
        "test_value_reads": 0,
        "upload_count": 0,
    }


def run_experiment(*, root: Path, data_dir: Path) -> dict[str, Any]:
    paths = _canonical_paths(root)
    config, paths = authorize_entry(
        root=root,
        data_dir=data_dir,
        requested_config=paths["config"],
        requested_output=paths["output"],
    )
    attempt = acquire_persistent_attempt_lock(
        paths["lock"],
        experiment_id=config["experiment_id"],
        config_sha256=EXPECTED_CONFIG_SHA256,
        created_at=_now(),
    )
    return _run_after_lock(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        attempt=attempt,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve(strict=True)
    data_dir = args.data_dir.resolve(strict=True)
    result = (
        check_only(root=root, data_dir=data_dir)
        if args.check_only
        else run_experiment(root=root, data_dir=data_dir)
    )
    if args.check_only:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
