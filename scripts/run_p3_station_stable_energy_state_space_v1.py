"""Run the append-only P3 stable energy state-space curve exactly once.

This deterministic CPU runner is train/OOF-only. It seals blind validation
predictions before attaching the already-scored comparator truth, never opens
anonymous-test values, and cannot create or upload a submission.
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

from ocean_goal.meaningful_score_ledger_v5 import validate_ledger
from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p3_wave.corrected_repeated_forward import CorrectedFold
from p3_wave.meaningful_learning_curve import (
    PREFIX_FRACTIONS,
    central_evidence,
    evaluate_hypothesis_gate,
    evaluate_point,
)
from p3_wave.one_shot_guard import acquire_persistent_attempt_lock, safe_new_stage_path
from p3_wave.stable_energy_state_space import (
    STATE_COLUMNS,
    StableEnergyStateSpaceConfig,
    build_wave_energy_state_sequences,
    fit_stable_energy_state_space,
    load_fitted_stable_energy_state_space,
    predict_stable_energy_state_space,
    save_fitted_stable_energy_state_space,
)
from p3_wave.validation import rmse

_GEN3_RUNNER_PATH = Path(__file__).with_name("run_p3_causal_spectral_kernel_v1.py")
_GEN3_SPEC = importlib.util.spec_from_file_location("p3_gen4_gen3_helpers", _GEN3_RUNNER_PATH)
if _GEN3_SPEC is None or _GEN3_SPEC.loader is None:
    raise ImportError("failed to load pinned Gen3 runner helpers")
gen3 = importlib.util.module_from_spec(_GEN3_SPEC)
sys.modules[_GEN3_SPEC.name] = gen3
_GEN3_SPEC.loader.exec_module(gen3)

EXPECTED_CONFIG_SHA256 = "2d2df3d2b566f795fe005368e7294ff0d9493e84c8c17cc813ff076d24b4fd03"
EXPECTED_CONFIG_DEEP_SHA256 = "ae99ea1e9499205e1ad594b7c715cf2d99f43f2b49b1bd426d848d964f56293d"
GEN3_CONFIG_SHA256 = "3bf5aaa053a702f46f34ab696b7876bdf954845dbea275279c58dc0c204e3f1b"
GEN3_CONFIG_DEEP_SHA256 = "3b92f8dfca516dd9cb350e60e08b846d7c5deb7399f97f9628fc7aae7f8bc851"
CANONICAL_CONFIG_RELATIVE = "configs/experiments/p3_station_stable_energy_state_space_v1.json"
CANONICAL_GEN3_CONFIG_RELATIVE = "configs/experiments/p3_causal_spectral_kernel_v1.json"
CANONICAL_GOAL_RELATIVE = "configs/goals/meaningful_score_maximization_v3.json"
CANONICAL_COMPACT_CACHE_RELATIVE = "artifacts/p3/features_all20_v1"
CANONICAL_SEQUENCE_CACHE_RELATIVE = "artifacts/p3/sequences_all20_v1"
CANONICAL_GEN1_RELATIVE = "artifacts/p3_meaningful_learning_curve_20260823_v1"
CANONICAL_GEN3_RELATIVE = "artifacts/p3_causal_spectral_kernel_20260823_v1"
CANONICAL_V5_LEDGER_RELATIVE = "artifacts/meaningful_score_goal_v5/registry.jsonl"
CANONICAL_OUTPUT_RELATIVE = "artifacts/p3_station_stable_energy_state_space_20260823_v1"
CANONICAL_LOCK_RELATIVE = (
    "artifacts/p3_station_stable_energy_state_space_20260823_v1.ATTEMPT_LOCK.json"
)
CANONICAL_EXECUTION_CLAIM_RELATIVE = (
    "artifacts/p3_station_stable_energy_state_space_20260823_v1.EXECUTION_CLAIM.json"
)
CANONICAL_WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DATA_DIR = Path(r"C:\Users\cedis\Downloads\p3\데이터셋_P3\P3_wave_forecast")
HYPOTHESIS = "station_partial_pooled_stable_wave_energy_state_space"


def _now() -> str:
    return gen3._now()


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
        "gen3_config": workspace / CANONICAL_GEN3_CONFIG_RELATIVE,
        "goal": workspace / CANONICAL_GOAL_RELATIVE,
        "compact_cache": workspace / CANONICAL_COMPACT_CACHE_RELATIVE,
        "sequence_cache": workspace / CANONICAL_SEQUENCE_CACHE_RELATIVE,
        "gen1": workspace / CANONICAL_GEN1_RELATIVE,
        "gen3": workspace / CANONICAL_GEN3_RELATIVE,
        "v5_ledger": workspace / CANONICAL_V5_LEDGER_RELATIVE,
        "output": workspace / CANONICAL_OUTPUT_RELATIVE,
        "lock": workspace / CANONICAL_LOCK_RELATIVE,
        "claim": workspace / CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }


def _implementation_paths(root: Path, paths: dict[str, Path]) -> dict[str, Path]:
    return {
        "gen3_runner": _GEN3_RUNNER_PATH,
        "gen3_config": paths["gen3_config"],
        "stable_energy_state_space_module": root / "src/p3_wave/stable_energy_state_space.py",
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
        "gen3_metrics": paths["gen3"] / "metrics.json",
        "gen3_learning_curve_evidence": paths["gen3"] / "learning_curve_evidence.json",
        "gen3_manifest": paths["gen3"] / "manifest.json",
        "gen3_validation_keys": paths["gen3"] / "validation_keys.parquet",
    }


def _verify_v5_anchor(root: Path, paths: dict[str, Path], config: dict[str, Any]) -> None:
    records = validate_ledger(root, paths["v5_ledger"])
    expected = config["central_ledger_anchor"]
    matching = [record for record in records if record.get("seq") == expected["gen3_event_seq"]]
    if len(matching) != 1:
        raise PermissionError("canonical v5 Gen3 event is absent or duplicated")
    event = matching[0]
    if event.get("event_sha256") != expected["gen3_event_sha256"]:
        raise PermissionError("canonical v5 Gen3 event SHA differs")
    payload = event.get("payload", {})
    if (
        payload.get("evidence", {}).get("sha256") != expected["gen3_evidence_sha256"]
        or payload.get("decision", {}).get("decision") != expected["gen3_decision"]
        or payload.get("upload_performed") is not False
    ):
        raise PermissionError("canonical v5 Gen3 event semantics differ")


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
        "gen3_artifact": CANONICAL_GEN3_RELATIVE,
        "v5_ledger": CANONICAL_V5_LEDGER_RELATIVE,
        "output": CANONICAL_OUTPUT_RELATIVE,
        "attempt_lock": CANONICAL_LOCK_RELATIVE,
        "execution_claim": CANONICAL_EXECUTION_CLAIM_RELATIVE,
    }
    if config.get("canonical_paths") != expected_paths:
        raise PermissionError("canonical path fields differ")
    if config.get("experiment_id") != "p3_station_stable_energy_state_space_v1":
        raise PermissionError("experiment identity differs")
    if config.get("created_before_first_fit") is not True:
        raise PermissionError("preregistration timing declaration differs")
    if tuple(item["id"] for item in config["hypotheses"]) != (HYPOTHESIS,):
        raise PermissionError("single structural hypothesis differs")
    if config["validation"]["training_prefix_fractions"] != list(PREFIX_FRACTIONS):
        raise PermissionError("prefix curve differs")
    if config["model"] != {
        "class": "StationPartialPooledStableWaveEnergyStateSpace",
        "global_transition": "closed_form_VAR1_ridge",
        "station_adjustment": "closed_form_residual_VAR1_ridge",
        "global_ridge": 0.001,
        "station_residual_ridge": 0.01,
        "maximum_spectral_radius": 0.995,
        "rollout_steps_20m": 72,
        "standardized_state_clip": 10.0,
        "official_lead_steps_20m": [9, 18, 27, 36, 54, 72],
        "deterministic": True,
        "seed_metric_replica_ids": [20260816, 20260817, 20260818],
        "seed_replica_rule": (
            "deterministic identical prediction copied into three protocol metric slots; "
            "seeds do not affect fit"
        ),
        "hyperparameter_search": False,
        "expected_actual_fit_cells": 15,
        "expected_protocol_seed_metric_cells": 45,
        "expected_closed_form_solves": 60,
    }:
        raise PermissionError("frozen stable state-space model contract differs")
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
        if gen3.gen2.gen1.base.sha256_file(path) != config["implementation_sha256"][name]:
            raise PermissionError(f"implementation SHA differs: {name}")
    for name, path in _reference_paths(root, paths).items():
        if gen3.gen2.gen1.base.sha256_file(path) != config["reference_evidence_sha256"][name]:
            raise PermissionError(f"reference evidence SHA differs: {name}")
    _verify_v5_anchor(root, paths, config)
    return config, paths


def _gen3_paths(root: Path) -> dict[str, Path]:
    return gen3._canonical_paths(root)


def _preflight(
    *, root: Path, data_dir: Path, config: dict[str, Any], paths: dict[str, Path]
) -> dict[str, Any]:
    gen3_config_bytes = paths["gen3_config"].read_bytes()
    if hashlib.sha256(gen3_config_bytes).hexdigest() != GEN3_CONFIG_SHA256:
        raise PermissionError("pinned Gen3 config byte SHA differs")
    gen3_config = json.loads(gen3_config_bytes)
    if _deep_sha(gen3_config) != GEN3_CONFIG_DEEP_SHA256:
        raise PermissionError("pinned Gen3 config deep JSON differs")
    preflight = gen3._preflight(
        root=root,
        data_dir=data_dir,
        config=gen3_config,
        paths=_gen3_paths(root),
    )
    gen3_metrics = json.loads((paths["gen3"] / "metrics.json").read_text(encoding="utf-8"))
    gen3_full = gen3_metrics["points"]["1.0"]
    diagnosis = config["gen3_failure_diagnosis"]
    if (
        gen3_metrics["central_goal_evaluator"]["decision"] != "RESEARCH_ONLY"
        or float(gen3_full["delta_candidate_minus_incumbent_m"])
        != diagnosis["full_delta_candidate_minus_incumbent_m"]
        or list(gen3_full["delta_ci90_m"]) != diagnosis["full_ci90_m"]
        or not diagnosis["all_five_prefix_point_estimates_worse"]
    ):
        raise ValueError("sealed Gen3 aggregate diagnosis differs")
    state_started = time.perf_counter()
    state_sequences = build_wave_energy_state_sequences(preflight["raw"])
    state_elapsed = time.perf_counter() - state_started
    anchor_time_ns = (
        pd.DatetimeIndex(
            pd.to_datetime(preflight["anchors"]["anchor_time"], utc=True, errors="raise")
        )
        .as_unit("ns")
        .asi8
    )
    if anchor_time_ns.shape != (24_360,):
        raise ValueError("anchor-time identity shape differs")
    preflight.update(
        {
            "state_sequences": state_sequences,
            "anchor_time_ns": anchor_time_ns,
            "state_build_elapsed_seconds": float(state_elapsed),
        }
    )
    return preflight


def _protected_roots(root: Path, data_dir: Path, paths: dict[str, Path]) -> tuple[Path, ...]:
    return (
        data_dir,
        paths["compact_cache"],
        paths["sequence_cache"],
        paths["gen1"],
        paths["gen3"],
        root / "submissions",
        root / "output",
        root / "데이터셋 원본",
    )


def _prefix_id_sha(ids: np.ndarray) -> str:
    return gen3._prefix_id_sha(ids)


def _write_npy_exclusive(path: Path, values: np.ndarray) -> str:
    return gen3._write_npy_exclusive(path, values)


def _postprocess(absolute_hs: np.ndarray, current_hs: np.ndarray) -> np.ndarray:
    delta = np.asarray(absolute_hs, dtype=np.float64) - np.asarray(current_hs)[:, None]
    return gen3._postprocess(delta, current_hs)


def _run_curve(
    *,
    root: Path,
    data_dir: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    preflight: dict[str, Any],
    stage: Path,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]], list[dict[str, Any]]]:
    del config
    states = preflight["state_sequences"]
    station = preflight["station"]
    times = preflight["anchor_time_ns"]
    anchors = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    protected = _protected_roots(root, data_dir, paths)
    model_config = StableEnergyStateSpaceConfig()
    anchor_lookup = anchors.set_index("anchor_id")
    blind_predictions: dict[tuple[float, str], np.ndarray] = {}
    receipts: list[dict[str, Any]] = []
    completed = 0

    for fraction in PREFIX_FRACTIONS:
        prefix_tag = f"p{int(round(fraction * 100)):03d}"
        for fold in folds:
            train_ids = preflight["prefix_ids"][fraction][fold.name]
            validation_ids = fold.validation_ids
            if np.intersect1d(train_ids, validation_ids).size:
                raise AssertionError("train/validation IDs overlap before fit")
            current_hs = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(np.float64)
            print(
                json.dumps(
                    {
                        "phase": "fit_stable_energy_state_space_cell",
                        "completed_before": completed,
                        "total_actual_fit_cells": 15,
                        "prefix": fraction,
                        "fold": fold.name,
                        "train_cases": len(train_ids),
                        "deterministic_seed_metric_replicas": 3,
                        "validation_target_values_read_by_model": 0,
                        "device": "cpu_closed_form",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            started = time.perf_counter()
            fitted, transition_receipt = fit_stable_energy_state_space(
                states,
                station,
                times,
                train_ids,
                forbidden_ids=validation_ids,
                config=model_config,
            )
            raw_prediction = predict_stable_energy_state_space(
                fitted, states, station, validation_ids
            )
            prediction = _postprocess(raw_prediction, current_hs)
            model_relative = f"models/{prefix_tag}/folds/{fold.name}/model.npz"
            model_path = safe_new_stage_path(stage, model_relative, protected_roots=protected)
            save_fitted_stable_energy_state_space(fitted, model_path)
            model_sha = gen3.gen2.gen1.base.sha256_file(model_path)
            blind_relative = f"blind_predictions/{prefix_tag}/{fold.name}.npy"
            blind_path = safe_new_stage_path(stage, blind_relative, protected_roots=protected)
            blind_sha = _write_npy_exclusive(blind_path, prediction.astype(np.float64))

            reloaded = load_fitted_stable_energy_state_space(model_path)
            reload_raw = predict_stable_energy_state_space(
                reloaded, states, station, validation_ids
            )
            reload_prediction = _postprocess(reload_raw, current_hs)
            reload_exact = bool(np.array_equal(reload_prediction, prediction))
            maximum_reload_difference = float(np.max(np.abs(reload_prediction - prediction)))
            if not reload_exact:
                raise RuntimeError("saved state-space reload failed exact reproduction")
            blind_predictions[(fraction, fold.name)] = prediction
            completed += 1
            receipts.append(
                {
                    "prefix_fraction": float(fraction),
                    "fold": fold.name,
                    "train_cases": int(len(train_ids)),
                    "validation_cases": int(len(validation_ids)),
                    "actual_fit_cells": 1,
                    "closed_form_solves": 4,
                    "deterministic_seed_metric_replica_ids": [20260816, 20260817, 20260818],
                    "train_id_sha256": _prefix_id_sha(train_ids),
                    "validation_id_sha256": _prefix_id_sha(validation_ids),
                    "scaler_fit_id_sha256": fitted.scaler.fit_ids_sha256,
                    "transition_key_sha256": fitted.transition_key_sha256,
                    "unique_station_time_transitions": fitted.transition_count,
                    "spectral_radius_before": fitted.spectral_radius_before.tolist(),
                    "spectral_radius_after": fitted.spectral_radius_after.tolist(),
                    "model_relative_path": model_relative,
                    "model_sha256": model_sha,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "blind_prediction_sealed_before_validation_truth_attachment": True,
                    "saved_model_reload_prediction_exact": reload_exact,
                    "saved_model_reload_max_abs_difference_m": maximum_reload_difference,
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "train_target_values_read_by_model": 0,
                    "validation_target_values_read_by_model": 0,
                    "test_or_hidden_value_reads": 0,
                }
            )

    if completed != 15 or sum(row["closed_form_solves"] for row in receipts) != 60:
        raise AssertionError("fit-cell or closed-form-solve count differs")
    comparator_truth = gen3._load_comparator_truth_after_blind(_gen3_paths(root))
    points: dict[float, dict[str, Any]] = {}
    all_frames: list[pd.DataFrame] = []
    seed_ids = [20260816, 20260817, 20260818]
    for fraction in PREFIX_FRACTIONS:
        fold_frames: list[pd.DataFrame] = []
        for fold in folds:
            comparator = gen3.gen2._cell_comparator(comparator_truth, fraction=fraction, fold=fold)
            comparator["challenger_prediction"] = blind_predictions[(fraction, fold.name)].reshape(
                -1
            )
            fold_frames.append(comparator)
        deterministic_frame = pd.concat(fold_frames, ignore_index=True)
        keys = ["fold", "anchor_id", "station", "lead_h"]
        invariant = ["target_hs", "current_hs", "persistence", "incumbent_prediction"]
        mean_frame = deterministic_frame[keys + invariant + ["challenger_prediction"]].copy()
        mean_frame["prefix_fraction"] = float(fraction)
        all_frames.append(mean_frame)
        point = evaluate_point(
            mean_frame,
            candidate_column="challenger_prediction",
            bootstrap_replicates=5000,
            bootstrap_seed=20260823 + int(round(fraction * 100)),
        )
        gen1_point = preflight["gen1_metrics"]["points_by_hypothesis"]["fixed_horizon_splice"][
            str(fraction)
        ]
        point["incumbent_seed_metrics"] = [
            float(value) for value in gen1_point["incumbent_seed_metrics"]
        ]
        deterministic_metric = float(
            rmse(mean_frame["target_hs"], mean_frame["challenger_prediction"])
        )
        point["challenger_seed_metrics"] = [deterministic_metric for _ in seed_ids]
        points[fraction] = point
        print(
            json.dumps(
                {
                    "phase": "prefix_scored_after_all_15_blind_predictions_sealed",
                    "prefix": fraction,
                    "completed_actual_fit_cells": completed,
                    "protocol_seed_metric_cells": 45,
                    "delta_candidate_minus_incumbent_m": point["delta_candidate_minus_incumbent_m"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return pd.concat(all_frames, ignore_index=True), points, receipts


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    return gen3._artifact_hashes(stage)


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
    if gen3.gen2.gen1.base.sha256_file(paths["lock"]) != attempt["sha256"]:
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
                "actual_fit_cells": 15,
                "closed_form_solves": 60,
                "device": "cpu_closed_form",
                "test_value_reads": 0,
                "upload_count": 0,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_station_stable_energy_ssm_v1_", dir=tmp_root))
    curve_oof, points, receipts = _run_curve(
        root=root,
        data_dir=data_dir,
        config=config,
        paths=paths,
        preflight=preflight,
        stage=stage,
    )
    gen3.gen2.gen1.base._atomic_parquet(stage / "oof/learning_curve_oof.parquet", curve_oof)
    gen3.gen2.gen1.base._atomic_parquet(
        stage / "validation_keys.parquet",
        preflight["selected"][["fold", "anchor_id", "station", "episode_id"]],
    )
    gen3.gen2.gen1.base._atomic_json(
        stage / "state_contract.json",
        {
            "state_count": len(STATE_COLUMNS),
            "state_columns": list(STATE_COLUMNS),
            "source": "train_only_raw_48h_context",
            "native_steps": 145,
            "native_cadence_minutes": 20,
            "transition_fit_target_values_used": 0,
        },
    )

    reproducibility_checks = {
        "canonical_config_path_sha_and_deep_json_equal": True,
        "sealed_gen1_comparator_fresh_refit_each_prefix_and_seed": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "same_prefix_ids_for_comparator_and_challenger": True,
        "challenger_fresh_refit_each_prefix_fold": True,
        "deterministic_three_seed_metric_replica_values_exact": True,
        "same_metric_clip_and_fixed_0p20_shrink": True,
        "hyperparameter_alpha_shrink_weight_and_seed_search_zero": True,
        "all_15_models_and_blind_predictions_saved_and_hashed": len(receipts) == 15,
        "all_saved_models_reload_prediction_exact": all(
            row["saved_model_reload_prediction_exact"] for row in receipts
        ),
        "blind_predictions_sealed_before_validation_truth_attachment": all(
            row["blind_prediction_sealed_before_validation_truth_attachment"] for row in receipts
        ),
        "model_fit_target_values_zero": True,
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
        "seed_ids": [20260816, 20260817, 20260818],
        "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
        "bootstrap_replicates": int(config["validation"]["bootstrap_replicates"]),
        "bootstrap_cluster": "whole_case",
        "incumbent_fresh_refit_each_prefix": True,
        "challenger_fresh_refit_each_prefix": True,
        "same_fold_keys_metric_postprocess": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        "inherited_from_gen1_fail_closed": True,
        "challenger_deterministic_seed_replica_policy": (
            "one exact fit per prefix-fold; identical prediction occupies three fixed "
            "protocol metric slots because the estimator has no stochastic state"
        ),
    }
    for point in evidence["points"]:
        fraction = float(point["fraction"])
        point["incumbent_seed_metrics"] = list(points[fraction]["incumbent_seed_metrics"])
        point["challenger_seed_metrics"] = list(points[fraction]["challenger_seed_metrics"])
    central = evaluate_learning_curve(load_contract(root, CANONICAL_GOAL_RELATIVE), evidence)
    if central["passed"]:
        raise AssertionError("known false exact-reference check must fail closed")
    gen3.gen2.gen1.base._atomic_json(stage / "learning_curve_evidence.json", evidence)

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
    inputs = gen3._input_paths(root, data_dir, _gen3_paths(root))
    input_after = gen3.gen2._verify_input_hashes(inputs, config["expected_sha256"])
    if input_after != preflight["snapshot"]:
        raise RuntimeError("source/cache/current/frozen inputs changed during run")
    status = "NO_CURVE_QUALIFICATION_RESEARCH_ONLY_STOPPED_BEFORE_TEST_READS"
    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "interpretation": (
            "Corrected same-surface Gen4 research evidence for a genuinely distinct "
            "station-partial-pooled stable wave-energy state-space model. It is not an "
            "official hidden score or upload authorization; inherited Gen1 exact-reference "
            "mismatch forces fail-close."
        ),
        "one_shot_attempt": attempt,
        "one_shot_execution_claim": execution_claim,
        "hypothesis": HYPOTHESIS,
        "gen3_failure_diagnosis": config["gen3_failure_diagnosis"],
        "points": {str(fraction): points[fraction] for fraction in PREFIX_FRACTIONS},
        "local_gate": gate,
        "central_goal_evaluator": central,
        "split_audit": preflight["split_audit"],
        "prefix_audit": preflight["prefix_audit"],
        "leakage_checks": preflight["leakage_checks"],
        "reproducibility_checks": reproducibility_checks,
        "state_build": {
            "shape": list(preflight["state_sequences"].shape),
            "state_columns": list(STATE_COLUMNS),
            "elapsed_seconds": preflight["state_build_elapsed_seconds"],
            "train_target_values_used": 0,
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
            "cpu_only_closed_form_execution": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    gen3.gen2.gen1.base._atomic_json(stage / "metrics.json", metrics)
    registry = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": status,
        "hypotheses": [
            {
                "id": HYPOTHESIS,
                "curve_qualified": False,
                "local_gate_passed": bool(gate["passed"]),
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
    gen3.gen2.gen1.base._atomic_json(stage / "registry.json", registry)

    implementation_paths = {
        "config": paths["config"],
        "runner": Path(__file__).resolve(),
        "runner_tests": root / "tests/test_p3_station_stable_energy_state_space_runner.py",
        "module_tests": root / "tests/test_p3_stable_energy_state_space.py",
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
            "gen3_path": CANONICAL_GEN3_RELATIVE,
            "v5_gen3_event_sha256": config["central_ledger_anchor"]["gen3_event_sha256"],
            "output_path": CANONICAL_OUTPUT_RELATIVE,
            "attempt_lock_path": CANONICAL_LOCK_RELATIVE,
            "attempt_lock_sha256": attempt["sha256"],
            "execution_claim_path": CANONICAL_EXECUTION_CLAIM_RELATIVE,
            "execution_claim_sha256": execution_claim["sha256"],
        },
        "implementation_sha256": {
            name: gen3.gen2.gen1.base.sha256_file(path)
            for name, path in implementation_paths.items()
        },
        "git": gen3.gen2.gen1.base._git_state(root),
        "input_sha256_before": preflight["snapshot"],
        "input_sha256_after": input_after,
        "source_cache_current_frozen_unchanged": True,
        "output_files": _artifact_hashes(stage),
        "curve_qualified": False,
        "local_gate_passed": bool(gate["passed"]),
        "candidate_created": False,
        "candidate_uploaded": False,
        "official_upload_count": 0,
        "access_counters": access,
    }
    gen3.gen2.gen1.base._atomic_json(stage / "manifest.json", manifest)
    manifest_sha = gen3.gen2.gen1.base.sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii", newline="\n"
    )
    if paths["output"].exists():
        raise FileExistsError("canonical output appeared before atomic move")
    stage.replace(paths["output"])
    result = {
        "status": status,
        "artifact_dir": CANONICAL_OUTPUT_RELATIVE,
        "metrics_sha256": gen3.gen2.gen1.base.sha256_file(paths["output"] / "metrics.json"),
        "oof_sha256": gen3.gen2.gen1.base.sha256_file(
            paths["output"] / "oof/learning_curve_oof.parquet"
        ),
        "learning_curve_evidence_sha256": gen3.gen2.gen1.base.sha256_file(
            paths["output"] / "learning_curve_evidence.json"
        ),
        "registry_sha256": gen3.gen2.gen1.base.sha256_file(paths["output"] / "registry.json"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": None,
        "local_gate_passed": bool(gate["passed"]),
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
        "actual_fit_cells": 15,
        "protocol_seed_metric_cells": 45,
        "closed_form_solves": 60,
        "state_shape": list(preflight["state_sequences"].shape),
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
