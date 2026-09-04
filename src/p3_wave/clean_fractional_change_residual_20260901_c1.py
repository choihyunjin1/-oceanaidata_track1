"""Clean, train-only P3 fractional-change residual confirmation cycle.

This module deliberately has no inference or submission surface.  It evaluates one
pre-registered scratch model on the clean historical OOF surface only.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import ResidualRegressor, compact_feature_columns, threshold_case_weights
from .one_shot_guard import acquire_persistent_attempt_lock
from .validation import build_forecast_folds, expand_leads, metric_slices, rmse

EXPERIMENT_ID = "p3_clean_fractional_change_residual_20260901_c1"
CONFIG_RELATIVE = f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER_RELATIVE = f"scripts/run_{EXPERIMENT_ID}.py"
MODULE_RELATIVE = "src/p3_wave/clean_fractional_change_residual_20260901_c1.py"
TEST_RELATIVE = f"tests/test_{EXPERIMENT_ID}.py"
OUTPUT_RELATIVE = f"artifacts/{EXPERIMENT_ID}"
LOCK_RELATIVE = f"artifacts/{EXPERIMENT_ID}.ATTEMPT_LOCK.json"

ALLOWED_INPUT_LABELS = frozenset(
    {
        "policy/00_ORGANIZER_DATA_POLICY.md",
        "policy/organizer_data_policy_20260901.json",
        "policy/p3_clean_incumbent_20260901.json",
        "source/README.md",
        "source/train_wave.csv",
        "source/train_atmos.csv",
        "cache/manifest.json",
        "cache/train_features.parquet",
        "cache/train_anchors.parquet",
        "clean_fallback/oof.parquet",
    }
)


class CleanCycleError(RuntimeError):
    """Raised when the clean-cycle contract fails closed."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def strict_json(path: Path) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise CleanCycleError(f"duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    parsed = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate)
    if not isinstance(parsed, dict):
        raise CleanCycleError(f"top-level JSON must be an object: {path.name}")
    return parsed


def fractional_target(delta: np.ndarray, current_hs: np.ndarray, *, offset_m: float) -> np.ndarray:
    delta_array = np.asarray(delta, dtype=np.float64)
    current = np.asarray(current_hs, dtype=np.float64)
    if delta_array.shape != current.shape:
        raise ValueError("delta/current shape mismatch")
    if offset_m <= 0.0 or not np.isfinite(offset_m):
        raise ValueError("offset_m must be finite and positive")
    denominator = current + offset_m
    if not np.isfinite(delta_array).all() or not np.isfinite(denominator).all():
        raise ValueError("fractional target inputs must be finite")
    if np.any(denominator <= 0.0):
        raise ValueError("fractional target denominator must be positive")
    return delta_array / denominator


def restore_delta(
    fractional_prediction: np.ndarray,
    current_hs: np.ndarray,
    *,
    offset_m: float,
) -> np.ndarray:
    fraction = np.asarray(fractional_prediction, dtype=np.float64)
    current = np.asarray(current_hs, dtype=np.float64)
    if fraction.shape != current.shape:
        raise ValueError("fraction/current shape mismatch")
    restored = fraction * (current + offset_m)
    if not np.isfinite(restored).all():
        raise ValueError("restored delta is non-finite")
    return restored


def blend_with_clean_fallback(
    fallback_prediction: np.ndarray,
    challenger_prediction: np.ndarray,
    *,
    challenger_weight: float,
) -> np.ndarray:
    fallback = np.asarray(fallback_prediction, dtype=np.float64)
    challenger = np.asarray(challenger_prediction, dtype=np.float64)
    if fallback.shape != challenger.shape:
        raise ValueError("fallback/challenger shape mismatch")
    if not 0.0 < challenger_weight <= 1.0:
        raise ValueError("challenger weight must be in (0, 1]")
    output = (1.0 - challenger_weight) * fallback + challenger_weight * challenger
    if not np.isfinite(output).all():
        raise ValueError("blended prediction is non-finite")
    return output


def _input_paths(root: Path, data_dir: Path) -> dict[str, Path]:
    cache = root / "artifacts" / "p3" / "features_all20_v1"
    fallback = root / "artifacts" / "p3_corrected_repeated_forward_catboost_v2"
    return {
        "policy/00_ORGANIZER_DATA_POLICY.md": root / "00_ORGANIZER_DATA_POLICY.md",
        "policy/organizer_data_policy_20260901.json": (
            root / "configs" / "compliance" / "organizer_data_policy_20260901.json"
        ),
        "policy/p3_clean_incumbent_20260901.json": (
            root / "configs" / "compliance" / "p3_clean_incumbent_20260901.json"
        ),
        "source/README.md": data_dir / "README.md",
        "source/train_wave.csv": data_dir / "train_wave.csv",
        "source/train_atmos.csv": data_dir / "train_atmos.csv",
        "cache/manifest.json": cache / "manifest.json",
        "cache/train_features.parquet": cache / "train_features.parquet",
        "cache/train_anchors.parquet": cache / "train_anchors.parquet",
        "clean_fallback/oof.parquet": fallback / "oof.parquet",
    }


def _verify_negative_fingerprint(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        CONFIG_RELATIVE,
        RUNNER_RELATIVE,
        MODULE_RELATIVE,
        TEST_RELATIVE,
    }
    observed: set[str] = set()
    search_roots = ("configs/experiments", "scripts", "src/p3_wave", "tests")
    token = str(config["negative_fingerprint"]["namespace_token"])
    for relative_root in search_roots:
        directory = root / relative_root
        for path in directory.rglob("*"):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.suffix != ".pyc"
                and token in path.name
            ):
                observed.add(path.relative_to(root).as_posix())
    if observed != expected:
        raise CleanCycleError(
            f"fresh namespace mismatch; expected {sorted(expected)}, observed {sorted(observed)}"
        )
    if int(config["negative_fingerprint"]["precreation_repository_matches"]) != 0:
        raise CleanCycleError("precreation negative fingerprint was not zero")
    return {
        "namespace_token": token,
        "precreation_repository_matches": 0,
        "current_expected_paths": sorted(observed),
        "fresh": True,
    }


def preflight(*, root: Path, data_dir: Path) -> dict[str, Any]:
    workspace = root.resolve(strict=True)
    source = data_dir.resolve(strict=True)
    config_path = (workspace / CONFIG_RELATIVE).resolve(strict=True)
    config = strict_json(config_path)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise CleanCycleError("experiment id mismatch")
    if config.get("status") != "PREREGISTERED_READY_NOT_EXECUTED":
        raise CleanCycleError("config is not in the fixed READY state")
    if set(config.get("inputs", {})) != ALLOWED_INPUT_LABELS:
        raise CleanCycleError("input allowlist is not exact")

    implementation = {
        "runner": workspace / RUNNER_RELATIVE,
        "module": workspace / MODULE_RELATIVE,
        "test": workspace / TEST_RELATIVE,
    }
    for name, path in implementation.items():
        path = path.resolve(strict=True)
        expected = str(config["implementation_sha256"][name])
        if sha256(path) != expected:
            raise CleanCycleError(f"implementation hash mismatch: {name}")

    paths = _input_paths(workspace, source)
    if set(paths) != ALLOWED_INPUT_LABELS:
        raise CleanCycleError("runtime input surface is not exact")
    verified_inputs: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        resolved = path.resolve(strict=True)
        expected = config["inputs"][label]
        observed_bytes = resolved.stat().st_size
        observed_hash = sha256(resolved)
        if observed_bytes != int(expected["bytes"]) or observed_hash != str(expected["sha256"]):
            raise CleanCycleError(f"input pin mismatch: {label}")
        verified_inputs[label] = {"bytes": observed_bytes, "sha256": observed_hash}

    clean_registry = strict_json(paths["policy/p3_clean_incumbent_20260901.json"])
    lineage = clean_registry.get("lineage", {})
    if not lineage.get("organizer_distributed_data_only"):
        raise CleanCycleError("clean incumbent registry lacks distributed-data-only seal")
    if not lineage.get("models_trained_from_scratch"):
        raise CleanCycleError("clean incumbent registry lacks scratch-model seal")
    if lineage.get("forbidden_lineages_referenced") != []:
        raise CleanCycleError("clean incumbent registry contains a forbidden lineage")

    output = (workspace / OUTPUT_RELATIVE).resolve(strict=False)
    lock = (workspace / LOCK_RELATIVE).resolve(strict=False)
    if output.exists() or lock.exists():
        raise CleanCycleError("fresh output or one-shot lock already exists")

    return {
        "schema_version": "p3.clean_fractional_change.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY",
        "config_sha256": sha256(config_path),
        "organizer_policy_highest_precedence": True,
        "source_lineage": {
            "distributed_train_only": True,
            "scratch_weights_only": True,
            "pretrained_weight_files_loaded": 0,
            "external_observation_rows": 0,
            "external_reanalysis_rows": 0,
            "external_forecast_rows": 0,
        },
        "official_access": {
            "test_context_rows": 0,
            "test_index_rows": 0,
            "sample_submission_rows": 0,
            "submission_csv_rows_created": 0,
            "uploads": 0,
            "hidden_truth_rows": 0,
        },
        "negative_fingerprint": _verify_negative_fingerprint(workspace, config),
        "verified_inputs": verified_inputs,
    }


def evaluate_gate(oof: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    truth = oof["target_hs"].to_numpy(dtype=np.float64)
    baseline = oof["clean_fallback_prediction"].to_numpy(dtype=np.float64)
    candidate = oof["candidate_prediction"].to_numpy(dtype=np.float64)
    baseline_rmse = rmse(truth, baseline)
    candidate_rmse = rmse(truth, candidate)

    folds: dict[str, dict[str, float | bool]] = {}
    improved_folds = 0
    for fold, group in oof.groupby("fold", sort=False, observed=True):
        fold_truth = group["target_hs"].to_numpy(dtype=np.float64)
        fold_base = group["clean_fallback_prediction"].to_numpy(dtype=np.float64)
        fold_candidate = group["candidate_prediction"].to_numpy(dtype=np.float64)
        before = rmse(fold_truth, fold_base)
        after = rmse(fold_truth, fold_candidate)
        improved = after < before
        improved_folds += int(improved)
        folds[str(fold)] = {
            "clean_fallback_rmse_m": before,
            "candidate_rmse_m": after,
            "delta_rmse_m": after - before,
            "improved": improved,
        }

    station_degradation: dict[str, float] = {}
    for station, group in oof.groupby("station", sort=True, observed=True):
        station_truth = group["target_hs"].to_numpy(dtype=np.float64)
        before = rmse(
            station_truth,
            group["clean_fallback_prediction"].to_numpy(dtype=np.float64),
        )
        after = rmse(station_truth, group["candidate_prediction"].to_numpy(dtype=np.float64))
        station_degradation[str(station)] = after - before

    long_group = oof.loc[oof["lead_h"].isin([18, 24])]
    long_truth = long_group["target_hs"].to_numpy(dtype=np.float64)
    long_degradation = rmse(
        long_truth,
        long_group["candidate_prediction"].to_numpy(dtype=np.float64),
    ) - rmse(
        long_truth,
        long_group["clean_fallback_prediction"].to_numpy(dtype=np.float64),
    )

    gate_config = config["gate"]
    checks = {
        "strict_pooled_improvement": candidate_rmse < baseline_rmse,
        "minimum_improved_folds": improved_folds >= int(gate_config["minimum_improved_folds"]),
        "station_degradation_within_guard": max(station_degradation.values())
        <= float(gate_config["maximum_station_degradation_m"]),
        "long_lead_degradation_within_guard": long_degradation
        <= float(gate_config["maximum_long_lead_degradation_m"]),
        "finite_and_physical": bool(
            np.isfinite(candidate).all()
            and np.min(candidate) >= float(gate_config["prediction_clip_m"][0])
            and np.max(candidate) <= float(gate_config["prediction_clip_m"][1])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "clean_fallback_rmse_m": baseline_rmse,
        "candidate_rmse_m": candidate_rmse,
        "delta_rmse_m": candidate_rmse - baseline_rmse,
        "improved_folds": improved_folds,
        "folds": folds,
        "station_delta_rmse_m": station_degradation,
        "lead_18_24_delta_rmse_m": long_degradation,
    }


def _load_train_only_surfaces(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cache = root / "artifacts" / "p3" / "features_all20_v1"
    fallback = root / "artifacts" / "p3_corrected_repeated_forward_catboost_v2"
    features = pd.read_parquet(cache / "train_features.parquet")
    anchors = pd.read_parquet(cache / "train_anchors.parquet")
    oof = pd.read_parquet(fallback / "oof.parquet")
    return features, anchors, oof


def execute(*, root: Path, data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    workspace = root.resolve(strict=True)
    source = data_dir.resolve(strict=True)
    first = preflight(root=workspace, data_dir=source)
    second = preflight(root=workspace, data_dir=source)
    first_bytes = canonical_json_bytes(first)
    if first_bytes != canonical_json_bytes(second):
        raise CleanCycleError("the two READY preflights are not byte-identical")

    config_path = workspace / CONFIG_RELATIVE
    config = strict_json(config_path)
    lock_path = workspace / LOCK_RELATIVE
    lock = acquire_persistent_attempt_lock(
        lock_path,
        experiment_id=EXPERIMENT_ID,
        config_sha256=sha256(config_path),
        created_at=str(config["created_at_kst"]),
    )
    output = workspace / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_bytes(first_bytes)

    try:
        features, anchors, fallback_oof = _load_train_only_surfaces(workspace)
        if len(features) != 24_360 or len(anchors) != 24_360 or len(fallback_oof) != 1_086:
            raise CleanCycleError("train-only surface row-count mismatch")
        if not features["anchor_id"].equals(anchors["anchor_id"]):
            raise CleanCycleError("feature/anchor alignment mismatch")
        required_oof = {
            "anchor_id",
            "station",
            "lead_h",
            "target_hs",
            "fold",
            "final_prediction",
        }
        if not required_oof.issubset(fallback_oof.columns):
            raise CleanCycleError("clean fallback OOF schema mismatch")

        feature_columns = compact_feature_columns(
            [column for column in features.columns if column not in {"anchor_id", "station"}]
        )
        if len(feature_columns) != int(config["features"]["expected_feature_count"]):
            raise CleanCycleError("fixed clean feature count mismatch")

        predictions: list[pd.DataFrame] = []
        training_receipts: list[dict[str, Any]] = []
        folds = build_forecast_folds(anchors)
        seeds = list(config["model"]["fold_seeds"])
        for fold, seed in zip(folds, seeds, strict=True):
            fold_started = time.perf_counter()
            expected_validation = fallback_oof.loc[
                fallback_oof["fold"].eq(fold.name), "anchor_id"
            ].drop_duplicates()
            if set(expected_validation.astype(int)) != set(fold.validation_ids.astype(int)):
                raise CleanCycleError(f"validation surface differs from clean fallback: {fold.name}")

            train_x, train_delta, train_meta = expand_leads(
                features,
                anchors,
                fold.train_ids,
                feature_columns,
            )
            valid_x, _, valid_meta = expand_leads(
                features,
                anchors,
                fold.validation_ids,
                feature_columns,
            )
            offset = float(config["model"]["fractional_offset_m"])
            train_current = train_meta["current_hs"].to_numpy(dtype=np.float64)
            transformed_target = fractional_target(train_delta, train_current, offset_m=offset)
            weights = threshold_case_weights(train_current)
            model = ResidualRegressor(
                "catboost",
                seed=int(seed),
                parameters=dict(config["model"]["catboost"]),
            )
            model.fit(train_x, transformed_target, sample_weight=weights)
            fractional_prediction = model.predict_delta(valid_x)
            valid_current = valid_meta["current_hs"].to_numpy(dtype=np.float64)
            challenger = valid_current + restore_delta(
                fractional_prediction,
                valid_current,
                offset_m=offset,
            )
            challenger = np.clip(
                challenger,
                float(config["gate"]["prediction_clip_m"][0]),
                float(config["gate"]["prediction_clip_m"][1]),
            )
            frame = valid_meta.copy()
            frame["fold"] = fold.name
            frame["fractional_challenger_prediction"] = challenger
            fallback = fallback_oof[
                ["anchor_id", "station", "lead_h", "target_hs", "final_prediction"]
            ].rename(
                columns={
                    "target_hs": "fallback_target_hs",
                    "final_prediction": "clean_fallback_prediction",
                }
            )
            frame = frame.merge(
                fallback,
                on=["anchor_id", "station", "lead_h"],
                how="left",
                validate="one_to_one",
            )
            if frame["clean_fallback_prediction"].isna().any():
                raise CleanCycleError(f"missing clean fallback rows: {fold.name}")
            if not np.array_equal(
                frame["target_hs"].to_numpy(dtype=np.float64),
                frame["fallback_target_hs"].to_numpy(dtype=np.float64),
            ):
                raise CleanCycleError(f"target mismatch against clean fallback: {fold.name}")
            frame["candidate_prediction"] = blend_with_clean_fallback(
                frame["clean_fallback_prediction"].to_numpy(dtype=np.float64),
                frame["fractional_challenger_prediction"].to_numpy(dtype=np.float64),
                challenger_weight=float(config["model"]["challenger_weight"]),
            )
            predictions.append(frame)
            training_receipts.append(
                {
                    "fold": fold.name,
                    "seed": int(seed),
                    "train_anchor_count": int(len(fold.train_ids)),
                    "train_rows": int(len(train_x)),
                    "validation_anchor_count": int(len(fold.validation_ids)),
                    "validation_rows": int(len(valid_x)),
                    "runtime_seconds": time.perf_counter() - fold_started,
                    "pretrained_weight_files_loaded": 0,
                }
            )

        oof = pd.concat(predictions, ignore_index=True)
        if len(oof) != 1_086 or oof.duplicated(["anchor_id", "station", "lead_h"]).any():
            raise CleanCycleError("candidate OOF surface mismatch")
        gate = evaluate_gate(oof, config)
        candidate_metrics = metric_slices(
            oof,
            oof["candidate_prediction"].to_numpy(dtype=np.float64),
        )
        fallback_metrics = metric_slices(
            oof,
            oof["clean_fallback_prediction"].to_numpy(dtype=np.float64),
        )
        oof_path = output / "oof.parquet"
        oof.to_parquet(oof_path, index=False)
        result = {
            "schema_version": "p3.clean_fractional_change.result.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "COMPLETE_GO_LOCAL_CONFIRMATION"
                if gate["passed"]
                else "COMPLETE_NO_GO_CLEAN_FRACTIONAL_CHANGE"
            ),
            "fit_count": len(training_receipts),
            "runtime_seconds": time.perf_counter() - started,
            "gate": gate,
            "metrics": {
                "clean_fallback": fallback_metrics,
                "candidate": candidate_metrics,
            },
            "training_receipts": training_receipts,
            "input_hashes": first["verified_inputs"],
            "preflight_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "oof": {
                "rows": int(len(oof)),
                "sha256": sha256(oof_path),
            },
            "source_lineage": first["source_lineage"],
            "official_access": first["official_access"],
            "one_shot_attempt": lock,
            "submission_ready": False,
            "next_action": (
                "independent clean confirmation only; no official materialization"
                if gate["passed"]
                else "retain clean incumbent; do not materialize or submit this challenger"
            ),
        }
        result_path = output / "result.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        terminal = {
            **result,
            "result_sha256": sha256(result_path),
            "terminal": True,
        }
        (output / "terminal_result.json").write_text(
            json.dumps(terminal, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return terminal
    except BaseException as exc:
        failure = {
            "schema_version": "p3.clean_fractional_change.terminal.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - started,
            "fit_count": 0,
            "source_lineage": first["source_lineage"],
            "official_access": first["official_access"],
            "one_shot_attempt": lock,
            "terminal": True,
            "automatic_restart_forbidden": True,
        }
        (output / "terminal_result.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise


def environment_paths() -> tuple[Path, Path]:
    root_text = os.environ.get("P3_WORKSPACE_ROOT")
    data_text = os.environ.get("P3_DATA_DIR")
    if not root_text or not data_text:
        raise CleanCycleError("P3_WORKSPACE_ROOT and P3_DATA_DIR are required")
    return Path(root_text), Path(data_text)
