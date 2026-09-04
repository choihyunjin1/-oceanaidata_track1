"""Split-repaired clean P3 fractional-change residual confirmation.

This successor preserves the c1 scientific contract and changes only the validation
split constructor so it exactly matches the clean corrected repeated-forward surface.
It has no test inference, submission materialization, or upload surface.
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

from .clean_fractional_change_residual_20260901_c1 import (
    CleanCycleError,
    blend_with_clean_fallback,
    canonical_json_bytes,
    evaluate_gate,
    fractional_target,
    restore_delta,
    sha256,
    strict_json,
)
from .corrected_repeated_forward import build_corrected_repeated_forward_folds
from .models import ResidualRegressor, compact_feature_columns, threshold_case_weights
from .one_shot_guard import acquire_persistent_attempt_lock
from .revin_patch import assign_storm_episodes_from_wave
from .validation import expand_leads, metric_slices

EXPERIMENT_ID = "p3_clean_fractional_change_residual_20260901_c1r1"
CONFIG_RELATIVE = f"configs/experiments/{EXPERIMENT_ID}.json"
RUNNER_RELATIVE = f"scripts/run_{EXPERIMENT_ID}.py"
MODULE_RELATIVE = "src/p3_wave/clean_fractional_change_residual_20260901_c1r1.py"
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
        "clean_fallback/validation_keys.parquet",
        "superseded/terminal_result.json",
        "superseded/attempt_lock.json",
    }
)


def _input_paths(root: Path, data_dir: Path) -> dict[str, Path]:
    cache = root / "artifacts" / "p3" / "features_all20_v1"
    fallback = root / "artifacts" / "p3_corrected_repeated_forward_catboost_v2"
    superseded = root / "artifacts" / "p3_clean_fractional_change_residual_20260901_c1"
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
        "clean_fallback/validation_keys.parquet": fallback / "validation_keys.parquet",
        "superseded/terminal_result.json": superseded / "terminal_result.json",
        "superseded/attempt_lock.json": (
            root
            / "artifacts"
            / "p3_clean_fractional_change_residual_20260901_c1.ATTEMPT_LOCK.json"
        ),
    }


def _verify_negative_fingerprint(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {CONFIG_RELATIVE, RUNNER_RELATIVE, MODULE_RELATIVE, TEST_RELATIVE}
    observed: set[str] = set()
    token = str(config["negative_fingerprint"]["namespace_token"])
    for relative_root in ("configs/experiments", "scripts", "src/p3_wave", "tests"):
        for path in (root / relative_root).rglob("*"):
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


def _canonical_validation_keys(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["fold", "anchor_id", "station", "episode_id"]
    if not set(columns).issubset(frame.columns):
        raise CleanCycleError("validation key surface is missing required columns")
    result = frame.loc[:, columns].copy()
    result["fold"] = result["fold"].astype(str)
    result["anchor_id"] = result["anchor_id"].astype(np.int64)
    result["station"] = result["station"].astype(str)
    result["episode_id"] = result["episode_id"].astype(np.int64)
    return result.sort_values(columns).reset_index(drop=True)


def assert_validation_surface_matches(
    selected: pd.DataFrame,
    reference: pd.DataFrame,
) -> None:
    observed = _canonical_validation_keys(selected)
    expected = _canonical_validation_keys(reference)
    if not observed.equals(expected):
        raise CleanCycleError("corrected validation keys differ from the clean fallback receipt")


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
        "shared_fractional_module": (
            workspace / "src/p3_wave/clean_fractional_change_residual_20260901_c1.py"
        ),
        "corrected_split_module": workspace / "src/p3_wave/corrected_repeated_forward.py",
        "episode_module": workspace / "src/p3_wave/revin_patch.py",
    }
    for name, path in implementation.items():
        resolved = path.resolve(strict=True)
        if sha256(resolved) != str(config["implementation_sha256"][name]):
            raise CleanCycleError(f"implementation hash mismatch: {name}")

    paths = _input_paths(workspace, source)
    if set(paths) != ALLOWED_INPUT_LABELS:
        raise CleanCycleError("runtime input surface is not exact")
    verified_inputs: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        resolved = path.resolve(strict=True)
        expected = config["inputs"][label]
        observed = {"bytes": resolved.stat().st_size, "sha256": sha256(resolved)}
        if observed["bytes"] != int(expected["bytes"]) or observed["sha256"] != str(
            expected["sha256"]
        ):
            raise CleanCycleError(f"input pin mismatch: {label}")
        verified_inputs[label] = observed

    registry = strict_json(paths["policy/p3_clean_incumbent_20260901.json"])
    lineage = registry.get("lineage", {})
    if not lineage.get("organizer_distributed_data_only"):
        raise CleanCycleError("clean incumbent registry lacks distributed-data-only seal")
    if not lineage.get("models_trained_from_scratch"):
        raise CleanCycleError("clean incumbent registry lacks scratch-model seal")
    if lineage.get("forbidden_lineages_referenced") != []:
        raise CleanCycleError("clean incumbent registry contains a forbidden lineage")

    prior = strict_json(paths["superseded/terminal_result.json"])
    if prior.get("status") != "TERMINAL_TECHNICAL_FAILURE" or int(prior.get("fit_count", -1)) != 0:
        raise CleanCycleError("superseded receipt is not the expected zero-fit technical failure")
    prior_lock = strict_json(paths["superseded/attempt_lock.json"])
    if prior_lock.get("status") != "ATTEMPT_CONSUMED_ONE_SHOT":
        raise CleanCycleError("superseded one-shot lock is not consumed")

    validation_keys = pd.read_parquet(paths["clean_fallback/validation_keys.parquet"])
    expected_counts = {"2024_h2_storm": 49, "winter_transition": 79, "2025_h1": 53}
    counts = {str(key): int(value) for key, value in validation_keys.groupby("fold").size().items()}
    if len(validation_keys) != 181 or validation_keys["anchor_id"].duplicated().any():
        raise CleanCycleError("clean validation key receipt has an unexpected shape")
    if counts != expected_counts:
        raise CleanCycleError("clean validation fold counts differ")

    output = (workspace / OUTPUT_RELATIVE).resolve(strict=False)
    lock = (workspace / LOCK_RELATIVE).resolve(strict=False)
    if output.exists() or lock.exists():
        raise CleanCycleError("fresh output or one-shot lock already exists")

    return {
        "schema_version": "p3.clean_fractional_change.split_repair.preflight.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "READY",
        "config_sha256": sha256(config_path),
        "repair_scope": "split constructor only; c1 science unchanged",
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
        "validation_receipt": {"cases": 181, "rows": 1086, "fold_counts": counts},
        "negative_fingerprint": _verify_negative_fingerprint(workspace, config),
        "verified_inputs": verified_inputs,
    }


def execute(*, root: Path, data_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    completed_fits = 0
    workspace = root.resolve(strict=True)
    source = data_dir.resolve(strict=True)
    first = preflight(root=workspace, data_dir=source)
    second = preflight(root=workspace, data_dir=source)
    first_bytes = canonical_json_bytes(first)
    if first_bytes != canonical_json_bytes(second):
        raise CleanCycleError("the two READY preflights are not byte-identical")

    config_path = workspace / CONFIG_RELATIVE
    config = strict_json(config_path)
    lock = acquire_persistent_attempt_lock(
        workspace / LOCK_RELATIVE,
        experiment_id=EXPERIMENT_ID,
        config_sha256=sha256(config_path),
        created_at=str(config["created_at_kst"]),
    )
    output = workspace / OUTPUT_RELATIVE
    output.mkdir(parents=True, exist_ok=False)
    (output / "preflight.json").write_bytes(first_bytes)

    try:
        cache = workspace / "artifacts" / "p3" / "features_all20_v1"
        fallback_dir = workspace / "artifacts" / "p3_corrected_repeated_forward_catboost_v2"
        features = pd.read_parquet(cache / "train_features.parquet")
        anchors = pd.read_parquet(cache / "train_anchors.parquet")
        fallback_oof = pd.read_parquet(fallback_dir / "oof.parquet")
        reference_keys = pd.read_parquet(fallback_dir / "validation_keys.parquet")
        if len(features) != 24_360 or len(anchors) != 24_360 or len(fallback_oof) != 1_086:
            raise CleanCycleError("train-only surface row-count mismatch")
        if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
            raise CleanCycleError("feature/anchor alignment mismatch")

        wave = pd.read_csv(source / "train_wave.csv")
        wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
        anchors = assign_storm_episodes_from_wave(anchors, wave)
        folds, selected, split_audit = build_corrected_repeated_forward_folds(
            anchors,
            windows=config["validation"]["windows"],
            gap_hours=int(config["validation"]["embargo_hours"]),
            footprint_hours=int(config["validation"]["footprint_hours"]),
        )
        assert_validation_surface_matches(selected, reference_keys)

        feature_columns = compact_feature_columns(
            [column for column in features.columns if column not in {"anchor_id", "station"}]
        )
        if len(feature_columns) != int(config["features"]["expected_feature_count"]):
            raise CleanCycleError("fixed clean feature count mismatch")

        predictions: list[pd.DataFrame] = []
        receipts: list[dict[str, Any]] = []
        seeds = list(config["model"]["fold_seeds"])
        for fold, seed in zip(folds, seeds, strict=True):
            fold_started = time.perf_counter()
            expected_validation = fallback_oof.loc[
                fallback_oof["fold"].eq(fold.name), "anchor_id"
            ].drop_duplicates()
            if set(expected_validation.astype(int)) != set(fold.validation_ids.astype(int)):
                raise CleanCycleError(f"validation surface differs from clean fallback: {fold.name}")

            train_x, train_delta, train_meta = expand_leads(
                features, anchors, fold.train_ids, feature_columns
            )
            valid_x, _, valid_meta = expand_leads(
                features, anchors, fold.validation_ids, feature_columns
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
            completed_fits += 1
            valid_current = valid_meta["current_hs"].to_numpy(dtype=np.float64)
            challenger = valid_current + restore_delta(
                model.predict_delta(valid_x), valid_current, offset_m=offset
            )
            challenger = np.clip(challenger, *map(float, config["gate"]["prediction_clip_m"]))
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
            receipts.append(
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
        oof_path = output / "oof.parquet"
        oof.to_parquet(oof_path, index=False)
        result = {
            "schema_version": "p3.clean_fractional_change.split_repair.result.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": (
                "COMPLETE_GO_LOCAL_CONFIRMATION"
                if gate["passed"]
                else "COMPLETE_NO_GO_CLEAN_FRACTIONAL_CHANGE"
            ),
            "fit_count": completed_fits,
            "runtime_seconds": time.perf_counter() - started,
            "repair_scope": "split constructor only; c1 science unchanged",
            "gate": gate,
            "metrics": {
                "clean_fallback": metric_slices(
                    oof, oof["clean_fallback_prediction"].to_numpy(dtype=np.float64)
                ),
                "candidate": metric_slices(
                    oof, oof["candidate_prediction"].to_numpy(dtype=np.float64)
                ),
            },
            "split_audit": split_audit,
            "validation_keys_sha256": first["verified_inputs"][
                "clean_fallback/validation_keys.parquet"
            ]["sha256"],
            "training_receipts": receipts,
            "input_hashes": first["verified_inputs"],
            "preflight_sha256": hashlib.sha256(first_bytes).hexdigest(),
            "oof": {"rows": int(len(oof)), "sha256": sha256(oof_path)},
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
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        terminal = {**result, "result_sha256": sha256(result_path), "terminal": True}
        (output / "terminal_result.json").write_text(
            json.dumps(terminal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return terminal
    except BaseException as exc:
        failure = {
            "schema_version": "p3.clean_fractional_change.split_repair.terminal.v1",
            "experiment_id": EXPERIMENT_ID,
            "status": "TERMINAL_TECHNICAL_FAILURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runtime_seconds": time.perf_counter() - started,
            "fit_count": completed_fits,
            "source_lineage": first["source_lineage"],
            "official_access": first["official_access"],
            "one_shot_attempt": lock,
            "terminal": True,
            "automatic_restart_forbidden": True,
        }
        (output / "terminal_result.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise


def environment_paths() -> tuple[Path, Path]:
    root_text = os.environ.get("P3_WORKSPACE_ROOT")
    data_text = os.environ.get("P3_DATA_DIR")
    if not root_text or not data_text:
        raise CleanCycleError("P3_WORKSPACE_ROOT and P3_DATA_DIR are required")
    return Path(root_text), Path(data_text)
