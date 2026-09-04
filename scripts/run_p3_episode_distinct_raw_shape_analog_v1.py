"""Dry-run and one-shot inner-only runner for the P3 raw-shape analog v1.

This runner has no outer or test mode.  It never opens the frozen 182-case OOF,
``test_context.parquet``, ``test_index.csv`` or any submission file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from p3_trajectory import build_trajectory_dataset
from p3_wave.episode_distinct_analog import (
    ACTIVE_LEADS,
    HISTORY_POINTS,
    LEADS,
    EpisodeAnalogError,
    EpisodeAnalogIndex,
    banded_dtw_distances,
    blend_candidate,
    evaluate_b_precheck,
    evaluate_inner_gate,
    lb_keogh_distances,
    prepare_histories,
    project_normalized_residual,
)
from p3_wave.models import threshold_case_weights
from p3_wave.revin_patch import build_inner_episode_split

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_episode_distinct_raw_shape_analog_v1"
CONFIG_PATH = (ROOT / f"configs/experiments/{EXPERIMENT_ID}.json").resolve()
DRY_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/dry_run").resolve()
INNER_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/inner_one_shot").resolve()
STATUS_PATH = (ROOT / f"artifacts/status/{EXPERIMENT_ID}.json").resolve()
ATTEMPT_LOCK = (
    ROOT / f"artifacts/experiment_locks/{EXPERIMENT_ID}.inner.attempt.lock"
).resolve()
CONFIRMATION_TOKEN = "RUN_P3_EPISODE_DISTINCT_ANALOG_INNER_ONCE"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: Path) -> str:
    from hashlib import sha256

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    with temporary.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, ensure_ascii=False, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())


def _status(
    *,
    state: str,
    phase: str,
    progress: float,
    detail: str,
    started: float,
    result: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "title": "P3 episode-distinct raw-shape analog v1",
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "phase": phase,
        "progress": float(progress),
        "detail": detail,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
        "updated_at": _now(),
    }
    if result is not None:
        payload["result"] = dict(result)
    _atomic_json(STATUS_PATH, payload)


def _load_config(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if resolved != CONFIG_PATH:
        raise PermissionError("config override is prohibited for this one-shot runner")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise EpisodeAnalogError("experiment id drifted")
    analog = config["analog"]
    exact = {
        "history_cadence_minutes": 20,
        "history_hours": 48,
        "history_points_including_current": 145,
        "mad_floor_m": 0.1,
        "station_scope": "same_station_only",
        "sakoe_chiba_radius_steps": 6,
        "neighbor_count": 8,
    }
    if any(analog.get(key) != value for key, value in exact.items()):
        raise EpisodeAnalogError("frozen analog structure changed")
    candidate = config["candidate"]
    if candidate["no_op_leads"] != [3, 6, 9] or candidate["active_leads"] != [12, 18, 24]:
        raise EpisodeAnalogError("frozen candidate lead routing changed")
    if candidate["alpha"] != 0.2:
        raise EpisodeAnalogError("frozen candidate alpha changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise EpisodeAnalogError("a safety prohibition was disabled")
    return config


def _resolve_data_dir(argument: str | None) -> Path:
    raw = argument or os.environ.get("P3_DATA_DIR")
    if not raw:
        raise FileNotFoundError("provide --p3-data-dir or set P3_DATA_DIR")
    root = Path(raw).resolve()
    if not (root / "train_wave.csv").is_file() or not (root / "README.md").is_file():
        raise FileNotFoundError("P3 data directory lacks train_wave.csv or README.md")
    return root


def _implementation_hashes() -> dict[str, str]:
    relatives = (
        f"configs/experiments/{EXPERIMENT_ID}.json",
        "src/p3_wave/episode_distinct_analog.py",
        f"scripts/run_{EXPERIMENT_ID}.py",
        f"tests/test_{EXPERIMENT_ID}.py",
        "src/p3_trajectory.py",
        "src/p3_wave/revin_patch.py",
        "src/p3_wave/models.py",
    )
    result: dict[str, str] = {}
    for relative in relatives:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"implementation input is missing: {relative}")
        result[relative] = _sha256(path)
    return result


def _verify_registered_inputs(
    config: Mapping[str, Any], data_dir: Path
) -> dict[str, str]:
    registered = config["registered_inputs"]
    paths = {
        "train_wave": data_dir / registered["train_wave"]["source_name"],
        "source_readme": data_dir / registered["source_readme"]["source_name"],
        "train_features": ROOT / registered["train_features"]["path"],
        "anchor_metadata_schema_reference": (
            ROOT / registered["anchor_metadata_schema_reference"]["path"]
        ),
        "feature_columns": ROOT / registered["feature_columns"]["path"],
        "trajectory_helper": ROOT / registered["trajectory_helper"]["path"],
        "episode_split_helper": ROOT / registered["episode_split_helper"]["path"],
        "weight_helper": ROOT / registered["weight_helper"]["path"],
    }
    result: dict[str, str] = {}
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"registered input is missing: {role}")
        actual = _sha256(path)
        if actual != registered[role]["sha256"]:
            raise EpisodeAnalogError(f"registered SHA changed: {role}")
        result[str(path)] = actual
    return result


def _schema_preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    registered = config["registered_inputs"]
    feature_path = ROOT / registered["train_features"]["path"]
    anchor_path = ROOT / registered["anchor_metadata_schema_reference"]["path"]
    feature_schema = parquet.read_schema(feature_path)
    anchor_schema = parquet.read_schema(anchor_path)
    feature_columns = json.loads(
        (ROOT / registered["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    if len(feature_columns) != 591 or len(set(feature_columns)) != 591:
        raise EpisodeAnalogError("frozen feature list is not 591 unique columns")
    if not {"anchor_id", "station", *feature_columns}.issubset(feature_schema.names):
        raise EpisodeAnalogError("feature cache lacks a frozen feature")
    expected_anchor = {
        "anchor_id",
        "station",
        "anchor_time",
        "grid_position",
        "current_hs",
        *[f"target_{lead}" for lead in LEADS],
    }
    if set(anchor_schema.names) != expected_anchor:
        raise EpisodeAnalogError("anchor schema reference changed")
    return {
        "feature_schema_column_count": len(feature_schema.names),
        "frozen_feature_count": len(feature_columns),
        "anchor_schema_columns": anchor_schema.names,
        "anchor_values_read": 0,
        "target_values_read": 0,
    }


def _synthetic_contract() -> dict[str, Any]:
    x = np.linspace(-1.0, 1.0, HISTORY_POINTS)
    candidates = np.vstack([x, x + 0.1])
    dtw = banded_dtw_distances(x, candidates, radius_steps=6)
    bound = lb_keogh_distances(x, candidates, radius_steps=6)
    history = np.vstack([2.0 + 0.2 * x, np.full(HISTORY_POINTS, 2.0)])
    prepared = prepare_histories(history, minimum_coverage=0.95, mad_floor=0.1)
    control = np.arange(6, dtype=np.float64) + 1.0
    analog = control + 1.0
    blended = blend_candidate(
        control,
        analog,
        np.asarray(LEADS, dtype=np.int64),
        alpha=0.2,
    )
    if not np.all(bound <= dtw + 1e-12):
        raise AssertionError("synthetic lower bound is not admissible")
    if not np.array_equal(blended[:3], control[:3]):
        raise AssertionError("synthetic short-lead no-op failed")
    return {
        "history_points": HISTORY_POINTS,
        "dtw_identity": float(dtw[0]),
        "lb_admissible": True,
        "prepared_finite": bool(np.isfinite(prepared.normalized).all()),
        "short_lead_exact_no_op": True,
        "active_lead_delta": (blended[3:] - control[3:]).tolist(),
        "model_fit_count": 0,
        "target_value_read_count": 0,
    }


def _git_provenance() -> dict[str, Any]:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_sha = "unavailable"
        dirty = True
    return {"git_sha": git_sha, "git_dirty": dirty}


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "pyarrow", "catboost", "scipy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }


def _dry_run(config: Mapping[str, Any], data_dir: Path, started: float) -> int:
    _status(
        state="dry_running",
        phase="hash_schema_synthetic_preflight",
        progress=20.0,
        detail="등록 SHA·schema·합성 계약 확인 중; model/target/outer/test/submission 0",
        started=started,
    )
    if ATTEMPT_LOCK.exists() or INNER_DIRECTORY.exists():
        raise PermissionError("inner one-shot state already exists")
    receipts = _verify_registered_inputs(config, data_dir)
    schema = _schema_preflight(config)
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "mode": "dry-run",
        "status": "READY_FOR_AUTHORIZED_INNER_ONLY_ONE_SHOT",
        "created_at": _now(),
        "registered_input_sha256": receipts,
        "implementation_sha256": _implementation_hashes(),
        "schema_preflight": schema,
        "synthetic_contract": _synthetic_contract(),
        "environment": _environment(),
        **_git_provenance(),
        "model_fit_count": 0,
        "target_value_read_count": 0,
        "outer_membership_read_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    DRY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DRY_DIRECTORY / "receipt.json"
    _atomic_json(path, receipt)
    result = {**receipt, "receipt_sha256": _sha256(path)}
    _status(
        state="dry_ready",
        phase="ready_for_inner_only",
        progress=100.0,
        detail="dry-run 완료; model/target/outer/test/submission 0",
        started=started,
        result={"receipt_sha256": result["receipt_sha256"]},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _read_wave(data_dir: Path) -> pd.DataFrame:
    wave = pd.read_csv(data_dir / "train_wave.csv")
    expected = {"station", "time", "hs"}
    if not expected.issubset(wave):
        raise EpisodeAnalogError(f"train_wave lacks columns: {sorted(expected - set(wave))}")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    if wave.duplicated(["station", "time"]).any():
        raise EpisodeAnalogError("train_wave keys are duplicated")
    return wave


def _inner_splits(dataset: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    anchors = dataset.anchors
    result: dict[str, Any] = {}
    for name, start, _ in config["validation"]["outer_windows_used_only_as_training_cutoffs"]:
        start_timestamp = pd.Timestamp(start, tz="UTC")
        cutoff = start_timestamp - pd.Timedelta(hours=78)
        outer_train_ids = anchors.loc[
            anchors["anchor_time"].lt(cutoff), "anchor_id"
        ].to_numpy(dtype=np.int64)
        split = build_inner_episode_split(
            anchors,
            outer_train_ids,
            validation_days=45,
            embargo_hours=78,
        )
        result[str(name)] = split
    return result


def _assert_inner_safety(dataset: Any, splits: Mapping[str, Any]) -> dict[str, Any]:
    lookup = dataset.anchors.set_index("anchor_id")
    result: dict[str, Any] = {}
    for name, split in splits.items():
        train = lookup.loc[split.train_ids]
        validation = lookup.loc[split.validation_ids]
        minimum_gap = np.inf
        for station, current in validation.groupby("station", observed=True):
            history = train.loc[train["station"].eq(station), "anchor_time"]
            if history.empty:
                raise EpisodeAnalogError(f"{name} lacks same-station training history")
            for timestamp in current["anchor_time"]:
                gap = (pd.Timestamp(timestamp) - history).dt.total_seconds().min() / 3600.0
                minimum_gap = min(minimum_gap, float(gap))
        shared = set(zip(train["station"], train["episode_id"], strict=True)).intersection(
            zip(validation["station"], validation["episode_id"], strict=True)
        )
        if shared or minimum_gap < 78.0:
            raise EpisodeAnalogError(f"{name} violates episode or 78h separation")
        result[name] = {
            "fit_anchors": int(len(train)),
            "inner_cases": int(len(validation)),
            "inner_cases_by_station": {
                str(key): int(value)
                for key, value in validation.groupby("station", observed=True).size().items()
            },
            "shared_station_episode_count": 0,
            "minimum_same_station_origin_gap_hours": float(minimum_gap),
            "validation_start": split.validation_start.isoformat(),
            "validation_end": split.validation_end.isoformat(),
        }
    return result


def _run_analog_precheck(
    *,
    dataset: Any,
    prepared: Any,
    splits: Mapping[str, Any],
    config: Mapping[str, Any],
    started: float,
) -> tuple[pd.DataFrame, dict[str, dict[int, np.ndarray]], dict[str, Any]]:
    analog_config = config["analog"]
    b_config = config["b_precheck"]
    anchors = dataset.anchors.set_index("anchor_id")
    normalized_residual = (
        dataset.official_target.astype(np.float64)
        - dataset.anchors["current_hs"].to_numpy(dtype=np.float64)[:, None]
    ) / prepared.scale[:, None]
    records: list[dict[str, Any]] = []
    predictions: dict[str, dict[int, np.ndarray]] = {}
    search_audit: dict[str, Any] = {}
    total_cases = sum(len(split.validation_ids) for split in splits.values())
    completed = 0

    for fold_name, split in splits.items():
        predictions[fold_name] = {}
        search_audit[fold_name] = {
            "eligible_queries": 0,
            "ineligible_queries": 0,
            "evaluated_candidates": 0,
            "total_candidates_if_exhaustive": 0,
            "library_anchors_by_station": {},
            "library_episodes_by_station": {},
        }
        station_indices: dict[str, tuple[EpisodeAnalogIndex, np.ndarray]] = {}
        fit_ids = np.asarray(split.train_ids, dtype=np.int64)
        for station in sorted(dataset.anchors["station"].astype(str).unique()):
            station_ids = fit_ids[
                anchors.loc[fit_ids, "station"].astype(str).to_numpy() == station
            ]
            station_ids = station_ids[prepared.eligible[station_ids]]
            index = EpisodeAnalogIndex(
                anchor_ids=station_ids,
                episode_ids=anchors.loc[station_ids, "episode_id"].to_numpy(dtype=np.int64),
                normalized_histories=prepared.normalized[station_ids],
                radius_steps=int(analog_config["sakoe_chiba_radius_steps"]),
                neighbor_count=int(analog_config["neighbor_count"]),
                batch_size=int(analog_config["exact_search_batch_size"]),
            )
            station_indices[station] = (index, station_ids)
            search_audit[fold_name]["library_anchors_by_station"][station] = int(len(station_ids))
            search_audit[fold_name]["library_episodes_by_station"][station] = int(
                len(index.unique_episodes)
            )

        for anchor_id in split.validation_ids:
            anchor_id = int(anchor_id)
            completed += 1
            if not prepared.eligible[anchor_id]:
                predictions[fold_name][anchor_id] = np.full(len(LEADS), np.nan)
                search_audit[fold_name]["ineligible_queries"] += 1
                continue
            station = str(anchors.loc[anchor_id, "station"])
            index, station_ids = station_indices[station]
            query = prepared.normalized[anchor_id]
            nearest = index.select_nearest(query)
            nearest_projection = project_normalized_residual(
                normalized_residual[station_ids[nearest.indices]],
                nearest.distances,
                distance_floor=float(analog_config["distance_floor"]),
            )
            random_panels = index.select_random_panels(
                query,
                seed=int(b_config["random_seed"]),
                query_key=f"{fold_name}|{anchor_id}",
                panel_count=int(b_config["random_panel_count"]),
            )
            random_projection = [
                project_normalized_residual(
                    normalized_residual[station_ids[selection.indices]],
                    selection.distances,
                    distance_floor=float(analog_config["distance_floor"]),
                )
                for selection in random_panels
            ]
            truth = normalized_residual[anchor_id]
            nearest_mse = float(np.mean(np.square(nearest_projection - truth)))
            random_mse = float(
                np.mean(
                    [np.mean(np.square(projection - truth)) for projection in random_projection]
                )
            )
            current = float(anchors.loc[anchor_id, "current_hs"])
            predictions[fold_name][anchor_id] = np.clip(
                current + prepared.scale[anchor_id] * nearest_projection,
                0.0,
                30.0,
            )
            records.append(
                {
                    "fold": fold_name,
                    "anchor_id": anchor_id,
                    "station": station,
                    "nearest_normalized_mse": nearest_mse,
                    "random_normalized_mse": random_mse,
                    "nearest_anchor_ids_sha256": _hash_integer_array(
                        index.anchor_ids[nearest.indices]
                    ),
                    "nearest_episode_ids_sha256": _hash_integer_array(nearest.episodes),
                    "nearest_max_distance": float(nearest.distances.max()),
                    "nearest_mean_distance": float(nearest.distances.mean()),
                    "evaluated_candidates": int(nearest.evaluated_candidates),
                    "total_candidates": int(nearest.total_candidates),
                }
            )
            search_audit[fold_name]["eligible_queries"] += 1
            search_audit[fold_name]["evaluated_candidates"] += int(
                nearest.evaluated_candidates
            )
            search_audit[fold_name]["total_candidates_if_exhaustive"] += int(
                nearest.total_candidates
            )
            if completed % 10 == 0 or completed == total_cases:
                _status(
                    state="inner_running_pre_outer",
                    phase="B_nearest_vs_random_precheck",
                    progress=10.0 + 40.0 * completed / max(total_cases, 1),
                    detail=(
                        f"B raw-shape retrieval {completed}/{total_cases}; "
                        "outer/test/submission 0"
                    ),
                    started=started,
                )
    frame = pd.DataFrame.from_records(records)
    gate = evaluate_b_precheck(
        frame,
        maximum_fold_mse_ratio=float(
            b_config["maximum_nearest_to_random_mse_ratio_per_passing_fold"]
        ),
        minimum_passing_folds=int(b_config["minimum_passing_folds"]),
    )
    return frame, predictions, {"gate": gate, "search": search_audit}


def _hash_integer_array(values: np.ndarray) -> str:
    from hashlib import sha256

    array = np.asarray(values, dtype="<i8")
    return sha256(array.tobytes()).hexdigest()


def _expand_fit_rows(
    *,
    features: pd.DataFrame,
    dataset: Any,
    anchor_ids: Sequence[int],
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = dataset.anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    metadata_rows: list[pd.DataFrame] = []
    for column, lead in enumerate(LEADS):
        block = feature_lookup.loc[ids, list(feature_columns)].reset_index(drop=True)
        station = anchor_lookup.loc[ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[ids, "current_hs"].to_numpy(dtype=np.float64)
        target = dataset.official_target[ids, column].astype(np.float64)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", str(lead))
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        targets.append(target - current)
        metadata_rows.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                    "target_hs": target,
                }
            )
        )
    return (
        pd.concat(blocks, ignore_index=True),
        np.concatenate(targets),
        pd.concat(metadata_rows, ignore_index=True),
    )


def _expand_predict_rows(
    *,
    features: pd.DataFrame,
    dataset: Any,
    anchor_ids: Sequence[int],
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = dataset.anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    metadata_rows: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[ids, list(feature_columns)].reset_index(drop=True)
        station = anchor_lookup.loc[ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[ids, "current_hs"].to_numpy(dtype=np.float64)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", str(lead))
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        metadata_rows.append(
            pd.DataFrame(
                {
                    "anchor_id": ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                }
            )
        )
    return pd.concat(blocks, ignore_index=True), pd.concat(metadata_rows, ignore_index=True)


def _control_model(parameters: Mapping[str, Any]) -> Any:
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        iterations=int(parameters["iterations"]),
        learning_rate=float(parameters["learning_rate"]),
        depth=int(parameters["depth"]),
        l2_leaf_reg=float(parameters["l2_leaf_reg"]),
        random_strength=float(parameters["random_strength"]),
        random_seed=int(parameters["random_seed"]),
        loss_function=str(parameters["loss_function"]),
        eval_metric="RMSE",
        task_type=str(parameters["task_type"]),
        thread_count=int(parameters["thread_count"]),
        allow_writing_files=bool(parameters["allow_writing_files"]),
        verbose=False,
    )


def _apply_control_shrink(
    prediction: np.ndarray, current: np.ndarray, lead_h: np.ndarray
) -> np.ndarray:
    result = np.asarray(prediction, dtype=np.float64).copy()
    active = np.isin(np.asarray(lead_h, dtype=np.int64), ACTIVE_LEADS)
    result[active] = 0.8 * result[active] + 0.2 * np.asarray(current, dtype=np.float64)[active]
    return np.clip(result, 0.0, 30.0)


def _run_c_gate(
    *,
    dataset: Any,
    splits: Mapping[str, Any],
    analog_predictions: Mapping[str, Mapping[int, np.ndarray]],
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    config: Mapping[str, Any],
    started: float,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, str]]:
    parameters = config["inner_control_proxy"]
    rows: list[pd.DataFrame] = []
    model_hashes: dict[str, str] = {}
    for fold_index, (fold_name, split) in enumerate(splits.items()):
        fit_matrix, fit_target, fit_meta = _expand_fit_rows(
            features=features,
            dataset=dataset,
            anchor_ids=split.train_ids,
            feature_columns=feature_columns,
        )
        validation_matrix, validation_meta = _expand_predict_rows(
            features=features,
            dataset=dataset,
            anchor_ids=split.validation_ids,
            feature_columns=feature_columns,
        )
        model = _control_model(parameters)
        model.fit(
            fit_matrix,
            fit_target,
            sample_weight=threshold_case_weights(
                fit_meta["current_hs"].to_numpy(dtype=np.float64)
            ),
            cat_features=[0, 1],
            verbose=False,
        )
        model_path = INNER_DIRECTORY / f"control_proxy_{fold_name}.cbm"
        model.save_model(model_path)
        model_hashes[fold_name] = _sha256(model_path)
        residual_prediction = np.asarray(
            model.predict(validation_matrix), dtype=np.float64
        )
        control_single = np.clip(
            validation_meta["current_hs"].to_numpy(dtype=np.float64)
            + residual_prediction,
            0.0,
            30.0,
        )
        validation_meta.insert(0, "fold", fold_name)
        validation_meta["control_single"] = control_single
        validation_meta["control_final"] = _apply_control_shrink(
            control_single,
            validation_meta["current_hs"].to_numpy(dtype=np.float64),
            validation_meta["lead_h"].to_numpy(dtype=np.int64),
        )
        lead_to_column = {lead: column for column, lead in enumerate(LEADS)}
        target_columns = validation_meta["lead_h"].map(lead_to_column).to_numpy(dtype=np.int64)
        validation_meta["target_hs"] = dataset.official_target[
            validation_meta["anchor_id"].to_numpy(dtype=np.int64), target_columns
        ].astype(np.float64)
        analog = np.full(len(validation_meta), np.nan, dtype=np.float64)
        for position, row in enumerate(validation_meta.itertuples(index=False)):
            path = analog_predictions[fold_name][int(row.anchor_id)]
            lead_column = LEADS.index(int(row.lead_h))
            analog[position] = path[lead_column]
        validation_meta["analog_prediction"] = analog
        validation_meta["candidate_final"] = blend_candidate(
            validation_meta["control_final"].to_numpy(dtype=np.float64),
            analog,
            validation_meta["lead_h"].to_numpy(dtype=np.int64),
            alpha=float(config["candidate"]["alpha"]),
        )
        short = validation_meta["lead_h"].isin([3, 6, 9])
        if not np.array_equal(
            validation_meta.loc[short, "candidate_final"].to_numpy(),
            validation_meta.loc[short, "control_final"].to_numpy(),
        ):
            raise AssertionError("short-lead candidate differs from exact inner control")
        rows.append(validation_meta)
        _status(
            state="inner_running_pre_outer",
            phase="C_fixed_control_proxy_gate",
            progress=60.0 + 10.0 * (fold_index + 1),
            detail=f"C {fold_name} control proxy 완료; outer/test/submission 0",
            started=started,
        )
    result = pd.concat(rows, ignore_index=True).sort_values(
        ["fold", "anchor_id", "lead_h"]
    ).reset_index(drop=True)
    gate_config = config["validation"]["inner_gate"]
    gate = evaluate_inner_gate(
        result,
        maximum_pooled_delta_m=float(
            gate_config["maximum_pooled_full_six_lead_delta_m"]
        ),
        minimum_improved_folds=int(gate_config["minimum_strictly_improved_folds"]),
        maximum_station_degradation_m=float(
            gate_config["maximum_any_station_rmse_degradation_m"]
        ),
    )
    return result, gate, model_hashes


def _load_feature_cache(
    config: Mapping[str, Any], dataset: Any
) -> tuple[pd.DataFrame, list[str]]:
    registered = config["registered_inputs"]
    feature_columns = json.loads(
        (ROOT / registered["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    features = pd.read_parquet(
        ROOT / registered["train_features"]["path"],
        columns=["anchor_id", "station", *feature_columns],
    )
    if features["anchor_id"].duplicated().any() or len(features) != len(dataset.anchors):
        raise EpisodeAnalogError("feature cache anchor contract changed")
    expected = np.arange(len(dataset.anchors), dtype=np.int64)
    if not np.array_equal(features["anchor_id"].to_numpy(dtype=np.int64), expected):
        raise EpisodeAnalogError("feature cache anchor ids differ from raw-wave reconstruction")
    if not features["station"].astype(str).equals(dataset.anchors["station"].astype(str)):
        raise EpisodeAnalogError("feature cache station differs from raw-wave reconstruction")
    return features, feature_columns


def _output_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _run_inner(config: Mapping[str, Any], data_dir: Path, started: float) -> int:
    dry_receipt_path = DRY_DIRECTORY / "receipt.json"
    if not dry_receipt_path.is_file():
        raise PermissionError("canonical dry-run receipt is required before inner execution")
    if ATTEMPT_LOCK.exists() or INNER_DIRECTORY.exists():
        raise PermissionError("inner one-shot has already been attempted")
    dry_receipt = json.loads(dry_receipt_path.read_text(encoding="utf-8"))
    implementation = _implementation_hashes()
    if dry_receipt.get("implementation_sha256") != implementation:
        raise PermissionError("implementation changed after dry-run")
    receipts = _verify_registered_inputs(config, data_dir)
    if dry_receipt.get("registered_input_sha256") != receipts:
        raise PermissionError("registered input changed after dry-run")
    lock_payload = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "B_then_C_inner_only_no_outer_no_test_no_submission",
        "dry_receipt_sha256": _sha256(dry_receipt_path),
        "implementation_sha256": implementation,
    }
    _write_exclusive(ATTEMPT_LOCK, lock_payload)
    INNER_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _status(
        state="inner_running_pre_outer",
        phase="raw_wave_reconstruction_and_inner_split",
        progress=5.0,
        detail="raw train_wave에서 episode/145점 history 구성; outer/test/submission 0",
        started=started,
    )
    wave = _read_wave(data_dir)
    dataset = build_trajectory_dataset(wave)
    if len(dataset.anchors) != 24360:
        raise EpisodeAnalogError("raw-wave anchor reconstruction count changed")
    prepared = prepare_histories(
        dataset.history,
        minimum_coverage=float(config["analog"]["minimum_history_coverage"]),
        mad_floor=float(config["analog"]["mad_floor_m"]),
    )
    splits = _inner_splits(dataset, config)
    split_audit = _assert_inner_safety(dataset, splits)

    b_rows, analog_predictions, b_payload = _run_analog_precheck(
        dataset=dataset,
        prepared=prepared,
        splits=splits,
        config=config,
        started=started,
    )
    b_path = INNER_DIRECTORY / "b_precheck_cases.parquet"
    b_result_path = INNER_DIRECTORY / "b_precheck.json"
    _atomic_parquet(b_path, b_rows)
    _atomic_json(b_result_path, b_payload)
    if not b_payload["gate"]["pass"]:
        result = {
            "decision": "NO_GO_B_NEAREST_NOT_BETTER_THAN_RANDOM",
            "b_gate": b_payload["gate"],
            "c_gate_executed": False,
            "model_fit_count": 0,
            "outer_membership_read_count": 0,
            "outer_designated_scoring_open_count": 0,
            "test_context_read_count": 0,
            "submission_write_count": 0,
        }
        _atomic_json(INNER_DIRECTORY / "result.json", result)
        manifest = {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "created_at": _now(),
            "scope": "inner_only_stopped_after_B",
            "decision": result["decision"],
            "split_audit": split_audit,
            "registered_input_sha256": receipts,
            "implementation_sha256": implementation,
            "attempt_lock_sha256": _sha256(ATTEMPT_LOCK),
            "dry_receipt_sha256": _sha256(dry_receipt_path),
            "output_sha256": _output_hashes(INNER_DIRECTORY),
            "environment": _environment(),
            **_git_provenance(),
            "target_access": {
                "inner_validation_cases": int(len(b_rows)),
                "outer_182_membership_or_target_rows": 0,
            },
            "model_fit_count": 0,
            "test_context_read_count": 0,
            "submission_write_count": 0,
        }
        _atomic_json(INNER_DIRECTORY / "manifest.json", manifest)
        _status(
            state="inner_stopped_pre_outer",
            phase="B_gate_no_go",
            progress=100.0,
            detail="B gate 실패; C/model/outer/test/submission 0; 종료",
            started=started,
            result=result,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    _status(
        state="inner_running_pre_outer",
        phase="C_fixed_control_proxy_gate",
        progress=55.0,
        detail="B gate 통과; 591-feature 고정 inner control 3개 학습 시작",
        started=started,
    )
    features, feature_columns = _load_feature_cache(config, dataset)
    inner_rows, c_gate, model_hashes = _run_c_gate(
        dataset=dataset,
        splits=splits,
        analog_predictions=analog_predictions,
        features=features,
        feature_columns=feature_columns,
        config=config,
        started=started,
    )
    predictions_path = INNER_DIRECTORY / "inner_predictions.parquet"
    gate_path = INNER_DIRECTORY / "inner_gate.json"
    _atomic_parquet(predictions_path, inner_rows)
    _atomic_json(gate_path, c_gate)
    decision = "GO_INNER_GATE" if c_gate["pass"] else "NO_GO_INNER_GATE"
    result = {
        "decision": decision,
        "b_gate": b_payload["gate"],
        "c_gate": c_gate,
        "model_fit_count": 3,
        "outer_membership_read_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
        "required_action": (
            "stop_without_outer_or_test_execution"
            if not c_gate["pass"]
            else "report_inner_result_only_and_await_separate_authorization"
        ),
    }
    _atomic_json(INNER_DIRECTORY / "result.json", result)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "B_and_C_inner_only_no_outer_no_test_no_submission",
        "decision": decision,
        "split_audit": split_audit,
        "history_eligible_fraction": float(prepared.eligible.mean()),
        "registered_input_sha256": receipts,
        "implementation_sha256": implementation,
        "attempt_lock_sha256": _sha256(ATTEMPT_LOCK),
        "dry_receipt_sha256": _sha256(dry_receipt_path),
        "control_proxy_model_sha256_by_fold": model_hashes,
        "output_sha256": _output_hashes(INNER_DIRECTORY),
        "environment": _environment(),
        **_git_provenance(),
        "target_access": {
            "inner_validation_cases": int(
                inner_rows[["fold", "anchor_id"]].drop_duplicates().shape[0]
            ),
            "outer_182_membership_or_target_rows": 0,
        },
        "model_fit_count": 3,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    _atomic_json(INNER_DIRECTORY / "manifest.json", manifest)
    _status(
        state="inner_complete_pre_outer",
        phase="C_gate_complete_stop",
        progress=100.0,
        detail=f"{decision}; outer/test/submission 0; 현재 권한 범위 종료",
        started=started,
        result=result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--p3-data-dir")
    parser.add_argument("--mode", choices=("dry-run", "inner"), default="dry-run")
    parser.add_argument("--confirm")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    started = time.perf_counter()
    config = _load_config(arguments.config)
    data_dir = _resolve_data_dir(arguments.p3_data_dir)
    if arguments.mode == "dry-run":
        if arguments.confirm is not None:
            raise PermissionError("dry-run does not accept a confirmation token")
        return _dry_run(config, data_dir, started)
    if arguments.confirm != CONFIRMATION_TOKEN:
        raise PermissionError("inner mode requires the exact one-shot confirmation token")
    return _run_inner(config, data_dir, started)


if __name__ == "__main__":
    raise SystemExit(main())
