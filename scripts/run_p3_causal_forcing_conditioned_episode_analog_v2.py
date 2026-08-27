"""One-shot B-only runner for the adaptive P3 forcing-conditioned analog v2."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as parquet

from p3_trajectory import build_trajectory_dataset
from p3_wave.causal_forcing_analog import (
    FORCING_COLUMNS,
    FORCING_DIMENSIONS,
    ForcingConditionedAnalogIndex,
)
from p3_wave.episode_distinct_analog import (
    EpisodeAnalogError,
    evaluate_b_precheck,
    prepare_histories,
    project_normalized_residual,
)
from p3_wave.revin_patch import build_inner_episode_split

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_causal_forcing_conditioned_episode_analog_v2"
CONFIG_PATH = (ROOT / f"configs/experiments/{EXPERIMENT_ID}.json").resolve()
DRY_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/dry_run").resolve()
B_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/B_one_shot").resolve()
STATUS_PATH = (ROOT / f"artifacts/status/{EXPERIMENT_ID}.json").resolve()
ATTEMPT_LOCK = (
    ROOT / f"artifacts/experiment_locks/{EXPERIMENT_ID}.B.attempt.lock"
).resolve()
CONFIRMATION_TOKEN = "RUN_P3_CAUSAL_FORCING_ANALOG_V2_B_ONCE"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_integer_array(values: np.ndarray) -> str:
    return sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


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
        "title": "P3 causal forcing-conditioned episode analog v2",
        "experiment_id": EXPERIMENT_ID,
        "status": state,
        "phase": phase,
        "progress": float(progress),
        "detail": detail,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "model_fit_count": 0,
        "C_execution_count": 0,
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
        raise PermissionError("config override is prohibited for the v2 one-shot runner")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise EpisodeAnalogError("experiment id drifted")
    forcing = config["forcing_state"]
    if tuple(forcing["columns_in_fixed_order"]) != FORCING_COLUMNS:
        raise EpisodeAnalogError("forcing columns changed after preregistration")
    selection = forcing["distance_and_selection"]
    if (
        selection["forcing_weight"] != 1.0
        or selection["selected_episode_count"] != 8
        or selection["search_batch_size"] != 1024
    ):
        raise EpisodeAnalogError("forcing selection contract changed")
    base = config["v1_contract_preserved"]
    expected = {
        "history_points_including_current": 145,
        "history_mad_floor_m": 0.1,
        "minimum_history_coverage": 0.95,
        "sakoe_chiba_radius_steps": 6,
        "neighbor_count": 8,
        "random_panel_count": 16,
        "random_seed": 20260822,
    }
    if any(base.get(key) != value for key, value in expected.items()):
        raise EpisodeAnalogError("v1 retrieval contract changed")
    gate = config["B_gate"]
    if gate["maximum_nearest_to_random_mse_ratio_per_passing_fold"] != 0.9:
        raise EpisodeAnalogError("B ratio threshold changed")
    if gate["minimum_passing_folds"] != 2 or gate["C_execution_in_v2"] is not False:
        raise EpisodeAnalogError("B/C gate contract changed")
    if config["adaptive_research_disclosure"]["is_adaptive_research"] is not True:
        raise EpisodeAnalogError("adaptive research disclosure was removed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise EpisodeAnalogError("a v2 safety prohibition was disabled")
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
        "src/p3_wave/causal_forcing_analog.py",
        f"scripts/run_{EXPERIMENT_ID}.py",
        f"tests/test_{EXPERIMENT_ID}.py",
        "src/p3_wave/episode_distinct_analog.py",
        "src/p3_trajectory.py",
        "src/p3_wave/revin_patch.py",
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
    }
    for role, details in registered.items():
        if role in paths:
            continue
        paths[role] = ROOT / details["path"]
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
    feature_path = ROOT / config["registered_inputs"]["train_features"]["path"]
    schema = parquet.read_schema(feature_path)
    required = {"anchor_id", "station", *FORCING_COLUMNS}
    if not required.issubset(schema.names):
        raise EpisodeAnalogError(
            f"forcing cache lacks columns: {sorted(required - set(schema.names))}"
        )
    return {
        "feature_cache_schema_columns": len(schema.names),
        "forcing_columns": list(FORCING_COLUMNS),
        "forcing_dimensions": FORCING_DIMENSIONS,
        "feature_values_read": 0,
        "target_values_read": 0,
        "v1_B_values_read": 0,
    }


def _synthetic_contract() -> dict[str, Any]:
    x = np.linspace(-1.0, 1.0, 145)
    histories = np.vstack([x + 0.01 * episode for episode in range(12)])
    forcing = np.vstack(
        [np.full(FORCING_DIMENSIONS, float(episode)) for episode in range(12)]
    )
    index = ForcingConditionedAnalogIndex(
        anchor_ids=np.arange(12),
        episode_ids=np.arange(12),
        normalized_histories=histories,
        forcing_state=forcing,
    )
    selected = index.select_nearest(x, np.zeros(FORCING_DIMENSIONS))
    fallback = index.select_nearest(x, np.full(FORCING_DIMENSIONS, np.nan))
    if not selected.conditioning_used or fallback.conditioning_used:
        raise AssertionError("synthetic forcing/fallback contract failed")
    if len(np.unique(selected.neighbors.episodes)) != 8:
        raise AssertionError("synthetic episode distinctness failed")
    return {
        "conditioning_used": selected.conditioning_used,
        "fallback_reason": fallback.fallback_reason,
        "selected_distinct_episodes": int(len(np.unique(selected.neighbors.episodes))),
        "model_fit_count": 0,
        "target_value_read_count": 0,
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "pandas", "pyarrow", "scipy"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
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


def _dry_run(config: Mapping[str, Any], data_dir: Path, started: float) -> int:
    if ATTEMPT_LOCK.exists() or B_DIRECTORY.exists():
        raise PermissionError("v2 B one-shot state already exists")
    _status(
        state="dry_running",
        phase="hash_schema_synthetic_preflight",
        progress=20.0,
        detail="사전 고정 forcing 계약·SHA·schema 확인; feature/target/model 0",
        started=started,
    )
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "mode": "dry-run",
        "status": "READY_FOR_B_INNER_ONLY_ONE_SHOT",
        "created_at": _now(),
        "adaptive_research": True,
        "registered_input_sha256": _verify_registered_inputs(config, data_dir),
        "implementation_sha256": _implementation_hashes(),
        "schema_preflight": _schema_preflight(config),
        "synthetic_contract": _synthetic_contract(),
        "environment": _environment(),
        **_git_provenance(),
        "model_fit_count": 0,
        "C_execution_count": 0,
        "outer_membership_read_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    DRY_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DRY_DIRECTORY / "receipt.json"
    _atomic_json(path, receipt)
    payload = {**receipt, "receipt_sha256": _sha256(path)}
    _status(
        state="dry_ready",
        phase="ready_for_B_only",
        progress=100.0,
        detail="dry-run 완료; forcing 값/target/model/outer/test/submission 0",
        started=started,
        result={"receipt_sha256": payload["receipt_sha256"]},
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _read_wave(data_dir: Path) -> pd.DataFrame:
    wave = pd.read_csv(data_dir / "train_wave.csv")
    if not {"station", "time", "hs"}.issubset(wave):
        raise EpisodeAnalogError("train_wave schema changed")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    if wave.duplicated(["station", "time"]).any():
        raise EpisodeAnalogError("train_wave keys are duplicated")
    return wave


def _inner_splits(dataset: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    anchors = dataset.anchors
    result: dict[str, Any] = {}
    for name, start, _ in config["validation"]["outer_windows_used_only_as_training_cutoffs"]:
        cutoff = pd.Timestamp(start, tz="UTC") - pd.Timedelta(hours=78)
        outer_train_ids = anchors.loc[
            anchors["anchor_time"].lt(cutoff), "anchor_id"
        ].to_numpy(dtype=np.int64)
        result[str(name)] = build_inner_episode_split(
            anchors,
            outer_train_ids,
            validation_days=45,
            embargo_hours=78,
        )
    return result


def _split_audit(dataset: Any, splits: Mapping[str, Any]) -> dict[str, Any]:
    lookup = dataset.anchors.set_index("anchor_id")
    result: dict[str, Any] = {}
    for name, split in splits.items():
        train = lookup.loc[split.train_ids]
        validation = lookup.loc[split.validation_ids]
        train_episode = set(zip(train["station"], train["episode_id"], strict=True))
        valid_episode = set(
            zip(validation["station"], validation["episode_id"], strict=True)
        )
        if train_episode.intersection(valid_episode):
            raise EpisodeAnalogError(f"{name} shares a station episode")
        minimum_gap = np.inf
        for station, group in validation.groupby("station", observed=True):
            prior = train.loc[train["station"].eq(station), "anchor_time"]
            for timestamp in group["anchor_time"]:
                gap = (pd.Timestamp(timestamp) - prior).dt.total_seconds().min() / 3600.0
                minimum_gap = min(minimum_gap, float(gap))
        if minimum_gap < 78.0:
            raise EpisodeAnalogError(f"{name} violates the 78h origin gap")
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


def _read_forcing_features(config: Mapping[str, Any], dataset: Any) -> np.ndarray:
    feature_path = ROOT / config["registered_inputs"]["train_features"]["path"]
    frame = pd.read_parquet(
        feature_path,
        columns=["anchor_id", "station", *FORCING_COLUMNS],
    )
    if len(frame) != len(dataset.anchors) or frame["anchor_id"].duplicated().any():
        raise EpisodeAnalogError("forcing feature cache anchor contract changed")
    expected = np.arange(len(dataset.anchors), dtype=np.int64)
    if not np.array_equal(frame["anchor_id"].to_numpy(dtype=np.int64), expected):
        raise EpisodeAnalogError("forcing feature anchor ids changed")
    if not frame["station"].astype(str).equals(dataset.anchors["station"].astype(str)):
        raise EpisodeAnalogError("forcing feature stations changed")
    return frame.loc[:, FORCING_COLUMNS].to_numpy(dtype=np.float64)


def _adaptive_comparison(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fold, group in rows.groupby("fold", sort=True, observed=True):
        v2 = float(group["nearest_normalized_mse"].mean())
        v1 = float(group["v1_nearest_normalized_mse"].mean())
        result[str(fold)] = {
            "cases": int(len(group)),
            "conditioned_cases": int(group["conditioning_used"].sum()),
            "fallback_cases": int((~group["conditioning_used"]).sum()),
            "v2_nearest_mse": v2,
            "sealed_v1_nearest_mse": v1,
            "v2_to_v1_ratio": float(v2 / v1),
            "v2_better_case_fraction": float(
                (group["nearest_normalized_mse"] < group["v1_nearest_normalized_mse"]).mean()
            ),
        }
    return result


def _run_B(
    *,
    dataset: Any,
    prepared: Any,
    forcing_values: np.ndarray,
    splits: Mapping[str, Any],
    config: Mapping[str, Any],
    started: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    base_config = config["v1_contract_preserved"]
    anchors = dataset.anchors.set_index("anchor_id")
    normalized_residual = (
        dataset.official_target.astype(np.float64)
        - dataset.anchors["current_hs"].to_numpy(dtype=np.float64)[:, None]
    ) / prepared.scale[:, None]
    records: list[dict[str, Any]] = []
    search_audit: dict[str, Any] = {}
    total_cases = sum(len(split.validation_ids) for split in splits.values())
    completed = 0
    for fold_name, split in splits.items():
        fold_audit = {
            "eligible_queries": 0,
            "history_ineligible_queries": 0,
            "forcing_conditioned_queries": 0,
            "forcing_fallback_queries": 0,
            "fallback_reason_counts": {},
            "library": {},
        }
        search_audit[fold_name] = fold_audit
        indices: dict[str, tuple[ForcingConditionedAnalogIndex, np.ndarray]] = {}
        fit_ids = np.asarray(split.train_ids, dtype=np.int64)
        for station in sorted(dataset.anchors["station"].astype(str).unique()):
            station_ids = fit_ids[
                anchors.loc[fit_ids, "station"].astype(str).to_numpy() == station
            ]
            station_ids = station_ids[prepared.eligible[station_ids]]
            index = ForcingConditionedAnalogIndex(
                anchor_ids=station_ids,
                episode_ids=anchors.loc[station_ids, "episode_id"].to_numpy(dtype=np.int64),
                normalized_histories=prepared.normalized[station_ids],
                forcing_state=forcing_values[station_ids],
                radius_steps=int(base_config["sakoe_chiba_radius_steps"]),
                neighbor_count=int(base_config["neighbor_count"]),
                batch_size=int(config["forcing_state"]["distance_and_selection"]["search_batch_size"]),
            )
            indices[station] = (index, station_ids)
            fold_audit["library"][station] = {
                "v1_eligible_anchors": int(len(station_ids)),
                "v1_eligible_episodes": int(len(index.base.unique_episodes)),
                "complete_forcing_anchors": int(len(index.complete_rows)),
                "complete_forcing_episodes": int(index.conditioned_library_episodes),
            }

        for anchor_id in split.validation_ids:
            anchor_id = int(anchor_id)
            completed += 1
            if not prepared.eligible[anchor_id]:
                fold_audit["history_ineligible_queries"] += 1
                continue
            station = str(anchors.loc[anchor_id, "station"])
            index, station_ids = indices[station]
            query = prepared.normalized[anchor_id]
            selected = index.select_nearest(query, forcing_values[anchor_id])
            nearest = selected.neighbors
            nearest_projection = project_normalized_residual(
                normalized_residual[station_ids[nearest.indices]],
                nearest.distances,
                distance_floor=float(base_config["inverse_distance_floor"]),
            )
            random_panels = index.select_random_panels(
                query,
                seed=int(base_config["random_seed"]),
                query_key=f"{fold_name}|{anchor_id}",
                panel_count=int(base_config["random_panel_count"]),
            )
            random_projection = [
                project_normalized_residual(
                    normalized_residual[station_ids[panel.indices]],
                    panel.distances,
                    distance_floor=float(base_config["inverse_distance_floor"]),
                )
                for panel in random_panels
            ]
            truth = normalized_residual[anchor_id]
            nearest_mse = float(np.mean(np.square(nearest_projection - truth)))
            random_mse = float(
                np.mean(
                    [np.mean(np.square(prediction - truth)) for prediction in random_projection]
                )
            )
            fold_audit["eligible_queries"] += 1
            fold_audit["forcing_conditioned_queries"] += int(selected.conditioning_used)
            fold_audit["forcing_fallback_queries"] += int(not selected.conditioning_used)
            if selected.fallback_reason is not None:
                reasons = fold_audit["fallback_reason_counts"]
                reasons[selected.fallback_reason] = int(reasons.get(selected.fallback_reason, 0)) + 1
            records.append(
                {
                    "fold": fold_name,
                    "anchor_id": anchor_id,
                    "station": station,
                    "nearest_normalized_mse": nearest_mse,
                    "random_normalized_mse": random_mse,
                    "conditioning_used": bool(selected.conditioning_used),
                    "fallback_reason": selected.fallback_reason,
                    "query_forcing_finite_components": int(
                        np.isfinite(forcing_values[anchor_id]).sum()
                    ),
                    "nearest_anchor_ids_sha256": _hash_integer_array(
                        index.anchor_ids[nearest.indices]
                    ),
                    "nearest_episode_ids_sha256": _hash_integer_array(nearest.episodes),
                    "nearest_combined_distance_mean": float(nearest.distances.mean()),
                    "nearest_combined_distance_max": float(nearest.distances.max()),
                    "nearest_forcing_distance_mean": selected.forcing_distance_mean,
                    "nearest_forcing_distance_max": selected.forcing_distance_max,
                    "conditioned_library_anchors": selected.conditioned_library_anchors,
                    "conditioned_library_episodes": selected.conditioned_library_episodes,
                    "evaluated_candidates": int(nearest.evaluated_candidates),
                    "total_search_candidates": int(nearest.total_candidates),
                }
            )
            if completed % 10 == 0 or completed == total_cases:
                _status(
                    state="B_running_pre_C_outer_test",
                    phase="forcing_conditioned_B_precheck",
                    progress=10.0 + 75.0 * completed / max(total_cases, 1),
                    detail=f"B {completed}/{total_cases}; C/model/outer/test/submission 0",
                    started=started,
                )

    rows = pd.DataFrame.from_records(records).sort_values(
        ["fold", "anchor_id"]
    ).reset_index(drop=True)
    v1_path = ROOT / config["registered_inputs"]["v1_B_cases"]["path"]
    v1 = pd.read_parquet(
        v1_path,
        columns=[
            "fold",
            "anchor_id",
            "station",
            "nearest_normalized_mse",
            "random_normalized_mse",
        ],
    ).rename(
        columns={
            "nearest_normalized_mse": "v1_nearest_normalized_mse",
            "random_normalized_mse": "v1_random_normalized_mse",
        }
    )
    v1 = v1.sort_values(["fold", "anchor_id"]).reset_index(drop=True)
    keys = ["fold", "anchor_id", "station"]
    if not rows[keys].equals(v1[keys]):
        raise EpisodeAnalogError("v2 B membership differs from sealed v1 B membership")
    rows = rows.merge(v1, on=keys, how="inner", validate="one_to_one", sort=False)
    random_difference = np.abs(
        rows["random_normalized_mse"].to_numpy(dtype=np.float64)
        - rows["v1_random_normalized_mse"].to_numpy(dtype=np.float64)
    )
    if float(random_difference.max()) > 1e-12:
        raise EpisodeAnalogError("v2 random reference differs from sealed v1")
    gate_config = config["B_gate"]
    gate = evaluate_b_precheck(
        rows,
        maximum_fold_mse_ratio=float(
            gate_config["maximum_nearest_to_random_mse_ratio_per_passing_fold"]
        ),
        minimum_passing_folds=int(gate_config["minimum_passing_folds"]),
    )
    payload = {
        "gate": gate,
        "adaptive_v2_vs_sealed_v1": _adaptive_comparison(rows),
        "random_reference_max_abs_difference_vs_v1": float(random_difference.max()),
        "search": search_audit,
    }
    return rows, payload


def _output_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _run_B_one_shot(config: Mapping[str, Any], data_dir: Path, started: float) -> int:
    dry_path = DRY_DIRECTORY / "receipt.json"
    if not dry_path.is_file():
        raise PermissionError("canonical dry-run receipt is required")
    if ATTEMPT_LOCK.exists() or B_DIRECTORY.exists():
        raise PermissionError("v2 B one-shot has already been attempted")
    dry = json.loads(dry_path.read_text(encoding="utf-8"))
    implementation = _implementation_hashes()
    receipts = _verify_registered_inputs(config, data_dir)
    if dry.get("implementation_sha256") != implementation:
        raise PermissionError("implementation changed after dry-run")
    if dry.get("registered_input_sha256") != receipts:
        raise PermissionError("registered input changed after dry-run")
    _write_exclusive(
        ATTEMPT_LOCK,
        {
            "experiment_id": EXPERIMENT_ID,
            "created_at": _now(),
            "scope": "adaptive_B_inner_only_no_C_model_outer_test_submission",
            "dry_receipt_sha256": _sha256(dry_path),
            "implementation_sha256": implementation,
        },
    )
    B_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _status(
        state="B_running_pre_C_outer_test",
        phase="raw_reconstruction_and_same_inner_split",
        progress=5.0,
        detail="v1 동일 inner split·forcing cache 구성; C/model/outer/test/submission 0",
        started=started,
    )
    dataset = build_trajectory_dataset(_read_wave(data_dir))
    if len(dataset.anchors) != 24360:
        raise EpisodeAnalogError("raw-wave anchor count changed")
    base = config["v1_contract_preserved"]
    prepared = prepare_histories(
        dataset.history,
        minimum_coverage=float(base["minimum_history_coverage"]),
        mad_floor=float(base["history_mad_floor_m"]),
    )
    splits = _inner_splits(dataset, config)
    split_audit = _split_audit(dataset, splits)
    forcing = _read_forcing_features(config, dataset)
    rows, B_payload = _run_B(
        dataset=dataset,
        prepared=prepared,
        forcing_values=forcing,
        splits=splits,
        config=config,
        started=started,
    )
    rows_path = B_DIRECTORY / "b_precheck_cases.parquet"
    result_path = B_DIRECTORY / "b_precheck.json"
    _atomic_parquet(rows_path, rows)
    _atomic_json(result_path, B_payload)
    passed = bool(B_payload["gate"]["pass"])
    decision = (
        "PASS_B_ADAPTIVE_INNER_ONLY_STOP"
        if passed
        else "NO_GO_B_FORCING_CONDITIONING_NOT_STABLE"
    )
    result = {
        "decision": decision,
        "B_gate": B_payload["gate"],
        "adaptive_research": True,
        "independent_confirmation": False,
        "required_action": (
            "stop_and_await_separate_authorization"
            if passed
            else "permanent_stop_without_threshold_relaxation_or_rerun"
        ),
        "model_fit_count": 0,
        "C_execution_count": 0,
        "outer_membership_read_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    _atomic_json(B_DIRECTORY / "result.json", result)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "adaptive_B_inner_only_no_C_model_outer_test_submission",
        "decision": decision,
        "adaptive_research": True,
        "same_split_reused": True,
        "split_audit": split_audit,
        "registered_input_sha256": receipts,
        "implementation_sha256": implementation,
        "attempt_lock_sha256": _sha256(ATTEMPT_LOCK),
        "dry_receipt_sha256": _sha256(dry_path),
        "output_sha256": _output_hashes(B_DIRECTORY),
        "environment": _environment(),
        **_git_provenance(),
        "target_access": {
            "adaptive_inner_B_cases": int(len(rows)),
            "outer_182_membership_or_target_rows": 0,
        },
        "model_fit_count": 0,
        "C_execution_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    _atomic_json(B_DIRECTORY / "manifest.json", manifest)
    _status(
        state="B_complete_stop",
        phase="B_gate_complete_no_C",
        progress=100.0,
        detail=f"{decision}; C/model/outer/test/submission 0; 권한 범위 종료",
        started=started,
        result=result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--p3-data-dir")
    parser.add_argument("--mode", choices=("dry-run", "B-inner"), default="dry-run")
    parser.add_argument("--confirm")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    started = time.perf_counter()
    config = _load_config(arguments.config)
    data_dir = _resolve_data_dir(arguments.p3_data_dir)
    if arguments.mode == "dry-run":
        if arguments.confirm is not None:
            raise PermissionError("dry-run does not accept confirmation")
        return _dry_run(config, data_dir, started)
    if arguments.confirm != CONFIRMATION_TOKEN:
        raise PermissionError("B-inner mode requires the exact confirmation token")
    return _run_B_one_shot(config, data_dir, started)


if __name__ == "__main__":
    raise SystemExit(main())
