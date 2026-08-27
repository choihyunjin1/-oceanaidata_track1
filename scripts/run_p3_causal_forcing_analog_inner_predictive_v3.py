"""One-shot C-inner predictive gate for the sealed P3 forcing analog v3."""

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
    ForcingConditionedAnalogIndex,
)
from p3_wave.causal_forcing_inner_predictive import (
    BLIND_CASE_COLUMNS,
    BLIND_PREDICTION_COLUMNS,
    FoldScope,
    InnerTargetVault,
    apply_fixed_control_shrink,
    attach_validation_targets,
    expand_control_fit_rows,
    expand_control_prediction_rows,
    hash_integer_array,
    independently_recalculate_C_metrics,
    validate_blind_cases,
    validate_blind_predictions,
)
from p3_wave.episode_distinct_analog import (
    LEADS,
    EpisodeAnalogError,
    blend_candidate,
    prepare_histories,
    project_normalized_residual,
)
from p3_wave.models import threshold_case_weights
from p3_wave.revin_patch import build_inner_episode_split

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p3_causal_forcing_analog_inner_predictive_v3"
CONFIG_PATH = (ROOT / f"configs/experiments/{EXPERIMENT_ID}.json").resolve()
DRY_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/dry_run").resolve()
C_DIRECTORY = (ROOT / f"artifacts/{EXPERIMENT_ID}/C_one_shot").resolve()
STATUS_PATH = (ROOT / f"artifacts/status/{EXPERIMENT_ID}.json").resolve()
ATTEMPT_LOCK = (
    ROOT / f"artifacts/experiment_locks/{EXPERIMENT_ID}.C.attempt.lock"
).resolve()
CONFIRMATION_TOKEN = "RUN_P3_CAUSAL_FORCING_ANALOG_V3_C_ONCE"


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _sha256(path: Path) -> str:
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
        "title": "P3 causal forcing analog inner predictive v3",
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
        raise PermissionError("config override is prohibited for the v3 one-shot runner")
    config = json.loads(resolved.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise EpisodeAnalogError("v3 experiment id drifted")
    if not config["adaptive_research_disclosure"]["is_adaptive_research"]:
        raise EpisodeAnalogError("adaptive research disclosure was removed")
    if config["sealed_v2_method"]["neighbor_count"] != 8:
        raise EpisodeAnalogError("sealed v2 k changed")
    if tuple(config["sealed_v2_method"]["forcing_columns"]) != FORCING_COLUMNS:
        raise EpisodeAnalogError("sealed v2 forcing columns changed")
    candidate = config["candidate"]
    if (
        candidate["no_op_leads"] != [3, 6, 9]
        or candidate["active_leads"] != [12, 18, 24]
        or candidate["alpha"] != 0.2
    ):
        raise EpisodeAnalogError("fixed candidate routing changed")
    gate = config["validation"]["C_gate"]
    if (
        gate["maximum_pooled_full_six_lead_delta_m"] != -0.005
        or gate["minimum_strictly_improved_folds"] != 2
        or gate["maximum_any_station_RMSE_degradation_m"] != 0.01
        or not gate["lead_18_must_not_degrade"]
        or not gate["lead_24_must_not_degrade"]
    ):
        raise EpisodeAnalogError("fixed C gate changed")
    if not all(bool(value) for value in config["prohibitions"].values()):
        raise EpisodeAnalogError("a v3 safety prohibition was disabled")
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
        "src/p3_wave/causal_forcing_inner_predictive.py",
        f"scripts/run_{EXPERIMENT_ID}.py",
        f"tests/test_{EXPERIMENT_ID}.py",
        "src/p3_wave/causal_forcing_analog.py",
        "src/p3_wave/episode_distinct_analog.py",
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
    registered = config["registered_inputs"]
    feature_path = ROOT / registered["train_features"]["path"]
    schema = parquet.read_schema(feature_path)
    feature_columns = json.loads(
        (ROOT / registered["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    if len(feature_columns) != 591 or len(set(feature_columns)) != 591:
        raise EpisodeAnalogError("frozen control feature list changed")
    required = {"anchor_id", "station", *feature_columns, *FORCING_COLUMNS}
    if not required.issubset(schema.names):
        raise EpisodeAnalogError(
            f"feature cache lacks columns: {sorted(required - set(schema.names))}"
        )
    v2_schema = parquet.read_schema(ROOT / registered["v2_B_cases"]["path"])
    required_v2 = {
        "fold",
        "anchor_id",
        "station",
        "nearest_normalized_mse",
        "conditioning_used",
        "fallback_reason",
        "nearest_anchor_ids_sha256",
        "nearest_episode_ids_sha256",
        "nearest_combined_distance_mean",
        "nearest_combined_distance_max",
    }
    if not required_v2.issubset(v2_schema.names):
        raise EpisodeAnalogError("sealed v2 B case schema changed")
    manifest = json.loads(
        (ROOT / registered["frozen_model_manifest"]["path"]).read_text(encoding="utf-8")
    )
    control = config["fixed_inner_control_proxy"]
    for key in (
        "iterations",
        "learning_rate",
        "depth",
        "l2_leaf_reg",
        "random_strength",
        "random_seed",
        "loss_function",
        "thread_count",
        "allow_writing_files",
    ):
        if manifest["parameters"]["single"].get(key) != control.get(key):
            raise EpisodeAnalogError(f"inner control differs from frozen manifest: {key}")
    v2_result = json.loads(
        (ROOT / registered["v2_result"]["path"]).read_text(encoding="utf-8")
    )
    if v2_result.get("decision") != "PASS_B_ADAPTIVE_INNER_ONLY_STOP":
        raise EpisodeAnalogError("sealed v2 B did not pass")
    return {
        "feature_cache_schema_columns": len(schema.names),
        "control_feature_count": len(feature_columns),
        "forcing_columns": list(FORCING_COLUMNS),
        "v2_B_schema_columns": v2_schema.names,
        "v2_B_decision": v2_result["decision"],
        "feature_values_read": 0,
        "target_values_read": 0,
        "model_fit_count": 0,
    }


def _synthetic_contract() -> dict[str, Any]:
    from p3_wave.causal_forcing_inner_predictive import (
        attach_validation_targets,
        validate_blind_predictions,
    )

    rows: list[dict[str, Any]] = []
    for anchor_id in range(3):
        for lead in LEADS:
            rows.append(
                {
                    "fold": f"f{anchor_id}",
                    "anchor_id": anchor_id,
                    "station": ("G-ORS", "I-ORS", "S-ORS")[anchor_id],
                    "lead_h": lead,
                    "current_hs": 2.0,
                    "query_mad_scale": 0.2,
                    "analog_applicable": True,
                    "analog_prediction": 1.5,
                    "control_single_prediction": 2.0,
                    "control_final": 2.0,
                    "candidate_final": 2.0 if lead in (3, 6, 9) else 1.9,
                }
            )
    blind = pd.DataFrame(rows, columns=BLIND_PREDICTION_COLUMNS)
    validate_blind_predictions(blind, expected_cases=3)
    targets = {
        f"f{anchor_id}": (
            np.asarray([anchor_id]),
            np.ones((1, len(LEADS)), dtype=np.float64),
        )
        for anchor_id in range(3)
    }
    evaluated = attach_validation_targets(blind, targets)
    metrics = independently_recalculate_C_metrics(evaluated)
    if not metrics["pass"]:
        raise AssertionError("synthetic C gate contract failed")
    return {
        "blind_columns": list(blind.columns),
        "blind_has_target": False,
        "short_lead_exact_no_op": True,
        "evaluated_rows": int(len(evaluated)),
        "synthetic_gate_pass": True,
        "model_fit_count": 0,
        "target_value_read_count": 0,
    }


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
    if ATTEMPT_LOCK.exists() or C_DIRECTORY.exists():
        raise PermissionError("v3 C one-shot state already exists")
    _status(
        state="dry_running",
        phase="hash_schema_synthetic_preflight",
        progress=20.0,
        detail="v2 seal·control schema·label vault 합성 감사; model/target/outer/test 0",
        started=started,
    )
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "mode": "dry-run",
        "status": "READY_FOR_C_INNER_ONLY_ONE_SHOT",
        "created_at": _now(),
        "adaptive_research": True,
        "registered_input_sha256": _verify_registered_inputs(config, data_dir),
        "implementation_sha256": _implementation_hashes(),
        "schema_preflight": _schema_preflight(config),
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
    payload = {**receipt, "receipt_sha256": _sha256(path)}
    _status(
        state="dry_ready",
        phase="ready_for_C_inner_only",
        progress=100.0,
        detail="dry-run 완료; model/target/outer/test/submission 0",
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
        shared = set(zip(train["station"], train["episode_id"], strict=True)).intersection(
            zip(validation["station"], validation["episode_id"], strict=True)
        )
        if shared:
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


def _load_feature_cache(
    config: Mapping[str, Any], dataset: Any
) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    registered = config["registered_inputs"]
    feature_columns = json.loads(
        (ROOT / registered["feature_columns"]["path"]).read_text(encoding="utf-8")
    )
    features = pd.read_parquet(
        ROOT / registered["train_features"]["path"],
        columns=["anchor_id", "station", *feature_columns],
    )
    expected = np.arange(len(dataset.anchors), dtype=np.int64)
    if (
        len(features) != len(dataset.anchors)
        or features["anchor_id"].duplicated().any()
        or not np.array_equal(features["anchor_id"].to_numpy(dtype=np.int64), expected)
    ):
        raise EpisodeAnalogError("feature cache anchor contract changed")
    if not features["station"].astype(str).equals(dataset.anchors["station"].astype(str)):
        raise EpisodeAnalogError("feature cache stations changed")
    forcing = features.loc[:, FORCING_COLUMNS].to_numpy(dtype=np.float64)
    return features, feature_columns, forcing


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


def _normalize_reason(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _reconstruct_analog_blind(
    *,
    fold_name: str,
    split: Any,
    dataset: Any,
    prepared: Any,
    forcing: np.ndarray,
    fit_targets: np.ndarray,
    sealed_v2: pd.DataFrame,
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    anchors = dataset.anchors.set_index("anchor_id")
    fit_ids = np.asarray(split.train_ids, dtype=np.int64)
    target_lookup = np.full((len(dataset.anchors), len(LEADS)), np.nan, dtype=np.float64)
    target_lookup[fit_ids] = fit_targets
    analog_by_id: dict[int, np.ndarray] = {}
    case_rows: list[dict[str, Any]] = []
    indices: dict[str, tuple[ForcingConditionedAnalogIndex, np.ndarray]] = {}
    for station in sorted(dataset.anchors["station"].astype(str).unique()):
        station_ids = fit_ids[
            anchors.loc[fit_ids, "station"].astype(str).to_numpy() == station
        ]
        station_ids = station_ids[prepared.eligible[station_ids]]
        indices[station] = (
            ForcingConditionedAnalogIndex(
                anchor_ids=station_ids,
                episode_ids=anchors.loc[station_ids, "episode_id"].to_numpy(dtype=np.int64),
                normalized_histories=prepared.normalized[station_ids],
                forcing_state=forcing[station_ids],
                radius_steps=6,
                neighbor_count=8,
                batch_size=1024,
            ),
            station_ids,
        )
    sealed_fold = sealed_v2.loc[sealed_v2["fold"].astype(str).eq(fold_name)].set_index(
        "anchor_id"
    )
    for anchor_id_value in split.validation_ids:
        anchor_id = int(anchor_id_value)
        station = str(anchors.loc[anchor_id, "station"])
        if not prepared.eligible[anchor_id]:
            analog_by_id[anchor_id] = np.full(len(LEADS), np.nan)
            case_rows.append(
                {
                    "fold": fold_name,
                    "anchor_id": anchor_id,
                    "station": station,
                    "history_eligible": False,
                    "conditioning_used": False,
                    "fallback_reason": "history_ineligible",
                    "query_mad_scale": np.nan,
                    "neighbor_anchor_ids_sha256": hash_integer_array([]),
                    "neighbor_episode_ids_sha256": hash_integer_array([]),
                    "neighbor_distance_mean": np.nan,
                    "neighbor_distance_max": np.nan,
                }
            )
            continue
        index, station_ids = indices[station]
        selected = index.select_nearest(
            prepared.normalized[anchor_id], forcing[anchor_id]
        )
        neighbors = selected.neighbors
        library_residual = (
            target_lookup[station_ids[neighbors.indices]]
            - anchors.loc[station_ids[neighbors.indices], "current_hs"].to_numpy(
                dtype=np.float64
            )[:, None]
        ) / prepared.scale[station_ids[neighbors.indices], None]
        if not np.isfinite(library_residual).all():
            raise EpisodeAnalogError("analog library target release is incomplete")
        projected = project_normalized_residual(
            library_residual,
            neighbors.distances,
            distance_floor=1e-6,
        )
        current = float(anchors.loc[anchor_id, "current_hs"])
        analog_by_id[anchor_id] = np.clip(
            current + prepared.scale[anchor_id] * projected,
            0.0,
            30.0,
        )
        anchor_hash = hash_integer_array(index.anchor_ids[neighbors.indices])
        episode_hash = hash_integer_array(neighbors.episodes)
        if anchor_id not in sealed_fold.index:
            raise EpisodeAnalogError("eligible v3 query is absent from sealed v2 B")
        expected = sealed_fold.loc[anchor_id]
        if str(expected["station"]) != station:
            raise EpisodeAnalogError("sealed v2 station differs")
        if bool(expected["conditioning_used"]) != bool(selected.conditioning_used):
            raise EpisodeAnalogError("sealed v2 conditioning flag differs")
        if _normalize_reason(expected["fallback_reason"]) != _normalize_reason(
            selected.fallback_reason
        ):
            raise EpisodeAnalogError("sealed v2 fallback reason differs")
        if expected["nearest_anchor_ids_sha256"] != anchor_hash:
            raise EpisodeAnalogError("sealed v2 neighbor anchor hash differs")
        if expected["nearest_episode_ids_sha256"] != episode_hash:
            raise EpisodeAnalogError("sealed v2 neighbor episode hash differs")
        tolerance = 1e-12
        if not np.isclose(
            float(expected["nearest_combined_distance_mean"]),
            float(neighbors.distances.mean()),
            rtol=0.0,
            atol=tolerance,
        ) or not np.isclose(
            float(expected["nearest_combined_distance_max"]),
            float(neighbors.distances.max()),
            rtol=0.0,
            atol=tolerance,
        ):
            raise EpisodeAnalogError("sealed v2 neighbor distance differs")
        case_rows.append(
            {
                "fold": fold_name,
                "anchor_id": anchor_id,
                "station": station,
                "history_eligible": True,
                "conditioning_used": bool(selected.conditioning_used),
                "fallback_reason": _normalize_reason(selected.fallback_reason),
                "query_mad_scale": float(prepared.scale[anchor_id]),
                "neighbor_anchor_ids_sha256": anchor_hash,
                "neighbor_episode_ids_sha256": episode_hash,
                "neighbor_distance_mean": float(neighbors.distances.mean()),
                "neighbor_distance_max": float(neighbors.distances.max()),
            }
        )
    return analog_by_id, pd.DataFrame(case_rows, columns=BLIND_CASE_COLUMNS)


def _fit_control_and_build_blind(
    *,
    fold_name: str,
    split: Any,
    dataset: Any,
    prepared: Any,
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    fit_targets: np.ndarray,
    analog_by_id: Mapping[int, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    fit_matrix, fit_residual, fit_meta = expand_control_fit_rows(
        features=features,
        anchors=dataset.anchors,
        anchor_ids=split.train_ids,
        target_matrix=fit_targets,
        feature_columns=feature_columns,
    )
    validation_matrix, validation_meta = expand_control_prediction_rows(
        features=features,
        anchors=dataset.anchors,
        anchor_ids=split.validation_ids,
        feature_columns=feature_columns,
    )
    model = _control_model(config["fixed_inner_control_proxy"])
    model.fit(
        fit_matrix,
        fit_residual,
        sample_weight=threshold_case_weights(
            fit_meta["current_hs"].to_numpy(dtype=np.float64)
        ),
        cat_features=[0, 1],
        verbose=False,
    )
    model_path = C_DIRECTORY / f"control_proxy_{fold_name}.cbm"
    model.save_model(model_path)
    residual = np.asarray(model.predict(validation_matrix), dtype=np.float64)
    control_single = np.clip(
        validation_meta["current_hs"].to_numpy(dtype=np.float64) + residual,
        0.0,
        30.0,
    )
    validation_meta.insert(0, "fold", fold_name)
    validation_meta["query_mad_scale"] = prepared.scale[
        validation_meta["anchor_id"].to_numpy(dtype=np.int64)
    ]
    analog = np.full(len(validation_meta), np.nan, dtype=np.float64)
    applicable = np.zeros(len(validation_meta), dtype=bool)
    for position, row in enumerate(validation_meta.itertuples(index=False)):
        path = analog_by_id[int(row.anchor_id)]
        lead_column = LEADS.index(int(row.lead_h))
        analog[position] = path[lead_column]
        applicable[position] = np.isfinite(path[lead_column])
    validation_meta["analog_applicable"] = applicable
    validation_meta["analog_prediction"] = analog
    validation_meta["control_single_prediction"] = control_single
    validation_meta["control_final"] = apply_fixed_control_shrink(
        control_single,
        validation_meta["current_hs"].to_numpy(dtype=np.float64),
        validation_meta["lead_h"].to_numpy(dtype=np.int64),
    )
    validation_meta["candidate_final"] = blend_candidate(
        validation_meta["control_final"].to_numpy(dtype=np.float64),
        analog,
        validation_meta["lead_h"].to_numpy(dtype=np.int64),
        alpha=0.2,
    )
    blind = validation_meta.loc[:, BLIND_PREDICTION_COLUMNS]
    return blind, _sha256(model_path)


def _verify_v2_B_post_label(
    evaluated: pd.DataFrame,
    blind_cases: pd.DataFrame,
    sealed_v2: pd.DataFrame,
) -> dict[str, Any]:
    eligible = blind_cases.loc[blind_cases["history_eligible"]].copy()
    keys = eligible[["fold", "anchor_id", "station"]].sort_values(
        ["fold", "anchor_id"]
    ).reset_index(drop=True)
    expected_keys = sealed_v2[["fold", "anchor_id", "station"]].sort_values(
        ["fold", "anchor_id"]
    ).reset_index(drop=True)
    if not keys.equals(expected_keys):
        raise EpisodeAnalogError("post-label v2 B membership differs")
    scale_lookup = eligible.set_index(["fold", "anchor_id"])["query_mad_scale"]
    work = evaluated.loc[evaluated["analog_applicable"]].copy()
    work["normalized_analog_squared_error"] = np.square(
        (work["analog_prediction"] - work["target_hs"])
        / [scale_lookup.loc[(row.fold, row.anchor_id)] for row in work.itertuples()]
    )
    reproduced = (
        work.groupby(["fold", "anchor_id", "station"], sort=True, observed=True)
        ["normalized_analog_squared_error"]
        .mean()
        .rename("reproduced_nearest_normalized_mse")
        .reset_index()
    )
    expected = sealed_v2[
        ["fold", "anchor_id", "station", "nearest_normalized_mse"]
    ].copy()
    comparison = reproduced.merge(
        expected,
        on=["fold", "anchor_id", "station"],
        how="inner",
        validate="one_to_one",
    )
    difference = np.abs(
        comparison["reproduced_nearest_normalized_mse"].to_numpy(dtype=np.float64)
        - comparison["nearest_normalized_mse"].to_numpy(dtype=np.float64)
    )
    maximum = float(difference.max())
    if maximum > 1e-12:
        raise EpisodeAnalogError("post-label nearest MSE differs from sealed v2 B")
    return {
        "cases": int(len(comparison)),
        "membership_exact": True,
        "neighbor_hashes_exact": True,
        "maximum_nearest_normalized_mse_abs_difference": maximum,
        "tolerance": 1e-12,
    }


def _assert_roundtrip(expected: pd.DataFrame, path: Path) -> pd.DataFrame:
    reloaded = pd.read_parquet(path)
    try:
        pd.testing.assert_frame_equal(reloaded, expected, check_exact=True)
    except AssertionError as error:
        raise EpisodeAnalogError(f"fsynced parquet changed after reload: {path.name}") from error
    return reloaded


def _output_hashes(directory: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _run_C_one_shot(config: Mapping[str, Any], data_dir: Path, started: float) -> int:
    dry_path = DRY_DIRECTORY / "receipt.json"
    if not dry_path.is_file():
        raise PermissionError("canonical dry-run receipt is required")
    if ATTEMPT_LOCK.exists() or C_DIRECTORY.exists():
        raise PermissionError("v3 C one-shot has already been attempted")
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
            "scope": "adaptive_C_inner_only_no_outer_test_submission",
            "dry_receipt_sha256": _sha256(dry_path),
            "implementation_sha256": implementation,
        },
    )
    C_DIRECTORY.mkdir(parents=True, exist_ok=False)
    _status(
        state="C_running_pre_outer",
        phase="raw_reconstruction_split_and_vault",
        progress=5.0,
        detail="동일 inner split·label vault 구성; validation/outer/test 0",
        started=started,
    )
    dataset = build_trajectory_dataset(_read_wave(data_dir))
    if len(dataset.anchors) != 24360:
        raise EpisodeAnalogError("raw-wave anchor count changed")
    prepared = prepare_histories(dataset.history, minimum_coverage=0.95, mad_floor=0.1)
    splits = _inner_splits(dataset, config)
    split_audit = _split_audit(dataset, splits)
    scopes = tuple(
        FoldScope(
            name=name,
            train_ids=np.asarray(split.train_ids, dtype=np.int64),
            validation_ids=np.asarray(split.validation_ids, dtype=np.int64),
        )
        for name, split in splits.items()
    )
    vault = InnerTargetVault(dataset.official_target, scopes)
    features, feature_columns, forcing = _load_feature_cache(config, dataset)
    v2_columns = [
        "fold",
        "anchor_id",
        "station",
        "nearest_normalized_mse",
        "conditioning_used",
        "fallback_reason",
        "nearest_anchor_ids_sha256",
        "nearest_episode_ids_sha256",
        "nearest_combined_distance_mean",
        "nearest_combined_distance_max",
    ]
    sealed_v2 = pd.read_parquet(
        ROOT / config["registered_inputs"]["v2_B_cases"]["path"],
        columns=v2_columns,
    )
    if len(sealed_v2) != 57 or sealed_v2.duplicated(["fold", "anchor_id"]).any():
        raise EpisodeAnalogError("sealed v2 B membership changed")

    blind_rows: list[pd.DataFrame] = []
    case_rows: list[pd.DataFrame] = []
    model_hashes: dict[str, str] = {}
    for fold_index, (fold_name, split) in enumerate(splits.items()):
        fit_targets = vault.read_fit(fold_name, split.train_ids)
        analog_by_id, cases = _reconstruct_analog_blind(
            fold_name=fold_name,
            split=split,
            dataset=dataset,
            prepared=prepared,
            forcing=forcing,
            fit_targets=fit_targets,
            sealed_v2=sealed_v2,
        )
        blind, model_hash = _fit_control_and_build_blind(
            fold_name=fold_name,
            split=split,
            dataset=dataset,
            prepared=prepared,
            features=features,
            feature_columns=feature_columns,
            fit_targets=fit_targets,
            analog_by_id=analog_by_id,
            config=config,
        )
        blind_rows.append(blind)
        case_rows.append(cases)
        model_hashes[fold_name] = model_hash
        _status(
            state="C_running_pre_outer",
            phase="fit_control_and_materialize_blind",
            progress=20.0 + 18.0 * (fold_index + 1),
            detail=f"{fold_name} blind 완료; validation labels/outer/test 0",
            started=started,
        )

    blind_cases = pd.concat(case_rows, ignore_index=True).sort_values(
        ["fold", "anchor_id"]
    ).reset_index(drop=True)
    blind_predictions = pd.concat(blind_rows, ignore_index=True).sort_values(
        ["fold", "anchor_id", "lead_h"]
    ).reset_index(drop=True)
    validate_blind_cases(blind_cases, expected_cases=61)
    validate_blind_predictions(blind_predictions, expected_cases=61)
    if int(blind_cases["history_eligible"].sum()) != 57:
        raise EpisodeAnalogError("v2 B eligible case count changed")
    if int(blind_cases["conditioning_used"].sum()) != 19:
        raise EpisodeAnalogError("v2 forcing-conditioned case count changed")
    case_path = C_DIRECTORY / "blind_case_audit.parquet"
    blind_path = C_DIRECTORY / "blind_inner_predictions.parquet"
    _atomic_parquet(case_path, blind_cases)
    _atomic_parquet(blind_path, blind_predictions)
    reloaded_cases = _assert_roundtrip(blind_cases, case_path)
    reloaded_blind = _assert_roundtrip(blind_predictions, blind_path)
    validate_blind_cases(reloaded_cases, expected_cases=61)
    validate_blind_predictions(reloaded_blind, expected_cases=61)
    blind_seal = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "all_three_fold_label_free_inner_predictions_before_validation_open",
        "blind_inner_predictions_sha256": _sha256(blind_path),
        "blind_case_audit_sha256": _sha256(case_path),
        "control_proxy_model_sha256_by_fold": model_hashes,
        "fit_target_access_log": vault.access_log,
        "validation_target_read_count": 0,
        "outer_membership_or_target_read_count": 0,
        "implementation_sha256": implementation,
        "registered_input_sha256": receipts,
        "attempt_lock_sha256": _sha256(ATTEMPT_LOCK),
    }
    blind_seal_path = C_DIRECTORY / "blind_seal.json"
    _atomic_json(blind_seal_path, blind_seal)
    blind_seal_sha = _sha256(blind_seal_path)
    if _implementation_hashes() != implementation:
        raise PermissionError("implementation changed before validation label open")
    if _verify_registered_inputs(config, data_dir) != receipts:
        raise PermissionError("registered input changed before validation label open")
    vault.seal_blind_predictions(blind_seal_sha)
    _status(
        state="C_running_pre_outer",
        phase="blind_sealed_open_inner_validation_only",
        progress=78.0,
        detail="blind fsync/SHA 봉인 완료; inner validation만 개봉; outer/test 0",
        started=started,
    )

    validation_targets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for fold_name, split in splits.items():
        validation_targets[fold_name] = (
            np.asarray(split.validation_ids, dtype=np.int64),
            vault.read_validation(fold_name, split.validation_ids),
        )
    evaluated = attach_validation_targets(reloaded_blind, validation_targets)
    v2_reproduction = _verify_v2_B_post_label(evaluated, reloaded_cases, sealed_v2)
    evaluated_path = C_DIRECTORY / "inner_predictions.parquet"
    _atomic_parquet(evaluated_path, evaluated)
    reloaded_evaluated = _assert_roundtrip(evaluated, evaluated_path)
    gate_config = config["validation"]["C_gate"]
    metrics = independently_recalculate_C_metrics(
        reloaded_evaluated,
        maximum_pooled_delta_m=float(
            gate_config["maximum_pooled_full_six_lead_delta_m"]
        ),
        minimum_improved_folds=int(gate_config["minimum_strictly_improved_folds"]),
        maximum_station_degradation_m=float(
            gate_config["maximum_any_station_RMSE_degradation_m"]
        ),
    )
    metrics["v2_B_reproduction"] = v2_reproduction
    metrics_path = C_DIRECTORY / "metrics.json"
    access_path = C_DIRECTORY / "target_access_log.json"
    _atomic_json(metrics_path, metrics)
    _atomic_json(
        access_path,
        {
            "blind_seal_sha256": blind_seal_sha,
            "access_log": vault.access_log,
            "current_or_future_fold_validation_overlap_count": 0,
            "outer_membership_or_target_read_count": 0,
        },
    )
    decision = (
        "PASS_C_ADAPTIVE_INNER_STOP_BEFORE_OUTER"
        if metrics["pass"]
        else "NO_GO_C_INNER_GATE"
    )
    result = {
        "decision": decision,
        "C_gate": metrics,
        "adaptive_research": True,
        "independent_confirmation": False,
        "required_action": (
            "stop_and_await_separate_outer_authorization"
            if metrics["pass"]
            else "permanent_stop_without_retuning_or_rerun"
        ),
        "model_fit_count": 3,
        "outer_membership_read_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    _atomic_json(C_DIRECTORY / "result.json", result)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at": _now(),
        "scope": "adaptive_C_inner_only_no_outer_test_submission",
        "decision": decision,
        "adaptive_research": True,
        "same_split_reused": True,
        "independent_confirmation": False,
        "split_audit": split_audit,
        "registered_input_sha256": receipts,
        "implementation_sha256": implementation,
        "attempt_lock_sha256": _sha256(ATTEMPT_LOCK),
        "dry_receipt_sha256": _sha256(dry_path),
        "blind_seal_sha256": blind_seal_sha,
        "control_proxy_model_sha256_by_fold": model_hashes,
        "output_sha256": _output_hashes(C_DIRECTORY),
        "environment": _environment(),
        **_git_provenance(),
        "target_access_log": vault.access_log,
        "model_fit_count": 3,
        "outer_membership_or_target_read_count": 0,
        "outer_designated_scoring_open_count": 0,
        "test_context_read_count": 0,
        "submission_write_count": 0,
    }
    _atomic_json(C_DIRECTORY / "manifest.json", manifest)
    _status(
        state="C_complete_stop_pre_outer",
        phase="C_gate_complete_no_outer",
        progress=100.0,
        detail=f"{decision}; outer/test/submission 0; 별도 승인 대기",
        started=started,
        result=result,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--p3-data-dir")
    parser.add_argument("--mode", choices=("dry-run", "C-inner"), default="dry-run")
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
        raise PermissionError("C-inner mode requires the exact confirmation token")
    return _run_C_one_shot(config, data_dir, started)


if __name__ == "__main__":
    raise SystemExit(main())
