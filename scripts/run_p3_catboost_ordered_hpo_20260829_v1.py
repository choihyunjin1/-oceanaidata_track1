#!/usr/bin/env python3
"""Static preflight and authorization boundary for P3 CatBoost ordered HPO.

The current contract permits only ``static-preflight``.  It reads schemas, hashes, and
historical aggregate timing receipts, but reads zero official rows and performs zero model fits.
Execution remains fail-closed until the root execute signal is bound into the config.
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRegressor, Pool

from p3_wave.catboost_ordered_hpo import (
    CONTROL_ID,
    EXPERIMENT_ID,
    HPOContractError,
    apply_frozen_kma_alpha,
    control_candidate,
    evaluate_confirmation_gate,
    evaluate_selection_gate,
    materialize_grid,
    metric_deltas,
    paired_case_bootstrap,
    rank_candidates,
    sha256_file,
    validate_schedule,
    validate_windows,
)
from p3_wave.corrected_repeated_forward import build_corrected_repeated_forward_folds
from p3_wave.kma_source_meta import integrate_frozen_router, read_frozen_router_components
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.validation import expand_leads

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/p3_catboost_ordered_hpo_20260829_v1.json"
FORBIDDEN_BASENAMES = {
    "test_context.parquet",
    "test_index.csv",
    "sample_submission.csv",
    "baseline_persistence.csv",
    "score.py",
}


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_physical", ctypes.c_ulonglong),
        ("available_physical", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("available_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("available_virtual", ctypes.c_ulonglong),
        ("available_extended_virtual", ctypes.c_ulonglong),
    ]


def _physical_memory_bytes() -> int:
    status = _MemoryStatusEx()
    status.length = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise HPOContractError("GlobalMemoryStatusEx failed")
    return int(status.total_physical)


def _contained(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise HPOContractError(f"path escapes repository: {relative}") from exc
    if path.name.lower() in FORBIDDEN_BASENAMES:
        raise HPOContractError(f"official path is forbidden: {path.name}")
    return path


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()


def load_config(path: Path) -> dict[str, Any]:
    if path.resolve() != DEFAULT_CONFIG.resolve():
        raise HPOContractError("only the canonical config path is allowed")
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise HPOContractError("experiment id changed")
    if config["data_boundary"] != {
        "historical_train_only": True,
        "official_rows_read": 0,
        "official_files_forbidden": [
            "test_context.parquet",
            "test_index.csv",
            "sample_submission.csv",
            "baseline_persistence.csv",
            "score.py",
        ],
        "absolute_official_time_reconstruction_allowed": False,
        "external_evaluation_period_matching_allowed": False,
        "csv_output_allowed": False,
        "submission_or_upload_allowed": False,
        "source_mutation_allowed": False,
    }:
        raise HPOContractError("data boundary changed")
    execution_state = (
        config["execution"]["allowed_mode_now"],
        config["execution"]["actual_authorized"],
    )
    if execution_state not in {("static-preflight", False), ("one-shot-execute", True)}:
        raise HPOContractError("execution authorization state is invalid")
    validate_windows(config)
    validate_schedule(config)
    return config


def _input_paths(config: dict[str, Any], data_dir: Path | None = None) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, item in config["inputs"].items():
        if "path" in item:
            result[name] = _contained(str(item["path"]))
    if data_dir is not None:
        historical_root = data_dir.resolve(strict=True)
        wave_relative = config["inputs"]["train_wave"]["relative_path"]
        if Path(wave_relative).name != "train_wave.csv":
            raise HPOContractError("historical wave relative path changed")
        result["train_wave"] = historical_root / wave_relative
    return result


def _verify_hashes(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise HPOContractError(f"missing frozen input: {name}")
        digest = sha256_file(path)
        if digest != config["inputs"][name]["sha256"]:
            raise HPOContractError(f"frozen input hash changed: {name}")
        observed[name] = digest
    return observed


def _schema_checks(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    feature_file = pq.ParquetFile(paths["train_features"])
    anchor_file = pq.ParquetFile(paths["train_anchors"])
    router_file = pq.ParquetFile(paths["frozen_router_oof"])
    kma_file = pq.ParquetFile(paths["frozen_kma_oof"])
    if feature_file.metadata.num_rows != config["inputs"]["train_features"]["expected_rows"]:
        raise HPOContractError("train feature row metadata changed")
    if anchor_file.metadata.num_rows != config["inputs"]["train_anchors"]["expected_rows"]:
        raise HPOContractError("train anchor row metadata changed")
    if router_file.metadata.num_rows != config["inputs"]["frozen_router_oof"]["expected_rows"]:
        raise HPOContractError("frozen router row metadata changed")
    if kma_file.metadata.num_rows != config["inputs"]["frozen_kma_oof"]["expected_rows"]:
        raise HPOContractError("frozen KMA row metadata changed")

    feature_schema = feature_file.schema_arrow.names
    compact = compact_feature_columns(
        [name for name in feature_schema if name not in {"anchor_id", "station"}]
    )
    expected_count = config["selection"]["expected_feature_count"]
    if len(compact) != expected_count or len(set(compact)) != expected_count:
        raise HPOContractError("compact feature surface is not exactly 591 unique columns")
    frozen_columns = json.loads(paths["feature_columns"].read_text(encoding="utf-8"))
    if frozen_columns != compact:
        raise HPOContractError("frozen feature column order differs from compact feature surface")

    anchor_required = {
        "anchor_id",
        "station",
        "anchor_time",
        "current_hs",
        "target_3",
        "target_6",
        "target_9",
        "target_12",
        "target_18",
        "target_24",
    }
    router_required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "single_prediction",
        "multi_prediction",
        "persistence",
        "weight_single",
        "weight_multi",
        "weight_persistence",
        "second_stage_persistence_weight",
        "prediction",
    }
    kma_required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "incumbent_final",
        "calibrated_source",
    }
    if not anchor_required.issubset(anchor_file.schema_arrow.names):
        raise HPOContractError("anchor schema changed")
    if not router_required.issubset(router_file.schema_arrow.names):
        raise HPOContractError("frozen router schema changed")
    if not kma_required.issubset(kma_file.schema_arrow.names):
        raise HPOContractError("frozen KMA schema changed")
    result = {
        "train_feature_rows_from_metadata": feature_file.metadata.num_rows,
        "train_anchor_rows_from_metadata": anchor_file.metadata.num_rows,
        "compact_feature_count_from_schema": len(compact),
        "frozen_router_rows_from_metadata": router_file.metadata.num_rows,
        "frozen_kma_rows_from_metadata": kma_file.metadata.num_rows,
        "official_or_confirmation_rows_materialized": 0,
    }
    if "train_wave" not in paths:
        result["historical_split_validation"] = "PENDING_EXPLICIT_DATA_DIR"
        return result

    anchors = pd.read_parquet(
        paths["train_anchors"], columns=["anchor_id", "station", "anchor_time", "current_hs"]
    )
    wave = pd.read_csv(paths["train_wave"], usecols=["station", "time", "hs"])
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["selection"]["windows"],
        gap_hours=config["selection"]["embargo_hours"],
        footprint_hours=config["selection"]["footprint_hours"],
    )
    observed_by_fold = [len(fold.validation_ids) for fold in folds]
    if len(selected) != config["selection"]["expected_validation_cases"]:
        raise HPOContractError("selection validation case count changed")
    if observed_by_fold != config["selection"]["expected_validation_cases_by_fold"]:
        raise HPOContractError("selection fold case counts changed")
    if split_audit["context48_plus_target24_footprint_overlap_pairs"] != 0:
        raise HPOContractError("selection footprints overlap")
    if min(split_audit["station_global_minimum_gap_hours"].values()) < 78.0:
        raise HPOContractError("selection station-global gap is below 78h")
    result.update(
        {
            "historical_split_validation": "PASS",
            "historical_rows_materialized": len(anchors) + len(wave),
            "selection_cases": len(selected),
            "selection_cases_by_fold": {
                fold.name: len(fold.validation_ids) for fold in folds
            },
            "selection_train_anchors_by_fold": {fold.name: len(fold.train_ids) for fold in folds},
            "selection_minimum_station_gap_hours": split_audit[
                "station_global_minimum_gap_hours"
            ],
            "selection_footprint_overlap_pairs": split_audit[
                "context48_plus_target24_footprint_overlap_pairs"
            ],
        }
    )
    return result


def _grid_checks(config: dict[str, Any], paths: dict[str, Path]) -> dict[str, Any]:
    grid_spec = json.loads(paths["grid"].read_text(encoding="utf-8"))
    challengers = materialize_grid(grid_spec)
    control = control_candidate(grid_spec)
    for candidate in [control, *challengers]:
        parameters = candidate["parameters"]
        if (
            parameters.get("boosting_type") == "Ordered"
            and parameters.get("grow_policy") != "SymmetricTree"
        ):
            raise HPOContractError(
                "CatBoost compatibility rejected before fit: "
                f"{candidate['candidate_id']} uses Ordered boosting with "
                f"grow_policy={parameters.get('grow_policy')}; "
                "Ordered boosting requires SymmetricTree"
            )
        model = CatBoostRegressor(**candidate["parameters"])
        params = model.get_params()
        if int(params["thread_count"]) != config["hardware"]["threads_per_fit"]:
            raise HPOContractError("CatBoost thread count differs from hardware contract")
    return {
        "challenger_count": len(challengers),
        "control_count": 1,
        "catboost_parameter_construction_count": len(challengers) + 1,
        "catboost_fit_count": 0,
        "maximum_authorized_future_fit_count": validate_schedule(config),
    }


def _runtime_calibration(paths: dict[str, Path]) -> dict[str, Any]:
    historical = json.loads(paths["runtime_calibration"].read_text(encoding="utf-8"))
    receipts = historical["training_receipts"]
    elapsed = [float(row["elapsed_seconds"]) for row in receipts]
    train_single_rows = [int(row["train_single_rows"]) for row in receipts]
    # Historical receipts include a 700-tree CPU single model and a 1200-tree GPU multi model.
    # A 300-tree control-only fit is therefore conservatively bounded below that combined time;
    # the range deliberately includes data setup and current-host variance.
    eta_low = max(20, int(round(min(elapsed) * 0.25)))
    eta_high = max(60, int(round(max(elapsed) * 0.60)))
    return {
        "source_experiment": historical["experiment_id"],
        "historical_combined_single700_multi1200_fold_seconds": elapsed,
        "historical_single_row_range": [min(train_single_rows), max(train_single_rows)],
        "projected_first_control_300_tree_seconds": [eta_low, eta_high],
        "projection_note": "No timing fit was run; estimate is conservatively scaled from pinned historical combined-fit receipts.",
    }


def static_preflight(config_path: Path, data_dir: Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    if _git_head() != config["base_commit"]:
        raise HPOContractError("repository HEAD differs from base commit")
    paths = _input_paths(config, data_dir)
    hashes = _verify_hashes(config, paths)
    logical_cpu = os.cpu_count()
    memory_bytes = _physical_memory_bytes()
    hardware = config["hardware"]
    if logical_cpu != hardware["required_logical_cpu"]:
        raise HPOContractError("logical CPU count differs from the frozen hardware contract")
    if memory_bytes < hardware["minimum_physical_memory_gib"] * (1 << 30):
        raise HPOContractError("physical memory is below the frozen hardware contract")
    if hardware["max_workers_after_execute_signal"] != 1 or hardware["threads_per_fit"] != 6:
        raise HPOContractError("authorized execution must remain one worker by six threads")

    output_root = _contained(config["outputs"]["artifact_dir"])
    attempt_lock = _contained(config["outputs"]["attempt_lock"])
    if attempt_lock.exists():
        raise HPOContractError("attempt lock already exists")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": (
            "STATIC_PREFLIGHT_PASS_ROOT_AUTHORIZED"
            if config["execution"]["actual_authorized"]
            else "STATIC_PREFLIGHT_PASS_AWAITING_ROOT_EXECUTE_SIGNAL"
        ),
        "base_commit": config["base_commit"],
        "config_sha256": sha256_file(config_path),
        "implementation_hashes": {
            "runner": sha256_file(Path(__file__)),
            "contract_module": sha256_file(ROOT / "src/p3_wave/catboost_ordered_hpo.py"),
            "tests": sha256_file(
                ROOT / "tests/test_p3_catboost_ordered_hpo_20260829_v1.py"
            ),
        },
        "input_hashes": hashes,
        "schema": _schema_checks(config, paths),
        "grid": _grid_checks(config, paths),
        "hardware": {
            "logical_cpu": logical_cpu,
            "physical_memory_bytes": memory_bytes,
            "physical_memory_gib": memory_bytes / (1 << 30),
            "future_max_workers": 1,
            "threads_per_fit": hardware["threads_per_fit"],
        },
        "runtime_calibration": _runtime_calibration(paths),
        "execution_boundary": {
            "model_fit_count": 0,
            "official_rows_read": 0,
            "csv_files_written": 0,
            "attempt_lock_created": False,
            "output_root_preexisting": output_root.exists(),
            "actual_authorized": config["execution"]["actual_authorized"],
        },
    }


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise HPOContractError("existing static preflight receipt differs")
        return
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        os.write(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("exclusive write made no progress")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    _exclusive_bytes(path, encoded + b"\n")


def _exclusive_parquet(path: Path, frame: pd.DataFrame) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression="zstd")
    _exclusive_bytes(path, buffer.getvalue())


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    result["lead_h"] = result["lead_h"].astype(str)
    return result


def _fit_predict(
    parameters: dict[str, Any],
    iterations: int,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    feature_columns: list[str],
    train_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    early_stopping_rounds: int | None = None,
) -> tuple[pd.DataFrame, int, float]:
    x_train, y_train, train_meta = expand_leads(
        features, anchors, train_ids, feature_columns
    )
    x_validation, _, validation_meta = expand_leads(
        features, anchors, validation_ids, feature_columns
    )
    x_train = _cat_frame(x_train)
    x_validation = _cat_frame(x_validation)
    params = dict(parameters)
    params["iterations"] = int(iterations)
    model = CatBoostRegressor(**params)
    train_pool = Pool(
        x_train,
        y_train,
        weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
    )
    started = time.perf_counter()
    if early_stopping_rounds is None:
        model.fit(train_pool, verbose=False)
    else:
        validation_delta = (
            validation_meta["target_hs"].to_numpy()
            - validation_meta["current_hs"].to_numpy()
        )
        model.fit(
            train_pool,
            eval_set=Pool(x_validation, validation_delta, cat_features=[0, 1]),
            early_stopping_rounds=int(early_stopping_rounds),
            use_best_model=True,
            verbose=False,
        )
    elapsed = time.perf_counter() - started
    prediction = np.clip(
        validation_meta["current_hs"].to_numpy() + model.predict(x_validation), 0.0, 30.0
    )
    output = validation_meta.copy()
    output["prediction"] = prediction
    observed_best_iteration = model.get_best_iteration()
    if observed_best_iteration is None or int(observed_best_iteration) < 0:
        best_iteration = int(iterations)
    else:
        best_iteration = int(observed_best_iteration) + 1
    return output, best_iteration, elapsed


def _aggregate_prediction(
    candidate_id: str, fold_name: str, prediction: pd.DataFrame
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    error = np.square(prediction["prediction"] - prediction["target_hs"])

    def add(slice_type: str, slice_value: str, indices: pd.Index) -> None:
        rows.append(
            {
                "candidate_id": candidate_id,
                "fold": fold_name,
                "slice_type": slice_type,
                "slice_value": slice_value,
                "squared_error_sum": float(error.loc[indices].sum()),
                "row_count": int(len(indices)),
            }
        )

    add("overall", "all", prediction.index)
    for station, group in prediction.groupby("station", sort=True, observed=True):
        add("station", str(station), group.index)
    for lead, group in prediction.groupby("lead_h", sort=True, observed=True):
        add("lead", str(int(lead)), group.index)
    return rows


def _metrics_from_aggregate(
    aggregate: pd.DataFrame, challenger_id: str
) -> dict[str, Any]:
    def score(candidate_id: str, slice_type: str, slice_value: str | None = None) -> float:
        selected = aggregate.loc[
            aggregate["candidate_id"].eq(candidate_id)
            & aggregate["slice_type"].eq(slice_type)
        ]
        if slice_value is not None:
            selected = selected.loc[selected["slice_value"].eq(slice_value)]
        return float(
            np.sqrt(selected["squared_error_sum"].sum() / selected["row_count"].sum())
        )

    def deltas(slice_type: str, key_column: str) -> dict[str, float]:
        values: dict[str, float] = {}
        if slice_type == "fold":
            keys = sorted(aggregate["fold"].unique())
            for key in keys:
                block = aggregate.loc[
                    aggregate["fold"].eq(key) & aggregate["slice_type"].eq("overall")
                ]
                challenger = block.loc[block["candidate_id"].eq(challenger_id)]
                control = block.loc[block["candidate_id"].eq(CONTROL_ID)]
                values[str(key)] = float(
                    np.sqrt(challenger["squared_error_sum"].sum() / challenger["row_count"].sum())
                    - np.sqrt(control["squared_error_sum"].sum() / control["row_count"].sum())
                )
            return values
        keys = sorted(aggregate.loc[aggregate["slice_type"].eq(slice_type), key_column].unique())
        for key in keys:
            values[str(key)] = score(challenger_id, slice_type, str(key)) - score(
                CONTROL_ID, slice_type, str(key)
            )
        return values

    control = score(CONTROL_ID, "overall")
    challenger = score(challenger_id, "overall")
    return {
        "control_rmse_m": control,
        "challenger_rmse_m": challenger,
        "delta_rmse_m": challenger - control,
        "by_fold": deltas("fold", "fold"),
        "by_station": deltas("station", "slice_value"),
        "by_lead": deltas("lead", "slice_value"),
    }


def _execution_data(
    config: dict[str, Any], paths: dict[str, Path]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], tuple[Any, ...]]:
    features = pd.read_parquet(paths["train_features"])
    anchors = pd.read_parquet(paths["train_anchors"])
    wave = pd.read_csv(paths["train_wave"], usecols=["station", "time", "hs"])
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    folds, selected, _ = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["selection"]["windows"],
        gap_hours=78,
        footprint_hours=72,
    )
    if len(selected) != config["selection"]["expected_validation_cases"]:
        raise HPOContractError("execution selection membership differs from preflight")
    return features, anchors, feature_columns, folds


def _confirmation_train_ids(
    anchors: pd.DataFrame,
    validation_ids: np.ndarray,
    validation_start: str,
) -> np.ndarray:
    validation = anchors.set_index("anchor_id").loc[validation_ids]
    cutoff = pd.Timestamp(validation_start, tz="UTC") - pd.Timedelta(hours=78)
    train = anchors.loc[anchors["anchor_time"].lt(cutoff)].copy()
    forbidden = set(
        zip(validation["station"].astype(str), validation["episode_id"].astype(int), strict=True)
    )
    keep = [
        (str(row.station), int(row.episode_id)) not in forbidden
        for row in train.itertuples(index=False)
    ]
    train = train.loc[np.asarray(keep, dtype=bool)]
    if np.intersect1d(train["anchor_id"], validation_ids).size:
        raise HPOContractError("confirmation train/validation ids overlap")
    return np.sort(train["anchor_id"].to_numpy(dtype=np.int64))


def execute_hpo(config_path: Path, data_dir: Path, authorization_token: str | None) -> dict[str, Any]:
    execution_started = time.monotonic()
    config = load_config(config_path)
    if config["execution"]["actual_authorized"] is not True:
        raise HPOContractError("execute is disabled until the root signal is bound to config")
    if authorization_token != config["execution"]["future_execute_token"]:
        raise HPOContractError("execute authorization token differs")
    wall_cap_seconds = float(config["hardware"]["projected_wall_time_hours_max"]) * 3600.0
    deadline = execution_started + wall_cap_seconds

    def emit(stage: str, **payload: Any) -> None:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "stage": stage,
                    "elapsed_seconds": time.monotonic() - execution_started,
                    **payload,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    def require_wall_time(stage: str) -> None:
        if time.monotonic() >= deadline:
            raise HPOContractError(f"12h wall cap reached before {stage}")

    preflight = static_preflight(config_path, data_dir)
    paths = _input_paths(config, data_dir)
    outputs = {name: _contained(path) for name, path in config["outputs"].items() if name != "artifact_dir"}
    consumed = [name for name, path in outputs.items() if name != "static_preflight_receipt" and path.exists()]
    if consumed:
        raise HPOContractError(f"one-shot output already exists: {sorted(consumed)}")
    _exclusive_json(
        outputs["attempt_lock"],
        {
            "experiment_id": EXPERIMENT_ID,
            "config_sha256": sha256_file(config_path),
            "grid_sha256": config["inputs"]["grid"]["sha256"],
            "rerun_forbidden": True,
        },
    )
    emit(
        "attempt_locked_execution_started",
        thread_count=config["hardware"]["threads_per_fit"],
        max_workers=config["hardware"]["max_workers_after_execute_signal"],
        maximum_selection_fits=config["successive_halving"]["maximum_fit_count"],
        wall_cap_hours=config["hardware"]["projected_wall_time_hours_max"],
    )

    features, anchors, feature_columns, folds = _execution_data(config, paths)
    grid_spec = json.loads(paths["grid"].read_text(encoding="utf-8"))
    challengers = materialize_grid(grid_spec)
    control = control_candidate(grid_spec)
    candidate_map = {row["candidate_id"]: row for row in [control, *challengers]}
    active_ids = [row["candidate_id"] for row in challengers]
    timing: list[dict[str, Any]] = []
    final_best_iterations: dict[str, list[int]] = {}
    rung_receipts: list[dict[str, Any]] = []
    final_rung_frame: pd.DataFrame | None = None
    completed_selection_fits = 0

    for rung in config["successive_halving"]["rungs"]:
        current_ids = [CONTROL_ID, *active_ids]
        rung_records: list[dict[str, Any]] = []
        for candidate_id in current_ids:
            candidate = candidate_map[candidate_id]
            for fold in folds[: int(rung["selection_fold_count"])]:
                require_wall_time(f"{rung['name']}:{candidate_id}:{fold.name}")
                prediction, best_iteration, elapsed = _fit_predict(
                    candidate["parameters"],
                    int(rung["iterations"]),
                    features,
                    anchors,
                    feature_columns,
                    fold.train_ids,
                    fold.validation_ids,
                    early_stopping_rounds=rung.get("early_stopping_rounds"),
                )
                completed_selection_fits += 1
                require_wall_time(f"post-fit:{rung['name']}:{candidate_id}:{fold.name}")
                rung_records.extend(_aggregate_prediction(candidate_id, fold.name, prediction))
                timing.append(
                    {
                        "rung": rung["name"],
                        "candidate_id": candidate_id,
                        "fold": fold.name,
                        "elapsed_seconds": elapsed,
                        "best_iteration": best_iteration,
                    }
                )
                if rung["name"] == "rung_2500":
                    final_best_iterations.setdefault(candidate_id, []).append(best_iteration)
                if completed_selection_fits == 1:
                    emit(
                        "first_control_fit_complete",
                        rung=rung["name"],
                        fold=fold.name,
                        fit_seconds=elapsed,
                        iterations=rung["iterations"],
                        completed_selection_fits=completed_selection_fits,
                    )
                elif completed_selection_fits % 10 == 0:
                    emit(
                        "selection_progress",
                        rung=rung["name"],
                        completed_selection_fits=completed_selection_fits,
                    )
        rung_frame = pd.DataFrame(rung_records)
        final_rung_frame = rung_frame
        overall = rung_frame.loc[rung_frame["slice_type"].eq("overall")]
        parameters = {
            candidate_id: candidate_map[candidate_id]["parameters"] for candidate_id in current_ids
        }
        ranking = rank_candidates(
            overall,
            parameters,
            tie_tolerance_rmse_m=config["selection"]["tie_tolerance_rmse_m"],
        )
        ranked_challengers = ranking.loc[
            ranking["candidate_id"].ne(CONTROL_ID), "candidate_id"
        ].tolist()
        active_ids = ranked_challengers[: int(rung["challenger_keep"])]
        rung_receipts.append(
            {
                "name": rung["name"],
                "iterations": rung["iterations"],
                "fit_count": len(current_ids) * int(rung["selection_fold_count"]),
                "surviving_challengers": active_ids,
                "ranking": ranking.to_dict(orient="records"),
            }
        )
        emit(
            "rung_complete",
            rung=rung["name"],
            completed_selection_fits=completed_selection_fits,
            surviving_challenger_count=len(active_ids),
        )

    selected_id = active_ids[0]
    if final_rung_frame is None:
        raise HPOContractError("successive-halving produced no final rung")
    last_records = final_rung_frame.loc[
        final_rung_frame["candidate_id"].isin([CONTROL_ID, selected_id])
    ]
    selection_metrics = _metrics_from_aggregate(last_records, selected_id)
    selection_gate = evaluate_selection_gate(selection_metrics, config["selection_gate"])
    selection_result = {
        "experiment_id": EXPERIMENT_ID,
        "selected_candidate_id": selected_id,
        "selected_parameters": candidate_map[selected_id]["parameters"],
        "selected_best_iterations": final_best_iterations[selected_id],
        "metrics": selection_metrics,
        "gate": selection_gate,
        "rungs": rung_receipts,
        "timing": timing,
        "aggregate_only_no_raw_predictions": True,
    }
    _exclusive_json(outputs["selection_aggregate"], selection_result)
    emit(
        "selection_gate_complete",
        selected_candidate_id=selected_id,
        selection_gate_pass=selection_gate["pass"],
        completed_selection_fits=completed_selection_fits,
    )
    if not selection_gate["pass"]:
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": "SELECTION_GATE_FAIL_HPO_CLOSED",
            "preflight": preflight,
            "selection": selection_result,
            "confirmation_fit_count": 0,
            "official_rows_read": 0,
            "csv_files_written": 0,
        }
        _exclusive_json(outputs["result"], result)
        emit("terminal_selection_gate_fail")
        return result

    selected_iteration = int(np.floor(np.median(final_best_iterations[selected_id])))
    router = read_frozen_router_components(paths["frozen_router_oof"])
    kma = pd.read_parquet(
        paths["frozen_kma_oof"],
        columns=["fold", "anchor_id", "station", "lead_h", "calibrated_source"],
    )
    blind_blocks: list[pd.DataFrame] = []
    confirmation_windows = {row[0]: row for row in config["confirmation"]["windows"]}
    for fold_name, router_fold in router.groupby("fold", sort=False, observed=True):
        require_wall_time(f"confirmation:{fold_name}")
        validation_ids = np.sort(router_fold["anchor_id"].unique().astype(np.int64))
        train_ids = _confirmation_train_ids(
            anchors, validation_ids, confirmation_windows[str(fold_name)][1]
        )
        challenger, _, _ = _fit_predict(
            candidate_map[selected_id]["parameters"],
            selected_iteration,
            features,
            anchors,
            feature_columns,
            train_ids,
            validation_ids,
        )
        challenger["fold"] = str(fold_name)
        control_single = router_fold[
            ["fold", "anchor_id", "station", "lead_h", "current_hs", "single_prediction"]
        ].rename(columns={"single_prediction": "control_single_prediction"})
        single = control_single.merge(
            challenger[["anchor_id", "station", "lead_h", "prediction"]].rename(
                columns={"prediction": "challenger_single_prediction"}
            ),
            on=["anchor_id", "station", "lead_h"],
            validate="one_to_one",
        )
        integrated = integrate_frozen_router(single, router_fold)
        if not np.allclose(
            integrated["control_final"],
            integrated["incumbent_final"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise HPOContractError("frozen control router/shrink reconstruction changed")
        blind_blocks.append(integrated)
        emit("confirmation_fold_fit_complete", fold=str(fold_name))
    integrated = pd.concat(blind_blocks, ignore_index=True)
    blind = integrated.merge(
        kma, on=["fold", "anchor_id", "station", "lead_h"], validate="one_to_one"
    )
    if len(blind) != config["inputs"]["frozen_router_oof"]["expected_rows"]:
        raise HPOContractError("confirmation intersection is not 182 complete cases")
    blind["control_prediction"] = apply_frozen_kma_alpha(
        blind["control_final"], blind["calibrated_source"], blind["lead_h"]
    )
    blind["challenger_prediction"] = apply_frozen_kma_alpha(
        blind["challenger_final"], blind["calibrated_source"], blind["lead_h"]
    )
    blind = blind[
        [
            "fold",
            "anchor_id",
            "station",
            "lead_h",
            "control_prediction",
            "challenger_prediction",
        ]
    ].sort_values(["fold", "anchor_id", "station", "lead_h"], kind="mergesort")
    _exclusive_parquet(outputs["confirmation_blind_predictions"], blind)
    seal = {
        "experiment_id": EXPERIMENT_ID,
        "phase": "confirmation_before_truth_metric",
        "prediction_sha256": sha256_file(outputs["confirmation_blind_predictions"]),
        "row_count": len(blind),
        "columns": list(blind.columns),
        "truth_columns_present": False,
        "config_sha256": sha256_file(config_path),
    }
    _exclusive_json(outputs["confirmation_seal"], seal)
    sealed = pd.read_parquet(outputs["confirmation_blind_predictions"])
    truth_blocks: list[pd.DataFrame] = []
    anchor_lookup = anchors.set_index("anchor_id")
    for lead in [3, 6, 9, 12, 18, 24]:
        block = sealed.loc[sealed["lead_h"].eq(lead), ["fold", "anchor_id", "station", "lead_h"]]
        block = block.copy()
        block["target_hs"] = anchor_lookup.loc[block["anchor_id"], f"target_{lead}"].to_numpy()
        truth_blocks.append(block)
    truth = pd.concat(truth_blocks, ignore_index=True)
    evaluated = sealed.merge(
        truth, on=["fold", "anchor_id", "station", "lead_h"], validate="one_to_one"
    )
    confirmation_metrics = metric_deltas(evaluated)
    bootstrap_config = config["confirmation"]["bootstrap"]
    bootstrap = paired_case_bootstrap(
        evaluated,
        replicates=bootstrap_config["replicates"],
        seed=bootstrap_config["seed"],
    )
    confirmation_gate = evaluate_confirmation_gate(
        confirmation_metrics, bootstrap, config["confirmation"]["gate"]
    )
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "CONFIRMATION_GATE_PASS" if confirmation_gate["pass"] else "CONFIRMATION_GATE_FAIL_HPO_CLOSED",
        "preflight": preflight,
        "selection": selection_result,
        "selected_iteration": selected_iteration,
        "confirmation": {
            "metrics": confirmation_metrics,
            "paired_case_bootstrap": bootstrap,
            "gate": confirmation_gate,
            "blind_prediction_sha256": seal["prediction_sha256"],
            "fit_count": 3,
        },
        "full_refit_fit_count": 0,
        "official_rows_read": 0,
        "csv_files_written": 0,
        "submission_or_upload_attempted": False,
    }
    _exclusive_json(outputs["result"], result)
    emit("terminal_confirmation_complete", confirmation_gate_pass=confirmation_gate["pass"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=["static-preflight", "execute"], default="static-preflight")
    parser.add_argument("--write-receipt", action="store_true")
    parser.add_argument("--authorization-token")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.mode == "execute":
        # This guard precedes all schema/data access and cannot create an attempt lock.
        if args.data_dir is None:
            raise HPOContractError("execute requires an explicit historical P3 data directory")
        result = execute_hpo(config_path, args.data_dir, args.authorization_token)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    receipt = static_preflight(config_path, args.data_dir)
    if args.write_receipt:
        config = load_config(config_path)
        receipt_path = _contained(config["outputs"]["static_preflight_receipt"])
        _write_receipt(receipt_path, receipt)
        receipt["static_preflight_receipt"] = str(receipt_path.relative_to(ROOT))
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
