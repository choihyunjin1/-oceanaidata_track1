"""Run one sealed P3 ERA5 joint wave-state multitask transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p3_wave.era5_context_transfer import (  # noqa: E402
    LEADS,
    LOCAL_CATBOOST_PARAMETERS,
    SOURCE_CATBOOST_PARAMETERS,
    FixedContextTransferRegressor,
    canonicalize_era5_hourly,
    common_feature_columns,
    select_source_year_validation,
    summarize_past_48h,
)
from p3_wave.joint_wave_state_multitask import (  # noqa: E402
    STATE_NAMES,
    JointWaveStateTransferRegressor,
    apply_frozen_persistence_shrink,
    apply_joint_increment,
)

EXPERIMENT_ID = "p3_era5_joint_wave_state_multitask_transfer_20260828_v1"
CONFIG = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
RUNNER = Path(__file__).resolve()
MODULE = ROOT / "src" / "p3_wave" / "joint_wave_state_multitask.py"
TEST = ROOT / "tests" / "test_p3_joint_wave_state_multitask_transfer_20260828_v1.py"
SOURCE_TRAIN_YEARS = tuple(range(2014, 2021))
SOURCE_HELD_YEARS = (2021, 2022, 2023)


class ContractError(RuntimeError):
    """Raised when the one-shot contract changes."""


def _load_helper() -> Any:
    path = ROOT / "scripts" / "run_p3_era5_wave_directional_energy_memory_20260828_v1.py"
    spec = importlib.util.spec_from_file_location("_p3_joint_helper", path)
    if spec is None or spec.loader is None:
        raise ContractError("could not load read-only validation helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPER = _load_helper()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_data_dir(argument: Path | None) -> Path:
    value = argument or (Path(os.environ["P3_DATA_DIR"]) if "P3_DATA_DIR" in os.environ else None)
    if value is None:
        raise ContractError("P3_DATA_DIR or --data-dir is required")
    return value.resolve()


def _verify(path: Path, record: dict[str, Any], label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    if "bytes" in record and path.stat().st_size != int(record["bytes"]):
        raise ContractError(f"{label} byte size changed")
    observed = sha256_file(path)
    if observed != str(record["sha256"]).lower():
        raise ContractError(f"{label} SHA changed")
    return observed


def _load_contract(data_dir: Path) -> tuple[dict[str, Any], Any, Any, dict[str, str]]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["experiment_id"] != EXPERIMENT_ID:
        raise ContractError("experiment ID changed")
    if config["status"] != "PREREGISTERED_ONE_SHOT_LOCAL_ONLY_EXECUTION_APPROVED_2026-08-28":
        raise ContractError("approval status changed")
    policy = config["execution_policy"]
    if (
        int(policy["maximum_executions"]) != 1
        or bool(policy["result_based_retry_or_tuning"])
        or bool(policy["official_test_context_index_sample_submission_access_allowed"])
        or bool(policy["submission_csv_generation_allowed"])
        or bool(policy["upload_allowed"])
    ):
        raise ContractError("execution boundary changed")
    if tuple(config["candidate"]["target_states"]) != STATE_NAMES:
        raise ContractError("joint target state order changed")
    if tuple(config["candidate"]["active_leads_h"]) != (18, 24):
        raise ContractError("active leads changed")
    if float(config["candidate"]["candidate_increment_weight"]) != 0.20:
        raise ContractError("candidate increment changed")
    if int(config["base_contract"]["frozen_feature_count"]) != len(common_feature_columns()):
        raise ContractError("frozen feature count changed")
    if dict(config["model"]["source_pretrain"]) != dict(SOURCE_CATBOOST_PARAMETERS):
        raise ContractError("source hyperparameters changed")
    local = dict(config["model"]["local_continuation"])
    local.pop("sample_weight")
    if local != dict(LOCAL_CATBOOST_PARAMETERS) or int(config["model"]["maximum_fits"]) != 4:
        raise ContractError("local schedule or fit budget changed")

    hashes = {
        "base_contract": _verify(
            ROOT / config["base_contract"]["path"], config["base_contract"], "base contract"
        )
    }
    for label, record in config["immutable_inputs"].items():
        hashes[label] = _verify(ROOT / record["path"], record, label)
    hashes["train_wave"] = _verify(
        data_dir / config["source_train_wave"]["filename"],
        config["source_train_wave"],
        "train_wave",
    )
    hashes.update(
        config=sha256_file(CONFIG),
        module=sha256_file(MODULE),
        runner=sha256_file(RUNNER),
        test=sha256_file(TEST),
    )
    frozen = HELPER._load_frozen_runner()
    _, _, paths = frozen._load_contract(ROOT)
    return config, frozen, paths, hashes


def _create_attempt_lock(path: Path, config_sha: str) -> str:
    payload = {
        "schema_version": "p3.joint_wave_state.attempt.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_sha,
        "pid": os.getpid(),
        "maximum_executions": 1,
        "official_access": False,
        "upload_allowed": False,
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return hashlib.sha256(raw).hexdigest()


def _build_source_surface(hourly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, float]] = []
    next_id = 1
    for station, group in hourly.groupby("station", sort=False, observed=True):
        canonical = canonicalize_era5_hourly(group.drop(columns="station"), time_column="time")
        first = canonical["time"].iloc[0] + pd.Timedelta(hours=48)
        last = canonical["time"].iloc[-1] - pd.Timedelta(hours=24)
        candidates = pd.date_range(first, last, freq="6h")
        hs_by_time = canonical.set_index("time")["hs"]
        current = hs_by_time.reindex(candidates).to_numpy(dtype=np.float64)
        complete = np.ones(len(candidates), dtype=bool)
        for lead in LEADS:
            future = hs_by_time.reindex(candidates + pd.Timedelta(hours=lead)).to_numpy(
                dtype=np.float64
            )
            complete &= np.isfinite(future) & (future >= 0.0)
        valid = np.isfinite(current) & (current >= 1.5) & complete
        times = canonical["time"].dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(
            np.int64
        )
        for anchor_time, current_hs in zip(candidates[valid], current[valid], strict=True):
            left = int(
                np.searchsorted(times, (anchor_time - pd.Timedelta(hours=48)).value, side="left")
            )
            right = int(np.searchsorted(times, anchor_time.value, side="right"))
            history = canonical.iloc[left:right]
            if len(history) != 49:
                raise ContractError("source context is not exactly 49 hourly rows")
            metadata_rows.append(
                {
                    "anchor_id": next_id,
                    "station": str(station),
                    "anchor_time": anchor_time,
                    "current_hs": float(current_hs),
                }
            )
            feature_rows.append(summarize_past_48h(history, anchor_time=anchor_time))
            next_id += 1
    metadata = pd.DataFrame(metadata_rows)
    features = pd.DataFrame(feature_rows, columns=common_feature_columns())
    if len(metadata) != len(features):
        raise ContractError("source metadata and features do not align")
    return metadata, features


def _source_split(metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    times = pd.to_datetime(metadata["anchor_time"], utc=True, errors="raise")
    years = times.dt.year
    complete_year = (
        (times - pd.Timedelta(hours=48)).dt.year.eq(years)
        & (times + pd.Timedelta(hours=24)).dt.year.eq(years)
    )
    train_ids = metadata.loc[
        complete_year & years.isin(SOURCE_TRAIN_YEARS), "anchor_id"
    ].to_numpy(dtype=np.int64)
    held = select_source_year_validation(metadata, held_years=SOURCE_HELD_YEARS)
    if len(train_ids) != 7311 or len(held) != 492:
        raise ContractError("frozen source split counts changed")
    return train_ids, held


def _joint_targets_from_hourly(
    hourly: pd.DataFrame, metadata: pd.DataFrame, anchor_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    rows = metadata.set_index("anchor_id").loc[ids]
    indexed = hourly.set_index(["station", "time"])[list(STATE_NAMES)]
    targets = np.full((len(ids), len(LEADS), len(STATE_NAMES)), np.nan, dtype=np.float64)
    current = np.full((len(ids), len(STATE_NAMES)), np.nan, dtype=np.float64)
    for position, row in enumerate(rows.itertuples()):
        key = (str(row.station), pd.Timestamp(row.anchor_time))
        current[position] = indexed.loc[key].to_numpy(dtype=np.float64)
        for lead_position, lead in enumerate(LEADS):
            targets[position, lead_position] = indexed.loc[
                (key[0], key[1] + pd.Timedelta(hours=lead))
            ].to_numpy(dtype=np.float64)
    eligible = (
        np.isfinite(current).all(axis=1)
        & (current >= 0.0).all(axis=1)
        & np.isfinite(targets).all(axis=(1, 2))
        & (targets >= 0.0).all(axis=(1, 2))
    )
    transformed = np.log1p(targets[eligible]) - np.log1p(current[eligible])[:, None, :]
    return ids[eligible], transformed, current[eligible]


def _read_local_joint_targets(
    wave: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: np.ndarray,
    frozen: Any,
    anchor_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    rows = anchors.set_index("anchor_id").loc[ids]
    indexed = wave.set_index(["station", "time"])[list(STATE_NAMES)]
    future = np.full((len(ids), len(LEADS), len(STATE_NAMES)), np.nan, dtype=np.float64)
    current = np.full((len(ids), len(STATE_NAMES)), np.nan, dtype=np.float64)
    for position, row in enumerate(rows.itertuples()):
        key = (str(row.station), pd.Timestamp(row.anchor_time))
        current[position] = indexed.loc[key].to_numpy(dtype=np.float64)
        for lead_position, lead in enumerate(LEADS):
            future[position, lead_position] = indexed.loc[
                (key[0], key[1] + pd.Timedelta(hours=lead))
            ].to_numpy(dtype=np.float64)
    eligible = (
        np.isfinite(current).all(axis=1)
        & (current >= 0.0).all(axis=1)
        & np.isfinite(future).all(axis=(1, 2))
        & (future >= 0.0).all(axis=(1, 2))
    )
    kept = ids[eligible]
    transformed = np.log1p(future[eligible]) - np.log1p(current[eligible])[:, None, :]
    frozen_target = frozen._read_training_targets(anchor_path, kept)
    frozen_log = frozen._log_delta_targets(frozen_target)
    if not np.allclose(transformed[:, :, 0], frozen_log, rtol=0.0, atol=1e-12):
        raise ContractError("train_wave Hs targets do not reproduce frozen anchor targets")
    return kept, transformed, current[eligible]


def _source_hs_truth(
    hourly: pd.DataFrame, metadata: pd.DataFrame, anchor_ids: np.ndarray
) -> np.ndarray:
    ids = np.asarray(anchor_ids, dtype=np.int64)
    rows = metadata.set_index("anchor_id").loc[ids]
    indexed = hourly.set_index(["station", "time"])["hs"]
    result = np.empty((len(ids), len(LEADS)), dtype=np.float64)
    for position, row in enumerate(rows.itertuples()):
        for lead_position, lead in enumerate(LEADS):
            result[position, lead_position] = float(
                indexed.loc[
                    (str(row.station), pd.Timestamp(row.anchor_time) + pd.Timedelta(hours=lead))
                ]
            )
    if not np.isfinite(result).all():
        raise ContractError("source Hs truth is invalid")
    return result


def _source_gate(frame: pd.DataFrame, config: dict[str, Any], auxiliary_fraction: float) -> dict[str, Any]:
    gate = config["validation"]["source_gate_first"]
    overall = HELPER._metric(frame)
    by_year = HELPER._breakdown(frame, "fold")
    by_station = HELPER._breakdown(frame, "station")
    by_lead = HELPER._breakdown(frame, "lead_h")
    bootstrap = HELPER._paired_bootstrap(
        frame, int(gate["bootstrap_replicates"]), int(gate["bootstrap_seed"])
    )
    maximum = max(
        float(item["delta_m"])
        for item in [*by_year.values(), *by_station.values(), *by_lead.values()]
    )
    checks = {
        "auxiliary_complete_fraction": auxiliary_fraction
        >= float(config["validation"]["minimum_auxiliary_complete_fraction"]),
        "pooled_delta_below_zero": float(overall["delta_m"])
        < float(gate["pooled_delta_below_m"]),
        "bootstrap_ci90_upper_below_zero": float(bootstrap["ci90_upper_m"])
        < float(gate["bootstrap_ci90_upper_below_m"]),
        "all_three_years_non_degrading": len(by_year) == 3
        and all(float(item["delta_m"]) <= 0.0 for item in by_year.values()),
        "maximum_slice_regression": maximum
        <= float(gate["maximum_year_station_or_lead_regression_m"]),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "overall": overall,
        "by_year": by_year,
        "by_station": by_station,
        "by_lead": by_lead,
        "maximum_year_station_or_lead_regression_m": maximum,
        "bootstrap": bootstrap,
        "auxiliary_complete_fraction": auxiliary_fraction,
    }


def _write_result(output: Path, result: dict[str, Any], artifact_hashes: dict[str, str]) -> dict[str, Any]:
    result["artifact_hashes"] = dict(artifact_hashes)
    result_hash = HELPER._atomic_json(output / "result.json", result)
    manifest = {
        "schema_version": "p3.joint_wave_state.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "result_sha256": result_hash,
        "artifacts": artifact_hashes,
        "official_rows_read": 0,
        "submission_generated": False,
        "uploaded": False,
    }
    manifest_hash = HELPER._atomic_json(output / "manifest.json", manifest)
    return {**result, "result_sha256": result_hash, "manifest_sha256": manifest_hash}


def check_only(data_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve():
        raise ContractError("runtime root changed")
    config, frozen, paths, hashes = _load_contract(data_dir)
    source_receipt, _, _ = frozen._external_preflight(paths)
    folds, selected, audit = HELPER._shadow_split(config, data_dir, paths.train_anchors)
    support = HELPER._support_receipt(selected)
    return {
        "schema_version": "p3.joint_wave_state.check.v1",
        "experiment_id": EXPERIMENT_ID,
        "passed": True,
        "writes": 0,
        "model_fits": 0,
        "official_rows_read": 0,
        "hashes": hashes,
        "environment": {
            "python": platform.python_version(),
            "catboost": catboost.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "executable": sys.executable,
        },
        "source_external_preflight": source_receipt,
        "source_feature_count": len(common_feature_columns()),
        "target_states": list(STATE_NAMES),
        "fresh_shadow_support": support,
        "fresh_shadow_support_passed": HELPER._support_passed(config, support),
        "fresh_shadow_audit": audit,
        "fresh_shadow_fold_count": len(folds),
        "output_exists": (ROOT / config["artifact_directory"]).exists(),
        "attempt_lock_exists": (
            ROOT / "artifacts" / f"{EXPERIMENT_ID}.attempt.lock"
        ).exists(),
    }


def execute_once(data_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    if root.resolve() != ROOT.resolve():
        raise ContractError("runtime root changed")
    started = time.perf_counter()
    config, frozen, paths, hashes = _load_contract(data_dir)
    output = ROOT / config["artifact_directory"]
    lock_path = ROOT / "artifacts" / f"{EXPERIMENT_ID}.attempt.lock"
    if output.exists() or lock_path.exists():
        raise FileExistsError("one-shot attempt is already consumed")
    preflight = check_only(data_dir, root)
    lock_sha = _create_attempt_lock(lock_path, hashes["config"])
    output.mkdir(parents=True, exist_ok=False)
    artifact_hashes: dict[str, str] = {}
    fit_count = 0

    source_hourly, source_provenance = frozen._load_source_hourly(paths)
    source_metadata, source_features = _build_source_surface(source_hourly)
    source_train_ids, source_held = _source_split(source_metadata)
    eligible_ids, source_joint_target, _ = _joint_targets_from_hourly(
        source_hourly, source_metadata, source_train_ids
    )
    auxiliary_fraction = float(len(eligible_ids) / len(source_train_ids))
    if auxiliary_fraction < float(config["validation"]["minimum_auxiliary_complete_fraction"]):
        return _write_result(
            output,
            {
                "schema_version": "p3.joint_wave_state.result.v1",
                "experiment_id": EXPERIMENT_ID,
                "status": "NO_GO_SOURCE_AUXILIARY_PREFLIGHT",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "attempt_lock_sha256": lock_sha,
                "preflight": preflight,
                "input_and_code_hashes": hashes,
                "source_auxiliary_complete_fraction": auxiliary_fraction,
                "fit_count": 0,
                "official_test_sample_submission_rows_read": 0,
                "submission_generated_or_uploaded": False,
                "result_based_retry": False,
                "runtime_seconds": time.perf_counter() - started,
            },
            artifact_hashes,
        )

    lookup = source_features.copy()
    lookup.insert(0, "anchor_id", source_metadata["anchor_id"].to_numpy(dtype=np.int64))
    lookup = lookup.set_index("anchor_id")
    train_x = lookup.loc[eligible_ids, list(common_feature_columns())].reset_index(drop=True)
    base_source = FixedContextTransferRegressor().fit_pretrain(
        train_x, source_joint_target[:, :, 0]
    )
    joint_source = JointWaveStateTransferRegressor().fit_pretrain(train_x, source_joint_target)
    fit_count += 2
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    base_source.source_model.save_model(str(model_dir / "source_base.cbm"))
    joint_source.save_model(model_dir / "source_joint.cbm")
    artifact_hashes["source_base_model"] = sha256_file(model_dir / "source_base.cbm")
    artifact_hashes["source_joint_model"] = sha256_file(model_dir / "source_joint.cbm")

    held_ids = source_held["anchor_id"].to_numpy(dtype=np.int64)
    held_x = lookup.loc[held_ids, list(common_feature_columns())].reset_index(drop=True)
    held_current = source_metadata.set_index("anchor_id").loc[held_ids, "current_hs"].to_numpy(
        dtype=np.float64
    )
    base_prediction = apply_frozen_persistence_shrink(
        base_source.predict_hs(held_x, current_hs=held_current), held_current
    )
    joint_prediction = apply_frozen_persistence_shrink(
        joint_source.predict_hs(held_x, current_hs=held_current), held_current
    )
    candidate = apply_joint_increment(base_prediction, joint_prediction)
    source_blind = HELPER._matrix_rows(
        source_metadata,
        held_ids,
        source_held.assign(year=source_held["year"].astype(str)),
        base_prediction,
        joint_prediction,
        candidate,
        fold_column="year",
    )
    artifact_hashes["source_prediction_seal"] = HELPER._atomic_parquet(
        output / "source_predictions_sealed.parquet", source_blind
    )
    artifact_hashes["source_prediction_receipt"] = HELPER._atomic_json(
        output / "source_prediction_seal.json",
        {
            "rows": int(len(source_blind)),
            "prediction_sha256": HELPER.array_sha256(source_blind["candidate_prediction"]),
            "numeric_held_target_values_attached_before_seal": 0,
            "fit_count": fit_count,
        },
    )
    source_truth = _source_hs_truth(source_hourly, source_metadata, held_ids)
    source_evaluated = HELPER._attach_matrix_truth(source_blind, held_ids, source_truth)
    source_gate = _source_gate(source_evaluated, config, auxiliary_fraction)
    base_result: dict[str, Any] = {
        "schema_version": "p3.joint_wave_state.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_only": True,
        "attempt_lock_sha256": lock_sha,
        "preflight": preflight,
        "input_and_code_hashes": hashes,
        "source_provenance": source_provenance,
        "source_split": {
            "train_cases": int(len(source_train_ids)),
            "matched_auxiliary_train_cases": int(len(eligible_ids)),
            "held_cases": int(len(held_ids)),
            "train_years": list(SOURCE_TRAIN_YEARS),
            "held_years": list(SOURCE_HELD_YEARS),
        },
        "source_gate": source_gate,
        "fit_count": fit_count,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "result_based_retry": False,
    }
    if not source_gate["passed"]:
        base_result.update(
            status="NO_GO_SOURCE_GATE",
            shadow_support=None,
            shadow_gate=None,
            runtime_seconds=time.perf_counter() - started,
        )
        return _write_result(output, base_result, artifact_hashes)

    folds, selected, shadow_audit = HELPER._shadow_split(config, data_dir, paths.train_anchors)
    support = HELPER._support_receipt(selected)
    base_result["shadow_support"] = support
    base_result["shadow_audit"] = shadow_audit
    if not HELPER._support_passed(config, support):
        base_result.update(
            status="NO_GO_SUPPORT", shadow_gate=None, runtime_seconds=time.perf_counter() - started
        )
        return _write_result(output, base_result, artifact_hashes)

    anchors = HELPER._read_local_metadata(paths.train_anchors)
    local_features = frozen._read_local_features(paths.train_features)
    local_lookup = local_features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    wave = pd.read_csv(data_dir / config["source_train_wave"]["filename"])
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    blocks: list[pd.DataFrame] = []
    local_auxiliary: dict[str, Any] = {}
    for fold in folds:
        train_ids = np.asarray(fold.train_ids, dtype=np.int64)
        validation_ids = np.asarray(fold.validation_ids, dtype=np.int64)
        kept, joint_target, state_current = _read_local_joint_targets(
            wave, anchors, train_ids, frozen, paths.train_anchors
        )
        fraction = float(len(kept) / len(train_ids))
        local_auxiliary[fold.name] = {
            "requested_cases": int(len(train_ids)),
            "matched_auxiliary_cases": int(len(kept)),
            "complete_fraction": fraction,
        }
        if fraction < float(config["validation"]["minimum_auxiliary_complete_fraction"]):
            base_result.update(
                status="NO_GO_LOCAL_AUXILIARY_PREFLIGHT",
                local_auxiliary=local_auxiliary,
                shadow_gate=None,
                runtime_seconds=time.perf_counter() - started,
            )
            return _write_result(output, base_result, artifact_hashes)
        train_x = local_lookup.loc[kept, list(common_feature_columns())].reset_index(drop=True)
        base_local = base_source.clone_pretrained().continue_local(
            train_x, joint_target[:, :, 0], current_hs=state_current[:, 0]
        )
        joint_local = joint_source.clone_pretrained().continue_local(
            train_x, joint_target, current_hs=state_current[:, 0]
        )
        fit_count += 2
        base_path = model_dir / f"{fold.name}_base.cbm"
        joint_path = model_dir / f"{fold.name}_joint.cbm"
        base_local.model.save_model(str(base_path))
        joint_local.save_model(joint_path)
        artifact_hashes[f"{fold.name}_base_model"] = sha256_file(base_path)
        artifact_hashes[f"{fold.name}_joint_model"] = sha256_file(joint_path)
        validation_x = local_lookup.loc[
            validation_ids, list(common_feature_columns())
        ].reset_index(drop=True)
        current = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(dtype=np.float64)
        base = apply_frozen_persistence_shrink(
            base_local.predict_hs(validation_x, current_hs=current), current
        )
        joint = apply_frozen_persistence_shrink(
            joint_local.predict_hs(validation_x, current_hs=current), current
        )
        candidate = apply_joint_increment(base, joint)
        fold_metadata = selected.loc[selected["fold"].eq(fold.name)].copy()
        blocks.append(
            HELPER._matrix_rows(
                anchors,
                validation_ids,
                fold_metadata,
                base,
                joint,
                candidate,
                fold_column="fold",
            )
        )
    if fit_count != int(config["model"]["maximum_fits"]):
        raise ContractError("fit count differs from preregistration")
    shadow_blind = pd.concat(blocks, ignore_index=True).sort_values(
        list(HELPER.KEY_COLUMNS), kind="mergesort"
    ).reset_index(drop=True)
    protected = shadow_blind["lead_h"].isin((3, 6, 9, 12))
    if not np.array_equal(
        shadow_blind.loc[protected, "candidate_prediction"].to_numpy(),
        shadow_blind.loc[protected, "base_prediction"].to_numpy(),
    ):
        raise ContractError("protected leads changed")
    artifact_hashes["shadow_prediction_seal"] = HELPER._atomic_parquet(
        output / "shadow_predictions_sealed.parquet", shadow_blind
    )
    artifact_hashes["shadow_prediction_receipt"] = HELPER._atomic_json(
        output / "shadow_prediction_seal.json",
        {
            "rows": int(len(shadow_blind)),
            "cases": int(shadow_blind["anchor_id"].nunique()),
            "candidate_sha256": HELPER.array_sha256(shadow_blind["candidate_prediction"]),
            "truth_rows_read_before_seal": 0,
            "fit_count": fit_count,
        },
    )
    validation_ids = selected.sort_values("anchor_id")["anchor_id"].to_numpy(dtype=np.int64)
    truth = HELPER._shadow_target_matrix(paths.train_anchors, validation_ids)
    evaluated = HELPER._attach_matrix_truth(shadow_blind, validation_ids, truth)
    shadow_gate = HELPER._shadow_gate(evaluated, config)
    base_result.update(
        status="RESEARCH_GATE_PASS" if shadow_gate["passed"] else "NO_GO_SHADOW_GATE",
        local_auxiliary=local_auxiliary,
        shadow_gate=shadow_gate,
        fit_count=fit_count,
        runtime_seconds=time.perf_counter() - started,
        promotion="research candidate only" if shadow_gate["passed"] else "no-go",
    )
    return _write_result(output, base_result, artifact_hashes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data_dir = _resolve_data_dir(args.data_dir)
    result = execute_once(data_dir, args.root) if args.execute else check_only(data_dir, args.root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
