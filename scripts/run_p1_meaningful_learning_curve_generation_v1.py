"""Run the sealed P1 meaningful-learning-curve generation.

This runner is intentionally one-shot and append-only.  It refits the exact
frozen XGBoost incumbent at every chronological training prefix, evaluates
three preregistered structural LightGBM alternatives, and defers all target
fold scoring until every prediction part has been persisted.  Test values are
loaded only if a branch passes every sealed curve gate.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from ocean_goal.meaningful_score import evaluate_learning_curve, load_contract
from p1_qc.config import P1QCConfig, load_config
from p1_qc.data import KEY_COLUMNS, load_dataset
from p1_qc.features import FeatureBundle
from p1_qc.pipeline import TabularEncoder, _fit_model, apply_postprocess
from p1_qc.rules import detect_plateaus, detect_singleton_spikes
from p1_qc.submission import build_submission, validate_submission, write_submission
from p1_qc.validation import paired_block_bootstrap

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "p1_meaningful_learning_curve_generation_v1.json"
DEPENDENCIES = (
    "src/p1_qc/pipeline.py",
    "src/p1_qc/models_tabular.py",
    "src/p1_qc/postprocess.py",
    "src/p1_qc/rules.py",
    "src/p1_qc/validation.py",
    "src/ocean_goal/meaningful_score.py",
)
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
TYPE_ORDER = ("spike", "noise", "flatline", "offset", "drift")
STATIONS = ("G-ORS", "I-ORS", "S-ORS")


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _int64_array_hash(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _json_load(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object: {path}")
    return parsed


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"time_kst": _now_kst(), "event": event, **payload},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _fraction_tag(fraction: float) -> str:
    return f"p{int(round(100 * fraction)):03d}"


def _data_dir() -> Path:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR must be set to the immutable P1 source directory")
    directory = Path(raw).expanduser().resolve(strict=True)
    for name in ("train.csv", "test.csv", "sample_submission.csv"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return directory


def _resolve_pin(name: str, data_dir: Path) -> Path:
    if name in {"train.csv", "test.csv", "sample_submission.csv", "baseline_rule.csv"}:
        return data_dir / name
    path = (PROJECT_ROOT / name).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"unsafe pinned path: {name}")
    return path


def _verify_pins(config: Mapping[str, Any], data_dir: Path) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, expected in config["immutable_inputs"].items():
        path = _resolve_pin(str(name), data_dir)
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(f"immutable pin mismatch for {name}: {observed} != {expected}")
        report[str(name)] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
    return report


def _artifact_dir(config: Mapping[str, Any]) -> Path:
    path = (PROJECT_ROOT / str(config["artifact_dir"])).resolve()
    if not path.is_relative_to(PROJECT_ROOT / "artifacts"):
        raise RuntimeError("artifact_dir must remain under the workspace artifacts directory")
    return path


def seal(config_path: Path) -> Path:
    config_path = config_path.resolve(strict=True)
    config = _json_load(config_path)
    data_dir = _data_dir()
    artifact = _artifact_dir(config)
    artifact.mkdir(parents=True, exist_ok=True)
    seal_path = artifact / "preexecution_seal.json"
    if seal_path.exists():
        raise FileExistsError(f"seal already exists: {seal_path}")
    pins = _verify_pins(config, data_dir)
    script_path = Path(__file__).resolve(strict=True)
    dependencies = {
        name: _sha256((PROJECT_ROOT / name).resolve(strict=True)) for name in DEPENDENCIES
    }
    receipt = {
        "schema_version": "p1_meaningful_learning_curve_preexecution_seal.v1",
        "experiment_id": config["experiment_id"],
        "sealed_at_kst": _now_kst(),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_path": str(script_path),
        "implementation_sha256": _sha256(script_path),
        "dependency_sha256": dependencies,
        "immutable_inputs": pins,
        "hypotheses": config["hypotheses"],
        "selection_rule": config["selection_rule"],
        "prefix_fractions": config["prefix_fractions"],
        "seeds": config["seeds"],
        "bootstrap": config["bootstrap"],
        "pass_gates": config["pass_gates"],
        "fixed_fold_postprocess": config["fixed_fold_postprocess"],
        "deployment_postprocess": config["deployment_postprocess"],
        "operation_counters_at_seal": {
            "model_fits": 0,
            "target_fold_scores": 0,
            "test_value_reads": 0,
            "candidate_files": 0,
            "uploads": 0,
            "source_mutations": 0,
            "frozen_mutations": 0,
        },
    }
    _json_new(seal_path, receipt)
    _emit("preexecution_sealed", path=str(seal_path), sha256=_sha256(seal_path))
    return seal_path


def _verify_seal(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    artifact = _artifact_dir(config)
    receipt = _json_load(artifact / "preexecution_seal.json")
    if receipt["config_sha256"] != _sha256(config_path):
        raise RuntimeError("config changed after preregistration")
    if receipt["implementation_sha256"] != _sha256(Path(__file__).resolve()):
        raise RuntimeError("implementation changed after preregistration")
    for name, expected in receipt["dependency_sha256"].items():
        if _sha256(PROJECT_ROOT / name) != expected:
            raise RuntimeError(f"dependency changed after preregistration: {name}")
    return receipt


def _load_feature_bundle(
    frame: pd.DataFrame,
    cache_path: Path,
    metadata_path: Path,
) -> tuple[FeatureBundle, dict[str, Any]]:
    metadata = _json_load(metadata_path)
    features = pd.read_parquet(cache_path)
    if len(features) != len(frame) or int(metadata["rows"]) != len(frame):
        raise RuntimeError("feature cache row count mismatch")
    if metadata["source_sha256"] != frame.attrs["source_sha256"]:
        raise RuntimeError("feature cache source binding mismatch")
    if _sha256(cache_path) != metadata["parquet_sha256"]:
        raise RuntimeError("feature cache parquet SHA mismatch")
    forbidden = {"label", "anomaly_type"}
    if forbidden.intersection(features.columns):
        raise RuntimeError("feature cache contains target columns")
    feature_columns = tuple(str(value) for value in metadata["feature_columns"])
    categorical = tuple(str(value) for value in metadata["categorical_columns"])
    if tuple(features.columns) != feature_columns:
        raise RuntimeError("feature cache schema/order differs from metadata")
    features.index = frame.index.copy()
    return FeatureBundle(features, feature_columns, categorical), metadata


def _key_positions(frame: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    source = pd.MultiIndex.from_frame(frame.loc[:, list(KEY_COLUMNS)])
    requested = pd.MultiIndex.from_frame(keys.loc[:, list(KEY_COLUMNS)])
    if source.has_duplicates or requested.has_duplicates:
        raise RuntimeError("duplicate P1 keys")
    positions = source.get_indexer(requested)
    if (positions < 0).any():
        raise RuntimeError("frozen OOF key not found in train")
    return positions.astype(np.int64, copy=False)


def _fold_runtime(
    train: pd.DataFrame,
    p1_config: P1QCConfig,
    frozen_oof_keys: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = pd.to_datetime(train["time"], errors="raise", utc=True, format="mixed")
    values = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    folds: list[dict[str, Any]] = []
    scope: dict[str, Any] = {}
    for ordinal, spec in enumerate(p1_config.splits.folds):
        if spec.name not in FOLD_ORDER:
            raise RuntimeError(f"unexpected fold: {spec.name}")
        train_end = pd.Timestamp(spec.train_end).tz_convert("UTC")
        val_start = pd.Timestamp(spec.val_start).tz_convert("UTC")
        val_end = pd.Timestamp(spec.val_end).tz_convert("UTC")
        train_idx = np.flatnonzero(parsed.le(train_end).to_numpy())
        frozen_part = frozen_oof_keys.loc[frozen_oof_keys["fold"].eq(spec.name)].copy()
        val_idx = _key_positions(train, frozen_part)
        if np.intersect1d(train_idx, val_idx).size:
            raise RuntimeError(f"train/validation overlap: {spec.name}")
        val_time = parsed.iloc[val_idx]
        before = int(val_time.lt(val_start).sum())
        after = int(val_time.ge(val_end).sum())
        scope[spec.name] = {
            "nominal_val_start_utc": val_start.isoformat(),
            "nominal_val_end_utc": val_end.isoformat(),
            "validation_rows": len(val_idx),
            "rows_before_nominal_start": before,
            "rows_at_or_after_nominal_end": after,
            "nominal_wall_clock_scope_exact": before == 0 and after == 0,
            "minimum_validation_time_utc": val_time.min().isoformat(),
            "maximum_validation_time_utc": val_time.max().isoformat(),
            "event_protected_keys_from_frozen_oof": True,
        }
        folds.append(
            {
                "name": spec.name,
                "ordinal": ordinal,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_end_ns": train_end.value,
                "val_start_ns": val_start.value,
                "val_end_ns": val_end.value,
                "time_ns": values,
                "frozen_keys": frozen_part.reset_index(drop=True),
            }
        )
    if tuple(item["name"] for item in folds) != FOLD_ORDER:
        raise RuntimeError("outer fold order changed")
    return folds, scope


def _safe_prefix(
    train: pd.DataFrame,
    fold_train_idx: np.ndarray,
    time_ns: np.ndarray,
    fraction: float,
    *,
    cadence_minutes: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    eligible_times = np.unique(time_ns[fold_train_idx])
    if not len(eligible_times):
        raise RuntimeError("empty eligible training prefix")
    if math.isclose(fraction, 1.0):
        nominal = int(eligible_times[-1])
        adjusted = nominal
        iterations = 0
        labels_examined_through = nominal
    else:
        ordinal = max(0, math.ceil(fraction * len(eligible_times)) - 1)
        nominal = int(eligible_times[ordinal])
        adjusted = nominal
        iterations = 0
        cadence_ns = int(pd.Timedelta(minutes=cadence_minutes).value)
        # Only labels at or before the nominal prefix cutoff are inspected.
        preliminary = fold_train_idx[time_ns[fold_train_idx] <= nominal]
        work = train.iloc[preliminary][["station", "layer", "time", "label"]].copy()
        work["__position"] = preliminary
        work["__time_ns"] = time_ns[preliminary]
        work["__label"] = pd.to_numeric(work["label"], errors="raise").to_numpy(dtype=np.int8)
        work.sort_values(["station", "layer", "__time_ns", "__position"], inplace=True)
        while True:
            included = work.loc[work["__time_ns"].le(adjusted)]
            retreat_candidates: list[int] = []
            for _, group in included.groupby(["station", "layer"], sort=False, observed=True):
                if group.empty or int(group["__label"].iloc[-1]) != 1:
                    continue
                times = group["__time_ns"].to_numpy(dtype=np.int64)
                labels = group["__label"].to_numpy(dtype=np.int8)
                if adjusted - int(times[-1]) > cadence_ns:
                    # A physical observation gap terminates the event segment.
                    continue
                start = len(group) - 1
                while (
                    start > 0
                    and labels[start - 1] == 1
                    and times[start] - times[start - 1] == cadence_ns
                ):
                    start -= 1
                retreat_candidates.append(int(times[start]) - 1)
            if not retreat_candidates:
                break
            new_adjusted = min([adjusted, *retreat_candidates])
            if new_adjusted >= adjusted:
                raise RuntimeError("event-safe cutoff did not retreat")
            adjusted = new_adjusted
            iterations += 1
            if iterations > 1000:
                raise RuntimeError("event-safe cutoff failed to converge")
        labels_examined_through = nominal
    result = fold_train_idx[time_ns[fold_train_idx] <= adjusted]
    if not len(result):
        raise RuntimeError("event-safe prefix became empty")
    audit = {
        "fraction": fraction,
        "eligible_rows": len(fold_train_idx),
        "eligible_unique_timestamps": len(eligible_times),
        "prefix_rows": len(result),
        "nominal_cutoff_utc": pd.Timestamp(nominal, tz="UTC").isoformat(),
        "adjusted_cutoff_utc": pd.Timestamp(adjusted, tz="UTC").isoformat(),
        "labels_examined_no_later_than_utc": pd.Timestamp(
            labels_examined_through, tz="UTC"
        ).isoformat(),
        "event_boundary_retreat_iterations": iterations,
        "maximum_retained_target_time_utc": pd.Timestamp(
            time_ns[result].max(), tz="UTC"
        ).isoformat(),
        "target_scope_not_after_nominal_cutoff": bool(time_ns[result].max() <= nominal),
        "prefix_subset_of_fold_train": bool(np.isin(result, fold_train_idx).all()),
    }
    return result, audit


def _binary_f1(truth: Sequence[int], prediction: Sequence[int]) -> float:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0


def _binary_weight(target: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)
    positive = max(1, int(y.sum()))
    negative = max(1, len(y) - positive)
    return np.where(y == 1, math.sqrt(negative / positive), 1.0).astype(np.float32)


def _event_day_weight(metadata: pd.DataFrame, target: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)
    work = metadata.loc[:, ["station", "layer", "time"]].reset_index(drop=True).copy()
    work["__position"] = np.arange(len(work), dtype=np.int64)
    work["__target"] = y
    work["__time"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "__time", "__position"], inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["__time"].diff().dt.total_seconds().eq(600)
    prior = grouped["__target"].shift(1).fillna(0).eq(1)
    starts = work["__target"].eq(1) & (~contiguous | ~prior)
    work["__event"] = starts.cumsum().where(work["__target"].eq(1), -1).astype(np.int64)
    positive = work["__target"].eq(1)
    event_length = work.loc[positive].groupby("__event", sort=False)["__event"].transform("size")
    pos_raw = 1.0 / np.sqrt(event_length.to_numpy(dtype=float))
    pos_raw /= pos_raw.mean()
    day = work["__time"].dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    normal = ~positive
    normal_length = (
        work.loc[normal]
        .assign(__day=day.loc[normal])
        .groupby(["station", "layer", "__day"], sort=False, observed=True)["__day"]
        .transform("size")
    )
    normal_raw = 1.0 / np.sqrt(normal_length.to_numpy(dtype=float))
    normal_raw /= normal_raw.mean()
    ordered_weight = np.empty(len(work), dtype=np.float64)
    ordered_weight[positive.to_numpy()] = pos_raw * math.sqrt(
        max(1, normal.sum()) / max(1, positive.sum())
    )
    ordered_weight[normal.to_numpy()] = normal_raw
    work["__weight"] = ordered_weight
    restored = work.sort_values("__position", kind="mergesort")
    result = restored["__weight"].to_numpy(dtype=np.float32)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise RuntimeError("invalid event/day training weight")
    return result


def _typed_target(anomaly_type: pd.Series, binary_target: np.ndarray) -> np.ndarray:
    result = np.zeros(len(binary_target), dtype=np.int8)
    values = anomaly_type.astype("string").fillna("").to_numpy(dtype=str)
    for position in np.flatnonzero(np.asarray(binary_target, dtype=np.int8) == 1):
        tokens = {token.strip() for token in values[position].split("+") if token.strip()}
        selected = next((name for name in TYPE_ORDER if name in tokens), None)
        if selected is None:
            raise RuntimeError("positive row lacks a recognized anomaly type")
        result[position] = TYPE_ORDER.index(selected) + 1
    return result


def _multiclass_weight(target: np.ndarray) -> np.ndarray:
    values, counts = np.unique(target, return_counts=True)
    raw = {
        int(value): math.sqrt(len(target) / int(count))
        for value, count in zip(values, counts, strict=True)
    }
    weight = np.asarray([raw[int(value)] for value in target], dtype=np.float64)
    weight /= weight.mean()
    return weight.astype(np.float32)


def _lgb_parameters(config: Mapping[str, Any], seed: int, *, multiclass: bool) -> dict[str, Any]:
    parameters = dict(config["lightgbm_parameters"])
    parameters.update(
        {
            "objective": "multiclass" if multiclass else "binary",
            "random_state": seed,
            "n_jobs": 8,
            "verbosity": -1,
            "deterministic": True,
            "force_row_wise": True,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "extra_seed": seed,
        }
    )
    if multiclass:
        parameters["num_class"] = 6
    return parameters


def _fit_candidate(
    branch: str,
    config: Mapping[str, Any],
    seeds: Sequence[int],
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_metadata: pd.DataFrame,
    train_anomaly_type: pd.Series,
    prediction_features: np.ndarray,
    prediction_metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, Any]], int, list[np.ndarray]]:
    seed_predictions: list[np.ndarray] = []
    packages: list[dict[str, Any]] = []
    fit_count = 0
    if branch == "event_day_balanced_binary_lgbm":
        weight = _event_day_weight(train_metadata, train_target)
        for seed in seeds:
            model = lgb.LGBMClassifier(**_lgb_parameters(config, int(seed), multiclass=False))
            model.fit(train_features, train_target, sample_weight=weight)
            seed_predictions.append(model.predict_proba(prediction_features)[:, 1])
            packages.append({"seed": int(seed), "global": model})
            fit_count += 1
    elif branch == "typed_multiclass_union_lgbm":
        typed = _typed_target(train_anomaly_type, train_target)
        weight = _multiclass_weight(typed)
        for seed in seeds:
            model = lgb.LGBMClassifier(**_lgb_parameters(config, int(seed), multiclass=True))
            model.fit(train_features, typed, sample_weight=weight)
            probability = model.predict_proba(prediction_features)
            normal_columns = np.flatnonzero(np.asarray(model.classes_) == 0)
            if len(normal_columns) != 1:
                raise RuntimeError("typed model has no unique normal class")
            seed_predictions.append(1.0 - probability[:, int(normal_columns[0])])
            packages.append({"seed": int(seed), "multiclass": model})
            fit_count += 1
    elif branch == "station_routed_binary_experts_lgbm":
        stations_train = train_metadata["station"].astype(str).to_numpy()
        stations_prediction = prediction_metadata["station"].astype(str).to_numpy()
        global_weight = _binary_weight(train_target)
        for seed in seeds:
            global_model = lgb.LGBMClassifier(
                **_lgb_parameters(config, int(seed), multiclass=False)
            )
            global_model.fit(train_features, train_target, sample_weight=global_weight)
            routed = global_model.predict_proba(prediction_features)[:, 1]
            station_models: dict[str, Any] = {}
            fit_count += 1
            for station in STATIONS:
                fit_mask = stations_train == station
                predict_mask = stations_prediction == station
                station_target = train_target[fit_mask]
                if (
                    int(fit_mask.sum()) < 5000
                    or len(np.unique(station_target)) != 2
                    or not predict_mask.any()
                ):
                    continue
                model = lgb.LGBMClassifier(**_lgb_parameters(config, int(seed), multiclass=False))
                model.fit(
                    train_features[fit_mask],
                    station_target,
                    sample_weight=_binary_weight(station_target),
                )
                routed[predict_mask] = model.predict_proba(prediction_features[predict_mask])[:, 1]
                station_models[station] = model
                fit_count += 1
            seed_predictions.append(routed)
            packages.append(
                {"seed": int(seed), "global": global_model, "station_models": station_models}
            )
    else:
        raise KeyError(branch)
    average = np.mean(np.vstack(seed_predictions), axis=0)
    if not np.isfinite(average).all() or ((average < 0) | (average > 1)).any():
        raise RuntimeError(f"invalid candidate probabilities: {branch}")
    return average.astype(np.float64), packages, fit_count, seed_predictions


def _predict_loaded_candidate(
    branch: str,
    packages: Sequence[Mapping[str, Any]],
    features: np.ndarray,
    metadata: pd.DataFrame,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    stations = metadata["station"].astype(str).to_numpy()
    for package in packages:
        if branch == "event_day_balanced_binary_lgbm":
            predictions.append(package["global"].predict_proba(features)[:, 1])
        elif branch == "typed_multiclass_union_lgbm":
            model = package["multiclass"]
            probability = model.predict_proba(features)
            normal = int(np.flatnonzero(np.asarray(model.classes_) == 0)[0])
            predictions.append(1.0 - probability[:, normal])
        elif branch == "station_routed_binary_experts_lgbm":
            routed = package["global"].predict_proba(features)[:, 1]
            for station, model in package["station_models"].items():
                mask = stations == station
                if mask.any():
                    routed[mask] = model.predict_proba(features[mask])[:, 1]
            predictions.append(routed)
        else:
            raise KeyError(branch)
    return np.mean(np.vstack(predictions), axis=0)


def _part_paths(artifact: Path, fold_name: str, fraction: float) -> tuple[Path, Path]:
    stem = f"{fold_name}_{_fraction_tag(fraction)}"
    return (
        artifact / "prediction_parts" / f"{stem}.parquet",
        artifact / "prediction_parts" / f"{stem}.json",
    )


def _write_parquet_new(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise RuntimeError(f"stale partial artifact requires manual audit: {partial}")
    frame.to_parquet(partial, index=False, compression="zstd")
    os.replace(partial, path)


def _build_prediction_parts(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    p1_config: P1QCConfig,
    config: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    artifact: Path,
) -> dict[str, Any]:
    branches = [str(item["id"]) for item in config["hypotheses"]]
    fractions = [float(value) for value in config["prefix_fractions"]]
    seeds = [int(value) for value in config["seeds"]]
    expected_parts = len(folds) * len(fractions)
    completed = 0
    total_fits = 0
    prefix_audits: dict[str, dict[str, Any]] = {}
    exact_parameters = dict(p1_config.raw["models"]["xgboost"])
    for fold in folds:
        prior_prefix: np.ndarray | None = None
        for fraction in fractions:
            part_path, audit_path = _part_paths(artifact, str(fold["name"]), fraction)
            if part_path.exists() or audit_path.exists():
                if not (part_path.exists() and audit_path.exists()):
                    raise RuntimeError(f"incomplete append-only prediction part: {part_path}")
                audit = _json_load(audit_path)
                if audit["parquet_sha256"] != _sha256(part_path):
                    raise RuntimeError(f"prediction part SHA mismatch: {part_path}")
                completed += 1
                total_fits += int(audit["model_fit_count"])
                reused_prefix, _ = _safe_prefix(
                    train,
                    np.asarray(fold["train_idx"], dtype=np.int64),
                    np.asarray(fold["time_ns"], dtype=np.int64),
                    fraction,
                    cadence_minutes=p1_config.data.cadence_minutes,
                )
                if _int64_array_hash(reused_prefix) != audit["prefix_positions_sha256"]:
                    raise RuntimeError("recomputed prefix differs from saved prediction part")
                prior_prefix = reused_prefix
                _emit(
                    "prediction_part_reused",
                    fold=fold["name"],
                    fraction=fraction,
                    completed=completed,
                    total=expected_parts,
                )
                continue
            prefix_idx, prefix_audit = _safe_prefix(
                train,
                np.asarray(fold["train_idx"], dtype=np.int64),
                np.asarray(fold["time_ns"], dtype=np.int64),
                fraction,
                cadence_minutes=p1_config.data.cadence_minutes,
            )
            if prior_prefix is not None and not np.isin(prior_prefix, prefix_idx).all():
                raise RuntimeError("chronological prefixes are not nested")
            prior_prefix = prefix_idx
            if np.intersect1d(prefix_idx, np.asarray(fold["val_idx"], dtype=np.int64)).size:
                raise RuntimeError("prefix overlaps target fold validation")
            encoder = TabularEncoder().fit(bundle, prefix_idx)
            train_features = encoder.transform(bundle, prefix_idx)
            validation_features = encoder.transform(
                bundle, np.asarray(fold["val_idx"], dtype=np.int64)
            )
            train_target = pd.to_numeric(train.iloc[prefix_idx]["label"], errors="raise").to_numpy(
                dtype=np.int8
            )
            if not np.isin(train_target, [0, 1]).all() or len(np.unique(train_target)) != 2:
                raise RuntimeError("training prefix must contain both binary classes")
            training_metadata = train.iloc[prefix_idx][["station", "layer", "time"]].reset_index(
                drop=True
            )
            training_types = train.iloc[prefix_idx]["anomaly_type"].reset_index(drop=True)
            validation_frame = train.iloc[np.asarray(fold["val_idx"], dtype=np.int64)][
                ["station", "year", "layer", "time", "temp", "psal", "depth"]
            ].copy()
            plateau = detect_plateaus(validation_frame).to_numpy(dtype=bool)
            spike = detect_singleton_spikes(validation_frame).to_numpy(dtype=bool)
            postprocess = config["fixed_fold_postprocess"][str(fold["name"])]
            baseline_seed_probabilities: list[np.ndarray] = []
            baseline_seed_predictions: list[np.ndarray] = []
            for registered_seed in seeds:
                baseline_model = _fit_model(
                    "xgboost",
                    exact_parameters,
                    int(registered_seed) + int(fold["ordinal"]),
                    int(p1_config.raw["project"]["threads"]),
                    train_features,
                    train_target,
                )
                seed_probability = baseline_model.predict_proba(validation_features)[:, 1]
                seed_prediction = apply_postprocess(
                    validation_frame,
                    seed_probability,
                    plateau,
                    spike,
                    postprocess,
                )
                baseline_seed_probabilities.append(seed_probability)
                baseline_seed_predictions.append(seed_prediction)
                del baseline_model
            baseline_probability = np.mean(np.vstack(baseline_seed_probabilities), axis=0)
            baseline_prediction = apply_postprocess(
                validation_frame,
                baseline_probability,
                plateau,
                spike,
                postprocess,
            )
            fit_count = len(seeds)
            part = validation_frame.loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
            part["row_position"] = np.asarray(fold["val_idx"], dtype=np.int64)
            part["fold"] = str(fold["name"])
            part["fraction"] = fraction
            part["baseline_probability"] = baseline_probability.astype(np.float32)
            part["baseline_prediction"] = baseline_prediction.astype(np.int8)
            for seed_index, registered_seed in enumerate(seeds):
                part[f"baseline__seed_{registered_seed}__probability"] = (
                    baseline_seed_probabilities[seed_index].astype(np.float32)
                )
                part[f"baseline__seed_{registered_seed}__prediction"] = baseline_seed_predictions[
                    seed_index
                ].astype(np.int8)
            part["plateau"] = plateau
            part["spike_candidate"] = spike
            branch_fit_counts: dict[str, int] = {}
            for branch in branches:
                probability, packages, branch_fits, seed_probabilities = _fit_candidate(
                    branch,
                    config,
                    [seed + int(fold["ordinal"]) for seed in seeds],
                    train_features,
                    train_target,
                    training_metadata,
                    training_types,
                    validation_features,
                    validation_frame[["station", "layer", "time"]].reset_index(drop=True),
                )
                prediction = apply_postprocess(
                    validation_frame,
                    probability,
                    plateau,
                    spike,
                    postprocess,
                )
                part[f"{branch}__probability"] = probability.astype(np.float32)
                part[f"{branch}__prediction"] = prediction.astype(np.int8)
                for seed_index, registered_seed in enumerate(seeds):
                    seed_probability = seed_probabilities[seed_index]
                    seed_prediction = apply_postprocess(
                        validation_frame,
                        seed_probability,
                        plateau,
                        spike,
                        postprocess,
                    )
                    part[f"{branch}__seed_{registered_seed}__probability"] = (
                        seed_probability.astype(np.float32)
                    )
                    part[f"{branch}__seed_{registered_seed}__prediction"] = seed_prediction.astype(
                        np.int8
                    )
                branch_fit_counts[branch] = branch_fits
                fit_count += branch_fits
                del packages, probability, prediction, seed_probabilities
                gc.collect()
            expected_keys = fold["frozen_keys"].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
            if not part.loc[:, list(KEY_COLUMNS)].equals(expected_keys):
                raise RuntimeError("prediction part key/order differs from frozen OOF")
            _write_parquet_new(part, part_path)
            prefix_audit.update(
                {
                    "schema_version": "p1_learning_curve_prediction_part.v1",
                    "experiment_id": config["experiment_id"],
                    "fold": fold["name"],
                    "fraction": fraction,
                    "prefix_positions_sha256": _int64_array_hash(prefix_idx),
                    "validation_rows": len(part),
                    "validation_key_order_matches_frozen_oof": True,
                    "target_fold_validation_labels_read_before_prediction": 0,
                    "model_fit_count": fit_count,
                    "branch_model_fit_counts": branch_fit_counts,
                    "baseline_structure": config["comparator"],
                    "fixed_postprocess": postprocess,
                    "parquet_path": str(part_path),
                    "parquet_sha256": _sha256(part_path),
                    "completed_at_kst": _now_kst(),
                }
            )
            _json_new(audit_path, prefix_audit)
            prefix_audits[f"{fold['name']}:{fraction}"] = prefix_audit
            completed += 1
            total_fits += fit_count
            _emit(
                "prediction_part_completed",
                fold=fold["name"],
                fraction=fraction,
                prefix_rows=len(prefix_idx),
                model_fits=fit_count,
                completed=completed,
                total=expected_parts,
            )
            del encoder, train_features, validation_features, train_target, part
            gc.collect()
    complete_path = artifact / "predictions_complete.json"
    part_records = []
    for fold in folds:
        for fraction in fractions:
            part_path, audit_path = _part_paths(artifact, str(fold["name"]), fraction)
            if not part_path.is_file() or not audit_path.is_file():
                raise RuntimeError("prediction matrix is incomplete")
            part_records.append(
                {
                    "fold": fold["name"],
                    "fraction": fraction,
                    "parquet": str(part_path.relative_to(PROJECT_ROOT)),
                    "parquet_sha256": _sha256(part_path),
                    "audit_sha256": _sha256(audit_path),
                }
            )
    payload = {
        "schema_version": "p1_learning_curve_predictions_complete.v1",
        "experiment_id": config["experiment_id"],
        "completed_at_kst": _now_kst(),
        "parts": part_records,
        "part_count": len(part_records),
        "model_fit_count": total_fits,
        "target_fold_validation_label_reads_before_its_prediction": 0,
        "aggregate_scores_computed_before_completion": 0,
        "branch_selection_before_completion": 0,
        "test_value_reads": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    if complete_path.exists():
        existing = _json_load(complete_path)
        if existing["parts"] != payload["parts"]:
            raise RuntimeError("predictions_complete receipt differs on resume")
        payload = existing
    else:
        _json_new(complete_path, payload)
    _emit(
        "all_predictions_complete_scoring_unlocked", parts=len(part_records), model_fits=total_fits
    )
    return payload


def _metric_slices(
    frame: pd.DataFrame,
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    columns: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, part in frame.assign(__row=np.arange(len(frame))).groupby(
        list(columns), sort=True, observed=True
    ):
        positions = part["__row"].to_numpy(dtype=np.int64)
        names = key if isinstance(key, tuple) else (key,)
        label = "|".join(str(value) for value in names)
        base = _binary_f1(truth[positions], baseline[positions])
        challenger = _binary_f1(truth[positions], candidate[positions])
        result[label] = {
            "rows": len(positions),
            "positive_rows": int(truth[positions].sum()),
            "incumbent_f1": base,
            "challenger_f1": challenger,
            "delta_candidate_minus_incumbent": challenger - base,
        }
    return result


def _score_all(
    train: pd.DataFrame,
    frozen_oof: pd.DataFrame,
    current_oof: pd.DataFrame,
    config: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    artifact: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not (artifact / "predictions_complete.json").is_file():
        raise RuntimeError("prediction completion receipt is required before scoring")
    branches = [str(item["id"]) for item in config["hypotheses"]]
    order = {str(item["id"]): int(item["order"]) for item in config["hypotheses"]}
    fractions = [float(value) for value in config["prefix_fractions"]]
    seeds = [int(value) for value in config["seeds"]]
    branch_reports: dict[str, Any] = {}
    full_baseline_exact = True
    all_key_order_exact = True
    for branch in branches:
        points: list[dict[str, Any]] = []
        full_fold_deltas: list[float] = []
        full_station_deltas: dict[str, float] = {}
        full_station_layer: dict[str, Any] = {}
        full_current_context: dict[str, Any] | None = None
        for fraction_index, fraction in enumerate(fractions):
            parts = []
            for fold in folds:
                part_path, _ = _part_paths(artifact, str(fold["name"]), fraction)
                parts.append(pd.read_parquet(part_path))
            combined = pd.concat(parts, ignore_index=True)
            row_positions = combined["row_position"].to_numpy(dtype=np.int64)
            truth = pd.to_numeric(train.iloc[row_positions]["label"], errors="raise").to_numpy(
                dtype=np.int8
            )
            baseline = combined["baseline_prediction"].to_numpy(dtype=np.int8)
            candidate = combined[f"{branch}__prediction"].to_numpy(dtype=np.int8)
            expected_keys = frozen_oof.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
            observed_keys = combined.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
            keys_exact = observed_keys.equals(expected_keys)
            all_key_order_exact &= keys_exact
            if not keys_exact:
                raise RuntimeError("aggregate OOF key/order differs from frozen OOF")
            incumbent_f1 = _binary_f1(truth, baseline)
            challenger_f1 = _binary_f1(truth, candidate)
            incumbent_seed_metrics = [
                _binary_f1(
                    truth,
                    combined[f"baseline__seed_{seed}__prediction"].to_numpy(dtype=np.int8),
                )
                for seed in seeds
            ]
            challenger_seed_metrics = [
                _binary_f1(
                    truth,
                    combined[f"{branch}__seed_{seed}__prediction"].to_numpy(dtype=np.int8),
                )
                for seed in seeds
            ]
            bootstrap = paired_block_bootstrap(
                truth,
                candidate,
                baseline,
                combined.loc[:, ["station", "layer", "time"]],
                replicates=int(config["bootstrap"]["replicates"]),
                seed=int(config["bootstrap"]["seed"]) + order[branch] * 100 + fraction_index,
                cadence_minutes=10,
                normal_day_timezone="Asia/Seoul",
            )
            folds_metric = _metric_slices(combined, truth, candidate, baseline, ["fold"])
            stations_metric = _metric_slices(combined, truth, candidate, baseline, ["station"])
            station_layer_metric = _metric_slices(
                combined, truth, candidate, baseline, ["station", "layer"]
            )
            point = {
                "fraction": fraction,
                "rows": len(combined),
                "incumbent": incumbent_f1,
                "challenger": challenger_f1,
                "delta_candidate_minus_incumbent": challenger_f1 - incumbent_f1,
                "delta_ci90": bootstrap["difference_ci90"],
                "incumbent_seed_metrics": incumbent_seed_metrics,
                "challenger_seed_metrics": challenger_seed_metrics,
                "paired_cluster_bootstrap": bootstrap,
                "folds": folds_metric,
                "stations": stations_metric,
                "station_layers": station_layer_metric,
                "key_order_exact": keys_exact,
            }
            points.append(point)
            oof = combined.loc[:, [*KEY_COLUMNS, "fold", "row_position"]].copy()
            oof["label"] = truth
            oof["incumbent_probability"] = combined["baseline_probability"].to_numpy(np.float32)
            oof["incumbent_prediction"] = baseline
            oof["challenger_probability"] = combined[f"{branch}__probability"].to_numpy(np.float32)
            oof["challenger_prediction"] = candidate
            oof_path = artifact / "oof" / f"{branch}_{_fraction_tag(fraction)}.parquet"
            if not oof_path.exists():
                _write_parquet_new(oof, oof_path)
            if math.isclose(fraction, 1.0):
                frozen_prediction = frozen_oof["prediction"].to_numpy(dtype=np.int8)
                reference_seed_prediction = combined[
                    f"baseline__seed_{seeds[0]}__prediction"
                ].to_numpy(dtype=np.int8)
                exact = np.array_equal(reference_seed_prediction, frozen_prediction)
                full_baseline_exact &= exact
                full_fold_deltas = [
                    float(folds_metric[name]["delta_candidate_minus_incumbent"])
                    for name in FOLD_ORDER
                ]
                full_station_deltas = {
                    station: float(stations_metric[station]["delta_candidate_minus_incumbent"])
                    for station in STATIONS
                }
                full_station_layer = station_layer_metric
                current = current_oof.loc[:, [*KEY_COLUMNS, "fold", "candidate_prediction"]]
                if (
                    not current.loc[:, [*KEY_COLUMNS, "fold"]]
                    .reset_index(drop=True)
                    .equals(expected_keys)
                ):
                    raise RuntimeError("current-cycle OOF key/order mismatch")
                current_prediction = current["candidate_prediction"].to_numpy(dtype=np.int8)
                current_f1 = _binary_f1(truth, current_prediction)
                full_current_context = {
                    "historical_research_only_f1": current_f1,
                    "challenger_minus_historical_research_only": challenger_f1 - current_f1,
                    "not_a_gate_comparator": True,
                }
        leakage_checks = {
            "target_fold_labels_not_used_before_its_prediction": True,
            "aggregate_scoring_deferred_until_all_prediction_parts_complete": True,
            "prefix_target_scope_never_after_registered_cutoff": True,
            "prefix_rows_are_subsets_of_fold_training_rows": True,
            "fold_train_validation_positions_disjoint": True,
            "feature_cache_excludes_label_and_anomaly_type": True,
            "test_values_not_read_during_curve": True,
            "fixed_postprocess_not_retuned": True,
        }
        reproducibility_checks = {
            "exact_three_registered_seeds": len(config["seeds"]) == 3,
            "exact_registered_prefixes": config["prefix_fractions"] == [0.4, 0.55, 0.7, 0.85, 1.0],
            "oof_key_and_order_exact": all_key_order_exact,
            "full_fraction_incumbent_reference_seed_refit_matches_frozen_oof": full_baseline_exact,
            "fixed_incumbent_structure_refit_each_prefix": True,
            "sealed_config_and_implementation_verified": True,
            "paired_bootstrap_replicates_exact": int(config["bootstrap"]["replicates"]) == 5000,
        }
        late = {
            point["fraction"]: point for point in points if point["fraction"] in {0.7, 0.85, 1.0}
        }
        full = late[1.0]
        gate_checks = {
            "late_fractions_all_improve": all(
                point["delta_candidate_minus_incumbent"] > 0 for point in late.values()
            ),
            "full_fraction_ci90_excludes_zero": float(full["delta_ci90"][0]) > 0,
            "another_late_fraction_ci90_excludes_zero": sum(
                float(late[value]["delta_ci90"][0]) > 0 for value in (0.7, 0.85)
            )
            >= 1,
            "full_effect_at_least_0_020_f1": float(full["delta_candidate_minus_incumbent"]) >= 0.02,
            "minimum_two_of_three_folds_improve": sum(value > 0 for value in full_fold_deltas) >= 2,
            "worst_station_regression_within_0_005": min(full_station_deltas.values()) >= -0.005,
            "all_leakage_checks": all(leakage_checks.values()),
            "all_reproducibility_checks": all(reproducibility_checks.values()),
        }
        report = {
            "schema_version": "p1_learning_curve_branch_metrics.v1",
            "experiment_id": config["experiment_id"],
            "branch": branch,
            "registered_order": order[branch],
            "points": points,
            "full_fraction_fold_deltas_candidate_minus_incumbent": full_fold_deltas,
            "full_fraction_station_deltas_candidate_minus_incumbent": full_station_deltas,
            "full_fraction_station_layer_metrics": full_station_layer,
            "historical_current_cycle_context": full_current_context,
            "leakage_checks": leakage_checks,
            "reproducibility_checks": reproducibility_checks,
            "gate_checks": gate_checks,
            "passed": all(gate_checks.values()),
            "decision": "PASS" if all(gate_checks.values()) else "RESEARCH_ONLY",
        }
        metrics_path = artifact / "metrics" / f"{branch}.json"
        if not metrics_path.exists():
            _json_new(metrics_path, report)
        branch_reports[branch] = report
        _emit(
            "branch_scored",
            branch=branch,
            full_delta=full["delta_candidate_minus_incumbent"],
            full_ci90=full["delta_ci90"],
            passed=report["passed"],
        )
    passing = [name for name, value in branch_reports.items() if value["passed"]]
    candidate_pool = passing if passing else branches
    selected = sorted(
        candidate_pool,
        key=lambda name: (
            -float(branch_reports[name]["points"][-1]["challenger"]),
            order[name],
        ),
    )[0]
    selected_report = branch_reports[selected]
    seal_receipt = _json_load(artifact / "preexecution_seal.json")
    evidence = {
        "problem": "P1",
        "selected_hypothesis": selected,
        "selection_status": "QUALIFIED_WINNER" if passing else "RESEARCH_ONLY_DIAGNOSTIC",
        "preregistration": {
            "generation_id": config["experiment_id"],
            "config_path": "configs/p1_meaningful_learning_curve_generation_v1.json",
            "config_sha256": seal_receipt["config_sha256"],
            "created_before_first_fit": True,
            "hypothesis_count": len(config["hypotheses"]),
            "score_derived_tuning": False,
        },
        "curve_protocol": {
            "prefix_fractions": fractions,
            "seed_ids": seeds,
            "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
            "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
            "bootstrap_cluster": "event_or_normal_day_by_station_layer",
            "incumbent_fresh_refit_each_prefix": True,
            "challenger_fresh_refit_each_prefix": True,
            "same_fold_keys_metric_postprocess": True,
            "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": bool(
                selected_report["reproducibility_checks"][
                    "full_fraction_incumbent_reference_seed_refit_matches_frozen_oof"
                ]
            ),
            "frozen_reproduction_reference_seed": seeds[0],
        },
        "points": [
            {
                "fraction": point["fraction"],
                "incumbent": point["incumbent"],
                "challenger": point["challenger"],
                "delta_ci90": point["delta_ci90"],
                "incumbent_seed_metrics": point["incumbent_seed_metrics"],
                "challenger_seed_metrics": point["challenger_seed_metrics"],
            }
            for point in selected_report["points"]
        ],
        "fold_deltas_candidate_minus_incumbent": selected_report[
            "full_fraction_fold_deltas_candidate_minus_incumbent"
        ],
        "slice_deltas_candidate_minus_incumbent": selected_report[
            "full_fraction_station_deltas_candidate_minus_incumbent"
        ],
        "leakage_checks": selected_report["leakage_checks"],
        "reproducibility_checks": selected_report["reproducibility_checks"],
        "operation_counters": {
            "uploads": 0,
            "source_mutations": 0,
            "frozen_mutations": 0,
        },
        "validation_scope_caveat": _json_load(artifact / "fold_scope_audit.json")[
            "strict_clock_caveat"
        ],
    }
    evidence_path = artifact / "learning_curve_evidence.json"
    if not evidence_path.exists():
        _json_new(evidence_path, evidence)
    contract = load_contract(PROJECT_ROOT, str(config["goal_contract"]))
    canonical_decision = evaluate_learning_curve(contract, evidence)
    decision_path = artifact / "canonical_curve_decision.json"
    if not decision_path.exists():
        _json_new(decision_path, canonical_decision)
    if bool(canonical_decision["passed"]) != bool(selected_report["passed"]):
        raise RuntimeError("local and canonical curve gates disagree")
    return branch_reports, evidence, canonical_decision


def _full_fit_and_reproduce(
    train: pd.DataFrame,
    train_bundle: FeatureBundle,
    data_dir: Path,
    config: Mapping[str, Any],
    selected: str,
    artifact: Path,
) -> dict[str, Any]:
    test = load_dataset(data_dir / "test.csv", kind="test", audit=False)
    test_cache = PROJECT_ROOT / "artifacts/cache/test_offline_c2a3877bdecea937.parquet"
    test_metadata_path = PROJECT_ROOT / "artifacts/cache/test_offline_c2a3877bdecea937.json"
    test_bundle, test_cache_metadata = _load_feature_bundle(test, test_cache, test_metadata_path)
    full_idx = np.arange(len(train), dtype=np.int64)
    encoder = TabularEncoder().fit(train_bundle, full_idx)
    train_features = encoder.transform(train_bundle, full_idx)
    test_features = encoder.transform(test_bundle)
    train_target = pd.to_numeric(train["label"], errors="raise").to_numpy(dtype=np.int8)
    train_metadata = train.loc[:, ["station", "layer", "time"]].reset_index(drop=True)
    test_metadata = test.loc[:, ["station", "layer", "time"]].reset_index(drop=True)
    probability, packages, fit_count, _ = _fit_candidate(
        selected,
        config,
        [int(value) for value in config["seeds"]],
        train_features,
        train_target,
        train_metadata,
        train["anomaly_type"].reset_index(drop=True),
        test_features,
        test_metadata,
    )
    plateau = detect_plateaus(test).to_numpy(dtype=bool)
    spike = detect_singleton_spikes(test).to_numpy(dtype=bool)
    prediction = apply_postprocess(
        test,
        probability,
        plateau,
        spike,
        config["deployment_postprocess"],
    )
    payload = {
        "schema_version": "p1_meaningful_learning_curve_saved_model.v1",
        "experiment_id": config["experiment_id"],
        "branch": selected,
        "feature_mode": "offline",
        "feature_columns": train_bundle.feature_columns,
        "categorical_columns": train_bundle.categorical_columns,
        "encoder": encoder,
        "packages": packages,
        "seeds": tuple(int(value) for value in config["seeds"]),
        "lightgbm_parameters": dict(config["lightgbm_parameters"]),
        "postprocess": dict(config["deployment_postprocess"]),
        "train_source_sha256": train.attrs["source_sha256"],
        "train_feature_cache_sha256": _sha256(
            PROJECT_ROOT / str(config["feature_cache"]["parquet"])
        ),
        "test_feature_cache_sha256": _sha256(test_cache),
        "model_count": fit_count,
    }
    model_path = artifact / "models" / f"{selected}_full.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        raise FileExistsError(model_path)
    joblib.dump(payload, model_path, compress=3)
    types = np.full(len(test), "", dtype=object)
    types[plateau & prediction.astype(bool)] = "flatline"
    types[spike & prediction.astype(bool)] = "spike"
    submission = build_submission(test, prediction, types)
    candidate_path = artifact / "candidate" / f"P1_MEANINGFUL_LC_V1_{selected.upper()}.csv"
    write_submission(submission, candidate_path)
    validation = validate_submission(candidate_path, test)
    loaded = joblib.load(model_path)
    if tuple(loaded["feature_columns"]) != test_bundle.feature_columns:
        raise RuntimeError("saved model/test feature schema mismatch")
    loaded_features = loaded["encoder"].transform(test_bundle)
    reproduced_probability = _predict_loaded_candidate(
        loaded["branch"], loaded["packages"], loaded_features, test_metadata
    )
    reproduced_prediction = apply_postprocess(
        test,
        reproduced_probability,
        plateau,
        spike,
        loaded["postprocess"],
    )
    reproduced_types = np.full(len(test), "", dtype=object)
    reproduced_types[plateau & reproduced_prediction.astype(bool)] = "flatline"
    reproduced_types[spike & reproduced_prediction.astype(bool)] = "spike"
    reproduced_submission = build_submission(test, reproduced_prediction, reproduced_types)
    reproduced_path = artifact / "candidate" / "reproduced.csv"
    write_submission(reproduced_submission, reproduced_path)
    candidate_sha = _sha256(candidate_path)
    reproduced_sha = _sha256(reproduced_path)
    if candidate_path.read_bytes() != reproduced_path.read_bytes():
        raise RuntimeError("saved model did not byte-reproduce the candidate")
    if int(validation["rows"]) != 169011:
        raise RuntimeError("P1 candidate row count is not 169011")
    return {
        "performed": True,
        "branch": selected,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "model_count": fit_count,
        "candidate_path": str(candidate_path),
        "candidate_sha256": candidate_sha,
        "candidate_bytes": candidate_path.stat().st_size,
        "candidate_rows": int(validation["rows"]),
        "candidate_positive_rows": int(validation["positive"]),
        "schema_key_order_valid": bool(validation["test_order_match"]),
        "reproduced_path": str(reproduced_path),
        "reproduced_sha256": reproduced_sha,
        "saved_model_byte_reproduces": candidate_sha == reproduced_sha,
        "test_value_reads": 2,
        "test_prediction_generations": 2,
        "test_feature_cache": test_cache_metadata,
    }


def _manifest(artifact: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(artifact.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or path.suffix == ".partial":
            continue
        files[str(path.relative_to(PROJECT_ROOT))] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {
        "schema_version": "p1_meaningful_learning_curve_manifest.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now_kst(),
        "artifacts": files,
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = config_path.resolve(strict=True)
    config = _json_load(config_path)
    seal_receipt = _verify_seal(config_path, config)
    data_dir = _data_dir()
    pins_before = _verify_pins(config, data_dir)
    artifact = _artifact_dir(config)
    if (artifact / "result.json").exists():
        raise FileExistsError("append-only experiment already has result.json")
    p1_config = load_config(PROJECT_ROOT / str(config["base_config"]))
    train = load_dataset(data_dir / "train.csv", kind="train", audit=False)
    train_cache = PROJECT_ROOT / str(config["feature_cache"]["parquet"])
    train_cache_metadata = PROJECT_ROOT / str(config["feature_cache"]["metadata"])
    bundle, feature_metadata = _load_feature_bundle(train, train_cache, train_cache_metadata)
    frozen_path = PROJECT_ROOT / "artifacts/runs/20260813T153038+0900_cv_378a4e89/oof.parquet"
    frozen_keys = pd.read_parquet(
        frozen_path,
        columns=[*KEY_COLUMNS, "fold", "prediction"],
    )
    folds, scope_audit = _fold_runtime(train, p1_config, frozen_keys)
    scope_path = artifact / "fold_scope_audit.json"
    if not scope_path.exists():
        _json_new(
            scope_path,
            {
                "schema_version": "p1_fold_scope_audit.v1",
                "folds": scope_audit,
                "strict_clock_caveat": {
                    "present": any(
                        not bool(value["nominal_wall_clock_scope_exact"])
                        for value in scope_audit.values()
                    ),
                    "meaning": "Frozen event-protected validation assignment can retain a complete positive event tail outside a nominal quarter; train/validation positions remain disjoint and every training cutoff remains earlier than validation start.",
                    "promotion_interpretation": "This is a validation-scope caveat, not a target leakage waiver, and must remain visible in downstream QA.",
                },
            },
        )
    prediction_receipt = _build_prediction_parts(train, bundle, p1_config, config, folds, artifact)
    # Only after the complete receipt exists are validation labels used for scores.
    frozen_oof = pd.read_parquet(frozen_path)
    current_oof = pd.read_parquet(
        PROJECT_ROOT / "artifacts/p1_full_improvement_cycle_20260822_v1/winner_oof.parquet"
    )
    branch_reports, evidence, canonical_decision = _score_all(
        train, frozen_oof, current_oof, config, folds, artifact
    )
    selected = str(evidence["selected_hypothesis"])
    if canonical_decision["passed"]:
        full_fit = _full_fit_and_reproduce(train, bundle, data_dir, config, selected, artifact)
        next_generation = None
    else:
        full_fit = {
            "performed": False,
            "reason": "no branch satisfied every sealed meaningful learning-curve gate",
            "test_value_reads": 0,
            "test_prediction_generations": 0,
            "candidate_files": 0,
        }
        next_generation = dict(config["next_generation_if_none_pass"])
    pins_after = _verify_pins(config, data_dir)
    protected_unchanged = {
        name: pins_before[name]["sha256"] == pins_after[name]["sha256"] for name in pins_before
    }
    if not all(protected_unchanged.values()):
        raise RuntimeError("an immutable input changed during the generation")
    result = {
        "schema_version": "p1_meaningful_learning_curve_result.v1",
        "experiment_id": config["experiment_id"],
        "completed_at_kst": _now_kst(),
        "elapsed_seconds": time.perf_counter() - started,
        "decision": canonical_decision["decision"],
        "passed": canonical_decision["passed"],
        "selected_hypothesis": selected,
        "canonical_curve_decision": canonical_decision,
        "branch_summary": {
            name: {
                "decision": report["decision"],
                "passed": report["passed"],
                "full_fraction_incumbent_f1": report["points"][-1]["incumbent"],
                "full_fraction_challenger_f1": report["points"][-1]["challenger"],
                "full_fraction_delta": report["points"][-1]["delta_candidate_minus_incumbent"],
                "full_fraction_ci90": report["points"][-1]["delta_ci90"],
                "gate_checks": report["gate_checks"],
            }
            for name, report in branch_reports.items()
        },
        "full_fit": full_fit,
        "exactly_one_next_structural_generation": next_generation,
        "strict_clock_caveat": _json_load(scope_path)["strict_clock_caveat"],
        "feature_mode": config["feature_cache"]["mode"],
        "feature_cache": feature_metadata,
        "prediction_phase": prediction_receipt,
        "operation_counters": {
            "curve_model_fits": prediction_receipt["model_fit_count"],
            "full_fit_model_fits": int(full_fit.get("model_count", 0)),
            "target_fold_scores": len(config["hypotheses"]) * len(config["prefix_fractions"]),
            "test_value_reads": int(full_fit["test_value_reads"]),
            "test_prediction_generations": int(full_fit["test_prediction_generations"]),
            "candidate_files": 1 if full_fit.get("performed") else 0,
            "uploads": 0,
            "source_mutations": 0,
            "frozen_mutations": 0,
        },
        "protected_input_sha256_unchanged": protected_unchanged,
        "preexecution_seal_sha256": _sha256(artifact / "preexecution_seal.json"),
        "config_sha256": seal_receipt["config_sha256"],
        "implementation_sha256": seal_receipt["implementation_sha256"],
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    _json_new(artifact / "result.json", result)
    registry = {
        "schema_version": "p1_meaningful_learning_curve_registry.v1",
        "experiment_id": config["experiment_id"],
        "registered_at_kst": _now_kst(),
        "decision": result["decision"],
        "selected_hypothesis": selected,
        "curve_evidence_path": str(
            (artifact / "learning_curve_evidence.json").relative_to(PROJECT_ROOT)
        ),
        "curve_evidence_sha256": _sha256(artifact / "learning_curve_evidence.json"),
        "canonical_decision_sha256": _sha256(artifact / "canonical_curve_decision.json"),
        "result_sha256": _sha256(artifact / "result.json"),
        "candidate": (
            {
                "path": full_fit["candidate_path"],
                "sha256": full_fit["candidate_sha256"],
                "model_path": full_fit["model_path"],
                "model_sha256": full_fit["model_sha256"],
                "byte_reproduced": full_fit["saved_model_byte_reproduces"],
            }
            if full_fit.get("performed")
            else None
        ),
        "uploads": 0,
        "source_mutations": 0,
        "frozen_mutations": 0,
    }
    _json_new(artifact / "registry.json", registry)
    _json_new(artifact / "manifest.json", _manifest(artifact, config))
    _emit(
        "generation_complete",
        decision=result["decision"],
        selected=selected,
        passed=result["passed"],
        elapsed_seconds=result["elapsed_seconds"],
    )
    return result


def self_check() -> None:
    truth = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    incumbent = np.asarray([0, 1, 0, 0, 0, 0], dtype=np.int8)
    challenger = np.asarray([0, 1, 1, 0, 1, 0], dtype=np.int8)
    assert _binary_f1(truth, challenger) == 1.0
    assert _binary_f1(truth, incumbent) < 1.0
    typed = _typed_target(pd.Series(["", "offset+drift", "spike", "", "flatline", ""]), truth)
    assert typed.tolist() == [0, 4, 1, 0, 3, 0]
    weight = _binary_weight(truth)
    assert weight.shape == truth.shape and np.isfinite(weight).all()
    config = _json_load(DEFAULT_CONFIG)
    assert len(config["hypotheses"]) == 3
    assert config["prefix_fractions"] == [0.4, 0.55, 0.7, 0.85, 1.0]
    assert len(config["seeds"]) == 3
    _emit("self_check_passed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--self-check", action="store_true")
    mode.add_argument("--seal-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
    elif args.seal_only:
        seal(args.config)
    else:
        run(args.config)


if __name__ == "__main__":
    main()
