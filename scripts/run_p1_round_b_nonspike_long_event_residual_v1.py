"""One-shot local-only P1 Round-B non-spike long-event residual screen.

The runner has no inference or export path.  It reuses the exact sealed Round-B
OOF surface, fits exactly three residual seeds per registered outer fold, saves
all predictions before scoring, and ends at a local screening decision.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from p1_qc.config import P1QCConfig, load_config
from p1_qc.data import KEY_COLUMNS, load_dataset
from p1_qc.features import FeatureBundle
from p1_qc.nonspike_long_event_residual import (
    binary_metrics,
    build_residual_training_view,
    connected_rescue,
)
from p1_qc.pipeline import TabularEncoder
from p1_qc.validation import normal_station_layer_day_fp, paired_block_bootstrap

DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "experiments"
    / "p1_round_b_nonspike_long_event_residual_v1.json"
)
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
BASE_PREFIX = "event_day_balanced_binary_lgbm"
MATCHED_PREFIX = "event_day_balanced_lightgbm__default"
DEPENDENCIES = (
    "src/p1_qc/nonspike_long_event_residual.py",
    "src/p1_qc/pipeline.py",
    "src/p1_qc/validation.py",
    "src/p1_qc/config.py",
    "src/p1_qc/data.py",
)


def _now_kst() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _parquet_new(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    frame.to_parquet(path, index=False, compression="zstd")


def _resolve_repo_path(value: str) -> Path:
    path = (PROJECT_ROOT / value).resolve(strict=True)
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"path escapes repository: {value}")
    return path


def _artifact_dir(config: Mapping[str, Any]) -> Path:
    path = (PROJECT_ROOT / str(config["artifact_dir"])).resolve()
    root = (PROJECT_ROOT / "artifacts").resolve()
    if not path.is_relative_to(root):
        raise RuntimeError("artifact_dir must remain under artifacts")
    return path


def _validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != (
        "p1_round_b_nonspike_long_event_residual.preregistration.v1"
    ):
        raise RuntimeError("unexpected preregistration schema")
    if config.get("experiment_id") != "p1_round_b_nonspike_long_event_residual_v1":
        raise RuntimeError("unexpected experiment id")
    surface = config["surface"]
    if tuple(surface["fold_order"]) != FOLD_ORDER:
        raise RuntimeError("outer fold order changed")
    if list(surface["seeds"]) != [20260813, 20260829, 20260847]:
        raise RuntimeError("registered seeds changed")
    if int(surface["expected_rows"]) != 421032:
        raise RuntimeError("OOF row contract changed")
    residual = config["residual_target"]
    decoder = config["rescue_decoder"]
    budget = config["resource_budget"]
    if int(residual["minimum_event_rows"]) != 19:
        raise RuntimeError("long-event lower bound changed")
    if int(residual["cadence_minutes"]) != 10:
        raise RuntimeError("cadence changed")
    if float(decoder["probability_threshold"]) != 0.8:
        raise RuntimeError("residual threshold changed")
    if int(decoder["maximum_anchor_distance_rows"]) != 18:
        raise RuntimeError("anchor distance changed")
    if decoder["threshold_tuning_grid"] != []:
        raise RuntimeError("threshold tuning is prohibited")
    if int(budget["round_b_base_model_fits"]) != 0:
        raise RuntimeError("Round-B refits are prohibited")
    if int(budget["residual_model_fits"]) != 9:
        raise RuntimeError("residual fit budget changed")
    if int(budget["result_driven_reruns"]) != 0:
        raise RuntimeError("result-driven reruns are prohibited")
    if not all(int(value) == 0 for value in config["prohibitions"].values()):
        raise RuntimeError("a prohibition counter is nonzero")


def _verify_immutable_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}

    def verify(name: str, spec: Mapping[str, Any]) -> Path:
        path = _resolve_repo_path(str(spec["path"]))
        observed = _sha256(path)
        if observed != str(spec["sha256"]):
            raise RuntimeError(f"immutable input SHA mismatch: {name}")
        records[name] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "bytes": path.stat().st_size,
            "sha256": observed,
        }
        return path

    base = config["base_config"]
    verify("base_config", base)
    immutable = config["immutable_inputs"]
    cache_path = verify("feature_cache", immutable["feature_cache"])
    metadata_path = verify("feature_metadata", immutable["feature_metadata"])
    verify("frozen_truth_oof", immutable["frozen_truth_oof"])
    verify("matched_budget_predictions", immutable["matched_budget_predictions"])
    for part in immutable["round_b_full_prefix_parts"]:
        verify(f"round_b_full_prefix:{part['fold']}", part)

    cache_file = pq.ParquetFile(cache_path)
    cache_spec = immutable["feature_cache"]
    if cache_file.metadata.num_rows != int(cache_spec["rows"]):
        raise RuntimeError("feature cache row count changed")
    if len(cache_file.schema_arrow.names) != int(cache_spec["columns"]):
        raise RuntimeError("feature cache column count changed")
    metadata = _json_load(metadata_path)
    if metadata.get("source_sha256") != cache_spec["source_sha256"]:
        raise RuntimeError("feature metadata source binding changed")
    if metadata.get("parquet_sha256") != cache_spec["sha256"]:
        raise RuntimeError("feature metadata parquet binding changed")
    if tuple(cache_file.schema_arrow.names) != tuple(metadata["feature_columns"]):
        raise RuntimeError("feature cache schema differs from metadata")
    if {"label", "anomaly_type"}.intersection(cache_file.schema_arrow.names):
        raise RuntimeError("feature cache contains protected target columns")
    return records


def _load_base_surface(config: Mapping[str, Any]) -> pd.DataFrame:
    seeds = [int(value) for value in config["surface"]["seeds"]]
    key_columns = [str(value) for value in config["surface"]["key_columns"]]
    columns = [
        *key_columns,
        "row_position",
        "fold",
        "fraction",
        f"{BASE_PREFIX}__probability",
        f"{BASE_PREFIX}__prediction",
        "spike_candidate",
    ]
    for seed in seeds:
        columns.extend(
            [
                f"{BASE_PREFIX}__seed_{seed}__probability",
                f"{BASE_PREFIX}__seed_{seed}__prediction",
            ]
        )
    parts: list[pd.DataFrame] = []
    immutable = config["immutable_inputs"]
    for spec in immutable["round_b_full_prefix_parts"]:
        part = pd.read_parquet(_resolve_repo_path(str(spec["path"])), columns=columns)
        if len(part) == 0 or not part["fold"].eq(str(spec["fold"])).all():
            raise RuntimeError(f"invalid Round-B part: {spec['fold']}")
        if not np.isclose(part["fraction"].to_numpy(dtype=float), 1.0).all():
            raise RuntimeError("Round-B part is not full-prefix p100")
        parts.append(part)
    surface = pd.concat(parts, ignore_index=True)
    if len(surface) != int(config["surface"]["expected_rows"]):
        raise RuntimeError("Round-B surface row count changed")
    if surface.duplicated(key_columns).any() or not surface["row_position"].is_unique:
        raise RuntimeError("Round-B surface keys or row positions are duplicated")
    if tuple(surface["fold"].drop_duplicates().tolist()) != FOLD_ORDER:
        raise RuntimeError("Round-B surface fold order changed")

    matched_columns = [*key_columns, MATCHED_PREFIX]
    matched_columns.extend(f"{MATCHED_PREFIX}__seed_{seed}" for seed in seeds)
    matched = pd.read_parquet(
        _resolve_repo_path(str(immutable["matched_budget_predictions"]["path"])),
        columns=matched_columns,
    )
    aligned = surface.loc[:, key_columns].merge(
        matched,
        on=key_columns,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(surface) or aligned[MATCHED_PREFIX].isna().any():
        raise RuntimeError("matched-budget Round-B alignment failed")
    comparisons = [
        (
            surface[f"{BASE_PREFIX}__prediction"].to_numpy(dtype=np.int8),
            aligned[MATCHED_PREFIX].to_numpy(dtype=np.int8),
        )
    ]
    for seed in seeds:
        comparisons.append(
            (
                surface[f"{BASE_PREFIX}__seed_{seed}__prediction"].to_numpy(dtype=np.int8),
                aligned[f"{MATCHED_PREFIX}__seed_{seed}"].to_numpy(dtype=np.int8),
            )
        )
    if any(not np.array_equal(left, right) for left, right in comparisons):
        raise RuntimeError("p100 Round-B predictions differ from matched-budget default")
    return surface


def preflight(config_path: Path) -> dict[str, Any]:
    config = _json_load(config_path)
    _validate_contract(config)
    pins = _verify_immutable_inputs(config)
    surface = _load_base_surface(config)
    return {
        "schema_version": "p1_round_b_nonspike_long_event_residual.preflight.v1",
        "experiment_id": config["experiment_id"],
        "status": "PASS_READY_TO_SEAL_OR_EXECUTE",
        "rows": len(surface),
        "fold_rows": {
            fold: int(surface["fold"].eq(fold).sum()) for fold in FOLD_ORDER
        },
        "row_positions_unique": bool(surface["row_position"].is_unique),
        "exact_round_b_default_equivalence": True,
        "round_b_base_model_fits_required": 0,
        "residual_model_fits_registered": 9,
        "immutable_inputs": pins,
        "protected_source_reads": 0,
        "outer_scores_computed": 0,
    }


def seal(config_path: Path) -> Path:
    config = _json_load(config_path)
    report = preflight(config_path)
    artifact = _artifact_dir(config)
    artifact.mkdir(parents=True, exist_ok=True)
    path = artifact / "preexecution_seal.json"
    if path.exists():
        raise FileExistsError(path)
    dependencies = {
        name: _sha256(_resolve_repo_path(name)) for name in DEPENDENCIES
    }
    receipt = {
        "schema_version": "p1_round_b_nonspike_long_event_residual.seal.v1",
        "experiment_id": config["experiment_id"],
        "status": "SEALED_BEFORE_OUTER_ONE_SHOT",
        "sealed_at_kst": _now_kst(),
        "config_path": str(config_path.relative_to(PROJECT_ROOT)),
        "config_sha256": _sha256(config_path),
        "runner_path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "runner_sha256": _sha256(Path(__file__).resolve()),
        "dependency_sha256": dependencies,
        "preflight": report,
        "registered_model_fits": 9,
        "registered_outer_scores": 1,
        "operation_counters_at_seal": {
            "round_b_base_model_fits": 0,
            "residual_model_fits": 0,
            "outer_scores": 0,
            "full_fits": 0,
            "candidate_files": 0,
            "uploads": 0,
        },
    }
    _json_new(path, receipt)
    return path


def _verify_seal(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _json_load(_artifact_dir(config) / "preexecution_seal.json")
    if receipt.get("status") != "SEALED_BEFORE_OUTER_ONE_SHOT":
        raise RuntimeError("invalid preexecution seal status")
    if receipt.get("config_sha256") != _sha256(config_path):
        raise RuntimeError("config changed after seal")
    if receipt.get("runner_sha256") != _sha256(Path(__file__).resolve()):
        raise RuntimeError("runner changed after seal")
    for name, expected in receipt["dependency_sha256"].items():
        if _sha256(_resolve_repo_path(name)) != expected:
            raise RuntimeError(f"dependency changed after seal: {name}")
    return receipt


def _training_source() -> Path:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR must identify the immutable P1 source directory")
    directory = Path(raw).expanduser().resolve(strict=True)
    path = directory / "train.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _load_feature_bundle(
    train: pd.DataFrame, config: Mapping[str, Any]
) -> tuple[FeatureBundle, dict[str, Any]]:
    cache_spec = config["immutable_inputs"]["feature_cache"]
    metadata_spec = config["immutable_inputs"]["feature_metadata"]
    cache_path = _resolve_repo_path(str(cache_spec["path"]))
    metadata = _json_load(_resolve_repo_path(str(metadata_spec["path"])))
    features = pd.read_parquet(cache_path)
    if len(features) != len(train) or int(metadata["rows"]) != len(train):
        raise RuntimeError("feature cache row count differs from training source")
    if train.attrs.get("source_sha256") != cache_spec["source_sha256"]:
        raise RuntimeError("training source differs from feature-cache source binding")
    if tuple(features.columns) != tuple(metadata["feature_columns"]):
        raise RuntimeError("feature cache schema changed")
    features.index = train.index.copy()
    return (
        FeatureBundle(
            features,
            tuple(str(value) for value in metadata["feature_columns"]),
            tuple(str(value) for value in metadata["categorical_columns"]),
        ),
        metadata,
    )


def _fold_runtime(
    train: pd.DataFrame,
    p1_config: P1QCConfig,
    surface: pd.DataFrame,
) -> list[dict[str, Any]]:
    parsed = pd.to_datetime(train["time"], errors="raise", utc=True, format="mixed")
    folds: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(p1_config.splits.folds):
        if spec.name not in FOLD_ORDER:
            raise RuntimeError(f"unexpected fold: {spec.name}")
        part = surface.loc[surface["fold"].eq(spec.name)].reset_index(drop=True)
        val_idx = part["row_position"].to_numpy(dtype=np.int64)
        if (val_idx < 0).any() or (val_idx >= len(train)).any():
            raise RuntimeError(f"invalid frozen row positions: {spec.name}")
        expected_keys = train.iloc[val_idx].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        if not expected_keys.equals(part.loc[:, list(KEY_COLUMNS)]):
            raise RuntimeError(f"frozen key/row-position binding changed: {spec.name}")
        train_end = pd.Timestamp(spec.train_end).tz_convert("UTC")
        val_start = pd.Timestamp(spec.val_start).tz_convert("UTC")
        train_idx = np.flatnonzero(parsed.le(train_end).to_numpy()).astype(np.int64)
        if np.intersect1d(train_idx, val_idx).size:
            raise RuntimeError(f"training and validation overlap: {spec.name}")
        if parsed.iloc[train_idx].max() >= val_start - pd.Timedelta(days=7):
            raise RuntimeError(f"seven-day outer purge changed: {spec.name}")
        folds.append(
            {
                "name": spec.name,
                "ordinal": ordinal,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "part": part,
                "train_end_utc": train_end.isoformat(),
                "val_start_utc": val_start.isoformat(),
            }
        )
    if tuple(item["name"] for item in folds) != FOLD_ORDER:
        raise RuntimeError("outer fold order changed")
    return folds


def _model_parameters(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    params = dict(config["residual_model"]["parameters"])
    params.update(
        {
            "objective": "binary",
            "random_state": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "extra_seed": seed,
        }
    )
    return params


def _fit_fold(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    config: Mapping[str, Any],
    fold: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_idx = np.asarray(fold["train_idx"], dtype=np.int64)
    val_idx = np.asarray(fold["val_idx"], dtype=np.int64)
    metadata = train.iloc[train_idx][["station", "layer", "time"]].reset_index(drop=True)
    truth = pd.to_numeric(train.iloc[train_idx]["label"], errors="raise").to_numpy(
        dtype=np.int8
    )
    anomaly_type = train.iloc[train_idx]["anomaly_type"].reset_index(drop=True)
    target_cfg = config["residual_target"]
    view = build_residual_training_view(
        truth,
        anomaly_type,
        metadata,
        min_event_rows=int(target_cfg["minimum_event_rows"]),
        cadence_minutes=int(target_cfg["cadence_minutes"]),
    )
    eligible_idx = train_idx[view.indices]
    encoder = TabularEncoder().fit(bundle, train_idx)
    fit_features = encoder.transform(bundle, eligible_idx)
    validation_features = encoder.transform(bundle, val_idx)
    part = fold["part"].copy()
    validation_metadata = part[["station", "layer", "time"]].reset_index(drop=True)
    spike = part["spike_candidate"].to_numpy(dtype=bool)
    decoder = config["rescue_decoder"]
    seeds = [int(value) for value in config["surface"]["seeds"]]
    probabilities: list[np.ndarray] = []
    seed_candidates: dict[int, np.ndarray] = {}
    seed_rescue_rows: dict[int, int] = {}
    for registered_seed in seeds:
        fit_seed = registered_seed + int(fold["ordinal"])
        model = lgb.LGBMClassifier(**_model_parameters(config, fit_seed))
        model.fit(
            fit_features,
            view.target,
            sample_weight=view.sample_weight,
        )
        probability = model.predict_proba(validation_features)[:, 1]
        if not np.isfinite(probability).all():
            raise RuntimeError("non-finite residual probability")
        probabilities.append(probability.astype(np.float64))
        base_seed = part[f"{BASE_PREFIX}__seed_{registered_seed}__prediction"].to_numpy(
            dtype=np.int8
        )
        candidate, rescue = connected_rescue(
            base_seed,
            probability,
            validation_metadata,
            spike,
            threshold=float(decoder["probability_threshold"]),
            max_distance_rows=int(decoder["maximum_anchor_distance_rows"]),
            cadence_minutes=int(target_cfg["cadence_minutes"]),
        )
        seed_candidates[registered_seed] = candidate
        seed_rescue_rows[registered_seed] = int(rescue.sum())

    ensemble_probability = np.mean(np.vstack(probabilities), axis=0)
    base = part[f"{BASE_PREFIX}__prediction"].to_numpy(dtype=np.int8)
    candidate, rescue = connected_rescue(
        base,
        ensemble_probability,
        validation_metadata,
        spike,
        threshold=float(decoder["probability_threshold"]),
        max_distance_rows=int(decoder["maximum_anchor_distance_rows"]),
        cadence_minutes=int(target_cfg["cadence_minutes"]),
    )
    output = part.loc[:, [*KEY_COLUMNS, "row_position", "fold"]].copy()
    output["round_b_prediction"] = base
    output["residual_probability"] = ensemble_probability.astype(np.float32)
    output["candidate_prediction"] = candidate
    output["rescue"] = rescue
    for index, registered_seed in enumerate(seeds):
        output[f"round_b__seed_{registered_seed}"] = part[
            f"{BASE_PREFIX}__seed_{registered_seed}__prediction"
        ].to_numpy(dtype=np.int8)
        output[f"residual_probability__seed_{registered_seed}"] = probabilities[index].astype(
            np.float32
        )
        output[f"candidate__seed_{registered_seed}"] = seed_candidates[registered_seed]
    audit = {
        "fold": fold["name"],
        "train_rows": len(train_idx),
        "residual_fit_rows": len(view.indices),
        "residual_positive_rows": view.positive_row_count,
        "residual_positive_events": view.positive_event_count,
        "excluded_positive_rows": view.excluded_positive_rows,
        "right_censored_event_count": view.right_censored_event_count,
        "validation_rows": len(output),
        "ensemble_rescue_rows": int(rescue.sum()),
        "seed_rescue_rows": seed_rescue_rows,
        "model_fits": len(seeds),
        "train_end_utc": fold["train_end_utc"],
        "validation_start_utc": fold["val_start_utc"],
        "target_fold_scores_computed": 0,
    }
    return output, audit


def _positive_event_count(truth: np.ndarray, metadata: pd.DataFrame) -> int:
    work = metadata[["station", "layer", "time"]].reset_index(drop=True).copy()
    work["truth"] = truth
    work["position"] = np.arange(len(work), dtype=np.int64)
    work["parsed"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "parsed", "position"], inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["parsed"].diff().dt.total_seconds().eq(600)
    prior = grouped["truth"].shift(1).fillna(0).eq(1)
    starts = work["truth"].eq(1) & (~contiguous | ~prior)
    return int(starts.sum())


def _slice_metrics(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    candidate_metrics = binary_metrics(truth, candidate)
    baseline_metrics = binary_metrics(truth, baseline)
    return {
        "rows": len(truth),
        "positive_rows": int(truth.sum()),
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "f1_delta": float(candidate_metrics["f1"] - baseline_metrics["f1"]),
        "precision_delta": float(
            candidate_metrics["precision"] - baseline_metrics["precision"]
        ),
        "recall_delta": float(candidate_metrics["recall"] - baseline_metrics["recall"]),
    }


def _structural_audit(
    baseline: np.ndarray,
    candidate: np.ndarray,
    metadata: pd.DataFrame,
) -> dict[str, int]:
    work = metadata[["station", "layer", "time"]].reset_index(drop=True).copy()
    work["base"] = baseline
    work["candidate"] = candidate
    work["position"] = np.arange(len(work), dtype=np.int64)
    work["parsed"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "parsed", "position"], inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["parsed"].diff().dt.total_seconds().eq(600)
    prior = grouped["candidate"].shift(1).fillna(0).eq(1)
    starts = work["candidate"].eq(1) & (~contiguous | ~prior)
    work["run"] = starts.cumsum().where(work["candidate"].eq(1), -1).astype(np.int64)
    runs = work.loc[work["candidate"].eq(1)].groupby("run", sort=False)
    disconnected = 0
    singleton = 0
    for _, group in runs:
        if int(group["base"].sum()) == 0:
            disconnected += 1
            if len(group) == 1:
                singleton += 1
    return {
        "removed_round_b_positive_rows": int(((baseline == 1) & (candidate == 0)).sum()),
        "rescued_rows": int(((baseline == 0) & (candidate == 1)).sum()),
        "new_disconnected_events": disconnected,
        "new_singletons": singleton,
    }


def _score(
    config: Mapping[str, Any],
    predictions: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, bool]]:
    immutable = config["immutable_inputs"]
    truth_frame = pd.read_parquet(
        _resolve_repo_path(str(immutable["frozen_truth_oof"]["path"])),
        columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
    )
    aligned = predictions.merge(
        truth_frame,
        on=[*KEY_COLUMNS, "fold"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if len(aligned) != len(predictions) or aligned["label"].isna().any():
        raise RuntimeError("truth OOF alignment failed")
    truth = aligned["label"].to_numpy(dtype=np.int8)
    baseline = aligned["round_b_prediction"].to_numpy(dtype=np.int8)
    candidate = aligned["candidate_prediction"].to_numpy(dtype=np.int8)
    metadata = aligned[["station", "layer", "time"]].reset_index(drop=True)
    pooled = _slice_metrics(truth, candidate, baseline)
    expected = config["surface"]["expected_base_metrics"]
    for name in ("tp", "fp", "fn"):
        if int(pooled["baseline"][name]) != int(expected[name]):
            raise RuntimeError(f"exact Round-B baseline count changed: {name}")
    for name in ("f1", "precision", "recall"):
        if not np.isclose(
            float(pooled["baseline"][name]), float(expected[name]), rtol=0.0, atol=1e-12
        ):
            raise RuntimeError(f"exact Round-B baseline metric changed: {name}")

    bootstrap_cfg = config["outer_protocol"]["paired_bootstrap"]
    bootstrap = paired_block_bootstrap(
        truth,
        candidate,
        baseline,
        metadata,
        replicates=int(bootstrap_cfg["replicates"]),
        seed=int(bootstrap_cfg["seed"]),
        cadence_minutes=int(config["residual_target"]["cadence_minutes"]),
        normal_day_timezone=str(bootstrap_cfg["normal_day_timezone"]),
    )
    fp_day = normal_station_layer_day_fp(truth, candidate, baseline, metadata)
    candidate_fp_rate = fp_day["candidate"]["false_positive_rows_per_normal_station_layer_day"]
    baseline_fp_rate = fp_day["baseline"]["false_positive_rows_per_normal_station_layer_day"]
    fp_ratio = (
        float(candidate_fp_rate / baseline_fp_rate)
        if baseline_fp_rate not in (None, 0)
        else (1.0 if candidate_fp_rate == baseline_fp_rate else float("inf"))
    )

    fold_metrics: dict[str, Any] = {}
    station_metrics: dict[str, Any] = {}
    cell_metrics: dict[str, Any] = {}
    for fold in FOLD_ORDER:
        mask = aligned["fold"].eq(fold).to_numpy()
        fold_metrics[fold] = _slice_metrics(truth[mask], candidate[mask], baseline[mask])
    stations = sorted(aligned["station"].astype(str).unique().tolist())
    for station in stations:
        mask = aligned["station"].astype(str).eq(station).to_numpy()
        station_metrics[station] = _slice_metrics(
            truth[mask], candidate[mask], baseline[mask]
        )

    support = config["outer_protocol"]["adequate_cell_support"]
    equalized_deltas: list[float] = []
    adequate_deltas: list[float] = []
    parsed = pd.to_datetime(aligned["time"], errors="raise", utc=True, format="mixed")
    for station in stations:
        for fold in FOLD_ORDER:
            mask = (
                aligned["station"].astype(str).eq(station)
                & aligned["fold"].astype(str).eq(fold)
            ).to_numpy()
            name = f"{station}|{fold}"
            values = _slice_metrics(truth[mask], candidate[mask], baseline[mask])
            events = _positive_event_count(truth[mask], metadata.loc[mask].reset_index(drop=True))
            positive_days = int(
                pd.DataFrame(
                    {
                        "station": aligned.loc[mask, "station"].astype(str).to_numpy(),
                        "layer": aligned.loc[mask, "layer"].astype(str).to_numpy(),
                        "day": parsed.loc[mask].dt.tz_convert("Asia/Seoul").dt.strftime(
                            "%Y-%m-%d"
                        ).to_numpy(),
                        "truth": truth[mask],
                    }
                )
                .loc[lambda frame: frame["truth"].eq(1), ["station", "layer", "day"]]
                .drop_duplicates()
                .shape[0]
            )
            adequate = bool(
                values["positive_rows"] >= int(support["minimum_positive_rows"])
                and events >= int(support["minimum_positive_events"])
                and positive_days >= int(support["minimum_positive_kst_days"])
            )
            values.update(
                {
                    "positive_events": events,
                    "positive_kst_days": positive_days,
                    "adequate_support": adequate,
                }
            )
            cell_metrics[name] = values
            if 0 < int(values["positive_rows"]) < int(values["rows"]):
                equalized_deltas.append(float(values["f1_delta"]))
            if adequate:
                adequate_deltas.append(float(values["f1_delta"]))

    equalized_delta = float(np.mean(equalized_deltas)) if equalized_deltas else float("nan")
    worst_adequate = float(min(adequate_deltas)) if adequate_deltas else float("nan")
    seeds = [int(value) for value in config["surface"]["seeds"]]
    seed_deltas: dict[str, float] = {}
    for seed in seeds:
        base_seed = aligned[f"round_b__seed_{seed}"].to_numpy(dtype=np.int8)
        candidate_seed = aligned[f"candidate__seed_{seed}"].to_numpy(dtype=np.int8)
        seed_deltas[str(seed)] = float(
            binary_metrics(truth, candidate_seed)["f1"]
            - binary_metrics(truth, base_seed)["f1"]
        )

    types = aligned["anomaly_type"].astype("string").fillna("")
    spike_mask = types.map(
        lambda value: "spike"
        in {token.strip() for token in str(value).split("+") if token.strip()}
    ).to_numpy()
    spike_truth = truth[spike_mask]
    spike_base_recall = (
        float(baseline[spike_mask].sum() / spike_truth.sum()) if spike_truth.sum() else 0.0
    )
    spike_candidate_recall = (
        float(candidate[spike_mask].sum() / spike_truth.sum()) if spike_truth.sum() else 0.0
    )
    structural = _structural_audit(baseline, candidate, metadata)
    gates = config["fail_fast_gates"]
    checks = {
        "pooled_micro_f1_delta": pooled["f1_delta"]
        >= float(gates["pooled_micro_f1_delta_gte"]),
        "paired_bootstrap_ci90_lower": float(bootstrap["difference_ci90"][0])
        > float(gates["paired_bootstrap_ci90_lower_gt"]),
        "nonnegative_fold_count": sum(
            float(value["f1_delta"]) >= 0 for value in fold_metrics.values()
        )
        >= int(gates["minimum_nonnegative_fold_count"]),
        "nonnegative_station_count": sum(
            float(value["f1_delta"]) >= 0 for value in station_metrics.values()
        )
        >= int(gates["minimum_nonnegative_station_count"]),
        "equal_weight_station_fold_f1_delta": equalized_delta
        >= float(gates["equal_weight_station_fold_f1_delta_gte"]),
        "adequately_supported_worst_cell_f1_delta": bool(adequate_deltas)
        and worst_adequate >= float(gates["adequately_supported_worst_cell_f1_delta_gte"]),
        "all_seed_f1_deltas": all(
            value >= float(gates["all_seed_f1_deltas_gte"])
            for value in seed_deltas.values()
        ),
        "precision_delta": pooled["precision_delta"] >= float(gates["precision_delta_gte"]),
        "recall_delta": pooled["recall_delta"] >= float(gates["recall_delta_gte"]),
        "normal_fp_per_day_ratio": fp_ratio <= float(gates["normal_fp_per_day_ratio_lte"]),
        "spike_recall_delta": spike_candidate_recall - spike_base_recall
        >= float(gates["spike_recall_delta_gte"]),
        "new_disconnected_events": structural["new_disconnected_events"]
        == int(gates["new_disconnected_events_eq"]),
        "new_singletons": structural["new_singletons"] == int(gates["new_singletons_eq"]),
    }
    metrics = {
        "schema_version": "p1_round_b_nonspike_long_event_residual.metrics.v1",
        "rows": len(aligned),
        "pooled": pooled,
        "bootstrap": bootstrap,
        "normal_fp_day": fp_day,
        "normal_fp_per_day_ratio": fp_ratio,
        "folds": fold_metrics,
        "stations": station_metrics,
        "station_fold_cells": cell_metrics,
        "equal_weight_station_fold_f1_delta": equalized_delta,
        "adequately_supported_worst_cell_f1_delta": worst_adequate,
        "seed_f1_deltas": seed_deltas,
        "spike": {
            "rows": int(spike_mask.sum()),
            "base_recall": spike_base_recall,
            "candidate_recall": spike_candidate_recall,
            "recall_delta": spike_candidate_recall - spike_base_recall,
        },
        "structural": structural,
        "gate_checks": checks,
        "passed_all_gates": all(checks.values()),
    }
    return metrics, checks


def execute(config_path: Path) -> Path:
    started = time.perf_counter()
    config = _json_load(config_path)
    _validate_contract(config)
    seal_receipt = _verify_seal(config_path, config)
    _verify_immutable_inputs(config)
    artifact = _artifact_dir(config)
    forbidden_existing = [
        artifact / "predictions_complete.json",
        artifact / "predictions.parquet",
        artifact / "metrics.json",
        artifact / "result.json",
        artifact / "manifest.json",
    ]
    forbidden_existing.extend(artifact / "prediction_parts" / f"{fold}.parquet" for fold in FOLD_ORDER)
    forbidden_existing.extend(artifact / "prediction_parts" / f"{fold}.json" for fold in FOLD_ORDER)
    if any(path.exists() for path in forbidden_existing):
        raise FileExistsError("one-shot output already exists; rerun is prohibited")

    train = load_dataset(_training_source(), kind="train", audit=False)
    bundle, feature_metadata = _load_feature_bundle(train, config)
    surface = _load_base_surface(config)
    p1_config = load_config(_resolve_repo_path(str(config["base_config"]["path"])))
    folds = _fold_runtime(train, p1_config, surface)
    part_frames: list[pd.DataFrame] = []
    fit_audits: dict[str, Any] = {}
    fit_count = 0
    wall_cap = float(config["resource_budget"]["wall_clock_cap_seconds"])
    for fold in folds:
        if time.perf_counter() - started > wall_cap:
            raise TimeoutError("wall-clock cap exceeded before next outer fold")
        output, audit = _fit_fold(train, bundle, config, fold)
        part_path = artifact / "prediction_parts" / f"{fold['name']}.parquet"
        audit_path = artifact / "prediction_parts" / f"{fold['name']}.json"
        _parquet_new(part_path, output)
        audit.update(
            {
                "parquet_path": str(part_path.relative_to(PROJECT_ROOT)),
                "parquet_sha256": _sha256(part_path),
                "completed_at_kst": _now_kst(),
            }
        )
        _json_new(audit_path, audit)
        part_frames.append(output)
        fit_audits[str(fold["name"])] = audit
        fit_count += int(audit["model_fits"])
    if fit_count != int(config["resource_budget"]["residual_model_fits"]):
        raise RuntimeError("residual fit count differs from preregistration")
    predictions = pd.concat(part_frames, ignore_index=True)
    if len(predictions) != int(config["surface"]["expected_rows"]):
        raise RuntimeError("completed prediction surface has wrong row count")
    predictions_path = artifact / "predictions.parquet"
    _parquet_new(predictions_path, predictions)
    complete_path = artifact / "predictions_complete.json"
    _json_new(
        complete_path,
        {
            "schema_version": "p1_round_b_nonspike_long_event_residual.predictions_complete.v1",
            "experiment_id": config["experiment_id"],
            "status": "ALL_OUTER_PREDICTIONS_FROZEN_BEFORE_SCORING",
            "rows": len(predictions),
            "residual_model_fits": fit_count,
            "round_b_base_model_fits": 0,
            "prediction_sha256": _sha256(predictions_path),
            "fold_audits": fit_audits,
            "target_fold_scores_computed_before_receipt": 0,
            "completed_at_kst": _now_kst(),
        },
    )
    if time.perf_counter() - started > wall_cap:
        raise TimeoutError("wall-clock cap exceeded after prediction freeze")

    metrics, checks = _score(config, predictions)
    metrics_path = artifact / "metrics.json"
    _json_new(metrics_path, metrics)
    passed = all(checks.values())
    result_path = artifact / "result.json"
    result = {
        "schema_version": "p1_round_b_nonspike_long_event_residual.result.v1",
        "experiment_id": config["experiment_id"],
        "status": "COMPLETE_LOCAL_SCREEN_ONLY",
        "decision": config["interpretation"]["pass_label"]
        if passed
        else config["interpretation"]["fail_label"],
        "passed_all_gates": passed,
        "completed_at_kst": _now_kst(),
        "elapsed_seconds": time.perf_counter() - started,
        "config_sha256": _sha256(config_path),
        "preexecution_seal_sha256": _sha256(artifact / "preexecution_seal.json"),
        "prediction_sha256": _sha256(predictions_path),
        "metrics_sha256": _sha256(metrics_path),
        "feature_cache_sha256": feature_metadata["parquet_sha256"],
        "operation_counters": {
            "round_b_base_model_fits": 0,
            "residual_model_fits": fit_count,
            "outer_scores": 1,
            "full_fits": 0,
            "candidate_files": 0,
            "uploads": 0,
            "source_mutations": 0,
            "frozen_artifact_mutations": 0,
        },
        "independent_confirmation": config["interpretation"]["independent_confirmation"],
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
        "seal_status": seal_receipt["status"],
    }
    _json_new(result_path, result)
    manifest_path = artifact / "manifest.json"
    files = [
        config_path,
        artifact / "preexecution_seal.json",
        predictions_path,
        complete_path,
        metrics_path,
        result_path,
    ]
    files.extend(artifact / "prediction_parts" / f"{fold}.parquet" for fold in FOLD_ORDER)
    files.extend(artifact / "prediction_parts" / f"{fold}.json" for fold in FOLD_ORDER)
    _json_new(
        manifest_path,
        {
            "schema_version": "p1_round_b_nonspike_long_event_residual.manifest.v1",
            "experiment_id": config["experiment_id"],
            "created_at_kst": _now_kst(),
            "artifacts": {
                str(path.relative_to(PROJECT_ROOT)): {
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in files
            },
        },
    )
    return result_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--seal", action="store_true")
    action.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve(strict=True)
    if not config_path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError("config must remain inside repository")
    if args.preflight:
        output: Any = preflight(config_path)
    elif args.seal:
        output = str(seal(config_path))
    else:
        output = str(execute(config_path))
    print(json.dumps({"status": "ok", "output": output}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
