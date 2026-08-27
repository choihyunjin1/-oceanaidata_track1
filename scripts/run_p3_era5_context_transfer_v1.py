"""One-shot, research-only runner for the preregistered P3 ERA5 transfer check.

The executable input surface is intentionally limited to the frozen local training
cache, its historical OOF benchmark, and the pre-2024 ERA5 quarantine.  The runner
never performs operational inference and never creates a deliverable prediction
file.  With no flag (or ``--check-only``) it performs a zero-write contract audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ocean_external.policy import preflight_external_use  # noqa: E402
from p3_wave import era5_pretrain_data as era5_data  # noqa: E402
from p3_wave.era5_context_transfer import (  # noqa: E402
    LEADS,
    LOCAL_CATBOOST_PARAMETERS,
    SOURCE_CATBOOST_PARAMETERS,
    FixedContextTransferRegressor,
    build_source_cases,
    common_feature_columns,
    select_common_cached_features,
    select_source_year_validation,
)
from p3_wave.validation import DEFAULT_WINDOWS, ForecastFold, build_forecast_folds  # noqa: E402

SEED = 20260824
BOOTSTRAP_REPLICATES = 5000
HELD_YEARS = (2021, 2022, 2023)
SOURCE_TRAIN_YEARS = tuple(range(2014, 2021))
SHRINK_LEADS = (12, 18, 24)
SHRINK_WEIGHT = 0.20
EXPECTED_CASES = 181
EXPECTED_ROWS = EXPECTED_CASES * len(LEADS)
EXPECTED_FEATURES = 286
SOURCE_ID = "era5_pre2024"
SOURCE_PURPOSE = "pretraining"
COMBINED_NAME = "era5_p3_context_pretrain_2014_2023.parquet"

LOCAL_CONTROL_STAGE1_PARAMETERS: Mapping[str, Any] = {
    "loss_function": "RMSE",
    "iterations": 600,
    "depth": 8,
    "learning_rate": 0.04,
    "l2_leaf_reg": 8.0,
    "random_seed": SEED,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": False,
}
LOCAL_CONTROL_STAGE2_PARAMETERS: Mapping[str, Any] = {
    "loss_function": "RMSE",
    "iterations": 250,
    "depth": 8,
    "learning_rate": 0.03,
    "l2_leaf_reg": 12.0,
    "random_seed": SEED,
    "thread_count": -1,
    "allow_writing_files": False,
    "verbose": False,
}
MATCHED_CONTROL_CONFIG: Mapping[str, Any] = {
    "stage1": {
        "training_data": "same_local_fold_prefix",
        "iterations": 600,
        "depth": 8,
        "learning_rate": 0.04,
        "l2_leaf_reg": 8.0,
        "sample_weight": "uniform",
        "init_model": False,
    },
    "stage2": {
        "training_data": "same_local_fold_prefix",
        "iterations": 250,
        "depth": 8,
        "learning_rate": 0.03,
        "l2_leaf_reg": 12.0,
        "sample_weight": "exp(-0.45*max(current_hs-1.5,0))",
        "init_model": "independent_stage1_local_model",
    },
    "random_seed": SEED,
    "same_286_features_log_delta_and_postprocess": True,
}

EXTERNAL_COLUMN_MAP: Mapping[str, str] = {
    "swh_m": "hs",
    "mwp_s": "tp",
    "hmax_m": "hmax",
    "mwd_deg": "wvdir",
    "wspd10_m_s": "wspd",
    "wdir10_from_deg": "wdir",
    "t2m_c": "airt",
    "relh2m_pct": "relh",
    "msl_hpa": "caph",
}

BLIND_COMPARATOR_COLUMNS = (
    "prefix_fraction",
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "current_hs",
    "incumbent_prediction",
)
TRUTH_COLUMNS = (
    "prefix_fraction",
    "fold",
    "anchor_id",
    "station",
    "lead_h",
    "target_hs",
)
KEY_COLUMNS = ("fold", "anchor_id", "station", "lead_h")
TARGET_COLUMNS = tuple(f"target_{lead}" for lead in LEADS)


class ContractError(RuntimeError):
    """A preregistration, provenance, split, or blind-scoring contract failed."""


@dataclass(frozen=True)
class RunPaths:
    root: Path
    experiment_config: Path
    external_scope: Path
    catalog: Path
    permission_receipt: Path
    train_features: Path
    train_anchors: Path
    validation_keys: Path
    incumbent_oof: Path
    quarantine: Path
    manifest: Path
    output: Path
    attempt_lock: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _bound(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not _inside(path, root):
        raise ContractError("configured path escapes repository root")
    return path


def _resolve_paths(root: Path, config: Mapping[str, Any], scope: Mapping[str, Any]) -> RunPaths:
    bindings = config["bindings"]
    output_rel = str(config["access_and_output"]["artifact_dir"])
    output = _bound(root, output_rel)
    external_scope = _bound(root, str(bindings["external_scope"]["path"]))
    quarantine = _bound(root, str(config["access_and_output"]["external_quarantine"]))
    paths = RunPaths(
        root=root.resolve(),
        experiment_config=_bound(root, "configs/experiments/p3_era5_context_transfer_v1.json"),
        external_scope=external_scope,
        catalog=_bound(root, str(scope["bindings"]["catalog"]["path"])),
        permission_receipt=_bound(
            root, str(scope["bindings"]["official_permission_receipt"]["path"])
        ),
        train_features=_bound(root, str(bindings["train_features"]["path"])),
        train_anchors=_bound(root, str(bindings["train_anchors"]["path"])),
        validation_keys=_bound(root, str(bindings["validation_keys"]["path"])),
        incumbent_oof=_bound(root, str(bindings["frozen_incumbent_oof"]["path"])),
        quarantine=quarantine,
        manifest=quarantine / "manifests" / "manifest.json",
        output=output,
        attempt_lock=output.with_name(f"{output.name}.attempt.lock"),
    )
    allowed_read_roots = (
        (root / "artifacts" / "p3").resolve(),
        (root / "artifacts" / "p3_meaningful_learning_curve_20260823_v1").resolve(),
        (
            root
            / "artifacts"
            / "p3_gen6_incumbent_preserving_residual_calibrator_20260823_v1r2"
        ).resolve(),
        quarantine.resolve(),
        (root / "configs").resolve(),
    )
    for candidate in (
        paths.experiment_config,
        paths.external_scope,
        paths.catalog,
        paths.permission_receipt,
        paths.train_features,
        paths.train_anchors,
        paths.validation_keys,
        paths.incumbent_oof,
        paths.manifest,
    ):
        if not any(_inside(candidate, parent) for parent in allowed_read_roots):
            raise ContractError(f"input path is outside the frozen training/quarantine surface: {candidate}")
    if not _inside(paths.output, root / "artifacts"):
        raise ContractError("research output must remain under artifacts")
    return paths


def _validate_prereg(config: Mapping[str, Any], scope: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != "p3_era5_context_transfer_v1":
        raise ContractError("wrong experiment preregistration")
    if config.get("status") != "PREREGISTERED_BEFORE_FIRST_ERA5_VALUE_DOWNLOAD":
        raise ContractError("experiment preregistration status changed")
    source = config["source_case_builder"]
    validation = config["validation"]
    decision = config["decision_gate"]
    postprocess = config["model"]["postprocess"]
    if tuple(source["source_train_years"]) != SOURCE_TRAIN_YEARS:
        raise ContractError("source train years changed")
    if tuple(source["source_validation_years_exactly_three"]) != HELD_YEARS:
        raise ContractError("source held years changed")
    if tuple(tuple(item) for item in validation["local_outer_windows_exactly_three"]) != DEFAULT_WINDOWS:
        raise ContractError("local validation windows changed")
    if int(validation["embargo_hours"]) != 78:
        raise ContractError("local embargo changed")
    if int(validation["bootstrap_replicates"]) != BOOTSTRAP_REPLICATES:
        raise ContractError("bootstrap replicate count changed")
    if int(validation["bootstrap_seed"]) != SEED:
        raise ContractError("bootstrap seed changed")
    if float(postprocess["fixed_long_lead_persistence_weight"]) != SHRINK_WEIGHT:
        raise ContractError("fixed shrink weight changed")
    if tuple(postprocess["active_leads_h"]) != SHRINK_LEADS:
        raise ContractError("fixed shrink leads changed")
    if config["model"].get("matched_local_only_control") != MATCHED_CONTROL_CONFIG:
        raise ContractError("matched local-only two-stage control changed")
    if dict(SOURCE_CATBOOST_PARAMETERS) != dict(LOCAL_CONTROL_STAGE1_PARAMETERS):
        raise ContractError("matched control stage 1 differs from source pretraining schedule")
    if dict(LOCAL_CATBOOST_PARAMETERS) != dict(LOCAL_CONTROL_STAGE2_PARAMETERS):
        raise ContractError("matched control stage 2 differs from local continuation schedule")
    if float(decision["full_delta_candidate_minus_incumbent_m_at_most"]) != -0.03:
        raise ContractError("primary delta threshold changed")
    if int(decision["minimum_improved_local_windows"]) != 2:
        raise ContractError("window sign threshold changed")
    if float(decision["maximum_critical_slice_regression_m"]) != 0.0075:
        raise ContractError("critical-slice threshold changed")
    if int(config["features"]["expanded_feature_count"]) != EXPECTED_FEATURES:
        raise ContractError("common feature count changed")
    if scope.get("source", {}).get("source_id") != SOURCE_ID:
        raise ContractError("external source changed")
    if scope.get("status") != "PREREGISTERED_ANALYTICS_WITH_POST_SMOKE_TRANSPORT_AMENDMENT_R2":
        raise ContractError("external scope amendment status changed")
    if scope.get("bindings", {}).get("transport_amendment") != config.get("bindings", {}).get(
        "transport_amendment_after_smoke"
    ):
        raise ContractError("transport-amendment bindings differ between scope and experiment")
    if scope["time_contract"]["forbidden_from_utc"] != "2023-12-31T15:00:00+00:00":
        raise ContractError("external cutoff changed")
    if not bool(scope["safety"]["research_only"]):
        raise ContractError("research-only boundary changed")


def _load_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any], RunPaths]:
    root = root.resolve()
    experiment_path = root / "configs" / "experiments" / "p3_era5_context_transfer_v1.json"
    config = _read_json(experiment_path)
    scope_path = _bound(root, str(config["bindings"]["external_scope"]["path"]))
    scope = _read_json(scope_path)
    _validate_prereg(config, scope)
    return config, scope, _resolve_paths(root, config, scope)


def _verify_sha(path: Path, expected: str, *, label: str) -> str:
    if not path.is_file():
        raise ContractError(f"missing {label}: {path}")
    observed = _sha256(path)
    if observed != str(expected).lower():
        raise ContractError(f"{label} SHA256 mismatch")
    return observed


def _columns_sha256(columns: Sequence[str]) -> str:
    raw = json.dumps(list(columns), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_all_preregistered_bindings(
    config: Mapping[str, Any], scope: Mapping[str, Any], paths: RunPaths
) -> dict[str, str]:
    """Verify every preregistered policy, feature, and implementation binding."""

    bindings = config["bindings"]
    checks: dict[str, str] = {}
    bound_files = {
        "external_scope": (paths.external_scope, bindings["external_scope"]["sha256"]),
        "transport_amendment": (
            _bound(paths.root, bindings["transport_amendment_after_smoke"]["path"]),
            bindings["transport_amendment_after_smoke"]["sha256"],
        ),
        "external_parent_preregistration": (
            _bound(paths.root, bindings["external_parent_preregistration"]["path"]),
            bindings["external_parent_preregistration"]["sha256"],
        ),
        "feature_contract": (
            _bound(paths.root, bindings["feature_contract"]["path"]),
            bindings["feature_contract"]["sha256"],
        ),
        "external_catalog": (paths.catalog, scope["bindings"]["catalog"]["sha256"]),
        "permission_receipt": (
            paths.permission_receipt,
            scope["bindings"]["official_permission_receipt"]["sha256"],
        ),
        "permission_evidence": (
            _bound(paths.root, scope["bindings"]["official_permission_evidence"]["path"]),
            scope["bindings"]["official_permission_evidence"]["sha256"],
        ),
    }
    for label, (path, expected) in bound_files.items():
        checks[label] = _verify_sha(path, expected, label=label)

    feature_contract = _read_json(bound_files["feature_contract"][0])
    columns = tuple(str(value) for value in feature_contract.get("columns", ()))
    column_hash = _columns_sha256(columns)
    if columns != common_feature_columns() or len(columns) != EXPECTED_FEATURES:
        raise ContractError("bound feature-contract columns differ from the implementation")
    if column_hash != feature_contract.get("columns_sha256"):
        raise ContractError("feature-contract columns SHA256 mismatch")
    if column_hash != bindings["feature_contract"].get("columns_sha256"):
        raise ContractError("experiment feature-columns SHA256 mismatch")
    checks["feature_columns"] = column_hash

    implementation = bindings.get("implementation_before_first_value_access", {})
    implementation_paths = {
        "era5_pretrain_data_sha256": paths.root / "src/p3_wave/era5_pretrain_data.py",
        "era5_context_transfer_sha256": paths.root / "src/p3_wave/era5_context_transfer.py",
        "prepare_era5_runner_sha256": paths.root / "scripts/prepare_p3_era5_pretrain.py",
        "p3_validation_sha256": paths.root / "src/p3_wave/validation.py",
        "ocean_external_policy_sha256": paths.root / "src/ocean_external/policy.py",
        "context_transfer_runner_sha256": Path(__file__).resolve(),
        "context_transfer_runner_tests_sha256": (
            paths.root / "tests/test_p3_era5_context_transfer_runner.py"
        ),
    }
    if set(implementation) != set(implementation_paths):
        missing = sorted(set(implementation_paths) - set(implementation))
        extra = sorted(set(implementation) - set(implementation_paths))
        raise ContractError(f"implementation bindings changed; missing={missing}, extra={extra}")
    for key, path in implementation_paths.items():
        checks[key] = _verify_sha(path, implementation[key], label=key)
    return checks


def _read_metadata(paths: RunPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchors = pd.read_parquet(
        paths.train_anchors,
        columns=["anchor_id", "station", "anchor_time", "current_hs"],
    )
    keys = pd.read_parquet(paths.validation_keys)
    if tuple(keys.columns) != ("fold", "anchor_id", "station", "episode_id"):
        raise ContractError("validation key schema changed")
    if len(keys) != EXPECTED_CASES or keys["anchor_id"].duplicated().any():
        raise ContractError("validation universe is not 181 unique cases")
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    return anchors, keys


def _index_unique(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    indexed = frame.set_index(column)
    if not indexed.index.is_unique:
        raise ContractError(f"{column} values are not unique")
    return indexed


def _restricted_folds(anchors: pd.DataFrame, keys: pd.DataFrame) -> tuple[ForecastFold, ...]:
    """Use the existing 78h builder for prefixes, but freeze validation to sealed keys."""

    generated = build_forecast_folds(anchors, windows=DEFAULT_WINDOWS, embargo_hours=78)
    names = tuple(fold.name for fold in generated)
    if set(keys["fold"].astype(str)) != set(names):
        raise ContractError("validation keys do not contain exactly the three registered folds")
    lookup = _index_unique(anchors, "anchor_id")
    result: list[ForecastFold] = []
    for fold in generated:
        train_ids = np.asarray(fold.train_ids, dtype=np.int64)
        if not len(train_ids) or len(np.unique(train_ids)) != len(train_ids):
            raise ContractError(f"invalid causal training IDs in {fold.name}")
        if not set(train_ids.astype(int)) <= set(lookup.index.astype(int)):
            raise ContractError(f"causal training ID is absent from anchors in {fold.name}")
        train_times = lookup.loc[train_ids, "anchor_time"]
        strict_train_end = fold.validation_start - pd.Timedelta(hours=78)
        if not train_times.lt(strict_train_end).all():
            raise ContractError(
                f"training anchor is not strictly earlier than the 78h embargo in {fold.name}"
            )
        independently_expected_train = anchors.loc[
            anchors["anchor_time"].lt(strict_train_end), "anchor_id"
        ].to_numpy(dtype=np.int64)
        if set(train_ids.astype(int)) != set(independently_expected_train.astype(int)):
            raise ContractError(f"validation helper causal prefix drifted in {fold.name}")
        selected = keys.loc[keys["fold"].eq(fold.name)].copy()
        ids = selected["anchor_id"].to_numpy(dtype=np.int64)
        if not len(ids) or len(np.unique(ids)) != len(ids):
            raise ContractError(f"invalid sealed validation IDs in {fold.name}")
        if not set(ids) <= set(lookup.index.astype(int)):
            raise ContractError(f"sealed validation ID is absent from anchors in {fold.name}")
        metadata = lookup.loc[ids]
        if not metadata["anchor_time"].ge(fold.validation_start).all() or not metadata[
            "anchor_time"
        ].lt(fold.validation_end).all():
            raise ContractError(f"sealed validation ID is outside {fold.name}")
        station_by_id = metadata["station"].astype(str)
        expected_station = selected.set_index("anchor_id")["station"].astype(str).loc[ids]
        if not np.array_equal(station_by_id.to_numpy(), expected_station.to_numpy()):
            raise ContractError(f"station binding changed in {fold.name}")
        for _, group in metadata.reset_index().groupby("station", observed=True):
            gaps = group.sort_values("anchor_time")["anchor_time"].diff().dropna()
            if not gaps.ge(pd.Timedelta(hours=78)).all():
                raise ContractError(f"sealed validation IDs violate the 78h gap in {fold.name}")
        if np.intersect1d(fold.train_ids, ids).size:
            raise ContractError(f"local train/validation overlap in {fold.name}")
        result.append(replace(fold, validation_ids=ids))
    if sum(len(fold.validation_ids) for fold in result) != EXPECTED_CASES:
        raise ContractError("restricted folds do not contain exactly 181 cases")
    return tuple(result)


def _load_blind_comparator(path: Path, keys: pd.DataFrame) -> pd.DataFrame:
    """Load predictor/key columns only; this function cannot decode the outcome column."""

    frame = pd.read_parquet(path, columns=list(BLIND_COMPARATOR_COLUMNS))
    frame = frame.loc[frame["prefix_fraction"].eq(1.0)].drop(columns="prefix_fraction")
    if len(frame) != EXPECTED_ROWS or frame.duplicated(list(KEY_COLUMNS)).any():
        raise ContractError("full-prefix comparator is not 1086 unique rows")
    if frame["anchor_id"].nunique() != EXPECTED_CASES:
        raise ContractError("full-prefix comparator is not 181 unique cases")
    if set(frame["lead_h"].astype(int)) != set(LEADS):
        raise ContractError("comparator lead set changed")
    case_keys = frame.loc[:, ["fold", "anchor_id", "station"]].drop_duplicates()
    expected = keys.loc[:, ["fold", "anchor_id", "station"]]
    joined = case_keys.merge(expected, how="outer", indicator=True)
    if not joined["_merge"].eq("both").all() or len(joined) != EXPECTED_CASES:
        raise ContractError("comparator rows differ from the sealed validation keys")
    return frame.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)


def _read_local_features(path: Path) -> pd.DataFrame:
    columns = ("anchor_id", *common_feature_columns())
    schema = tuple(pq.ParquetFile(path).schema_arrow.names)
    if len(common_feature_columns()) != EXPECTED_FEATURES or not set(columns) <= set(schema):
        raise ContractError("frozen local cache does not contain the exact 286 features")
    frame = pd.read_parquet(path, columns=list(columns))
    selected = select_common_cached_features(frame.drop(columns="anchor_id"), require_all=True)
    if tuple(selected.columns) != common_feature_columns():
        raise ContractError("local common-feature order changed")
    if frame["anchor_id"].duplicated().any():
        raise ContractError("local feature IDs are not unique")
    return frame


def _read_training_targets(path: Path, anchor_ids: Sequence[int]) -> pd.DataFrame:
    """Predicate-push down a fold's causal prefix; held rows are not decoded."""

    ids = np.asarray(anchor_ids, dtype=np.int64)
    if not len(ids) or len(np.unique(ids)) != len(ids):
        raise ContractError("local training IDs must be non-empty and unique")
    table = ds.dataset(path, format="parquet").to_table(
        columns=["anchor_id", "current_hs", *TARGET_COLUMNS],
        filter=ds.field("anchor_id").isin(ids.tolist()),
    )
    frame = _index_unique(table.to_pandas(), "anchor_id").loc[ids].reset_index()
    if len(frame) != len(ids):
        raise ContractError("predicate-pushed local target read lost IDs")
    values = frame.loc[:, ["current_hs", *TARGET_COLUMNS]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ContractError("local training target values are invalid")
    return frame


def _log_delta_targets(frame: pd.DataFrame) -> np.ndarray:
    current = frame["current_hs"].to_numpy(dtype=np.float64)
    future = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    return np.log1p(future) - np.log1p(current)[:, None]


def _apply_fixed_shrink(raw: np.ndarray, current: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != len(LEADS) or current.shape != (len(raw),):
        raise ContractError("fixed-shrink inputs have wrong shape")
    result = raw.copy()
    for position, lead in enumerate(LEADS):
        if lead in SHRINK_LEADS:
            result[:, position] = (1.0 - SHRINK_WEIGHT) * raw[:, position] + (
                SHRINK_WEIGHT * current
            )
    return result


def _predict_source_with_fixed_postprocess(
    model: Any, features: pd.DataFrame, current_hs: np.ndarray
) -> np.ndarray:
    """Apply the preregistered model-level postprocess to source held years too."""

    raw = model.predict_hs(features, current_hs=current_hs)
    return _apply_fixed_shrink(raw, current_hs)


class _LocalOnlyControl:
    """Matched two-stage schedule using local data in place of source pretraining."""

    def __init__(self) -> None:
        self._stage1_model: Any | None = None
        self._model: Any | None = None

    @staticmethod
    def _long(features: pd.DataFrame) -> pd.DataFrame:
        if tuple(features.columns) != common_feature_columns():
            raise ContractError("local-only control feature surface changed")
        long = features.loc[features.index.repeat(len(LEADS))].reset_index(drop=True)
        long["lead_h"] = np.tile(np.asarray(LEADS, dtype=np.float64), len(features))
        return long

    def fit(
        self,
        features: pd.DataFrame,
        targets: np.ndarray,
        *,
        current_hs: np.ndarray,
    ) -> _LocalOnlyControl:
        from catboost import CatBoostRegressor

        current = np.asarray(current_hs, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        if target.shape != (len(features), len(LEADS)) or current.shape != (len(features),):
            raise ContractError("local-only control fit arrays changed shape")
        if not np.isfinite(target).all() or not np.isfinite(current).all() or (current < 0).any():
            raise ContractError("local-only control fit arrays contain invalid values")
        long = self._long(features)
        flattened_target = target.reshape(-1)
        stage1 = CatBoostRegressor(**dict(LOCAL_CONTROL_STAGE1_PARAMETERS))
        # This is the matched counterfactual to source pretraining: same first-stage
        # schedule, but the causal local prefix is the only training domain.
        stage1.fit(long, flattened_target, verbose=False)
        try:
            stage1_init = copy.deepcopy(stage1)
        except Exception as error:  # pragma: no cover - backend-specific defensive path
            raise ContractError("local-only stage-1 model could not be cloned") from error
        if stage1_init is stage1:
            raise ContractError("local-only stage-1 clone unexpectedly shares identity")
        weights = np.exp(-0.45 * np.maximum(current - 1.5, 0.0))
        stage2 = CatBoostRegressor(**dict(LOCAL_CONTROL_STAGE2_PARAMETERS))
        stage2.fit(
            long,
            flattened_target,
            sample_weight=np.repeat(weights, len(LEADS)),
            init_model=stage1_init,
            verbose=False,
        )
        self._stage1_model = stage1
        self._model = stage2
        return self

    def predict_hs(self, features: pd.DataFrame, *, current_hs: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ContractError("local-only control is not fitted")
        current = np.asarray(current_hs, dtype=np.float64)
        if current.shape != (len(features),) or not np.isfinite(current).all() or (current < 0).any():
            raise ContractError("local-only control current values are invalid")
        prediction = np.asarray(self._model.predict(self._long(features)), dtype=np.float64)
        if prediction.shape != (len(features) * len(LEADS),):
            raise ContractError("local-only control prediction shape changed")
        log_delta = prediction.reshape(len(features), len(LEADS))
        forecast = np.expm1(np.log1p(current)[:, None] + log_delta)
        return np.clip(forecast, 0.0, 30.0)


def _produce_local_blind_predictions(
    *,
    pretrained: Any,
    folds: Sequence[ForecastFold],
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    keys: pd.DataFrame,
    target_reader: Callable[[Sequence[int]], pd.DataFrame],
    control_factory: Callable[[], Any] = _LocalOnlyControl,
) -> pd.DataFrame:
    feature_lookup = _index_unique(features, "anchor_id")
    anchor_lookup = _index_unique(anchors, "anchor_id")
    blocks: list[pd.DataFrame] = []
    clones: list[Any] = []
    for fold in folds:
        validation_ids = np.asarray(fold.validation_ids, dtype=np.int64)
        if np.intersect1d(fold.train_ids, validation_ids).size:
            raise ContractError(f"target exclusion failed in {fold.name}")
        local_targets = target_reader(fold.train_ids)
        if set(local_targets["anchor_id"].astype(int)) & set(validation_ids.astype(int)):
            raise ContractError(f"held targets entered local continuation in {fold.name}")
        train_ids = local_targets["anchor_id"].to_numpy(dtype=np.int64)
        expected_train_current = anchor_lookup.loc[train_ids, "current_hs"].to_numpy(
            dtype=np.float64
        )
        if not np.allclose(
            local_targets["current_hs"].to_numpy(dtype=np.float64),
            expected_train_current,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ContractError(f"local training current value drifted in {fold.name}")
        train_x = feature_lookup.loc[train_ids, list(common_feature_columns())].reset_index(drop=True)
        validation_x = feature_lookup.loc[
            validation_ids, list(common_feature_columns())
        ].reset_index(drop=True)
        model = pretrained.clone_pretrained()
        if model is pretrained or any(model is previous for previous in clones):
            raise ContractError("fold continuation did not receive an independent clone")
        clones.append(model)
        model.continue_local(
            train_x,
            _log_delta_targets(local_targets),
            current_hs=local_targets["current_hs"].to_numpy(dtype=np.float64),
        )
        current = anchor_lookup.loc[validation_ids, "current_hs"].to_numpy(dtype=np.float64)
        raw = model.predict_hs(validation_x, current_hs=current)
        prediction = _apply_fixed_shrink(raw, current)
        local_control = control_factory().fit(
            train_x,
            _log_delta_targets(local_targets),
            current_hs=local_targets["current_hs"].to_numpy(dtype=np.float64),
        )
        control_raw = local_control.predict_hs(validation_x, current_hs=current)
        control_prediction = _apply_fixed_shrink(control_raw, current)
        fold_keys = keys.loc[keys["fold"].eq(fold.name)].set_index("anchor_id")
        stations = anchor_lookup.loc[validation_ids, "station"].astype(str).to_numpy()
        for row_position, anchor_id in enumerate(validation_ids):
            expected_station = str(fold_keys.loc[anchor_id, "station"])
            if stations[row_position] != expected_station:
                raise ContractError("fold station binding changed")
            for lead_position, lead in enumerate(LEADS):
                blocks.append(
                    pd.DataFrame(
                        {
                            "fold": [fold.name],
                            "anchor_id": [int(anchor_id)],
                            "station": [stations[row_position]],
                            "lead_h": [int(lead)],
                            "model_current_hs": [float(current[row_position])],
                            "transfer_prediction": [float(prediction[row_position, lead_position])],
                            "local_control_prediction": [
                                float(control_prediction[row_position, lead_position])
                            ],
                        }
                    )
                )
    blind = pd.concat(blocks, ignore_index=True)
    if len(blind) != EXPECTED_ROWS or blind.duplicated(list(KEY_COLUMNS)).any():
        raise ContractError("three-fold blind prediction set is incomplete")
    return blind


@contextmanager
def _repository_cwd(root: Path) -> Iterable[None]:
    previous = Path.cwd()
    os.chdir(root)
    try:
        yield
    finally:
        os.chdir(previous)


def _external_preflight(paths: RunPaths) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    if not paths.manifest.is_file():
        raise ContractError("external quarantine manifest is missing")
    manifest = _read_json(paths.manifest)
    local_value = Path(str(manifest.get("local_file", "")))
    candidate = local_value if local_value.is_absolute() else paths.root / local_value
    candidate = candidate.resolve()
    if not _inside(candidate, paths.quarantine) or candidate.name != COMBINED_NAME:
        raise ContractError("manifest candidate is outside the fixed ERA5 quarantine")
    with _repository_cwd(paths.root):
        receipt = preflight_external_use(
            catalog_path=paths.catalog,
            approval_receipt_path=paths.permission_receipt,
            manifest_path=paths.manifest,
            problem="P3",
            source_id=SOURCE_ID,
            purpose=SOURCE_PURPOSE,
        )
    return receipt, candidate, manifest


def _load_source_hourly(paths: RunPaths) -> tuple[pd.DataFrame, dict[str, Any]]:
    receipt, candidate, manifest = _external_preflight(paths)
    layout = era5_data.QuarantineLayout(paths.quarantine)
    canonical_candidate = layout.assert_inside(layout.derived / COMBINED_NAME)
    if candidate != canonical_candidate:
        raise ContractError("generic preflight candidate differs from the canonical combined file")
    era5_data._validate_completed_manifest_payload(layout, manifest)
    selections = era5_data.read_selected_cells(layout)
    expected_selected_cells = [
        selections[name].public_dict() for name in era5_data.STATIONS
    ]
    if manifest.get("selected_cells") != expected_selected_cells:
        raise ContractError(
            "manifest selected cells differ from the frozen smoke-selection receipt"
        )
    frame, summary = era5_data.load_validated_combined_file(layout, selections)
    if int(manifest["row_count"]) != len(frame):
        raise ContractError("ERA5 manifest row count differs from the combined file")
    if (
        summary["row_count"] != manifest["row_count"]
        or summary["observed_start"] != manifest["observed_start"]
        or summary["observed_end"] != manifest["observed_end"]
    ):
        raise ContractError("validated ERA5 coverage differs from its manifest")
    renamed = frame.rename(columns={"time_utc": "time", **EXTERNAL_COLUMN_MAP})
    renamed["time"] = pd.to_datetime(renamed["time"], utc=True, errors="raise")
    if renamed[["station", "time"]].duplicated().any():
        raise ContractError("combined ERA5 station-hours are not unique")
    if renamed["time"].max() >= pd.Timestamp("2023-12-31T15:00:00Z"):
        raise ContractError("combined ERA5 values crossed the fixed cutoff")
    dynamic_columns = tuple(
        column
        for column in era5_data.DERIVED_COLUMNS
        if column not in {"station", "time_utc", "latitude", "longitude", "land_sea_mask"}
    )
    finite = {
        column: float(
            np.isfinite(pd.to_numeric(frame[column], errors="coerce")).mean()
        )
        for column in dynamic_columns
    }
    return renamed.loc[:, ["station", "time", *EXTERNAL_COLUMN_MAP.values()]], {
        "generic_preflight": receipt,
        "manifest_sha256": _sha256(paths.manifest),
        "combined_sha256": _sha256(candidate),
        "row_count": int(len(renamed)),
        "finite_fraction": finite,
        "complete_hourly_coverage": True,
        "physical_hard_bounds_validated": True,
        "fixed_selected_cells_validated": True,
        "combined_validation_summary": summary,
    }


def _source_split(cases: Any) -> tuple[np.ndarray, pd.DataFrame]:
    anchors = cases.anchors.copy()
    times = pd.to_datetime(anchors["anchor_time"], utc=True, errors="raise")
    years = times.dt.year
    complete_year = (
        (times - pd.Timedelta(hours=48)).dt.year.eq(years)
        & (times + pd.Timedelta(hours=24)).dt.year.eq(years)
    )
    train_ids = anchors.loc[complete_year & years.isin(SOURCE_TRAIN_YEARS), "anchor_id"].to_numpy(
        dtype=np.int64
    )
    held = select_source_year_validation(
        anchors,
        held_years=HELD_YEARS,
        station_column="station",
    )
    if not len(train_ids) or set(train_ids.astype(int)) & set(held["anchor_id"].astype(int)):
        raise ContractError("source train/held split failed")
    if set(held["year"].astype(int)) != set(HELD_YEARS):
        raise ContractError("source split lost a held year")
    return train_ids, held


def _source_prediction_rows(cases: Any, held: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    anchors = _index_unique(cases.anchors, "anchor_id")
    if prediction.shape != (len(held), len(LEADS)):
        raise ContractError("source prediction shape changed")
    rows: list[dict[str, Any]] = []
    for position, held_row in enumerate(held.itertuples(index=False)):
        anchor = anchors.loc[int(held_row.anchor_id)]
        for lead_position, lead in enumerate(LEADS):
            rows.append(
                {
                    "year": int(held_row.year),
                    "episode_id": int(held_row.episode_id),
                    "anchor_id": int(held_row.anchor_id),
                    "station": str(held_row.station),
                    "lead_h": int(lead),
                    "persistence": float(anchor["current_hs"]),
                    "transfer_prediction": float(prediction[position, lead_position]),
                    "source_future_hs": float(anchor[f"future_hs_{lead}h"]),
                }
            )
    return pd.DataFrame(rows)


def _rmse(values: Sequence[float], prediction: Sequence[float]) -> float:
    truth = np.asarray(values, dtype=np.float64)
    forecast = np.asarray(prediction, dtype=np.float64)
    if truth.shape != forecast.shape or not len(truth):
        raise ContractError("RMSE arrays are empty or misaligned")
    return float(np.sqrt(np.mean(np.square(forecast - truth))))


def _metric_pair(
    frame: pd.DataFrame, baseline: str, *, truth_column: str = "target_hs"
) -> dict[str, float | int]:
    candidate_rmse = _rmse(frame[truth_column], frame["transfer_prediction"])
    baseline_rmse = _rmse(frame[truth_column], frame[baseline])
    return {
        "rows": int(len(frame)),
        "candidate_rmse_m": candidate_rmse,
        "baseline_rmse_m": baseline_rmse,
        "delta_m": candidate_rmse - baseline_rmse,
    }


def _paired_episode_bootstrap(
    frame: pd.DataFrame,
    *,
    baseline: str,
    truth_column: str = "target_hs",
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = SEED,
) -> dict[str, Any]:
    required = {"episode_id", truth_column, "transfer_prediction", baseline}
    if not required <= set(frame.columns):
        raise ContractError("bootstrap frame is missing required columns")
    grouped = []
    for episode, block in frame.groupby("episode_id", sort=True, observed=True):
        truth = block[truth_column].to_numpy(dtype=np.float64)
        candidate = block["transfer_prediction"].to_numpy(dtype=np.float64)
        base = block[baseline].to_numpy(dtype=np.float64)
        grouped.append(
            (
                episode,
                float(np.square(candidate - truth).sum()),
                float(np.square(base - truth).sum()),
                int(len(block)),
            )
        )
    if not grouped:
        raise ContractError("bootstrap has no episodes")
    candidate_sse = np.asarray([item[1] for item in grouped], dtype=np.float64)
    baseline_sse = np.asarray([item[2] for item in grouped], dtype=np.float64)
    counts = np.asarray([item[3] for item in grouped], dtype=np.float64)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for draw in range(replicates):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        denominator = counts[selected].sum()
        deltas[draw] = np.sqrt(candidate_sse[selected].sum() / denominator) - np.sqrt(
            baseline_sse[selected].sum() / denominator
        )
    return {
        "unit": "episode_id",
        "episodes": int(len(grouped)),
        "replicates": int(replicates),
        "seed": int(seed),
        "observed_delta_m": _metric_pair(
            frame, baseline, truth_column=truth_column
        )["delta_m"],
        "ci90_lower_m": float(np.quantile(deltas, 0.05)),
        "ci90_upper_m": float(np.quantile(deltas, 0.95)),
    }


def _source_gate(rows: pd.DataFrame, finite_fraction: Mapping[str, float]) -> dict[str, Any]:
    by_year = {
        str(year): _metric_pair(
            rows.loc[rows["year"].eq(year)],
            "persistence",
            truth_column="source_future_hs",
        )
        for year in HELD_YEARS
    }
    by_lead = {
        str(lead): _metric_pair(
            rows.loc[rows["lead_h"].eq(lead)],
            "persistence",
            truth_column="source_future_hs",
        )
        for lead in LEADS
    }
    bootstrap = _paired_episode_bootstrap(
        rows,
        baseline="persistence",
        truth_column="source_future_hs",
    )
    checks = {
        "finite_coverage_at_least_0_995": bool(
            finite_fraction and min(finite_fraction.values()) >= 0.995
        ),
        "all_three_years_better_than_persistence": all(
            by_year[str(year)]["delta_m"] < 0.0 for year in HELD_YEARS
        ),
        "pooled_episode_ci90_upper_below_zero": bootstrap["ci90_upper_m"] < 0.0,
        "lead_18_non_degrade": by_lead["18"]["delta_m"] <= 0.0,
        "lead_24_non_degrade": by_lead["24"]["delta_m"] <= 0.0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "finite_fraction": dict(finite_fraction),
        "pooled": _metric_pair(rows, "persistence", truth_column="source_future_hs"),
        "by_year": by_year,
        "by_lead": by_lead,
        "paired_episode_bootstrap": bootstrap,
    }


def _domain_classifier_auc(source: pd.DataFrame, local: pd.DataFrame) -> float:
    """Fixed diagnostic only; it never changes or gates the transfer continuation."""

    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    columns = list(common_feature_columns())
    x = pd.concat([source.loc[:, columns], local.loc[:, columns]], ignore_index=True)
    y = np.concatenate(
        [np.zeros(len(source), dtype=np.int8), np.ones(len(local), dtype=np.int8)]
    )
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = cross_val_predict(model, x, y, cv=folds, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, probability))


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _atomic_text(text: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256(path)


def _atomic_json(value: Mapping[str, Any], path: Path) -> str:
    return _atomic_text(json.dumps(value, indent=2, sort_keys=True) + "\n", path)


def _create_attempt_lock(path: Path, config_sha256: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "p3_era5_context_transfer_v1.attempt.v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_sha256": config_sha256,
        "research_only": True,
        "operational_prediction_allowed": False,
        "upload_allowed": False,
        "pid": os.getpid(),
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("attempt-lock write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.read_bytes() != raw:
        raise ContractError("attempt lock changed after exclusive write")
    return hashlib.sha256(raw).hexdigest()


def _attach_truth_after_seal(
    blind: pd.DataFrame,
    *,
    seal_path: Path,
    seal_sha256: str,
    truth_path: Path,
) -> pd.DataFrame:
    """The only OOF outcome reader; it is unreachable until the blind seal exists."""

    if not seal_path.is_file() or _sha256(seal_path) != seal_sha256:
        raise ContractError("blind predictions were not durably sealed before scoring")
    truth = pd.read_parquet(truth_path, columns=list(TRUTH_COLUMNS))
    truth = truth.loc[truth["prefix_fraction"].eq(1.0)].drop(columns="prefix_fraction")
    if len(truth) != EXPECTED_ROWS or truth.duplicated(list(KEY_COLUMNS)).any():
        raise ContractError("full-prefix truth rows changed")
    evaluated = blind.merge(truth, on=list(KEY_COLUMNS), how="outer", validate="one_to_one", indicator=True)
    if not evaluated["_merge"].eq("both").all() or len(evaluated) != EXPECTED_ROWS:
        raise ContractError("blind prediction/truth key alignment failed")
    return evaluated.drop(columns="_merge")


def _local_gate(evaluated: pd.DataFrame) -> dict[str, Any]:
    overall = _metric_pair(evaluated, "incumbent_prediction")
    by_fold = {
        str(name): _metric_pair(block, "incumbent_prediction")
        for name, block in evaluated.groupby("fold", sort=True, observed=True)
    }
    bootstrap = _paired_episode_bootstrap(evaluated, baseline="incumbent_prediction")
    slices: dict[str, dict[str, float | int]] = {}
    for station in ("G-ORS", "I-ORS", "S-ORS"):
        slices[station] = _metric_pair(
            evaluated.loc[evaluated["station"].eq(station)], "incumbent_prediction"
        )
    slices["winter"] = _metric_pair(
        evaluated.loc[evaluated["fold"].eq("winter_transition")], "incumbent_prediction"
    )
    for lead in SHRINK_LEADS:
        slices[f"lead_{lead}"] = _metric_pair(
            evaluated.loc[evaluated["lead_h"].eq(lead)], "incumbent_prediction"
        )
    max_slice = max(float(value["delta_m"]) for value in slices.values())
    checks = {
        "full_delta_at_most_minus_0_03": float(overall["delta_m"]) <= -0.03,
        "at_least_two_of_three_windows_improve": sum(
            float(item["delta_m"]) < 0.0 for item in by_fold.values()
        )
        >= 2,
        "paired_episode_ci90_upper_below_zero": float(bootstrap["ci90_upper_m"]) < 0.0,
        "critical_slice_regression_at_most_0_0075": max_slice <= 0.0075,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "overall": overall,
        "by_fold": by_fold,
        "critical_slices": slices,
        "maximum_critical_slice_delta_m": max_slice,
        "paired_episode_bootstrap": bootstrap,
    }


def _viewpoint_gate(evaluated: pd.DataFrame) -> dict[str, Any]:
    """Require the source-initialized model to beat its matched local-only control."""

    overall = _metric_pair(evaluated, "local_control_prediction")
    by_fold = {
        str(name): _metric_pair(block, "local_control_prediction")
        for name, block in evaluated.groupby("fold", sort=True, observed=True)
    }
    bootstrap = _paired_episode_bootstrap(evaluated, baseline="local_control_prediction")
    checks = {
        "pooled_transfer_better_than_local_control": float(overall["delta_m"]) < 0.0,
        "at_least_two_of_three_windows_improve": sum(
            float(item["delta_m"]) < 0.0 for item in by_fold.values()
        )
        >= 2,
        "paired_episode_ci90_upper_below_zero": float(bootstrap["ci90_upper_m"]) < 0.0,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "comparison": "transfer_minus_matched_local_only_control",
        "control_parameters": {
            "stage1_unweighted_local": dict(LOCAL_CONTROL_STAGE1_PARAMETERS),
            "stage2_weighted_local": dict(LOCAL_CONTROL_STAGE2_PARAMETERS),
        },
        "only_stage1_training_domain_differs_from_transfer": True,
        "same_features_targets_schedule_seed_and_fixed_shrink": True,
        "overall": overall,
        "by_fold": by_fold,
        "paired_episode_bootstrap": bootstrap,
    }


def check_only(root: Path = ROOT) -> dict[str, Any]:
    """Read only frozen metadata/schema surfaces and create no files."""

    config, scope, paths = _load_contract(root)
    bindings = config["bindings"]
    hashes = {
        "train_features": _verify_sha(
            paths.train_features, bindings["train_features"]["sha256"], label="train features"
        ),
        "train_anchors": _verify_sha(
            paths.train_anchors, bindings["train_anchors"]["sha256"], label="train anchors"
        ),
        "validation_keys": _verify_sha(
            paths.validation_keys,
            bindings["validation_keys"]["sha256"],
            label="validation keys",
        ),
        "incumbent_oof": _verify_sha(
            paths.incumbent_oof,
            bindings["frozen_incumbent_oof"]["sha256"],
            label="incumbent OOF",
        ),
        "experiment_config": _sha256(paths.experiment_config),
        "external_scope": _sha256(paths.external_scope),
    }
    hashes.update(_verify_all_preregistered_bindings(config, scope, paths))
    feature_schema = tuple(pq.ParquetFile(paths.train_features).schema_arrow.names)
    if not set(common_feature_columns()) <= set(feature_schema):
        raise ContractError("frozen train cache is missing a common feature")
    anchors, keys = _read_metadata(paths)
    folds = _restricted_folds(anchors, keys)
    comparator = _load_blind_comparator(paths.incumbent_oof, keys)
    manifest_preview = _read_json(paths.manifest) if paths.manifest.is_file() else {}
    preview_value = manifest_preview.get("local_file")
    preview_path = (
        Path(str(preview_value))
        if preview_value
        else paths.quarantine / "__incomplete_external_value__"
    )
    if not preview_path.is_absolute():
        preview_path = paths.root / preview_path
    source_ready = bool(
        preview_value
        and int(manifest_preview.get("row_count", 0)) > 0
        and preview_path.is_file()
    )
    source_preflight: dict[str, Any] | None = None
    if source_ready:
        source_preflight, _, _ = _external_preflight(paths)
    return {
        "mode": "check-only",
        "passed": True,
        "writes": 0,
        "model_fits": 0,
        "outcome_values_read": 0,
        "hashes": hashes,
        "common_feature_count": len(common_feature_columns()),
        "local_anchor_rows": int(len(anchors)),
        "validation_cases": int(len(keys)),
        "full_prefix_rows": int(len(comparator)),
        "folds": {
            fold.name: {
                "train_cases": int(len(fold.train_ids)),
                "validation_cases": int(len(fold.validation_ids)),
                "embargo_hours": 78,
            }
            for fold in folds
        },
        "source_quarantine_ready": source_ready,
        "source_preflight": source_preflight,
        "research_only": True,
    }


def _write_readme(result: Mapping[str, Any], path: Path) -> None:
    local = result.get("local_gate")
    lines = [
        "# P3 ERA5 context transfer v1",
        "",
        "Research-only one-shot historical validation. No operational inference was performed.",
        "",
        f"- Status: `{result['status']}`",
        f"- Source gate: `{result['source_gate']['passed']}`",
    ]
    if isinstance(local, Mapping):
        lines.extend(
            [
                f"- Local gate: `{local['passed']}`",
                f"- Local RMSE delta: `{local['overall']['delta_m']:.6f} m`",
                (
                    "- Paired episode bootstrap 90% CI: "
                    f"`[{local['paired_episode_bootstrap']['ci90_lower_m']:.6f}, "
                    f"{local['paired_episode_bootstrap']['ci90_upper_m']:.6f}] m`"
                ),
            ]
        )
    viewpoint = result.get("viewpoint_signal_gate")
    if isinstance(viewpoint, Mapping):
        lines.extend(
            [
                f"- Matched local-only viewpoint gate: `{viewpoint['passed']}`",
                f"- Transfer minus local-only RMSE: `{viewpoint['overall']['delta_m']:.6f} m`",
            ]
        )
    lines.extend(
        [
            "",
            "The source model was fit once. Each local fold used an independent deep clone and only its causal prefix.",
            "All 1,086 historical blind predictions were atomically sealed and hashed before the OOF outcome column was opened.",
            "The result remains a research decision only.",
            "",
        ]
    )
    _atomic_text("\n".join(lines), path)


def execute_once(root: Path = ROOT) -> dict[str, Any]:
    config, scope, paths = _load_contract(root)
    if paths.output.exists():
        raise FileExistsError("canonical research output already exists")
    if paths.attempt_lock.exists():
        raise FileExistsError("one-shot attempt was already consumed")
    # Static checks and all frozen hashes are verified before the irreversible lock.
    check = check_only(root)
    lock_sha = _create_attempt_lock(paths.attempt_lock, _sha256(paths.experiment_config))
    paths.output.mkdir(parents=False, exist_ok=False)

    source_hourly, source_provenance = _load_source_hourly(paths)
    source_cases = build_source_cases(source_hourly, time_column="time", group_column="station")
    source_train_ids, source_held = _source_split(source_cases)
    position = pd.Series(
        np.arange(len(source_cases.anchors), dtype=np.int64),
        index=source_cases.anchors["anchor_id"].to_numpy(dtype=np.int64),
    )
    train_positions = position.loc[source_train_ids].to_numpy(dtype=np.int64)
    held_positions = position.loc[source_held["anchor_id"].to_numpy(dtype=np.int64)].to_numpy(
        dtype=np.int64
    )
    pretrained = FixedContextTransferRegressor().fit_pretrain(
        source_cases.features.iloc[train_positions].reset_index(drop=True),
        source_cases.log_delta_targets[train_positions],
    )
    source_prediction = _predict_source_with_fixed_postprocess(
        pretrained,
        source_cases.features.iloc[held_positions].reset_index(drop=True),
        source_cases.current_hs[held_positions],
    )
    source_rows = _source_prediction_rows(source_cases, source_held, source_prediction)
    source_gate = _source_gate(source_rows, source_provenance["finite_fraction"])

    anchors, keys = _read_metadata(paths)
    local_features = _read_local_features(paths.train_features)
    source_auc = _domain_classifier_auc(
        source_cases.features.iloc[train_positions].reset_index(drop=True),
        local_features.loc[:, common_feature_columns()],
    )
    base_result: dict[str, Any] = {
        "schema_version": "p3_era5_context_transfer_v1.result.v1",
        "experiment_id": config["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "research_only": True,
        "attempt_lock_sha256": lock_sha,
        "check_only_preflight": check,
        "external_scope_sha256": _sha256(paths.external_scope),
        "source_provenance": source_provenance,
        "source_split": {
            "train_years": list(SOURCE_TRAIN_YEARS),
            "held_years": list(HELD_YEARS),
            "train_cases": int(len(source_train_ids)),
            "held_cases": int(len(source_held)),
            "held_cases_by_year": {
                str(year): int(source_held["year"].eq(year).sum()) for year in HELD_YEARS
            },
            "complete_footprints_only": True,
            "station_gap_hours": 78,
        },
        "source_gate": source_gate,
        "domain_classifier_auc": source_auc,
        "domain_classifier_restriction": (
            "Direct row pooling and pretrain-only promotion remain prohibited"
            if source_auc >= 0.98
            else "Diagnostic only; direct row pooling remains preregistration-prohibited"
        ),
    }
    if not source_gate["passed"]:
        base_result["status"] = "NO_GO_SOURCE_GATE"
        base_result["local_gate"] = None
        base_result["viewpoint_signal_gate"] = None
        _atomic_json(base_result, paths.output / "result.json")
        _write_readme(base_result, paths.output / "README.md")
        return base_result

    folds = _restricted_folds(anchors, keys)
    comparator = _load_blind_comparator(paths.incumbent_oof, keys)
    transfer = _produce_local_blind_predictions(
        pretrained=pretrained,
        folds=folds,
        features=local_features,
        anchors=anchors,
        keys=keys,
        target_reader=lambda ids: _read_training_targets(paths.train_anchors, ids),
    )
    blind = comparator.merge(
        transfer,
        on=list(KEY_COLUMNS),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not blind["_merge"].eq("both").all() or len(blind) != EXPECTED_ROWS:
        raise ContractError("blind transfer/comparator alignment failed")
    if not np.allclose(
        blind["current_hs"].to_numpy(dtype=np.float64),
        blind["model_current_hs"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ContractError("comparator current values differ from frozen anchor metadata")
    blind = blind.drop(columns=["_merge", "model_current_hs"]).merge(
        keys.loc[:, ["fold", "anchor_id", "episode_id"]],
        on=["fold", "anchor_id"],
        how="left",
        validate="many_to_one",
    )
    if blind["episode_id"].isna().any():
        raise ContractError("episode IDs did not attach to every blind row")
    blind = blind.sort_values(list(KEY_COLUMNS), kind="mergesort").reset_index(drop=True)
    if "target_hs" in blind.columns:
        raise ContractError("outcome column entered the blind prediction seal")
    seal_path = paths.output / "sealed_historical_blind_predictions.parquet"
    seal_sha = _atomic_parquet(blind, seal_path)
    evaluated = _attach_truth_after_seal(
        blind,
        seal_path=seal_path,
        seal_sha256=seal_sha,
        truth_path=paths.incumbent_oof,
    )
    local_gate = _local_gate(evaluated)
    viewpoint_gate = _viewpoint_gate(evaluated)
    passed = bool(local_gate["passed"] and viewpoint_gate["passed"])
    base_result.update(
        {
            "status": "RESEARCH_GATE_PASS" if passed else "NO_GO_LOCAL_OR_VIEWPOINT_GATE",
            "blind_seal": {
                "path": str(seal_path.relative_to(paths.root)).replace("\\", "/"),
                "sha256": seal_sha,
                "rows": int(len(blind)),
                "outcome_column_present": False,
                "sealed_before_outcome_read": True,
            },
            "local_split": {
                fold.name: {
                    "train_cases": int(len(fold.train_ids)),
                    "validation_cases": int(len(fold.validation_ids)),
                    "held_targets_excluded": True,
                    "independent_pretrained_clone": True,
                }
                for fold in folds
            },
            "local_gate": local_gate,
            "viewpoint_signal_gate": viewpoint_gate,
            "promotion": "research candidate only" if passed else "no-go",
        }
    )
    _atomic_json(base_result, paths.output / "result.json")
    _write_readme(base_result, paths.output / "README.md")
    return base_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute:
        result = execute_once(args.root)
    else:
        result = check_only(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
