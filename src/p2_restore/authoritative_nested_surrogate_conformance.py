"""Implementation adapters for the sealed P2 nested-surrogate contract.

This module owns only split, masking, OOF-ledger, seed, and meta-refit
semantics.  It intentionally does not launch the registered 45-cell fit.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from p2_restore.deep_data import CADENCE_MINUTES, P2Panel
from p2_restore.deep_training import train_fold, train_full_model
from p2_restore.model import fit_model
from p2_restore.regime_gate import STATE_FEATURES, fit_soft_gate

TARGET_LAYERS = (2, 3, 4)
COMPONENTS = (
    "router_400",
    "depth_query_bitcn",
    "moment_units_scratch",
    "lsti_style",
    "timemixerpp_style",
)
DEEP_COMPONENTS = COMPONENTS[1:]
KEY_COLUMNS = ("inner_fold", "station", "layer", "time")
SEED_NAMESPACE = "P2_AUTH_NESTED_V1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_index(values: Sequence[Any] | pd.Series | pd.Index) -> pd.DatetimeIndex:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    return pd.DatetimeIndex(parsed)


def _timestamp_iso(value: pd.Timestamp) -> str:
    return value.tz_convert("Asia/Seoul").isoformat()


@dataclass(frozen=True)
class InnerFoldPlan:
    inner_fold: str
    train_times: pd.DatetimeIndex
    validation_times: pd.DatetimeIndex
    validation_start_utc: pd.Timestamp
    validation_stop_utc: pd.Timestamp
    embargo_threshold_utc: pd.Timestamp

    def summary(self) -> dict[str, Any]:
        return {
            "inner_fold": self.inner_fold,
            "train_time_count": len(self.train_times),
            "validation_time_count": len(self.validation_times),
            "train_first_kst": _timestamp_iso(self.train_times[0]),
            "train_last_kst": _timestamp_iso(self.train_times[-1]),
            "validation_start_kst": _timestamp_iso(self.validation_start_utc),
            "validation_stop_exclusive_kst": _timestamp_iso(self.validation_stop_utc),
            "embargo_threshold_kst": _timestamp_iso(self.embargo_threshold_utc),
            "strict_embargo_pass": bool(
                self.train_times[-1] < self.embargo_threshold_utc
            ),
        }


@dataclass(frozen=True)
class PrefixPlan:
    outer_fold: str
    fraction: float
    outer_validation_start_utc: pd.Timestamp
    outer_validation_stop_utc: pd.Timestamp
    eligible_times: pd.DatetimeIndex
    prefix_times: pd.DatetimeIndex
    cutoff_utc: pd.Timestamp
    inner_folds: tuple[InnerFoldPlan, ...]

    @property
    def scope_id(self) -> str:
        return f"{self.outer_fold}__p{int(round(self.fraction * 100)):03d}"

    def summary(self) -> dict[str, Any]:
        return {
            "scope_id": self.scope_id,
            "outer_fold": self.outer_fold,
            "prefix_fraction": self.fraction,
            "eligible_time_count": len(self.eligible_times),
            "prefix_time_count": len(self.prefix_times),
            "prefix_count_rule_expected": int(
                math.ceil(self.fraction * len(self.eligible_times))
            ),
            "cutoff_kst": _timestamp_iso(self.cutoff_utc),
            "outer_validation_start_kst": _timestamp_iso(
                self.outer_validation_start_utc
            ),
            "outer_validation_stop_exclusive_kst": _timestamp_iso(
                self.outer_validation_stop_utc
            ),
            "inner_folds": [item.summary() for item in self.inner_folds],
        }


def build_prefix_plan(
    metadata: pd.DataFrame,
    *,
    outer_fold: str,
    validation_start_kst: str,
    validation_stop_kst: str,
    fraction: float,
    embargo_days: int = 7,
    cadence_minutes: int = CADENCE_MINUTES,
) -> PrefixPlan:
    """Build the exact expanding-time prefix and its three inner folds."""

    _require("time" in metadata, "metadata has no time column")
    _require(0.0 < float(fraction) <= 1.0, "prefix fraction is invalid")
    _require(embargo_days == 7, "sealed embargo changed")
    _require(cadence_minutes == 10, "sealed P2 cadence changed")
    unique_times = _utc_index(metadata["time"]).unique().sort_values()
    _require(len(unique_times) >= 8, "too few timestamps for nested prefixes")
    validation_start = pd.Timestamp(validation_start_kst).tz_convert("UTC")
    validation_stop = pd.Timestamp(validation_stop_kst).tz_convert("UTC")
    _require(validation_start < validation_stop, "outer validation interval is invalid")
    outer_threshold = validation_start - pd.Timedelta(days=embargo_days)
    eligible = unique_times[unique_times < outer_threshold]
    _require(len(eligible) >= 8, "outer fold has too few eligible timestamps")
    prefix_count = int(math.ceil(float(fraction) * len(eligible)))
    prefix = eligible[:prefix_count]
    _require(len(prefix) == prefix_count, "prefix count rule was not honored")
    _require(prefix[-1] < outer_threshold, "prefix crossed the outer embargo")

    boundaries = tuple((index * len(prefix)) // 4 for index in range(5))
    _require(boundaries[0] == 0 and boundaries[-1] == len(prefix), "bad inner boundaries")
    _require(
        all(
            left < right
            for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        ),
        "empty inner block",
    )
    inner: list[InnerFoldPlan] = []
    cadence = pd.Timedelta(minutes=cadence_minutes)
    for number in range(1, 4):
        validation = prefix[boundaries[number] : boundaries[number + 1]]
        start = validation[0]
        stop = validation[-1] + cadence
        threshold = start - pd.Timedelta(days=embargo_days)
        training = prefix[: boundaries[number]]
        training = training[training < threshold]
        _require(len(training) > 0, f"inner fold {number} has no training timestamps")
        _require(training[-1] < threshold, f"inner fold {number} embargo failed")
        _require(
            len(training.intersection(validation)) == 0,
            f"inner fold {number} train/validation overlap",
        )
        inner.append(
            InnerFoldPlan(
                inner_fold=f"inner_{number}",
                train_times=training,
                validation_times=validation,
                validation_start_utc=start,
                validation_stop_utc=stop,
                embargo_threshold_utc=threshold,
            )
        )
    return PrefixPlan(
        outer_fold=outer_fold,
        fraction=float(fraction),
        outer_validation_start_utc=validation_start,
        outer_validation_stop_utc=validation_stop,
        eligible_times=eligible,
        prefix_times=prefix,
        cutoff_utc=prefix[-1],
        inner_folds=tuple(inner),
    )


def build_all_prefix_plans(
    metadata: pd.DataFrame,
    recipe: Mapping[str, Any],
) -> tuple[PrefixPlan, ...]:
    nested = recipe["authoritative_nested_surrogate_recipe"]
    folds = nested["outer_fold_contract"]["folds"]
    fractions = nested["chronological_prefix_contract"]["fractions"]
    embargo = int(nested["outer_fold_contract"]["embargo_days"])
    plans = tuple(
        build_prefix_plan(
            metadata,
            outer_fold=str(fold["id"]),
            validation_start_kst=str(fold["validation_half_open_kst"][0]),
            validation_stop_kst=str(fold["validation_half_open_kst"][1]),
            fraction=float(fraction),
            embargo_days=embargo,
        )
        for fold in folds
        for fraction in fractions
    )
    _require(len(plans) == 15, "sealed 15-cell outer/prefix plan changed")
    _require(len({item.scope_id for item in plans}) == 15, "duplicate prefix scope")
    return plans


def build_seeded_execution_plan(
    plans: Sequence[PrefixPlan], seeds: Sequence[int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for plan in plans:
        for seed in seeds:
            result.append(
                {
                    "cell_id": f"{plan.scope_id}__s{int(seed)}",
                    "scope_id": plan.scope_id,
                    "outer_fold": plan.outer_fold,
                    "prefix_fraction": plan.fraction,
                    "cutoff_kst": _timestamp_iso(plan.cutoff_utc),
                    "complete_pipeline_seed": int(seed),
                    "inner_fold_count": 3,
                    "fit_authorized": False,
                }
            )
    _require(len(result) == 45, "sealed seeded execution plan changed")
    _require(len({item["cell_id"] for item in result}) == 45, "duplicate seeded cell")
    return result


def joint_mask_target_observations(
    observations: pd.DataFrame,
    allowed_times: Sequence[Any] | pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Jointly hide target-layer TEMP/PSAL outside ``allowed_times``."""

    required = {"time", "layer", "temp", "psal"}
    _require(required.issubset(observations), "observation masking schema is incomplete")
    result = observations.copy()
    time = _utc_index(result["time"])
    allowed = _utc_index(allowed_times).unique()
    target = result["layer"].isin(TARGET_LAYERS).to_numpy()
    outside = ~time.isin(allowed)
    mask = target & outside
    preserved_public_temp = result.loc[~target, "temp"].copy()
    preserved_public_psal = result.loc[~target, "psal"].copy()
    result.loc[mask, ["temp", "psal"]] = np.nan
    _require(result.loc[mask, ["temp", "psal"]].isna().all().all(), "joint mask failed")
    _require(
        result.loc[~target, "temp"].equals(preserved_public_temp)
        and result.loc[~target, "psal"].equals(preserved_public_psal),
        "public-layer values changed during target masking",
    )
    return result, {
        "target_rows_outside_allowed": int(mask.sum()),
        "temp_rows_masked": int(result.loc[mask, "temp"].isna().sum()),
        "psal_rows_masked": int(result.loc[mask, "psal"].isna().sum()),
        "joint_mask_equal": True,
        "public_rows_changed": 0,
    }


def _subset_panel(panel: P2Panel, selected_times: pd.DatetimeIndex) -> P2Panel:
    selected = panel.times.isin(selected_times)
    _require(bool(selected.any()), "panel subset is empty")
    times = panel.times[selected]
    delta = times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    segment_ids = (
        np.cumsum(np.r_[True, ~np.isclose(delta[1:], CADENCE_MINUTES)]).astype(np.int32)
        - 1
    )
    return P2Panel(
        times=times,
        inputs=panel.inputs[selected].copy(),
        input_names=panel.input_names,
        baseline=panel.baseline[selected].copy(),
        target=panel.target[selected].copy(),
        target_mask=panel.target_mask[selected].copy(),
        segment_ids=segment_ids,
    )


def adapt_panel_for_inner_fold(
    panel: P2Panel, inner: InnerFoldPlan
) -> tuple[P2Panel, dict[str, Any]]:
    """Return a panel whose complement split cannot see embargo/future rows."""

    allowed = inner.train_times.append(inner.validation_times).sort_values()
    adapted = _subset_panel(panel, allowed)
    validation = (adapted.times >= inner.validation_start_utc) & (
        adapted.times < inner.validation_stop_utc
    )
    training = ~validation
    _require(adapted.times[training].equals(inner.train_times), "deep train view differs")
    _require(
        adapted.times[validation].equals(inner.validation_times),
        "deep validation view differs",
    )
    _require(adapted.times[training][-1] < inner.embargo_threshold_utc, "deep embargo failed")
    return adapted, {
        "inner_fold": inner.inner_fold,
        "panel_time_count": len(adapted.times),
        "training_time_count": int(training.sum()),
        "validation_time_count": int(validation.sum()),
        "future_or_embargo_time_count_in_panel": 0,
        "train_fold_start": inner.validation_start_utc.isoformat(),
        "train_fold_stop": inner.validation_stop_utc.isoformat(),
        "complement_split_train_equals_registered_inner_train": True,
    }


def adapt_panel_for_full_prefix(
    panel: P2Panel, plan: PrefixPlan
) -> tuple[P2Panel, dict[str, Any]]:
    adapted = _subset_panel(panel, plan.prefix_times)
    _require(adapted.times.equals(plan.prefix_times), "full-prefix panel differs")
    _require(adapted.times[-1] == plan.cutoff_utc, "full-prefix cutoff differs")
    return adapted, {
        "scope_id": plan.scope_id,
        "panel_time_count": len(adapted.times),
        "cutoff_kst": _timestamp_iso(adapted.times[-1]),
        "later_time_count_in_panel": 0,
        "train_full_model_uses_only_registered_prefix": True,
    }


def child_seed(
    complete_seed: int,
    component: str,
    outer_fold: str,
    prefix_fraction: float,
    inner_fold_or_full: str,
) -> int:
    _require(component in COMPONENTS, f"unknown component: {component}")
    _require(
        inner_fold_or_full in {"inner_1", "inner_2", "inner_3", "full"},
        "unknown child-seed phase",
    )
    fraction_token = f"{float(prefix_fraction):.2f}"
    preimage = "|".join(
        (
            SEED_NAMESPACE,
            str(int(complete_seed)),
            component,
            outer_fold,
            fraction_token,
            inner_fold_or_full,
        )
    )
    return int(hashlib.sha256(preimage.encode("utf-8")).hexdigest()[:16], 16) % 2147483647


def select_best_epoch(history: Sequence[Mapping[str, Any]]) -> int:
    _require(len(history) > 0, "empty epoch history")
    rows = [(float(item["rmse"]), int(item["epoch"])) for item in history]
    _require(all(epoch >= 1 and np.isfinite(score) for score, epoch in rows), "bad epoch history")
    return min(rows, key=lambda item: (item[0], item[1]))[1]


def middle_epoch(best_epochs: Sequence[int]) -> int:
    values = sorted(int(value) for value in best_epochs)
    _require(len(values) == 3 and values[0] >= 1, "three positive inner epochs required")
    return values[1]


def _ordered_key_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {*KEY_COLUMNS, "truth"}
    _require(required.issubset(frame), "component OOF schema is incomplete")
    result = frame.loc[:, [*KEY_COLUMNS, "truth"]].copy()
    result["station"] = result["station"].astype(str)
    result["layer"] = result["layer"].astype(int)
    result["time"] = _utc_index(result["time"]).astype(str)
    result["truth"] = result["truth"].astype(float)
    _require(not result.duplicated(list(KEY_COLUMNS)).any(), "duplicate component OOF keys")
    _require(np.isfinite(result["truth"]).all(), "component OOF truth is non-finite")
    return result


def _digest_rows(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        encoded = [float(value).hex() if isinstance(value, float) else value for value in row]
        digest.update(
            json.dumps(encoded, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def merge_component_oof(
    component_frames: Mapping[str, pd.DataFrame],
    *,
    expected_keys: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Require exact ordered key/truth identity, then form a wide OOF ledger."""

    _require(tuple(component_frames) == COMPONENTS, "component order or set changed")
    reference_keys: pd.DataFrame | None = None
    merged: pd.DataFrame | None = None
    prediction_digests: dict[str, str] = {}
    for component, source in component_frames.items():
        _require("prediction" in source, f"{component} has no prediction")
        current_keys = _ordered_key_frame(source)
        prediction = source["prediction"].to_numpy(dtype=np.float64)
        _require(np.isfinite(prediction).all(), f"{component} prediction is non-finite")
        if reference_keys is None:
            reference_keys = current_keys
            merged = current_keys.copy()
        else:
            _require(
                current_keys.equals(reference_keys),
                f"{component} ordered key/truth surface differs",
            )
        assert merged is not None
        merged[f"pred_{component}"] = prediction
        prediction_digests[component] = _digest_rows(
            pd.DataFrame({"prediction": prediction}), ["prediction"]
        )
    assert reference_keys is not None and merged is not None
    if expected_keys is not None:
        expected = expected_keys.loc[:, list(KEY_COLUMNS)].copy()
        expected["station"] = expected["station"].astype(str)
        expected["layer"] = expected["layer"].astype(int)
        expected["time"] = _utc_index(expected["time"]).astype(str)
        _require(
            reference_keys.loc[:, list(KEY_COLUMNS)].equals(expected),
            "component OOF keys differ from registered inner validation keys",
        )
    key_digest = _digest_rows(reference_keys, KEY_COLUMNS)
    truth_digest = _digest_rows(reference_keys, [*KEY_COLUMNS, "truth"])
    return merged, {
        "rows": len(merged),
        "ordered_key_sha256": key_digest,
        "ordered_key_truth_sha256": truth_digest,
        "component_prediction_sha256": prediction_digests,
        "component_count": len(component_frames),
        "duplicate_keys": 0,
        "nonfinite_truth": 0,
        "nonfinite_predictions": 0,
        "same_ordered_key_and_truth_across_components": True,
    }


def fit_nnls_stack(
    oof: pd.DataFrame, prediction_columns: Sequence[str]
) -> dict[int, np.ndarray]:
    columns = tuple(prediction_columns)
    _require(len(columns) > 0, "stack needs contributors")
    result: dict[int, np.ndarray] = {}
    for layer in TARGET_LAYERS:
        selected = oof["layer"].to_numpy(int) == layer
        inputs = oof.loc[selected, columns].to_numpy(float)
        truth = oof.loc[selected, "truth"].to_numpy(float)
        _require(len(truth) > 0, f"stack layer {layer} is empty")
        _require(np.isfinite(inputs).all() and np.isfinite(truth).all(), "bad stack arrays")
        weights, _ = nnls(inputs, truth)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 1e-14:
            weights = np.full(len(columns), 1.0 / len(columns), dtype=np.float64)
        else:
            weights = weights / total
        _require((weights >= 0).all(), "negative NNLS stack weight")
        _require(np.isclose(weights.sum(), 1.0, atol=1e-12), "stack is not sum-one")
        result[layer] = weights
    return result


def fit_prefix_local_meta(
    oof: pd.DataFrame,
    *,
    scope_id: str,
    prediction_columns: Sequence[str],
    feature_names: Sequence[str] = STATE_FEATURES,
    gate_regularization: float = 10.0,
) -> dict[str, Any]:
    """Freshly fit stack and gate using only one scope's nested OOF ledger."""

    columns = tuple(prediction_columns)
    features = tuple(feature_names)
    required = {"layer", "truth", *columns, *features, *KEY_COLUMNS}
    _require(required.issubset(oof), "prefix-local meta frame is incomplete")
    _require(scope_id != "", "meta scope id is empty")
    key_frame = _ordered_key_frame(oof)
    key_truth_digest = _digest_rows(key_frame, [*KEY_COLUMNS, "truth"])
    stack = fit_nnls_stack(oof, columns)
    gate = fit_soft_gate(
        oof,
        feature_names=features,
        prediction_columns=columns,
        regularization=float(gate_regularization),
    )
    stack_payload = {
        str(layer): {column: float(weight) for column, weight in zip(columns, stack[layer], strict=True)}
        for layer in TARGET_LAYERS
    }
    gate_payload: dict[str, Any] = {}
    for layer, fitted in gate.layers.items():
        gate_payload[str(layer)] = {
            "prior": fitted.prior.tolist(),
            "coefficient_shape": list(fitted.coefficients.shape),
            "coefficient_sha256": hashlib.sha256(
                np.ascontiguousarray(fitted.coefficients, dtype="<f8").tobytes()
            ).hexdigest(),
            "transform_center_sha256": hashlib.sha256(
                np.ascontiguousarray(fitted.transform.center, dtype="<f8").tobytes()
            ).hexdigest(),
            "transform_scale_sha256": hashlib.sha256(
                np.ascontiguousarray(fitted.transform.scale, dtype="<f8").tobytes()
            ).hexdigest(),
            "optimizer_iterations": int(fitted.optimizer_iterations),
            "objective_mse": float(fitted.objective_mse),
        }
    return {
        "scope_id": scope_id,
        "rows": len(oof),
        "ordered_key_truth_sha256": key_truth_digest,
        "parameter_source": "CURRENT_SCOPE_NESTED_COMPONENT_OOF_ONLY",
        "frozen_stack_reused": False,
        "frozen_gate_reused": False,
        "stack_method": "SCIPY_NNLS_THEN_SUM_NORMALIZE_UNIFORM_IF_ALL_ZERO",
        "stack_weights": stack_payload,
        "gate_regularization": float(gate_regularization),
        "gate_feature_names": list(features),
        "gate_layers": gate_payload,
        "stack_and_gate_refit_for_this_scope": True,
    }


def build_epoch_refit_receipt(
    histories: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    _require(tuple(histories) == DEEP_COMPONENTS, "deep component epoch set changed")
    result: dict[str, Any] = {}
    for component, by_inner in histories.items():
        _require(
            tuple(by_inner) == ("inner_1", "inner_2", "inner_3"),
            f"{component} inner epoch folds changed",
        )
        best = [select_best_epoch(by_inner[name]) for name in by_inner]
        result[component] = {
            "best_epoch_by_inner": dict(zip(by_inner, best, strict=True)),
            "full_prefix_epochs": middle_epoch(best),
            "selection": "MIN_RMSE_EARLIEST_TIE",
            "aggregation": "MIDDLE_OF_THREE_INTEGER_BEST_EPOCHS",
            "full_prefix_refit": True,
            "frozen_epoch_reused": False,
        }
    return result


def source_api_conformance() -> dict[str, Any]:
    """Assert that the existing model APIs can consume the adapters."""

    signatures = {
        "train_fold": tuple(inspect.signature(train_fold).parameters),
        "train_full_model": tuple(inspect.signature(train_full_model).parameters),
        "fit_model": tuple(inspect.signature(fit_model).parameters),
        "fit_soft_gate": tuple(inspect.signature(fit_soft_gate).parameters),
    }
    _require(
        signatures["train_fold"] == ("panel", "block", "start", "stop", "config", "progress"),
        "deep train_fold API changed",
    )
    _require(
        signatures["train_full_model"] == ("panel", "config", "epochs", "progress"),
        "deep full-refit API changed",
    )
    _require(signatures["fit_model"] == ("table", "rows", "seed"), "router fit API changed")
    _require(
        signatures["fit_soft_gate"]
        == ("frame", "feature_names", "prediction_columns", "regularization"),
        "gate fit API changed",
    )
    return {
        "status": "PASS",
        "signatures": {name: list(value) for name, value in signatures.items()},
        "adapter_path": {
            "deep_inner": "adapt_panel_for_inner_fold -> train_fold",
            "deep_full": "adapt_panel_for_full_prefix -> train_full_model",
            "router": "joint_mask_target_observations + registered row mask -> fit_model",
            "meta": "merge_component_oof -> fresh fit_nnls_stack + fit_soft_gate",
        },
    }
