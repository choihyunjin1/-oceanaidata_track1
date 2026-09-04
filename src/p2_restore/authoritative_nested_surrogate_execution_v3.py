"""Data-executable v3 P2 authoritative nested-surrogate contract.

The failed v2 contract formed chronological prefixes from every observation
timestamp, including periods before target labels existed.  V3 forms all
fractions and inner folds from one prospective, supervised-eligible ledger:
every target layer (2, 3, 4) must have finite TEMP and PSAL and must be
executable by both the router and deep panel.

Deep models still receive continuous public-covariate context.  Only their
supervision mask is restricted to the registered ledger.  The semantic
preflight below invokes the exact v3 backend adapters, but never fits a model,
writes a file, creates a lock, predicts, or scores.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from p2_restore import authoritative_nested_surrogate_execution as v2
from p2_restore.architecture_matched_stage_a_execution_v2 import (
    RouterContext,
    _build_router_context,
    _joint_masked_panel,
)
from p2_restore.authoritative_nested_surrogate_conformance import (
    COMPONENTS,
    DEEP_COMPONENTS,
    TARGET_LAYERS,
    InnerFoldPlan,
    PrefixPlan,
    child_seed,
)
from p2_restore.deep_data import P2Panel, PanelNormalizer, make_chunk_bounds
from p2_restore.deep_training import train_full_model
from p2_restore.regime_gate import STATE_FEATURES
from p2_restore.state_conditional import compute_state_partition

SUPERVISED_LEDGER_ID = "P2_TARGET_234_TEMP_PSAL_ROUTER_DEEP_COMMON_V3"
CADENCE_MINUTES = 10
MINIMUM_CHUNK_TARGET_VALUES = 24


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _utc_index(values: Sequence[Any] | pd.Series | pd.Index) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))


def _timestamp_iso(value: pd.Timestamp) -> str:
    return value.tz_convert("Asia/Seoul").isoformat()


def _resegment(times: pd.DatetimeIndex) -> np.ndarray:
    delta = times.to_series().diff().dt.total_seconds().div(60).to_numpy()
    return np.cumsum(np.r_[True, ~np.isclose(delta[1:], CADENCE_MINUTES)]).astype(
        np.int32
    ) - 1


def _masked_panel_subset(
    panel: P2Panel,
    *,
    context_rows: np.ndarray,
    allowed_label_times: pd.DatetimeIndex,
) -> P2Panel:
    selected = np.asarray(context_rows, dtype=bool)
    _require(selected.shape == (len(panel.times),) and selected.any(), "deep context is empty")
    times = panel.times[selected]
    allowed = np.asarray(times.isin(allowed_label_times), dtype=bool)
    target = panel.target[selected].copy()
    target_mask = panel.target_mask[selected].copy() & allowed[:, None]
    target[~target_mask] = np.nan
    _require(bool(target_mask.any()), "deep supervised label mask is empty")
    return P2Panel(
        times=times,
        inputs=panel.inputs[selected].copy(),
        input_names=panel.input_names,
        baseline=panel.baseline[selected].copy(),
        target=target,
        target_mask=target_mask,
        segment_ids=_resegment(times),
    )


def adapt_panel_for_inner_fold_v3(
    panel: P2Panel, inner: InnerFoldPlan
) -> tuple[P2Panel, dict[str, Any]]:
    """Keep continuous train/validation public context and only registered labels.

    The training context is all public timestamps through the last registered
    training label.  The validation context is the complete public timestamp
    interval spanning the registered validation ledger.  The seven-day gap
    separates the two contexts, so resegmentation prevents a training chunk
    from crossing into validation.
    """

    training_context = panel.times <= inner.train_times[-1]
    validation_context = (panel.times >= inner.validation_start_utc) & (
        panel.times < inner.validation_stop_utc
    )
    adapted = _masked_panel_subset(
        panel,
        context_rows=np.asarray(training_context | validation_context, dtype=bool),
        allowed_label_times=inner.train_times.append(inner.validation_times).unique(),
    )
    is_validation = (adapted.times >= inner.validation_start_utc) & (
        adapted.times < inner.validation_stop_utc
    )
    train_label_times = adapted.times[(~is_validation) & adapted.target_mask.all(axis=1)]
    validation_label_times = adapted.times[is_validation & adapted.target_mask.all(axis=1)]
    _require(train_label_times.equals(inner.train_times), "v3 deep train ledger differs")
    _require(
        validation_label_times.equals(inner.validation_times),
        "v3 deep validation ledger differs",
    )
    _require(
        adapted.times[~is_validation].max() < inner.embargo_threshold_utc,
        "v3 deep public train context crossed embargo",
    )
    return adapted, {
        "schema_version": "p2_authoritative_deep_inner_adapter.v3",
        "inner_fold": inner.inner_fold,
        "panel_time_count": len(adapted.times),
        "continuous_training_public_time_count": int((~is_validation).sum()),
        "continuous_validation_public_time_count": int(is_validation.sum()),
        "training_supervised_time_count": len(inner.train_times),
        "validation_supervised_time_count": len(inner.validation_times),
        "masked_nonregistered_target_values": int((~adapted.target_mask).sum()),
        "training_context_last_kst": _timestamp_iso(adapted.times[~is_validation].max()),
        "validation_context_first_kst": _timestamp_iso(adapted.times[is_validation].min()),
        "strict_embargo_pass": True,
        "continuous_public_covariates_preserved": True,
        "labels_restricted_to_registered_common_ledger": True,
    }


def adapt_panel_for_full_prefix_v3(
    panel: P2Panel, plan: PrefixPlan
) -> tuple[P2Panel, dict[str, Any]]:
    """Keep every public timestamp through cutoff; expose only prefix labels."""

    context = np.asarray(panel.times <= plan.cutoff_utc, dtype=bool)
    adapted = _masked_panel_subset(
        panel,
        context_rows=context,
        allowed_label_times=plan.prefix_times,
    )
    supervised = adapted.times[adapted.target_mask.all(axis=1)]
    _require(supervised.equals(plan.prefix_times), "v3 full-prefix label ledger differs")
    _require(adapted.times[-1] == plan.cutoff_utc, "v3 full-prefix cutoff differs")
    return adapted, {
        "schema_version": "p2_authoritative_deep_full_adapter.v3",
        "scope_id": plan.scope_id,
        "continuous_public_time_count": len(adapted.times),
        "supervised_time_count": len(supervised),
        "cutoff_kst": _timestamp_iso(adapted.times[-1]),
        "later_public_time_count": 0,
        "continuous_public_covariates_preserved": True,
        "labels_restricted_to_registered_common_ledger": True,
    }


def fully_joint_target_times(observations: pd.DataFrame) -> pd.DatetimeIndex:
    """Return timestamps where every target layer has finite TEMP and PSAL."""

    required = {"station", "layer", "time", "temp", "psal"}
    _require(required.issubset(observations), "v3 target-ledger schema is incomplete")
    target = observations.loc[
        observations["layer"].isin(TARGET_LAYERS),
        ["station", "layer", "time", "temp", "psal"],
    ].copy()
    target["_time"] = _utc_index(target["time"])
    _require(
        not target.duplicated(["station", "layer", "_time"]).any(),
        "v3 target-ledger keys are duplicated",
    )
    target["_joint"] = np.isfinite(target["temp"]) & np.isfinite(target["psal"])
    wide = target.pivot(index=["station", "_time"], columns="layer", values="_joint")
    wide = wide.reindex(columns=list(TARGET_LAYERS)).fillna(False).astype(bool)
    complete = wide.all(axis=1)
    station_count = int(target["station"].nunique())
    _require(station_count == 1, "v3 execution requires the sealed single station")
    times = pd.DatetimeIndex(wide.index.get_level_values("_time")[complete]).unique().sort_values()
    _require(len(times) >= 8, "v3 joint target ledger is too small")
    return times


def _subset_router_context(context: RouterContext, times: pd.DatetimeIndex) -> RouterContext:
    rows = np.asarray(context.joint_rows, dtype=bool) & np.asarray(
        context.times.isin(times), dtype=bool
    )
    _require(bool(rows.any()), "v3 router scope is empty")
    positions = np.flatnonzero(rows)

    def subset(table: Any) -> Any:
        return type(table)(table.frame.loc[rows].reset_index(drop=True), table.feature_columns)

    return RouterContext(
        base=subset(context.base),
        lean=subset(context.lean),
        phase=subset(context.phase),
        public_state=context.public_state.iloc[positions].reset_index(drop=True),
        joint_rows=np.ones(len(positions), dtype=bool),
        times=pd.DatetimeIndex(context.times[positions]),
    )


def supervised_common_ledger(
    observations: pd.DataFrame,
    *,
    panel: P2Panel,
    router_context: RouterContext,
) -> tuple[pd.DatetimeIndex, dict[str, Any]]:
    """Intersect label, deep, and router executability into one time ledger."""

    target_times = fully_joint_target_times(observations)
    deep_times = panel.times[panel.target_mask.all(axis=1)]
    router = router_context.base.frame.loc[
        np.asarray(router_context.joint_rows, dtype=bool), ["station", "layer", "time"]
    ].copy()
    router["_time"] = _utc_index(router["time"])
    router_wide = (
        router.assign(_ready=True)
        .pivot(index=["station", "_time"], columns="layer", values="_ready")
        .reindex(columns=list(TARGET_LAYERS))
        .fillna(False)
        .astype(bool)
    )
    router_complete = router_wide.all(axis=1)
    router_times = pd.DatetimeIndex(
        router_wide.index.get_level_values("_time")[router_complete]
    ).unique().sort_values()
    common = target_times.intersection(deep_times).intersection(router_times).sort_values()
    _require(len(common) >= 8, "v3 executable common ledger is too small")
    _require(common.is_unique and common.is_monotonic_increasing, "v3 common ledger order failed")
    return common, {
        "schema_version": "p2_supervised_common_ledger.v3",
        "ledger_id": SUPERVISED_LEDGER_ID,
        "target_all_three_temp_psal_time_count": len(target_times),
        "deep_all_three_executable_time_count": len(deep_times),
        "router_all_three_executable_time_count": len(router_times),
        "common_supervised_time_count": len(common),
        "excluded_target_joint_times": len(target_times) - len(common),
        "first_kst": _timestamp_iso(common[0]),
        "last_kst": _timestamp_iso(common[-1]),
        "ordered_time_sha256": v2.canonical_sha256([value.isoformat() for value in common]),
        "target_layers": list(TARGET_LAYERS),
        "finite_columns": ["temp", "psal"],
        "all_target_layers_required_at_each_timestamp": True,
        "prospective_label_availability_only": True,
    }


def build_prefix_plan_v3(
    supervised_times: pd.DatetimeIndex,
    *,
    outer_fold: str,
    validation_start_kst: str,
    validation_stop_kst: str,
    fraction: float,
    embargo_days: int = 7,
) -> PrefixPlan:
    """Build one prefix and three inner folds from supervised timestamps only."""

    _require(0.0 < float(fraction) <= 1.0, "v3 prefix fraction is invalid")
    _require(embargo_days == 7, "v3 sealed embargo changed")
    unique = _utc_index(supervised_times).unique().sort_values()
    validation_start = pd.Timestamp(validation_start_kst).tz_convert("UTC")
    validation_stop = pd.Timestamp(validation_stop_kst).tz_convert("UTC")
    outer_threshold = validation_start - pd.Timedelta(days=embargo_days)
    eligible = unique[unique < outer_threshold]
    _require(len(eligible) >= 8, "v3 outer fold has too few supervised timestamps")
    prefix_count = int(math.ceil(float(fraction) * len(eligible)))
    prefix = eligible[:prefix_count]
    boundaries = tuple((index * len(prefix)) // 4 for index in range(5))
    _require(
        all(left < right for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)),
        "v3 supervised inner block is empty",
    )
    cadence = pd.Timedelta(minutes=CADENCE_MINUTES)
    inner: list[InnerFoldPlan] = []
    for number in range(1, 4):
        validation = prefix[boundaries[number] : boundaries[number + 1]]
        start = validation[0]
        stop = validation[-1] + cadence
        threshold = start - pd.Timedelta(days=embargo_days)
        training = prefix[: boundaries[number]]
        training = training[training < threshold]
        _require(len(training) > 0, f"v3 inner fold {number} has no supervised train labels")
        _require(training[-1] < threshold, f"v3 inner fold {number} embargo failed")
        _require(
            len(training.intersection(validation)) == 0,
            f"v3 inner fold {number} overlap",
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


def build_all_prefix_plans_v3(
    supervised_times: pd.DatetimeIndex,
    recipe: Mapping[str, Any],
) -> tuple[PrefixPlan, ...]:
    nested = recipe["authoritative_nested_surrogate_recipe"]
    folds = nested["outer_fold_contract"]["folds"]
    fractions = nested["chronological_prefix_contract"]["fractions"]
    embargo = int(nested["outer_fold_contract"]["embargo_days"])
    plans = tuple(
        build_prefix_plan_v3(
            supervised_times,
            outer_fold=str(fold["id"]),
            validation_start_kst=str(fold["validation_half_open_kst"][0]),
            validation_stop_kst=str(fold["validation_half_open_kst"][1]),
            fraction=float(fraction),
            embargo_days=embargo,
        )
        for fold in folds
        for fraction in fractions
    )
    _require(len(plans) == 15, "v3 15-scope plan changed")
    _require(len({plan.scope_id for plan in plans}) == 15, "v3 duplicate scope")
    return plans


class ActualTrainingBackendV3(v2.ActualTrainingBackend):
    """Exact production backend with v3 split/context adapters."""

    def __init__(self, observations: pd.DataFrame, config: Mapping[str, Any]) -> None:
        super().__init__(observations, config)
        self.full_router_context = _build_router_context(observations)

    def _masked_router_context(self, times: pd.DatetimeIndex) -> RouterContext:
        # Public features are target-label independent.  Subsetting the one full
        # context by joint-label rows is exactly the v3 registered router view.
        return _subset_router_context(self.full_router_context, _utc_index(times).unique())

    def fit_inner(
        self,
        component: str,
        plan: PrefixPlan,
        inner: InnerFoldPlan,
        pipeline_seed: int,
    ) -> v2.JobProduct:
        if component == "router_400":
            return super().fit_inner(component, plan, inner, pipeline_seed)
        seed = child_seed(
            pipeline_seed,
            component,
            plan.outer_fold,
            plan.fraction,
            inner.inner_fold,
        )
        adapted, adapter = adapt_panel_for_inner_fold_v3(self.panel, inner)
        config = self._deep_config(component, seed)
        start = inner.validation_start_utc.tz_convert("Asia/Seoul").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        stop = inner.validation_stop_utc.tz_convert("Asia/Seoul").strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        result = v2.train_fold_exact_min_rmse(
            adapted,
            block=inner.inner_fold,
            start=start,
            stop=stop,
            config=config,
        )
        frame = result.oof.copy()
        frame.insert(0, "station", "S-ORS")
        receipt = {
            "component": component,
            "phase": inner.inner_fold,
            "seed": seed,
            "best_epoch": int(result.best_epoch),
            "best_rmse_c": float(result.best_rmse),
            "parameter_count": int(result.parameter_count),
            "history": result.history,
            "adapter": adapter,
            "future_or_outer_labels_in_fit": False,
        }
        checkpoint = v2._deep_checkpoint_payload(
            result, epochs=result.best_epoch, role="inner_best_checkpoint"
        )
        return v2.JobProduct(
            frame, receipt, {"checkpoint.pt": v2._torch_bytes(checkpoint)}
        )

    def fit_full(
        self,
        component: str,
        plan: PrefixPlan,
        pipeline_seed: int,
        full_epochs: int | None,
    ) -> v2.JobProduct:
        if component == "router_400":
            return super().fit_full(component, plan, pipeline_seed, full_epochs)
        seed = child_seed(
            pipeline_seed,
            component,
            plan.outer_fold,
            plan.fraction,
            "full",
        )
        _require(full_epochs is not None and full_epochs >= 1, "v3 deep full epochs missing")
        training_panel, adapter = adapt_panel_for_full_prefix_v3(self.panel, plan)
        config = replace(
            self._deep_config(component, seed),
            max_epochs=int(full_epochs),
            patience=int(full_epochs),
        )
        v2.set_deterministic_seed(seed)
        result = train_full_model(training_panel, config, epochs=int(full_epochs))
        outer_times = self.panel.times[
            (self.panel.times >= plan.outer_validation_start_utc)
            & (self.panel.times < plan.outer_validation_stop_utc)
        ]
        prediction_panel = v2._subset_panel(self.panel, outer_times, expose_targets=False)
        prediction = v2._predict_full_result(result, prediction_panel)
        truth_panel = v2._subset_panel(self.panel, outer_times, expose_targets=True)
        rows: list[pd.DataFrame] = []
        for offset, layer in enumerate(TARGET_LAYERS):
            keep = truth_panel.target_mask[:, offset]
            rows.append(
                pd.DataFrame(
                    {
                        "station": "S-ORS",
                        "layer": layer,
                        "time": truth_panel.times[keep].astype(str),
                        "truth": truth_panel.target[keep, offset],
                        "prediction": prediction[keep, offset],
                    }
                )
            )
        frame = pd.concat(rows, ignore_index=True)
        receipt = {
            "component": component,
            "phase": "full",
            "seed": seed,
            "epochs": int(full_epochs),
            "parameter_count": int(result.parameter_count),
            "final_train_mse_c": float(result.final_train_mse_c),
            "adapter": adapter,
            "future_or_outer_labels_in_fit": False,
        }
        checkpoint = v2._deep_checkpoint_payload(
            result, epochs=int(full_epochs), role="full_prefix_refit"
        )
        return v2.JobProduct(
            frame, receipt, {"checkpoint.pt": v2._torch_bytes(checkpoint)}
        )


def _router_keys(context: RouterContext) -> pd.DataFrame:
    frame = context.base.frame.loc[:, ["station", "layer", "time"]].copy()
    frame["station"] = frame["station"].astype(str)
    frame["layer"] = frame["layer"].astype(int)
    frame["time"] = _utc_index(frame["time"]).astype(str)
    return frame.reset_index(drop=True)


def _deep_keys(panel: P2Panel, *, validation: np.ndarray | None = None) -> pd.DataFrame:
    selected_time = (
        np.ones(len(panel.times), dtype=bool)
        if validation is None
        else np.asarray(validation, dtype=bool)
    )
    rows: list[pd.DataFrame] = []
    for offset, layer in enumerate(TARGET_LAYERS):
        keep = selected_time & panel.target_mask[:, offset]
        rows.append(
            pd.DataFrame(
                {
                    "station": "S-ORS",
                    "layer": layer,
                    "time": panel.times[keep].astype(str),
                }
            )
        )
    return pd.concat(rows, ignore_index=True).reset_index(drop=True)


def _chunk_support(
    panel: P2Panel,
    *,
    selected_training_times: np.ndarray,
    chunk_length: int,
    chunk_stride: int,
) -> dict[str, Any]:
    selected = np.asarray(selected_training_times, dtype=bool)
    normalizer = PanelNormalizer.fit(panel, selected)
    _, mask = normalizer.transform_targets(panel)
    training_mask = mask & selected[:, None]
    bounds = make_chunk_bounds(
        panel.segment_ids,
        length=int(chunk_length),
        stride=int(chunk_stride),
    )
    supervised = tuple(
        bound
        for bound in bounds
        if training_mask[bound[0] : bound[1]].sum() >= MINIMUM_CHUNK_TARGET_VALUES
    )
    _require(bool(supervised), "v3 preflight found no deep supervised chunks")
    per_layer = {
        str(layer): int(training_mask[:, offset].sum())
        for offset, layer in enumerate(TARGET_LAYERS)
    }
    _require(all(value > 0 for value in per_layer.values()), "v3 deep layer support is empty")
    return {
        "training_values": int(training_mask.sum()),
        "training_rows_by_layer": per_layer,
        "all_chunk_count": len(bounds),
        "supervised_chunk_count": len(supervised),
        "minimum_values_per_kept_chunk": int(
            min(training_mask[start:stop].sum() for start, stop in supervised)
        ),
    }


def semantic_preflight_actual_data(
    observations: pd.DataFrame,
    *,
    recipe: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[tuple[PrefixPlan, ...], dict[str, Any]]:
    """Exercise exact v3 row/chunk/state adapters across all scopes, with zero fit."""

    backend = ActualTrainingBackendV3(observations, config)
    ledger, ledger_receipt = supervised_common_ledger(
        observations,
        panel=backend.panel,
        router_context=backend.full_router_context,
    )
    plans = build_all_prefix_plans_v3(ledger, recipe)
    deep_specs = {
        (
            int(config["component_hyperparameters"]["deep"][name]["chunk_length"]),
            int(config["component_hyperparameters"]["deep"][name]["chunk_stride"]),
        )
        for name in DEEP_COMPONENTS
    }
    _require(len(deep_specs) == 1, "v3 deep chunk hyperparameters differ across contributors")
    chunk_length, chunk_stride = next(iter(deep_specs))
    scope_rows: list[dict[str, Any]] = []
    outer_by_fold: dict[str, dict[str, Any]] = {}
    minima = {
        "router_train_rows_per_layer": 2**63 - 1,
        "router_validation_rows_per_layer": 2**63 - 1,
        "deep_train_rows_per_layer": 2**63 - 1,
        "deep_validation_rows_per_layer": 2**63 - 1,
        "deep_supervised_chunks": 2**63 - 1,
        "router_mixed_partition_rows": 2**63 - 1,
        "router_stratified_partition_rows": 2**63 - 1,
        "meta_oof_rows_per_layer": 2**63 - 1,
        "full_deep_supervised_chunks": 2**63 - 1,
    }
    for plan in plans:
        full_router = backend._masked_router_context(plan.prefix_times)
        full_partition = compute_state_partition(full_router.lean)
        full_panel, full_adapter = adapt_panel_for_full_prefix_v3(backend.panel, plan)
        full_chunk = _chunk_support(
            full_panel,
            selected_training_times=np.ones(len(full_panel.times), dtype=bool),
            chunk_length=chunk_length,
            chunk_stride=chunk_stride,
        )
        minima["full_deep_supervised_chunks"] = min(
            minima["full_deep_supervised_chunks"], full_chunk["supervised_chunk_count"]
        )
        inner_rows: list[dict[str, Any]] = []
        meta_keys: list[pd.DataFrame] = []
        for inner in plan.inner_folds:
            train_router = backend._masked_router_context(inner.train_times)
            validation_router = backend._masked_router_context(inner.validation_times)
            train_layer_counts = (
                train_router.base.frame["layer"].astype(int).value_counts().to_dict()
            )
            validation_layer_counts = (
                validation_router.base.frame["layer"].astype(int).value_counts().to_dict()
            )
            _require(
                all(int(train_layer_counts.get(layer, 0)) > 0 for layer in TARGET_LAYERS),
                "v3 router train layer support is empty",
            )
            _require(
                all(int(validation_layer_counts.get(layer, 0)) > 0 for layer in TARGET_LAYERS),
                "v3 router validation layer support is empty",
            )
            partition = compute_state_partition(train_router.lean)
            adapted, adapter = adapt_panel_for_inner_fold_v3(backend.panel, inner)
            validation_mask = (adapted.times >= inner.validation_start_utc) & (
                adapted.times < inner.validation_stop_utc
            )
            chunk = _chunk_support(
                adapted,
                selected_training_times=~validation_mask,
                chunk_length=chunk_length,
                chunk_stride=chunk_stride,
            )
            deep_validation_counts = {
                str(layer): int((validation_mask & adapted.target_mask[:, offset]).sum())
                for offset, layer in enumerate(TARGET_LAYERS)
            }
            _require(
                all(value > 0 for value in deep_validation_counts.values()),
                "v3 deep validation layer support is empty",
            )
            router_keys = _router_keys(validation_router)
            deep_keys = _deep_keys(adapted, validation=validation_mask)
            _require(router_keys.equals(deep_keys), "v3 router/deep validation ledger differs")
            keyed = router_keys.copy()
            keyed.insert(0, "inner_fold", inner.inner_fold)
            meta_keys.append(keyed)
            train_min = min(int(train_layer_counts[layer]) for layer in TARGET_LAYERS)
            validation_min = min(
                int(validation_layer_counts[layer]) for layer in TARGET_LAYERS
            )
            deep_train_min = min(chunk["training_rows_by_layer"].values())
            deep_validation_min = min(deep_validation_counts.values())
            minima["router_train_rows_per_layer"] = min(
                minima["router_train_rows_per_layer"], train_min
            )
            minima["router_validation_rows_per_layer"] = min(
                minima["router_validation_rows_per_layer"], validation_min
            )
            minima["deep_train_rows_per_layer"] = min(
                minima["deep_train_rows_per_layer"], deep_train_min
            )
            minima["deep_validation_rows_per_layer"] = min(
                minima["deep_validation_rows_per_layer"], deep_validation_min
            )
            minima["deep_supervised_chunks"] = min(
                minima["deep_supervised_chunks"], chunk["supervised_chunk_count"]
            )
            minima["router_mixed_partition_rows"] = min(
                minima["router_mixed_partition_rows"], int(partition.mixed_rows.sum())
            )
            minima["router_stratified_partition_rows"] = min(
                minima["router_stratified_partition_rows"],
                int(partition.stratified_rows.sum()),
            )
            inner_rows.append(
                {
                    "inner_fold": inner.inner_fold,
                    "train_supervised_times": len(inner.train_times),
                    "validation_supervised_times": len(inner.validation_times),
                    "router_train_rows_by_layer": {
                        str(layer): int(train_layer_counts[layer]) for layer in TARGET_LAYERS
                    },
                    "router_validation_rows_by_layer": {
                        str(layer): int(validation_layer_counts[layer])
                        for layer in TARGET_LAYERS
                    },
                    "router_mixed_partition_rows": int(partition.mixed_rows.sum()),
                    "router_stratified_partition_rows": int(
                        partition.stratified_rows.sum()
                    ),
                    "deep_training": chunk,
                    "deep_validation_rows_by_layer": deep_validation_counts,
                    "deep_adapter": adapter,
                    "router_deep_key_support_equal": True,
                    "ordered_validation_key_sha256": v2.canonical_sha256(
                        router_keys.to_dict("records")
                    ),
                }
            )
        combined = pd.concat(meta_keys, ignore_index=True)
        _require(
            not combined.duplicated(["inner_fold", "station", "layer", "time"]).any(),
            "v3 meta OOF keys are duplicated",
        )
        meta_counts = combined["layer"].astype(int).value_counts().to_dict()
        _require(
            all(int(meta_counts.get(layer, 0)) > 0 for layer in TARGET_LAYERS),
            "v3 meta OOF layer support is empty",
        )
        minima["meta_oof_rows_per_layer"] = min(
            minima["meta_oof_rows_per_layer"],
            min(int(meta_counts[layer]) for layer in TARGET_LAYERS),
        )
        scope_rows.append(
            {
                "scope_id": plan.scope_id,
                "outer_fold": plan.outer_fold,
                "prefix_fraction": plan.fraction,
                "eligible_supervised_time_count": len(plan.eligible_times),
                "prefix_supervised_time_count": len(plan.prefix_times),
                "cutoff_kst": _timestamp_iso(plan.cutoff_utc),
                "full_router_rows": len(full_router.base.frame),
                "full_router_mixed_partition_rows": int(full_partition.mixed_rows.sum()),
                "full_router_stratified_partition_rows": int(
                    full_partition.stratified_rows.sum()
                ),
                "full_deep_training": full_chunk,
                "full_deep_adapter": full_adapter,
                "meta_oof_rows_by_layer": {
                    str(layer): int(meta_counts[layer]) for layer in TARGET_LAYERS
                },
                "meta_oof_ordered_key_sha256": v2.canonical_sha256(
                    combined.to_dict("records")
                ),
                "inner_scopes": inner_rows,
            }
        )
        if plan.outer_fold not in outer_by_fold:
            outer_times = backend.panel.times[
                (backend.panel.times >= plan.outer_validation_start_utc)
                & (backend.panel.times < plan.outer_validation_stop_utc)
            ]
            outer_router = backend._masked_router_context(outer_times)
            router_keys = _router_keys(outer_router)
            outer_panel = v2._subset_panel(backend.panel, outer_times, expose_targets=True)
            deep_keys = _deep_keys(outer_panel)
            _require(router_keys.equals(deep_keys), "v3 outer router/deep key surface differs")
            layer_counts = router_keys["layer"].astype(int).value_counts().to_dict()
            outer_by_fold[plan.outer_fold] = {
                "rows": len(router_keys),
                "rows_by_layer": {
                    str(layer): int(layer_counts[layer]) for layer in TARGET_LAYERS
                },
                "ordered_key_sha256": v2.canonical_sha256(router_keys.to_dict("records")),
                "router_deep_key_support_equal": True,
            }
    expected_outer = int(config["metrics"]["expected_evaluation_rows_per_fraction"])
    observed_outer = sum(int(value["rows"]) for value in outer_by_fold.values())
    _require(observed_outer == expected_outer, "v3 outer metric population changed")
    _require(len(scope_rows) == 15, "v3 semantic scope count changed")
    _require(
        sum(len(scope["inner_scopes"]) for scope in scope_rows) == 45,
        "v3 semantic inner-scope count changed",
    )
    _require(all(value < 2**63 - 1 for value in minima.values()), "v3 minima incomplete")
    receipt = {
        "schema_version": "p2_authoritative_actual_data_semantic_preflight.v3",
        "status": "PASS_DATA_EXECUTABLE_ZERO_FIT",
        "access_scope": "PINNED_OBSERVATIONS_CSV_ONLY",
        "supervised_common_ledger": ledger_receipt,
        "outer_prefix_scopes_checked": 15,
        "inner_scopes_checked": 45,
        "component_model_fits": 0,
        "predictions_materialized": 0,
        "scores_computed": 0,
        "locks_created": 0,
        "partial_directories_created": 0,
        "model_files_written": 0,
        "router_feature_counts": {
            "base": len(backend.full_router_context.base.feature_columns),
            "lean": len(backend.full_router_context.lean.feature_columns),
            "phase": len(backend.full_router_context.phase.feature_columns),
        },
        "deep_input_channels": len(backend.panel.input_names),
        "deep_chunk_contract": {
            "chunk_length": chunk_length,
            "chunk_stride": chunk_stride,
            "minimum_supervised_values": MINIMUM_CHUNK_TARGET_VALUES,
        },
        "minimum_support_across_all_scopes": minima,
        "outer_validation_by_fold": outer_by_fold,
        "outer_evaluation_rows_per_fraction": observed_outer,
        "scope_support": scope_rows,
        "guards": {
            "fractions_derived_from_supervised_common_ledger": True,
            "all_three_target_layers_temp_psal_finite_per_split_time": True,
            "continuous_deep_public_context_preserved": True,
            "nonregistered_target_labels_masked": True,
            "router_and_deep_validation_keys_equal": True,
            "router_state_partitions_executable": True,
            "deep_supervised_chunks_executable": True,
            "meta_oof_all_layers_supported": True,
            "outer_windows_unchanged": True,
            "model_hyperparameters_unchanged": True,
        },
    }
    receipt["semantic_receipt_sha256"] = v2.canonical_sha256(receipt)
    return plans, receipt


def execute_authorized_curve_v3(
    *,
    observations: pd.DataFrame,
    plans: Sequence[PrefixPlan],
    parent_recipe: Mapping[str, Any],
    config: Mapping[str, Any],
    output_dir: Path,
    contract_sha256: str,
) -> dict[str, Any]:
    """Execute the unchanged 45-cell DAG on the repaired v3 plans."""

    seeds = tuple(
        int(value)
        for value in parent_recipe["authoritative_nested_surrogate_recipe"][
            "complete_pipeline_seed_contract"
        ]["seeds"]
    )
    _require(len(plans) == 15 and len(seeds) == 3, "v3 45-cell graph changed")
    backend = ActualTrainingBackendV3(observations, config)
    jobs = v2.JobStore(output_dir / "jobs", contract_sha256=contract_sha256)
    cell_store = v2.JobStore(output_dir / "cells", contract_sha256=contract_sha256)
    fractions = sorted({plan.fraction for plan in plans})
    by_fraction: dict[float, dict[int, list[pd.DataFrame]]] = {
        fraction: {seed: [] for seed in seeds} for fraction in fractions
    }
    for plan in plans:
        for seed in seeds:
            cell_id = f"cell__{plan.scope_id}__s{seed}"
            product = cell_store.materialize(
                cell_id,
                lambda plan=plan, seed=seed: v2.execute_cell_seed(
                    plan=plan,
                    pipeline_seed=seed,
                    backend=backend,
                    jobs=jobs,
                    gate_regularization=float(config["meta_refit"]["gate_regularization"]),
                ),
            )
            by_fraction[plan.fraction][seed].append(product.frame)
    spec = v2._causal_spec(config)
    context = v2.build_local_context(observations, spec)
    result: dict[str, Any] = {}
    population_digest: str | None = None
    expected_rows = int(config["metrics"]["expected_evaluation_rows_per_fraction"])
    for fraction, by_seed in by_fraction.items():
        frames = {seed: pd.concat(parts, ignore_index=True) for seed, parts in by_seed.items()}
        output, metrics = v2.evaluate_fraction(
            frames=frames,
            context=context,
            spec=spec,
            bootstrap=config["metrics"]["bootstrap"],
        )
        _require(len(output) == expected_rows, "v3 outer evaluation population changed")
        current_digest = v2.canonical_sha256(
            output.loc[:, [*v2.OUTER_KEY_COLUMNS, "truth"]].astype(str).to_dict("records")
        )
        if population_digest is None:
            population_digest = current_digest
        else:
            _require(current_digest == population_digest, "v3 population differs by fraction")
        token = f"{int(round(fraction * 100)):03d}"
        payload = v2._dataframe_bytes(output)
        result[token] = {
            **metrics,
            "ordered_key_truth_sha256": current_digest,
            "evaluated_oof_publish": v2.atomic_write_or_verify(
                output_dir / f"evaluated_oof_{token}.parquet", payload
            ),
        }
    return {
        "status": "COMPLETE_LOCAL_AUTHORITATIVE_SURROGATE_V3_NO_PROMOTION",
        "outer_prefix_cells": 15,
        "seeded_cells": 45,
        "component_jobs_new_this_invocation": jobs.new_jobs,
        "component_jobs_reused_this_invocation": jobs.reused_jobs,
        "cell_jobs_new_this_invocation": cell_store.new_jobs,
        "cell_jobs_reused_this_invocation": cell_store.reused_jobs,
        "metrics_by_prefix": result,
        "top_level_component_jobs_total": 900,
        "underlying_base_estimator_fits_total": 1440,
        "underlying_deep_fits_total": 720,
        "underlying_lightgbm_fits_total": 720,
        "meta_optimizations_total": 405,
        "same_population_digest_across_fractions": population_digest,
        "submission_files_generated": 0,
        "uploads": 0,
    }
