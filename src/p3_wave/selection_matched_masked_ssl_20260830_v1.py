"""Fixed train-only masked-history experiment primitives for P3.

The neural training core is intentionally reused from :mod:`p1_qc.models_ssl`.
This module only supplies the P3 sequence transform, frozen Huber head, and a
paired inference-only reconstruction of the corrected CatBoost reference.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor

from p1_qc.models_ssl import (
    SSLModelConfig,
    SSLTrainConfig,
    extract_ssl_embeddings,
    train_masked_reconstruction,
)
from p3_wave.corrected_repeated_forward import fixed_prequential_lead_router
from p3_wave.features import summarize_context
from p3_wave.loss_router import (
    RouterConfig,
    build_case_router_data,
    expand_case_router_rows,
)
from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.sequences import CONTEXT_ROWS, RAW_COLUMNS
from p3_wave.validation import expand_leads, rmse

LEADS = (3, 6, 9, 12, 18, 24)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
VALUE_CHANNELS = (
    "hs",
    "tp",
    "hmax",
    "wspd",
    "gust",
    "airt",
    "relh",
    "caph",
    "wvdir_sin",
    "wvdir_cos",
    "wdir_sin",
    "wdir_cos",
)


@dataclass(frozen=True)
class RobustTransform:
    median: np.ndarray
    scale: np.ndarray
    clip_abs: float


@dataclass(frozen=True)
class FoldPrediction:
    frame: pd.DataFrame
    receipt: dict[str, Any]


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(str(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def extract_history_sequences(grid: pd.DataFrame, anchors: pd.DataFrame) -> np.ndarray:
    """Extract one closed 48-hour history per anchor without changing row membership."""

    required_grid = {"station", "time", *RAW_COLUMNS}
    required_anchor = {"anchor_id", "station", "grid_position"}
    if missing := required_grid.difference(grid.columns):
        raise ValueError(f"grid columns missing: {sorted(missing)}")
    if missing := required_anchor.difference(anchors.columns):
        raise ValueError(f"anchor columns missing: {sorted(missing)}")
    if anchors["anchor_id"].duplicated().any():
        raise ValueError("anchor_id must be unique")
    result = np.empty((len(anchors), CONTEXT_ROWS, len(RAW_COLUMNS)), dtype=np.float32)
    by_station = {
        str(station): part.sort_values("time").reset_index(drop=True)
        for station, part in grid.groupby("station", sort=False, observed=True)
    }
    for output_row, row in enumerate(anchors.itertuples(index=False)):
        source = by_station[str(row.station)]
        stop = int(row.grid_position) + 1
        start = stop - CONTEXT_ROWS
        if start < 0 or stop > len(source):
            raise ValueError("anchor lacks a closed 48-hour history")
        result[output_row] = source.iloc[start:stop][list(RAW_COLUMNS)].to_numpy(
            dtype=np.float32
        )
    if result.shape[1:] != (CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise AssertionError("history extraction shape changed")
    return result


def _physical_channels(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (CONTEXT_ROWS, len(RAW_COLUMNS)):
        raise ValueError("raw histories must have shape (cases, 289, 10)")
    scalar_indices = (0, 1, 2, 4, 5, 7, 8, 9)
    scalar = values[:, :, scalar_indices]
    scalar_mask = np.isfinite(scalar)
    direction_parts: list[np.ndarray] = []
    direction_masks: list[np.ndarray] = []
    for index in (3, 6):
        direction = values[:, :, index]
        finite = np.isfinite(direction)
        radians = np.deg2rad(direction)
        direction_parts.extend((np.sin(radians), np.cos(radians)))
        direction_masks.extend((finite, finite))
    physical = np.concatenate(
        [scalar, *(part[:, :, None] for part in direction_parts)], axis=2
    ).astype(np.float32)
    observed = np.concatenate(
        [scalar_mask, *(part[:, :, None] for part in direction_masks)], axis=2
    )
    if physical.shape[2] != len(VALUE_CHANNELS) or observed.shape != physical.shape:
        raise AssertionError("physical channel transform changed")
    return physical, observed


def fit_sequence_transform(raw_train: np.ndarray, *, clip_abs: float) -> RobustTransform:
    physical, observed = _physical_channels(raw_train)
    median = np.empty(physical.shape[2], dtype=np.float32)
    scale = np.empty(physical.shape[2], dtype=np.float32)
    for channel in range(physical.shape[2]):
        selected = physical[:, :, channel][observed[:, :, channel]]
        if len(selected) == 0:
            raise ValueError(f"training channel has no finite values: {VALUE_CHANNELS[channel]}")
        median[channel] = float(np.median(selected))
        q25, q75 = np.quantile(selected, [0.25, 0.75])
        width = float(q75 - q25)
        scale[channel] = width if np.isfinite(width) and width > 1.0e-8 else 1.0
    if not np.isfinite(median).all() or not np.isfinite(scale).all():
        raise ValueError("non-finite train-only robust transform")
    return RobustTransform(median=median, scale=scale, clip_abs=float(clip_abs))


def transform_sequences(raw: np.ndarray, transform: RobustTransform) -> np.ndarray:
    physical, observed = _physical_channels(raw)
    normalized = (physical - transform.median[None, None, :]) / transform.scale[
        None, None, :
    ]
    normalized = np.clip(normalized, -transform.clip_abs, transform.clip_abs)
    normalized = np.where(observed, normalized, 0.0).astype(np.float32)
    result = np.concatenate([normalized, observed.astype(np.float32)], axis=2)
    if result.shape[:2] != physical.shape[:2] or result.shape[2] != 2 * len(VALUE_CHANNELS):
        raise AssertionError("masked representation input shape changed")
    if not np.isfinite(result).all():
        raise ValueError("representation input contains non-finite values")
    return result


def _flatten_ssl_inputs(
    values: np.ndarray, anchor_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    identifiers = np.asarray(anchor_ids, dtype=np.int64)
    if identifiers.shape != (len(values),) or len(np.unique(identifiers)) != len(identifiers):
        raise ValueError("SSL anchor ids must be aligned and unique")
    segments = np.repeat(identifiers, CONTEXT_ROWS)
    steps = np.tile(np.arange(CONTEXT_ROWS, dtype=np.int64), len(identifiers))
    row_ids = segments * CONTEXT_ROWS + steps
    if len(np.unique(row_ids)) != len(row_ids):
        raise AssertionError("SSL provenance ids collided")
    return values.reshape(-1, values.shape[2]), segments, row_ids


def _aggregate_embeddings(embeddings: np.ndarray, case_count: int) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float32).reshape(case_count, CONTEXT_ROWS, -1)
    result = np.concatenate(
        [values.mean(axis=1), values.std(axis=1), values[:, -1, :]], axis=1
    ).astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("aggregated SSL embedding is non-finite")
    return result


def fit_masked_embeddings(
    raw_train: np.ndarray,
    raw_validation: np.ndarray,
    train_anchor_ids: np.ndarray,
    validation_anchor_ids: np.ndarray,
    *,
    representation_config: dict[str, Any],
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit one fold-local encoder and return frozen train/validation embeddings."""

    normalization = representation_config["normalization"]
    transform = fit_sequence_transform(
        raw_train, clip_abs=float(normalization["normalized_clip_abs"])
    )
    train_values = transform_sequences(raw_train, transform)
    validation_values = transform_sequences(raw_validation, transform)
    train_flat, train_segments, train_rows = _flatten_ssl_inputs(
        train_values, train_anchor_ids
    )
    validation_flat, validation_segments, _ = _flatten_ssl_inputs(
        validation_values, validation_anchor_ids
    )
    model_spec = representation_config["model"]
    train_spec = representation_config["masked_training"]
    model_config = SSLModelConfig(
        input_dim=train_values.shape[2],
        channels=tuple(int(value) for value in model_spec["channels"]),
        kernel_size=int(model_spec["kernel_size"]),
        dropout=float(model_spec["dropout"]),
        causal=bool(model_spec["causal"]),
    )
    train_config = SSLTrainConfig(
        window_steps=int(train_spec["window_steps"]),
        stride_steps=int(train_spec["stride_steps"]),
        mask_fraction=float(train_spec["mask_fraction"]),
        mask_block_steps=int(train_spec["mask_block_steps"]),
        batch_size=int(train_spec["batch_size"]),
        max_epochs=int(train_spec["maximum_epochs"]),
        patience=int(train_spec["patience"]),
        learning_rate=float(train_spec["learning_rate"]),
        weight_decay=float(train_spec["weight_decay"]),
        use_bfloat16=bool(train_spec["use_bfloat16"]),
        seed=int(seed),
        deterministic=bool(train_spec["deterministic"]),
        normalized_abs_limit=max(100.0, float(normalization["normalized_clip_abs"])),
    )
    started = time.perf_counter()
    fitted = train_masked_reconstruction(
        train_flat,
        train_segments,
        train_rows,
        model_config=model_config,
        train_config=train_config,
        device=device,
    )
    train_embedding = extract_ssl_embeddings(
        fitted, train_flat, train_segments, device=device
    )
    validation_embedding = extract_ssl_embeddings(
        fitted, validation_flat, validation_segments, device=device
    )
    elapsed = float(time.perf_counter() - started)
    train_aggregate = _aggregate_embeddings(train_embedding, len(raw_train))
    validation_aggregate = _aggregate_embeddings(validation_embedding, len(raw_validation))
    receipt = {
        "fit_count": 1,
        "seed": int(seed),
        "device": device,
        "train_cases": int(len(raw_train)),
        "outer_validation_cases_exposed_to_ssl_fit_or_early_stop": 0,
        "epochs_completed": int(len(fitted.history)),
        "best_epoch_zero_based": int(fitted.best_epoch),
        "best_masked_reconstruction_loss": float(fitted.best_validation_loss),
        "history": [
            {
                "epoch": int(item["epoch"]),
                "train_loss": float(item["train_loss"]),
                "train_remask_monitor_loss": float(item["validation_loss"]),
            }
            for item in fitted.history
        ],
        "encoder_parameter_count": int(sum(p.numel() for p in fitted.model.parameters())),
        "aggregated_embedding_dimension": int(train_aggregate.shape[1]),
        "transform_sha256": _array_sha256(transform.median, transform.scale),
        "elapsed_seconds": elapsed,
    }
    return train_aggregate, validation_aggregate, receipt


def _head_design(embedding: np.ndarray, anchors: pd.DataFrame) -> np.ndarray:
    if len(embedding) != len(anchors):
        raise ValueError("embedding and anchor rows differ")
    current = anchors["current_hs"].to_numpy(dtype=float)[:, None]
    rise = anchors["rise_12h"].to_numpy(dtype=float)[:, None]
    base = np.concatenate([np.asarray(embedding, dtype=float), current, rise], axis=1)
    repeated = np.repeat(base, len(LEADS), axis=0)
    station = anchors["station"].astype(str).to_numpy()
    station_one_hot = np.column_stack([station == value for value in STATIONS]).astype(float)
    station_one_hot = np.repeat(station_one_hot, len(LEADS), axis=0)
    lead_one_hot = np.tile(np.eye(len(LEADS), dtype=float), (len(anchors), 1))
    design = np.concatenate([repeated, station_one_hot, lead_one_hot], axis=1)
    if not np.isfinite(design).all():
        raise ValueError("Huber design contains non-finite values")
    return design


def _head_targets(anchors: pd.DataFrame) -> np.ndarray:
    current = anchors["current_hs"].to_numpy(dtype=float)
    targets = np.column_stack(
        [anchors[f"target_{lead}"].to_numpy(dtype=float) for lead in LEADS]
    )
    return (targets - current[:, None]).reshape(-1)


def _make_prediction_frame(
    anchors: pd.DataFrame, prediction: np.ndarray, *, fold: str, column: str
) -> pd.DataFrame:
    values = np.asarray(prediction, dtype=float)
    if values.shape != (len(anchors) * len(LEADS),):
        raise ValueError("prediction row count differs from six-lead anchors")
    targets = np.column_stack(
        [anchors[f"target_{lead}"].to_numpy(dtype=float) for lead in LEADS]
    ).reshape(-1)
    current = np.repeat(anchors["current_hs"].to_numpy(dtype=float), len(LEADS))
    return pd.DataFrame(
        {
            "fold": fold,
            "anchor_id": np.repeat(anchors["anchor_id"].to_numpy(dtype=np.int64), len(LEADS)),
            "station": np.repeat(anchors["station"].astype(str).to_numpy(), len(LEADS)),
            "lead_h": np.tile(np.asarray(LEADS, dtype=int), len(anchors)),
            "current_hs": current,
            "target_hs": targets,
            column: values,
        }
    )


def fit_candidate_fold(
    raw_train: np.ndarray,
    raw_validation: np.ndarray,
    train_anchors: pd.DataFrame,
    validation_anchors: pd.DataFrame,
    *,
    fold: str,
    representation_config: dict[str, Any],
    head_config: dict[str, Any],
    seed: int,
    device: str,
) -> FoldPrediction:
    train_embedding, validation_embedding, ssl_receipt = fit_masked_embeddings(
        raw_train,
        raw_validation,
        train_anchors["anchor_id"].to_numpy(dtype=np.int64),
        validation_anchors["anchor_id"].to_numpy(dtype=np.int64),
        representation_config=representation_config,
        seed=seed,
        device=device,
    )
    x_train = _head_design(train_embedding, train_anchors)
    x_validation = _head_design(validation_embedding, validation_anchors)
    y_train = _head_targets(train_anchors)
    median = np.median(x_train, axis=0)
    q25, q75 = np.quantile(x_train, [0.25, 0.75], axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale <= 1.0e-8)] = 1.0
    x_train = (x_train - median) / scale
    x_validation = (x_validation - median) / scale
    model = HuberRegressor(
        epsilon=float(head_config["epsilon"]),
        alpha=float(head_config["alpha"]),
        max_iter=int(head_config["max_iter"]),
        tol=float(head_config["tol"]),
        fit_intercept=bool(head_config["fit_intercept"]),
    )
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, y_train)
    head_elapsed = float(time.perf_counter() - started)
    residual = np.asarray(model.predict(x_validation), dtype=float)
    current = np.repeat(
        validation_anchors["current_hs"].to_numpy(dtype=float), len(LEADS)
    )
    lower, upper = map(float, head_config["prediction_clip_m"])
    prediction = np.clip(current + residual, lower, upper)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Huber head produced non-finite predictions")
    frame = _make_prediction_frame(
        validation_anchors, prediction, fold=fold, column="candidate_prediction"
    )
    receipt = {
        "fold": fold,
        "train_cases": int(len(train_anchors)),
        "validation_cases": int(len(validation_anchors)),
        "train_supervised_rows": int(len(y_train)),
        "validation_prediction_rows": int(len(prediction)),
        "ssl": ssl_receipt,
        "huber": {
            "fit_count": 1,
            "n_iter": int(model.n_iter_),
            "convergence_warning_count": int(
                sum(issubclass(item.category, ConvergenceWarning) for item in caught)
            ),
            "design_columns": int(x_train.shape[1]),
            "design_transform_sha256": _array_sha256(median, scale),
            "elapsed_seconds": head_elapsed,
        },
        "rows_deleted": 0,
    }
    return FoldPrediction(frame=frame, receipt=receipt)


def summarize_validation_histories(
    raw_validation: np.ndarray, validation_anchors: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for raw, anchor in zip(raw_validation, validation_anchors.itertuples(index=False), strict=True):
        row: dict[str, Any] = {
            "anchor_id": int(anchor.anchor_id),
            "station": str(anchor.station),
        }
        row.update(summarize_context(pd.DataFrame(raw, columns=RAW_COLUMNS)))
        rows.append(row)
    return pd.DataFrame(rows)


def saved_catboost_component_predictions(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    fold_order: tuple[str, ...],
    feature_columns: list[str],
    model_root: Path,
) -> pd.DataFrame:
    """Infer with the six saved fold models; no CatBoost ``fit`` is called."""

    outputs: list[pd.DataFrame] = []
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    for fold in fold_order:
        fold_anchor = anchors.loc[anchors["fold"].astype(str).eq(fold)].copy()
        anchor_ids = fold_anchor["anchor_id"].to_numpy(dtype=np.int64)
        x_single, _, metadata = expand_leads(
            features, anchors, anchor_ids, feature_columns
        )
        x_single = x_single.copy()
        x_single["station"] = x_single["station"].astype(str)
        x_single["lead_h"] = x_single["lead_h"].astype(str)
        single = CatBoostRegressor()
        single.load_model(model_root / fold / "single.cbm")
        single_prediction = np.clip(
            metadata["current_hs"].to_numpy(dtype=float) + single.predict(x_single),
            0.0,
            30.0,
        )
        multi_x = feature_lookup.loc[
            anchor_ids, ["station", *feature_columns]
        ].reset_index(drop=True)
        multi_x["station"] = multi_x["station"].astype(str)
        multi = CatBoostRegressor()
        multi.load_model(model_root / fold / "multi.cbm")
        multi_delta = np.asarray(multi.predict(multi_x), dtype=float)
        current_case = anchor_lookup.loc[anchor_ids, "current_hs"].to_numpy(dtype=float)
        multi_prediction = np.clip(current_case[:, None] + multi_delta, 0.0, 30.0)
        multi_frame = pd.DataFrame(
            {
                "anchor_id": np.repeat(anchor_ids, len(LEADS)),
                "station": np.repeat(
                    anchor_lookup.loc[anchor_ids, "station"].astype(str).to_numpy(),
                    len(LEADS),
                ),
                "lead_h": np.tile(np.asarray(LEADS, dtype=int), len(anchor_ids)),
                "multi_prediction": multi_prediction.reshape(-1),
            }
        )
        current = metadata.copy()
        current["fold"] = fold
        current["single_prediction"] = single_prediction
        current = current.merge(
            multi_frame,
            on=["anchor_id", "station", "lead_h"],
            how="left",
            validate="one_to_one",
        )
        current["persistence"] = current["current_hs"]
        outputs.append(current)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(["fold", "anchor_id", "station", "lead_h"]).any():
        raise ValueError("saved CatBoost component keys duplicated")
    return result


def apply_paired_prequential_reference(
    component_oof: pd.DataFrame,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    *,
    fold_order: tuple[str, ...],
    reference_config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    case_x, case_metadata, case_components, _ = build_case_router_data(
        component_oof, features, anchors
    )
    lookup = anchors.set_index("anchor_id")
    truth = np.column_stack(
        [
            lookup.loc[case_metadata["anchor_id"], f"target_{lead}"].to_numpy(dtype=float)
            for lead in LEADS
        ]
    )
    row_x, row_metadata, row_components, row_losses = expand_case_router_rows(
        case_x, case_metadata, case_components, truth
    )
    router_spec = reference_config["router"]
    routed, weights, receipts = fixed_prequential_lead_router(
        row_x,
        row_metadata,
        row_components,
        row_losses,
        fold_order=fold_order,
        config=RouterConfig(
            alpha=float(router_spec["alpha"]),
            temperature_multiplier=float(router_spec["temperature_multiplier"]),
            strength=float(router_spec["strength"]),
            name=str(router_spec["name"]),
        ),
        active_leads=tuple(int(value) for value in router_spec["active_leads_hours"]),
    )
    shrink_spec = reference_config["long_lead_persistence_shrink"]
    persistence = row_components[:, 2]
    final = apply_long_lead_persistence_shrink(
        routed,
        persistence,
        row_metadata["lead_h"].to_numpy(dtype=int),
        config=LongLeadPersistenceShrink(
            weight=float(shrink_spec["weight"]),
            active_leads=tuple(int(value) for value in shrink_spec["active_leads_hours"]),
        ),
    )
    output = row_metadata[["fold", "anchor_id", "station", "lead_h"]].copy()
    output["incumbent_prediction"] = final
    output["persistence"] = persistence
    output = output.merge(
        component_oof[["fold", "anchor_id", "station", "lead_h", "target_hs", "current_hs"]],
        on=["fold", "anchor_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    fit_count = int(sum(item["past_fit_rows"] > 0 for item in receipts))
    receipt = {
        "catboost_fit_count": 0,
        "catboost_model_load_count": int(2 * len(fold_order)),
        "fixed_router_fit_count": fit_count,
        "router_receipts": receipts,
        "weight_summary": {
            name: {
                "mean": float(weights[:, index].mean()),
                "minimum": float(weights[:, index].min()),
                "maximum": float(weights[:, index].max()),
            }
            for index, name in enumerate(("single", "multi", "persistence"))
        },
    }
    return output, receipt


def _comparison_slice(frame: pd.DataFrame) -> dict[str, Any]:
    truth = frame["target_hs"].to_numpy(dtype=float)
    candidate = frame["candidate_prediction"].to_numpy(dtype=float)
    incumbent = frame["incumbent_prediction"].to_numpy(dtype=float)
    persistence = frame["persistence"].to_numpy(dtype=float)
    candidate_rmse = rmse(truth, candidate)
    incumbent_rmse = rmse(truth, incumbent)
    return {
        "cases": int(frame["anchor_id"].nunique()),
        "rows": int(len(frame)),
        "candidate_rmse_m": candidate_rmse,
        "paired_incumbent_rmse_m": incumbent_rmse,
        "persistence_rmse_m": rmse(truth, persistence),
        "delta_candidate_minus_incumbent_m": candidate_rmse - incumbent_rmse,
    }


def comparison_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    required = {
        "fold",
        "anchor_id",
        "station",
        "lead_h",
        "target_hs",
        "candidate_prediction",
        "incumbent_prediction",
        "persistence",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"comparison columns missing: {sorted(missing)}")
    if frame.duplicated(["anchor_id", "lead_h"]).any():
        raise ValueError("comparison keys duplicated")
    return {
        "overall": _comparison_slice(frame),
        "by_window": {
            str(key): _comparison_slice(group)
            for key, group in frame.groupby("fold", sort=True, observed=True)
        },
        "by_station": {
            str(key): _comparison_slice(group)
            for key, group in frame.groupby("station", sort=True, observed=True)
        },
        "by_lead": {
            str(int(key)): _comparison_slice(group)
            for key, group in frame.groupby("lead_h", sort=True, observed=True)
        },
    }


def paired_case_bootstrap(
    frame: pd.DataFrame, *, replicates: int, seed: int
) -> dict[str, Any]:
    ordered = frame.sort_values(["anchor_id", "lead_h"]).reset_index(drop=True)
    counts = ordered.groupby("anchor_id", sort=False)["lead_h"].agg(tuple)
    if not counts.map(lambda value: tuple(int(item) for item in value) == LEADS).all():
        raise ValueError("bootstrap cases must contain the six official ordered leads")
    cases = len(counts)
    truth = ordered["target_hs"].to_numpy(dtype=float).reshape(cases, len(LEADS))
    candidate = ordered["candidate_prediction"].to_numpy(dtype=float).reshape(
        cases, len(LEADS)
    )
    incumbent = ordered["incumbent_prediction"].to_numpy(dtype=float).reshape(
        cases, len(LEADS)
    )
    rng = np.random.default_rng(int(seed))
    delta = np.empty(int(replicates), dtype=float)
    for number in range(int(replicates)):
        selected = rng.integers(0, cases, size=cases)
        delta[number] = rmse(truth[selected], candidate[selected]) - rmse(
            truth[selected], incumbent[selected]
        )
    return {
        "unit": "complete_six_lead_anchor_case",
        "cases": int(cases),
        "replicates": int(replicates),
        "seed": int(seed),
        "delta_candidate_minus_incumbent_ci90_m": [
            float(value) for value in np.quantile(delta, [0.05, 0.95])
        ],
        "delta_candidate_minus_incumbent_median_m": float(np.median(delta)),
    }


def evaluate_promotion_gate(
    metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    *,
    gate_config: dict[str, Any],
    integrity_checks: dict[str, bool],
) -> dict[str, Any]:
    overall_delta = float(metrics["overall"]["delta_candidate_minus_incumbent_m"])
    window_deltas = [
        float(item["delta_candidate_minus_incumbent_m"])
        for item in metrics["by_window"].values()
    ]
    lead_deltas = [
        float(item["delta_candidate_minus_incumbent_m"])
        for item in metrics["by_lead"].values()
    ]
    minimum = float(gate_config["minimum_pooled_improvement_vs_paired_incumbent_m"])
    checks = {
        "all_integrity_checks_pass": bool(integrity_checks) and all(integrity_checks.values()),
        "pooled_improvement_at_least_preregistered_margin": overall_delta <= -minimum,
        "paired_case_ci90_upper_below_zero": float(
            bootstrap["delta_candidate_minus_incumbent_ci90_m"][1]
        )
        < 0.0,
        "minimum_improved_forward_windows": int(sum(value < 0.0 for value in window_deltas))
        >= int(gate_config["minimum_improved_forward_windows"]),
        "worst_lead_regression_within_cap": max(lead_deltas)
        <= float(gate_config["maximum_worst_lead_regression_m"]),
    }
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "decision": (
            "RESEARCH_PROMOTION_CANDIDATE_ONLY_NO_OFFICIAL_INFERENCE"
            if passed
            else "NO_GO_CLOSE_THIS_EXACT_RECIPE"
        ),
        "checks": checks,
        "integrity_checks": integrity_checks,
        "improved_forward_window_count": int(sum(value < 0.0 for value in window_deltas)),
        "worst_lead_delta_candidate_minus_incumbent_m": float(max(lead_deltas)),
    }


def recipe_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Return a compact immutable recipe summary used by static tests and receipts."""

    return {
        "official_leads_hours": list(LEADS),
        "history_rows": CONTEXT_ROWS,
        "raw_columns": list(RAW_COLUMNS),
        "value_channels": list(VALUE_CHANNELS),
        "fit_budget": dict(config["fit_and_runtime_budget"]),
        "representation_model": dict(config["representation"]["model"]),
        "masked_training": dict(config["representation"]["masked_training"]),
        "robust_head": dict(config["robust_residual_head"]),
        "reference_router": dict(config["paired_incumbent_reference"]["router"]),
        "normalization": asdict(
            RobustTransform(
                median=np.zeros(len(VALUE_CHANNELS), dtype=np.float32),
                scale=np.ones(len(VALUE_CHANNELS), dtype=np.float32),
                clip_abs=float(
                    config["representation"]["normalization"]["normalized_clip_abs"]
                ),
            )
        )["clip_abs"],
    }
