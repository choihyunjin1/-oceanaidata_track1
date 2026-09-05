"""Minimal clean recipe functions extracted without algorithm changes."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .corrected_repeated_forward import (
    CorrectedFold,
    evaluate_candidate_gate,
    fixed_prequential_lead_router,
    paired_case_bootstrap,
)
from .loss_router import RouterConfig, build_case_router_data, expand_case_router_rows
from .models import threshold_case_weights
from .persistence_shrink import LongLeadPersistenceShrink, apply_long_lead_persistence_shrink
from .validation import expand_leads, metric_slices, rmse

LEADS = (3, 6, 9, 12, 18, 24)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cat_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["station"] = result["station"].astype(str)
    result["lead_h"] = result["lead_h"].astype(str)
    return result


def _single_model(config: dict[str, Any], seed: int) -> CatBoostRegressor:
    parameters = dict(config["model"]["single"])
    return CatBoostRegressor(
        **parameters,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )


def _multi_model(config: dict[str, Any], seed: int) -> CatBoostRegressor:
    parameters = dict(config["model"]["multi"])
    return CatBoostRegressor(
        **parameters,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
    )


def _multi_target(anchors: pd.DataFrame, anchor_ids: np.ndarray) -> np.ndarray:
    lookup = anchors.set_index("anchor_id")
    current = lookup.loc[anchor_ids, "current_hs"].to_numpy(dtype=float)
    return np.column_stack(
        [lookup.loc[anchor_ids, f"target_{lead}"].to_numpy(dtype=float) - current for lead in LEADS]
    )


def _multi_validation_frame(
    anchors: pd.DataFrame, anchor_ids: np.ndarray, prediction: np.ndarray
) -> pd.DataFrame:
    lookup = anchors.set_index("anchor_id")
    current = lookup.loc[anchor_ids, "current_hs"].to_numpy(dtype=float)
    absolute = np.clip(current[:, None] + prediction, 0.0, 30.0)
    return pd.DataFrame(
        {
            "anchor_id": np.repeat(anchor_ids, len(LEADS)),
            "station": np.repeat(
                lookup.loc[anchor_ids, "station"].astype(str).to_numpy(), len(LEADS)
            ),
            "lead_h": np.tile(np.asarray(LEADS, dtype=int), len(anchor_ids)),
            "multi_prediction": absolute.reshape(-1),
        }
    )


def _fit_fold_components(
    *,
    fold: CorrectedFold,
    fold_number: int,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    feature_columns: list[str],
    config: dict[str, Any],
    model_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed = int(config["model"]["fold_seeds"][fold_number])
    started = time.perf_counter()
    x_train, y_train, train_meta = expand_leads(features, anchors, fold.train_ids, feature_columns)
    x_valid, _, valid_meta = expand_leads(features, anchors, fold.validation_ids, feature_columns)
    single = _single_model(config, seed)
    single.fit(
        _cat_frame(x_train),
        y_train,
        sample_weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    single_prediction = np.clip(
        valid_meta["current_hs"].to_numpy(dtype=float) + single.predict(_cat_frame(x_valid)),
        0.0,
        30.0,
    )

    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    multi_x_train = feature_lookup.loc[fold.train_ids, ["station", *feature_columns]].reset_index(
        drop=True
    )
    multi_x_valid = feature_lookup.loc[
        fold.validation_ids, ["station", *feature_columns]
    ].reset_index(drop=True)
    multi_x_train["station"] = multi_x_train["station"].astype(str)
    multi_x_valid["station"] = multi_x_valid["station"].astype(str)
    multi = _multi_model(config, seed)
    multi.fit(
        multi_x_train,
        _multi_target(anchors, fold.train_ids),
        sample_weight=threshold_case_weights(
            anchor_lookup.loc[fold.train_ids, "current_hs"].to_numpy(dtype=float)
        ),
        cat_features=[0],
        verbose=False,
    )
    multi_delta = np.asarray(multi.predict(multi_x_valid), dtype=float)
    multi_frame = _multi_validation_frame(anchors, fold.validation_ids, multi_delta)

    oof = valid_meta.copy()
    oof["fold"] = fold.name
    oof["single_prediction"] = single_prediction
    oof = oof.merge(
        multi_frame,
        on=["anchor_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    oof["persistence"] = oof["current_hs"]
    oof["equal_prediction"] = 0.5 * (oof["single_prediction"] + oof["multi_prediction"])
    if not np.isfinite(
        oof[["single_prediction", "multi_prediction", "equal_prediction"]].to_numpy()
    ).all():
        raise ValueError(f"non-finite fold component prediction: {fold.name}")

    destination = model_dir / fold.name
    destination.mkdir(parents=True, exist_ok=True)
    single_path = destination / "single.cbm"
    multi_path = destination / "multi.cbm"
    single.save_model(single_path)
    multi.save_model(multi_path)
    receipt = {
        "fold": fold.name,
        "seed": seed,
        "train_anchor_count": int(len(fold.train_ids)),
        "train_single_rows": int(len(x_train)),
        "validation_case_count": int(len(fold.validation_ids)),
        "validation_rows": int(len(oof)),
        "elapsed_seconds": float(time.perf_counter() - started),
        "model_sha256": {
            "single.cbm": sha256_file(single_path),
            "multi.cbm": sha256_file(multi_path),
        },
    }
    return oof, receipt


def _router_config(config: dict[str, Any]) -> RouterConfig:
    router = config["router"]
    return RouterConfig(
        alpha=float(router["alpha"]),
        temperature_multiplier=float(router["temperature_multiplier"]),
        strength=float(router["strength"]),
        name=str(router["name"]),
    )


def _evaluate_fixed_structure(
    *,
    component_oof: pd.DataFrame,
    train_features: pd.DataFrame,
    anchors: pd.DataFrame,
    fold_order: tuple[str, ...],
    config: dict[str, Any],
    split_audit: dict[str, Any],
    expected_validation_ids: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    oof = component_oof.copy()
    if oof.duplicated(["fold", "anchor_id", "station", "lead_h"]).any():
        raise ValueError("component OOF keys are duplicated")
    if set(oof["lead_h"].astype(int)) != set(LEADS):
        raise ValueError("component OOF has unexpected leads")
    if set(oof["anchor_id"].astype(int)) != set(expected_validation_ids.astype(int)):
        raise ValueError("component OOF cases differ from corrected validation selection")
    case_leads = oof.groupby(["fold", "anchor_id"], observed=True)["lead_h"].agg(
        lambda values: tuple(sorted(values.astype(int)))
    )
    if not case_leads.map(lambda value: value == LEADS).all():
        raise ValueError("component OOF cases are not complete")

    case_x, case_meta, case_components, _ = build_case_router_data(oof, train_features, anchors)
    anchor_lookup = anchors.set_index("anchor_id")
    truth = np.column_stack(
        [
            anchor_lookup.loc[case_meta["anchor_id"], f"target_{lead}"].to_numpy(dtype=float)
            for lead in LEADS
        ]
    )
    row_x, row_meta, row_components, row_losses = expand_case_router_rows(
        case_x, case_meta, case_components, truth
    )
    routed, weights, router_receipts = fixed_prequential_lead_router(
        row_x,
        row_meta,
        row_components,
        row_losses,
        fold_order=fold_order,
        config=_router_config(config),
        active_leads=tuple(config["router"]["active_leads"]),
    )
    routed_frame = row_meta[["fold", "anchor_id", "station", "lead_h"]].copy()
    routed_frame["routed_prediction"] = routed
    routed_frame[["weight_single", "weight_multi", "weight_persistence"]] = weights
    oof = oof.merge(
        routed_frame,
        on=["fold", "anchor_id", "station", "lead_h"],
        how="left",
        validate="one_to_one",
    )
    shrink = LongLeadPersistenceShrink(
        weight=float(config["shrink"]["persistence_weight"]),
        active_leads=tuple(config["shrink"]["active_leads"]),
    )
    oof["final_prediction"] = apply_long_lead_persistence_shrink(
        oof["routed_prediction"].to_numpy(dtype=float),
        oof["persistence"].to_numpy(dtype=float),
        oof["lead_h"].to_numpy(dtype=int),
        config=shrink,
    )
    short = oof["lead_h"].isin([3, 6, 9]).to_numpy()
    short_router_error = float(
        np.max(
            np.abs(
                oof.loc[short, "routed_prediction"].to_numpy(dtype=float)
                - oof.loc[short, "equal_prediction"].to_numpy(dtype=float)
            )
        )
    )
    short_shrink_error = float(
        np.max(
            np.abs(
                oof.loc[short, "final_prediction"].to_numpy(dtype=float)
                - oof.loc[short, "routed_prediction"].to_numpy(dtype=float)
            )
        )
    )
    numeric = oof[
        [
            "target_hs",
            "single_prediction",
            "multi_prediction",
            "equal_prediction",
            "routed_prediction",
            "final_prediction",
            "persistence",
        ]
    ].to_numpy(dtype=float)
    finite_and_range = bool(
        np.isfinite(numeric).all() and np.all(oof["final_prediction"].between(0.0, 30.0).to_numpy())
    )
    metrics = {
        name: metric_slices(oof, oof[column].to_numpy(dtype=float))
        for name, column in (
            ("single", "single_prediction"),
            ("multi", "multi_prediction"),
            ("equal", "equal_prediction"),
            ("routed", "routed_prediction"),
            ("final", "final_prediction"),
            ("persistence", "persistence"),
        )
    }
    metrics["folds"] = {
        str(name): {
            "final": metric_slices(group, group["final_prediction"].to_numpy(dtype=float)),
            "persistence": metric_slices(group, group["persistence"].to_numpy(dtype=float)),
            "delta_final_minus_persistence_m": rmse(group["target_hs"], group["final_prediction"])
            - rmse(group["target_hs"], group["persistence"]),
        }
        for name, group in oof.groupby("fold", sort=True, observed=True)
    }
    bootstrap = paired_case_bootstrap(
        oof,
        candidate_column="final_prediction",
        baseline_column="persistence",
        replicates=int(config["validation"]["bootstrap_replicates"]),
        seed=int(config["validation"]["bootstrap_seed"]),
    )
    contract_checks = {
        "station_global_gap_at_least_78h": all(
            value >= 78.0 for value in split_audit["station_global_minimum_gap_hours"].values()
        ),
        "validation_storm_episode_distinct": split_audit["repeated_station_episode_count"] == 0,
        "validation_72h_footprints_disjoint": split_audit[
            "context48_plus_target24_footprint_overlap_pairs"
        ]
        == 0,
        "all_fold_train_validation_episodes_disjoint": all(
            row["shared_train_validation_station_episode_count"] == 0
            for row in split_audit["folds"].values()
        ),
        "all_fold_train_validation_gaps_at_least_78h": all(
            row["minimum_train_validation_anchor_gap_hours"] >= 78.0
            for row in split_audit["folds"].values()
        ),
        "complete_unique_oof_keys": len(oof) == split_audit["validation_case_count"] * len(LEADS),
        "fixed_router_no_current_fold_target": all(
            not row["current_fold_target_used_for_router"] for row in router_receipts
        ),
        "short_lead_router_is_exact_equal_ensemble": short_router_error <= 1e-12,
        "short_lead_shrink_is_exact_no_op": short_shrink_error == 0.0,
        "finite_and_range_valid": finite_and_range,
        "hyperparameter_search_run_zero": config["router"]["hyperparameter_search"] is False,
    }
    gate = evaluate_candidate_gate(
        oof,
        bootstrap=bootstrap,
        contract_checks=contract_checks,
        minimum_improved_folds=int(config["gate"]["minimum_improved_folds"]),
    )
    detail = {
        "metrics": metrics,
        "paired_case_bootstrap": bootstrap,
        "gate": gate,
        "router_receipts": router_receipts,
        "router_weight_summary": {
            name: {
                "mean": float(weights[:, index].mean()),
                "p10": float(np.quantile(weights[:, index], 0.10)),
                "p90": float(np.quantile(weights[:, index], 0.90)),
            }
            for index, name in enumerate(("single", "multi", "persistence"))
        },
        "short_lead_max_abs_error": {
            "router_vs_equal": short_router_error,
            "shrink_vs_router": short_shrink_error,
        },
    }
    router_material = {
        "row_features": row_x,
        "row_losses": row_losses,
        "case_features": case_x,
        "case_metadata": case_meta,
        "case_components": case_components,
    }
    return oof, detail, router_material
