"""Run the append-only P3 corrected repeated-forward CatBoost evaluation.

The run performs no model-family or hyperparameter search.  Anonymous test features and
``test_index`` are parsed only after the preregistered corrected-validation gate passes.
It creates a local candidate and never mutates or uploads the frozen/current submission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from catboost.utils import get_gpu_device_count

from p3_wave.corrected_repeated_forward import (
    ACTIVE_ROUTER_LEADS,
    CorrectedFold,
    build_corrected_repeated_forward_folds,
    evaluate_candidate_gate,
    fixed_prequential_lead_router,
    paired_case_bootstrap,
)
from p3_wave.loss_router import (
    OBSERVED_FEATURES,
    ComponentLossRouter,
    RouterConfig,
    build_case_router_data,
    build_inference_router_features,
    expand_case_router_features,
    expand_case_router_rows,
    route_row_predictions,
)
from p3_wave.models import compact_feature_columns, threshold_case_weights
from p3_wave.persistence_shrink import (
    LongLeadPersistenceShrink,
    apply_long_lead_persistence_shrink,
)
from p3_wave.revin_patch import assign_storm_episodes_from_wave
from p3_wave.submission import build_submission, validate_submission, write_submission
from p3_wave.validation import expand_leads, metric_slices, rmse

LEADS = (3, 6, 9, 12, 18, 24)
KEYS = ["case_id", "station", "lead_h"]
DEFAULT_CONFIG = "configs/experiments/p3_corrected_repeated_forward_catboost_v1.json"
DEFAULT_OUTPUT = "artifacts/p3_corrected_repeated_forward_catboost_v1"

SOURCE_FILE_MAP = {
    "source/train_wave.csv": "train_wave.csv",
    "source/train_atmos.csv": "train_atmos.csv",
    "source/test_context.parquet": "test_context.parquet",
    "source/test_index.csv": "test_index.csv",
    "source/sample_submission.csv": "sample_submission.csv",
    "source/baseline_persistence.csv": "baseline_persistence.csv",
}
CACHE_FILE_MAP = {
    "cache/manifest.json": "manifest.json",
    "cache/train_features.parquet": "train_features.parquet",
    "cache/train_anchors.parquet": "train_anchors.parquet",
    "cache/test_features.parquet": "test_features.parquet",
}
REPO_IMMUTABLE_FILE_MAP = {
    "frozen/equal_submission.csv": "submissions/p3_frozen_catboost/submission.csv",
    "frozen/equal_manifest.json": "submissions/p3_frozen_catboost/manifest.json",
    "frozen/router_submission.csv": "submissions/p3_lead_long_loss_router/submission.csv",
    "frozen/router_manifest.json": "submissions/p3_lead_long_loss_router/manifest.json",
    "frozen/current_submission.csv": "submissions/p3_long_persistence_shrink/submission.csv",
    "frozen/current_manifest.json": "submissions/p3_long_persistence_shrink/manifest.json",
    "current/ready_submission.csv": "output/2026-08-20/ready/P3_submission.csv",
}


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def _git_state(root: Path) -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in status.stdout.splitlines() if line.strip()]
    return {
        "head": sha.stdout.strip() if sha.returncode == 0 else "unknown",
        "dirty": bool(lines),
        "dirty_entry_count": int(len(lines)),
    }


def _resolved_input_paths(
    *, root: Path, data_dir: Path, cache_dir: Path
) -> dict[str, Path]:
    paths = {
        logical: data_dir / relative for logical, relative in SOURCE_FILE_MAP.items()
    }
    paths.update(
        {logical: cache_dir / relative for logical, relative in CACHE_FILE_MAP.items()}
    )
    paths.update(
        {logical: root / relative for logical, relative in REPO_IMMUTABLE_FILE_MAP.items()}
    )
    return paths


def _verify_input_hashes(paths: dict[str, Path], expected: dict[str, str]) -> dict[str, str]:
    if set(paths) != set(expected):
        missing = sorted(set(paths).difference(expected))
        extra = sorted(set(expected).difference(paths))
        raise ValueError(f"input SHA contract differs; missing={missing}, extra={extra}")
    observed: dict[str, str] = {}
    for logical, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"required pinned input is missing: {logical}")
        digest = sha256_file(path)
        if digest != str(expected[logical]).lower():
            raise ValueError(f"pinned input SHA differs: {logical}")
        observed[logical] = digest
    return observed


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment_id") != "p3_corrected_repeated_forward_catboost_v1":
        raise ValueError("unexpected experiment id")
    validation = config["validation"]
    if (
        validation["gap_hours"] != 78
        or validation["context_hours"] != 48
        or validation["target_hours"] != 24
        or validation["footprint_hours"] != 72
    ):
        raise ValueError("corrected split hours differ from the frozen contract")
    if config["features"]["expected_feature_count"] != 591:
        raise ValueError("feature count contract changed")
    if config["model"]["single_weight"] != 0.5 or config["model"]["multi_weight"] != 0.5:
        raise ValueError("component weights changed")
    router = config["router"]
    if router != {
        "granularity": "lead_long",
        "name": "smooth_medium",
        "alpha": 10.0,
        "temperature_multiplier": 2.0,
        "strength": 0.5,
        "active_leads": [12, 18, 24],
        "fold_one": "exact_equal_component_no_op",
        "later_folds": "fixed_config_refit_on_completed_corrected_oof_only",
        "hyperparameter_search": False,
    }:
        raise ValueError("fixed router contract changed")
    if config["shrink"] != {"active_leads": [12, 18, 24], "persistence_weight": 0.2}:
        raise ValueError("fixed shrink contract changed")
    if not all(config["prohibitions"].values()):
        raise ValueError("all prohibited actions must remain true")


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
        [
            lookup.loc[anchor_ids, f"target_{lead}"].to_numpy(dtype=float) - current
            for lead in LEADS
        ]
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
    x_train, y_train, train_meta = expand_leads(
        features, anchors, fold.train_ids, feature_columns
    )
    x_valid, _, valid_meta = expand_leads(
        features, anchors, fold.validation_ids, feature_columns
    )
    single = _single_model(config, seed)
    single.fit(
        _cat_frame(x_train),
        y_train,
        sample_weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    single_prediction = np.clip(
        valid_meta["current_hs"].to_numpy(dtype=float)
        + single.predict(_cat_frame(x_valid)),
        0.0,
        30.0,
    )

    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    multi_x_train = feature_lookup.loc[
        fold.train_ids, ["station", *feature_columns]
    ].reset_index(drop=True)
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
    oof["equal_prediction"] = 0.5 * (
        oof["single_prediction"] + oof["multi_prediction"]
    )
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

    case_x, case_meta, case_components, _ = build_case_router_data(
        oof, train_features, anchors
    )
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
        np.isfinite(numeric).all()
        and np.all(oof["final_prediction"].between(0.0, 30.0).to_numpy())
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
            "delta_final_minus_persistence_m": rmse(
                group["target_hs"], group["final_prediction"]
            )
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
            value >= 78.0
            for value in split_audit["station_global_minimum_gap_hours"].values()
        ),
        "validation_storm_episode_distinct": split_audit["repeated_station_episode_count"]
        == 0,
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
        "complete_unique_oof_keys": len(oof)
        == split_audit["validation_case_count"] * len(LEADS),
        "fixed_router_no_current_fold_target": all(
            not row["current_fold_target_used_for_router"] for row in router_receipts
        ),
        "short_lead_router_is_exact_equal_ensemble": short_router_error <= 1e-12,
        "short_lead_shrink_is_exact_no_op": short_shrink_error == 0.0,
        "finite_and_range_valid": finite_and_range,
        "hyperparameter_search_run_zero": config["router"]["hyperparameter_search"]
        is False,
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


def _fit_full_and_infer(
    *,
    root: Path,
    data_dir: Path,
    cache_dir: Path,
    stage: Path,
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    feature_columns: list[str],
    router_material: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    access = {
        "test_feature_cache_value_reads": 0,
        "test_index_value_reads": 0,
        "test_context_value_reads": 0,
        "test_target_or_hidden_label_reads": 0,
        "absolute_test_timestamp_recovery_attempts": 0,
        "current_or_frozen_submission_value_reads": 0,
        "current_or_frozen_submission_writes": 0,
        "upload_attempts": 0,
    }
    seed = int(config["model"]["full_train_seed"])
    train_ids = anchors["anchor_id"].to_numpy(dtype=np.int64)
    x_train, y_train, train_meta = expand_leads(
        features, anchors, train_ids, feature_columns
    )
    single = _single_model(config, seed)
    single.fit(
        _cat_frame(x_train),
        y_train,
        sample_weight=threshold_case_weights(train_meta["current_hs"].to_numpy()),
        cat_features=[0, 1],
        verbose=False,
    )
    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    multi_train = feature_lookup.loc[train_ids, ["station", *feature_columns]].reset_index(
        drop=True
    )
    multi_train["station"] = multi_train["station"].astype(str)
    multi = _multi_model(config, seed)
    multi.fit(
        multi_train,
        _multi_target(anchors, train_ids),
        sample_weight=threshold_case_weights(
            anchor_lookup.loc[train_ids, "current_hs"].to_numpy(dtype=float)
        ),
        cat_features=[0],
        verbose=False,
    )
    router = ComponentLossRouter(_router_config(config)).fit(
        router_material["row_features"], router_material["row_losses"]
    )

    # Gate has passed before the first parsed read of either anonymous-test artifact.
    test_features = pd.read_parquet(cache_dir / "test_features.parquet")
    access["test_feature_cache_value_reads"] += 1
    test_index = pd.read_csv(data_dir / "test_index.csv")
    access["test_index_value_reads"] += 1
    if list(test_index.columns) != KEYS or len(test_index) != 1_200:
        raise ValueError("test_index contract differs")
    if test_index.duplicated(KEYS).any() or test_index[KEYS].isna().any().any():
        raise ValueError("test_index keys are missing or duplicated")
    lead_tuple = test_index.groupby("case_id", sort=False, observed=True)["lead_h"].agg(tuple)
    if len(lead_tuple) != 200 or not lead_tuple.map(lambda value: value == LEADS).all():
        raise ValueError("test_index must have 200 cases with the six ordered leads")
    if len(test_features) != 200 or test_features.duplicated(["case_id", "station"]).any():
        raise ValueError("same-case test feature cache must contain 200 unique cases")
    forbidden_test_columns = {
        column
        for column in test_features.columns
        if column.lower() in {"time", "timestamp", "date", "target_hs", "truth", "label"}
    }
    if forbidden_test_columns:
        raise ValueError(f"forbidden anonymous-test feature columns: {sorted(forbidden_test_columns)}")
    case_order = test_index[["case_id", "station"]].drop_duplicates().reset_index(drop=True)
    test_cases = case_order.merge(
        test_features,
        on=["case_id", "station"],
        how="left",
        validate="one_to_one",
    )
    if len(test_cases) != 200 or test_cases[feature_columns].isna().all(axis=1).any():
        raise ValueError("same-case test feature alignment failed")

    source = test_cases.set_index(["case_id", "station"])
    repeated_keys = pd.MultiIndex.from_frame(test_index[["case_id", "station"]])
    single_x = source.loc[repeated_keys, feature_columns].reset_index(drop=True)
    single_x.insert(0, "lead_h", test_index["lead_h"].to_numpy())
    single_x.insert(0, "station", test_index["station"].astype(str).to_numpy())
    current_rows = source.loc[repeated_keys, "hs_current"].to_numpy(dtype=float)
    single_x.insert(2, "current_hs_for_residual", current_rows)
    single_prediction = np.clip(
        current_rows + single.predict(_cat_frame(single_x)), 0.0, 30.0
    )

    multi_test = test_cases[["station", *feature_columns]].copy()
    multi_test["station"] = multi_test["station"].astype(str)
    multi_delta = np.asarray(multi.predict(multi_test), dtype=float)
    current_case = test_cases["hs_current"].to_numpy(dtype=float)
    multi_prediction = np.clip(current_case[:, None] + multi_delta, 0.0, 30.0)
    single_matrix = single_prediction.reshape(200, len(LEADS))
    persistence_matrix = np.repeat(current_case[:, None], len(LEADS), axis=1)
    components = np.stack(
        [single_matrix, multi_prediction, persistence_matrix], axis=2
    )

    test_case_x = build_inference_router_features(
        test_cases.loc[:, OBSERVED_FEATURES],
        test_cases["station"].to_numpy(str),
        current_case,
        components,
    )
    test_meta = pd.DataFrame(
        {
            "fold": "anonymous_test",
            "anchor_id": np.arange(200, dtype=np.int64),
            "station": test_cases["station"].astype(str),
            "anchor_time": pd.NaT,
        }
    )
    test_row_x, test_row_meta, test_row_components = expand_case_router_features(
        test_case_x, test_meta, components
    )
    weights = router.predict_weights(test_row_x)
    inactive = ~test_row_meta["lead_h"].isin(ACTIVE_ROUTER_LEADS).to_numpy()
    weights[inactive] = np.array([0.5, 0.5, 0.0])
    routed = route_row_predictions(test_row_components, weights)
    final = apply_long_lead_persistence_shrink(
        routed,
        persistence_matrix.reshape(-1),
        test_row_meta["lead_h"].to_numpy(dtype=int),
        config=LongLeadPersistenceShrink(
            weight=float(config["shrink"]["persistence_weight"]),
            active_leads=tuple(config["shrink"]["active_leads"]),
        ),
    )
    candidate = build_submission(test_index, final)
    validate_submission(candidate, test_index)
    candidate_path = write_submission(
        candidate, test_index, stage / config["output"]["candidate_relative_path"]
    )
    reread = pd.read_csv(candidate_path)
    validate_submission(reread, test_index)
    if not reread[KEYS].equals(test_index[KEYS]):
        raise AssertionError("candidate key/order changed after write")

    full_model_dir = stage / "models/full"
    full_model_dir.mkdir(parents=True, exist_ok=True)
    single_path = full_model_dir / "single.cbm"
    multi_path = full_model_dir / "multi.cbm"
    router_path = full_model_dir / "router.joblib"
    single.save_model(single_path)
    multi.save_model(multi_path)
    joblib.dump(router, router_path)
    receipt = {
        "status": "passed_local_candidate_contract_not_uploaded",
        "rows": int(len(candidate)),
        "cases": int(candidate["case_id"].nunique()),
        "key_order_exact": True,
        "finite": bool(np.isfinite(final).all()),
        "range_0_to_30_m": bool(np.all((final >= 0.0) & (final <= 30.0))),
        "prediction_aggregate": {
            "minimum_m": float(np.min(final)),
            "median_m": float(np.median(final)),
            "maximum_m": float(np.max(final)),
            "mean_m": float(np.mean(final)),
        },
        "weight_aggregate": {
            name: {
                "mean": float(weights[:, index].mean()),
                "p10": float(np.quantile(weights[:, index], 0.10)),
                "p90": float(np.quantile(weights[:, index], 0.90)),
            }
            for index, name in enumerate(("single", "multi", "persistence"))
        },
        "same_case_only": True,
        "test_context_values_read_directly": False,
        "test_feature_cache_rows": int(len(test_features)),
        "absolute_test_time_features": 0,
        "test_target_or_hidden_label_reads": 0,
        "sha256": {
            "candidate/submission.csv": sha256_file(candidate_path),
            "models/full/single.cbm": sha256_file(single_path),
            "models/full/multi.cbm": sha256_file(multi_path),
            "models/full/router.joblib": sha256_file(router_path),
        },
        "uploaded": False,
    }
    _atomic_json(stage / "candidate/validation.json", receipt)
    return receipt, access


def _artifact_hashes(stage: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(value for value in stage.rglob("*") if value.is_file()):
        relative = path.relative_to(stage).as_posix()
        if relative in {"manifest.json", "manifest.sha256"}:
            continue
        result[relative] = {"bytes": int(path.stat().st_size), "sha256": sha256_file(path)}
    return result


def _preflight(
    *, root: Path, data_dir: Path, cache_dir: Path, config: dict[str, Any], output: Path
) -> dict[str, Any]:
    _validate_config(config)
    if output.exists():
        raise FileExistsError("append-only output generation already exists")
    paths = _resolved_input_paths(root=root, data_dir=data_dir, cache_dir=cache_dir)
    snapshot = _verify_input_hashes(paths, config["expected_sha256"])
    if get_gpu_device_count() < 1:
        raise RuntimeError("fixed multi-output CatBoost requires GPU device 0")
    features = pd.read_parquet(cache_dir / "train_features.parquet")
    anchors = pd.read_parquet(cache_dir / "train_anchors.parquet")
    if len(features) != len(anchors) or len(anchors) != 24_360:
        raise ValueError("training cache row contract differs")
    if not features[["anchor_id", "station"]].equals(anchors[["anchor_id", "station"]]):
        raise ValueError("training feature/anchor key alignment differs")
    feature_columns = compact_feature_columns(
        [column for column in features if column not in {"anchor_id", "station"}]
    )
    if len(feature_columns) != int(config["features"]["expected_feature_count"]):
        raise ValueError("compact feature surface changed")
    wave = pd.read_csv(data_dir / "train_wave.csv")
    wave["time"] = pd.to_datetime(wave["time"], utc=True, errors="raise")
    anchors = assign_storm_episodes_from_wave(anchors, wave)
    folds, selected, split_audit = build_corrected_repeated_forward_folds(
        anchors,
        windows=config["validation"]["windows"],
        gap_hours=int(config["validation"]["gap_hours"]),
        footprint_hours=int(config["validation"]["footprint_hours"]),
    )
    return {
        "input_paths": paths,
        "input_snapshot": snapshot,
        "features": features,
        "anchors": anchors,
        "feature_columns": feature_columns,
        "folds": folds,
        "selected": selected,
        "split_audit": split_audit,
        "aggregate": {
            "status": "CHECK_ONLY_PASS",
            "feature_count": int(len(feature_columns)),
            "training_anchor_count": int(len(anchors)),
            "validation_case_count": int(len(selected)),
            "validation_row_count": int(len(selected) * len(LEADS)),
            "fold_validation_cases": {
                fold.name: int(len(fold.validation_ids)) for fold in folds
            },
            "fold_train_anchors": {fold.name: int(len(fold.train_ids)) for fold in folds},
            "station_global_minimum_gap_hours": split_audit[
                "station_global_minimum_gap_hours"
            ],
            "repeated_station_episode_count": split_audit[
                "repeated_station_episode_count"
            ],
            "footprint_overlap_pairs": split_audit[
                "context48_plus_target24_footprint_overlap_pairs"
            ],
            "gpu_device_count": int(get_gpu_device_count()),
            "output_absent": True,
        },
    }


def run_experiment(
    *,
    root: Path,
    data_dir: Path,
    cache_dir: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    preflight = _preflight(
        root=root, data_dir=data_dir, cache_dir=cache_dir, config=config, output=output
    )
    print(json.dumps(preflight["aggregate"], ensure_ascii=False), flush=True)
    tmp_root = root / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="p3_corrected_rf_v1_", dir=tmp_root))
    features: pd.DataFrame = preflight["features"]
    anchors: pd.DataFrame = preflight["anchors"]
    folds: tuple[CorrectedFold, ...] = preflight["folds"]
    selected: pd.DataFrame = preflight["selected"]
    feature_columns: list[str] = preflight["feature_columns"]

    fold_oof: list[pd.DataFrame] = []
    training_receipts: list[dict[str, Any]] = []
    for number, fold in enumerate(folds):
        print(
            json.dumps(
                {
                    "phase": "fit_corrected_fold",
                    "fold": fold.name,
                    "number": number + 1,
                    "total": len(folds),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        current, receipt = _fit_fold_components(
            fold=fold,
            fold_number=number,
            features=features,
            anchors=anchors,
            feature_columns=feature_columns,
            config=config,
            model_dir=stage / "models/folds",
        )
        fold_oof.append(current)
        training_receipts.append(receipt)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)

    component_oof = pd.concat(fold_oof, ignore_index=True)
    oof, evaluation, router_material = _evaluate_fixed_structure(
        component_oof=component_oof,
        train_features=features,
        anchors=anchors,
        fold_order=tuple(fold.name for fold in folds),
        config=config,
        split_audit=preflight["split_audit"],
        expected_validation_ids=selected["anchor_id"].to_numpy(dtype=np.int64),
    )
    _atomic_parquet(stage / "oof.parquet", oof)
    split_keys = selected[["fold", "anchor_id", "station", "episode_id"]].copy()
    _atomic_parquet(stage / "validation_keys.parquet", split_keys)
    _atomic_json(stage / "feature_columns.json", {"columns": feature_columns})

    access = {
        "test_feature_cache_value_reads": 0,
        "test_index_value_reads": 0,
        "test_context_value_reads": 0,
        "test_target_or_hidden_label_reads": 0,
        "absolute_test_timestamp_recovery_attempts": 0,
        "current_or_frozen_submission_value_reads": 0,
        "current_or_frozen_submission_writes": 0,
        "upload_attempts": 0,
    }
    candidate_receipt: dict[str, Any] | None = None
    if evaluation["gate"]["passed"]:
        print(
            json.dumps(
                {
                    "phase": "gate_passed_full_refit_and_same_case_test_inference",
                    "candidate_rmse_m": evaluation["gate"]["candidate_rmse_m"],
                    "persistence_rmse_m": evaluation["gate"]["persistence_rmse_m"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        candidate_receipt, access = _fit_full_and_infer(
            root=root,
            data_dir=data_dir,
            cache_dir=cache_dir,
            stage=stage,
            features=features,
            anchors=anchors,
            feature_columns=feature_columns,
            router_material=router_material,
            config=config,
        )

    input_after = _verify_input_hashes(
        preflight["input_paths"], config["expected_sha256"]
    )
    if input_after != preflight["input_snapshot"]:
        raise RuntimeError("source/cache/current/frozen SHA changed during the run")
    metrics = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": (
            "CORRECTED_RESEARCH_EVIDENCE_GATE_PASS_CANDIDATE_CREATED_NOT_UPLOADED"
            if evaluation["gate"]["passed"]
            else "CORRECTED_RESEARCH_EVIDENCE_GATE_FAIL_NO_TEST_INFERENCE"
        ),
        "interpretation": (
            "Corrected repeated-forward research evidence; not an official hidden score, "
            "not fresh confirmation, and not upload authorization."
        ),
        "official_scoring_note": (
            "T=0.624165 is the organizer's policy/scoring constant, not a hidden model score."
        ),
        "split_audit": preflight["split_audit"],
        "training_receipts": training_receipts,
        **evaluation,
        "candidate_validation": candidate_receipt,
        "access_counters": access,
        "invariants": {
            "hyperparameter_search_run": False,
            "external_observations_used": 0,
            "test_absolute_timestamp_recovered": False,
            "test_target_or_hidden_labels_used": 0,
            "current_or_frozen_submission_mutated": False,
            "submission_uploaded": False,
            "source_cache_current_frozen_sha_unchanged": True,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    _atomic_json(stage / "metrics.json", metrics)

    implementation_paths = {
        "config": config_path,
        "module": root / "src/p3_wave/corrected_repeated_forward.py",
        "runner": Path(__file__).resolve(),
        "tests": root / "tests/test_p3_corrected_repeated_forward_catboost_v1.py",
        "feature_builder": root / "src/p3_wave/features.py",
    }
    manifest = {
        "created_at": _now(),
        "experiment_id": config["experiment_id"],
        "status": metrics["status"],
        "append_only_generation": True,
        "artifact_root": config["output"]["artifact_dir"],
        "config_sha256": sha256_file(config_path),
        "implementation_sha256": {
            name: sha256_file(path) for name, path in implementation_paths.items()
        },
        "git": _git_state(root),
        "input_sha256_before": preflight["input_snapshot"],
        "input_sha256_after": input_after,
        "source_cache_current_frozen_unchanged": True,
        "output_files": _artifact_hashes(stage),
        "gate_passed": bool(evaluation["gate"]["passed"]),
        "candidate_created": candidate_receipt is not None,
        "candidate_uploaded": False,
        "access_counters": access,
        "no_raw_values_in_manifest": True,
    }
    _atomic_json(stage / "manifest.json", manifest)
    manifest_sha = sha256_file(stage / "manifest.json")
    (stage / "manifest.sha256").write_text(
        f"{manifest_sha}  manifest.json\n", encoding="ascii"
    )
    if output.exists():
        raise FileExistsError("append-only output appeared before finalization")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage.replace(output)
    result = {
        "status": metrics["status"],
        "artifact_dir": output.relative_to(root).as_posix(),
        "metrics_sha256": sha256_file(output / "metrics.json"),
        "oof_sha256": sha256_file(output / "oof.parquet"),
        "manifest_sha256": manifest_sha,
        "candidate_sha256": (
            sha256_file(output / config["output"]["candidate_relative_path"])
            if candidate_receipt is not None
            else None
        ),
        "gate": evaluation["gate"],
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", default="artifacts/p3/features_all20_v1")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("check-only", "run"), default="check-only")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = (root / args.config).resolve()
    cache_dir = (root / args.cache_dir).resolve()
    output = (root / args.output_dir).resolve()
    data_dir = Path(args.data_dir).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.mode == "check-only":
        result = _preflight(
            root=root,
            data_dir=data_dir,
            cache_dir=cache_dir,
            config=config,
            output=output,
        )["aggregate"]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    run_experiment(
        root=root,
        data_dir=data_dir,
        cache_dir=cache_dir,
        config_path=config_path,
        output=output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
