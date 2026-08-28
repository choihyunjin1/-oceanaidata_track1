"""Check or execute the sealed asynchronous P1 latent-state subset scan once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from p1_qc.p1_async_latent_state_gp_subset_scan_anchor_union_20260828_v1 import (  # noqa: E402
    binary_metrics,
    block_bootstrap_delta,
    block_proposals,
    conformal_threshold,
    loo_matern_smoother,
    robust_center_scale,
    sha256_file,
)

EXPERIMENT_ID = "p1_async_latent_state_gp_subset_scan_anchor_union_20260828_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
MODULE = ROOT / "src/p1_qc" / f"{EXPERIMENT_ID}.py"
RUNNER = Path(__file__).resolve()
TEST = ROOT / "tests" / f"test_{EXPERIMENT_ID}.py"
KEYS = ["station", "year", "layer", "time"]
CADENCE = pd.Timedelta(minutes=10)


class ContractError(RuntimeError):
    """Raised when an immutable preregistration condition is violated."""


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".partial"
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".partial") as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _prediction_sha(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int8).tobytes()).hexdigest()


def _parse_time(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["time"] = pd.to_datetime(output["time"], utc=True, format="mixed").dt.tz_convert(
        "Asia/Seoul"
    )
    return output


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    if int(config["split"]["purge_days"]) != 15:
        raise ContractError("purge changed")
    if int(config["split"]["calibration_days"]) != 84:
        raise ContractError("calibration changed")
    if float(config["conformal"]["alpha"]) != 0.01:
        raise ContractError("conformal alpha changed")
    if config["state_model"]["lengthscale_hours"] != [6.0, 48.0]:
        raise ContractError("Matérn lengthscales changed")
    if config["scope"]["target_layer_measurement_updates_allowed"] is not False:
        raise ContractError("target-layer measurement updates are forbidden")
    if config["scan"]["result_based_threshold_or_duration_search"] is not False:
        raise ContractError("result-based scan search is forbidden")
    policy = config["execution_policy"]
    if (
        policy["local_execution_authorized"] is not True
        or int(policy["maximum_executions"]) != 1
        or policy["result_based_retry"] is not False
        or policy["official_test_sample_submission_read_allowed"] is not False
        or policy["submission_csv_generation_allowed"] is not False
        or policy["official_upload_authorized"] is not False
    ):
        raise ContractError("execution policy drifted")
    for record in config["immutable_inputs"].values():
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
            raise ContractError(f"immutable input changed: {path}")
    return config


def _load_e150(config: dict[str, Any], outer_keys: pd.DataFrame) -> np.ndarray:
    output = np.zeros(len(outer_keys), dtype=np.int8)
    records = {
        "2025_q2": config["immutable_inputs"]["e150_q2"]["path"],
        "2025_q3": config["immutable_inputs"]["e150_q3"]["path"],
        "2025_q4": config["immutable_inputs"]["e150_q4"]["path"],
    }
    for fold, relative in records.items():
        positions = np.flatnonzero(outer_keys["fold"].astype(str).eq(fold).to_numpy())
        with np.load(ROOT / relative, allow_pickle=False) as archive:
            if fold == "2025_q2":
                model = np.flatnonzero((archive["widths"] == 512) & (archive["epochs"] == 150))
                threshold = np.flatnonzero(np.isclose(archive["thresholds"], 0.8))
                if len(model) != 1 or len(threshold) != 1:
                    raise ContractError("Q2 e150 recipe is not unique")
                prediction = archive["candidate"][int(model[0]), int(threshold[0])]
            else:
                epoch = np.flatnonzero(archive["epochs"] == 150)
                if len(epoch) != 1:
                    raise ContractError(f"{fold} e150 recipe is not unique")
                prediction = archive["candidate"][int(epoch[0])]
        if len(prediction) != len(positions):
            raise ContractError(f"{fold} e150 length mismatch")
        output[positions] = np.asarray(prediction, dtype=np.int8)
    return output


def _prefix_labels(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    dataset = ds.dataset(path, format="parquet")
    table = dataset.to_table(
        columns=[*KEYS, "label"],
        filter=ds.field("time") < cutoff.isoformat(),
    )
    return _parse_time(table.to_pandas())


def _load_features(config: dict[str, Any]) -> pd.DataFrame:
    key_path = ROOT / config["immutable_inputs"]["feature_keys"]["path"]
    matrix_path = ROOT / config["immutable_inputs"]["feature_matrix"]["path"]
    keys = _parse_time(pd.read_parquet(key_path, columns=["ordinal", *KEYS]))
    values = pd.read_parquet(matrix_path, columns=[config["scope"]["feature"]])
    if len(keys) != len(values):
        raise ContractError("feature/key row count mismatch")
    if not np.array_equal(keys["ordinal"].to_numpy(dtype=np.int64), np.arange(len(keys))):
        raise ContractError("feature ordinals are not canonical")
    keys["value"] = values.iloc[:, 0].to_numpy(dtype=np.float64)
    return keys


def _cell_stats(
    feature: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    fit_stop: pd.Timestamp,
    stations: list[str],
    minimum_scale: float,
) -> dict[tuple[str, int], tuple[float, float]]:
    prefix = feature.loc[
        feature["station"].astype(str).isin(stations) & (feature["time"] < fit_stop),
        [*KEYS, "value"],
    ]
    merged = prefix.merge(labels, on=KEYS, how="left", validate="one_to_one")
    output: dict[tuple[str, int], tuple[float, float]] = {}
    for (station, layer), group in merged.loc[merged["label"].eq(0)].groupby(
        ["station", "layer"], sort=True
    ):
        output[(str(station), int(layer))] = robust_center_scale(
            group["value"], minimum_scale
        )
    return output


def _cell_state(
    period: pd.DataFrame,
    *,
    station: str,
    target_layer: int,
    stats: dict[tuple[str, int], tuple[float, float]],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    station_rows = period.loc[period["station"].astype(str).eq(station)].copy()
    target = station_rows.loc[station_rows["layer"].eq(target_layer)].sort_values("time").copy()
    target = target.drop_duplicates("time", keep=False).reset_index(drop=True)
    if target.empty or (station, target_layer) not in stats:
        return pd.DataFrame(columns=["ordinal", *KEYS, "residual", "peer_count"]), {
            "rows": 0,
            "peer_coverage": 0.0,
            "posterior_finite_coverage": 0.0,
            "matched_rows": 0,
        }
    target_center, target_scale = stats[(station, target_layer)]
    target_z = (target["value"].to_numpy(dtype=np.float64) - target_center) / target_scale
    peer_columns: list[np.ndarray] = []
    tolerance = pd.Timedelta(minutes=int(config["scope"]["peer_tolerance_minutes"]))
    peer_layers = sorted(
        int(value)
        for value in station_rows["layer"].dropna().unique()
        if int(value) != target_layer and (station, int(value)) in stats
    )
    left = target[["time"]].sort_values("time")
    for peer_layer in peer_layers:
        center, scale = stats[(station, peer_layer)]
        peer = station_rows.loc[station_rows["layer"].eq(peer_layer), ["time", "value"]].copy()
        peer["peer_z"] = (peer["value"].to_numpy(dtype=np.float64) - center) / scale
        peer = peer.sort_values("time").drop_duplicates("time", keep=False)
        matched = pd.merge_asof(
            left,
            peer[["time", "peer_z"]],
            on="time",
            direction="nearest",
            tolerance=tolerance,
        )
        peer_columns.append(matched["peer_z"].to_numpy(dtype=np.float64))
    if peer_columns:
        matrix = np.column_stack(peer_columns)
        peer_count = np.isfinite(matrix).sum(axis=1).astype(np.int16)
        with np.errstate(all="ignore"):
            peer_observation = np.nanmedian(matrix, axis=1)
        peer_observation[peer_count == 0] = np.nan
    else:
        peer_count = np.zeros(len(target), dtype=np.int16)
        peer_observation = np.full(len(target), np.nan)
    state = config["state_model"]
    latent_mean, latent_variance, observed = loo_matern_smoother(
        pd.DatetimeIndex(target["time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64),
        peer_observation,
        peer_count,
        minimum_peers=int(config["scope"]["minimum_peer_layers"]),
        lengthscale_hours=state["lengthscale_hours"],
        factor_variance=state["factor_variance"],
        observation_variance=float(state["peer_observation_variance"]),
    )
    denominator = np.sqrt(
        latent_variance + float(state["target_idiosyncratic_variance"])
    )
    residual = (target_z - latent_mean) / denominator
    residual[~np.isfinite(target_z)] = np.nan
    output = target[["ordinal", *KEYS]].copy()
    output["residual"] = residual
    output["peer_count"] = peer_count
    posterior_finite = np.isfinite(latent_mean) & np.isfinite(latent_variance)
    return output, {
        "rows": int(len(output)),
        "peer_layers": int(len(peer_layers)),
        "peer_coverage": float(np.mean(observed)) if len(observed) else 0.0,
        "posterior_finite_coverage": float(np.mean(posterior_finite))
        if len(posterior_finite)
        else 0.0,
        "matched_rows": int(observed.sum()),
    }


def _attach_labels(state: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return state.merge(labels, on=KEYS, how="left", validate="one_to_one")


def _event_recoveries(evaluation: pd.DataFrame) -> list[dict[str, Any]]:
    relevant = evaluation.loc[
        evaluation["label"].eq(1)
        & evaluation["anomaly_type"].fillna("").str.contains("noise|offset|drift", regex=True)
    ].sort_values(["fold", "station", "layer", "time"])
    recovered: list[dict[str, Any]] = []
    for (fold, station, layer), group in relevant.groupby(["fold", "station", "layer"]):
        boundary = np.ones(len(group), dtype=bool)
        if len(group) > 1:
            boundary[1:] = pd.DatetimeIndex(group["time"]).to_series().diff().iloc[1:].ne(CADENCE).to_numpy()
        event_ids = np.cumsum(boundary)
        for event_id, event in group.groupby(event_ids):
            added = event["scan_prediction"].eq(1) & event["anchor_prediction"].eq(0)
            if not added.any():
                continue
            text = "|".join(event["anomaly_type"].dropna().astype(str).unique()).lower()
            anomaly_type = next((name for name in ("noise", "offset", "drift") if name in text), "mixed")
            recovered.append(
                {
                    "fold": str(fold),
                    "station": str(station),
                    "layer": int(layer),
                    "event_id": int(event_id),
                    "type": anomaly_type,
                    "added_true_rows": int(added.sum()),
                }
            )
    return recovered


def _write_terminal_no_go(
    output_directory: Path,
    *,
    diagnostics: dict[str, Any],
    checks: dict[str, bool],
    anchor: np.ndarray,
    scan: np.ndarray,
    started: float,
) -> dict[str, Any]:
    sealed = output_directory / "sealed_predictions.npz"
    union = np.maximum(anchor, scan).astype(np.int8)
    _atomic_npz(sealed, anchor_prediction=anchor, scan_prediction=scan, union_prediction=union)
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_rows": int(len(anchor)),
        "anchor_sha256": _prediction_sha(anchor),
        "scan_sha256": _prediction_sha(scan),
        "union_sha256": _prediction_sha(union),
        "sealed_file_sha256": sha256_file(sealed),
        "truth_rows_read_before_commitment": 0,
        "official_rows_read": 0,
    }
    _atomic_json(output_directory / "prediction_commitment.json", commitment)
    result = {
        "schema_version": "p1.async_latent_state_gp_subset_scan.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "NO_GO_SUPPORT_EXACT_E150_NO_OP",
        "scientific_execution_count": 1,
        "fixed_hyperparameter_model_fit_count": 0,
        "truth_rows_read_after_commitment": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "anchor_deletions": 0,
        "gate_checks": checks,
        "diagnostics": diagnostics,
        "commitment": commitment,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_directory / "result.json", result)
    return result


def execute(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    output_directory = ROOT / config["artifact_directory"]
    if output_directory.exists():
        raise FileExistsError(f"one-shot artifact already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    feature = _load_features(config)
    truth_path = ROOT / config["immutable_inputs"]["outer_truth_after_prediction_seal"]["path"]
    outer_keys = _parse_time(pd.read_parquet(truth_path, columns=[*KEYS, "fold"]))
    if outer_keys.duplicated([*KEYS, "fold"]).any():
        raise ContractError("outer keys are not unique")
    anchor = _load_e150(config, outer_keys)
    scan = np.zeros(len(outer_keys), dtype=np.int8)
    outer_keys = outer_keys.reset_index(drop=True)
    outer_keys["outer_position"] = np.arange(len(outer_keys), dtype=np.int64)
    feature_lookup = feature[["ordinal", *KEYS]]
    outer_binding = outer_keys.merge(feature_lookup, on=KEYS, how="left", validate="one_to_one")
    if outer_binding["ordinal"].isna().any():
        raise ContractError("outer keys do not bind to feature ordinals")
    ordinal_to_outer = pd.Series(
        outer_binding["outer_position"].to_numpy(dtype=np.int64),
        index=outer_binding["ordinal"].to_numpy(dtype=np.int64),
    )

    diagnostics: dict[str, Any] = {}
    all_calibration_scores: dict[str, list[float]] = {}
    validation_proposals: dict[str, list[tuple[float, np.ndarray]]] = {}
    support_checks: dict[str, bool] = {}
    label_path = ROOT / config["immutable_inputs"]["training_labels"]["path"]
    minimum_scale = float(config["state_model"]["minimum_mad"])
    durations = config["scan"]["duration_rows"]
    for fold in config["split"]["folds"]:
        fold_keys = outer_keys.loc[outer_keys["fold"].astype(str).eq(fold)]
        validation_start = fold_keys["time"].min()
        validation_stop = fold_keys["time"].max()
        cutoff = validation_start - pd.Timedelta(days=int(config["split"]["purge_days"]))
        calibration_start = cutoff - pd.Timedelta(days=int(config["split"]["calibration_days"]))
        labels = _prefix_labels(label_path, cutoff)
        stats = _cell_stats(
            feature,
            labels,
            fit_stop=calibration_start,
            stations=config["scope"]["stations"],
            minimum_scale=minimum_scale,
        )
        calibration_period = feature.loc[
            (feature["time"] >= calibration_start) & (feature["time"] < cutoff)
        ]
        validation_period = feature.loc[
            (feature["time"] >= validation_start) & (feature["time"] <= validation_stop)
        ]
        fold_scores: list[float] = []
        fold_candidates: list[tuple[float, np.ndarray]] = []
        fold_diagnostics: dict[str, Any] = {"cells": {}}
        layers_by_station = (
            fold_keys.loc[fold_keys["station"].astype(str).isin(config["scope"]["stations"])]
            .groupby("station")["layer"]
            .unique()
        )
        for station, layer_values in layers_by_station.items():
            for layer_value in sorted(layer_values):
                layer = int(layer_value)
                calibration_state, calibration_diag = _cell_state(
                    calibration_period,
                    station=str(station),
                    target_layer=layer,
                    stats=stats,
                    config=config,
                )
                validation_state, validation_diag = _cell_state(
                    validation_period,
                    station=str(station),
                    target_layer=layer,
                    stats=stats,
                    config=config,
                )
                calibration_state = _attach_labels(calibration_state, labels).sort_values(
                    "time"
                ).reset_index(drop=True)
                validation_state = validation_state.sort_values("time").reset_index(drop=True)
                calibration_blocks = block_proposals(
                    calibration_state,
                    durations,
                    block_days=int(config["split"]["conformal_block_days"]),
                )
                normal_blocks = 0
                for proposal in calibration_blocks:
                    block_labels = calibration_state.iloc[proposal.block_positions]["label"]
                    if len(block_labels) and block_labels.notna().all() and block_labels.eq(0).all():
                        fold_scores.append(proposal.score)
                        normal_blocks += 1
                validation_blocks = block_proposals(
                    validation_state,
                    durations,
                    block_days=int(config["split"]["conformal_block_days"]),
                )
                for proposal in validation_blocks:
                    ordinals = validation_state.iloc[proposal.row_positions]["ordinal"].to_numpy(
                        dtype=np.int64
                    )
                    fold_candidates.append((proposal.score, ordinals))
                cell_key = f"{station}/L{layer}"
                fold_diagnostics["cells"][cell_key] = {
                    "calibration": calibration_diag,
                    "validation": validation_diag,
                    "normal_calibration_blocks": normal_blocks,
                    "validation_blocks": int(len(validation_blocks)),
                }
                minimum_peer = float(config["scope"]["minimum_peer_coverage"])
                minimum_rows = int(config["scope"]["minimum_matched_rows_per_cell"])
                minimum_posterior = float(config["gates"]["minimum_posterior_finite_coverage"])
                support_checks[f"{fold}:{cell_key}:peer_coverage"] = (
                    calibration_diag["peer_coverage"] >= minimum_peer
                    and validation_diag["peer_coverage"] >= minimum_peer
                )
                support_checks[f"{fold}:{cell_key}:matched_rows"] = (
                    calibration_diag["matched_rows"] >= minimum_rows
                    and validation_diag["matched_rows"] >= minimum_rows
                )
                support_checks[f"{fold}:{cell_key}:posterior_coverage"] = (
                    calibration_diag["posterior_finite_coverage"] >= minimum_posterior
                    and validation_diag["posterior_finite_coverage"] >= minimum_posterior
                )
        all_calibration_scores[fold] = fold_scores
        validation_proposals[fold] = fold_candidates
        fold_diagnostics["normal_calibration_blocks"] = int(len(fold_scores))
        diagnostics[fold] = fold_diagnostics
        support_checks[f"{fold}:normal_calibration_blocks"] = len(fold_scores) >= int(
            config["conformal"]["minimum_normal_calibration_blocks"]
        )

    thresholds: dict[str, float] = {}
    if all(support_checks.values()):
        for fold in config["split"]["folds"]:
            threshold = conformal_threshold(
                all_calibration_scores[fold], float(config["conformal"]["alpha"])
            )
            thresholds[fold] = threshold
            for score, ordinals in validation_proposals[fold]:
                if score <= threshold:
                    continue
                positions = ordinal_to_outer.reindex(ordinals)
                if positions.isna().any():
                    raise ContractError("candidate ordinals do not bind to the outer fold")
                scan[positions.to_numpy(dtype=np.int64)] = np.int8(1)
    diagnostics["conformal_thresholds"] = thresholds
    diagnostics["support_check_count"] = int(len(support_checks))
    diagnostics["support_check_failures"] = sorted(
        name for name, passed in support_checks.items() if not passed
    )
    if not all(support_checks.values()):
        result = _write_terminal_no_go(
            output_directory,
            diagnostics=diagnostics,
            checks=support_checks,
            anchor=anchor,
            scan=scan,
            started=started,
        )
        _finalize_qa(output_directory, config, result)
        return result

    union = np.maximum(anchor, scan).astype(np.int8)
    sealed = output_directory / "sealed_predictions.npz"
    _atomic_npz(sealed, anchor_prediction=anchor, scan_prediction=scan, union_prediction=union)
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_rows": int(len(anchor)),
        "anchor_sha256": _prediction_sha(anchor),
        "scan_sha256": _prediction_sha(scan),
        "union_sha256": _prediction_sha(union),
        "sealed_file_sha256": sha256_file(sealed),
        "truth_rows_read_before_commitment": 0,
        "official_rows_read": 0,
    }
    _atomic_json(output_directory / "prediction_commitment.json", commitment)

    truth = _parse_time(
        pd.read_parquet(truth_path, columns=[*KEYS, "fold", "label", "anomaly_type"])
    )
    if not outer_keys[[*KEYS, "fold"]].astype(str).equals(truth[[*KEYS, "fold"]].astype(str)):
        raise ContractError("truth/key alignment changed after commitment")
    evaluation = truth.copy()
    evaluation["anchor_prediction"] = anchor
    evaluation["scan_prediction"] = scan
    evaluation["union_prediction"] = union
    pooled_anchor = binary_metrics(evaluation["label"], anchor)
    pooled_union = binary_metrics(evaluation["label"], union)
    fold_metrics: dict[str, Any] = {}
    added_precision_checks: list[bool] = []
    for fold, group in evaluation.groupby("fold", sort=True):
        old = binary_metrics(group["label"], group["anchor_prediction"])
        new = binary_metrics(group["label"], group["union_prediction"])
        added = group["scan_prediction"].eq(1) & group["anchor_prediction"].eq(0)
        added_rows = int(added.sum())
        added_true = int((added & group["label"].eq(1)).sum())
        added_precision = added_true / added_rows if added_rows else 0.0
        fold_metrics[str(fold)] = {
            "anchor": old,
            "challenger": new,
            "delta_f1": float(new["f1"] - old["f1"]),
            "added_rows": added_rows,
            "added_true_rows": added_true,
            "added_precision": added_precision,
            "minimum_added_precision": float(old["f1"] / 2.0),
        }
        added_precision_checks.append(added_precision > float(old["f1"] / 2.0))
    added = evaluation["scan_prediction"].eq(1) & evaluation["anchor_prediction"].eq(0)
    recovered = _event_recoveries(evaluation)
    normal = evaluation.loc[evaluation["label"].eq(0)].copy()
    normal["date"] = normal["time"].dt.date
    cell_rates: dict[str, float] = {}
    for (station, layer), group in normal.groupby(["station", "layer"], sort=True):
        days = group["date"].nunique()
        false_rows = int(
            (group["scan_prediction"].eq(1) & group["anchor_prediction"].eq(0)).sum()
        )
        cell_rates[f"{station}/L{int(layer)}"] = false_rows / max(days, 1)
    maximum_cell_rate = max(cell_rates.values(), default=0.0)
    bootstrap = block_bootstrap_delta(
        evaluation,
        block_days=int(config["gates"]["bootstrap_blocks_days"]),
        replicates=int(config["gates"]["bootstrap_replicates"]),
        seed=int(config["gates"]["bootstrap_seed"]),
    )
    gates = config["gates"]
    scientific_checks = {
        "support_preflight": all(support_checks.values()),
        "anchor_deletions_zero": bool(np.all(union[anchor == 1] == 1)),
        "pooled_delta": float(pooled_union["f1"] - pooled_anchor["f1"]) > 0.0,
        "positive_folds": sum(value["delta_f1"] > 0.0 for value in fold_metrics.values())
        >= int(gates["minimum_positive_folds"]),
        "fold_floor": min(value["delta_f1"] for value in fold_metrics.values())
        >= float(gates["minimum_fold_delta_f1"]),
        "fold_added_precision": all(added_precision_checks),
        "recovered_events": len(recovered) >= int(gates["minimum_recovered_long_events"]),
        "recovery_folds": len({event["fold"] for event in recovered})
        >= int(gates["minimum_recovery_folds"]),
        "recovery_cells": len({(event["station"], event["layer"]) for event in recovered})
        >= int(gates["minimum_recovery_cells"]),
        "recovery_types": len({event["type"] for event in recovered})
        >= int(gates["minimum_recovery_types"]),
        "false_positive_cell_day_cap": maximum_cell_rate
        <= float(gates["maximum_normal_false_positive_rows_per_cell_day"]),
        "bootstrap_probability": bootstrap["probability_delta_positive"]
        >= float(gates["minimum_probability_delta_positive"]),
        "bootstrap_ci90_lower": bootstrap["ci90_lower"] >= float(gates["minimum_ci90_lower"]),
    }
    passed = all(scientific_checks.values())
    selected = union if passed else anchor
    result = {
        "schema_version": "p1.async_latent_state_gp_subset_scan.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "GO_LOCAL_ONLY" if passed else "NO_GO_QUALIFICATION_EXACT_E150_NO_OP",
        "scientific_execution_count": 1,
        "fixed_hyperparameter_model_fit_count": 0,
        "fixed_state_filter_cell_periods": int(
            sum(len(value["cells"]) * 2 for value in diagnostics.values() if isinstance(value, dict) and "cells" in value)
        ),
        "pooled_anchor": pooled_anchor,
        "pooled_challenger": pooled_union,
        "delta_f1": float(pooled_union["f1"] - pooled_anchor["f1"]),
        "added_rows": int(added.sum()),
        "added_true_rows": int((added & evaluation["label"].eq(1)).sum()),
        "added_precision": float(
            (added & evaluation["label"].eq(1)).sum() / max(int(added.sum()), 1)
        ),
        "fold_metrics": fold_metrics,
        "recovered_long_events": recovered,
        "normal_false_positive_rows_per_cell_day": cell_rates,
        "maximum_normal_false_positive_rows_per_cell_day": maximum_cell_rate,
        "bootstrap": bootstrap,
        "support_checks": support_checks,
        "gate_checks": scientific_checks,
        "diagnostics": diagnostics,
        "selected_arm": "challenger_union" if passed else "exact_e150_no_op",
        "selected_prediction_sha256": _prediction_sha(selected),
        "commitment": commitment,
        "truth_rows_read_after_commitment": int(len(truth)),
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "anchor_deletions": int(np.sum((anchor == 1) & (union == 0))),
        "result_based_retry_or_retune": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(output_directory / "result.json", result)
    _finalize_qa(output_directory, config, result)
    return result


def _finalize_qa(
    output_directory: Path, config: dict[str, Any], result: dict[str, Any]
) -> None:
    sealed = output_directory / "sealed_predictions.npz"
    commitment = json.loads((output_directory / "prediction_commitment.json").read_text(encoding="utf-8"))
    with np.load(sealed, allow_pickle=False) as archive:
        anchor = archive["anchor_prediction"].astype(np.int8)
        scan = archive["scan_prediction"].astype(np.int8)
        union = archive["union_prediction"].astype(np.int8)
    qa_checks = {
        "experiment_id": result["experiment_id"] == EXPERIMENT_ID,
        "execution_count_one": int(result["scientific_execution_count"]) == 1,
        "sealed_file_hash": sha256_file(sealed) == commitment["sealed_file_sha256"],
        "anchor_hash": _prediction_sha(anchor) == commitment["anchor_sha256"],
        "scan_hash": _prediction_sha(scan) == commitment["scan_sha256"],
        "union_hash": _prediction_sha(union) == commitment["union_sha256"],
        "anchor_union": bool(np.array_equal(union, np.maximum(anchor, scan))),
        "anchor_deletions_zero": bool(np.all(union[anchor == 1] == 1)),
        "truth_late": commitment["truth_rows_read_before_commitment"] == 0,
        "official_reads_zero": result["official_test_sample_submission_rows_read"] == 0,
        "no_submission": result["submission_generated_or_uploaded"] is False,
        "no_csv": not any(output_directory.rglob("*.csv")),
        "no_retry": config["execution_policy"]["result_based_retry"] is False,
    }
    qa = {
        "experiment_id": EXPERIMENT_ID,
        "decision": "PASS" if all(qa_checks.values()) else "FAIL",
        "checks": qa_checks,
    }
    _atomic_json(output_directory / "independent_qa.json", qa)
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "artifact_hashes": {
            path.name: sha256_file(path)
            for path in sorted(output_directory.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
        "code_hashes": {
            "config": sha256_file(CONFIG),
            "module": sha256_file(MODULE),
            "runner": sha256_file(RUNNER),
            "test": sha256_file(TEST),
        },
        "immutable_input_hashes": {
            name: record["sha256"] for name, record in config["immutable_inputs"].items()
        },
    }
    _atomic_json(output_directory / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.check_only == arguments.execute:
        raise SystemExit("choose exactly one of --check-only or --execute")
    config = load_config()
    output_directory = ROOT / config["artifact_directory"]
    if output_directory.exists():
        raise FileExistsError(f"one-shot artifact already exists: {output_directory}")
    if arguments.check_only:
        print(
            json.dumps(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "status": "PASS_CHECK_ONLY",
                    "artifact_absent": True,
                    "official_rows_read": 0,
                    "truth_rows_read": 0,
                    "submission_generated_or_uploaded": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    result = execute(config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
