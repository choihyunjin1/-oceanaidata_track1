"""Execute the sealed P1 station-pooled hierarchical residual scan exactly once."""

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

from p1_qc.p1_station_pooled_hierarchical_residual_subset_scan_anchor_union_20260828_v1 import (  # noqa: E402
    binary_metrics,
    block_bootstrap_delta,
    block_proposals,
    conformal_threshold,
    hierarchical_center_scale,
    sha256_file,
    tail_layer_share,
)

EXPERIMENT_ID = "p1_station_pooled_hierarchical_residual_subset_scan_anchor_union_20260828_v1"
CONFIG = ROOT / "configs/experiments" / f"{EXPERIMENT_ID}.json"
MODULE = ROOT / "src/p1_qc" / f"{EXPERIMENT_ID}.py"
RUNNER = Path(__file__).resolve()
TEST = ROOT / "tests" / f"test_{EXPERIMENT_ID}.py"
KEYS = ["station", "year", "layer", "time"]
CADENCE = pd.Timedelta(minutes=10)


class ContractError(RuntimeError):
    """Raised when the sealed one-shot contract is violated."""


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


def _time(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["time"] = pd.to_datetime(output["time"], utc=True, format="mixed").dt.tz_convert(
        "Asia/Seoul"
    )
    return output


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID drifted")
    if config["split"] != {
        "folds": ["2025_q2", "2025_q3", "2025_q4"],
        "purge_days": 15,
        "historical_fit_fraction": 0.5,
        "block_days": 7,
    }:
        raise ContractError("split drifted")
    if float(config["conformal"]["alpha"]) != 0.01:
        raise ContractError("alpha drifted")
    if config["scan"]["result_based_search"] is not False:
        raise ContractError("result-based search forbidden")
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
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ContractError(f"immutable input missing or resized: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ContractError(f"immutable input hash changed: {path}")
    return config


def _prefix_labels(path: Path, cutoff: pd.Timestamp) -> pd.DataFrame:
    table = ds.dataset(path, format="parquet").to_table(
        columns=[*KEYS, "label"], filter=ds.field("time") < cutoff.isoformat()
    )
    return _time(table.to_pandas())


def _load_feature(config: dict[str, Any]) -> pd.DataFrame:
    keys = _time(
        pd.read_parquet(
            ROOT / config["immutable_inputs"]["feature_keys"]["path"],
            columns=["ordinal", *KEYS],
        )
    )
    value = pd.read_parquet(
        ROOT / config["immutable_inputs"]["feature_matrix"]["path"],
        columns=[config["scope"]["feature"]],
    ).iloc[:, 0]
    if len(keys) != len(value) or not np.array_equal(
        keys["ordinal"].to_numpy(dtype=np.int64), np.arange(len(keys))
    ):
        raise ContractError("feature/row-key alignment failed")
    keys["value"] = value.to_numpy(dtype=np.float64)
    return keys


def _e150(config: dict[str, Any], outer: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(outer), dtype=np.int8)
    paths = {
        "2025_q2": config["immutable_inputs"]["e150_q2"]["path"],
        "2025_q3": config["immutable_inputs"]["e150_q3"]["path"],
        "2025_q4": config["immutable_inputs"]["e150_q4"]["path"],
    }
    for fold, relative in paths.items():
        rows = np.flatnonzero(outer["fold"].astype(str).eq(fold).to_numpy())
        with np.load(ROOT / relative, allow_pickle=False) as archive:
            if fold == "2025_q2":
                model = np.flatnonzero((archive["widths"] == 512) & (archive["epochs"] == 150))
                threshold = np.flatnonzero(np.isclose(archive["thresholds"], 0.8))
                prediction = archive["candidate"][int(model[0]), int(threshold[0])]
            else:
                epoch = np.flatnonzero(archive["epochs"] == 150)
                prediction = archive["candidate"][int(epoch[0])]
        if len(prediction) != len(rows):
            raise ContractError(f"e150 length mismatch: {fold}")
        result[rows] = np.asarray(prediction, dtype=np.int8)
    return result


def _fit_stats(
    history: pd.DataFrame,
    labels: pd.DataFrame,
    fit_stop: pd.Timestamp,
    config: dict[str, Any],
) -> tuple[dict[tuple[str, int], tuple[float, float, float]], dict[str, int]]:
    fit = history.loc[history["time"] < fit_stop].merge(
        labels, on=KEYS, how="left", validate="one_to_one"
    )
    fit = fit.loc[fit["label"].eq(0) & np.isfinite(fit["value"])]
    settings = config["partial_pooling"]
    output: dict[tuple[str, int], tuple[float, float, float]] = {}
    station_support: dict[str, int] = {}
    for station, station_rows in fit.groupby("station", sort=True):
        station_values = station_rows["value"].to_numpy(dtype=np.float64)
        station_support[str(station)] = int(len(station_values))
        if len(station_values) < int(settings["minimum_station_fit_normal_rows"]):
            continue
        for layer, cell in station_rows.groupby("layer", sort=True):
            if len(cell) < int(settings["minimum_cell_fit_normal_rows"]):
                continue
            output[(str(station), int(layer))] = hierarchical_center_scale(
                cell["value"],
                station_values,
                prior_strength=int(settings["cell_prior_strength_rows"]),
                minimum_scale=float(settings["minimum_scale"]),
            )
    return output, station_support


def _standardized_period(
    period: pd.DataFrame,
    station: str,
    layer: int,
    stats: dict[tuple[str, int], tuple[float, float, float]],
) -> pd.DataFrame:
    if (station, layer) not in stats:
        return pd.DataFrame(columns=["ordinal", *KEYS, "residual"])
    center, scale, _ = stats[(station, layer)]
    cell = period.loc[
        period["station"].astype(str).eq(station) & period["layer"].eq(layer)
    ].sort_values("time").copy()
    cell["residual"] = (cell["value"].to_numpy(dtype=np.float64) - center) / scale
    return cell[["ordinal", *KEYS, "residual"]].reset_index(drop=True)


def _seal(
    directory: Path, anchor: np.ndarray, scan: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    union = np.maximum(anchor, scan).astype(np.int8)
    sealed = directory / "sealed_predictions.npz"
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
    _atomic_json(directory / "prediction_commitment.json", commitment)
    return union, commitment


def _recoveries(frame: pd.DataFrame) -> list[dict[str, Any]]:
    relevant = frame.loc[
        frame["label"].eq(1)
        & frame["anomaly_type"].fillna("").str.contains("noise|offset|drift", regex=True)
    ].sort_values(["fold", "station", "layer", "time"])
    output: list[dict[str, Any]] = []
    for (fold, station, layer), group in relevant.groupby(["fold", "station", "layer"]):
        times = pd.DatetimeIndex(group["time"])
        boundary = np.ones(len(group), dtype=bool)
        if len(group) > 1:
            boundary[1:] = (times[1:] - times[:-1]) != CADENCE
        for event_id, event in group.groupby(np.cumsum(boundary)):
            added = event["scan_prediction"].eq(1) & event["anchor_prediction"].eq(0)
            if added.any():
                output.append(
                    {
                        "fold": str(fold),
                        "station": str(station),
                        "layer": int(layer),
                        "event_id": int(event_id),
                        "added_true_rows": int(added.sum()),
                    }
                )
    return output


def execute(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    directory = ROOT / config["artifact_directory"]
    if directory.exists():
        raise FileExistsError(f"one-shot output exists: {directory}")
    directory.mkdir(parents=True)
    feature = _load_feature(config)
    truth_path = ROOT / config["immutable_inputs"]["outer_truth_after_prediction_seal"]["path"]
    outer = _time(pd.read_parquet(truth_path, columns=[*KEYS, "fold"])).reset_index(drop=True)
    outer["outer_position"] = np.arange(len(outer), dtype=np.int64)
    anchor = _e150(config, outer)
    scan = np.zeros(len(outer), dtype=np.int8)
    binding = outer.merge(feature[["ordinal", *KEYS]], on=KEYS, how="left", validate="one_to_one")
    if binding["ordinal"].isna().any():
        raise ContractError("outer/feature binding failed")
    ordinal_to_outer = pd.Series(
        binding["outer_position"].to_numpy(dtype=np.int64),
        index=binding["ordinal"].to_numpy(dtype=np.int64),
    )
    label_path = ROOT / config["immutable_inputs"]["training_labels"]["path"]
    diagnostics: dict[str, Any] = {}
    active_folds: set[str] = set()
    for fold in config["split"]["folds"]:
        fold_outer = outer.loc[outer["fold"].astype(str).eq(fold)]
        validation_start = fold_outer["time"].min()
        validation_stop = fold_outer["time"].max()
        cutoff = validation_start - pd.Timedelta(days=int(config["split"]["purge_days"]))
        history = feature.loc[
            feature["station"].astype(str).isin(config["scope"]["stations"])
            & (feature["time"] < cutoff)
        ]
        history_start = history["time"].min()
        fit_stop = history_start + (cutoff - history_start) * float(
            config["split"]["historical_fit_fraction"]
        )
        labels = _prefix_labels(label_path, cutoff)
        stats, station_fit_support = _fit_stats(history, labels, fit_stop, config)
        calibration = history.loc[history["time"] >= fit_stop]
        validation = feature.loc[
            (feature["time"] >= validation_start) & (feature["time"] <= validation_stop)
        ]
        fold_diag: dict[str, Any] = {"fit_support": station_fit_support, "stations": {}}
        for station in config["scope"]["stations"]:
            station_scores: list[float] = []
            station_score_layers: list[int] = []
            validation_proposals: list[tuple[float, np.ndarray]] = []
            layers = sorted(layer for name, layer in stats if name == station)
            for layer in layers:
                calibration_cell = _standardized_period(calibration, station, layer, stats)
                calibration_cell = calibration_cell.merge(
                    labels, on=KEYS, how="left", validate="one_to_one"
                )
                for proposal in block_proposals(
                    calibration_cell,
                    config["scan"]["duration_rows"],
                    block_days=int(config["split"]["block_days"]),
                ):
                    block_label = calibration_cell.iloc[proposal.block_positions]["label"]
                    if block_label.notna().all() and block_label.eq(0).all():
                        station_scores.append(proposal.score)
                        station_score_layers.append(layer)
                validation_cell = _standardized_period(validation, station, layer, stats)
                for proposal in block_proposals(
                    validation_cell,
                    config["scan"]["duration_rows"],
                    block_days=int(config["split"]["block_days"]),
                ):
                    validation_proposals.append(
                        (
                            proposal.score,
                            validation_cell.iloc[proposal.row_positions]["ordinal"].to_numpy(
                                dtype=np.int64
                            ),
                        )
                    )
            conformal = config["conformal"]
            distinct_layers = len(set(station_score_layers))
            tail_share = tail_layer_share(
                station_scores,
                station_score_layers,
                tail_fraction=float(conformal["tail_fraction_for_homogeneity_audit"]),
            )
            eligible = (
                len(station_scores) >= int(conformal["minimum_normal_cell_blocks_per_station"])
                and distinct_layers >= int(conformal["minimum_distinct_calibration_layers"])
                and tail_share <= float(conformal["maximum_single_layer_share_of_tail"])
            )
            threshold = (
                conformal_threshold(station_scores, float(conformal["alpha"]))
                if eligible
                else None
            )
            decoded_rows = 0
            if eligible and threshold is not None:
                active_folds.add(fold)
                for score, ordinals in validation_proposals:
                    if score <= threshold:
                        continue
                    positions = ordinal_to_outer.reindex(ordinals)
                    if positions.isna().any():
                        raise ContractError("candidate rows escaped the outer fold")
                    scan[positions.to_numpy(dtype=np.int64)] = np.int8(1)
                    decoded_rows += len(ordinals)
            fold_diag["stations"][station] = {
                "fitted_layers": int(len(layers)),
                "normal_calibration_cell_blocks": int(len(station_scores)),
                "distinct_calibration_layers": int(distinct_layers),
                "tail_maximum_layer_share": float(tail_share),
                "eligible": bool(eligible),
                "threshold": threshold,
                "validation_proposals": int(len(validation_proposals)),
                "decoded_rows_before_anchor_union": int(decoded_rows),
            }
        diagnostics[fold] = fold_diag
    union, commitment = _seal(directory, anchor, scan)
    if len(active_folds) < int(config["gates"]["minimum_active_folds"]):
        result = {
            "schema_version": "p1.station_pooled_hierarchical_scan.result.v1",
            "experiment_id": EXPERIMENT_ID,
            "decision": "NO_GO_SUPPORT_EXACT_E150_NO_OP",
            "scientific_execution_count": 1,
            "active_folds": sorted(active_folds),
            "diagnostics": diagnostics,
            "commitment": commitment,
            "truth_rows_read_after_commitment": 0,
            "official_test_sample_submission_rows_read": 0,
            "submission_generated_or_uploaded": False,
            "anchor_deletions": 0,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(directory / "result.json", result)
        _finalize(directory, config, result)
        return result

    truth = _time(
        pd.read_parquet(truth_path, columns=[*KEYS, "fold", "label", "anomaly_type"])
    )
    if not outer[[*KEYS, "fold"]].astype(str).equals(truth[[*KEYS, "fold"]].astype(str)):
        raise ContractError("truth alignment failed after prediction seal")
    evaluation = truth.copy()
    evaluation["anchor_prediction"] = anchor
    evaluation["scan_prediction"] = scan
    evaluation["union_prediction"] = union
    pooled_anchor = binary_metrics(evaluation["label"], anchor)
    pooled_union = binary_metrics(evaluation["label"], union)
    fold_metrics: dict[str, Any] = {}
    active_precision: list[bool] = []
    for fold, group in evaluation.groupby("fold", sort=True):
        old = binary_metrics(group["label"], group["anchor_prediction"])
        new = binary_metrics(group["label"], group["union_prediction"])
        added = group["scan_prediction"].eq(1) & group["anchor_prediction"].eq(0)
        added_rows = int(added.sum())
        added_true = int((added & group["label"].eq(1)).sum())
        precision = added_true / added_rows if added_rows else 0.0
        if str(fold) in active_folds:
            active_precision.append(precision > float(old["f1"]) / 2.0)
        fold_metrics[str(fold)] = {
            "anchor": old,
            "challenger": new,
            "delta_f1": float(new["f1"] - old["f1"]),
            "added_rows": added_rows,
            "added_true_rows": added_true,
            "added_precision": precision,
        }
    recovered = _recoveries(evaluation)
    added = evaluation["scan_prediction"].eq(1) & evaluation["anchor_prediction"].eq(0)
    normal = evaluation.loc[evaluation["label"].eq(0)].copy()
    normal["date"] = normal["time"].dt.date
    cell_rates: dict[str, float] = {}
    for (station, layer), group in normal.groupby(["station", "layer"], sort=True):
        false_rows = int(
            (group["scan_prediction"].eq(1) & group["anchor_prediction"].eq(0)).sum()
        )
        cell_rates[f"{station}/L{int(layer)}"] = false_rows / max(group["date"].nunique(), 1)
    maximum_cell_rate = max(cell_rates.values(), default=0.0)
    bootstrap = block_bootstrap_delta(
        evaluation,
        block_days=int(config["gates"]["bootstrap_blocks_days"]),
        replicates=int(config["gates"]["bootstrap_replicates"]),
        seed=int(config["gates"]["bootstrap_seed"]),
    )
    gates = config["gates"]
    active_values = [fold_metrics[fold] for fold in sorted(active_folds)]
    checks = {
        "active_fold_support": len(active_folds) >= int(gates["minimum_active_folds"]),
        "pooled_delta": float(pooled_union["f1"] - pooled_anchor["f1"]) > 0.0,
        "positive_active_folds": sum(value["delta_f1"] > 0.0 for value in active_values)
        >= int(gates["minimum_positive_active_folds"]),
        "fold_floor": min(value["delta_f1"] for value in active_values)
        >= float(gates["minimum_fold_delta_f1"]),
        "added_precision": all(active_precision),
        "recovered_events": len(recovered) >= int(gates["minimum_recovered_long_events"]),
        "recovery_folds": len({record["fold"] for record in recovered})
        >= int(gates["minimum_recovery_folds"]),
        "recovery_cells": len({(record["station"], record["layer"]) for record in recovered})
        >= int(gates["minimum_recovery_cells"]),
        "false_positive_cap": maximum_cell_rate
        <= float(gates["maximum_normal_false_positive_rows_per_cell_day"]),
        "bootstrap_probability": bootstrap["probability_delta_positive"]
        >= float(gates["minimum_probability_delta_positive"]),
        "bootstrap_ci": bootstrap["ci90_lower"] >= float(gates["minimum_ci90_lower"]),
        "anchor_deletions_zero": bool(np.all(union[anchor == 1] == 1)),
    }
    passed = all(checks.values())
    selected = union if passed else anchor
    result = {
        "schema_version": "p1.station_pooled_hierarchical_scan.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "decision": "GO_LOCAL_ONLY" if passed else "NO_GO_QUALIFICATION_EXACT_E150_NO_OP",
        "scientific_execution_count": 1,
        "active_folds": sorted(active_folds),
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
        "maximum_normal_false_positive_rows_per_cell_day": maximum_cell_rate,
        "bootstrap": bootstrap,
        "gate_checks": checks,
        "diagnostics": diagnostics,
        "selected_arm": "challenger_union" if passed else "exact_e150_no_op",
        "selected_prediction_sha256": _prediction_sha(selected),
        "commitment": commitment,
        "truth_rows_read_after_commitment": int(len(truth)),
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
        "anchor_deletions": 0,
        "result_based_retry_or_retune": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _atomic_json(directory / "result.json", result)
    _finalize(directory, config, result)
    return result


def _finalize(directory: Path, config: dict[str, Any], result: dict[str, Any]) -> None:
    sealed = directory / "sealed_predictions.npz"
    commitment = json.loads((directory / "prediction_commitment.json").read_text(encoding="utf-8"))
    with np.load(sealed, allow_pickle=False) as archive:
        anchor = archive["anchor_prediction"]
        scan = archive["scan_prediction"]
        union = archive["union_prediction"]
    checks = {
        "execution_count_one": result["scientific_execution_count"] == 1,
        "sealed_hash": sha256_file(sealed) == commitment["sealed_file_sha256"],
        "anchor_hash": _prediction_sha(anchor) == commitment["anchor_sha256"],
        "scan_hash": _prediction_sha(scan) == commitment["scan_sha256"],
        "union_hash": _prediction_sha(union) == commitment["union_sha256"],
        "anchor_union": bool(np.array_equal(union, np.maximum(anchor, scan))),
        "truth_late": commitment["truth_rows_read_before_commitment"] == 0,
        "official_zero": result["official_test_sample_submission_rows_read"] == 0,
        "no_submission": result["submission_generated_or_uploaded"] is False,
        "no_csv": not any(directory.rglob("*.csv")),
        "no_retry": config["execution_policy"]["result_based_retry"] is False,
    }
    _atomic_json(
        directory / "independent_qa.json",
        {"experiment_id": EXPERIMENT_ID, "decision": "PASS" if all(checks.values()) else "FAIL", "checks": checks},
    )
    _atomic_json(
        directory / "manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "artifact_hashes": {
                path.name: sha256_file(path)
                for path in sorted(directory.iterdir())
                if path.is_file() and path.name != "manifest.json"
            },
            "code_hashes": {
                "config": sha256_file(CONFIG),
                "module": sha256_file(MODULE),
                "runner": sha256_file(RUNNER),
                "test": sha256_file(TEST),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.check_only == args.execute:
        raise SystemExit("choose exactly one mode")
    config = load_config()
    directory = ROOT / config["artifact_directory"]
    if directory.exists():
        raise FileExistsError(f"one-shot output exists: {directory}")
    if args.check_only:
        print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "PASS_CHECK_ONLY", "truth_rows_read": 0, "official_rows_read": 0}, indent=2))
        return
    print(json.dumps(execute(config), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
