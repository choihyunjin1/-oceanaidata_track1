"""Run one frozen Q2 event-verifier experiment without opening Q3/Q4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingClassifier

from p1_qc.frozen_direct_event_verifier import (
    EventProposal,
    assign_split,
    binary_metrics,
    build_event_proposals,
    chronological_boundaries,
    decode_additions,
    evaluate_union,
    split_intervals,
    utility_targets,
)

EXPERIMENT_ID = "p1_frozen_direct_event_verifier_blocked_20260828_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
MODULE_PATH = ROOT / "src" / "p1_qc" / "frozen_direct_event_verifier.py"
SOURCE_CONFIG_PATH = (
    ROOT / "configs" / "experiments" / "p1_incumbent_preserving_mstcn_asrf_v2.json"
)
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
KEY_COLUMNS = ("station", "year", "layer", "time")
Q2_FOLD = "2025_q2"


class ContractError(RuntimeError):
    """Raised when a preregistered immutable contract is not satisfied."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
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
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config() -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    generator = config["generator"]
    verifier = config["verifier"]
    if (
        float(generator["query_threshold"]) != 0.5
        or int(generator["minimum_component_rows"]) != 19
        or generator["retrain_allowed"] is not False
    ):
        raise ContractError("frozen generator contract changed")
    if (
        int(verifier["max_depth"]) != 3
        or int(verifier["max_iter"]) > 100
        or int(verifier["fit_count"]) != 1
        or verifier["threshold_search"] is not False
        or verifier["grid_search"] is not False
        or verifier["result_based_retry"] is not False
    ):
        raise ContractError("single bounded verifier contract changed")
    if config["features"]["station_and_layer_ids_as_model_features"] is not False:
        raise ContractError("station/layer IDs entered the model contract")
    return config


def _verify_registered(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ContractError(f"missing immutable input: {path}")
    observed = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if observed["bytes"] != int(record["bytes"]) or observed["sha256"] != record["sha256"]:
        raise ContractError(f"immutable input changed: {path}")
    return observed


def _ordered_key_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, KEY_COLUMNS].astype(str).itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _kst_text(value: pd.Timestamp) -> str:
    return value.tz_convert("Asia/Seoul").isoformat()


def _row_mask(
    times: pd.Series,
    interval: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> np.ndarray:
    start, end = interval
    mask = np.ones(len(times), dtype=bool)
    if start is not None:
        mask &= (times >= start).to_numpy()
    if end is not None:
        mask &= (times < end).to_numpy()
    return np.flatnonzero(mask)


def _q2_membership(path: Path) -> pd.DataFrame:
    scanner = ds.dataset(path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "fold"],
        filter=ds.field("fold") == Q2_FOLD,
        use_threads=True,
    )
    frame = scanner.to_table().to_pandas().reset_index(drop=True)
    if frame.empty or frame.duplicated(list(KEY_COLUMNS)).any():
        raise ContractError("Q2 membership is empty or duplicated")
    return frame


def _load_q2_covariates(
    source_config: dict[str, Any], selected_numeric: Sequence[str]
) -> tuple[pd.DataFrame, np.ndarray, dict[str, dict[str, Any]]]:
    immutable = source_config["immutable_inputs"]
    truth_record = immutable["frozen_truth_and_folds"]
    key_record = immutable["feature_key_sidecar"]
    cache_record = immutable["feature_cache"]
    paths = {
        "truth": ROOT / truth_record["path"],
        "keys": ROOT / key_record["path"],
        "features": ROOT / cache_record["path"],
    }
    verified = {
        "truth": _verify_registered(paths["truth"], truth_record),
        "keys": _verify_registered(paths["keys"], key_record),
        "features": _verify_registered(paths["features"], cache_record),
    }
    membership = _q2_membership(paths["truth"])
    all_keys = pd.read_parquet(paths["keys"], columns=["ordinal", *KEY_COLUMNS])
    lookup = pd.MultiIndex.from_frame(all_keys.loc[:, KEY_COLUMNS].astype(str))
    if not lookup.is_unique:
        raise ContractError("feature key sidecar is not unique")
    requested = pd.MultiIndex.from_frame(membership.loc[:, KEY_COLUMNS].astype(str))
    ordinals = lookup.get_indexer(requested)
    if np.any(ordinals < 0) or len(np.unique(ordinals)) != len(ordinals):
        raise ContractError("Q2 membership does not bind one-to-one to feature rows")
    features = pd.read_parquet(paths["features"], columns=list(selected_numeric))
    numeric = features.iloc[ordinals].to_numpy(dtype=np.float32, copy=True)
    del features, all_keys
    return membership, numeric, verified


def _load_generator(config: dict[str, Any], expected_rows: int) -> dict[str, np.ndarray]:
    generator = config["generator"]
    path = ROOT / generator["prediction_path"]
    if path.stat().st_size != int(generator["prediction_bytes"]):
        raise ContractError("frozen generator byte size changed")
    if _sha256(path) != generator["prediction_sha256"]:
        raise ContractError("frozen generator hash changed")
    receipt = _load_json(ROOT / generator["receipt_path"])
    if receipt.get("score_sha256") != generator["prediction_sha256"]:
        raise ContractError("generator receipt no longer binds the score file")
    with np.load(path, allow_pickle=False) as archive:
        result = {
            name: archive[name].copy()
            for name in ("confidence", "proposal", "anchor")
        }
    if any(value.shape != (expected_rows,) for value in result.values()):
        raise ContractError("generator array shape differs from Q2 membership")
    if not np.isin(result["proposal"], [0, 1]).all() or not np.isin(
        result["anchor"], [0, 1]
    ).all():
        raise ContractError("generator proposal/anchor is not binary")
    replay = (result["confidence"] >= float(generator["query_threshold"])).astype(
        np.int8
    )
    if not np.array_equal(replay, result["proposal"].astype(np.int8)):
        raise ContractError("stored proposal differs from fixed threshold 0.5 replay")
    return result


def _proposal_bank_sha(proposals: Sequence[EventProposal]) -> str:
    digest = hashlib.sha256()
    for proposal in proposals:
        digest.update(proposal.proposal_id.encode())
        digest.update(str(proposal.start_time).encode())
        digest.update(str(proposal.end_time).encode())
        digest.update(np.asarray(proposal.features, dtype="<f8").tobytes())
    return digest.hexdigest()


def _truth_filter(
    interval: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> Any:
    expression = ds.field("fold") == Q2_FOLD
    start, end = interval
    if start is not None:
        expression &= ds.field("time") >= _kst_text(start)
    if end is not None:
        expression &= ds.field("time") < _kst_text(end)
    return expression


def _open_truth_split(
    path: Path,
    membership: pd.DataFrame,
    rows: np.ndarray,
    interval: tuple[pd.Timestamp | None, pd.Timestamp | None],
) -> tuple[np.ndarray, np.ndarray]:
    scanner = ds.dataset(path, format="parquet").scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type"],
        filter=_truth_filter(interval),
        use_threads=True,
    )
    frame = scanner.to_table().to_pandas().reset_index(drop=True)
    expected = membership.iloc[rows].reset_index(drop=True)
    if len(frame) != len(expected):
        raise ContractError("split truth row count differs from blind membership")
    for column in KEY_COLUMNS:
        if not np.array_equal(
            frame[column].astype(str).to_numpy(),
            expected[column].astype(str).to_numpy(),
        ):
            raise ContractError(f"split truth key order differs: {column}")
    labels = frame["label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, [0, 1]).all():
        raise ContractError("Q2 truth is not binary")
    anomaly = frame["anomaly_type"].fillna("").astype(str).to_numpy()
    return labels, anomaly


def _proposal_support(
    proposals: Sequence[EventProposal], targets: np.ndarray
) -> dict[str, Any]:
    positive = int(targets.sum())
    negative = int(len(targets) - positive)
    hard_negative_ratio = negative / positive if positive else None
    cell_counts: dict[str, int] = {}
    for proposal, target in zip(proposals, targets, strict=True):
        if target:
            cell = f"{proposal.station}|L{proposal.layer}"
            cell_counts[cell] = cell_counts.get(cell, 0) + 1
    maximum_cell_share = max(cell_counts.values()) / positive if positive else None
    return {
        "proposals": len(proposals),
        "utility_positive": positive,
        "hard_negative": negative,
        "hard_negative_to_positive_ratio": hard_negative_ratio,
        "maximum_single_station_layer_positive_share": maximum_cell_share,
        "positive_station_layer_cells": len(cell_counts),
    }


def _calibration_pass(metrics: dict[str, Any], config: dict[str, Any]) -> bool:
    gate = config["calibration_arm"]["choose_model_if_all"]
    return bool(
        metrics["delta_f1"] > float(gate["delta_f1_strictly_above"])
        and metrics["added_precision"] > metrics["anchor_f1_over_2"]
        and metrics["fp_per_day_relative"]
        <= float(gate["normal_fp_per_day_relative_lte"])
    )


def _type_mask(values: np.ndarray, name: str) -> np.ndarray:
    pattern = re.compile(rf"(?<![a-z]){re.escape(name.lower())}(?![a-z])")
    return np.asarray([bool(pattern.search(str(value).lower())) for value in values])


def _direction_diagnostics(
    truth: np.ndarray,
    anomaly: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    membership: pd.DataFrame,
    rows: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    minimum = int(config["qualification_gate"]["supported_positive_rows_min"])
    local_truth = truth[rows]
    local_anchor = anchor[rows]
    local_candidate = candidate[rows]
    local_types = anomaly[rows]
    type_rows: dict[str, Any] = {}
    supported_type_deltas: list[float] = []
    for name in config["target"]["eligible_long_event_types"]:
        selected = (local_truth == 1) & _type_mask(local_types, name)
        positives = int(selected.sum())
        if positives >= minimum:
            anchor_recall = float(np.mean(local_anchor[selected] == 1))
            candidate_recall = float(np.mean(local_candidate[selected] == 1))
            delta = candidate_recall - anchor_recall
            supported_type_deltas.append(delta)
            type_rows[name] = {
                "positive_rows": positives,
                "anchor_recall": anchor_recall,
                "candidate_recall": candidate_recall,
                "delta_recall": delta,
            }
        else:
            type_rows[name] = {"positive_rows": positives, "supported": False}

    local_keys = membership.iloc[rows].reset_index(drop=True)
    cell_rows: dict[str, Any] = {}
    supported_cell_deltas: list[float] = []
    grouped = local_keys.groupby(["station", "layer"], sort=True, observed=True)
    for (station, layer), part in grouped:
        positions = part.index.to_numpy(dtype=np.int64)
        positives = int(np.sum(local_truth[positions] == 1))
        key = f"{station}|L{int(layer)}"
        if positives >= minimum:
            base = binary_metrics(local_truth[positions], local_anchor[positions])
            new = binary_metrics(local_truth[positions], local_candidate[positions])
            delta = float(new["f1"] - base["f1"])
            supported_cell_deltas.append(delta)
            cell_rows[key] = {
                "positive_rows": positives,
                "anchor_f1": base["f1"],
                "candidate_f1": new["f1"],
                "delta_f1": delta,
            }
        else:
            cell_rows[key] = {"positive_rows": positives, "supported": False}
    return {
        "types": type_rows,
        "station_layers": cell_rows,
        "supported_type_count": len(supported_type_deltas),
        "nonnegative_supported_type_fraction": (
            float(np.mean(np.asarray(supported_type_deltas) >= 0.0))
            if supported_type_deltas
            else None
        ),
        "supported_station_layer_count": len(supported_cell_deltas),
        "nonnegative_supported_station_layer_fraction": (
            float(np.mean(np.asarray(supported_cell_deltas) >= 0.0))
            if supported_cell_deltas
            else None
        ),
    }


def _newly_recovered_long_events(
    truth: np.ndarray,
    anomaly: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    membership: pd.DataFrame,
    rows: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    minimum = int(config["target"]["minimum_long_event_rows"])
    eligible = np.zeros(len(rows), dtype=bool)
    for name in config["target"]["eligible_long_event_types"]:
        eligible |= _type_mask(anomaly[rows], name)
    eligible &= truth[rows] == 1
    local = membership.iloc[rows].reset_index(drop=True).copy()
    local["__position"] = np.arange(len(rows), dtype=np.int64)
    events = 0
    anchor_missed = 0
    newly_recovered = 0
    duration_rows: list[int] = []
    for _group, part in local.groupby(
        ["station", "year", "layer"], sort=False, observed=True
    ):
        positions = part["__position"].to_numpy(dtype=np.int64)
        times = pd.to_datetime(part["time"], utc=True, format="mixed").to_numpy()
        active = eligible[positions]
        cursor = 0
        while cursor < len(positions):
            if not active[cursor]:
                cursor += 1
                continue
            end = cursor + 1
            while (
                end < len(positions)
                and active[end]
                and times[end] - times[end - 1] == pd.Timedelta(minutes=10)
            ):
                end += 1
            if end - cursor >= minimum:
                event_rows = rows[positions[cursor:end]]
                events += 1
                duration_rows.append(len(event_rows))
                missed = not bool(np.any(anchor[event_rows] == 1))
                anchor_missed += int(missed)
                newly_recovered += int(missed and bool(np.any(candidate[event_rows] == 1)))
            cursor = end
    return {
        "eligible_long_events": events,
        "anchor_fully_missed_long_events": anchor_missed,
        "newly_recovered_long_events": newly_recovered,
        "duration_rows_min": min(duration_rows) if duration_rows else None,
        "duration_rows_max": max(duration_rows) if duration_rows else None,
    }


def _bootstrap_delta(
    truth: np.ndarray,
    anchor: np.ndarray,
    candidate: np.ndarray,
    membership: pd.DataFrame,
    rows: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    times = pd.to_datetime(membership.iloc[rows]["time"], utc=True, format="mixed")
    days = times.dt.floor("D").astype(str).to_numpy()
    unique_days = np.unique(days)
    if len(unique_days) < 2:
        return {"available": False, "reason": "fewer than two qualification days"}
    daily = np.zeros((len(unique_days), 6), dtype=np.int64)
    for index, day in enumerate(unique_days):
        selected = days == day
        y = truth[rows][selected]
        base = anchor[rows][selected]
        new = candidate[rows][selected]
        for offset, prediction in ((0, base), (3, new)):
            daily[index, offset] = np.sum((y == 1) & (prediction == 1))
            daily[index, offset + 1] = np.sum((y == 0) & (prediction == 1))
            daily[index, offset + 2] = np.sum((y == 1) & (prediction == 0))
    gate = config["qualification_gate"]
    replicates = int(gate["block_bootstrap_replicates"])
    block_days = int(gate["block_bootstrap_days"])
    rng = np.random.default_rng(int(config["verifier"]["random_state"]))
    deltas = np.empty(replicates, dtype=np.float64)
    blocks = int(np.ceil(len(unique_days) / block_days))
    for replicate in range(replicates):
        starts = rng.integers(0, len(unique_days), size=blocks)
        sampled = np.concatenate(
            [
                (np.arange(start, start + block_days, dtype=np.int64) % len(unique_days))
                for start in starts
            ]
        )[: len(unique_days)]
        counts = daily[sampled].sum(axis=0)
        base_denominator = 2 * counts[0] + counts[1] + counts[2]
        new_denominator = 2 * counts[3] + counts[4] + counts[5]
        base_f1 = 2 * counts[0] / base_denominator if base_denominator else 0.0
        new_f1 = 2 * counts[3] / new_denominator if new_denominator else 0.0
        deltas[replicate] = new_f1 - base_f1
    quantiles = np.quantile(deltas, [0.05, 0.50, 0.95])
    return {
        "available": True,
        "replicates": replicates,
        "block_days": block_days,
        "unique_days": len(unique_days),
        "delta_f1_ci90_lower": float(quantiles[0]),
        "delta_f1_median": float(quantiles[1]),
        "delta_f1_ci90_upper": float(quantiles[2]),
        "positive_delta_fraction": float(np.mean(deltas > 0.0)),
    }


def _qualification_pass(
    metrics: dict[str, Any],
    event_metrics: dict[str, Any],
    directions: dict[str, Any],
    bootstrap: dict[str, Any],
    config: dict[str, Any],
) -> tuple[bool, dict[str, bool]]:
    gate = config["qualification_gate"]
    checks = {
        "delta_f1": metrics["delta_f1"] > float(gate["delta_f1_strictly_above"]),
        "added_precision": metrics["added_precision"]
        > metrics["anchor_f1_over_2"],
        "fp_per_day": metrics["fp_per_day_relative"]
        <= float(gate["normal_fp_per_day_relative_lte"]),
        "long_events": event_metrics["newly_recovered_long_events"]
        >= int(gate["minimum_newly_recovered_long_events"]),
        "type_direction": directions["nonnegative_supported_type_fraction"]
        is not None
        and directions["nonnegative_supported_type_fraction"]
        >= float(gate["minimum_nonnegative_supported_type_fraction"]),
        "cell_direction": directions["nonnegative_supported_station_layer_fraction"]
        is not None
        and directions["nonnegative_supported_station_layer_fraction"]
        >= float(gate["minimum_nonnegative_supported_station_layer_fraction"]),
        "bootstrap": bootstrap.get("available", False)
        and bootstrap["delta_f1_ci90_lower"]
        >= float(gate["block_bootstrap_ci90_lower_gte"]),
        "anchor_preserved": metrics["anchor_positive_removed_rows"] == 0,
    }
    return all(checks.values()), checks


def _write_manifest(source_inputs: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for path in sorted(ARTIFACT_DIR.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(CONFIG_PATH),
        "module_sha256": _sha256(MODULE_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "generator_sha256": _load_config()["generator"]["prediction_sha256"],
        "immutable_inputs": source_inputs,
        "q3_q4_fold_surface_reads": 0,
        "q3_q4_truth_reads": 0,
        "official_test_sample_submission_reads": 0,
        "artifacts": artifacts,
    }
    _atomic_json(ARTIFACT_DIR / "manifest.json", manifest)
    return manifest


def check_only() -> dict[str, Any]:
    config = _load_config()
    if ARTIFACT_DIR.exists():
        raise ContractError("append-only artifact namespace already exists")
    if not SOURCE_CONFIG_PATH.is_file():
        raise ContractError("source config missing")
    generator = config["generator"]
    generator_path = ROOT / generator["prediction_path"]
    receipt_path = ROOT / generator["receipt_path"]
    if not generator_path.is_file() or not receipt_path.is_file():
        raise ContractError("frozen Q2 generator dependency missing")
    if generator_path.stat().st_size != int(generator["prediction_bytes"]):
        raise ContractError("frozen Q2 generator size changed")
    if _sha256(generator_path) != generator["prediction_sha256"]:
        raise ContractError("frozen Q2 generator hash changed")
    receipt = _load_json(receipt_path)
    if receipt.get("score_sha256") != generator["prediction_sha256"]:
        raise ContractError("frozen Q2 generator receipt changed")
    return {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": _sha256(CONFIG_PATH),
        "module_sha256": _sha256(MODULE_PATH),
        "runner_sha256": _sha256(Path(__file__)),
        "generator_sha256": generator["prediction_sha256"],
        "generator_threshold": generator["query_threshold"],
        "minimum_component_rows": generator["minimum_component_rows"],
        "historical_surface": "Q2 historically exposed; no fresh claim allowed",
        "q3_q4_fold_surface_reads": 0,
        "q3_q4_truth_reads": 0,
        "official_test_sample_submission_reads": 0,
        "result": "PASS",
    }


def _terminal(status: str, **fields: Any) -> dict[str, Any]:
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": status,
        "historical_surface": "Q2 historically exposed; directional evidence only",
        "q3_q4_fold_surface_reads": 0,
        "q3_q4_truth_reads": 0,
        "official_test_sample_submission_reads": 0,
        "submission_generated_or_uploaded": False,
        **fields,
    }
    _atomic_json(ARTIFACT_DIR / "terminal_result.json", result)
    return result


def execute() -> dict[str, Any]:
    preflight = check_only()
    config = _load_config()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    _atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
    source_config = _load_json(SOURCE_CONFIG_PATH)
    selected_numeric = tuple(config["features"]["numeric_sources"])
    membership, numeric, verified = _load_q2_covariates(
        source_config, selected_numeric
    )
    generator = _load_generator(config, len(membership))
    receipt = _load_json(ROOT / config["generator"]["receipt_path"])
    if _ordered_key_sha(membership) != receipt["ordered_key_sha256"]:
        raise ContractError("Q2 key order differs from frozen generator receipt")
    times = pd.to_datetime(membership["time"], utc=True, format="mixed")
    boundary1, boundary2 = chronological_boundaries(times)
    intervals = split_intervals(
        boundary1, boundary2, int(config["split"]["boundary_purge_days"])
    )
    split_rows = {
        name: _row_mask(times, interval) for name, interval in intervals.items()
    }
    proposals, feature_names = build_event_proposals(
        membership,
        numeric,
        selected_numeric,
        generator["confidence"].astype(np.float64),
        generator["proposal"].astype(np.int8),
        generator["anchor"].astype(np.int8),
        selected_numeric=selected_numeric,
        minimum_rows=int(config["generator"]["minimum_component_rows"]),
        context_rows=int(config["features"]["context_rows_each_side"]),
    )
    split_proposals: dict[str, list[EventProposal]] = {
        name: [] for name in intervals
    }
    dropped = 0
    for proposal in proposals:
        split = assign_split(proposal, intervals)
        if split is None:
            dropped += 1
        else:
            split_proposals[split].append(proposal)
    proposal_receipt = {
        "schema_version": "p1.frozen_direct_event_verifier.proposal_receipt.v1",
        "generator_path": config["generator"]["prediction_path"],
        "generator_bytes": config["generator"]["prediction_bytes"],
        "generator_sha256": config["generator"]["prediction_sha256"],
        "generator_threshold": config["generator"]["query_threshold"],
        "minimum_component_rows": config["generator"]["minimum_component_rows"],
        "ordered_key_sha256": _ordered_key_sha(membership),
        "q2_rows": len(membership),
        "boundaries_utc": [boundary1.isoformat(), boundary2.isoformat()],
        "purge_days": config["split"]["boundary_purge_days"],
        "split_rows": {name: len(rows) for name, rows in split_rows.items()},
        "split_proposals": {
            name: len(values) for name, values in split_proposals.items()
        },
        "dropped_or_purged_proposals": dropped,
        "feature_count": len(feature_names),
        "feature_schema_sha256": _bytes_sha(
            json.dumps(feature_names, separators=(",", ":")).encode()
        ),
        "proposal_bank_sha256": _proposal_bank_sha(proposals),
        "station_layer_ids_as_model_features": False,
        "truth_columns_read_before_receipt": 0,
        "q3_q4_fold_surface_reads": 0,
        "q3_q4_truth_reads": 0,
        "official_test_sample_submission_reads": 0,
    }
    _atomic_json(ARTIFACT_DIR / "proposal_pretruth_receipt.json", proposal_receipt)
    del numeric

    truth_path = ROOT / source_config["immutable_inputs"]["frozen_truth_and_folds"][
        "path"
    ]
    truth = np.full(len(membership), -1, dtype=np.int8)
    anomaly = np.full(len(membership), "", dtype=object)
    for name in ("train", "calibration"):
        labels, types = _open_truth_split(
            truth_path, membership, split_rows[name], intervals[name]
        )
        truth[split_rows[name]] = labels
        anomaly[split_rows[name]] = types

    targets: dict[str, np.ndarray] = {}
    support: dict[str, Any] = {}
    target_diagnostics: dict[str, Any] = {}
    for name in ("train", "calibration"):
        targets[name], details = utility_targets(
            split_proposals[name],
            truth,
            generator["anchor"].astype(np.int8),
            split_rows[name],
        )
        support[name] = _proposal_support(split_proposals[name], targets[name])
        target_diagnostics[name] = {
            "added_tp": int(sum(item["added_tp"] for item in details)),
            "added_fp": int(sum(item["added_fp"] for item in details)),
            "utility_sum": float(sum(item["utility"] for item in details)),
        }
    minimums = config["support_preflight"][
        "minimum_utility_positive_proposals"
    ]
    train_ratio = support["train"]["hard_negative_to_positive_ratio"]
    train_share = support["train"]["maximum_single_station_layer_positive_share"]
    initial_checks = {
        "train_positive": support["train"]["utility_positive"]
        >= int(minimums["train"]),
        "calibration_positive": support["calibration"]["utility_positive"]
        >= int(minimums["calibration"]),
        "train_hard_negative_ratio": train_ratio is not None
        and train_ratio
        >= float(
            config["support_preflight"][
                "minimum_hard_negative_to_positive_ratio_train"
            ]
        ),
        "train_cell_concentration": train_share is not None
        and train_share
        <= float(
            config["support_preflight"][
                "maximum_single_station_layer_positive_share_train"
            ]
        ),
    }
    support_record = {
        "stage": "pre_fit_train_calibration",
        "support": support,
        "target_diagnostics": target_diagnostics,
        "checks": initial_checks,
        "qualification_truth_opened": False,
    }
    _atomic_json(ARTIFACT_DIR / "support_preflight_train_calibration.json", support_record)
    if not all(initial_checks.values()):
        result = _terminal(
            "NO_GO_SUPPORT_PREFLIGHT",
            fit_count=0,
            qualification_truth_opened=False,
            qualification_evaluated=False,
            failed_checks=[name for name, passed in initial_checks.items() if not passed],
        )
        _write_manifest(verified)
        return result

    train_x = np.vstack(
        [proposal.features for proposal in split_proposals["train"]]
    )
    calibration_x = np.vstack(
        [proposal.features for proposal in split_proposals["calibration"]]
    )
    verifier = config["verifier"]
    model = HistGradientBoostingClassifier(
        learning_rate=float(verifier["learning_rate"]),
        max_iter=int(verifier["max_iter"]),
        max_depth=int(verifier["max_depth"]),
        l2_regularization=float(verifier["l2_regularization"]),
        class_weight=str(verifier["class_weight"]),
        random_state=int(verifier["random_state"]),
    )
    model.fit(train_x, targets["train"])
    calibration_scores = model.predict_proba(calibration_x)[:, 1]
    selected_calibration = calibration_scores >= float(
        verifier["probability_threshold"]
    )
    calibration_additions = decode_additions(
        len(membership), split_proposals["calibration"], selected_calibration
    )
    zero_add = np.zeros(len(membership), dtype=np.int8)
    no_op_metrics = evaluate_union(
        truth,
        generator["anchor"].astype(np.int8),
        zero_add,
        split_rows["calibration"],
    )
    model_metrics = evaluate_union(
        truth,
        generator["anchor"].astype(np.int8),
        calibration_additions,
        split_rows["calibration"],
    )
    use_model = _calibration_pass(model_metrics, config)
    calibration_record = {
        "arms": {
            "ZERO_ADD_NO_OP": no_op_metrics,
            "FIXED_SCORE_GTE_0.5": model_metrics,
        },
        "selected_arm": "FIXED_SCORE_GTE_0.5" if use_model else "ZERO_ADD_NO_OP",
        "fixed_probability_threshold": verifier["probability_threshold"],
        "fit_count": 1,
        "threshold_search_count": 0,
        "grid_search_count": 0,
        "result_based_retry_count": 0,
    }
    _atomic_json(ARTIFACT_DIR / "calibration_arm.json", calibration_record)
    if not use_model:
        _atomic_json(
            ARTIFACT_DIR / "changed_row_manifest.json",
            {
                "calibration_added_rows": model_metrics["added_rows"],
                "calibration_addition_sha256": _bytes_sha(
                    calibration_additions.astype(np.int8).tobytes()
                ),
                "selected_arm": "ZERO_ADD_NO_OP",
                "qualification_not_opened": True,
            },
        )
        result = _terminal(
            "NO_GO_CALIBRATION_NO_OP",
            fit_count=1,
            selected_arm="ZERO_ADD_NO_OP",
            calibration_delta_f1=model_metrics["delta_f1"],
            calibration_added_precision=model_metrics["added_precision"],
            qualification_truth_opened=False,
            qualification_evaluated=False,
        )
        _write_manifest(verified)
        return result

    qualification_x = np.vstack(
        [proposal.features for proposal in split_proposals["qualification"]]
    )
    qualification_scores = model.predict_proba(qualification_x)[:, 1]
    score_path = ARTIFACT_DIR / "qualification_blind_event_scores.npz"
    _atomic_npz(score_path, score=qualification_scores.astype(np.float32))
    _atomic_json(
        ARTIFACT_DIR / "qualification_blind_event_scores_receipt.json",
        {
            "score_path": score_path.name,
            "score_bytes": score_path.stat().st_size,
            "score_sha256": _sha256(score_path),
            "proposal_bank_sha256": _proposal_bank_sha(
                split_proposals["qualification"]
            ),
            "fixed_probability_threshold": verifier["probability_threshold"],
            "fit_count": 1,
            "qualification_truth_columns_read_before_receipt": 0,
            "q3_q4_truth_reads": 0,
            "official_test_sample_submission_reads": 0,
        },
    )
    labels, types = _open_truth_split(
        truth_path,
        membership,
        split_rows["qualification"],
        intervals["qualification"],
    )
    truth[split_rows["qualification"]] = labels
    anomaly[split_rows["qualification"]] = types
    targets["qualification"], qualification_details = utility_targets(
        split_proposals["qualification"],
        truth,
        generator["anchor"].astype(np.int8),
        split_rows["qualification"],
    )
    support["qualification"] = _proposal_support(
        split_proposals["qualification"], targets["qualification"]
    )
    qualification_support_pass = support["qualification"][
        "utility_positive"
    ] >= int(minimums["qualification"])
    _atomic_json(
        ARTIFACT_DIR / "support_preflight_qualification.json",
        {
            "stage": "post_blind_receipt_pre_metric",
            "support": support["qualification"],
            "target_diagnostics": {
                "added_tp": int(
                    sum(item["added_tp"] for item in qualification_details)
                ),
                "added_fp": int(
                    sum(item["added_fp"] for item in qualification_details)
                ),
                "utility_sum": float(
                    sum(item["utility"] for item in qualification_details)
                ),
            },
            "minimum_utility_positive": minimums["qualification"],
            "pass": qualification_support_pass,
            "qualification_truth_opened_after_blind_receipt": True,
        },
    )
    if not qualification_support_pass:
        result = _terminal(
            "NO_GO_SUPPORT_PREFLIGHT_QUALIFICATION",
            fit_count=1,
            selected_arm="FIXED_SCORE_GTE_0.5",
            qualification_truth_opened=True,
            qualification_evaluated=False,
            qualification_utility_positive=support["qualification"][
                "utility_positive"
            ],
        )
        _write_manifest(verified)
        return result

    selected_qualification = qualification_scores >= float(
        verifier["probability_threshold"]
    )
    qualification_additions = decode_additions(
        len(membership), split_proposals["qualification"], selected_qualification
    )
    candidate = np.maximum(
        generator["anchor"].astype(np.int8), qualification_additions
    ).astype(np.int8)
    qualification_metrics = evaluate_union(
        truth,
        generator["anchor"].astype(np.int8),
        qualification_additions,
        split_rows["qualification"],
    )
    event_metrics = _newly_recovered_long_events(
        truth,
        anomaly,
        generator["anchor"].astype(np.int8),
        candidate,
        membership,
        split_rows["qualification"],
        config,
    )
    directions = _direction_diagnostics(
        truth,
        anomaly,
        generator["anchor"].astype(np.int8),
        candidate,
        membership,
        split_rows["qualification"],
        config,
    )
    bootstrap = _bootstrap_delta(
        truth,
        generator["anchor"].astype(np.int8),
        candidate,
        membership,
        split_rows["qualification"],
        config,
    )
    passed, checks = _qualification_pass(
        qualification_metrics, event_metrics, directions, bootstrap, config
    )
    aggregate = {
        "experiment_id": EXPERIMENT_ID,
        "historical_surface": "Q2 historically exposed; directional evidence only",
        "calibration": calibration_record,
        "qualification": qualification_metrics,
        "long_events": event_metrics,
        "directions": directions,
        "bootstrap": bootstrap,
        "qualification_gate_checks": checks,
        "qualification_gate_pass": passed,
        "strict_fresh_claim_allowed": False,
    }
    _atomic_json(ARTIFACT_DIR / "aggregate_metrics.json", aggregate)
    changed = (
        generator["anchor"].astype(np.int8) == 0
    ) & (qualification_additions == 1)
    _atomic_json(
        ARTIFACT_DIR / "changed_row_manifest.json",
        {
            "calibration_added_rows": model_metrics["added_rows"],
            "calibration_addition_sha256": _bytes_sha(
                calibration_additions.astype(np.int8).tobytes()
            ),
            "qualification_added_rows": int(changed.sum()),
            "qualification_changed_mask_sha256": _bytes_sha(
                changed.astype(np.int8).tobytes()
            ),
            "qualification_candidate_sha256": _bytes_sha(
                candidate.astype(np.int8).tobytes()
            ),
            "anchor_positive_removed_rows": qualification_metrics[
                "anchor_positive_removed_rows"
            ],
            "raw_keys_persisted": False,
        },
    )
    status = "GO_INFORMATION_VALUE" if passed else "NO_GO_QUALIFICATION_GATE"
    result = _terminal(
        status,
        fit_count=1,
        selected_arm="FIXED_SCORE_GTE_0.5",
        qualification_truth_opened=True,
        qualification_evaluated=True,
        qualification_delta_f1=qualification_metrics["delta_f1"],
        qualification_added_precision=qualification_metrics["added_precision"],
        newly_recovered_long_events=event_metrics["newly_recovered_long_events"],
        qualification_gate_checks=checks,
        strict_fresh_claim_allowed=False,
    )
    _write_manifest(verified)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.check_only == args.execute:
        parser.error("choose exactly one of --check-only or --execute")
    result = execute() if args.execute else check_only()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
