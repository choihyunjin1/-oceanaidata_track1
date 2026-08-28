"""Run one bounded conditional real-event donor pilot on pre-Q2 history and Q2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.ensemble import HistGradientBoostingClassifier

from p1_qc.p1_conditional_real_event_donor_20260828_v1 import (
    binary_metrics,
    block_bootstrap_delta,
    conditional_transplant,
    decode_scores,
    evaluate_anchor_union,
    event_support,
    extract_long_events,
    extract_mask_events,
    newly_recovered_events,
    proposal_support_metrics,
)

EXPERIMENT_ID = "p1_conditional_real_event_donor_20260828_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
MODULE_PATH = ROOT / "src" / "p1_qc" / f"{EXPERIMENT_ID}.py"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
KEY_COLUMNS = ("station", "year", "layer", "time")


class ContractError(RuntimeError):
    """Raised when a sealed experiment contract is not satisfied."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_key_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, KEY_COLUMNS].astype(str).itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
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


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID changed")
    model = config["model"]
    synthetic = config["synthetic"]
    if (
        int(model["fit_count"]) != 1
        or bool(model["threshold_search"])
        or bool(model["grid_search"])
        or bool(model["result_based_retry"])
    ):
        raise ContractError("single fixed fit contract changed")
    if float(synthetic["normal_row_budget_fraction"]) != 0.0025:
        raise ContractError("synthetic rate changed")
    if int(config["split"]["boundary_purge_days"]) != 15:
        raise ContractError("purge contract changed")
    if config["features"]["station_and_layer_ids_as_model_features"] is not False:
        raise ContractError("station/layer IDs entered model features")
    return config


def verify_inputs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    verified: dict[str, dict[str, Any]] = {}
    for name, record in config["immutable_inputs"].items():
        path = ROOT / record["path"]
        if not path.is_file():
            raise ContractError(f"missing immutable input: {name}")
        observed = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        if observed["bytes"] != int(record["bytes"]) or observed["sha256"] != record["sha256"]:
            raise ContractError(f"immutable input changed: {name}")
        verified[name] = observed
    return verified


def multiindex(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, KEY_COLUMNS].astype(str))


def map_ordinals(all_keys: pd.DataFrame, requested: pd.DataFrame) -> np.ndarray:
    lookup = multiindex(all_keys)
    if not lookup.is_unique:
        raise ContractError("key sidecar is not unique")
    ordinals = lookup.get_indexer(multiindex(requested))
    if np.any(ordinals < 0) or len(np.unique(ordinals)) != len(ordinals):
        raise ContractError("requested keys do not bind one-to-one")
    return ordinals.astype(np.int64)


def load_surfaces(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    immutable = config["immutable_inputs"]
    all_keys = pd.read_parquet(
        ROOT / immutable["feature_key_sidecar"]["path"], columns=["ordinal", *KEY_COLUMNS]
    )
    if not np.array_equal(all_keys["ordinal"].to_numpy(), np.arange(len(all_keys))):
        raise ContractError("feature ordinals are not contiguous")
    split = config["split"]
    cutoff = pd.Timestamp(split["calibration_end_exclusive_utc"]).tz_convert(
        "Asia/Seoul"
    ).isoformat()
    label_scanner = ds.dataset(
        ROOT / immutable["training_labels"]["path"], format="parquet"
    ).scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type"],
        filter=ds.field("time") < cutoff,
        use_threads=True,
    )
    historical = label_scanner.to_table().to_pandas().reset_index(drop=True)
    historical["ordinal"] = map_ordinals(all_keys, historical)
    q2_scanner = ds.dataset(
        ROOT / immutable["frozen_truth_and_folds"]["path"], format="parquet"
    ).scanner(
        columns=[*KEY_COLUMNS, "fold"],
        filter=ds.field("fold") == str(split["q2_fold"]),
        use_threads=True,
    )
    q2 = q2_scanner.to_table().to_pandas().reset_index(drop=True)
    q2["ordinal"] = map_ordinals(all_keys, q2)
    return all_keys, historical, q2


def split_rows(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, np.ndarray]:
    times = pd.to_datetime(frame["time"], utc=True, format="mixed")
    split = config["split"]
    return {
        "train": np.flatnonzero(
            (times < pd.Timestamp(split["training_end_exclusive_utc"])).to_numpy()
        ),
        "calibration": np.flatnonzero(
            (
                (times >= pd.Timestamp(split["calibration_start_utc"]))
                & (times < pd.Timestamp(split["calibration_end_exclusive_utc"]))
            ).to_numpy()
        ),
    }


def normal_segments(keys: pd.DataFrame, labels: np.ndarray, rows: np.ndarray) -> list[np.ndarray]:
    chosen = rows[labels[rows] == 0]
    frame = keys.iloc[chosen].loc[:, ["station", "year", "layer", "time"]].copy()
    frame["position"] = chosen
    frame["time"] = pd.to_datetime(frame["time"], utc=True, format="mixed")
    output: list[np.ndarray] = []
    for _, part in frame.groupby(["station", "year", "layer"], sort=False, observed=True):
        part = part.sort_values("time")
        positions = part["position"].to_numpy(dtype=np.int64)
        times = part["time"].to_numpy()
        groups = np.cumsum(np.r_[True, np.diff(times) != pd.Timedelta(minutes=10)])
        output.extend(positions[groups == group] for group in np.unique(groups))
    return output


def build_synthetic_rows(
    historical: pd.DataFrame,
    feature_values: np.ndarray,
    train_rows: np.ndarray,
    donor_events: Sequence[Any],
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    labels = historical["label"].to_numpy(dtype=np.int8)
    segments = normal_segments(historical, labels, train_rows)
    times = pd.to_datetime(historical["time"], utc=True, format="mixed")
    feature_names = list(config["features"]["numeric_sources"])
    preserved = set(config["features"]["recipient_preserved_sources"])
    replace_indices = [index for index, name in enumerate(feature_names) if name not in preserved]
    normal_rows = int(np.sum(labels[train_rows] == 0))
    real_rows = int(sum(len(event.rows) for event in donor_events))
    target_rows = min(
        int(np.floor(normal_rows * float(config["synthetic"]["normal_row_budget_fraction"]))),
        int(np.floor(real_rows * float(config["synthetic"]["synthetic_positive_weight_share_cap"]))),
    )
    rng = np.random.default_rng(int(config["model"]["random_state"]))
    donor_order = rng.permutation(len(donor_events))
    candidates: dict[tuple[str, int, int], list[np.ndarray]] = {}
    for segment in segments:
        first = int(segment[0])
        key = (
            str(historical.iloc[first]["station"]),
            int(historical.iloc[first]["layer"]),
            int(times.iloc[first].quarter),
        )
        candidates.setdefault(key, []).append(segment)
    transplants: list[np.ndarray] = []
    donor_cells: set[tuple[str, int]] = set()
    attempts = 0
    cursor = 0
    while sum(len(item) for item in transplants) < target_rows and attempts < max(100, 20 * len(donor_events)):
        event = donor_events[int(donor_order[cursor % len(donor_order)])]
        quarter = int(times.iloc[int(event.rows[0])].quarter)
        key = (event.station, event.layer, quarter)
        eligible = [segment for segment in candidates.get(key, []) if len(segment) >= len(event.rows)]
        attempts += 1
        cursor += 1
        if not eligible:
            continue
        segment = eligible[int(rng.integers(0, len(eligible)))]
        start = int(rng.integers(0, len(segment) - len(event.rows) + 1))
        recipient_rows = segment[start : start + len(event.rows)]
        donor_ordinals = historical.iloc[event.rows]["ordinal"].to_numpy(dtype=np.int64)
        recipient_ordinals = historical.iloc[recipient_rows]["ordinal"].to_numpy(dtype=np.int64)
        transplant = conditional_transplant(
            feature_values[donor_ordinals],
            feature_values[recipient_ordinals],
            replace_indices=replace_indices,
        )
        remaining = target_rows - sum(len(item) for item in transplants)
        transplants.append(transplant[:remaining])
        donor_cells.add((event.station, event.layer))
    if sum(len(item) for item in transplants) < target_rows:
        raise ContractError("same-cell same-quarter recipients cannot fill synthetic budget")
    matrix = np.vstack(transplants) if transplants else np.empty((0, len(feature_names)), dtype=np.float32)
    return matrix, {
        "target_rows": target_rows,
        "generated_rows": len(matrix),
        "donor_events_available": len(donor_events),
        "donor_station_layer_cells_used": len(donor_cells),
        "same_station_layer": True,
        "same_calendar_quarter": True,
        "replaced_feature_count": len(replace_indices),
        "preserved_feature_count": len(preserved),
    }


def q2_truth(config: dict[str, Any], q2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    record = config["immutable_inputs"]["frozen_truth_and_folds"]
    scanner = ds.dataset(ROOT / record["path"], format="parquet").scanner(
        columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
        filter=ds.field("fold") == str(config["split"]["q2_fold"]),
        use_threads=True,
    )
    frame = scanner.to_table().to_pandas().reset_index(drop=True)
    if not multiindex(frame).equals(multiindex(q2)):
        raise ContractError("Q2 truth order differs from sealed prediction order")
    return frame["label"].to_numpy(dtype=np.int8), frame["anomaly_type"].fillna("").to_numpy()


def write_manifest(verified: dict[str, dict[str, Any]]) -> None:
    files = {}
    for path in sorted(ARTIFACT_DIR.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            files[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(CONFIG_PATH),
        "module_sha256": sha256_file(MODULE_PATH),
        "runner_sha256": sha256_file(Path(__file__)),
        "immutable_inputs": verified,
        "artifacts": files,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }
    atomic_json(ARTIFACT_DIR / "manifest.json", manifest)


def check_only() -> dict[str, Any]:
    config = load_config()
    verified = verify_inputs(config)
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY",
        "verified_inputs": verified,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
    }


def execute() -> dict[str, Any]:
    config = load_config()
    verified = verify_inputs(config)
    if ARTIFACT_DIR.exists():
        raise FileExistsError(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True)
    all_keys, historical, q2 = load_surfaces(config)
    rows = split_rows(historical, config)
    labels = historical["label"].to_numpy(dtype=np.int8)
    anomaly = historical["anomaly_type"].fillna("").to_numpy()
    event_config = config["event"]
    donor_events = extract_long_events(
        historical,
        labels,
        anomaly,
        rows["train"],
        eligible_types=event_config["eligible_types"],
        minimum_rows=int(event_config["minimum_rows"]),
    )
    calibration_truth_events = extract_long_events(
        historical,
        labels,
        anomaly,
        rows["calibration"],
        eligible_types=event_config["eligible_types"],
        minimum_rows=int(event_config["minimum_rows"]),
    )
    train_support = event_support(donor_events)
    calibration_support = event_support(calibration_truth_events)
    support_config = config["support_preflight"]
    checks = {
        "train_donor_events": int(train_support["events"])
        >= int(support_config["minimum_train_donor_events"]),
        "train_donor_cells": int(train_support["station_layer_cells"])
        >= int(support_config["minimum_train_station_layer_cells"]),
        "train_cell_concentration": float(train_support["maximum_single_cell_share"])
        <= float(support_config["maximum_train_single_cell_share"]),
        "calibration_truth_events": int(calibration_support["events"])
        >= int(support_config["minimum_calibration_truth_events"]),
        "calibration_truth_cells": int(calibration_support["station_layer_cells"])
        >= int(support_config["minimum_calibration_truth_cells"]),
    }
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "pre_fit_real_event_support",
        "historical_rows_read": len(historical),
        "train_rows": len(rows["train"]),
        "calibration_rows": len(rows["calibration"]),
        "train": train_support,
        "calibration": calibration_support,
        "checks": checks,
        "support_gate_pass": all(checks.values()),
        "model_fit_count": 0,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
    }
    atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
    if not all(checks.values()):
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": "NO_GO_SUPPORT",
            "model_fit_count": 0,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "q2_truth_rows_read": 0,
        }
        atomic_json(ARTIFACT_DIR / "result.json", result)
        atomic_json(
            ARTIFACT_DIR / "qa.json",
            {"terminal_before_fit": True, "contract_pass": True, "anchor_rows_removed": 0},
        )
        write_manifest(verified)
        return result

    feature_names = list(config["features"]["numeric_sources"])
    feature_values = pd.read_parquet(
        ROOT / config["immutable_inputs"]["feature_cache"]["path"],
        columns=feature_names,
    ).to_numpy(dtype=np.float32, copy=True)
    feature_values[np.isinf(feature_values)] = np.nan
    synthetic_rows, synthetic_receipt = build_synthetic_rows(
        historical, feature_values, rows["train"], donor_events, config
    )
    real_positive_positions = np.unique(
        np.concatenate([event.rows for event in donor_events])
    )
    real_positive_ordinals = historical.iloc[real_positive_positions]["ordinal"].to_numpy(
        dtype=np.int64
    )
    normal_positions = rows["train"][labels[rows["train"]] == 0]
    normal_ordinals = historical.iloc[normal_positions]["ordinal"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(int(config["model"]["random_state"]))
    negative_count = min(
        len(normal_ordinals),
        int(np.ceil(len(real_positive_ordinals) * float(config["model"]["negative_to_real_positive_row_ratio"]))),
    )
    negative_ordinals = rng.choice(normal_ordinals, size=negative_count, replace=False)
    train_x = np.vstack(
        [feature_values[real_positive_ordinals], synthetic_rows, feature_values[negative_ordinals]]
    )
    train_y = np.concatenate(
        [
            np.ones(len(real_positive_ordinals) + len(synthetic_rows), dtype=np.int8),
            np.zeros(len(negative_ordinals), dtype=np.int8),
        ]
    )
    synthetic_weight = min(
        1.0,
        float(config["synthetic"]["synthetic_positive_weight_share_cap"])
        * len(real_positive_ordinals)
        / max(1, len(synthetic_rows)),
    )
    positive_weight_total = len(real_positive_ordinals) + synthetic_weight * len(synthetic_rows)
    train_weights = np.concatenate(
        [
            np.ones(len(real_positive_ordinals), dtype=np.float64),
            np.full(len(synthetic_rows), synthetic_weight, dtype=np.float64),
            np.full(len(negative_ordinals), positive_weight_total / len(negative_ordinals)),
        ]
    )
    model_config = config["model"]
    model = HistGradientBoostingClassifier(
        max_depth=int(model_config["max_depth"]),
        max_iter=int(model_config["max_iter"]),
        learning_rate=float(model_config["learning_rate"]),
        l2_regularization=float(model_config["l2_regularization"]),
        random_state=int(model_config["random_state"]),
    )
    model.fit(train_x, train_y, sample_weight=train_weights)

    calibration_frame = historical.iloc[rows["calibration"]].reset_index(drop=True)
    calibration_ordinals = calibration_frame["ordinal"].to_numpy(dtype=np.int64)
    calibration_scores = model.predict_proba(feature_values[calibration_ordinals])[:, 1]
    q2_ordinals = q2["ordinal"].to_numpy(dtype=np.int64)
    q2_scores = model.predict_proba(feature_values[q2_ordinals])[:, 1]
    decoder = config["decoder"]
    calibration_additions = decode_scores(
        calibration_frame,
        calibration_scores,
        threshold=float(model_config["threshold"]),
        smoothing_rows=int(decoder["score_smoothing_rows"]),
        minimum_rows=int(decoder["minimum_component_rows"]),
        bridge_rows=int(decoder["bridge_rows"]),
    )
    q2_additions = decode_scores(
        q2,
        q2_scores,
        threshold=float(model_config["threshold"]),
        smoothing_rows=int(decoder["score_smoothing_rows"]),
        minimum_rows=int(decoder["minimum_component_rows"]),
        bridge_rows=int(decoder["bridge_rows"]),
    )
    prediction_path = ARTIFACT_DIR / "sealed_predictions.npz"
    atomic_npz(
        prediction_path,
        calibration_scores=calibration_scores.astype(np.float16),
        calibration_additions=calibration_additions,
        q2_scores=q2_scores.astype(np.float16),
        q2_additions=q2_additions,
    )
    anchor_record = config["immutable_inputs"]["q2_anchor"]
    receipt = json.loads((ROOT / anchor_record["receipt"]).read_text(encoding="utf-8"))
    if ordered_key_sha(q2) != receipt["ordered_key_sha256"]:
        raise ContractError("Q2 order differs from frozen anchor receipt")
    blind_receipt = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_path": prediction_path.name,
        "prediction_bytes": prediction_path.stat().st_size,
        "prediction_sha256": sha256_file(prediction_path),
        "q2_rows": len(q2),
        "q2_ordered_key_sha256": ordered_key_sha(q2),
        "model_fit_count": 1,
        "threshold": model_config["threshold"],
        "synthetic": synthetic_receipt,
        "q2_truth_rows_read_before_receipt": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
    }
    atomic_json(ARTIFACT_DIR / "blind_prediction_receipt.json", blind_receipt)

    calibration_labels = calibration_frame["label"].to_numpy(dtype=np.int8)
    calibration_anomaly = calibration_frame["anomaly_type"].fillna("").to_numpy()
    calibration_truth_local = extract_long_events(
        calibration_frame,
        calibration_labels,
        calibration_anomaly,
        np.arange(len(calibration_frame)),
        eligible_types=event_config["eligible_types"],
        minimum_rows=int(event_config["minimum_rows"]),
    )
    calibration_proposals = extract_mask_events(
        calibration_frame,
        calibration_additions,
        minimum_rows=int(event_config["minimum_rows"]),
    )
    calibration_proposal_metrics = proposal_support_metrics(
        calibration_truth_local,
        calibration_proposals,
        calibration_labels,
        iou_threshold=float(event_config["iou_match_threshold"]),
    )
    calibration_row_metrics = binary_metrics(calibration_labels, calibration_additions)
    calibration_gate = config["calibration_gate"]
    calibration_checks = {
        "matched_proposals": int(calibration_proposal_metrics["matched_proposals"])
        >= int(calibration_gate["minimum_matched_proposals"]),
        "matched_cells": int(calibration_proposal_metrics["matched_station_layer_cells"])
        >= int(calibration_gate["minimum_matched_station_layer_cells"]),
        "real_event_recall": float(calibration_proposal_metrics["real_event_recall"])
        >= float(calibration_gate["minimum_real_event_recall"]),
        "row_precision": float(calibration_row_metrics["precision"])
        >= float(calibration_gate["minimum_row_precision"]),
    }
    calibration_record = {
        "row_metrics": calibration_row_metrics,
        "proposal_metrics": calibration_proposal_metrics,
        "checks": calibration_checks,
        "gate_pass": all(calibration_checks.values()),
    }
    if not all(calibration_checks.values()):
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": "NO_GO_CALIBRATION",
            "model_fit_count": 1,
            "training": {
                "real_positive_rows": len(real_positive_ordinals),
                "synthetic_positive_rows": len(synthetic_rows),
                "negative_rows": len(negative_ordinals),
                "synthetic_weight": synthetic_weight,
            },
            "calibration": calibration_record,
            "q2_truth_rows_read": 0,
            "q2_evaluated": False,
            "selected_arm": "ZERO_ADD_NO_OP",
        }
        atomic_json(ARTIFACT_DIR / "result.json", result)
        atomic_json(
            ARTIFACT_DIR / "qa.json",
            {
                "preflight_pass": True,
                "single_fit": True,
                "threshold_search_count": 0,
                "result_based_retry_count": 0,
                "q2_truth_opened": False,
                "anchor_rows_removed": 0,
                "contract_pass": True,
            },
        )
        write_manifest(verified)
        return result

    q2_labels, q2_anomaly = q2_truth(config, q2)
    with np.load(ROOT / anchor_record["path"], allow_pickle=False) as archive:
        q2_anchor = archive[str(anchor_record["array"])].astype(np.int8)
    if q2_anchor.shape != (len(q2),):
        raise ContractError("Q2 anchor shape changed")
    q2_metrics = evaluate_anchor_union(q2_labels, q2_anchor, q2_additions)
    q2_truth_events = extract_long_events(
        q2,
        q2_labels,
        q2_anomaly,
        np.arange(len(q2)),
        eligible_types=event_config["eligible_types"],
        minimum_rows=int(event_config["minimum_rows"]),
    )
    recovered = newly_recovered_events(q2_truth_events, q2_anchor, q2_additions)
    outer = config["q2_outer_gate"]
    bootstrap = block_bootstrap_delta(
        q2,
        q2_labels,
        q2_anchor,
        q2_additions,
        replicates=int(outer["block_bootstrap_replicates"]),
        block_days=int(outer["block_days"]),
        seed=int(outer["block_bootstrap_seed"]),
    )
    fp_relative = q2_metrics["normal_fp_per_day_relative"]
    q2_checks = {
        "delta_f1": float(q2_metrics["delta_f1"]) > float(outer["delta_f1_strictly_above"]),
        "added_precision": float(q2_metrics["added_precision"])
        > float(q2_metrics["anchor_f1_over_2"]),
        "fp_per_day": fp_relative is not None
        and float(fp_relative) <= float(outer["normal_fp_per_day_relative_lte"]),
        "newly_recovered_long_events": recovered
        >= int(outer["minimum_newly_recovered_long_events"]),
        "bootstrap_ci90": float(bootstrap["ci90_lower"])
        >= float(outer["ci90_lower_gte"]),
        "anchor_preserved": int(q2_metrics["anchor_positive_removed_rows"]) == 0,
    }
    passed = all(q2_checks.values())
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "GO_LOCAL_DIRECTIONAL" if passed else "NO_GO_Q2_OUTER",
        "model_fit_count": 1,
        "training": {
            "real_positive_rows": len(real_positive_ordinals),
            "synthetic_positive_rows": len(synthetic_rows),
            "negative_rows": len(negative_ordinals),
            "synthetic_weight": synthetic_weight,
        },
        "calibration": calibration_record,
        "q2_outer": {
            "metrics": q2_metrics,
            "newly_recovered_long_events": recovered,
            "truth_long_events": len(q2_truth_events),
            "bootstrap": bootstrap,
            "checks": q2_checks,
            "gate_pass": passed,
        },
        "selected_arm": "CONDITIONAL_DONOR_UNION" if passed else "ZERO_ADD_NO_OP",
        "q2_truth_rows_read": len(q2),
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }
    atomic_json(ARTIFACT_DIR / "result.json", result)
    atomic_json(
        ARTIFACT_DIR / "qa.json",
        {
            "preflight_pass": True,
            "calibration_gate_pass": True,
            "single_fit": True,
            "threshold_search_count": 0,
            "grid_search_count": 0,
            "result_based_retry_count": 0,
            "q2_truth_opened_after_blind_receipt": True,
            "anchor_rows_removed": q2_metrics["anchor_positive_removed_rows"],
            "contract_pass": int(q2_metrics["anchor_positive_removed_rows"]) == 0,
        },
    )
    write_manifest(verified)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = execute() if args.execute else check_only()
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
