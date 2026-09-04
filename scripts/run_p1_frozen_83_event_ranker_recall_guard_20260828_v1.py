"""Run the single bounded frozen-83 P1 event-ranker experiment."""

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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler

from p1_qc.p1_conditional_real_event_donor_20260828_v1 import (
    EventSpan,
    block_bootstrap_delta,
    evaluate_anchor_union,
    extract_long_events,
    extract_mask_events,
    newly_recovered_events,
)
from p1_qc.p1_frozen_83_event_ranker_recall_guard_20260828_v1 import (
    chronological_split,
    mask_from_proposals,
    moving_block_bootstrap_delta_f1,
    proposal_feature_matrix,
    proposal_partition_metrics,
    proposal_truth_matrix,
    remove_cross_split_truth_matches,
    select_recall_threshold,
)

EXPERIMENT_ID = "p1_frozen_83_event_ranker_recall_guard_20260828_v1"
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
MODULE_PATH = ROOT / "src" / "p1_qc" / f"{EXPERIMENT_ID}.py"
TEST_PATH = ROOT / "tests" / f"test_{EXPERIMENT_ID}.py"
QA_SCRIPT_PATH = ROOT / "scripts" / f"qa_{EXPERIMENT_ID}.py"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
KEY_COLUMNS = ("station", "year", "layer", "time")


class ContractError(RuntimeError):
    """Raised when the preregistered contract or an immutable input changes."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
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
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def ordered_key_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for row in frame.loc[:, KEY_COLUMNS].astype(str).itertuples(index=False, name=None):
        digest.update("\x1f".join(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def multiindex(frame: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(frame.loc[:, KEY_COLUMNS].astype(str))


def map_ordinals(all_keys: pd.DataFrame, requested: pd.DataFrame) -> np.ndarray:
    lookup = multiindex(all_keys)
    if not lookup.is_unique:
        raise ContractError("feature key sidecar is not unique")
    ordinals = lookup.get_indexer(multiindex(requested))
    if np.any(ordinals < 0) or len(np.unique(ordinals)) != len(ordinals):
        raise ContractError("surface keys do not bind one-to-one")
    return ordinals.astype(np.int64)


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment ID changed")
    frozen = config["frozen_generator"]
    split = config["split"]
    model = config["model"]
    threshold = config["threshold"]
    if (
        float(frozen["threshold"]) != 0.5
        or int(frozen["score_smoothing_rows"]) != 19
        or int(frozen["minimum_component_rows"]) != 19
        or int(frozen["bridge_rows"]) != 2
        or int(frozen["expected_calibration_proposals"]) != 83
        or bool(frozen["regeneration_allowed"])
    ):
        raise ContractError("frozen proposal decoder contract changed")
    if split["chronological_proposal_fractions"] != [0.4, 0.3, 0.3] or int(
        split["boundary_purge_days"]
    ) != 15:
        raise ContractError("chronological split contract changed")
    if (
        model["family"] != "sklearn LogisticRegression"
        or float(model["C"]) != 1.0
        or model["class_weight"] != "balanced"
        or int(model["fit_count"]) != 1
        or int(model["cpu_threads"]) > 2
        or bool(model["feature_selection"])
        or bool(model["hyperparameter_search"])
        or bool(model["result_based_retry"])
    ):
        raise ContractError("single fixed ranker contract changed")
    if (
        float(threshold["minimum_raw_matched_event_retention"]) != 0.9
        or int(threshold["selection_count"]) != 1
        or bool(threshold["precision_optimization"])
        or bool(threshold["grid_search"])
    ):
        raise ContractError("single recall threshold contract changed")
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


def verify_frozen_archive(config: dict[str, Any]) -> dict[str, Any]:
    frozen = config["immutable_inputs"]["frozen_proposals"]
    with np.load(ROOT / frozen["path"], allow_pickle=False) as archive:
        required = {
            str(frozen["calibration_score_array"]),
            str(frozen["calibration_addition_array"]),
            str(frozen["q2_score_array"]),
            str(frozen["q2_addition_array"]),
        }
        if set(archive.files) != required:
            raise ContractError("frozen proposal arrays changed")
        calibration_scores = archive[str(frozen["calibration_score_array"])]
        calibration_additions = archive[str(frozen["calibration_addition_array"])]
        q2_scores = archive[str(frozen["q2_score_array"])]
        q2_additions = archive[str(frozen["q2_addition_array"])]
        if calibration_scores.shape != calibration_additions.shape or q2_scores.shape != q2_additions.shape:
            raise ContractError("frozen score/addition shapes differ")
        if int(calibration_additions.sum()) != int(
            config["frozen_generator"]["expected_calibration_addition_rows"]
        ):
            raise ContractError("frozen calibration addition rows changed")
        if not set(np.unique(calibration_additions)).issubset({0, 1}) or not set(
            np.unique(q2_additions)
        ).issubset({0, 1}):
            raise ContractError("frozen additions are not binary")
        return {
            "calibration_rows": len(calibration_scores),
            "calibration_addition_rows": int(calibration_additions.sum()),
            "q2_rows": len(q2_scores),
            "q2_addition_rows": int(q2_additions.sum()),
            "archive_arrays": sorted(required),
        }


def check_only() -> dict[str, Any]:
    config = load_config()
    verified = verify_inputs(config)
    archive = verify_frozen_archive(config)
    receipt = json.loads(
        (ROOT / config["immutable_inputs"]["frozen_proposal_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if receipt["prediction_sha256"] != verified["frozen_proposals"]["sha256"]:
        raise ContractError("frozen receipt does not bind proposal archive")
    return {
        "experiment_id": EXPERIMENT_ID,
        "status": "READY_CHECK_ONLY",
        "verified_inputs": verified,
        "frozen_archive": archive,
        "artifact_directory_absent": not ARTIFACT_DIR.exists(),
        "model_fit_count": 0,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
    }


def load_surfaces(
    config: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    immutable = config["immutable_inputs"]
    all_keys = pd.read_parquet(
        ROOT / immutable["feature_key_sidecar"]["path"],
        columns=["ordinal", *KEY_COLUMNS],
    )
    if not np.array_equal(all_keys["ordinal"].to_numpy(), np.arange(len(all_keys))):
        raise ContractError("feature ordinals are not contiguous")
    surface = config["surface"]
    start = pd.Timestamp(surface["calibration_start_utc"]).tz_convert("Asia/Seoul").isoformat()
    stop = pd.Timestamp(surface["calibration_end_exclusive_utc"]).tz_convert(
        "Asia/Seoul"
    ).isoformat()
    historical = (
        ds.dataset(ROOT / immutable["training_labels"]["path"], format="parquet")
        .scanner(
            columns=[*KEY_COLUMNS, "label", "anomaly_type"],
            filter=(ds.field("time") >= start) & (ds.field("time") < stop),
            use_threads=False,
        )
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
    )
    historical["ordinal"] = map_ordinals(all_keys, historical)
    q2 = (
        ds.dataset(ROOT / immutable["frozen_truth_and_folds"]["path"], format="parquet")
        .scanner(
            columns=[*KEY_COLUMNS, "fold"],
            filter=ds.field("fold") == str(surface["q2_fold"]),
            use_threads=False,
        )
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
    )
    q2["ordinal"] = map_ordinals(all_keys, q2)
    feature_names = list(config["features"]["numeric_sources"])
    all_features = pd.read_parquet(
        ROOT / immutable["feature_cache"]["path"], columns=feature_names
    ).to_numpy(dtype=np.float32, copy=True)
    all_features[~np.isfinite(all_features)] = np.nan
    historical_features = all_features[
        historical["ordinal"].to_numpy(dtype=np.int64)
    ]
    q2_features = all_features[q2["ordinal"].to_numpy(dtype=np.int64)]
    frozen = immutable["frozen_proposals"]
    with np.load(ROOT / frozen["path"], allow_pickle=False) as archive:
        historical_scores = archive[str(frozen["calibration_score_array"])].astype(
            np.float64
        )
        historical_additions = archive[
            str(frozen["calibration_addition_array"])
        ].astype(np.int8)
        q2_scores = archive[str(frozen["q2_score_array"])].astype(np.float64)
        q2_additions = archive[str(frozen["q2_addition_array"])].astype(np.int8)
    if historical_scores.shape != (len(historical),) or q2_scores.shape != (len(q2),):
        raise ContractError("frozen arrays do not align with historical/Q2 surfaces")
    receipt = json.loads(
        (ROOT / immutable["frozen_proposal_receipt"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if ordered_key_sha(q2) != receipt["q2_ordered_key_sha256"]:
        raise ContractError("Q2 ordered key hash changed")
    return (
        historical,
        q2,
        historical_features,
        q2_features,
        (historical_scores, historical_additions, q2_scores, q2_additions),
    )


def truth_subset(
    events: Sequence[EventSpan], start: pd.Timestamp, stop: pd.Timestamp | None
) -> list[EventSpan]:
    return [
        event
        for event in events
        if event.start_time >= start and (stop is None or event.end_time <= stop)
    ]


def split_record(
    indices: np.ndarray,
    proposals: Sequence[EventSpan],
    matches: np.ndarray,
) -> dict[str, int]:
    positive = np.any(matches[indices], axis=1) if len(indices) else np.zeros(0, dtype=bool)
    cells = {
        (proposals[int(index)].station, proposals[int(index)].layer)
        for index, keep in zip(indices, positive, strict=True)
        if keep
    }
    matched_truth = int(np.sum(np.any(matches[indices], axis=0))) if len(indices) else 0
    return {
        "proposals": len(indices),
        "positive_proposals": int(positive.sum()),
        "positive_cells": len(cells),
        "matched_truth_events": matched_truth,
    }


def array_without_masks(value: dict[str, object]) -> dict[str, object]:
    return {key: item for key, item in value.items() if key not in {"raw_mask", "selected_mask"}}


def fit_ranker(
    train_x: np.ndarray,
    train_y: np.ndarray,
    config: dict[str, Any],
) -> tuple[SimpleImputer, RobustScaler, LogisticRegression]:
    model_config = config["model"]
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()
    transformed = scaler.fit_transform(imputer.fit_transform(train_x))
    model = LogisticRegression(
        C=float(model_config["C"]),
        class_weight=str(model_config["class_weight"]),
        solver=str(model_config["solver"]),
        max_iter=int(model_config["max_iter"]),
        random_state=int(model_config["random_state"]),
        n_jobs=1,
    )
    model.fit(transformed, train_y)
    return imputer, scaler, model


def rank_scores(
    values: np.ndarray,
    imputer: SimpleImputer,
    scaler: RobustScaler,
    model: LogisticRegression,
) -> np.ndarray:
    return model.predict_proba(scaler.transform(imputer.transform(values)))[:, 1]


def q2_truth(config: dict[str, Any], q2: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    record = config["immutable_inputs"]["frozen_truth_and_folds"]
    frame = (
        ds.dataset(ROOT / record["path"], format="parquet")
        .scanner(
            columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
            filter=ds.field("fold") == str(config["surface"]["q2_fold"]),
            use_threads=False,
        )
        .to_table()
        .to_pandas()
        .reset_index(drop=True)
    )
    if not multiindex(frame).equals(multiindex(q2)):
        raise ContractError("Q2 truth order differs from committed predictions")
    return frame["label"].to_numpy(dtype=np.int8), frame["anomaly_type"].fillna("").to_numpy()


def write_manifest(verified: dict[str, dict[str, Any]]) -> None:
    artifacts = {}
    for path in sorted(ARTIFACT_DIR.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts[path.name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    sources = {}
    for path in [CONFIG_PATH, MODULE_PATH, Path(__file__), TEST_PATH, QA_SCRIPT_PATH]:
        if path.is_file():
            sources[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    atomic_json(
        ARTIFACT_DIR / "manifest.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "sources": sources,
            "immutable_inputs": verified,
            "artifacts": artifacts,
            "q3_q4_rows_read": 0,
            "official_test_sample_submission_rows_read": 0,
            "submission_generated_or_uploaded": False,
        },
    )


def terminal_no_go_support(
    verified: dict[str, dict[str, Any]],
    preflight: dict[str, Any],
    failed: list[str],
    q2_anchor: np.ndarray,
) -> dict[str, Any]:
    zero = np.zeros(int(preflight["historical_rows"]), dtype=np.int8)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "NO_GO_SUPPORT",
        "model_fit_count": 0,
        "failed_checks": failed,
        "support": preflight["support"],
        "selected_arm": "EXACT_ZERO_ADD_NO_OP",
        "no_op": {
            "historical_zero_add_sha256": array_sha256(zero),
            "historical_changed_rows": 0,
            "q2_anchor_sha256": array_sha256(q2_anchor),
            "q2_candidate_sha256": array_sha256(q2_anchor.copy()),
            "q2_anchor_positive_removed_rows": 0,
            "byte_equivalent": True,
        },
        "q2_truth_rows_read": 0,
        "q2_outer_evaluated": False,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }
    atomic_json(ARTIFACT_DIR / "result.json", result)
    atomic_json(
        ARTIFACT_DIR / "qa.json",
        {
            "terminal_before_fit": True,
            "single_execution_contract": True,
            "model_fit_count": 0,
            "threshold_selection_count": 0,
            "q2_truth_opened": False,
            "anchor_rows_removed": 0,
            "contract_pass": True,
        },
    )
    write_manifest(verified)
    return result


def execute() -> dict[str, Any]:
    config = load_config()
    verified = verify_inputs(config)
    archive_check = verify_frozen_archive(config)
    if ARTIFACT_DIR.exists():
        raise FileExistsError(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True)
    (
        historical,
        q2,
        historical_features,
        q2_features,
        frozen_arrays,
    ) = load_surfaces(config)
    historical_scores, historical_additions, q2_scores, q2_additions = frozen_arrays
    historical_scores = np.asarray(historical_scores, dtype=np.float64)
    historical_additions = np.asarray(historical_additions, dtype=np.int8)
    q2_scores = np.asarray(q2_scores, dtype=np.float64)
    q2_additions = np.asarray(q2_additions, dtype=np.int8)
    event_config = config["event"]
    proposals = extract_mask_events(
        historical,
        historical_additions,
        minimum_rows=int(event_config["minimum_rows"]),
    )
    if len(proposals) != int(config["frozen_generator"]["expected_calibration_proposals"]):
        raise ContractError("frozen calibration proposal count changed")
    labels = historical["label"].to_numpy(dtype=np.int8)
    anomaly_types = historical["anomaly_type"].fillna("").to_numpy()
    truth_events = extract_long_events(
        historical,
        labels,
        anomaly_types,
        np.arange(len(historical), dtype=np.int64),
        eligible_types=event_config["eligible_types"],
        minimum_rows=int(event_config["minimum_rows"]),
    )
    matches = proposal_truth_matrix(
        proposals,
        truth_events,
        iou_threshold=float(event_config["iou_match_threshold"]),
    )
    split = chronological_split(
        proposals,
        first_fraction=0.4,
        second_cumulative_fraction=0.7,
        purge_days=int(config["split"]["boundary_purge_days"]),
    )
    split, cross_purged = remove_cross_split_truth_matches(split, matches)
    times = pd.to_datetime(historical["time"], utc=True, format="mixed")
    embargo = pd.Timedelta(days=int(config["split"]["boundary_purge_days"]))
    calibration_start = split.first_boundary + embargo
    qualification_start = split.second_boundary + embargo
    reference_rows = np.flatnonzero((times <= split.first_boundary).to_numpy())
    feature_matrix, feature_names, feature_support = proposal_feature_matrix(
        historical,
        historical_features,
        list(config["features"]["numeric_sources"]),
        proposals,
        historical_scores,
        reference_rows,
        frozen_threshold=float(config["frozen_generator"]["threshold"]),
        context_rows_each_side=int(event_config["context_rows_each_side"]),
    )
    total_positive = np.any(matches, axis=1)
    total_positive_cells = {
        (proposal.station, proposal.layer)
        for proposal, positive in zip(proposals, total_positive, strict=True)
        if positive
    }
    support = {
        "total": {
            "proposals": len(proposals),
            "positive_proposals": int(total_positive.sum()),
            "positive_cells": len(total_positive_cells),
            "matched_truth_events": int(np.sum(np.any(matches, axis=0))),
            "truth_events": len(truth_events),
        },
        "train": split_record(split.train, proposals, matches),
        "calibration": split_record(split.calibration, proposals, matches),
        "qualification": split_record(split.qualification, proposals, matches),
        "purged_proposals": len(split.purged),
        "cross_truth_group_purged_proposals": len(cross_purged),
        "normality_reference": feature_support,
    }
    support_config = config["support_preflight"]
    checks = {
        "frozen_proposal_count": len(proposals)
        == int(config["frozen_generator"]["expected_calibration_proposals"]),
        "total_positive_proposals": support["total"]["positive_proposals"]
        >= int(support_config["minimum_total_positive_proposals"]),
        "total_positive_cells": support["total"]["positive_cells"]
        >= int(support_config["minimum_total_positive_cells"]),
        "train_positive_proposals": support["train"]["positive_proposals"]
        >= int(support_config["minimum_train_positive_proposals"]),
        "calibration_positive_proposals": support["calibration"]["positive_proposals"]
        >= int(support_config["minimum_calibration_positive_proposals"]),
        "qualification_positive_proposals": support["qualification"]["positive_proposals"]
        >= int(support_config["minimum_qualification_positive_proposals"]),
        "calibration_matched_truth_events": support["calibration"]["matched_truth_events"]
        >= int(support_config["minimum_calibration_matched_truth_events"]),
        "qualification_matched_truth_events": support["qualification"]["matched_truth_events"]
        >= int(support_config["minimum_qualification_matched_truth_events"]),
        "normality_reference_coverage": float(
            feature_support["primary_reference_coverage"]
        )
        >= float(support_config["minimum_normality_reference_coverage"]),
    }
    preflight = {
        "experiment_id": EXPERIMENT_ID,
        "stage": "frozen_bank_support_before_fit",
        "historical_rows": len(historical),
        "q2_covariate_rows": len(q2),
        "archive_check": archive_check,
        "split": {
            "first_boundary": split.first_boundary.isoformat(),
            "second_boundary": split.second_boundary.isoformat(),
            "calibration_start_after_purge": calibration_start.isoformat(),
            "qualification_start_after_purge": qualification_start.isoformat(),
        },
        "feature_count": len(feature_names),
        "feature_name_sha256": array_sha256(np.asarray(feature_names, dtype="U")),
        "support": support,
        "checks": checks,
        "support_gate_pass": all(checks.values()),
        "model_fit_count": 0,
        "q2_truth_rows_read": 0,
        "q3_q4_rows_read": 0,
    }
    atomic_json(ARTIFACT_DIR / "preflight.json", preflight)
    anchor_record = config["immutable_inputs"]["q2_anchor"]
    with np.load(ROOT / anchor_record["path"], allow_pickle=False) as anchor_archive:
        q2_anchor = anchor_archive[str(anchor_record["array"])].astype(np.int8)
    if q2_anchor.shape != (len(q2),):
        raise ContractError("Q2 anchor shape changed")
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        return terminal_no_go_support(verified, preflight, failed, q2_anchor)

    train_y = total_positive[split.train].astype(np.int8)
    if len(np.unique(train_y)) != 2:
        raise ContractError("support gate allowed one-class ranker fit")
    imputer, scaler, model = fit_ranker(feature_matrix[split.train], train_y, config)
    ranker_scores = rank_scores(feature_matrix, imputer, scaler, model)
    calibration_matches = matches[split.calibration]
    threshold, threshold_record = select_recall_threshold(
        ranker_scores[split.calibration],
        calibration_matches,
        minimum_retention=float(config["threshold"]["minimum_raw_matched_event_retention"]),
    )
    calibration_selected = ranker_scores[split.calibration] >= threshold
    qualification_selected = ranker_scores[split.qualification] >= threshold

    combined_keys = pd.concat([historical, q2], ignore_index=True)
    combined_features = np.vstack([historical_features, q2_features])
    combined_scores = np.concatenate([historical_scores, q2_scores])
    q2_proposals_local = extract_mask_events(
        q2,
        q2_additions,
        minimum_rows=int(event_config["minimum_rows"]),
    )
    shifted_q2_proposals = [
        EventSpan(
            proposal.station,
            proposal.layer,
            proposal.start_time,
            proposal.end_time,
            proposal.rows + len(historical),
        )
        for proposal in q2_proposals_local
    ]
    q2_feature_matrix, q2_feature_names, q2_feature_support = proposal_feature_matrix(
        combined_keys,
        combined_features,
        list(config["features"]["numeric_sources"]),
        shifted_q2_proposals,
        combined_scores,
        reference_rows,
        frozen_threshold=float(config["frozen_generator"]["threshold"]),
        context_rows_each_side=int(event_config["context_rows_each_side"]),
    )
    if q2_feature_names != feature_names:
        raise ContractError("Q2 feature contract differs from historical feature contract")
    q2_ranker_scores = rank_scores(q2_feature_matrix, imputer, scaler, model)
    q2_selected_proposals = q2_ranker_scores >= threshold
    q2_selected_additions = mask_from_proposals(
        len(q2), q2_proposals_local, q2_selected_proposals
    )
    commitment_path = ARTIFACT_DIR / "prediction_commitment.npz"
    atomic_npz(
        commitment_path,
        historical_ranker_scores=ranker_scores.astype(np.float32),
        calibration_proposal_indices=split.calibration.astype(np.int32),
        calibration_selected=calibration_selected.astype(np.int8),
        qualification_proposal_indices=split.qualification.astype(np.int32),
        qualification_selected=qualification_selected.astype(np.int8),
        q2_ranker_scores=q2_ranker_scores.astype(np.float32),
        q2_selected_additions=q2_selected_additions.astype(np.int8),
    )
    commitment = {
        "experiment_id": EXPERIMENT_ID,
        "prediction_path": commitment_path.name,
        "prediction_bytes": commitment_path.stat().st_size,
        "prediction_sha256": sha256_file(commitment_path),
        "model_fit_count": 1,
        "threshold_selection_count": 1,
        "threshold": threshold,
        "threshold_record": threshold_record,
        "q2_rows": len(q2),
        "q2_proposals": len(q2_proposals_local),
        "q2_selected_proposals": int(q2_selected_proposals.sum()),
        "q2_selected_addition_rows": int(q2_selected_additions.sum()),
        "q2_feature_primary_reference_coverage": q2_feature_support[
            "primary_reference_coverage"
        ],
        "q2_truth_rows_read_before_commitment": 0,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
    }
    atomic_json(ARTIFACT_DIR / "prediction_commitment_receipt.json", commitment)

    calibration_stop = split.second_boundary
    qualification_stop = pd.Timestamp(config["surface"]["calibration_end_exclusive_utc"])
    calibration_truth = truth_subset(truth_events, calibration_start, calibration_stop)
    qualification_truth = truth_subset(truth_events, qualification_start, qualification_stop)
    calibration_eval_rows = np.flatnonzero(
        ((times >= calibration_start) & (times <= calibration_stop)).to_numpy()
    )
    qualification_eval_rows = np.flatnonzero(
        ((times >= qualification_start) & (times < qualification_stop)).to_numpy()
    )
    calibration_metrics = proposal_partition_metrics(
        labels,
        calibration_truth,
        [proposals[int(index)] for index in split.calibration],
        calibration_selected,
        iou_threshold=float(event_config["iou_match_threshold"]),
        evaluation_rows=calibration_eval_rows,
    )
    qualification_metrics = proposal_partition_metrics(
        labels,
        qualification_truth,
        [proposals[int(index)] for index in split.qualification],
        qualification_selected,
        iou_threshold=float(event_config["iou_match_threshold"]),
        evaluation_rows=qualification_eval_rows,
    )
    qualification_bootstrap = moving_block_bootstrap_delta_f1(
        historical.iloc[qualification_eval_rows].reset_index(drop=True),
        labels[qualification_eval_rows],
        np.asarray(qualification_metrics["raw_mask"])[qualification_eval_rows],
        np.asarray(qualification_metrics["selected_mask"])[qualification_eval_rows],
        replicates=int(config["qualification_gate"]["bootstrap_replicates"]),
        block_days=int(config["qualification_gate"]["bootstrap_block_days"]),
        seed=int(config["qualification_gate"]["bootstrap_seed"]),
    )
    gate = config["qualification_gate"]
    raw_record = qualification_metrics["raw"]
    selected_record = qualification_metrics["selected"]
    qualification_checks = {
        "matched_event_retention": float(
            qualification_metrics["matched_event_retention"]
        )
        >= float(gate["minimum_raw_matched_event_retention"]),
        "row_precision": float(selected_record["row_metrics"]["precision"])
        > float(raw_record["row_metrics"]["precision"]),
        "row_f1": float(selected_record["row_metrics"]["f1"])
        > float(raw_record["row_metrics"]["f1"]),
        "false_proposal_reduction": float(
            qualification_metrics["false_proposal_reduction"]
        )
        >= float(gate["minimum_false_proposal_reduction"]),
        "matched_cells": int(selected_record["matched_cells"])
        >= int(raw_record["matched_cells"]),
        "eligible_truth_row_recall": float(
            qualification_metrics["eligible_truth_row_recall_relative"]
        )
        >= float(gate["minimum_eligible_truth_row_recall_relative"]),
        "bootstrap_probability": float(
            qualification_bootstrap["probability_positive"]
        )
        >= float(gate["bootstrap_probability_positive_gte"]),
        "bootstrap_ci90": float(qualification_bootstrap["ci90_lower"])
        >= float(gate["bootstrap_ci90_lower_gte"]),
    }
    historical_gate_pass = all(qualification_checks.values())
    historical_record = {
        "calibration": array_without_masks(calibration_metrics),
        "qualification": {
            **array_without_masks(qualification_metrics),
            "bootstrap": qualification_bootstrap,
            "checks": qualification_checks,
            "gate_pass": historical_gate_pass,
        },
    }
    if not historical_gate_pass:
        zero = np.zeros(len(historical), dtype=np.int8)
        result = {
            "experiment_id": EXPERIMENT_ID,
            "status": "NO_GO_QUALIFICATION",
            "model_fit_count": 1,
            "threshold_selection_count": 1,
            "support": support,
            "threshold": threshold_record | {"value": threshold},
            "historical": historical_record,
            "selected_arm": "EXACT_ZERO_ADD_NO_OP",
            "no_op": {
                "historical_zero_add_sha256": array_sha256(zero),
                "historical_changed_rows": 0,
                "q2_anchor_sha256": array_sha256(q2_anchor),
                "q2_candidate_sha256": array_sha256(q2_anchor.copy()),
                "q2_anchor_positive_removed_rows": 0,
                "byte_equivalent": True,
            },
            "q2_truth_rows_read": 0,
            "q2_outer_evaluated": False,
            "q3_q4_rows_read": 0,
            "official_test_sample_submission_rows_read": 0,
            "submission_generated_or_uploaded": False,
        }
        atomic_json(ARTIFACT_DIR / "result.json", result)
        atomic_json(
            ARTIFACT_DIR / "qa.json",
            {
                "preflight_pass": True,
                "single_fit": True,
                "model_fit_count": 1,
                "threshold_selection_count": 1,
                "hyperparameter_search_count": 0,
                "result_based_retry_count": 0,
                "q2_truth_opened": False,
                "anchor_rows_removed": 0,
                "exact_no_op": True,
                "contract_pass": True,
            },
        )
        write_manifest(verified)
        return result

    q2_labels, q2_anomaly_types = q2_truth(config, q2)
    q2_truth_events = extract_long_events(
        q2,
        q2_labels,
        q2_anomaly_types,
        np.arange(len(q2), dtype=np.int64),
        eligible_types=event_config["eligible_types"],
        minimum_rows=int(event_config["minimum_rows"]),
    )
    q2_partition = proposal_partition_metrics(
        q2_labels,
        q2_truth_events,
        q2_proposals_local,
        q2_selected_proposals,
        iou_threshold=float(event_config["iou_match_threshold"]),
    )
    raw_union = evaluate_anchor_union(q2_labels, q2_anchor, q2_additions)
    selected_union = evaluate_anchor_union(q2_labels, q2_anchor, q2_selected_additions)
    recovered = newly_recovered_events(
        q2_truth_events, q2_anchor, q2_selected_additions
    )
    q2_bootstrap = block_bootstrap_delta(
        q2,
        q2_labels,
        q2_anchor,
        q2_selected_additions,
        replicates=int(config["q2_outer_gate"]["block_bootstrap_replicates"]),
        block_days=int(config["q2_outer_gate"]["block_days"]),
        seed=int(config["q2_outer_gate"]["block_bootstrap_seed"]),
    )
    raw_candidate_fp_day = float(raw_union["candidate_fp_per_day"])
    selected_candidate_fp_day = float(selected_union["candidate_fp_per_day"])
    selected_to_raw_fp = (
        selected_candidate_fp_day / raw_candidate_fp_day
        if raw_candidate_fp_day > 0
        else (1.0 if selected_candidate_fp_day == 0 else None)
    )
    outer = config["q2_outer_gate"]
    raw_q2_record = q2_partition["raw"]
    selected_q2_record = q2_partition["selected"]
    q2_checks = {
        "matched_event_retention": float(q2_partition["matched_event_retention"])
        >= float(outer["minimum_raw_matched_event_retention"]),
        "delta_f1": float(selected_union["delta_f1"])
        > float(outer["delta_f1_strictly_above"]),
        "added_precision": float(selected_union["added_precision"])
        > float(selected_union["anchor_f1_over_2"]),
        "selected_fp_per_day_relative_to_raw": selected_to_raw_fp is not None
        and float(selected_to_raw_fp)
        <= float(outer["selected_fp_per_day_relative_to_raw_bank_lte"]),
        "newly_recovered_long_events": recovered
        >= int(outer["minimum_newly_recovered_long_events"]),
        "matched_cells": int(selected_q2_record["matched_cells"])
        >= int(raw_q2_record["matched_cells"]),
        "bootstrap_ci90": float(q2_bootstrap["ci90_lower"])
        >= float(outer["ci90_lower_gte"]),
        "anchor_preserved": int(selected_union["anchor_positive_removed_rows"]) == 0,
    }
    q2_pass = all(q2_checks.values())
    selected_candidate = np.maximum(q2_anchor, q2_selected_additions).astype(np.int8)
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "GO_LOCAL_DIRECTIONAL" if q2_pass else "NO_GO_Q2_OUTER",
        "model_fit_count": 1,
        "threshold_selection_count": 1,
        "support": support,
        "threshold": threshold_record | {"value": threshold},
        "historical": historical_record,
        "q2_outer": {
            "proposal_metrics": array_without_masks(q2_partition),
            "raw_anchor_union": raw_union,
            "selected_anchor_union": selected_union,
            "selected_fp_per_day_relative_to_raw_bank": selected_to_raw_fp,
            "newly_recovered_long_events": recovered,
            "truth_long_events": len(q2_truth_events),
            "bootstrap": q2_bootstrap,
            "checks": q2_checks,
            "gate_pass": q2_pass,
        },
        "selected_arm": "RANKED_PROPOSAL_ANCHOR_UNION" if q2_pass else "EXACT_ANCHOR_NO_OP",
        "no_op": {
            "q2_anchor_sha256": array_sha256(q2_anchor),
            "q2_selected_candidate_sha256": array_sha256(selected_candidate),
            "q2_anchor_positive_removed_rows": int(
                selected_union["anchor_positive_removed_rows"]
            ),
            "failure_fallback_candidate_sha256": array_sha256(q2_anchor.copy()),
            "failure_fallback_byte_equivalent": True,
        },
        "q2_truth_rows_read": len(q2),
        "q2_outer_evaluated": True,
        "q3_q4_rows_read": 0,
        "official_test_sample_submission_rows_read": 0,
        "submission_generated_or_uploaded": False,
    }
    atomic_json(ARTIFACT_DIR / "result.json", result)
    atomic_json(
        ARTIFACT_DIR / "qa.json",
        {
            "preflight_pass": True,
            "historical_gate_pass": True,
            "single_fit": True,
            "model_fit_count": 1,
            "threshold_selection_count": 1,
            "hyperparameter_search_count": 0,
            "result_based_retry_count": 0,
            "q2_truth_opened_after_prediction_commitment": True,
            "anchor_rows_removed": selected_union["anchor_positive_removed_rows"],
            "failure_fallback_exact_no_op": True,
            "contract_pass": int(selected_union["anchor_positive_removed_rows"]) == 0,
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
