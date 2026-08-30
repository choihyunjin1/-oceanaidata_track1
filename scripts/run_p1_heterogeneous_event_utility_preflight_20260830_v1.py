"""Run the sealed zero-fit heterogeneous-event utility support preflight.

Only immutable, registered P1 train/OOF artifacts are eligible inputs.  The
runner never opens raw training data or any official interface file, never
fits a model, and can persist only one aggregate JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_heterogeneous_event_utility_preflight_20260830_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
CONFIG_SHA256 = "e1f1f350295010c7b54ef1d181e94311c1c52f2b3af69e5671c89112a96db622"


class ContractError(RuntimeError):
    """Raised when the immutable preregistration itself is invalid."""


class ProvenanceError(RuntimeError):
    """Raised when a registered OOF artifact cannot be authenticated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _normal_path(value: str) -> str:
    return value.replace("\\", "/").lstrip("./").lower()


def _path_matches(value: str, registered: str) -> bool:
    observed = _normal_path(value)
    expected = _normal_path(registered)
    return observed == expected or observed == Path(expected).name


def _resolve_registered(
    root: Path,
    value: str,
    forbidden_tokens: Sequence[str],
) -> Path:
    relative = _normal_path(value)
    if any(token.lower() in relative for token in forbidden_tokens):
        raise ProvenanceError("registered input contains a forbidden path token")
    try:
        candidate = (root / value).resolve(strict=True)
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError("registered input is missing or inaccessible") from error
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ProvenanceError("registered input escaped the repository root")
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProvenanceError("registered receipt is unreadable") from error
    if not isinstance(value, dict):
        raise ProvenanceError("registered receipt is not a JSON object")
    return value


def _dict_has_digest(record: Mapping[str, Any], expected: str) -> bool:
    return any(
        "sha" in str(key).lower() and str(value).lower() == expected.lower()
        for key, value in record.items()
    )


def _dict_has_bytes(record: Mapping[str, Any], expected: int) -> bool:
    values = [
        value
        for key, value in record.items()
        if "byte" in str(key).lower() and isinstance(value, (int, float))
    ]
    return bool(values) and any(int(value) == expected for value in values)


def _receipt_binds_artifact(
    receipt: Mapping[str, Any],
    artifact_path: str,
    artifact_sha256: str,
    artifact_bytes: int,
) -> bool:
    """Accept both manifest mappings and path/sha/bytes sibling receipts."""

    def walk(node: Any) -> bool:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if _path_matches(str(key), artifact_path) and isinstance(value, Mapping):
                    if _dict_has_digest(value, artifact_sha256) and _dict_has_bytes(
                        value, artifact_bytes
                    ):
                        return True
            path_is_bound = any(
                isinstance(value, str) and _path_matches(value, artifact_path)
                for key, value in node.items()
                if "path" in str(key).lower() or str(key).lower() == "output"
            )
            if (
                path_is_bound
                and _dict_has_digest(node, artifact_sha256)
                and _dict_has_bytes(node, artifact_bytes)
            ):
                return True
            return any(walk(value) for value in node.values())
        if isinstance(node, list):
            return any(walk(value) for value in node)
        return False

    return walk(receipt)


def _verify_file_record(
    root: Path,
    record: Mapping[str, Any],
    forbidden_tokens: Sequence[str],
    cache: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any]]:
    relative = str(record["path"])
    path = _resolve_registered(root, relative, forbidden_tokens)
    key = str(path)
    if key not in cache:
        try:
            cache[key] = {
                "path": _normal_path(relative),
                "bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            }
        except OSError as error:
            raise ProvenanceError(
                "registered input disappeared during authentication"
            ) from error
    observed = cache[key]
    if (
        int(observed["bytes"]) != int(record["bytes"])
        or observed["sha256"] != str(record["sha256"])
    ):
        raise ProvenanceError("registered artifact bytes or sha256 changed")
    return path, observed


def _verify_artifact_pair(
    root: Path,
    pair: Mapping[str, Any],
    forbidden_tokens: Sequence[str],
    file_cache: dict[str, dict[str, Any]],
    receipt_cache: dict[str, dict[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    artifact_path, artifact_observed = _verify_file_record(
        root, pair["artifact"], forbidden_tokens, file_cache
    )
    receipt_path, receipt_observed = _verify_file_record(
        root, pair["receipt"], forbidden_tokens, file_cache
    )
    receipt_key = str(receipt_path)
    if receipt_key not in receipt_cache:
        receipt_cache[receipt_key] = _load_json(receipt_path)
    receipt = receipt_cache[receipt_key]
    artifact = pair["artifact"]
    if not _receipt_binds_artifact(
        receipt,
        str(artifact["path"]),
        str(artifact["sha256"]),
        int(artifact["bytes"]),
    ):
        raise ProvenanceError("registered receipt does not bind its artifact")
    observed = {
        "artifact": artifact_observed,
        "receipt": receipt_observed,
    }
    return artifact_path, receipt, observed


def _ordered_key_sha(frame: pd.DataFrame, key_columns: Sequence[str]) -> str:
    """Replay the key digest used by the sealed MS-TCN receipts."""
    digest = hashlib.sha256()
    for column in key_columns:
        digest.update(column.encode("ascii") + b"\0")
        for value in frame[column].tolist():
            raw = str(value).encode("utf-8")
            digest.update(len(raw).to_bytes(4, "little"))
            digest.update(raw)
    return digest.hexdigest()


def _validate_neural_receipt(
    receipt: Mapping[str, Any],
    record: Mapping[str, Any],
    fold: str,
    row_count: int,
    ordered_key_sha256: str,
) -> None:
    """Authenticate blind-fold identity before any neural array is opened."""
    if (
        receipt.get("experiment_id") != record["expected_experiment_id"]
        or receipt.get("fold") != fold
        or int(receipt.get("holdout_rows", -1)) != row_count
        or receipt.get("ordered_holdout_key_sha256") != ordered_key_sha256
        or int(
            receipt.get("same_fold_holdout_truth_columns_opened_before_receipt", -1)
        )
        != 0
    ):
        raise ProvenanceError("blind neural OOF receipt contract changed")


def _assert_binary(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or not np.isin(array, [0, 1]).all():
        raise ProvenanceError(f"{name} is not a one-dimensional binary vector")
    return array.astype(np.int8, copy=False)


def _bind_frame(
    truth: pd.DataFrame,
    other: pd.DataFrame,
    key_columns: Sequence[str],
    value_columns: Sequence[str],
    source_name: str,
) -> pd.DataFrame:
    required = [*key_columns, "fold", *value_columns]
    if set(required) - set(other.columns):
        raise ProvenanceError(f"{source_name} is missing required columns")
    if other.duplicated(list(key_columns)).any():
        raise ProvenanceError(f"{source_name} has duplicate OOF keys")
    right = other.loc[:, required].rename(columns={"fold": "__source_fold"})
    left = truth.assign(__ordinal=np.arange(len(truth), dtype=np.int64))
    merged = left.merge(
        right,
        on=list(key_columns),
        how="left",
        sort=False,
        validate="one_to_one",
        indicator=True,
    )
    if len(merged) != len(truth) or not merged["_merge"].eq("both").all():
        raise ProvenanceError(f"{source_name} OOF keys do not exactly cover truth")
    if not merged["fold"].astype(str).equals(merged["__source_fold"].astype(str)):
        raise ProvenanceError(f"{source_name} fold binding differs from truth")
    merged = merged.sort_values("__ordinal", kind="stable").reset_index(drop=True)
    return merged.drop(columns=["__ordinal", "__source_fold", "_merge"])


def _select_npz_candidate(
    archive: Mapping[str, np.ndarray], selector: Mapping[str, Any]
) -> np.ndarray:
    kind = str(selector["kind"])
    candidate = np.asarray(archive[str(selector["candidate_array"])])
    if kind == "direct":
        selected = candidate
    elif kind == "epoch_indexed":
        epochs = np.asarray(archive[str(selector["epochs_array"])])
        positions = np.flatnonzero(epochs == int(selector["epoch"]))
        if len(positions) != 1:
            raise ProvenanceError("epoch-indexed candidate selector is not unique")
        selected = candidate[int(positions[0])]
    elif kind == "width_epoch_threshold_grid":
        widths = np.asarray(archive[str(selector["widths_array"])])
        epochs = np.asarray(archive[str(selector["epochs_array"])])
        thresholds = np.asarray(archive[str(selector["thresholds_array"])])
        capacity = np.flatnonzero(
            (widths == int(selector["width"]))
            & (epochs == int(selector["epoch"]))
        )
        threshold = np.flatnonzero(
            np.isclose(thresholds, float(selector["threshold"]), rtol=0.0, atol=1e-12)
        )
        if len(capacity) != 1 or len(threshold) != 1:
            raise ProvenanceError("grid candidate selector is not unique")
        selected = candidate[int(capacity[0]), int(threshold[0])]
    else:
        raise ContractError(f"unsupported frozen candidate selector: {kind}")
    return _assert_binary(np.asarray(selected), "frozen neural candidate")


def _load_npz_arrays(
    path: Path,
    selector: Mapping[str, Any],
    cache: dict[str, dict[str, np.ndarray]],
) -> Mapping[str, np.ndarray]:
    key = str(path)
    requested = {str(selector["candidate_array"])}
    for field in ("widths_array", "epochs_array", "thresholds_array"):
        if field in selector:
            requested.add(str(selector[field]))
    stored = cache.setdefault(key, {})
    missing = requested - set(stored)
    if missing:
        with np.load(path, allow_pickle=False) as archive:
            absent = missing - set(archive.files)
            if absent:
                raise ProvenanceError("registered NPZ is missing selector arrays")
            for name in missing:
                stored[name] = archive[name].copy()
    return stored


def _load_inputs(
    root: Path, config: Mapping[str, Any]
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    key_columns = [str(value) for value in config["key_columns"]]
    fold_order = [str(value) for value in config["fold_order"]]
    forbidden = [str(value).lower() for value in config["forbidden_path_tokens"]]
    file_cache: dict[str, dict[str, Any]] = {}
    receipt_cache: dict[str, dict[str, Any]] = {}
    npz_cache: dict[str, dict[str, np.ndarray]] = {}
    inputs = config["immutable_inputs"]

    truth_record = inputs["truth_oof"]
    truth_path, _, truth_observed = _verify_artifact_pair(
        root, truth_record, forbidden, file_cache, receipt_cache
    )
    required_truth = [str(value) for value in truth_record["required_columns"]]
    truth = pd.read_parquet(truth_path, columns=required_truth)
    if truth.duplicated(key_columns).any() or truth.empty:
        raise ProvenanceError("sealed OOF truth is empty or has duplicate keys")
    truth["label"] = _assert_binary(truth["label"].to_numpy(), "OOF truth label")
    if set(truth["fold"].astype(str).unique()) != set(fold_order):
        raise ProvenanceError("sealed OOF truth fold inventory changed")

    anchor_record = inputs["anchor_oof"]
    anchor_path, anchor_receipt, anchor_observed = _verify_artifact_pair(
        root, anchor_record, forbidden, file_cache, receipt_cache
    )
    if (
        anchor_receipt.get("status") != anchor_record["required_receipt_status"]
        or int(anchor_receipt.get("truth_columns_read", -1))
        != int(anchor_record["required_truth_columns_read"])
    ):
        raise ProvenanceError("Router anchor receipt lost its label-free status")
    anchor_column = str(anchor_record["prediction_column"])
    anchor = pd.read_parquet(
        anchor_path, columns=[*key_columns, "fold", anchor_column]
    )
    frame = _bind_frame(truth, anchor, key_columns, [anchor_column], "Router anchor")
    frame = frame.rename(columns={anchor_column: "anchor"})
    frame["anchor"] = _assert_binary(frame["anchor"].to_numpy(), "Router anchor")

    tabular_sources = [
        source
        for source in config["proposal_sources"]
        if source["kind"] == "parquet_column"
    ]
    tabular_record = inputs["tabular_oof"]
    tabular_path, _, tabular_observed = _verify_artifact_pair(
        root, tabular_record, forbidden, file_cache, receipt_cache
    )
    tabular_columns = [str(source["column"]) for source in tabular_sources]
    tabular = pd.read_parquet(
        tabular_path, columns=[*key_columns, "fold", *tabular_columns]
    )
    frame = _bind_frame(
        frame, tabular, key_columns, tabular_columns, "matched-budget family OOF"
    )
    source_names: list[str] = []
    for source in tabular_sources:
        name = str(source["name"])
        column = str(source["column"])
        frame[name] = _assert_binary(frame[column].to_numpy(), name)
        frame = frame.drop(columns=column)
        source_names.append(name)

    key_hashes: dict[str, str] = {}
    for fold in fold_order:
        selected = frame["fold"].astype(str).eq(fold)
        key_hashes[fold] = _ordered_key_sha(frame.loc[selected], key_columns)

    neural_observed: dict[str, Any] = {}
    for source in config["proposal_sources"]:
        if source["kind"] != "fold_npz_candidate":
            continue
        name = str(source["name"])
        values = np.zeros(len(frame), dtype=np.int8)
        source_observed: dict[str, Any] = {}
        for fold in fold_order:
            record = source["folds"][fold]
            artifact_path, receipt, observed = _verify_artifact_pair(
                root, record, forbidden, file_cache, receipt_cache
            )
            selected = frame["fold"].astype(str).eq(fold).to_numpy()
            row_count = int(selected.sum())
            _validate_neural_receipt(
                receipt, record, fold, row_count, key_hashes[fold]
            )
            archive = _load_npz_arrays(artifact_path, record["selector"], npz_cache)
            candidate = _select_npz_candidate(archive, record["selector"])
            if candidate.shape != (row_count,):
                raise ProvenanceError("neural candidate row count differs from OOF fold")
            values[selected] = candidate
            source_observed[fold] = observed
        frame[name] = values
        source_names.append(name)
        neural_observed[name] = source_observed

    if len(source_names) != len(set(source_names)):
        raise ContractError("proposal source names are not unique")
    provenance = {
        "complete": True,
        "truth_oof": truth_observed,
        "anchor_oof": anchor_observed,
        "tabular_oof": tabular_observed,
        "neural_oof": neural_observed,
        "ordered_key_sha256_by_fold": key_hashes,
        "registered_source_names": source_names,
        "registered_file_count": len(file_cache),
    }
    return frame, source_names, provenance


def _binary_f1_fraction(
    truth: np.ndarray, prediction: np.ndarray
) -> tuple[float, int, int]:
    truth = _assert_binary(truth, "metric truth")
    prediction = _assert_binary(prediction, "metric prediction")
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    denominator = 2 * tp + fp + fn
    numerator = 2 * tp
    value = float(numerator / denominator) if denominator else 0.0
    return value, numerator, denominator


def _bank_digest(
    frame: pd.DataFrame,
    proposal_mask: np.ndarray,
    source_names: Sequence[str],
    key_columns: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    columns = [*key_columns, *source_names]
    for row in frame.loc[proposal_mask, columns].itertuples(index=False, name=None):
        digest.update("\x1f".join(map(str, row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_events(
    frame: pd.DataFrame,
    source_names: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    key_columns = [str(value) for value in config["key_columns"]]
    sources = frame.loc[:, source_names].to_numpy(dtype=np.int8)
    anchor = frame["anchor"].to_numpy(dtype=np.int8)
    proposal_mask = (anchor == 0) & np.any(sources == 1, axis=1)
    union = np.maximum(anchor, proposal_mask.astype(np.int8))
    removed = int(np.sum((anchor == 1) & (union == 0)))
    if removed != 0:
        raise ContractError("anchor-preserving union removed incumbent positives")

    proposed = frame.loc[
        proposal_mask,
        [*key_columns, "fold", "label", *source_names],
    ].copy()
    proposed["__time"] = pd.to_datetime(proposed["time"], errors="raise", utc=True)
    group_columns = [str(value) for value in config["proposal_contract"]["event_group"]]
    proposed = proposed.sort_values(
        [*group_columns, "__time"], kind="stable"
    ).reset_index(drop=True)
    cadence = pd.Timedelta(
        minutes=int(config["proposal_contract"]["contiguous_cadence_minutes"])
    )
    events: list[dict[str, Any]] = []
    if not proposed.empty:
        group_changed = proposed.loc[:, group_columns].ne(
            proposed.loc[:, group_columns].shift()
        ).any(axis=1)
        time_gap = proposed["__time"].diff().ne(cadence)
        event_ids = (group_changed | time_gap).cumsum()
        for _, event in proposed.groupby(event_ids, sort=False, observed=True):
            labels = event["label"].to_numpy(dtype=np.int8)
            source_bits = event.loc[:, source_names].to_numpy(dtype=np.int8)
            events.append(
                {
                    "fold": str(event["fold"].iloc[0]),
                    "cell": (
                        f"{event['station'].iloc[0]}|L{int(event['layer'].iloc[0])}"
                    ),
                    "rows": int(len(event)),
                    "tp": int(np.sum(labels == 1)),
                    "fp": int(np.sum(labels == 0)),
                    "active_source_count": int(np.any(source_bits == 1, axis=0).sum()),
                }
            )
    source_rows = {
        name: int(np.sum(proposal_mask & frame[name].to_numpy(dtype=np.int8).astype(bool)))
        for name in source_names
    }
    bank = {
        "proposal_rows": int(proposal_mask.sum()),
        "proposal_events": len(events),
        "source_proposal_rows": source_rows,
        "anchor_positive_rows": int(anchor.sum()),
        "anchor_positive_removed_rows": removed,
        "candidate_materialized_or_persisted": False,
        "ordered_bank_sha256": _bank_digest(
            frame, proposal_mask, source_names, key_columns
        ),
    }
    return events, bank


def _support_summary(
    frame: pd.DataFrame,
    events: Sequence[Mapping[str, Any]],
    folds: Iterable[str],
) -> dict[str, Any]:
    selected_folds = {str(value) for value in folds}
    row_mask = frame["fold"].astype(str).isin(selected_folds).to_numpy()
    anchor_f1, anchor_f1_numerator, anchor_f1_denominator = _binary_f1_fraction(
        frame.loc[row_mask, "label"].to_numpy(dtype=np.int8),
        frame.loc[row_mask, "anchor"].to_numpy(dtype=np.int8),
    )
    requirement = anchor_f1 / 2.0
    selected_events = [event for event in events if event["fold"] in selected_folds]
    proposal_rows = int(sum(int(event["rows"]) for event in selected_events))
    added_tp = int(sum(int(event["tp"]) for event in selected_events))
    added_fp = int(sum(int(event["fp"]) for event in selected_events))
    precision = float(added_tp / proposal_rows) if proposal_rows else None
    utility_positive: list[Mapping[str, Any]] = []
    utility_numerators: list[int] = []
    utility_denominator = anchor_f1_denominator or 1
    for event in selected_events:
        if anchor_f1_denominator:
            utility_numerator = (
                (2 * anchor_f1_denominator - anchor_f1_numerator)
                * int(event["tp"])
                - anchor_f1_numerator * int(event["fp"])
            )
        else:
            utility_numerator = 2 * int(event["tp"])
        utility_numerators.append(utility_numerator)
        if utility_numerator > 0:
            utility_positive.append(event)
    cell_counts = Counter(str(event["cell"]) for event in utility_positive)
    positive_count = len(utility_positive)
    maximum_count = max(cell_counts.values()) if cell_counts else 0
    maximum_share = (
        float(maximum_count / positive_count) if positive_count else None
    )
    positive_precisions = [
        float(int(event["tp"]) / int(event["rows"]))
        for event in utility_positive
    ]
    return {
        "folds": sorted(selected_folds),
        "rows": int(row_mask.sum()),
        "anchor_f1": anchor_f1,
        "anchor_f1_numerator": anchor_f1_numerator,
        "anchor_f1_denominator": anchor_f1_denominator,
        "proposal_precision_requirement_f1_over_2": requirement,
        "proposal_events": len(selected_events),
        "proposal_rows": proposal_rows,
        "proposal_added_tp": added_tp,
        "proposal_added_fp": added_fp,
        "proposal_precision": precision,
        "overall_proposal_precision_strictly_above_requirement": (
            sum(utility_numerators) > 0 if precision is not None else False
        ),
        "utility_positive_events": positive_count,
        "utility_positive_station_layer_cells": len(cell_counts),
        "maximum_single_station_layer_utility_positive_event_count": maximum_count,
        "maximum_single_station_layer_utility_positive_share": maximum_share,
        "minimum_utility_positive_event_precision": (
            min(positive_precisions) if positive_precisions else None
        ),
        "utility_numerator": int(sum(utility_numerators)),
        "utility_denominator": utility_denominator,
        "utility_sum": float(sum(utility_numerators) / utility_denominator),
    }


def evaluate_support(
    frame: pd.DataFrame,
    source_names: Sequence[str],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate only preregistered support gates; this function cannot fit."""
    events, bank = _build_events(frame, source_names, config)
    gates = config["support_gates"]
    provenance_complete = (provenance or {}).get("complete", False) is True
    prefix_receipts: dict[str, Any] = {}
    failure_reasons: list[str] = []
    for prefix in config["prefixes"]:
        name = str(prefix["name"])
        fit = _support_summary(frame, events, prefix["fit_folds"])
        calibration = _support_summary(frame, events, prefix["calibration_folds"])
        share_limit = Fraction(
            str(gates["maximum_fit_single_station_layer_utility_positive_share"])
        )
        checks = {
            "fit_utility_positive_events_gte_10": int(
                fit["utility_positive_events"]
            )
            >= int(gates["minimum_fit_utility_positive_events"]),
            "calibration_utility_positive_events_gte_4": int(
                calibration["utility_positive_events"]
            )
            >= int(gates["minimum_calibration_utility_positive_events"]),
            "fit_utility_positive_station_layer_cells_gte_3": int(
                fit["utility_positive_station_layer_cells"]
            )
            >= int(gates["minimum_fit_utility_positive_station_layer_cells"]),
            "fit_maximum_single_station_layer_share_lte_0_70": (
                int(fit["utility_positive_events"]) > 0
                and share_limit.denominator
                * int(
                    fit[
                        "maximum_single_station_layer_utility_positive_event_count"
                    ]
                )
                <= share_limit.numerator * int(fit["utility_positive_events"])
            ),
            "provenance_complete": provenance_complete,
        }
        for check, passed in checks.items():
            if not passed:
                failure_reasons.append(f"{name}:{check}")
        prefix_receipts[name] = {
            "fit": fit,
            "calibration": calibration,
            "checks": checks,
            "pass": all(checks.values()),
        }
    passed = bool(prefix_receipts) and all(
        receipt["pass"] for receipt in prefix_receipts.values()
    )
    return {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "status": (
            "PASS_ZERO_FIT_SUPPORT_PREFLIGHT_RESEARCH_ONLY"
            if passed
            else "NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT"
        ),
        "historical_surface": config["historical_surface"],
        "proposal_bank": bank,
        "prefixes": prefix_receipts,
        "failure_reasons": failure_reasons,
        "provenance": dict(provenance or {"complete": False}),
        "execution_audit": {
            "model_fit_count": 0,
            "threshold_search_count": 0,
            "prediction_materialization_count": 0,
            "prediction_csv_count": 0,
            "official_interface_rows_read": 0,
            "raw_training_rows_read": 0,
            "raw_temp_rows_read": 0,
            "auxiliary_psal_depth_rows_read": 0,
            "target_positive_rows_removed": 0,
            "anchor_positive_removed_rows": int(
                bank["anchor_positive_removed_rows"]
            ),
            "upload_count": 0,
        },
        "outlier_policy": config["outlier_policy"],
    }


def _provenance_no_go(
    config: Mapping[str, Any], error: ProvenanceError
) -> dict[str, Any]:
    return {
        "schema_version": config["schema_version"],
        "experiment_id": config["experiment_id"],
        "status": "NO_GO_ZERO_FIT_SUPPORT_PREFLIGHT",
        "historical_surface": config["historical_surface"],
        "proposal_bank": None,
        "prefixes": {},
        "failure_reasons": ["provenance_complete"],
        "provenance": {"complete": False, "error": str(error)},
        "execution_audit": {
            "model_fit_count": 0,
            "threshold_search_count": 0,
            "prediction_materialization_count": 0,
            "prediction_csv_count": 0,
            "official_interface_rows_read": 0,
            "raw_training_rows_read": 0,
            "raw_temp_rows_read": 0,
            "auxiliary_psal_depth_rows_read": 0,
            "target_positive_rows_removed": 0,
            "anchor_positive_removed_rows": 0,
            "upload_count": 0,
        },
        "outlier_policy": config["outlier_policy"],
    }


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise ContractError("experiment identity changed")
    execution = config["execution_contract"]
    if (
        int(execution["model_fit_count"]) != 0
        or int(execution["threshold_search_count"]) != 0
        or int(execution["prediction_materialization_count"]) != 0
    ):
        raise ContractError("zero-fit execution contract changed")
    gates = config["support_gates"]
    expected_gates = {
        "minimum_fit_utility_positive_events": 10,
        "minimum_calibration_utility_positive_events": 4,
        "minimum_fit_utility_positive_station_layer_cells": 3,
        "maximum_fit_single_station_layer_utility_positive_share": 0.7,
    }
    if any(float(gates[key]) != value for key, value in expected_gates.items()):
        raise ContractError("support gates changed")
    if (
        gates["require_complete_oof_provenance"] is not True
        or gates["apply_to_every_prefix"] is not True
        or gates["kill_on_any_failure"] is not True
    ):
        raise ContractError("support-gate terminality changed")
    expected_prefixes = [
        ("q2_to_q3", ("2025_q2",), ("2025_q3",)),
        ("q2_q3_to_q4", ("2025_q2", "2025_q3"), ("2025_q4",)),
    ]
    observed_prefixes = [
        (
            prefix["name"],
            tuple(prefix["fit_folds"]),
            tuple(prefix["calibration_folds"]),
        )
        for prefix in config["prefixes"]
    ]
    if observed_prefixes != expected_prefixes:
        raise ContractError("chronological prefix contract changed")
    source_names = {source["name"] for source in config["proposal_sources"]}
    if source_names != {
        "offline_xgboost_default",
        "round_a_causal_default",
        "round_b_event_day_default",
        "mstcn_e125",
        "mstcn_e150",
    }:
        raise ContractError("frozen heterogeneous source inventory changed")
    proposal = config["proposal_contract"]
    if (
        proposal["bank_rule"]
        != "current_router_prediction == 0 AND any frozen source prediction == 1"
        or proposal["anchor_negative_only"] is not True
        or proposal["anchor_union_immutable"] is not True
        or proposal["event_group"] != ["fold", "station", "year", "layer"]
        or int(proposal["contiguous_cadence_minutes"]) != 10
        or proposal["utility"]
        != "(2 - anchor_f1) * added_tp - anchor_f1 * added_fp"
        or float(proposal["utility_positive_if_strictly_above"]) != 0.0
        or proposal["equivalent_event_precision_requirement"]
        != "event_precision > anchor_f1 / 2"
        or proposal["persist_raw_keys"] is not False
    ):
        raise ContractError("anchor or event contract changed")
    outlier = config["outlier_policy"]
    if (
        outlier["remove_label_1_rows_or_events"] is not False
        or outlier["clip_or_remove_raw_temp_anomaly_signal"] is not False
        or outlier["auxiliary_quality_may_change_support_or_gate"] is not False
        or int(outlier["target_positive_rows_removed"]) != 0
    ):
        raise ContractError("target anomaly protection changed")
    if not all(config["prohibitions"].values()):
        raise ContractError("a prohibition was disabled")
    outputs = config["outputs"]
    if (
        outputs["terminal_receipt"]
        != "artifacts/p1_heterogeneous_event_utility_preflight_20260830_v1/terminal_result.json"
        or outputs["append_only"] is not True
        or outputs["allowed_file_extensions"] != [".json"]
        or int(outputs["prediction_csv_count"]) != 0
    ):
        raise ContractError("non-JSON output was authorized")


def run_registered() -> dict[str, Any]:
    """Run only the compiled-in registration; callers cannot override inputs."""
    if _sha256(CONFIG_PATH) != CONFIG_SHA256:
        raise ContractError("sealed preregistration sha256 changed")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _validate_config(config)
    try:
        frame, source_names, provenance = _load_inputs(ROOT, config)
    except ProvenanceError as error:
        result = _provenance_no_go(config, error)
    else:
        result = evaluate_support(frame, source_names, config, provenance)
    result["config_sha256"] = CONFIG_SHA256
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link is an atomic create-if-absent operation.
        # It cannot overwrite a winner created by a rival process.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-receipt",
        action="store_true",
        help="Persist only the preregistered aggregate terminal JSON receipt.",
    )
    args = parser.parse_args()
    result = run_registered()
    if args.write_receipt:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        output = (ROOT / config["outputs"]["terminal_receipt"]).resolve()
        _atomic_json(output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
