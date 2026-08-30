from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from p1_qc.clean_state_capa import (
    INPUT_ONLY_COLUMNS,
    KEY_COLUMNS,
    apply_clean_state,
    decode_frame,
    fit_clean_state,
    protected_union,
    synthetic_contract_audit,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p1_clean_state_capa_falsification_20260831_v1"
CONFIG_PATH = ROOT / "configs" / "experiments" / f"{EXPERIMENT_ID}.json"
ARTIFACT_DIR = ROOT / "artifacts" / EXPERIMENT_ID
ATTEMPT_LOCK = ROOT / "artifacts" / f"{EXPERIMENT_ID}.ATTEMPT_LOCK.json"
REPORT_DIR = ROOT / "reports" / EXPERIMENT_ID
TERMINAL_PATH = ARTIFACT_DIR / "terminal_result.json"
KST = timezone(timedelta(hours=9))
PART_COLUMNS = [*KEY_COLUMNS, "row_position", "baseline_probability", "baseline_prediction"]
SOURCE_PATHS = {
    "typed_duration_semimarkov": ROOT / "src" / "p1_qc" / "typed_duration_semimarkov.py",
    "long_event_segment_proposal_rescore": ROOT
    / "src"
    / "p1_qc"
    / "long_event_segment_proposal_rescore.py",
    "v6_slow_unary": ROOT / "src" / "p1_qc" / "multiscale_cross_layer_offset_drift_v6r2.py",
}


class RunnerContractError(RuntimeError):
    pass


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _int64_hash(values: Any) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _exclusive_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        os.write(descriptor, encoded + b"\n")
    finally:
        os.close(descriptor)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise RunnerContractError(f"JSON object required: {path}")
    return value


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")


def _validate_train_path(path: Path, config: dict[str, Any]) -> Path:
    resolved = path.resolve()
    if resolved.name.casefold() != "train.csv":
        raise RunnerContractError("only the historical P1 train.csv is allowed")
    lowered = str(resolved).casefold()
    for forbidden in config["input_contract"]["forbidden_paths"]:
        if str(forbidden).casefold() in lowered:
            raise RunnerContractError(f"forbidden path token: {forbidden}")
    if not resolved.is_file():
        raise RunnerContractError(f"train CSV does not exist: {resolved}")
    if _sha256_file(resolved) != config["input_contract"]["train_csv_sha256"]:
        raise RunnerContractError("historical train CSV hash differs")
    return resolved


def preflight(train_csv: Path, *, require_namespace: bool = True) -> dict[str, Any]:
    config = _load_json(CONFIG_PATH)
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RunnerContractError("experiment id differs")
    if config["authorization"] != {
        "user_message": "진행하세요",
        "maximum_executions": 1,
        "result_based_retry_or_retune": 0,
        "base_commit": "23788ffc1cbeac3b419eb0f1d62c72e68ba6f386",
    }:
        raise RunnerContractError("exactly-once authorization differs")
    if config["decision"]["official_submission_authorized"] is not False:
        raise RunnerContractError("official submission must remain unauthorized")
    if require_namespace and (ARTIFACT_DIR.exists() or ATTEMPT_LOCK.exists()):
        raise RunnerContractError("exactly-once namespace or attempt lock already exists")
    resolved_train = _validate_train_path(train_csv, config)
    upstream: dict[str, str] = {}
    for role, source_path in SOURCE_PATHS.items():
        actual = _sha256_file(source_path)
        expected = config["semantic_distinction"][role]["sha256"]
        if actual != expected:
            raise RunnerContractError(f"semantic predecessor hash differs: {role}")
        upstream[role] = actual
    part_checks: dict[str, Any] = {}
    for fold, item in config["incumbent_full_fraction_parts"].items():
        parquet = ROOT / item["path"]
        audit = ROOT / item["audit_path"]
        if _sha256_file(parquet) != item["sha256"]:
            raise RunnerContractError(f"incumbent part hash differs: {fold}")
        part_audit = _load_json(audit)
        if (
            part_audit.get("fold") != fold
            or part_audit.get("fraction") != 1.0
            or part_audit.get("target_fold_validation_labels_read_before_prediction") != 0
        ):
            raise RunnerContractError(f"incumbent part audit differs: {fold}")
        part_checks[fold] = {
            "parquet_sha256": item["sha256"],
            "audit_sha256": _sha256_file(audit),
            "validation_rows": int(part_audit["validation_rows"]),
            "prefix_rows": int(part_audit["prefix_rows"]),
        }
    synthetic = synthetic_contract_audit()
    if synthetic["status"] != "PASS":
        raise RunnerContractError("synthetic frozen decoder contract failed")
    structure = _historical_structure_audit(resolved_train, config)
    return {
        "status": "READY_EXACTLY_ONCE_RESEARCH_ONLY",
        "experiment_id": EXPERIMENT_ID,
        "train_csv": str(resolved_train),
        "train_csv_sha256": config["input_contract"]["train_csv_sha256"],
        "config_sha256": _sha256_file(CONFIG_PATH),
        "runner_sha256": _sha256_file(Path(__file__)),
        "module_sha256": _sha256_file(ROOT / "src" / "p1_qc" / "clean_state_capa.py"),
        "upstream_source_sha256": upstream,
        "incumbent_parts": part_checks,
        "historical_structure": structure,
        "synthetic_contract": synthetic,
        "official_test_sample_submission_value_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }


def _key_order_matches(expected: pd.DataFrame, observed: pd.DataFrame) -> bool:
    if len(expected) != len(observed):
        return False
    for column in ("station", "time"):
        if not np.array_equal(
            expected[column].astype(str).to_numpy(), observed[column].astype(str).to_numpy()
        ):
            return False
    for column in ("year", "layer"):
        if not np.array_equal(
            expected[column].to_numpy(dtype=np.int64),
            observed[column].to_numpy(dtype=np.int64),
        ):
            return False
    expected_time = pd.to_datetime(expected["time"], utc=True, errors="raise", format="mixed")
    observed_time = pd.to_datetime(observed["time"], utc=True, errors="raise", format="mixed")
    return np.array_equal(
        expected_time.to_numpy(dtype="datetime64[ns]"),
        observed_time.to_numpy(dtype="datetime64[ns]"),
    )


def _historical_structure_audit(train_csv: Path, config: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(train_csv, usecols=list(INPUT_ONLY_COLUMNS)).loc[
        :, list(INPUT_ONLY_COLUMNS)
    ]
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise RunnerContractError("historical input keys are not unique")
    parsed = pd.to_datetime(frame["time"], utc=True, errors="raise", format="mixed")
    fold_audits: dict[str, Any] = {}
    all_validation_positions: list[np.ndarray] = []
    for fold in config["folds"]:
        item = config["incumbent_full_fraction_parts"][fold]
        part_audit = _load_json(ROOT / item["audit_path"])
        part = pd.read_parquet(ROOT / item["path"], columns=PART_COLUMNS)
        positions = part["row_position"].to_numpy(dtype=np.int64)
        if (
            len(positions) != int(part_audit["validation_rows"])
            or len(np.unique(positions)) != len(positions)
            or positions.min(initial=0) < 0
            or positions.max(initial=-1) >= len(frame)
        ):
            raise RunnerContractError(f"invalid validation positions during preflight: {fold}")
        expected = frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        if not _key_order_matches(expected, part.loc[:, list(KEY_COLUMNS)]):
            raise RunnerContractError(f"validation keys differ during preflight: {fold}")
        cutoff = pd.Timestamp(part_audit["adjusted_cutoff_utc"])
        prefix_positions = np.flatnonzero((parsed <= cutoff).to_numpy()).astype(np.int64)
        if (
            len(prefix_positions) != int(part_audit["prefix_rows"])
            or _int64_hash(prefix_positions) != part_audit["prefix_positions_sha256"]
        ):
            raise RunnerContractError(f"prefix receipt differs during preflight: {fold}")
        if not bool((parsed.iloc[positions] > cutoff).all()):
            raise RunnerContractError(f"prefix/validation time overlap during preflight: {fold}")
        prefix = frame.iloc[prefix_positions]
        validation = frame.iloc[positions]
        counts = prefix.groupby(["station", "layer"], sort=False, observed=True).size()
        validation_groups = validation.loc[:, ["station", "layer"]].drop_duplicates()
        group_prefix_rows = [
            int(counts.get((row.station, row.layer), 0))
            for row in validation_groups.itertuples(index=False)
        ]
        minimum_rows = min(group_prefix_rows)
        unsupported_groups = sum(value < 32 for value in group_prefix_rows)
        if not np.isfinite(prefix["temp"].to_numpy(dtype=np.float64)).all():
            raise RunnerContractError(f"prefix temperature is nonfinite: {fold}")
        if not np.isfinite(validation["temp"].to_numpy(dtype=np.float64)).all():
            raise RunnerContractError(f"validation temperature is nonfinite: {fold}")
        incumbent = part["baseline_prediction"].to_numpy(dtype=np.int8)
        probability = part["baseline_probability"].to_numpy(dtype=np.float64)
        if not set(np.unique(incumbent)).issubset({0, 1}) or not np.isfinite(probability).all():
            raise RunnerContractError(f"incumbent surface is invalid: {fold}")
        all_validation_positions.append(positions)
        fold_audits[fold] = {
            "prefix_rows": int(len(prefix_positions)),
            "validation_rows": int(len(positions)),
            "validation_group_count": int(len(validation_groups)),
            "minimum_validation_group_prefix_rows": int(minimum_rows),
            "unsupported_validation_groups": int(unsupported_groups),
            "unsupported_group_policy": "zero_signal_abstention_no_additions",
            "key_order_matches": True,
            "prefix_receipt_matches": True,
            "prefix_validation_disjoint_by_time": True,
            "temperature_finite": True,
            "incumbent_binary_and_probability_finite": True,
        }
    pooled = np.concatenate(all_validation_positions)
    if len(np.unique(pooled)) != len(pooled):
        raise RunnerContractError("historical validation folds overlap by row position")
    return {
        "historical_input_rows": int(len(frame)),
        "pooled_validation_rows": int(len(pooled)),
        "pooled_validation_positions_unique": True,
        "folds": fold_audits,
        "target_columns_read": 0,
    }


def _binary_metrics(truth: Any, prediction: Any) -> dict[str, Any]:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    if y.ndim != 1 or p.shape != y.shape or not set(np.unique(y)).issubset({0, 1}):
        raise RunnerContractError("binary metric vectors differ or are not binary")
    if not set(np.unique(p)).issubset({0, 1}):
        raise RunnerContractError("prediction is not binary")
    tp = int(np.count_nonzero((y == 1) & (p == 1)))
    fp = int(np.count_nonzero((y == 0) & (p == 1)))
    fn = int(np.count_nonzero((y == 1) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return {
        "rows": int(len(y)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "f1": float((2 * tp) / denominator) if denominator else 0.0,
    }


def _type_recall(truth: np.ndarray, prediction: np.ndarray, types: pd.Series, name: str) -> float:
    tokens = types.astype("string").fillna("").str.split("|")
    type_mask = tokens.map(lambda values: name in {str(item).strip() for item in values}).to_numpy()
    eligible = (truth == 1) & type_mask
    return float(prediction[eligible].mean()) if eligible.any() else 0.0


def _score_surface(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    additions: np.ndarray,
    anomaly_type: pd.Series,
    metadata: pd.DataFrame,
) -> dict[str, Any]:
    incumbent_metrics = _binary_metrics(truth, incumbent)
    candidate_metrics = _binary_metrics(truth, candidate)
    additions_mask = additions.astype(bool)
    additions_precision = float(truth[additions_mask].mean()) if additions_mask.any() else 0.0
    station_layer: list[dict[str, Any]] = []
    work = metadata.loc[:, ["station", "layer"]].copy()
    work["_position"] = np.arange(len(work), dtype=np.int64)
    for (station, layer), positions_raw in work.groupby(
        ["station", "layer"], sort=True, observed=True
    ).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        base_f1 = _binary_metrics(truth[positions], incumbent[positions])["f1"]
        candidate_f1 = _binary_metrics(truth[positions], candidate[positions])["f1"]
        station_layer.append(
            {
                "station": str(station),
                "layer": int(layer),
                "rows": int(len(positions)),
                "delta_f1": float(candidate_f1 - base_f1),
                "additions": int(additions_mask[positions].sum()),
            }
        )
    parsed = pd.to_datetime(metadata["time"], utc=True, errors="raise", format="mixed")
    june = (parsed.dt.tz_convert("Asia/Seoul").dt.month == 6).to_numpy()
    june_delta = None
    if june.any():
        june_delta = float(
            _binary_metrics(truth[june], candidate[june])["f1"]
            - _binary_metrics(truth[june], incumbent[june])["f1"]
        )
    return {
        "incumbent": incumbent_metrics,
        "candidate": candidate_metrics,
        "delta_f1": float(candidate_metrics["f1"] - incumbent_metrics["f1"]),
        "additions": int(additions_mask.sum()),
        "additions_true_positive": int(truth[additions_mask].sum()),
        "additions_precision": additions_precision,
        "incumbent_positive_removals": int(np.count_nonzero((incumbent == 1) & (candidate == 0))),
        "offset_recall": _type_recall(truth, candidate, anomaly_type, "offset"),
        "drift_recall": _type_recall(truth, candidate, anomaly_type, "drift"),
        "june_kst_delta_f1": june_delta,
        "station_layer_diagnostics": station_layer,
    }


def _cluster_ids(metadata: pd.DataFrame, truth: np.ndarray) -> np.ndarray:
    parsed = pd.to_datetime(metadata["time"], utc=True, errors="raise", format="mixed")
    time_ns = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    station = metadata["station"].astype(str).to_numpy()
    layer = metadata["layer"].to_numpy(dtype=np.int64)
    order = np.lexsort((time_ns, layer, station))
    unit = np.empty(len(metadata), dtype=object)
    event_number = 0
    prior_station = ""
    prior_layer = -1
    prior_time = np.iinfo(np.int64).min
    prior_positive = False
    maximum_gap = 30 * 60 * 1_000_000_000
    kst_dates = parsed.dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d").to_numpy()
    for position in order:
        if truth[position] == 1:
            continuation = (
                prior_positive
                and station[position] == prior_station
                and layer[position] == prior_layer
                and time_ns[position] - prior_time <= maximum_gap
            )
            if not continuation:
                event_number += 1
            unit[position] = f"event|{event_number}"
            prior_positive = True
        else:
            unit[position] = f"normal|{station[position]}|{layer[position]}|{kst_dates[position]}"
            prior_positive = False
        prior_station = station[position]
        prior_layer = int(layer[position])
        prior_time = int(time_ns[position])
    codes, _uniques = pd.factorize(unit, sort=True)
    if (codes < 0).any():
        raise RunnerContractError("cluster assignment is incomplete")
    return codes.astype(np.int64)


def _paired_cluster_bootstrap(
    truth: np.ndarray,
    incumbent: np.ndarray,
    candidate: np.ndarray,
    metadata: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    cluster = _cluster_ids(metadata, truth)
    cluster_count = int(cluster.max()) + 1
    vectors: list[np.ndarray] = []
    for prediction in (incumbent, candidate):
        vectors.extend(
            [
                np.bincount(cluster, weights=(truth & prediction), minlength=cluster_count),
                np.bincount(
                    cluster,
                    weights=((truth == 0) & (prediction == 1)),
                    minlength=cluster_count,
                ),
                np.bincount(
                    cluster,
                    weights=((truth == 1) & (prediction == 0)),
                    minlength=cluster_count,
                ),
            ]
        )
    base_tp, base_fp, base_fn, cand_tp, cand_fp, cand_fn = vectors
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = rng.integers(0, cluster_count, size=cluster_count)
        counts = np.bincount(sampled, minlength=cluster_count)
        btp = float(counts @ base_tp)
        bfp = float(counts @ base_fp)
        bfn = float(counts @ base_fn)
        ctp = float(counts @ cand_tp)
        cfp = float(counts @ cand_fp)
        cfn = float(counts @ cand_fn)
        base_denominator = 2.0 * btp + bfp + bfn
        cand_denominator = 2.0 * ctp + cfp + cfn
        base_f1 = 2.0 * btp / base_denominator if base_denominator else 0.0
        cand_f1 = 2.0 * ctp / cand_denominator if cand_denominator else 0.0
        deltas[replicate] = cand_f1 - base_f1
    lower, upper = np.quantile(deltas, [0.05, 0.95])
    return {
        "cluster_count": cluster_count,
        "replicates": replicates,
        "seed": seed,
        "ci90": [float(lower), float(upper)],
        "bootstrap_mean_delta_f1": float(deltas.mean()),
        "positive_probability": float(np.mean(deltas > 0.0)),
    }


def _save_fold_prediction(
    fold: str,
    row_position: np.ndarray,
    incumbent: np.ndarray,
    additions: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    path = ARTIFACT_DIR / f"{fold}_blind_predictions.npz"
    np.savez_compressed(
        path,
        row_position=np.asarray(row_position, dtype=np.int64),
        incumbent=np.asarray(incumbent, dtype=np.int8),
        additions=np.asarray(additions, dtype=bool),
        candidate=np.asarray(candidate, dtype=np.int8),
    )
    return {
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "rows": int(len(row_position)),
        "incumbent_positives": int(np.asarray(incumbent).sum()),
        "addition_rows": int(np.asarray(additions).sum()),
        "candidate_positives": int(np.asarray(candidate).sum()),
    }


def execute(train_csv: Path) -> dict[str, Any]:
    started = time.monotonic()
    preflight_result = preflight(train_csv, require_namespace=True)
    config = _load_json(CONFIG_PATH)
    lock = {
        "schema_version": "p1.clean_state_capa.attempt_lock.v1",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": _now_kst(),
        "authorization": config["authorization"],
        "config_sha256": preflight_result["config_sha256"],
        "runner_sha256": preflight_result["runner_sha256"],
        "maximum_executions": 1,
    }
    _exclusive_json(ATTEMPT_LOCK, lock)
    ARTIFACT_DIR.mkdir(parents=False, exist_ok=False)
    _json_new(ARTIFACT_DIR / "preflight.json", preflight_result)

    train_path = Path(preflight_result["train_csv"])
    input_frame = pd.read_csv(train_path, usecols=list(INPUT_ONLY_COLUMNS))
    input_frame = input_frame.loc[:, list(INPUT_ONLY_COLUMNS)]
    if input_frame.duplicated(list(KEY_COLUMNS)).any():
        raise RunnerContractError("historical input keys are not unique")
    all_time = pd.to_datetime(input_frame["time"], utc=True, errors="raise", format="mixed")
    fold_seals: list[dict[str, Any]] = []
    for fold in config["folds"]:
        item = config["incumbent_full_fraction_parts"][fold]
        part_path = ROOT / item["path"]
        part_audit_path = ROOT / item["audit_path"]
        part_audit = _load_json(part_audit_path)
        part = pd.read_parquet(part_path, columns=PART_COLUMNS)
        positions = part["row_position"].to_numpy(dtype=np.int64)
        if (
            len(np.unique(positions)) != len(positions)
            or positions.min(initial=0) < 0
            or positions.max(initial=-1) >= len(input_frame)
        ):
            raise RunnerContractError(f"invalid validation row positions: {fold}")
        expected_keys = input_frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        if not _key_order_matches(expected_keys, part.loc[:, list(KEY_COLUMNS)]):
            raise RunnerContractError(f"validation key/order differs: {fold}")
        cutoff = pd.Timestamp(part_audit["adjusted_cutoff_utc"])
        prefix_positions = np.flatnonzero((all_time <= cutoff).to_numpy()).astype(np.int64)
        if (
            len(prefix_positions) != int(part_audit["prefix_rows"])
            or _int64_hash(prefix_positions) != part_audit["prefix_positions_sha256"]
        ):
            raise RunnerContractError(f"historical prefix differs: {fold}")
        validation_time = all_time.iloc[positions]
        if not bool((validation_time > cutoff).all()):
            raise RunnerContractError(f"prefix and validation time overlap: {fold}")

        prefix = input_frame.iloc[prefix_positions].reset_index(drop=True)
        validation = input_frame.iloc[positions].reset_index(drop=True)
        state = fit_clean_state(prefix)
        projection = apply_clean_state(validation, state)
        additions, proposals, decoder_audit = decode_frame(validation, projection)
        incumbent = part["baseline_prediction"].to_numpy(dtype=np.int8)
        candidate = protected_union(incumbent, additions)
        prediction_seal = _save_fold_prediction(fold, positions, incumbent, additions, candidate)
        proposal_path = ARTIFACT_DIR / f"{fold}_proposals.json"
        _json_new(
            proposal_path,
            {
                "schema_version": "p1.clean_state_capa.proposals.v1",
                "experiment_id": EXPERIMENT_ID,
                "fold": fold,
                "clean_state_sha256": state.sha256,
                "decoder_audit": decoder_audit,
                "proposals": proposals,
                "target_columns_read": 0,
            },
        )
        seal = {
            "fold": fold,
            "adjusted_cutoff_utc": str(part_audit["adjusted_cutoff_utc"]),
            "prefix_rows": int(len(prefix_positions)),
            "prefix_positions_sha256": _int64_hash(prefix_positions),
            "validation_rows": int(len(positions)),
            "validation_key_order_matches": True,
            "prefix_validation_disjoint_by_time": True,
            "clean_state_sha256": state.sha256,
            "decoder_audit": decoder_audit,
            "proposal_path": _relative(proposal_path),
            "proposal_sha256": _sha256_file(proposal_path),
            "prediction": prediction_seal,
            "target_columns_read_before_seal": 0,
        }
        seal_path = ARTIFACT_DIR / f"{fold}_seal.json"
        _json_new(seal_path, seal)
        seal["seal_path"] = _relative(seal_path)
        seal["seal_sha256"] = _sha256_file(seal_path)
        fold_seals.append(seal)

    pooled_positions = np.concatenate(
        [
            np.load(ROOT / seal["prediction"]["path"], allow_pickle=False)["row_position"]
            for seal in fold_seals
        ]
    )
    if len(np.unique(pooled_positions)) != len(pooled_positions):
        raise RunnerContractError("historical validation folds overlap by row position")
    predictions_complete = {
        "schema_version": "p1.clean_state_capa.predictions_complete.v1",
        "experiment_id": EXPERIMENT_ID,
        "completed_at_kst": _now_kst(),
        "fold_count": len(fold_seals),
        "fold_seals": fold_seals,
        "pooled_validation_rows": int(len(pooled_positions)),
        "pooled_row_positions_sha256": _int64_hash(pooled_positions),
        "top_level_clean_state_fits": len(fold_seals),
        "supervised_model_fits": 0,
        "target_columns_read_before_completion": 0,
        "official_test_sample_submission_value_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    completion_path = ARTIFACT_DIR / "predictions_complete.json"
    _json_new(completion_path, predictions_complete)
    completion_sha256 = _sha256_file(completion_path)

    targets = pd.read_csv(train_path, usecols=["label", "anomaly_type"])
    truth_all = pd.to_numeric(targets["label"], errors="raise").to_numpy(dtype=np.int8)
    if len(targets) != len(input_frame) or not set(np.unique(truth_all)).issubset({0, 1}):
        raise RunnerContractError("historical targets do not align or are not binary")
    fold_scores: list[dict[str, Any]] = []
    pooled_truth: list[np.ndarray] = []
    pooled_incumbent: list[np.ndarray] = []
    pooled_candidate: list[np.ndarray] = []
    pooled_additions: list[np.ndarray] = []
    pooled_types: list[pd.Series] = []
    pooled_metadata: list[pd.DataFrame] = []
    for seal in fold_seals:
        fold = seal["fold"]
        prediction_path = ROOT / seal["prediction"]["path"]
        if _sha256_file(prediction_path) != seal["prediction"]["sha256"]:
            raise RunnerContractError(f"sealed prediction hash differs: {fold}")
        with np.load(prediction_path, allow_pickle=False) as arrays:
            positions = arrays["row_position"].astype(np.int64, copy=False)
            incumbent = arrays["incumbent"].astype(np.int8, copy=False)
            additions = arrays["additions"].astype(bool, copy=False)
            candidate = arrays["candidate"].astype(np.int8, copy=False)
        truth = truth_all[positions]
        anomaly = targets.iloc[positions]["anomaly_type"].reset_index(drop=True)
        metadata = input_frame.iloc[positions].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
        score = _score_surface(truth, incumbent, candidate, additions, anomaly, metadata)
        fold_scores.append({"fold": fold, **score})
        pooled_truth.append(truth)
        pooled_incumbent.append(incumbent)
        pooled_candidate.append(candidate)
        pooled_additions.append(additions)
        pooled_types.append(anomaly)
        pooled_metadata.append(metadata)

    truth = np.concatenate(pooled_truth)
    incumbent = np.concatenate(pooled_incumbent)
    candidate = np.concatenate(pooled_candidate)
    additions = np.concatenate(pooled_additions)
    anomaly_type = pd.concat(pooled_types, ignore_index=True)
    metadata = pd.concat(pooled_metadata, ignore_index=True)
    pooled_score = _score_surface(truth, incumbent, candidate, additions, anomaly_type, metadata)
    bootstrap = _paired_cluster_bootstrap(
        truth,
        incumbent,
        candidate,
        metadata,
        replicates=int(config["decision"]["bootstrap_replicates"]),
        seed=int(config["decision"]["seed"]),
    )
    favorable = (
        pooled_score["delta_f1"] > 0.0
        and pooled_score["additions_precision"] > pooled_score["incumbent"]["f1"] / 2.0
    )
    decision = (
        config["decision"]["favorable_label"] if favorable else config["decision"]["otherwise"]
    )
    result = {
        "schema_version": "p1.clean_state_capa.result.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "decision": decision,
        "completed_at_kst": _now_kst(),
        "runtime_seconds": float(time.monotonic() - started),
        "fold_scores": fold_scores,
        "pooled": pooled_score,
        "paired_cluster_bootstrap": bootstrap,
        "decision_checks": {
            "pooled_row_binary_micro_f1_delta_strictly_positive": bool(
                pooled_score["delta_f1"] > 0.0
            ),
            "additions_precision_strictly_greater_than_incumbent_f1_divided_by_2": bool(
                pooled_score["additions_precision"] > pooled_score["incumbent"]["f1"] / 2.0
            ),
            "incumbent_positive_removals_zero": pooled_score["incumbent_positive_removals"] == 0,
        },
        "fits": {
            "top_level_clean_state_fits": len(fold_seals),
            "supervised_model_fits": 0,
        },
        "hashes": {
            "config_sha256": preflight_result["config_sha256"],
            "runner_sha256": preflight_result["runner_sha256"],
            "module_sha256": preflight_result["module_sha256"],
            "predictions_complete_sha256": completion_sha256,
            "attempt_lock_sha256": _sha256_file(ATTEMPT_LOCK),
        },
        "access_audit": {
            "historical_input_rows_read": int(len(input_frame)),
            "historical_target_rows_read_after_predictions_complete": int(len(targets)),
            "official_test_sample_submission_value_reads": 0,
            "hidden_label_reads": 0,
            "submission_csv_created": 0,
            "uploads": 0,
        },
        "promotion_scope": "RESEARCH_ONLY_NO_OFFICIAL_MATERIALIZATION",
    }
    result_path = ARTIFACT_DIR / "result.json"
    _json_new(result_path, result)
    manifest = {
        "schema_version": "p1.clean_state_capa.manifest.v1",
        "experiment_id": EXPERIMENT_ID,
        "files": [
            {
                "path": _relative(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(ARTIFACT_DIR.iterdir())
            if path.is_file() and path.name not in {"manifest.json", "terminal_result.json"}
        ],
        "official_test_sample_submission_value_reads": 0,
        "submission_csv_created": 0,
        "uploads": 0,
    }
    _json_new(ARTIFACT_DIR / "manifest.json", manifest)
    terminal = {
        "schema_version": "p1.clean_state_capa.terminal.v1",
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "decision": decision,
        "completed_at_kst": _now_kst(),
        "result_path": _relative(result_path),
        "result_sha256": _sha256_file(result_path),
        "manifest_sha256": _sha256_file(ARTIFACT_DIR / "manifest.json"),
        "attempt_lock_sha256": _sha256_file(ATTEMPT_LOCK),
    }
    _json_new(TERMINAL_PATH, terminal)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _json_new(REPORT_DIR / "result.json", result)
    return terminal


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--train-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.check:
        print(json.dumps(preflight(args.train_csv), ensure_ascii=False, indent=2))
        return 0
    try:
        terminal = execute(args.train_csv)
    except Exception as exc:
        if ARTIFACT_DIR.is_dir() and not TERMINAL_PATH.exists():
            _json_new(
                TERMINAL_PATH,
                {
                    "schema_version": "p1.clean_state_capa.terminal.v1",
                    "experiment_id": EXPERIMENT_ID,
                    "status": "TECHNICAL_FAILURE",
                    "completed_at_kst": _now_kst(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "automatic_restart_authorized": False,
                },
            )
        raise
    print(json.dumps(terminal, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
