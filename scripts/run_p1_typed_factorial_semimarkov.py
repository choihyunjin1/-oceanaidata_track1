"""Run the preregistered P1 typed factorial semi-Markov historical experiment.

The executable has no outer-fold, test, submission, or upload mode.  It opens
exactly one fail-fast attempt and, only when that structural check passes,
evaluates the frozen configuration on three pre-outer historical blocks.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_qc.data import load_dataset  # noqa: E402
from p1_qc.features import FeatureBundle  # noqa: E402
from p1_qc.models_tabular import make_tabular_classifier  # noqa: E402
from p1_qc.pipeline import TabularEncoder  # noqa: E402
from p1_qc.typed_factorial_semimarkov import (  # noqa: E402
    ANOMALY_TYPES,
    DecoderConfig,
    GrammarAudit,
    binary_counts,
    build_grammar_audit,
    chronological_split_masks,
    decode_frame,
    duration_is_decomposable,
    recall_by_type,
    rowwise_union,
    station_layer_f1,
)

EXPERIMENT_ID = "p1_typed_factorial_semimarkov_v1"
CANONICAL_CONFIG = (
    PROJECT_ROOT / "configs" / "experiments" / "p1_typed_factorial_semimarkov_v1.json"
)
HELPER = PROJECT_ROOT / "src" / "p1_qc" / "typed_factorial_semimarkov.py"
RUNNER = Path(__file__).resolve()
TESTS = PROJECT_ROOT / "tests" / "test_typed_factorial_semimarkov.py"
HISTORICAL_HARD_CUTOFF = pd.Timestamp("2025-03-01T23:50:00+09:00").tz_convert("UTC")


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _read_config() -> dict[str, Any]:
    config = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("canonical experiment ID mismatch")
    authorization = config.get("authorization", {})
    forbidden = (
        "outer_validation_or_scoring",
        "test_read_or_prediction",
        "submission_generation_or_mutation",
        "upload",
        "existing_file_mutation",
        "hyperparameter_or_threshold_search",
    )
    if any(bool(authorization.get(item)) for item in forbidden):
        raise RuntimeError("canonical authorization enables a forbidden operation")
    if config["grammar"]["embargo_days"] != 8:
        raise RuntimeError("embargo must remain exactly eight days")
    if config["grammar"]["maximum_concurrent_events"] != 2:
        raise RuntimeError("factorial maximum concurrency must remain two")
    if len(config["historical_inner_blocks"]) != 3:
        raise RuntimeError("exactly three historical blocks are required")
    configured_cutoff = pd.Timestamp(
        config["source_contract"]["historical_truth_cutoff_inclusive"]
    ).tz_convert("UTC")
    if configured_cutoff != HISTORICAL_HARD_CUTOFF:
        raise RuntimeError("historical hard cutoff changed")
    return config


def _write_json_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False, default=str
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    if reloaded != json.loads(payload):
        raise RuntimeError(f"atomic JSON reload mismatch: {path.name}")


def _create_attempt_lock(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        payload = (
            json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode("utf-8")
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError("attempt lock reload mismatch")


def _implementation_hashes() -> dict[str, str]:
    return {
        _relative(CANONICAL_CONFIG): _sha256(CANONICAL_CONFIG),
        _relative(HELPER): _sha256(HELPER),
        _relative(RUNNER): _sha256(RUNNER),
        _relative(TESTS): _sha256(TESTS),
    }


class Gauge:
    def __init__(self, path: Path, started: float) -> None:
        self.path = path
        self.started = started

    def update(
        self,
        *,
        phase: str,
        progress_percent: float,
        detail: str,
        eta_seconds: float | None = None,
        terminal: bool = False,
    ) -> None:
        elapsed = time.perf_counter() - self.started
        _write_json_fsync(
            self.path,
            {
                "schema_version": "1.0",
                "experiment_id": EXPERIMENT_ID,
                "updated_at_kst": datetime.now().astimezone().isoformat(),
                "phase": phase,
                "detail": detail,
                "progress_percent": round(float(progress_percent), 2),
                "elapsed_seconds": round(elapsed, 3),
                "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
                "historical_inner_only": True,
                "outer_validation_or_scoring_count": 0,
                "test_prediction_count": 0,
                "submission_generation_count": 0,
                "upload_count": 0,
                "terminal": bool(terminal),
            },
        )


def _duration_mapping(config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    raw = config["grammar"]["duration_rows_inclusive"]
    return {name: (int(raw[name][0]), int(raw[name][1])) for name in ANOMALY_TYPES}


def _load_feature_bundle(
    config: dict[str, Any], *, source_sha256: str, source_rows: int
) -> tuple[FeatureBundle, dict[str, Any], dict[str, str]]:
    contract = config["source_contract"]
    metadata_path = PROJECT_ROOT / contract["feature_cache_metadata"]
    parquet_path = PROJECT_ROOT / contract["feature_cache_parquet"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_sha256") != source_sha256:
        raise RuntimeError("feature cache source SHA differs from immutable train source")
    if int(metadata.get("rows", -1)) != source_rows:
        raise RuntimeError("feature cache row count differs from train source")
    parquet_sha = _sha256(parquet_path)
    if parquet_sha != metadata.get("parquet_sha256"):
        raise RuntimeError("feature cache parquet SHA mismatch")
    feature_frame = pd.read_parquet(parquet_path)
    if len(feature_frame) != source_rows:
        raise RuntimeError("feature cache frame length mismatch")
    feature_columns = tuple(metadata["feature_columns"])
    categorical_columns = tuple(metadata["categorical_columns"])
    if not set(feature_columns).issubset(feature_frame.columns):
        raise RuntimeError("feature cache is missing registered columns")
    return (
        FeatureBundle(feature_frame, feature_columns, categorical_columns),
        metadata,
        {
            _relative(metadata_path): _sha256(metadata_path),
            _relative(parquet_path): parquet_sha,
        },
    )


def _sensitive_signature(sequence: tuple[str, ...]) -> bool:
    if len(sequence) < 2:
        return False
    if len(sequence) != len(set(sequence)):
        return True
    canonical = tuple(sorted(sequence, key=ANOMALY_TYPES.index))
    return sequence != canonical


def _precheck_metrics(
    frame: pd.DataFrame,
    audit: GrammarAudit,
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    label = frame["label"].to_numpy(dtype=np.int8)
    positive = label == 1
    positive_count = int(positive.sum())
    if positive_count == 0:
        raise RuntimeError("precheck scope has no anomaly rows")
    run_counts = Counter()
    for run in audit.atomic_runs:
        if run.occurrence_rank != 1:
            continue
        minimum, maximum = _duration_mapping(config)[run.anomaly_type]
        if duration_is_decomposable(run.length, minimum, maximum):
            run_counts[run.anomaly_type] += 1
    composite_events = 0
    for event in audit.super_events:
        positions = slice(event.start_position, event.stop_position + 1)
        if (audit.token_counts[positions].sum(axis=1) >= 2).any():
            composite_events += 1
    sensitive = {
        item.raw for item in audit.sequences if item.raw and _sensitive_signature(item.tokens)
    }
    thresholds = config["fail_fast_precheck"]
    recognized_coverage = float(audit.recognized_positive_rows.sum() / positive_count)
    max_two_coverage = float(audit.max_two_positive_rows.sum() / positive_count)
    duration_coverage = float(audit.duration_decomposable_positive_rows.sum() / positive_count)
    checks = {
        "all_five_types_present": all(run_counts[name] > 0 for name in ANOMALY_TYPES),
        "minimum_complete_presence_runs_per_type": all(
            run_counts[name] >= int(thresholds["minimum_complete_presence_runs_per_type"])
            for name in ANOMALY_TYPES
        ),
        "minimum_composite_super_events": composite_events
        >= int(thresholds["minimum_composite_super_events"]),
        "minimum_order_or_multiplicity_sensitive_signatures": len(sensitive)
        >= int(thresholds["minimum_order_or_multiplicity_sensitive_signatures"]),
        "minimum_positive_row_recognized_coverage": recognized_coverage
        >= float(thresholds["minimum_positive_row_recognized_coverage"]),
        "minimum_positive_row_max_two_coverage": max_two_coverage
        >= float(thresholds["minimum_positive_row_max_two_coverage"]),
        "minimum_positive_row_duration_decomposable_coverage": duration_coverage
        >= float(thresholds["minimum_positive_row_duration_decomposable_coverage"]),
    }
    metrics = {
        "scope_rows": len(frame),
        "positive_rows": positive_count,
        "positive_fraction": float(positive.mean()),
        "recognized_positive_row_coverage": recognized_coverage,
        "maximum_two_concurrent_positive_row_coverage": max_two_coverage,
        "duration_decomposable_positive_row_coverage": duration_coverage,
        "atomic_presence_run_counts_rank_one": {
            name: int(run_counts[name]) for name in ANOMALY_TYPES
        },
        "super_event_count": len(audit.super_events),
        "composite_super_event_count": composite_events,
        "order_or_multiplicity_sensitive_distinct_signature_count": len(sensitive),
        "order_or_multiplicity_sensitive_signatures": sorted(sensitive),
        "raw_signature_counts": dict(audit.signature_counts),
        "checks": checks,
        "passed": all(checks.values()),
    }
    return metrics, bool(metrics["passed"])


def _sample_weights(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=np.int8)
    positives = max(1, int(target.sum()))
    negatives = max(1, len(target) - positives)
    weight = float(np.sqrt(negatives / positives))
    return np.where(target == 1, weight, 1.0).astype(np.float32)


def _block_metrics(
    validation_frame: pd.DataFrame,
    membership: np.ndarray,
    truth: np.ndarray,
    control: np.ndarray,
    decoder: np.ndarray,
    no_op: np.ndarray,
) -> dict[str, Any]:
    control_counts = binary_counts(truth, control)
    decoder_counts = binary_counts(truth, decoder)
    control_recall = recall_by_type(membership, control)
    decoder_recall = recall_by_type(membership, decoder)
    type_delta = {
        name: (
            None
            if decoder_recall[name] is None or control_recall[name] is None
            else float(decoder_recall[name] - control_recall[name])
        )
        for name in ANOMALY_TYPES
    }
    composite = membership.sum(axis=1) >= 2
    control_group = station_layer_f1(validation_frame, truth, control)
    decoder_group = station_layer_f1(validation_frame, truth, decoder)
    group_delta = {name: float(decoder_group[name] - control_group[name]) for name in control_group}
    fp_relative = float(
        (int(decoder_counts["fp"]) - int(control_counts["fp"])) / max(1, int(control_counts["fp"]))
    )
    return {
        "rows": len(truth),
        "positive_rows": int(truth.sum()),
        "control": control_counts,
        "decoder": decoder_counts,
        "micro_f1_delta": float(decoder_counts["f1"] - control_counts["f1"]),
        "control_recall_by_type": control_recall,
        "decoder_recall_by_type": decoder_recall,
        "recall_delta_by_type": type_delta,
        "composite_rows": int(composite.sum()),
        "control_composite_recall": float(control[composite].mean()) if composite.any() else None,
        "decoder_composite_recall": float(decoder[composite].mean()) if composite.any() else None,
        "normal_false_positive_relative_delta": fp_relative,
        "station_layer_f1_delta": group_delta,
        "worst_station_layer_f1_delta": min(group_delta.values()) if group_delta else 0.0,
        "no_op_rows": int(no_op.sum()),
        "no_op_share": float(no_op.mean()),
    }


def _aggregate_and_decide(
    blocks: list[dict[str, Any]],
    pooled: dict[str, list[np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = np.concatenate(pooled["truth"])
    control = np.concatenate(pooled["control"])
    decoder = np.concatenate(pooled["decoder"])
    membership = np.concatenate(pooled["membership"])
    control_counts = binary_counts(truth, control)
    decoder_counts = binary_counts(truth, decoder)
    control_recall = recall_by_type(membership, control)
    decoder_recall = recall_by_type(membership, decoder)
    recall_delta = {
        name: float(decoder_recall[name] - control_recall[name])
        for name in ANOMALY_TYPES
        if decoder_recall[name] is not None and control_recall[name] is not None
    }
    if set(recall_delta) != set(ANOMALY_TYPES):
        raise RuntimeError("pooled historical validation lacks at least one anomaly type")
    group_deltas = [
        float(value)
        for block in blocks
        for value in block["metrics"]["station_layer_f1_delta"].values()
    ]
    fp_relative = float(
        (int(decoder_counts["fp"]) - int(control_counts["fp"])) / max(1, int(control_counts["fp"]))
    )
    gate = config["inner_decision_gate"]
    checks = {
        "pooled_micro_f1_delta": float(decoder_counts["f1"] - control_counts["f1"])
        > float(gate["minimum_pooled_micro_f1_delta_exclusive"]),
        "improved_blocks": sum(block["metrics"]["micro_f1_delta"] > 0 for block in blocks)
        >= int(gate["minimum_improved_blocks"]),
        "decoder_pooled_micro_f1": float(decoder_counts["f1"])
        >= float(gate["minimum_decoder_pooled_micro_f1"]),
        "offset_or_drift_recall_gain": max(recall_delta["offset"], recall_delta["drift"])
        >= float(gate["minimum_offset_or_drift_recall_gain"]),
        "non_slow_type_recall_drop": min(
            recall_delta["spike"], recall_delta["noise"], recall_delta["flatline"]
        )
        >= -float(gate["maximum_other_type_recall_drop"]),
        "normal_false_positive_relative_increase": fp_relative
        <= float(gate["maximum_normal_false_positive_relative_increase"]),
        "worst_station_layer_f1_drop": (min(group_deltas) if group_deltas else 0.0)
        >= -float(gate["maximum_worst_station_layer_f1_drop"]),
    }
    passed = all(checks.values())
    return {
        "rows": len(truth),
        "positive_rows": int(truth.sum()),
        "control": control_counts,
        "decoder": decoder_counts,
        "micro_f1_delta": float(decoder_counts["f1"] - control_counts["f1"]),
        "improved_blocks": sum(block["metrics"]["micro_f1_delta"] > 0 for block in blocks),
        "control_recall_by_type": control_recall,
        "decoder_recall_by_type": decoder_recall,
        "recall_delta_by_type": recall_delta,
        "normal_false_positive_relative_delta": fp_relative,
        "worst_station_layer_f1_delta": min(group_deltas) if group_deltas else 0.0,
        "no_op_rows": int(sum(block["metrics"]["no_op_rows"] for block in blocks)),
        "no_op_share": float(sum(block["metrics"]["no_op_rows"] for block in blocks) / len(truth)),
        "gate_checks": checks,
        "gate_passed": passed,
        "decision": gate["promotion_target"] if passed else gate["failure_target"],
    }


def run(data_dir: Path) -> int:
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    config = _read_config()
    output_dir = PROJECT_ROOT / config["artifacts"]["output_dir"]
    status_path = PROJECT_ROOT / config["artifacts"]["status_json"]
    attempt_lock = PROJECT_ROOT / config["artifacts"]["attempt_lock"]
    gauge = Gauge(status_path, started)
    if attempt_lock.exists() or (output_dir / "result.json").exists():
        raise FileExistsError("the single historical-inner attempt already exists")

    data_dir = data_dir.expanduser().resolve(strict=True)
    train_path = data_dir / config["source_contract"]["train_file"]
    readme_path = data_dir / "README.md"
    if not train_path.is_file() or not readme_path.is_file():
        raise FileNotFoundError(
            "data directory must contain the immutable P1 train.csv and README.md"
        )
    implementation = _implementation_hashes()
    _create_attempt_lock(
        attempt_lock,
        {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "created_at_kst": datetime.now().astimezone().isoformat(),
            "implementation_sha256": implementation,
            "historical_inner_only": True,
            "outer_validation_or_scoring": False,
            "test_prediction": False,
            "submission_generation": False,
            "upload": False,
        },
    )
    gauge.update(phase="load", progress_percent=2, detail="immutable train audit")
    source = load_dataset(train_path, kind="train", audit=True, strict=True)
    train_sha = str(source.attrs["source_sha256"])
    source_rows = len(source)
    source_times = pd.to_datetime(source["time"], errors="raise", utc=True)
    historical_mask = source_times <= HISTORICAL_HARD_CUTOFF
    historical = source.loc[
        historical_mask, ["station", "layer", "time", "label", "anomaly_type"]
    ].copy()
    historical["__source_position"] = np.flatnonzero(historical_mask)
    historical["__time_utc"] = source_times[historical_mask].to_numpy()
    historical = historical.sort_values(
        ["station", "layer", "__time_utc", "__source_position"], kind="mergesort"
    ).reset_index(drop=True)
    if historical["__time_utc"].max() > HISTORICAL_HARD_CUTOFF:
        raise RuntimeError("historical truth cutoff was violated")
    del source, source_times
    gc.collect()
    peak_rss = max(peak_rss, int(process.memory_info().rss))

    gauge.update(
        phase="features", progress_percent=6, detail="verify and load frozen feature cache"
    )
    bundle, feature_metadata, cache_hashes = _load_feature_bundle(
        config, source_sha256=train_sha, source_rows=source_rows
    )
    peak_rss = max(peak_rss, int(process.memory_info().rss))
    duration_rows = _duration_mapping(config)
    gauge.update(phase="precheck", progress_percent=10, detail="single fail-fast grammar audit")
    precheck_end = pd.Timestamp(
        config["source_contract"]["precheck_scope_end_inclusive"]
    ).tz_convert("UTC")
    precheck_frame = historical.loc[historical["__time_utc"] <= precheck_end].reset_index(drop=True)
    precheck_audit = build_grammar_audit(
        precheck_frame, cadence_minutes=10, duration_rows=duration_rows
    )
    precheck, precheck_passed = _precheck_metrics(precheck_frame, precheck_audit, config)
    del precheck_frame, precheck_audit
    gc.collect()

    historical_audit = build_grammar_audit(
        historical, cadence_minutes=10, duration_rows=duration_rows
    )
    split_counts: dict[str, dict[str, int]] = {}
    split_masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for block in config["historical_inner_blocks"]:
        fit, validation = chronological_split_masks(
            historical,
            historical_audit,
            fit_end_inclusive=block["fit_end_inclusive"],
            validation_start_inclusive=block["validation_start_inclusive"],
            validation_end_inclusive=block["validation_end_inclusive"],
            embargo_days=int(config["grammar"]["embargo_days"]),
        )
        if not fit.any() or not validation.any():
            raise RuntimeError(f"empty historical split: {block['name']}")
        split_masks[block["name"]] = (fit, validation)
        split_counts[block["name"]] = {
            "fit_rows": int(fit.sum()),
            "validation_rows": int(validation.sum()),
            "fit_positive_rows": int(historical.loc[fit, "label"].sum()),
            "validation_positive_rows": int(historical.loc[validation, "label"].sum()),
        }
    precheck["historical_split_counts"] = split_counts
    precheck["all_splits_super_event_disjoint"] = True
    precheck["outer_validation_or_scoring_count"] = 0
    precheck["maximum_truth_time_used"] = str(historical["__time_utc"].max())
    _write_json_fsync(output_dir / "precheck.json", precheck)
    if not precheck_passed:
        result = {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "decision": "NO_GO_PRECHECK",
            "precheck_passed": False,
            "inner_blocks_executed": 0,
            "outer_validation_or_scoring_count": 0,
            "test_prediction_count": 0,
            "submission_generation_count": 0,
            "upload_count": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss,
        }
        _write_json_fsync(output_dir / "result.json", result)
        gauge.update(
            phase="complete",
            progress_percent=100,
            detail="fail-fast precheck did not pass",
            eta_seconds=0,
            terminal=True,
        )
        return 2

    decoder_section = config["decoder"]
    decoder_config = DecoderConfig(
        duration_rows=duration_rows,
        beam_width=int(decoder_section["beam_width"]),
        maximum_concurrent_events=int(decoder_section["maximum_concurrent_events"]),
        maximum_start_candidates_per_row=int(decoder_section["maximum_start_candidates_per_row"]),
        start_penalty=float(decoder_section["start_penalty"]),
        overlap_penalty=float(decoder_section["overlap_penalty"]),
        duplicate_type_start_penalty=float(decoder_section["duplicate_type_start_penalty"]),
        probability_clip=float(decoder_section["probability_clip"]),
    )
    unary_section = config["unary_model"]
    parameters = dict(unary_section["parameters"])
    threads = int(unary_section["threads"])
    seed = int(unary_section["seed"])
    threshold = float(config["rowwise_control"]["threshold"])
    blocks: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {
        "truth": [],
        "control": [],
        "decoder": [],
        "membership": [],
    }
    completed_folds = 0
    for fold_index, block in enumerate(config["historical_inner_blocks"]):
        block_started = time.perf_counter()
        fit_mask, validation_mask = split_masks[block["name"]]
        fit_hist_positions = np.flatnonzero(fit_mask)
        validation_hist_positions = np.flatnonzero(validation_mask)
        fit_source_positions = historical.loc[fit_mask, "__source_position"].to_numpy(
            dtype=np.int64
        )
        validation_source_positions = historical.loc[validation_mask, "__source_position"].to_numpy(
            dtype=np.int64
        )
        encoder = TabularEncoder().fit(bundle, fit_source_positions)
        x_fit = encoder.transform(bundle, fit_source_positions)
        x_validation = encoder.transform(bundle, validation_source_positions)
        membership_fit = historical_audit.token_counts[fit_hist_positions]
        membership_validation = historical_audit.token_counts[validation_hist_positions]
        probabilities = np.zeros((len(validation_hist_positions), len(ANOMALY_TYPES)), dtype=float)
        head_summaries: dict[str, Any] = {}
        for type_index, anomaly_type in enumerate(ANOMALY_TYPES):
            y_fit = (membership_fit[:, type_index] > 0).astype(np.int8)
            if y_fit.min() == y_fit.max():
                raise RuntimeError(f"{block['name']} {anomaly_type} unary target is degenerate")
            head_started = time.perf_counter()
            model = make_tabular_classifier(
                "xgboost",
                seed=seed + type_index,
                n_jobs=threads,
                parameters=parameters,
            )
            model.fit(x_fit, y_fit, sample_weight=_sample_weights(y_fit))
            probabilities[:, type_index] = model.predict_proba(x_validation)[:, 1]
            importance = model.feature_importances_
            top_indices = np.argsort(-importance, kind="stable")[:10]
            head_summaries[anomaly_type] = {
                "fit_positive_rows": int(y_fit.sum()),
                "fit_positive_fraction": float(y_fit.mean()),
                "elapsed_seconds": time.perf_counter() - head_started,
                "top_feature_importance": [
                    {
                        "feature": encoder.feature_columns[int(index)],
                        "importance": float(importance[int(index)]),
                    }
                    for index in top_indices
                ],
            }
            del model
            gc.collect()
            peak_rss = max(peak_rss, int(process.memory_info().rss))
            model_step = fold_index * len(ANOMALY_TYPES) + type_index + 1
            gauge.update(
                phase="unary_fit",
                progress_percent=12 + 58 * model_step / 15,
                detail=f"{block['name']} unary head {type_index + 1}/5",
            )
        if (
            not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
        ):
            raise RuntimeError("unary probability contract failed")
        control = rowwise_union(probabilities, threshold=threshold)
        validation_frame = historical.loc[validation_mask].reset_index(drop=True)
        gauge.update(
            phase="decode",
            progress_percent=72 + 7 * fold_index,
            detail=f"{block['name']} factorial beam/MAP",
        )
        decoder, typed_decoder, no_op = decode_frame(
            validation_frame,
            probabilities,
            decoder_config,
            threshold=threshold,
            cadence_minutes=10,
        )
        if typed_decoder.sum(axis=1).max(initial=0) > 2:
            raise RuntimeError("decoder exceeded maximum concurrency")
        truth = validation_frame["label"].to_numpy(dtype=np.int8)
        metrics = _block_metrics(
            validation_frame,
            membership_validation,
            truth,
            control,
            decoder,
            no_op,
        )
        blocks.append(
            {
                "name": block["name"],
                "fit_end_inclusive": block["fit_end_inclusive"],
                "validation_start_inclusive": block["validation_start_inclusive"],
                "validation_end_inclusive": block["validation_end_inclusive"],
                "fit_rows": len(fit_source_positions),
                "validation_rows": len(validation_source_positions),
                "unary_heads": head_summaries,
                "metrics": metrics,
                "elapsed_seconds": time.perf_counter() - block_started,
            }
        )
        pooled["truth"].append(truth)
        pooled["control"].append(control)
        pooled["decoder"].append(decoder)
        pooled["membership"].append(membership_validation)
        completed_folds += 1
        elapsed = time.perf_counter() - started
        remaining = (elapsed / completed_folds) * (3 - completed_folds)
        gauge.update(
            phase="inner_blocks",
            progress_percent=79 + 6 * completed_folds,
            detail=f"historical blocks complete {completed_folds}/3",
            eta_seconds=remaining,
        )
        del x_fit, x_validation, probabilities, validation_frame, typed_decoder
        gc.collect()
        peak_rss = max(peak_rss, int(process.memory_info().rss))

    aggregate = _aggregate_and_decide(blocks, pooled, config)
    inner_result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "comparison": "factorial_decoder_vs_same_unary_rowwise_union",
        "blocks": blocks,
        "aggregate": aggregate,
        "grammar": {
            "type_order": list(ANOMALY_TYPES),
            "duration_rows_inclusive": {name: list(duration_rows[name]) for name in ANOMALY_TYPES},
            "maximum_concurrent_events": 2,
            "embargo_days": 8,
            "raw_order_and_multiplicity_preserved": True,
        },
        "outer_validation_or_scoring_count": 0,
        "test_prediction_count": 0,
        "submission_generation_count": 0,
        "upload_count": 0,
    }
    _write_json_fsync(output_dir / "inner_result.json", inner_result)
    peak_rss = max(peak_rss, int(process.memory_info().rss))
    inputs = {
        "train.csv": {"sha256": train_sha, "rows": source_rows},
        "README.md": {"sha256": _sha256(readme_path)},
        **{name: {"sha256": digest} for name, digest in cache_hashes.items()},
    }
    artifacts = {
        "attempt.lock": _sha256(attempt_lock),
        "precheck.json": _sha256(output_dir / "precheck.json"),
        "inner_result.json": _sha256(output_dir / "inner_result.json"),
    }
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "decision": aggregate["decision"],
        "precheck_passed": True,
        "inner_blocks_executed": 3,
        "aggregate": aggregate,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "peak_rss_gib": peak_rss / 1024**3,
        "implementation_sha256": implementation,
        "inputs": inputs,
        "artifacts_before_result": artifacts,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "feature_count": len(feature_metadata["feature_columns"]),
            "threads": threads,
        },
        "outer_validation_or_scoring_count": 0,
        "test_prediction_count": 0,
        "submission_generation_count": 0,
        "upload_count": 0,
        "existing_frozen_artifacts_modified": False,
    }
    _write_json_fsync(output_dir / "result.json", result)
    manifest = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "result_sha256": _sha256(output_dir / "result.json"),
        "artifact_sha256": {
            **artifacts,
            "result.json": _sha256(output_dir / "result.json"),
        },
        "implementation_sha256": implementation,
        "decision": aggregate["decision"],
        "generated_at_kst": datetime.now().astimezone().isoformat(),
    }
    _write_json_fsync(output_dir / "manifest.json", manifest)
    gauge.update(
        phase="complete",
        progress_percent=100,
        detail=aggregate["decision"],
        eta_seconds=0,
        terminal=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one preregistered P1 historical-inner typed semi-Markov experiment."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    arguments = parser.parse_args()
    return run(arguments.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
