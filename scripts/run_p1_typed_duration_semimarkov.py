"""Dry-check and run P1 typed duration semi-Markov historical inner v2."""

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
from p1_qc.typed_duration_semimarkov import (  # noqa: E402
    DurationDecoderConfig,
    decode_independent_types,
    same_unary_control,
)
from p1_qc.typed_factorial_semimarkov import (  # noqa: E402
    ANOMALY_TYPES,
    GrammarAudit,
    binary_counts,
    build_grammar_audit,
    chronological_split_masks,
    duration_is_decomposable,
    recall_by_type,
    station_layer_f1,
)

EXPERIMENT_ID = "p1_typed_duration_semimarkov_v2"
CANONICAL_CONFIG = PROJECT_ROOT / "configs" / "experiments" / "p1_typed_duration_semimarkov_v2.json"
HELPER = PROJECT_ROOT / "src" / "p1_qc" / "typed_duration_semimarkov.py"
GRAMMAR_HELPER = PROJECT_ROOT / "src" / "p1_qc" / "typed_factorial_semimarkov.py"
RUNNER = Path(__file__).resolve()
TESTS = PROJECT_ROOT / "tests" / "test_typed_duration_semimarkov.py"
HISTORICAL_HARD_CUTOFF = pd.Timestamp("2025-03-01T23:50:00+09:00").tz_convert("UTC")


def _sha256(path: Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def _write_json_fsync(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    with path.open("r+b") as handle:
        os.fsync(handle.fileno())
    if json.loads(path.read_text(encoding="utf-8")) != json.loads(payload):
        raise RuntimeError(f"JSON reload mismatch: {path.name}")


def _create_exclusive(path: Path, value: dict[str, Any]) -> None:
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
        raise RuntimeError(f"exclusive receipt reload mismatch: {path.name}")


def _implementation_hashes() -> dict[str, str]:
    return {
        _relative(CANONICAL_CONFIG): _sha256(CANONICAL_CONFIG),
        _relative(HELPER): _sha256(HELPER),
        _relative(GRAMMAR_HELPER): _sha256(GRAMMAR_HELPER),
        _relative(RUNNER): _sha256(RUNNER),
        _relative(TESTS): _sha256(TESTS),
    }


def _read_config() -> dict[str, Any]:
    config = json.loads(CANONICAL_CONFIG.read_text(encoding="utf-8"))
    if config.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("canonical experiment ID mismatch")
    forbidden = (
        "outer_validation_or_scoring",
        "test_read_or_prediction",
        "submission_generation_or_mutation",
        "upload",
        "existing_file_mutation",
        "hyperparameter_threshold_or_penalty_search",
    )
    if any(bool(config["authorization"].get(key)) for key in forbidden):
        raise RuntimeError("canonical authorization enables a forbidden operation")
    disclosure = config["adaptive_research_disclosure"]
    if not disclosure.get("adaptive") or disclosure.get("independent_confirmation"):
        raise RuntimeError("adaptive research disclosure is incomplete")
    grammar = config["grammar"]
    if any(
        bool(grammar[key])
        for key in (
            "overlap_interaction_modeled",
            "raw_token_order_modeled",
            "raw_token_multiplicity_modeled",
        )
    ):
        raise RuntimeError("v2 must remain duration-only")
    if not config["decoder"].get("same_type_restart_requires_one_normal_row"):
        raise RuntimeError("same-type restart separator contract changed")
    if grammar["embargo_days"] != 8 or len(config["historical_inner_blocks"]) != 2:
        raise RuntimeError("late two-block split contract changed")
    if (
        pd.Timestamp(config["source_contract"]["historical_truth_cutoff_inclusive"]).tz_convert(
            "UTC"
        )
        != HISTORICAL_HARD_CUTOFF
    ):
        raise RuntimeError("historical cutoff changed")
    return config


def _paths(config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    artifacts = config["artifacts"]
    return (
        PROJECT_ROOT / artifacts["output_dir"],
        PROJECT_ROOT / artifacts["status_json"],
        PROJECT_ROOT / artifacts["dry_receipt"],
        PROJECT_ROOT / artifacts["attempt_lock"],
    )


class Gauge:
    def __init__(self, path: Path, started: float) -> None:
        self.path = path
        self.started = started

    def update(
        self,
        phase: str,
        progress: float,
        detail: str,
        *,
        eta_seconds: float | None = None,
        terminal: bool = False,
        actual_started: bool = False,
    ) -> None:
        _write_json_fsync(
            self.path,
            {
                "schema_version": "1.0",
                "experiment_id": EXPERIMENT_ID,
                "updated_at_kst": datetime.now().astimezone().isoformat(),
                "phase": phase,
                "detail": detail,
                "progress_percent": round(float(progress), 2),
                "elapsed_seconds": round(time.perf_counter() - self.started, 3),
                "eta_seconds": None if eta_seconds is None else round(float(eta_seconds), 3),
                "actual_started": actual_started,
                "historical_inner_only": True,
                "outer_score_count": 0,
                "test_prediction_count": 0,
                "submission_count": 0,
                "upload_count": 0,
                "terminal": terminal,
            },
        )


def _duration_rows(config: dict[str, Any]) -> dict[str, tuple[int, int]]:
    values = config["grammar"]["duration_rows_inclusive"]
    return {name: (int(values[name][0]), int(values[name][1])) for name in ANOMALY_TYPES}


def _source_paths(config: dict[str, Any], data_dir: Path) -> tuple[Path, Path, Path, Path]:
    data_dir = data_dir.expanduser().resolve(strict=True)
    train = data_dir / config["source_contract"]["train_file"]
    readme = data_dir / "README.md"
    metadata = PROJECT_ROOT / config["source_contract"]["feature_cache_metadata"]
    parquet = PROJECT_ROOT / config["source_contract"]["feature_cache_parquet"]
    for path in (train, readme, metadata, parquet):
        if not path.is_file():
            raise FileNotFoundError(f"required immutable input is absent: {path.name}")
    return train, readme, metadata, parquet


def dry_run(data_dir: Path) -> int:
    started = time.perf_counter()
    config = _read_config()
    output, status, dry_receipt, attempt = _paths(config)
    gauge = Gauge(status, started)
    if dry_receipt.exists() or attempt.exists() or (output / "result.json").exists():
        raise FileExistsError("dry or actual generation already exists")
    train, readme, metadata_path, parquet_path = _source_paths(config, data_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    train_sha = _sha256(train)
    parquet_sha = _sha256(parquet_path)
    if metadata.get("source_sha256") != train_sha:
        raise RuntimeError("feature cache source SHA mismatch")
    if metadata.get("parquet_sha256") != parquet_sha:
        raise RuntimeError("feature cache parquet SHA mismatch")
    implementation = _implementation_hashes()
    receipt = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "created_at_kst": datetime.now().astimezone().isoformat(),
        "implementation_sha256": implementation,
        "input_sha256": {
            "train.csv": train_sha,
            "README.md": _sha256(readme),
            _relative(metadata_path): _sha256(metadata_path),
            _relative(parquet_path): parquet_sha,
        },
        "registered_train_rows": int(metadata["rows"]),
        "registered_feature_count": len(metadata["feature_columns"]),
        "checks": {
            "adaptive_disclosure": True,
            "duration_only": True,
            "two_late_blocks": True,
            "eight_day_embargo": True,
            "fixed_model_threshold_and_penalty": True,
            "same_unary_control": True,
        },
        "actual_model_fit_count": 0,
        "historical_inner_score_count": 0,
        "outer_score_count": 0,
        "test_prediction_count": 0,
        "submission_count": 0,
        "upload_count": 0,
        "actual_ready": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _create_exclusive(dry_receipt, receipt)
    gauge.update("dry_complete", 0, "dry receipt sealed; actual not started")
    return 0


def _load_bundle(
    metadata_path: Path,
    parquet_path: Path,
    *,
    source_sha: str,
    source_rows: int,
) -> tuple[FeatureBundle, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("source_sha256") != source_sha or int(metadata.get("rows", -1)) != source_rows:
        raise RuntimeError("feature cache source contract mismatch")
    if _sha256(parquet_path) != metadata.get("parquet_sha256"):
        raise RuntimeError("feature cache parquet hash drift")
    feature_frame = pd.read_parquet(parquet_path)
    if len(feature_frame) != source_rows:
        raise RuntimeError("feature cache row count mismatch")
    return (
        FeatureBundle(
            feature_frame,
            tuple(metadata["feature_columns"]),
            tuple(metadata["categorical_columns"]),
        ),
        metadata,
    )


def _complete_runs(
    audit: GrammarAudit,
    mask: np.ndarray,
    duration_rows: dict[str, tuple[int, int]],
) -> dict[str, int]:
    counts = Counter()
    for run in audit.atomic_runs:
        if run.occurrence_rank != 1:
            continue
        if not (mask[run.start_position] and mask[run.stop_position]):
            continue
        minimum, maximum = duration_rows[run.anomaly_type]
        if duration_is_decomposable(run.length, minimum, maximum):
            counts[run.anomaly_type] += 1
    return {name: int(counts[name]) for name in ANOMALY_TYPES}


def _precheck(
    frame: pd.DataFrame,
    audit: GrammarAudit,
    splits: dict[str, tuple[np.ndarray, np.ndarray]],
    config: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    duration_rows = _duration_rows(config)
    contract = config["fail_fast_precheck"]
    blocks: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for block in config["historical_inner_blocks"]:
        name = block["name"]
        fit, validation = splits[name]
        fit_positive = fit & (frame["label"].to_numpy(dtype=np.int8) == 1)
        val_positive = validation & (frame["label"].to_numpy(dtype=np.int8) == 1)
        fit_runs = _complete_runs(audit, fit, duration_rows)
        val_runs = _complete_runs(audit, validation, duration_rows)
        fit_coverage = float(audit.duration_decomposable_positive_rows[fit_positive].mean())
        val_coverage = float(audit.duration_decomposable_positive_rows[val_positive].mean())
        local_checks = {
            "fit_all_five_types": all(value > 0 for value in fit_runs.values()),
            "validation_all_five_types": all(value > 0 for value in val_runs.values()),
            "minimum_fit_runs": min(fit_runs.values())
            >= int(contract["minimum_complete_legal_fit_runs_per_type"]),
            "minimum_validation_runs": min(val_runs.values())
            >= int(contract["minimum_complete_legal_validation_runs_per_type"]),
            "fit_duration_coverage": fit_coverage
            >= float(contract["minimum_fit_positive_row_duration_coverage"]),
            "validation_duration_coverage": val_coverage
            >= float(contract["minimum_validation_positive_row_duration_coverage"]),
        }
        blocks[name] = {
            "fit_rows": int(fit.sum()),
            "validation_rows": int(validation.sum()),
            "fit_positive_rows": int(fit_positive.sum()),
            "validation_positive_rows": int(val_positive.sum()),
            "fit_complete_legal_runs": fit_runs,
            "validation_complete_legal_runs": val_runs,
            "fit_duration_coverage": fit_coverage,
            "validation_duration_coverage": val_coverage,
            "checks": local_checks,
        }
        checks[name] = all(local_checks.values())
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "duration_only": True,
        "composite_or_order_sensitive_support_required": False,
        "blocks": blocks,
        "block_checks": checks,
        "all_splits_super_event_disjoint": True,
        "passed": all(checks.values()),
        "outer_score_count": 0,
        "test_prediction_count": 0,
        "submission_count": 0,
        "upload_count": 0,
    }
    return result, bool(result["passed"])


def _weights(target: np.ndarray) -> np.ndarray:
    positives = max(1, int(target.sum()))
    negatives = max(1, len(target) - positives)
    return np.where(target == 1, np.sqrt(negatives / positives), 1.0).astype(np.float32)


def _block_metrics(
    frame: pd.DataFrame,
    membership: np.ndarray,
    truth: np.ndarray,
    control: np.ndarray,
    candidate: np.ndarray,
    no_op: np.ndarray,
) -> dict[str, Any]:
    control_counts = binary_counts(truth, control)
    candidate_counts = binary_counts(truth, candidate)
    control_recall = recall_by_type(membership, control)
    candidate_recall = recall_by_type(membership, candidate)
    recall_delta = {
        name: (
            None
            if control_recall[name] is None or candidate_recall[name] is None
            else float(candidate_recall[name] - control_recall[name])
        )
        for name in ANOMALY_TYPES
    }
    control_group = station_layer_f1(frame, truth, control)
    candidate_group = station_layer_f1(frame, truth, candidate)
    group_delta = {key: float(candidate_group[key] - control_group[key]) for key in control_group}
    return {
        "rows": len(truth),
        "positive_rows": int(truth.sum()),
        "control": control_counts,
        "candidate": candidate_counts,
        "micro_f1_delta": float(candidate_counts["f1"] - control_counts["f1"]),
        "control_recall_by_type": control_recall,
        "candidate_recall_by_type": candidate_recall,
        "recall_delta_by_type": recall_delta,
        "normal_false_positive_relative_delta": float(
            (int(candidate_counts["fp"]) - int(control_counts["fp"]))
            / max(1, int(control_counts["fp"]))
        ),
        "station_layer_f1_delta": group_delta,
        "worst_station_layer_f1_delta": min(group_delta.values()) if group_delta else 0.0,
        "no_op_rows": int(no_op.any(axis=1).sum()),
        "no_op_share": float(no_op.any(axis=1).mean()),
        "no_op_cells_by_type": {
            name: int(no_op[:, index].sum()) for index, name in enumerate(ANOMALY_TYPES)
        },
    }


def _aggregate(
    block_results: list[dict[str, Any]],
    pooled: dict[str, list[np.ndarray]],
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = np.concatenate(pooled["truth"])
    control = np.concatenate(pooled["control"])
    candidate = np.concatenate(pooled["candidate"])
    membership = np.concatenate(pooled["membership"])
    control_counts = binary_counts(truth, control)
    candidate_counts = binary_counts(truth, candidate)
    control_recall = recall_by_type(membership, control)
    candidate_recall = recall_by_type(membership, candidate)
    if any(value is None for value in (*control_recall.values(), *candidate_recall.values())):
        raise RuntimeError("pooled validation lacks an anomaly type")
    recall_delta = {
        name: float(candidate_recall[name] - control_recall[name])  # type: ignore[operator]
        for name in ANOMALY_TYPES
    }
    fp_delta = float(
        (int(candidate_counts["fp"]) - int(control_counts["fp"]))
        / max(1, int(control_counts["fp"]))
    )
    group_deltas = [
        float(value)
        for block in block_results
        for value in block["metrics"]["station_layer_f1_delta"].values()
    ]
    gate = config["inner_decision_gate"]
    checks = {
        "strict_improvement_each_block": all(
            block["metrics"]["micro_f1_delta"] > 0 for block in block_results
        ),
        "minimum_improved_blocks": sum(
            block["metrics"]["micro_f1_delta"] > 0 for block in block_results
        )
        >= int(gate["minimum_improved_blocks"]),
        "minimum_pooled_micro_f1_delta": float(candidate_counts["f1"] - control_counts["f1"])
        >= float(gate["minimum_pooled_micro_f1_delta"]),
        "minimum_decoder_pooled_micro_f1": float(candidate_counts["f1"])
        >= float(gate["minimum_decoder_pooled_micro_f1"]),
        "offset_or_drift_recall_gain": max(recall_delta["offset"], recall_delta["drift"])
        >= float(gate["minimum_offset_or_drift_recall_gain"]),
        "spike_noise_flatline_recall_drop": min(
            recall_delta["spike"], recall_delta["noise"], recall_delta["flatline"]
        )
        >= -float(gate["maximum_spike_noise_flatline_recall_drop"]),
        "normal_false_positive_relative_increase": fp_delta
        <= float(gate["maximum_normal_false_positive_relative_increase"]),
        "worst_station_layer_f1_drop": (min(group_deltas) if group_deltas else 0.0)
        >= -float(gate["maximum_worst_station_layer_f1_drop"]),
    }
    passed = all(checks.values())
    return {
        "rows": len(truth),
        "positive_rows": int(truth.sum()),
        "control": control_counts,
        "candidate": candidate_counts,
        "micro_f1_delta": float(candidate_counts["f1"] - control_counts["f1"]),
        "improved_blocks": sum(block["metrics"]["micro_f1_delta"] > 0 for block in block_results),
        "control_recall_by_type": control_recall,
        "candidate_recall_by_type": candidate_recall,
        "recall_delta_by_type": recall_delta,
        "normal_false_positive_relative_delta": fp_delta,
        "worst_station_layer_f1_delta": min(group_deltas) if group_deltas else 0.0,
        "no_op_rows": int(sum(block["metrics"]["no_op_rows"] for block in block_results)),
        "no_op_share": float(
            sum(block["metrics"]["no_op_rows"] for block in block_results) / len(truth)
        ),
        "gate_checks": checks,
        "gate_passed": passed,
        "decision": gate["promotion_target"] if passed else gate["failure_target"],
    }


def actual_run(data_dir: Path) -> int:
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = int(process.memory_info().rss)
    config = _read_config()
    output, status, dry_receipt_path, attempt_path = _paths(config)
    gauge = Gauge(status, started)
    if not dry_receipt_path.is_file():
        raise FileNotFoundError("sealed dry receipt is required before actual")
    if attempt_path.exists() or (output / "result.json").exists():
        raise FileExistsError("the one-shot actual already exists")
    dry_receipt = json.loads(dry_receipt_path.read_text(encoding="utf-8"))
    implementation = _implementation_hashes()
    if dry_receipt.get("implementation_sha256") != implementation:
        raise RuntimeError("implementation drift after dry receipt")
    train, readme, metadata_path, parquet_path = _source_paths(config, data_dir)
    current_inputs = {
        "train.csv": _sha256(train),
        "README.md": _sha256(readme),
        _relative(metadata_path): _sha256(metadata_path),
        _relative(parquet_path): _sha256(parquet_path),
    }
    if dry_receipt.get("input_sha256") != current_inputs:
        raise RuntimeError("input drift after dry receipt")
    _create_exclusive(
        attempt_path,
        {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "created_at_kst": datetime.now().astimezone().isoformat(),
            "dry_receipt_sha256": _sha256(dry_receipt_path),
            "implementation_sha256": implementation,
            "historical_inner_only": True,
            "outer_score": False,
            "test_prediction": False,
            "submission": False,
            "upload": False,
        },
    )
    gauge.update("load", 2, "immutable train audit", actual_started=True)
    source = load_dataset(train, kind="train", audit=True, strict=True)
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
    del source, source_times
    gc.collect()
    peak_rss = max(peak_rss, int(process.memory_info().rss))
    gauge.update("features", 6, "load sealed offline feature cache", actual_started=True)
    bundle, feature_metadata = _load_bundle(
        metadata_path,
        parquet_path,
        source_sha=current_inputs["train.csv"],
        source_rows=source_rows,
    )
    peak_rss = max(peak_rss, int(process.memory_info().rss))
    duration_rows = _duration_rows(config)
    audit = build_grammar_audit(historical, duration_rows=duration_rows)
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for block in config["historical_inner_blocks"]:
        splits[block["name"]] = chronological_split_masks(
            historical,
            audit,
            fit_end_inclusive=block["fit_end_inclusive"],
            validation_start_inclusive=block["validation_start_inclusive"],
            validation_end_inclusive=block["validation_end_inclusive"],
            embargo_days=int(config["grammar"]["embargo_days"]),
        )
    gauge.update("precheck", 10, "duration-only structural precheck", actual_started=True)
    precheck, passed = _precheck(historical, audit, splits, config)
    _write_json_fsync(output / "precheck.json", precheck)
    if not passed:
        result = {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "decision": "NO_GO_PRECHECK",
            "inner_blocks_executed": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak_rss,
            "outer_score_count": 0,
            "test_prediction_count": 0,
            "submission_count": 0,
            "upload_count": 0,
        }
        _write_json_fsync(output / "result.json", result)
        gauge.update(
            "complete", 100, "NO_GO_PRECHECK", eta_seconds=0, terminal=True, actual_started=True
        )
        return 2

    decoder_config = DurationDecoderConfig(
        duration_rows=duration_rows,
        start_penalty=float(config["decoder"]["start_penalty"]),
        stop_penalty=float(config["decoder"]["stop_penalty"]),
        probability_clip=float(config["decoder"]["probability_clip"]),
    )
    unary = config["unary_model"]
    parameters = dict(unary["parameters"])
    threads = int(unary["threads"])
    seed = int(unary["seed"])
    threshold = float(config["rowwise_control"]["threshold"])
    blocks: list[dict[str, Any]] = []
    pooled: dict[str, list[np.ndarray]] = {
        "truth": [],
        "control": [],
        "candidate": [],
        "membership": [],
    }
    for block_index, block in enumerate(config["historical_inner_blocks"]):
        block_started = time.perf_counter()
        fit_mask, validation_mask = splits[block["name"]]
        fit_hist = np.flatnonzero(fit_mask)
        val_hist = np.flatnonzero(validation_mask)
        fit_source = historical.loc[fit_mask, "__source_position"].to_numpy(dtype=np.int64)
        val_source = historical.loc[validation_mask, "__source_position"].to_numpy(dtype=np.int64)
        encoder = TabularEncoder().fit(bundle, fit_source)
        x_fit = encoder.transform(bundle, fit_source)
        x_val = encoder.transform(bundle, val_source)
        fit_membership = audit.token_counts[fit_hist]
        val_membership = audit.token_counts[val_hist]
        probabilities = np.zeros((len(val_hist), len(ANOMALY_TYPES)), dtype=float)
        head_summaries: dict[str, Any] = {}
        for type_index, anomaly_type in enumerate(ANOMALY_TYPES):
            target = (fit_membership[:, type_index] > 0).astype(np.int8)
            if target.min() == target.max():
                raise RuntimeError(f"degenerate unary target: {block['name']} {anomaly_type}")
            head_started = time.perf_counter()
            model = make_tabular_classifier(
                "xgboost",
                seed=seed + type_index,
                n_jobs=threads,
                parameters=parameters,
            )
            model.fit(x_fit, target, sample_weight=_weights(target))
            probabilities[:, type_index] = model.predict_proba(x_val)[:, 1]
            importance = model.feature_importances_
            top = np.argsort(-importance, kind="stable")[:10]
            head_summaries[anomaly_type] = {
                "fit_positive_rows": int(target.sum()),
                "elapsed_seconds": time.perf_counter() - head_started,
                "top_feature_importance": [
                    {
                        "feature": encoder.feature_columns[int(index)],
                        "importance": float(importance[int(index)]),
                    }
                    for index in top
                ],
            }
            del model
            gc.collect()
            peak_rss = max(peak_rss, int(process.memory_info().rss))
            step = block_index * 5 + type_index + 1
            gauge.update(
                "unary_fit",
                12 + 60 * step / 10,
                f"{block['name']} fixed unary {type_index + 1}/5",
                actual_started=True,
            )
        if (
            not np.isfinite(probabilities).all()
            or not ((probabilities >= 0) & (probabilities <= 1)).all()
        ):
            raise RuntimeError("unary probability contract failed")
        control = same_unary_control(probabilities, threshold=threshold)
        validation_frame = historical.loc[validation_mask].reset_index(drop=True)
        gauge.update(
            "decode",
            75 + 8 * block_index,
            f"{block['name']} five exact duration chains",
            actual_started=True,
        )
        candidate, typed_candidate, no_op = decode_independent_types(
            validation_frame,
            probabilities,
            decoder_config,
            threshold=threshold,
        )
        truth = validation_frame["label"].to_numpy(dtype=np.int8)
        metrics = _block_metrics(
            validation_frame,
            val_membership,
            truth,
            control,
            candidate,
            no_op,
        )
        blocks.append(
            {
                "name": block["name"],
                "fit_end_inclusive": block["fit_end_inclusive"],
                "validation_start_inclusive": block["validation_start_inclusive"],
                "validation_end_inclusive": block["validation_end_inclusive"],
                "fit_rows": len(fit_source),
                "validation_rows": len(val_source),
                "unary_heads": head_summaries,
                "metrics": metrics,
                "elapsed_seconds": time.perf_counter() - block_started,
            }
        )
        pooled["truth"].append(truth)
        pooled["control"].append(control)
        pooled["candidate"].append(candidate)
        pooled["membership"].append(val_membership)
        elapsed = time.perf_counter() - started
        remaining = elapsed / (block_index + 1) * (1 - block_index)
        gauge.update(
            "inner_blocks",
            88 + 5 * (block_index + 1),
            f"historical blocks complete {block_index + 1}/2",
            eta_seconds=max(0.0, remaining),
            actual_started=True,
        )
        del x_fit, x_val, probabilities, validation_frame, typed_candidate
        gc.collect()
        peak_rss = max(peak_rss, int(process.memory_info().rss))

    aggregate = _aggregate(blocks, pooled, config)
    inner_result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "adaptive_research": True,
        "duration_only": True,
        "overlap_order_or_multiplicity_modeled": False,
        "comparison": "independent_duration_decoder_union_vs_same_unary_rowwise_union",
        "blocks": blocks,
        "aggregate": aggregate,
        "outer_score_count": 0,
        "test_prediction_count": 0,
        "submission_count": 0,
        "upload_count": 0,
    }
    _write_json_fsync(output / "inner_result.json", inner_result)
    result = {
        "schema_version": "1.0",
        "experiment_id": EXPERIMENT_ID,
        "decision": aggregate["decision"],
        "precheck_passed": True,
        "inner_blocks_executed": 2,
        "aggregate": aggregate,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_rss,
        "peak_rss_gib": peak_rss / 1024**3,
        "implementation_sha256": implementation,
        "input_sha256": current_inputs,
        "dry_receipt_sha256": _sha256(dry_receipt_path),
        "attempt_lock_sha256": _sha256(attempt_path),
        "precheck_sha256": _sha256(output / "precheck.json"),
        "inner_result_sha256": _sha256(output / "inner_result.json"),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "threads": threads,
            "feature_count": len(feature_metadata["feature_columns"]),
        },
        "outer_score_count": 0,
        "test_prediction_count": 0,
        "submission_count": 0,
        "upload_count": 0,
        "existing_frozen_artifacts_modified": False,
    }
    _write_json_fsync(output / "result.json", result)
    _write_json_fsync(
        output / "manifest.json",
        {
            "schema_version": "1.0",
            "experiment_id": EXPERIMENT_ID,
            "decision": aggregate["decision"],
            "result_sha256": _sha256(output / "result.json"),
            "artifact_sha256": {
                "dry_receipt.json": _sha256(dry_receipt_path),
                "attempt.lock": _sha256(attempt_path),
                "precheck.json": _sha256(output / "precheck.json"),
                "inner_result.json": _sha256(output / "inner_result.json"),
                "result.json": _sha256(output / "result.json"),
            },
            "implementation_sha256": implementation,
            "generated_at_kst": datetime.now().astimezone().isoformat(),
        },
    )
    gauge.update(
        "complete",
        100,
        aggregate["decision"],
        eta_seconds=0,
        terminal=True,
        actual_started=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="P1 duration-only historical inner v2")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    return dry_run(arguments.data_dir) if arguments.dry_run else actual_run(arguments.data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
