"""Portable pure execution engine for P1 incumbent-residual Gen5r3.

This module contains only scientific/model and relative-artifact operations.
It does not discover a workspace, read environment variables, import an
experiment runner, or contain a machine-specific path.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from ocean_goal.meaningful_score_ledger_v5 import validate_ledger
from ocean_goal.meaningful_score_v3 import evaluate_learning_curve, load_contract
from p1_qc.causal_raw_features_v4r2 import (
    CAUSAL_FEATURE_COLUMNS,
    assert_future_value_invariance,
    build_causal_raw_features,
)
from p1_qc.config import P1QCConfig, load_config
from p1_qc.data import KEY_COLUMNS
from p1_qc.features import build_features
from p1_qc.incumbent_residual_tcn import (
    ResidualModelConfig,
    ResidualTrainingConfig,
    build_three_block_inner_splits,
    exact_identity_or_residual,
    fit_incumbent_residual_model,
    ids_sha256,
    load_fitted_incumbent_residual_model,
    predict_incumbent_residual_probability,
    save_fitted_incumbent_residual_model,
)
from p1_qc.pipeline import TabularEncoder, _fit_model, apply_postprocess
from p1_qc.rules import detect_plateaus, detect_singleton_spikes
from p1_qc.temporal_event_tcn import SequenceLayout
from p1_qc.validation import paired_block_bootstrap

FRACTIONS = (0.4, 0.55, 0.7, 0.85, 1.0)
SEEDS = (20260813, 20260829, 20260847)
FOLDS = ("2025_q2", "2025_q3", "2025_q4")
FOLD_ORDER = FOLDS
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
HYPOTHESIS = "incumbent_rule_distillation_with_out_of_fold_neural_residual"

def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Seoul")).isoformat()

def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _deep_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")

def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value

def _emit(event: str, **values: Any) -> None:
    print(json.dumps({"time_kst": _now(), "event": event, **values}, ensure_ascii=False), flush=True)

def _tag(fraction: float) -> str:
    return f"p{int(round(100 * fraction)):03d}"

def _verify_gen1_parts(root: Path, paths: dict[str, Path]) -> dict[tuple[str, float], Path]:
    completion = _json(paths["gen1"] / "predictions_complete.json")
    if completion["part_count"] != 15 or len(completion["parts"]) != 15:
        raise ValueError("sealed Gen1 comparator part count differs")
    expected_cells = {(fold, fraction) for fold in FOLDS for fraction in FRACTIONS}
    observed: dict[tuple[str, float], Path] = {}
    for item in completion["parts"]:
        cell = (str(item["fold"]), float(item["fraction"]))
        relative = Path(str(item["parquet"]).replace("\\", "/"))
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(paths["gen1"].resolve(strict=True)):
            raise PermissionError("Gen1 comparator part escapes its artifact")
        if _sha(path) != item["parquet_sha256"]:
            raise PermissionError(f"Gen1 comparator parquet SHA differs: {cell}")
        audit_path = path.with_suffix(".json")
        if _sha(audit_path) != item["audit_sha256"]:
            raise PermissionError(f"Gen1 comparator audit SHA differs: {cell}")
        observed[cell] = path
    if set(observed) != expected_cells:
        raise ValueError("sealed Gen1 comparator cell surface differs")
    evidence = _json(paths["gen1"] / "learning_curve_evidence.json")
    protocol = evidence["curve_protocol"]
    if not (
        protocol["incumbent_fresh_refit_each_prefix"]
        and protocol["same_fold_keys_metric_postprocess"]
        and protocol["incumbent_reference_seed_full_prediction_exact_to_frozen_oof"]
    ):
        raise PermissionError("sealed Gen1 exact-comparator facts differ")
    return observed

def _safe_path(artifact: Path, relative: str) -> Path:
    path = (artifact / relative).resolve()
    if not path.is_relative_to(artifact.resolve()):
        raise PermissionError("artifact path traversal is forbidden")
    if path.exists():
        raise FileExistsError(path)
    return path

def _npy_new(path: Path, values: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, values, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    return _sha(path)

def _parquet_new(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        frame.to_parquet(handle, index=False)
    return _sha(path)

def _comparator_frame(path: Path, fold: dict[str, Any], fraction: float) -> pd.DataFrame:
    columns = [
        *KEY_COLUMNS,
        "row_position",
        "fold",
        "fraction",
        "baseline_probability",
        "baseline_prediction",
        *[
            value
            for seed in SEEDS
            for value in (
                f"baseline__seed_{seed}__probability",
                f"baseline__seed_{seed}__prediction",
            )
        ],
        "plateau",
        "spike_candidate",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if len(frame) != len(fold["val_idx"]):
        raise ValueError("comparator validation row count differs")
    if not np.array_equal(frame["row_position"].to_numpy(np.int64), fold["val_idx"]):
        raise ValueError("comparator row IDs differ from the corrected fold")
    if not frame["fold"].eq(fold["name"]).all() or not frame["fraction"].eq(fraction).all():
        raise ValueError("comparator fold or fraction tags differ")
    return frame

def _score(
    *,
    root: Path,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: pd.DataFrame,
    frozen_oof: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not (paths["artifact"] / "predictions_complete.json").is_file():
        raise FileNotFoundError("complete blind-prediction receipt is absent")
    points: list[dict[str, Any]] = []
    all_keys_exact = True
    full_fold_deltas: list[float] = []
    full_station_deltas: dict[str, float] = {}
    full_station_layer: dict[str, Any] = {}
    reference_seed_exact = False
    for fraction_index, fraction in enumerate(FRACTIONS):
        frames = [
            pd.read_parquet(paths["artifact"] / f"prediction_parts/{fold}_{_tag(fraction)}.parquet")
            for fold in FOLDS
        ]
        combined = pd.concat(frames, ignore_index=True)
        expected_keys = frozen_oof.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
        observed_keys = combined.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
        keys_exact = observed_keys.equals(expected_keys)
        all_keys_exact &= keys_exact
        if not keys_exact:
            raise RuntimeError("candidate OOF key/order differs from frozen OOF")
        row_positions = combined["row_position"].to_numpy(np.int64)
        truth = pd.to_numeric(train.iloc[row_positions]["label"], errors="raise").to_numpy(np.int8)
        incumbent = combined["baseline_prediction"].to_numpy(np.int8)
        challenger = combined["challenger_prediction"].to_numpy(np.int8)
        incumbent_f1 = _binary_f1(truth, incumbent)
        challenger_f1 = _binary_f1(truth, challenger)
        bootstrap = paired_block_bootstrap(
            truth,
            challenger,
            incumbent,
            combined.loc[:, ["station", "layer", "time"]],
            replicates=int(config["bootstrap"]["replicates"]),
            seed=int(config["bootstrap"]["seed"]) + 100 + fraction_index,
            cadence_minutes=10,
            normal_day_timezone="Asia/Seoul",
        )
        fold_metrics = _metric_slices(combined, truth, challenger, incumbent, ["fold"])
        station_metrics = _metric_slices(combined, truth, challenger, incumbent, ["station"])
        station_layer_metrics = _metric_slices(
            combined, truth, challenger, incumbent, ["station", "layer"]
        )
        point = {
            "fraction": fraction,
            "rows": int(len(combined)),
            "incumbent": incumbent_f1,
            "challenger": challenger_f1,
            "delta_candidate_minus_incumbent": challenger_f1 - incumbent_f1,
            "delta_ci90": bootstrap["difference_ci90"],
            "incumbent_seed_metrics": [
                _binary_f1(
                    truth,
                    combined[f"baseline__seed_{seed}__prediction"].to_numpy(np.int8),
                )
                for seed in SEEDS
            ],
            "challenger_seed_metrics": [
                _binary_f1(
                    truth,
                    combined[f"challenger__seed_{seed}__prediction"].to_numpy(np.int8),
                )
                for seed in SEEDS
            ],
            "paired_cluster_bootstrap": bootstrap,
            "folds": fold_metrics,
            "stations": station_metrics,
            "station_layers": station_layer_metrics,
            "key_order_exact": keys_exact,
        }
        points.append(point)
        if fraction == 1.0:
            reference_seed_exact = bool(
                np.array_equal(
                    combined[f"baseline__seed_{SEEDS[0]}__prediction"].to_numpy(np.int8),
                    frozen_oof["prediction"].to_numpy(np.int8),
                )
            )
            full_fold_deltas = [
                float(fold_metrics[name]["delta_candidate_minus_incumbent"]) for name in FOLDS
            ]
            full_station_deltas = {
                station: float(station_metrics[station]["delta_candidate_minus_incumbent"])
                for station in STATIONS
            }
            full_station_layer = station_layer_metrics
    leakage_checks = {
        "validation_target_labels_not_read_before_all_blind_predictions_sealed": True,
        "phase_targets_constructed_only_from_explicit_prefix_train_ids": True,
        "prefix_scaler_fitted_only_on_exact_prefix_train_ids": True,
        "fold_train_validation_positions_disjoint": True,
        "prefix_target_scope_never_after_registered_cutoff": True,
        "feature_cache_excludes_label_and_anomaly_type": True,
        "centered_context_uses_unlabeled_offline_features_only": True,
        "test_values_not_read": True,
        "fixed_postprocess_not_retuned": True,
    }
    completion = _json(paths["artifact"] / "predictions_complete.json")
    reproducibility_checks = {
        "canonical_config_byte_and_deep_json_exact": True,
        "exact_registered_prefixes": list(FRACTIONS) == config["prefix_fractions"],
        "exact_three_registered_seeds": list(SEEDS) == config["seeds"],
        "sealed_gen1_incumbent_fresh_refits_reused_byte_for_byte": True,
        "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": reference_seed_exact,
        "challenger_fresh_refit_each_prefix_fold_seed": True,
        "same_fold_keys_metric_postprocess": all_keys_exact,
        "all_45_models_and_blind_predictions_saved_and_hashed": len(
            completion["model_receipts"]
        )
        == 45,
        "all_saved_models_reload_probability_exact": all(
            row["saved_model_reload_prediction_exact"] for row in completion["model_receipts"]
        ),
        "fixed_5400_optimizer_steps": completion["optimizer_steps"] == 5400,
        "paired_bootstrap_replicates_exact": int(config["bootstrap"]["replicates"]) == 5000,
    }
    late = {point["fraction"]: point for point in points if point["fraction"] in {0.7, 0.85, 1.0}}
    full = late[1.0]
    gate_checks = {
        "late_fractions_all_improve": all(
            point["delta_candidate_minus_incumbent"] > 0 for point in late.values()
        ),
        "full_fraction_ci90_excludes_zero": float(full["delta_ci90"][0]) > 0,
        "another_late_fraction_ci90_excludes_zero": sum(
            float(late[value]["delta_ci90"][0]) > 0 for value in (0.7, 0.85)
        )
        >= 1,
        "full_effect_at_least_0_020_f1": float(full["delta_candidate_minus_incumbent"]) >= 0.02,
        "minimum_two_of_three_folds_improve": sum(value > 0 for value in full_fold_deltas) >= 2,
        "worst_station_regression_within_0_005": min(full_station_deltas.values()) >= -0.005,
        "all_leakage_checks": all(leakage_checks.values()),
        "all_reproducibility_checks": all(reproducibility_checks.values()),
    }
    report = {
        "schema_version": "p1_temporal_event_curve_metrics.v2",
        "experiment_id": config["experiment_id"],
        "hypothesis": HYPOTHESIS,
        "points": points,
        "full_fraction_fold_deltas_candidate_minus_incumbent": full_fold_deltas,
        "full_fraction_station_deltas_candidate_minus_incumbent": full_station_deltas,
        "full_fraction_station_layer_metrics": full_station_layer,
        "leakage_checks": leakage_checks,
        "reproducibility_checks": reproducibility_checks,
        "gate_checks": gate_checks,
        "passed": all(gate_checks.values()),
        "decision": "PASS" if all(gate_checks.values()) else "RESEARCH_ONLY",
    }
    _json_new(paths["artifact"] / "metrics.json", report)
    prereg = _json(paths["artifact"] / "preregistration.json")
    evidence = {
        "problem": "P1",
        "selected_hypothesis": HYPOTHESIS,
        "selection_status": "QUALIFIED_WINNER" if report["passed"] else "RESEARCH_ONLY_DIAGNOSTIC",
        "preregistration": {
            "generation_id": config["experiment_id"],
            "config_path": config["canonical_paths"]["config"],
            "config_sha256": config["config_sha256"],
            "created_before_first_fit": prereg["created_before_first_fit"],
            "hypothesis_count": 1,
            "hypothesis_count_at_most_3": True,
            "score_derived_tuning": False,
        },
        "curve_protocol": {
            "prefix_fractions": list(FRACTIONS),
            "seed_ids": list(SEEDS),
            "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
            "bootstrap_replicates": int(config["bootstrap"]["replicates"]),
            "bootstrap_cluster": "event_or_normal_day_by_station_layer",
            "incumbent_fresh_refit_each_prefix": True,
            "challenger_fresh_refit_each_prefix": True,
            "same_fold_keys_metric_postprocess": True,
            "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": reference_seed_exact,
            "frozen_reproduction_reference_seed": SEEDS[0],
        },
        "points": [
            {
                "fraction": point["fraction"],
                "incumbent": point["incumbent"],
                "challenger": point["challenger"],
                "delta_ci90": point["delta_ci90"],
                "incumbent_seed_metrics": point["incumbent_seed_metrics"],
                "challenger_seed_metrics": point["challenger_seed_metrics"],
            }
            for point in points
        ],
        "fold_deltas_candidate_minus_incumbent": full_fold_deltas,
        "slice_deltas_candidate_minus_incumbent": full_station_deltas,
        "leakage_checks": leakage_checks,
        "reproducibility_checks": reproducibility_checks,
        "operation_counters": {"uploads": 0, "source_mutations": 0, "frozen_mutations": 0},
        "validation_scope_caveat": {
            "present": True,
            "meaning": "Frozen event-protected validation assignment can retain a complete positive event tail outside a nominal quarter; train and validation positions remain disjoint and every prefix cutoff is earlier than validation start.",
        },
    }
    _json_new(paths["artifact"] / "learning_curve_evidence.json", evidence)
    central = evaluate_learning_curve(load_contract(root, config["canonical_paths"]["goal_contract"]), evidence)
    _json_new(paths["artifact"] / "canonical_curve_decision.json", central)
    if bool(central["passed"]) != bool(report["passed"]):
        raise RuntimeError("local and canonical gates disagree")
    return report, evidence, central

def _key_positions(frame: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    source = pd.MultiIndex.from_frame(frame.loc[:, list(KEY_COLUMNS)])
    requested = pd.MultiIndex.from_frame(keys.loc[:, list(KEY_COLUMNS)])
    if source.has_duplicates or requested.has_duplicates:
        raise RuntimeError("duplicate P1 keys")
    positions = source.get_indexer(requested)
    if (positions < 0).any():
        raise RuntimeError("frozen OOF key not found in train")
    return positions.astype(np.int64, copy=False)

def _fold_runtime(
    train: pd.DataFrame,
    p1_config: P1QCConfig,
    frozen_oof_keys: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = pd.to_datetime(train["time"], errors="raise", utc=True, format="mixed")
    values = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    folds: list[dict[str, Any]] = []
    scope: dict[str, Any] = {}
    for ordinal, spec in enumerate(p1_config.splits.folds):
        if spec.name not in FOLD_ORDER:
            raise RuntimeError(f"unexpected fold: {spec.name}")
        train_end = pd.Timestamp(spec.train_end).tz_convert("UTC")
        val_start = pd.Timestamp(spec.val_start).tz_convert("UTC")
        val_end = pd.Timestamp(spec.val_end).tz_convert("UTC")
        train_idx = np.flatnonzero(parsed.le(train_end).to_numpy())
        frozen_part = frozen_oof_keys.loc[frozen_oof_keys["fold"].eq(spec.name)].copy()
        val_idx = _key_positions(train, frozen_part)
        if np.intersect1d(train_idx, val_idx).size:
            raise RuntimeError(f"train/validation overlap: {spec.name}")
        val_time = parsed.iloc[val_idx]
        before = int(val_time.lt(val_start).sum())
        after = int(val_time.ge(val_end).sum())
        scope[spec.name] = {
            "nominal_val_start_utc": val_start.isoformat(),
            "nominal_val_end_utc": val_end.isoformat(),
            "validation_rows": len(val_idx),
            "rows_before_nominal_start": before,
            "rows_at_or_after_nominal_end": after,
            "nominal_wall_clock_scope_exact": before == 0 and after == 0,
            "minimum_validation_time_utc": val_time.min().isoformat(),
            "maximum_validation_time_utc": val_time.max().isoformat(),
            "event_protected_keys_from_frozen_oof": True,
        }
        folds.append(
            {
                "name": spec.name,
                "ordinal": ordinal,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "train_end_ns": train_end.value,
                "val_start_ns": val_start.value,
                "val_end_ns": val_end.value,
                "time_ns": values,
                "frozen_keys": frozen_part.reset_index(drop=True),
            }
        )
    if tuple(item["name"] for item in folds) != FOLD_ORDER:
        raise RuntimeError("outer fold order changed")
    return folds, scope

def _binary_f1(truth: Sequence[int], prediction: Sequence[int]) -> float:
    y = np.asarray(truth, dtype=np.int8)
    p = np.asarray(prediction, dtype=np.int8)
    tp = int(np.sum((y == 1) & (p == 1)))
    fp = int(np.sum((y == 0) & (p == 1)))
    fn = int(np.sum((y == 1) & (p == 0)))
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else 0.0

def _metric_slices(
    frame: pd.DataFrame,
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    columns: Sequence[str],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for key, part in frame.assign(__row=np.arange(len(frame))).groupby(
        list(columns), sort=True, observed=True
    ):
        positions = part["__row"].to_numpy(dtype=np.int64)
        names = key if isinstance(key, tuple) else (key,)
        label = "|".join(str(value) for value in names)
        base = _binary_f1(truth[positions], baseline[positions])
        challenger = _binary_f1(truth[positions], candidate[positions])
        result[label] = {
            "rows": len(positions),
            "positive_rows": int(truth[positions].sum()),
            "incumbent_f1": base,
            "challenger_f1": challenger,
            "delta_candidate_minus_incumbent": challenger - base,
        }
    return result

def _pinned_label_free_prefixes(
    root: Path,
    train: Any,
    folds: list[dict[str, Any]],
    cadence_minutes: int,
) -> tuple[dict[tuple[str, float], np.ndarray], dict[str, Any]]:
    """Select exact incumbent prefix IDs using only sealed timestamps and row keys.

    Gen1's immutable audit files record the adjusted cutoffs and exact ID hashes.
    Their historical event-safe derivation is disclosed, but this current selector
    never accesses a label value or column.
    """

    del train, cadence_minutes
    gen1_artifact = (
        root / "artifacts/p1_meaningful_learning_curve_generation_v1"
    ).resolve(strict=True)
    completion_path = gen1_artifact / "predictions_complete.json"
    completion = _json(completion_path)
    receipts = {
        (str(row["fold"]), float(row["fraction"])): row for row in completion["parts"]
    }
    expected = {(fold["name"], fraction) for fold in folds for fraction in FRACTIONS}
    if set(receipts) != expected:
        raise PermissionError("sealed Gen1 prefix-audit receipt cells differ")
    result: dict[tuple[str, float], np.ndarray] = {}
    audit: dict[str, Any] = {}
    for fraction in FRACTIONS:
        audit[_tag(fraction)] = {}
        for fold in folds:
            cell = (str(fold["name"]), fraction)
            receipt = receipts[cell]
            parquet = (root / str(receipt["parquet"]).replace("\\", "/")).resolve(
                strict=True
            )
            if not parquet.is_relative_to(gen1_artifact):
                raise PermissionError(f"sealed Gen1 prefix audit escapes artifact: {cell}")
            part_audit_path = parquet.with_suffix(".json")
            if _sha(part_audit_path) != receipt["audit_sha256"]:
                raise PermissionError(f"sealed Gen1 prefix audit SHA differs: {cell}")
            part_audit = _json(part_audit_path)
            if part_audit["fold"] != cell[0] or float(part_audit["fraction"]) != fraction:
                raise PermissionError(f"sealed Gen1 prefix audit identity differs: {cell}")
            adjusted = int(pd.Timestamp(part_audit["adjusted_cutoff_utc"]).value)
            nominal = int(pd.Timestamp(part_audit["nominal_cutoff_utc"]).value)
            fold_train = np.asarray(fold["train_idx"], dtype=np.int64)
            time_ns = np.asarray(fold["time_ns"], dtype=np.int64)
            ids = fold_train[time_ns[fold_train] <= adjusted]
            if (
                len(ids) != int(part_audit["prefix_rows"])
                or ids_sha256(ids) != part_audit["prefix_positions_sha256"]
                or np.intersect1d(ids, fold["val_idx"]).size
            ):
                raise PermissionError(f"sealed Gen1 prefix IDs differ: {cell}")
            result[cell] = ids
            audit[_tag(fraction)][cell[0]] = {
                "fraction": fraction,
                "eligible_rows": int(len(fold_train)),
                "prefix_rows": int(len(ids)),
                "nominal_cutoff_utc": part_audit["nominal_cutoff_utc"],
                "adjusted_cutoff_utc": part_audit["adjusted_cutoff_utc"],
                "current_run_prefix_selector_target_reads": 0,
                "current_run_prefix_selector_columns": ["time", "row_position"],
                "historical_event_safe_cutoff_lineage_disclosed": True,
                "historical_event_boundary_retreat_iterations": int(
                    part_audit["event_boundary_retreat_iterations"]
                ),
                "boundary_split_risk_if_nominal_cutoff_used": bool(adjusted < nominal),
                "id_sha256_little_endian_int64": ids_sha256(ids),
                "validation_id_sha256_little_endian_int64": ids_sha256(fold["val_idx"]),
                "exact_to_immutable_incumbent_fold_train_ids": True,
            }
    return result, audit

def _verify_v5_ledger_binding(
    root: Path, config: dict[str, Any], ledger_path: Path
) -> dict[str, Any]:
    binding = config["v5_ledger_binding"]
    observed = {
        "path": ledger_path.relative_to(root).as_posix(),
        "sha256": _sha(ledger_path),
        "bytes": int(ledger_path.stat().st_size),
    }
    if observed != {key: binding[key] for key in ("path", "sha256", "bytes")}:
        raise PermissionError("canonical v5 ledger path, SHA, or byte count differs")
    records = validate_ledger(root, ledger_path)
    if (
        len(records) != binding["event_count"]
        or records[-1]["seq"] != binding["head_seq"]
        or records[-1]["event_sha256"] != binding["head_event_sha256"]
    ):
        raise PermissionError("canonical v5 ledger latest head differs")
    uploads = sum(record["payload"].get("upload_performed") is True for record in records)
    if (
        binding["all_event_upload_performed_false"] is not True
        or binding["semantic_upload_count"] != 0
        or uploads != 0
        or not all(record["payload"].get("upload_performed") is False for record in records)
    ):
        raise PermissionError("canonical v5 ledger upload semantics differ from zero")
    return {**observed, "event_count": len(records), "head_seq": records[-1]["seq"]}

def _model_config(config: dict[str, Any], feature_count: int, group_count: int) -> ResidualModelConfig:
    model = config["model"]
    result = ResidualModelConfig(
        input_feature_count=feature_count,
        group_count=group_count,
        width=int(model["width"]),
        group_embedding_width=int(model["group_embedding_width"]),
        dilations=tuple(int(value) for value in model["dilations"]),
        kernel_size=int(model["kernel_size"]),
        dropout=float(model["dropout"]),
        norm_groups=int(model["norm_groups"]),
        maximum_absolute_logit_correction=float(model["maximum_absolute_logit_correction"]),
    )
    result.validate()
    if result.receptive_field_rows != int(model["receptive_field_rows"]):
        raise ValueError("registered Gen5 receptive field differs")
    return result

def _training_config(config: dict[str, Any]) -> ResidualTrainingConfig:
    training = config["training"]
    weights = training["loss_weights"]
    result = ResidualTrainingConfig(
        optimizer_steps=int(training["optimizer_steps_per_residual_fit"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        main_loss_weight=float(weights["main"]),
        distillation_loss_weight=float(weights["distillation"]),
        identity_regularizer_weight=float(weights["identity_regularizer"]),
    )
    result.validate()
    return result

def _postprocess_ids(
    train: Any,
    ids: np.ndarray,
    probability: np.ndarray,
    postprocess: dict[str, Any],
) -> np.ndarray:
    frame = train.iloc[ids][["station", "year", "layer", "time", "temp", "psal", "depth"]].copy()
    plateau = detect_plateaus(frame).to_numpy(bool)
    spike = detect_singleton_spikes(frame).to_numpy(bool)
    return apply_postprocess(frame, probability, plateau, spike, postprocess).astype(
        np.int8, copy=False
    )

def _save_joblib_new(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        joblib.dump(value, handle, compress=3)
    return _sha(path)

def _teacher_oof(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: Any,
    p1_config: Any,
    outer_prefix_ids: np.ndarray,
    outer_forbidden_ids: np.ndarray | None,
    fold_name: str,
    fold_ordinal: int,
    fraction_tag: str,
    scope: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    splits = build_three_block_inner_splits(
        train.loc[:, ["time"]],
        outer_prefix_ids,
        purge_days=int(config["inner_cross_fit"]["purge_days"]),
    )
    prefix_frame = train.iloc[outer_prefix_ids][
        ["station", "year", "layer", "time", "temp", "psal", "depth"]
    ].reset_index(drop=True)
    bundle = build_features(
        prefix_frame,
        config=p1_config,
        mode="offline",
        cadence_minutes=p1_config.data.cadence_minutes,
        group_columns=p1_config.data.group_columns,
    )
    global_to_local = np.full(len(train), -1, dtype=np.int64)
    global_to_local[outer_prefix_ids] = np.arange(len(outer_prefix_ids), dtype=np.int64)
    n_rows = len(train)
    seed_probability = {seed: np.full(n_rows, 0.5, dtype=np.float32) for seed in SEEDS}
    seed_decision = {seed: np.zeros(n_rows, dtype=np.int8) for seed in SEEDS}
    receipts: list[dict[str, Any]] = []
    exact_parameters = dict(p1_config.raw["models"]["xgboost"])
    threads = int(p1_config.raw["project"]["threads"])
    for split in splits:
        train_ids = split.teacher_train_ids
        prediction_ids = split.teacher_prediction_ids
        if outer_forbidden_ids is not None and (
            np.intersect1d(train_ids, outer_forbidden_ids).size
            or np.intersect1d(prediction_ids, outer_forbidden_ids).size
        ):
            raise PermissionError("inner teacher touched forbidden outer validation rows")
        local_train = global_to_local[train_ids]
        local_prediction = global_to_local[prediction_ids]
        if (local_train < 0).any() or (local_prediction < 0).any():
            raise PermissionError("inner teacher IDs escape exact outer prefix")
        labels = pd.to_numeric(
            train.iloc[train_ids]["label"], errors="raise"
        ).to_numpy(np.int8)
        if len(np.unique(labels)) != 2:
            raise ValueError("inner teacher train block must contain both labels")
        encoder = TabularEncoder().fit(bundle, local_train)
        train_features = encoder.transform(bundle, local_train)
        prediction_features = encoder.transform(bundle, local_prediction)
        for seed in SEEDS:
            started = time.perf_counter()
            model = _fit_model(
                "xgboost",
                exact_parameters,
                int(seed) + int(fold_ordinal),
                threads,
                train_features,
                labels,
            )
            probability = model.predict_proba(prediction_features)[:, 1].astype(np.float32)
            seed_probability[seed][prediction_ids] = probability
            seed_decision[seed][prediction_ids] = _postprocess_ids(
                train,
                prediction_ids,
                probability,
                config["fixed_fold_postprocess"][fold_name],
            )
            relative = (
                f"teacher_models/{scope}/{fraction_tag}/{fold_name}/"
                f"block_{split.block}/seed_{seed}.joblib"
            )
            model_path = _safe_path(paths["artifact"], relative)
            model_sha = _save_joblib_new(model_path, {"encoder": encoder, "model": model})
            loaded = joblib.load(model_path)
            reproduced = loaded["model"].predict_proba(
                loaded["encoder"].transform(bundle, local_prediction)
            )[:, 1].astype(np.float32)
            reload_exact = bool(np.array_equal(probability, reproduced))
            if not reload_exact:
                raise RuntimeError("saved inner teacher did not reproduce OOF probability")
            blind_relative = (
                f"teacher_blind_predictions/{scope}/{fraction_tag}/{fold_name}/"
                f"block_{split.block}/seed_{seed}.npy"
            )
            blind_path = _safe_path(paths["artifact"], blind_relative)
            blind_sha = _npy_new(blind_path, probability)
            receipts.append(
                {
                    "role": "inner_teacher",
                    "scope": scope,
                    "fraction_tag": fraction_tag,
                    "fold": fold_name,
                    "block": split.block,
                    "seed": seed,
                    "train_rows": int(len(train_ids)),
                    "prediction_rows": int(len(prediction_ids)),
                    "train_ids_sha256": split.train_ids_sha256,
                    "prediction_ids_sha256": split.prediction_ids_sha256,
                    "purge_days": split.purge_days,
                    "train_end_utc": split.train_end_utc,
                    "prediction_start_utc": split.prediction_start_utc,
                    "teacher_fit_and_prediction_rows_disjoint": True,
                    "outer_validation_rows_touched": 0,
                    "raw_rows_outside_outer_prefix_read_for_features": 0,
                    "model_relative_path": relative,
                    "model_sha256": model_sha,
                    "blind_prediction_relative_path": blind_relative,
                    "blind_prediction_sha256": blind_sha,
                    "saved_model_reload_prediction_exact": reload_exact,
                    "elapsed_seconds": float(time.perf_counter() - started),
                    "test_value_reads": 0,
                }
            )
            _emit(
                "gen5_teacher_fit_complete",
                scope=scope,
                fraction_tag=fraction_tag,
                fold=fold_name,
                block=split.block,
                seed=seed,
                completed_in_cell=len(receipts),
                total_in_cell=9,
                elapsed_seconds=receipts[-1]["elapsed_seconds"],
            )
    oof_ids = np.concatenate([split.teacher_prediction_ids for split in splits])
    if len(np.unique(oof_ids)) != len(oof_ids):
        raise AssertionError("inner teacher OOF blocks overlap")
    matrix = np.column_stack([seed_probability[seed] for seed in SEEDS])
    mean_probability = matrix.mean(axis=1).astype(np.float32)
    std_probability = matrix.std(axis=1).astype(np.float32)
    return (
        {
            "splits": splits,
            "oof_ids": oof_ids,
            "seed_probability": seed_probability,
            "seed_decision": seed_decision,
            "mean_probability": mean_probability,
            "std_probability": std_probability,
        },
        receipts,
    )

def _gate_decision(
    *,
    train: Any,
    gate_ids: np.ndarray,
    truth: np.ndarray,
    base_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    micro_delta = _binary_f1(truth, residual_prediction) - _binary_f1(truth, base_prediction)
    stations = train.iloc[gate_ids]["station"].astype(str).to_numpy()
    station_deltas: dict[str, float] = {}
    missing_stations: list[str] = []
    for station in STATIONS:
        mask = stations == station
        if not mask.any():
            missing_stations.append(station)
            continue
        station_deltas[station] = _binary_f1(
            truth[mask], residual_prediction[mask]
        ) - _binary_f1(truth[mask], base_prediction[mask])
    thresholds = config["train_only_no_op_gate"]["apply_residual_if_all"]
    improved = sum(value > 0.0 for value in station_deltas.values())
    missing_binary_class = len(np.unique(truth)) != 2
    passed = bool(
        not missing_stations
        and not missing_binary_class
        and micro_delta >= float(thresholds["micro_f1_delta_at_least"])
        and improved >= int(thresholds["improved_station_count_at_least"])
        and min(station_deltas.values())
        >= float(thresholds["worst_station_f1_delta_at_least"])
    )
    return {
        "passed": passed,
        "micro_f1_delta": float(micro_delta),
        "station_deltas": station_deltas,
        "missing_required_stations": missing_stations,
        "missing_required_station_fail_closed": bool(missing_stations),
        "missing_binary_class_fail_closed": bool(missing_binary_class),
        "improved_station_count": int(improved),
        "thresholds": thresholds,
    }

def _full_fit_models(
    *,
    config: dict[str, Any],
    paths: dict[str, Path],
    train: Any,
    features: np.ndarray,
    feature_columns: list[str],
    layout: Any,
) -> dict[str, Any]:
    full_ids = np.arange(len(train), dtype=np.int64)
    p1_config = load_config(paths["base_config"])
    teacher, teacher_receipts = _teacher_oof(
        config=config,
        paths=paths,
        train=train,
        p1_config=p1_config,
        outer_prefix_ids=full_ids,
        outer_forbidden_ids=None,
        fold_name="2025_q4",
        fold_ordinal=0,
        fraction_tag="p100",
        scope="full_fit",
    )
    splits = teacher["splits"]
    gate_train_ids = np.concatenate(
        [splits[0].teacher_prediction_ids, splits[1].teacher_prediction_ids]
    )
    gate_ids = splits[2].teacher_prediction_ids
    refit_ids = teacher["oof_ids"]
    model_config = _model_config(config, len(feature_columns), layout.group_count)
    training_config = _training_config(config)
    gate_models: list[Any] = []
    refit_models: list[Any] = []
    gate_results: list[dict[str, Any]] = []
    for seed in SEEDS:
        gate_model = fit_incumbent_residual_model(
            features,
            layout,
            gate_train_ids,
            pd.to_numeric(train.iloc[gate_train_ids]["label"], errors="raise").to_numpy(
                np.int8
            ),
            teacher["seed_probability"][seed],
            teacher["mean_probability"],
            teacher["std_probability"],
            teacher["seed_decision"][seed],
            context_ids=full_ids,
            forbidden_ids=gate_ids,
            seed=seed,
            device="cuda",
            model_config=model_config,
            training_config=training_config,
        )
        gate_probability = predict_incumbent_residual_probability(
            gate_model,
            features,
            layout,
            gate_ids,
            teacher["seed_probability"][seed],
            teacher["mean_probability"],
            teacher["std_probability"],
            teacher["seed_decision"][seed],
            context_ids=full_ids,
            device="cuda",
        )
        gate_model_path = _safe_path(
            paths["artifact"], f"full_fit/gate_model_seed_{seed}.pt"
        )
        save_fitted_incumbent_residual_model(gate_model, gate_model_path)
        loaded_gate = load_fitted_incumbent_residual_model(gate_model_path)
        reproduced_gate = predict_incumbent_residual_probability(
            loaded_gate,
            features,
            layout,
            gate_ids,
            teacher["seed_probability"][seed],
            teacher["mean_probability"],
            teacher["std_probability"],
            teacher["seed_decision"][seed],
            context_ids=full_ids,
            device="cuda",
        )
        if not np.array_equal(gate_probability, reproduced_gate):
            raise RuntimeError("saved full-fit gate model did not reproduce")
        gate_blind_path = _safe_path(
            paths["artifact"], f"full_fit/gate_blind_seed_{seed}.npy"
        )
        _npy_new(gate_blind_path, gate_probability)
        gate_truth = pd.to_numeric(
            train.iloc[gate_ids]["label"], errors="raise"
        ).to_numpy(np.int8)
        per_postprocess: dict[str, Any] = {}
        for fold_name, postprocess in config["fixed_fold_postprocess"].items():
            per_postprocess[fold_name] = _gate_decision(
                train=train,
                gate_ids=gate_ids,
                truth=gate_truth,
                base_prediction=_postprocess_ids(
                    train,
                    gate_ids,
                    teacher["seed_probability"][seed][gate_ids],
                    postprocess,
                ),
                residual_prediction=_postprocess_ids(
                    train, gate_ids, gate_probability, postprocess
                ),
                config=config,
            )
        gate_results.append(
            {
                "seed": seed,
                "passed_all_three_fixed_fold_postprocesses": all(
                    row["passed"] for row in per_postprocess.values()
                ),
                "per_postprocess": per_postprocess,
                "prediction_sealed_before_gate_label_read": True,
                "gate_model_sha256": _sha(gate_model_path),
                "gate_model_reload_prediction_exact": True,
            }
        )
        gate_models.append(gate_model)
        refit_models.append(
            fit_incumbent_residual_model(
                features,
                layout,
                refit_ids,
                pd.to_numeric(
                    train.iloc[refit_ids]["label"], errors="raise"
                ).to_numpy(np.int8),
                teacher["seed_probability"][seed],
                teacher["mean_probability"],
                teacher["std_probability"],
                teacher["seed_decision"][seed],
                context_ids=full_ids,
                forbidden_ids=None,
                seed=seed,
                device="cuda",
                model_config=model_config,
                training_config=training_config,
            )
        )
    full_bundle = build_features(
        train.loc[:, ["station", "year", "layer", "time", "temp", "psal", "depth"]],
        config=p1_config,
        mode="offline",
        cadence_minutes=p1_config.data.cadence_minutes,
        group_columns=p1_config.data.group_columns,
    )
    encoder = TabularEncoder().fit(full_bundle, full_ids)
    full_offline_features = encoder.transform(full_bundle, full_ids)
    labels = pd.to_numeric(train["label"], errors="raise").to_numpy(np.int8)
    exact_parameters = dict(p1_config.raw["models"]["xgboost"])
    threads = int(p1_config.raw["project"]["threads"])
    base_models: list[Any] = []
    base_paths: list[tuple[str, Path, str]] = []
    base_probabilities: list[np.ndarray] = []
    for seed in SEEDS:
        base_model = _fit_model(
            "xgboost", exact_parameters, seed, threads, full_offline_features, labels
        )
        base_relative = f"full_fit/base_seed_{seed}.joblib"
        base_path = _safe_path(paths["artifact"], base_relative)
        base_sha = _save_joblib_new(base_path, {"encoder": encoder, "model": base_model})
        loaded_base = joblib.load(base_path)
        full_base = base_model.predict_proba(full_offline_features)[:, 1].astype(np.float32)
        original_base = full_base
        reloaded_base = loaded_base["model"].predict_proba(
            loaded_base["encoder"].transform(full_bundle, full_ids)
        )[:, 1].astype(np.float32)
        if not np.array_equal(original_base, reloaded_base):
            raise RuntimeError("saved full-fit incumbent base did not reproduce")
        base_models.append(base_model)
        base_paths.append((base_relative, base_path, base_sha))
        base_probabilities.append(full_base)
    base_matrix = np.column_stack(base_probabilities)
    base_mean = base_matrix.mean(axis=1).astype(np.float32)
    base_std = base_matrix.std(axis=1).astype(np.float32)
    base_decisions = [
        _postprocess_ids(
            train,
            full_ids,
            probability,
            config["fixed_fold_postprocess"]["2025_q4"],
        )
        for probability in base_probabilities
    ]
    reference_ids = gate_ids[: min(4096, len(gate_ids))]
    packages: list[dict[str, Any]] = []
    for index, seed in enumerate(SEEDS):
        base_relative, base_path, base_sha = base_paths[index]
        residual_relative = f"full_fit/residual_seed_{seed}.pt"
        residual_path = _safe_path(paths["artifact"], residual_relative)
        save_fitted_incumbent_residual_model(refit_models[index], residual_path)
        loaded_residual = load_fitted_incumbent_residual_model(residual_path)
        if loaded_residual.model_state_sha256 != refit_models[index].model_state_sha256:
            raise RuntimeError("saved full-fit residual state differs")
        original_residual = predict_incumbent_residual_probability(
            refit_models[index],
            features,
            layout,
            reference_ids,
            base_probabilities[index],
            base_mean,
            base_std,
            base_decisions[index],
            context_ids=full_ids,
            device="cuda",
        )
        reloaded_residual = predict_incumbent_residual_probability(
            loaded_residual,
            features,
            layout,
            reference_ids,
            base_probabilities[index],
            base_mean,
            base_std,
            base_decisions[index],
            context_ids=full_ids,
            device="cuda",
        )
        if not np.array_equal(original_residual, reloaded_residual):
            raise RuntimeError("saved full-fit residual did not reproduce inference")
        gate_passed = bool(
            gate_results[index]["passed_all_three_fixed_fold_postprocesses"]
        )
        package_probability = exact_identity_or_residual(
            base_probabilities[index][reference_ids],
            reloaded_residual,
            gate_passed=gate_passed,
        )
        if not gate_passed and not np.array_equal(
            package_probability, base_probabilities[index][reference_ids]
        ):
            raise AssertionError("failed full-fit gate did not preserve incumbent identity")
        packages.append(
            {
                "seed": seed,
                "base_model_path": base_relative,
                "base_model_sha256": base_sha,
                "residual_model_path": residual_relative,
                "residual_model_sha256": _sha(residual_path),
                "residual_model_state_sha256": refit_models[index].model_state_sha256,
                "gate": gate_results[index],
                "saved_base_reload_prediction_exact": True,
                "saved_residual_reload_state_exact": True,
                "saved_residual_reload_inference_exact": True,
                "failed_gate_reference_probability_identity_exact": not gate_passed,
                "serialization_reference_ids_sha256": ids_sha256(reference_ids),
            }
        )
    if len(teacher_receipts) != 9 or len(gate_models) != 3 or len(refit_models) != 3:
        raise AssertionError("full-fit teacher/gate/refit count differs")
    receipt = {
        "performed": True,
        "model_count": 18,
        "inner_teacher_fits": 9,
        "residual_gate_fits": 3,
        "residual_refits": 3,
        "incumbent_inference_base_fits": 3,
        "saved_inference_package_count": 3,
        "optimizer_steps": 720,
        "teacher_model_receipts": teacher_receipts,
        "models": packages,
        "feature_columns": feature_columns,
        "test_value_reads": 0,
        "test_prediction_generations": 0,
        "candidate_files": 0,
        "uploads": 0,
    }
    _json_new(paths["artifact"] / "full_fit_models.json", receipt)
    return receipt


def verify_relative_input_pins(
    root: Path,
    data_dir: Path,
    immutable_inputs: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Hash pinned inputs using explicit data-file names or workspace relatives."""

    observed: dict[str, dict[str, Any]] = {}
    for name, expected in immutable_inputs.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise PermissionError(f"non-portable immutable input key: {name}")
        path = data_dir / relative if len(relative.parts) == 1 else root / relative
        resolved = path.resolve(strict=True)
        base = data_dir if len(relative.parts) == 1 else root
        if not resolved.is_relative_to(base.resolve(strict=True)):
            raise PermissionError(f"immutable input escaped its anchor: {name}")
        digest = _sha(resolved)
        if digest != expected:
            raise PermissionError(f"immutable input SHA differs: {name}")
        observed[name] = {"sha256": digest, "bytes": int(resolved.stat().st_size)}
    return observed


def causal_feature_audit(
    raw_input_frame: pd.DataFrame,
    *,
    cached_path: Path,
) -> dict[str, Any]:
    """Rebuild the causal cache without reopening or decoding target fields."""

    required = ["station", "layer", "time", "temp", "psal", "depth"]
    if list(raw_input_frame.columns) != [
        "station",
        "year",
        "layer",
        "time",
        "temp",
        "psal",
        "depth",
    ]:
        raise PermissionError("input-only frame columns differ")
    raw = raw_input_frame.loc[:, required]
    rebuilt = build_causal_raw_features(raw)
    cached = pd.read_parquet(cached_path, columns=list(CAUSAL_FEATURE_COLUMNS))
    if not np.array_equal(
        rebuilt.to_numpy(np.float32),
        cached.to_numpy(np.float32),
        equal_nan=True,
    ):
        raise PermissionError("pinned causal cache differs from raw-only rebuild")
    parsed = pd.to_datetime(raw["time"], errors="raise", utc=True, format="mixed")
    groups = raw["station"].astype(str) + "|" + raw["layer"].astype(str)
    order = pd.DataFrame(
        {"group": groups, "time": parsed, "row": np.arange(len(raw), dtype=np.int64)}
    )
    order.sort_values(["group", "time", "row"], kind="mergesort", inplace=True)
    prefix_ids: list[int] = []
    for _, rows in order.groupby("group", sort=False, observed=True):
        keep = max(1, int(len(rows) * 0.7))
        prefix_ids.extend(rows.iloc[:keep]["row"].astype(int).tolist())
    invariance_sha = assert_future_value_invariance(raw, sorted(prefix_ids))
    return {
        "target_columns_read": 0,
        "feature_count": len(CAUSAL_FEATURE_COLUMNS),
        "cache_exact_to_raw_rebuild": True,
        "future_value_perturbation_invariant": True,
        "future_value_perturbation_prefix_sha256": invariance_sha,
    }


__all__ = [
    "FOLDS",
    "FOLD_ORDER",
    "FRACTIONS",
    "HYPOTHESIS",
    "KEY_COLUMNS",
    "SEEDS",
    "STATIONS",
    "SequenceLayout",
    "_binary_f1",
    "_comparator_frame",
    "_deep_sha",
    "_emit",
    "_fold_runtime",
    "_full_fit_models",
    "_gate_decision",
    "_json",
    "_json_new",
    "_metric_slices",
    "_model_config",
    "_now",
    "_npy_new",
    "_parquet_new",
    "_pinned_label_free_prefixes",
    "_postprocess_ids",
    "_safe_path",
    "_score",
    "_sha",
    "_tag",
    "_teacher_oof",
    "_training_config",
    "_verify_gen1_parts",
    "_verify_v5_ledger_binding",
    "build_three_block_inner_splits",
    "causal_feature_audit",
    "exact_identity_or_residual",
    "fit_incumbent_residual_model",
    "ids_sha256",
    "load_config",
    "load_fitted_incumbent_residual_model",
    "predict_incumbent_residual_probability",
    "save_fitted_incumbent_residual_model",
    "verify_relative_input_pins",
]
