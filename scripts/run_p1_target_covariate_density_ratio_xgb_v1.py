"""Execute the sealed P1 target-covariate density-ratio XGBoost experiment.

No submission upload is performed.  The official test input is used only as
an unlabeled covariate distribution and for final prediction after all local
prediction parts have been sealed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_qc.config import P1QCConfig, load_config
from p1_qc.data import load_dataset
from p1_qc.experiment import stable_hash
from p1_qc.features import FeatureBundle
from p1_qc.improvement_cycle import _predict_ensemble
from p1_qc.metrics import group_row_shares, micro_f1, weighted_group_f1
from p1_qc.models_tabular import make_tabular_classifier
from p1_qc.pipeline import (
    SavedTabularModel,
    TabularEncoder,
    apply_postprocess,
    load_or_build_features,
    predict_submission,
)
from p1_qc.rules import detect_plateaus, detect_singleton_spikes
from p1_qc.submission import build_submission, validate_submission, write_submission
from p1_qc.target_covariate_density_ratio import (
    DOMAIN_CATEGORICAL_COLUMNS,
    DOMAIN_FEATURE_COLUMNS,
    DOMAIN_FORBIDDEN_COLUMNS,
    DOMAIN_NUMERIC_COLUMNS,
    build_daily_domain_covariates,
    combined_training_weight,
    effective_sample_fraction,
    estimate_source_daily_density_ratio,
    map_daily_ratio_to_rows,
    paired_weighted_block_bootstrap,
    sha256_array,
)
from p1_qc.validation import normal_station_layer_day_fp


KST = ZoneInfo("Asia/Seoul")
KEY_COLUMNS = ("station", "year", "layer", "time")
DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/p1_target_covariate_density_ratio_xgb_v1.json"


def _now_kst() -> str:
    return datetime.now(KST).isoformat()


def _emit(event: str, **values: Any) -> None:
    print(
        json.dumps(
            {"time_kst": _now_kst(), "event": event, **values},
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError(f"expected JSON object: {path}")
    return parsed


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial JSON: {temporary}")
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
    os.replace(temporary, path)


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    if temporary.exists():
        raise RuntimeError(f"stale partial parquet: {temporary}")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _resolve_repo(relative: str, *, must_exist: bool = True) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise RuntimeError(f"path escapes project root: {relative}")
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def _data_dir() -> Path:
    raw = os.environ.get("P1_DATA_DIR")
    if not raw:
        raise RuntimeError("P1_DATA_DIR must point to the immutable P1_qc_anomaly directory")
    directory = Path(raw).expanduser().resolve(strict=True)
    for name in ("README.md", "train.csv", "test.csv", "sample_submission.csv"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    return directory


def _artifact_dir(config: Mapping[str, Any]) -> Path:
    path = _resolve_repo(str(config["artifact_dir"]), must_exist=False)
    if not path.is_relative_to(PROJECT_ROOT / "artifacts"):
        raise RuntimeError("artifact_dir must remain under project artifacts")
    return path


def _validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p1_target_covariate_density_ratio_xgb.v1":
        raise ValueError("unexpected schema_version")
    if config.get("experiment_id") != "p1_target_covariate_density_ratio_xgb_v1":
        raise ValueError("unexpected experiment_id")
    domain = config["domain"]
    if tuple(domain["categorical_features"]) != DOMAIN_CATEGORICAL_COLUMNS:
        raise ValueError("domain categorical feature contract changed")
    if tuple(domain["numeric_features"]) != DOMAIN_NUMERIC_COLUMNS:
        raise ValueError("domain numeric feature contract changed")
    if tuple(domain["forbidden_features"]) != DOMAIN_FORBIDDEN_COLUMNS:
        raise ValueError("domain forbidden feature contract changed")
    if list(config["validation"]["prefix_fractions"]) != [0.4, 0.55, 0.7, 0.85, 1.0]:
        raise ValueError("prefix fractions changed")
    seeds = [
        int(config["validation"]["primary_seed"]),
        *[int(value) for value in config["validation"]["robustness_seeds"]],
    ]
    if seeds != [20260813, 20260829, 20260847]:
        raise ValueError("registered seeds changed")
    expected_model = {
        "n_estimators": 700,
        "learning_rate": 0.04,
        "max_depth": 7,
        "min_child_weight": 20.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "device": "cpu",
    }
    if dict(config["model"]["parameters"]) != expected_model:
        raise ValueError("frozen XGBoost parameters changed")
    expected_postprocess = {
        "high_threshold": 0.2,
        "low_threshold": 0.1,
        "close_gap_rows": 0,
        "minimum_positive_run": 12,
    }
    if dict(config["model"]["deployment_postprocess"]) != expected_postprocess:
        raise ValueError("deployment postprocess changed")


def _verify_immutable(config: Mapping[str, Any], data_dir: Path) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, expected in config["immutable_sha256"].items():
        path = data_dir / name if name in {"README.md", "train.csv", "test.csv", "sample_submission.csv"} else _resolve_repo(name)
        digest = _sha256(path)
        if digest != expected:
            raise RuntimeError(f"immutable input drift: {name}: {digest} != {expected}")
        observed[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}
    return observed


def _load_cached_bundle(
    frame: pd.DataFrame,
    parquet_path: Path,
    metadata_path: Path,
) -> FeatureBundle:
    metadata = _json(metadata_path)
    features = pd.read_parquet(parquet_path)
    if len(features) != len(frame) or int(metadata["rows"]) != len(frame):
        raise RuntimeError("feature cache row count mismatch")
    if metadata["source_sha256"] != frame.attrs["source_sha256"]:
        raise RuntimeError("feature cache source hash mismatch")
    if _sha256(parquet_path) != metadata["parquet_sha256"]:
        raise RuntimeError("feature cache parquet hash mismatch")
    columns = tuple(str(value) for value in metadata["feature_columns"])
    categorical = tuple(str(value) for value in metadata["categorical_columns"])
    if tuple(features.columns) != columns or {"label", "anomaly_type"}.intersection(columns):
        raise RuntimeError("feature cache schema is unsafe")
    features.index = frame.index.copy()
    features.attrs.update(
        {
            "feature_columns": columns,
            "categorical_columns": categorical,
            "feature_mode": "offline",
        }
    )
    return FeatureBundle(features, columns, categorical)


def _effective_p1_config(config: Mapping[str, Any], data_dir: Path) -> P1QCConfig:
    loaded = load_config(
        _resolve_repo(str(config["paths"]["base_config"])),
        env={"P1_DATA_DIR": str(data_dir), "P1QC_MODE": "offline"},
    )
    return replace(loaded, mode="offline", features=replace(loaded.features, mode="offline"))


def _round_a_reproduction(
    config: Mapping[str, Any],
    p1_config: P1QCConfig,
    data_dir: Path,
    test: pd.DataFrame,
    test_bundle: FeatureBundle,
    artifact: Path,
) -> dict[str, Any]:
    output = artifact / "round_a" / "reproduced.csv"
    expected = _resolve_repo(str(config["paths"]["round_a_candidate"]))
    model_path = _resolve_repo(str(config["paths"]["round_a_model"]))
    seal = _json(_resolve_repo(str(config["paths"]["round_a_resume_seal"])))
    implementation_paths = {
        "module": _resolve_repo("src/p1_qc/improvement_cycle_resume.py"),
        "runner": _resolve_repo("scripts/resume_p1_full_improvement_cycle.py"),
        "tests": _resolve_repo("tests/test_p1_full_improvement_cycle_resume.py"),
    }
    implementation_matches = {
        name: _sha256(path) == seal["resume_implementation_sha256"][name]
        for name, path in implementation_paths.items()
    }
    if not all(implementation_matches.values()):
        raise RuntimeError("Round A sealed reproduction implementation drifted")
    if not output.exists():
        causal_config = replace(
            p1_config, mode="causal", features=replace(p1_config.features, mode="causal")
        )
        causal_bundle = load_or_build_features(test, causal_config, kind="test", use_cache=True)
        ensemble = joblib.load(model_path)
        reproduced = _predict_ensemble(ensemble, test, test_bundle, causal_bundle)
        output.parent.mkdir(parents=True, exist_ok=True)
        write_submission(reproduced, output)
    validation = validate_submission(output, test)
    byte_identical = output.read_bytes() == expected.read_bytes()
    if not byte_identical:
        raise RuntimeError("Round A saved model no longer reproduces byte-identically")
    return {
        "decision": "QA_PASS",
        "candidate_path": str(expected),
        "candidate_sha256": _sha256(expected),
        "reproduced_path": str(output),
        "reproduced_sha256": _sha256(output),
        "byte_identical": True,
        "implementation_matches_seal": implementation_matches,
        "validation": validation,
        "submission_uploads": 0,
        "test_label_reads": 0,
    }


def _key_positions(frame: pd.DataFrame, keys: pd.DataFrame) -> np.ndarray:
    source = pd.MultiIndex.from_frame(frame.loc[:, list(KEY_COLUMNS)])
    requested = pd.MultiIndex.from_frame(keys.loc[:, list(KEY_COLUMNS)])
    if source.has_duplicates or requested.has_duplicates:
        raise RuntimeError("duplicate P1 keys")
    positions = source.get_indexer(requested)
    if (positions < 0).any():
        raise RuntimeError("reference OOF key is absent from train")
    return positions.astype(np.int64, copy=False)


def _fold_runtime(
    train: pd.DataFrame,
    p1_config: P1QCConfig,
    incumbent_oof: pd.DataFrame,
) -> list[dict[str, Any]]:
    parsed = pd.to_datetime(train["time"], errors="raise", utc=True, format="mixed")
    time_ns = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    folds: list[dict[str, Any]] = []
    for ordinal, spec in enumerate(p1_config.splits.folds):
        train_end = pd.Timestamp(spec.train_end).tz_convert("UTC")
        train_idx = np.flatnonzero(parsed.le(train_end).to_numpy())
        frozen_part = incumbent_oof.loc[incumbent_oof["fold"].eq(spec.name)].copy()
        val_idx = _key_positions(train, frozen_part)
        if np.intersect1d(train_idx, val_idx).size:
            raise RuntimeError(f"fold overlap: {spec.name}")
        folds.append(
            {
                "name": spec.name,
                "ordinal": ordinal,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "time_ns": time_ns,
                "frozen_keys": frozen_part.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True),
            }
        )
    return folds


def _int64_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes(order="C")).hexdigest()


def _safe_prefix(
    train: pd.DataFrame,
    fold_train_idx: np.ndarray,
    time_ns: np.ndarray,
    fraction: float,
    *,
    cadence_minutes: int,
) -> np.ndarray:
    eligible_times = np.unique(time_ns[fold_train_idx])
    if math.isclose(fraction, 1.0):
        adjusted = int(eligible_times[-1])
    else:
        ordinal = max(0, math.ceil(fraction * len(eligible_times)) - 1)
        adjusted = int(eligible_times[ordinal])
        nominal = adjusted
        cadence_ns = int(pd.Timedelta(minutes=cadence_minutes).value)
        preliminary = fold_train_idx[time_ns[fold_train_idx] <= nominal]
        work = train.iloc[preliminary][["station", "layer", "time", "label"]].copy()
        work["__position"] = preliminary
        work["__time_ns"] = time_ns[preliminary]
        work["__label"] = pd.to_numeric(work["label"], errors="raise").to_numpy(dtype=np.int8)
        work.sort_values(["station", "layer", "__time_ns", "__position"], inplace=True)
        for _ in range(1001):
            included = work.loc[work["__time_ns"].le(adjusted)]
            retreats: list[int] = []
            for _, group in included.groupby(["station", "layer"], sort=False, observed=True):
                if group.empty or int(group["__label"].iloc[-1]) != 1:
                    continue
                times = group["__time_ns"].to_numpy(dtype=np.int64)
                labels = group["__label"].to_numpy(dtype=np.int8)
                if adjusted - int(times[-1]) > cadence_ns:
                    continue
                start = len(group) - 1
                while (
                    start > 0
                    and labels[start - 1] == 1
                    and times[start] - times[start - 1] == cadence_ns
                ):
                    start -= 1
                retreats.append(int(times[start]) - 1)
            if not retreats:
                break
            new_adjusted = min([adjusted, *retreats])
            if new_adjusted >= adjusted:
                raise RuntimeError("safe prefix failed to retreat")
            adjusted = new_adjusted
        else:
            raise RuntimeError("safe prefix failed to converge")
    result = fold_train_idx[time_ns[fold_train_idx] <= adjusted]
    if not len(result):
        raise RuntimeError("empty safe prefix")
    return result


def _fit_weighted_xgb(
    parameters: Mapping[str, Any],
    *,
    seed: int,
    threads: int,
    features: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
):
    model = make_tabular_classifier(
        "xgboost", seed=seed, n_jobs=threads, parameters=dict(parameters)
    )
    model.fit(features, target, sample_weight=weight)
    return model


def _domain_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[dict[int, np.ndarray], dict[str, Any], bool]:
    domain = config["domain"]
    source_daily = build_daily_domain_covariates(
        train.loc[:, ["station", "layer", "time", "temp", "psal", "depth"]],
        cadence_minutes=int(domain["cadence_minutes"]),
        expected_rows_per_day=int(domain["expected_rows_per_day"]),
    )
    target_daily = build_daily_domain_covariates(
        test.loc[:, ["station", "layer", "time", "temp", "psal", "depth"]],
        cadence_minutes=int(domain["cadence_minutes"]),
        expected_rows_per_day=int(domain["expected_rows_per_day"]),
    )
    seeds = [
        int(config["validation"]["primary_seed"]),
        *[int(value) for value in config["validation"]["robustness_seeds"]],
    ]
    gates = domain["support_gates"]
    row_ratios: dict[int, np.ndarray] = {}
    audits: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for seed in seeds:
        result = estimate_source_daily_density_ratio(
            source_daily,
            target_daily,
            seed=seed,
            n_splits=int(domain["oof"]["n_splits"]),
            regularization_c=float(domain["classifier"]["C"]),
            ratio_clip=tuple(float(value) for value in domain["ratio"]["clip"]),
        )
        mapped = map_daily_ratio_to_rows(train, source_daily, result.source_daily_ratio)
        row_ratios[seed] = mapped
        per_domain = result.audit["per_station_layer_daily_ratio_ess_fraction"]
        seed_checks = {
            "group_overlap_zero": bool(result.audit["all_groups_disjoint"]),
            "forbidden_feature_intersection_empty": not result.audit[
                "forbidden_feature_intersection"
            ],
            "target_support_complete": not result.audit[
                "missing_target_station_layer_support"
            ],
            "daily_ess": float(result.audit["daily_ratio_ess_fraction"])
            >= float(gates["minimum_daily_ratio_ess_fraction"]),
            "row_ess": effective_sample_fraction(mapped)
            >= float(gates["minimum_row_ratio_ess_fraction"]),
            "per_station_layer_daily_ess": min(float(value) for value in per_domain.values())
            >= float(gates["minimum_station_layer_daily_ratio_ess_fraction"]),
            "row_mapping_complete": len(mapped) == len(train),
            "weights_finite_positive": bool(np.isfinite(mapped).all() and (mapped > 0).all()),
        }
        checks[f"seed_{seed}"] = all(seed_checks.values())
        audits[str(seed)] = {
            **result.audit,
            "row_ratio_ess_fraction": effective_sample_fraction(mapped),
            "row_ratio_sha256": sha256_array(mapped),
            "checks": seed_checks,
        }
    audit = {
        "schema_version": "p1_target_covariate_density_ratio.domain_audit.v1",
        "source_days": len(source_daily),
        "target_days": len(target_daily),
        "domain_feature_count": len(DOMAIN_FEATURE_COLUMNS),
        "seeds": audits,
        "seed_gate_pass": checks,
        "structural_gate_passed": all(checks.values()),
        "label_reads_by_domain_model": 0,
        "anomaly_type_reads_by_domain_model": 0,
        "official_score_reads": 0,
        "test_label_reads": 0,
    }
    return row_ratios, audit, bool(audit["structural_gate_passed"])


def _reference_part(config: Mapping[str, Any], fold: str, fraction: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    tag = f"p{int(round(100 * fraction)):03d}"
    directory = _resolve_repo(str(config["paths"]["reference_curve_parts"]))
    parquet = directory / f"{fold}_{tag}.parquet"
    audit_path = directory / f"{fold}_{tag}.json"
    audit = _json(audit_path)
    if _sha256(parquet) != audit["parquet_sha256"]:
        raise RuntimeError(f"reference prediction part drift: {parquet}")
    return pd.read_parquet(parquet), audit


def _build_prediction_parts(
    train: pd.DataFrame,
    bundle: FeatureBundle,
    p1_config: P1QCConfig,
    config: Mapping[str, Any],
    folds: Sequence[Mapping[str, Any]],
    row_ratios: Mapping[int, np.ndarray],
    artifact: Path,
) -> dict[str, Any]:
    fractions = [float(value) for value in config["validation"]["prefix_fractions"]]
    seeds = [
        int(config["validation"]["primary_seed"]),
        *[int(value) for value in config["validation"]["robustness_seeds"]],
    ]
    parameters = dict(config["model"]["parameters"])
    threads = int(p1_config.raw["project"]["threads"])
    complete: list[dict[str, Any]] = []
    fit_count = 0
    for fold in folds:
        for fraction in fractions:
            tag = f"p{int(round(100 * fraction)):03d}"
            part_path = artifact / "prediction_parts" / f"{fold['name']}_{tag}.parquet"
            audit_path = artifact / "prediction_parts" / f"{fold['name']}_{tag}.json"
            prefix = _safe_prefix(
                train,
                np.asarray(fold["train_idx"], dtype=np.int64),
                np.asarray(fold["time_ns"], dtype=np.int64),
                fraction,
                cadence_minutes=p1_config.data.cadence_minutes,
            )
            reference, reference_audit = _reference_part(
                config, str(fold["name"]), fraction
            )
            if _int64_hash(prefix) != reference_audit["prefix_positions_sha256"]:
                raise RuntimeError("safe prefix differs from frozen reference")
            if part_path.exists() or audit_path.exists():
                if not (part_path.exists() and audit_path.exists()):
                    raise RuntimeError(f"incomplete prediction part: {part_path}")
                audit = _json(audit_path)
                if _sha256(part_path) != audit["parquet_sha256"]:
                    raise RuntimeError(f"prediction part hash mismatch: {part_path}")
                complete.append(audit)
                fit_count += int(audit["model_fit_count"])
                _emit("p1_density_part_reused", fold=fold["name"], fraction=fraction)
                continue

            encoder = TabularEncoder().fit(bundle, prefix)
            train_features = encoder.transform(bundle, prefix)
            validation_idx = np.asarray(fold["val_idx"], dtype=np.int64)
            validation_features = encoder.transform(bundle, validation_idx)
            target = pd.to_numeric(train.iloc[prefix]["label"], errors="raise").to_numpy(
                dtype=np.int8
            )
            validation_frame = train.iloc[validation_idx].loc[
                :, ["station", "year", "layer", "time", "temp", "psal", "depth"]
            ].copy()
            plateau = detect_plateaus(validation_frame).to_numpy(dtype=bool)
            spike = detect_singleton_spikes(validation_frame).to_numpy(dtype=bool)
            postprocess = dict(reference_audit["fixed_postprocess"])
            seed_probabilities: dict[int, np.ndarray] = {}
            seed_predictions: dict[int, np.ndarray] = {}
            weight_audits: dict[str, Any] = {}
            started = perf_counter()
            for seed in seeds:
                weight, weight_audit = combined_training_weight(
                    target, np.asarray(row_ratios[seed])[prefix]
                )
                if abs(float(weight_audit["sum_difference"])) > 1e-8:
                    raise RuntimeError("combined sample-weight normalization failed")
                model_seed = seed + int(fold["ordinal"])
                model = _fit_weighted_xgb(
                    parameters,
                    seed=model_seed,
                    threads=threads,
                    features=train_features,
                    target=target,
                    weight=weight,
                )
                probability = model.predict_proba(validation_features)[:, 1]
                prediction = apply_postprocess(
                    validation_frame, probability, plateau, spike, postprocess
                )
                seed_probabilities[seed] = probability
                seed_predictions[seed] = prediction
                weight_audits[str(seed)] = weight_audit
                del model, weight
                gc.collect()
            primary = int(config["validation"]["primary_seed"])
            expected_keys = fold["frozen_keys"].loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
            if not validation_frame.loc[:, list(KEY_COLUMNS)].reset_index(drop=True).equals(
                expected_keys
            ):
                raise RuntimeError("validation key/order differs from incumbent OOF")
            if not reference.loc[:, list(KEY_COLUMNS)].reset_index(drop=True).equals(expected_keys):
                raise RuntimeError("reference prediction key/order differs")
            part = validation_frame.loc[:, list(KEY_COLUMNS)].reset_index(drop=True)
            part["row_position"] = validation_idx
            part["fold"] = str(fold["name"])
            part["fraction"] = fraction
            part["plateau"] = plateau
            part["spike_candidate"] = spike
            for seed in seeds:
                part[f"incumbent__seed_{seed}__probability"] = reference[
                    f"baseline__seed_{seed}__probability"
                ].to_numpy(dtype=np.float32)
                part[f"incumbent__seed_{seed}__prediction"] = reference[
                    f"baseline__seed_{seed}__prediction"
                ].to_numpy(dtype=np.int8)
                part[f"candidate__seed_{seed}__probability"] = seed_probabilities[seed].astype(
                    np.float32
                )
                part[f"candidate__seed_{seed}__prediction"] = seed_predictions[seed].astype(
                    np.int8
                )
            part["incumbent_probability"] = part[
                f"incumbent__seed_{primary}__probability"
            ]
            part["incumbent_prediction"] = part[
                f"incumbent__seed_{primary}__prediction"
            ]
            part["candidate_probability"] = part[f"candidate__seed_{primary}__probability"]
            part["candidate_prediction"] = part[f"candidate__seed_{primary}__prediction"]
            _write_parquet_atomic(part, part_path)
            audit = {
                "schema_version": "p1_target_covariate_density_ratio.prediction_part.v1",
                "experiment_id": config["experiment_id"],
                "fold": fold["name"],
                "fraction": fraction,
                "prefix_rows": len(prefix),
                "prefix_positions_sha256": _int64_hash(prefix),
                "validation_rows": len(part),
                "validation_key_order_exact": True,
                "target_fold_validation_label_reads_before_prediction": 0,
                "model_fit_count": len(seeds),
                "registered_seeds": seeds,
                "primary_seed": primary,
                "seed_probabilities_averaged": False,
                "postprocess": postprocess,
                "sample_weight": weight_audits,
                "elapsed_seconds": perf_counter() - started,
                "parquet_path": str(part_path),
                "parquet_sha256": _sha256(part_path),
                "completed_at_kst": _now_kst(),
            }
            _write_json_new(audit_path, audit)
            complete.append(audit)
            fit_count += len(seeds)
            _emit(
                "p1_density_part_completed",
                fold=fold["name"],
                fraction=fraction,
                prefix_rows=len(prefix),
                elapsed_seconds=round(audit["elapsed_seconds"], 2),
            )
            del train_features, validation_features, seed_probabilities, seed_predictions
            gc.collect()
    receipt = {
        "schema_version": "p1_target_covariate_density_ratio.predictions_complete.v1",
        "part_count": len(complete),
        "expected_part_count": len(folds) * len(fractions),
        "model_fit_count": fit_count,
        "all_predictions_complete": len(complete) == len(folds) * len(fractions),
        "target_fold_validation_label_reads_before_prediction": 0,
        "test_label_reads": 0,
        "submission_uploads": 0,
        "parts": [
            {
                "fold": row["fold"],
                "fraction": row["fraction"],
                "parquet_sha256": row["parquet_sha256"],
            }
            for row in complete
        ],
    }
    path = artifact / "predictions_complete.json"
    if path.exists():
        if _json(path)["parts"] != receipt["parts"]:
            raise RuntimeError("prediction completion receipt changed")
    else:
        _write_json_new(path, receipt)
    return receipt


def _slice_metric(
    frame: pd.DataFrame,
    truth: np.ndarray,
    candidate: np.ndarray,
    incumbent: np.ndarray,
    *,
    column: str,
    test_shares: Mapping[Any, float],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, positions in frame.groupby(column, sort=True, observed=True).indices.items():
        idx = np.asarray(positions, dtype=np.int64)
        if column == "fold":
            base_score = weighted_group_f1(truth[idx], incumbent[idx], frame.iloc[idx], test_shares)
            candidate_score = weighted_group_f1(
                truth[idx], candidate[idx], frame.iloc[idx], test_shares
            )
            metric = "test_share_weighted_row_F1"
        else:
            base_score = micro_f1(truth[idx], incumbent[idx])
            candidate_score = micro_f1(truth[idx], candidate[idx])
            metric = "within_station_row_F1"
        result[str(value)] = {
            "rows": len(idx),
            "positive_rows": int(truth[idx].sum()),
            "incumbent": base_score,
            "candidate": candidate_score,
            "delta": candidate_score - base_score,
            "metric": metric,
        }
    return result


def _score_predictions(
    train: pd.DataFrame,
    test: pd.DataFrame,
    incumbent_oof: pd.DataFrame,
    config: Mapping[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    if not (artifact / "predictions_complete.json").is_file():
        raise RuntimeError("predictions must be sealed before scoring")
    fractions = [float(value) for value in config["validation"]["prefix_fractions"]]
    seeds = [
        int(config["validation"]["primary_seed"]),
        *[int(value) for value in config["validation"]["robustness_seeds"]],
    ]
    primary = seeds[0]
    test_shares = group_row_shares(test)
    points: list[dict[str, Any]] = []
    full_fold: dict[str, Any] = {}
    full_station: dict[str, Any] = {}
    full_fp: dict[str, Any] = {}
    full_seed_deltas: dict[str, float] = {}
    exact_incumbent = False
    for fraction_index, fraction in enumerate(fractions):
        tag = f"p{int(round(100 * fraction)):03d}"
        paths = sorted((artifact / "prediction_parts").glob(f"*_{tag}.parquet"))
        if len(paths) != 3:
            raise RuntimeError(f"expected three fold parts for {fraction}")
        combined = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        positions = combined["row_position"].to_numpy(dtype=np.int64)
        truth = pd.to_numeric(train.iloc[positions]["label"], errors="raise").to_numpy(
            dtype=np.int8
        )
        incumbent = combined["incumbent_prediction"].to_numpy(dtype=np.int8)
        candidate = combined["candidate_prediction"].to_numpy(dtype=np.int8)
        incumbent_weighted = weighted_group_f1(truth, incumbent, combined, test_shares)
        candidate_weighted = weighted_group_f1(truth, candidate, combined, test_shares)
        seed_metrics: dict[str, Any] = {}
        for seed in seeds:
            seed_incumbent = combined[f"incumbent__seed_{seed}__prediction"].to_numpy(
                dtype=np.int8
            )
            seed_candidate = combined[f"candidate__seed_{seed}__prediction"].to_numpy(
                dtype=np.int8
            )
            base_score = weighted_group_f1(truth, seed_incumbent, combined, test_shares)
            challenger_score = weighted_group_f1(
                truth, seed_candidate, combined, test_shares
            )
            seed_metrics[str(seed)] = {
                "incumbent_weighted_f1": base_score,
                "candidate_weighted_f1": challenger_score,
                "delta": challenger_score - base_score,
            }
        bootstrap = paired_weighted_block_bootstrap(
            truth,
            candidate,
            incumbent,
            combined,
            test_shares,
            replicates=int(config["validation"]["bootstrap"]["replicates"]),
            seed=int(config["validation"]["bootstrap"]["seed"]) + fraction_index,
        )
        point = {
            "fraction": fraction,
            "rows": len(combined),
            "incumbent_weighted_f1": incumbent_weighted,
            "candidate_weighted_f1": candidate_weighted,
            "weighted_f1_delta": candidate_weighted - incumbent_weighted,
            "incumbent_micro_f1": micro_f1(truth, incumbent),
            "candidate_micro_f1": micro_f1(truth, candidate),
            "seed_metrics": seed_metrics,
            "paired_weighted_bootstrap": bootstrap,
        }
        points.append(point)
        if math.isclose(fraction, 1.0):
            expected_keys = incumbent_oof.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
            observed_keys = combined.loc[:, [*KEY_COLUMNS, "fold"]].reset_index(drop=True)
            if not observed_keys.equals(expected_keys):
                raise RuntimeError("full prediction key/order differs from incumbent OOF")
            exact_incumbent = np.array_equal(
                incumbent, incumbent_oof["prediction"].to_numpy(dtype=np.int8)
            )
            if not exact_incumbent:
                raise RuntimeError("full primary-seed incumbent did not reproduce")
            full_fold = _slice_metric(
                combined,
                truth,
                candidate,
                incumbent,
                column="fold",
                test_shares=test_shares,
            )
            full_station = _slice_metric(
                combined,
                truth,
                candidate,
                incumbent,
                column="station",
                test_shares=test_shares,
            )
            full_fp = normal_station_layer_day_fp(truth, candidate, incumbent, combined)
            full_seed_deltas = {
                seed: float(values["delta"]) for seed, values in seed_metrics.items()
            }

    full = points[-1]
    baseline_fp = float(
        full_fp["baseline"]["false_positive_rows_per_normal_station_layer_day"]
    )
    candidate_fp = float(
        full_fp["candidate"]["false_positive_rows_per_normal_station_layer_day"]
    )
    relative_fp_increase = (
        (candidate_fp - baseline_fp) / baseline_fp
        if baseline_fp > 0
        else (0.0 if candidate_fp == 0 else float("inf"))
    )
    safety = config["validation"]["safety_gate"]
    safety_checks = {
        "full_weighted_f1_noncatastrophic": float(full["weighted_f1_delta"])
        >= float(safety["minimum_full_weighted_f1_delta"]),
        "minimum_two_noncatastrophic_folds": sum(
            float(row["delta"]) >= float(safety["minimum_noncatastrophic_fold_delta"])
            for row in full_fold.values()
        )
        >= int(safety["minimum_noncatastrophic_fold_count"]),
        "fp_per_normal_day_increase_within_limit": relative_fp_increase
        <= float(safety["maximum_relative_fp_per_normal_day_increase"]),
    }
    promotion = config["validation"]["research_promotion_gate"]
    late = [point for point in points if point["fraction"] in {0.7, 0.85, 1.0}]
    promotion_checks = {
        "full_weighted_f1_delta_material": float(full["weighted_f1_delta"])
        >= float(promotion["minimum_full_weighted_f1_delta"]),
        "weighted_ci90_lower_above_zero": float(
            full["paired_weighted_bootstrap"]["difference_ci90"][0]
        )
        > 0,
        "all_late_fractions_positive": all(
            float(point["weighted_f1_delta"]) > 0 for point in late
        ),
        "minimum_two_improved_folds": sum(float(row["delta"]) > 0 for row in full_fold.values())
        >= int(promotion["minimum_improved_fold_count"]),
        "worst_station_regression_within_limit": min(
            float(row["delta"]) for row in full_station.values()
        )
        >= -float(promotion["maximum_worst_station_f1_regression"]),
        "minimum_two_positive_seeds": sum(value > 0 for value in full_seed_deltas.values())
        >= int(promotion["minimum_positive_seed_count"]),
        "fp_per_normal_day_increase_within_limit": relative_fp_increase
        <= float(promotion["maximum_relative_fp_per_normal_day_increase"]),
    }
    return {
        "schema_version": "p1_target_covariate_density_ratio.metrics.v1",
        "metric": "test_share_weighted_row_F1",
        "primary_seed": primary,
        "robustness_seeds_not_averaged_or_selected": seeds[1:],
        "points": points,
        "full_fraction_fold_metrics": full_fold,
        "full_fraction_station_metrics": full_station,
        "full_fraction_seed_deltas": full_seed_deltas,
        "full_fraction_false_positive_diagnostics": full_fp,
        "relative_fp_per_normal_day_increase": relative_fp_increase,
        "full_primary_incumbent_exact_to_frozen_oof": exact_incumbent,
        "safety_checks": safety_checks,
        "safety_gate_passed": all(safety_checks.values()),
        "research_promotion_checks": promotion_checks,
        "research_promotion_local_gate_passed": all(promotion_checks.values()),
        "official_public_score_used_in_model_or_selection": False,
        "test_label_reads": 0,
        "submission_uploads": 0,
    }


def _full_fit_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_bundle: FeatureBundle,
    test_bundle: FeatureBundle,
    p1_config: P1QCConfig,
    config: Mapping[str, Any],
    row_ratio: np.ndarray,
    artifact: Path,
) -> dict[str, Any]:
    primary = int(config["validation"]["primary_seed"])
    encoder = TabularEncoder().fit(train_bundle, np.arange(len(train), dtype=np.int64))
    train_features = encoder.transform(train_bundle)
    test_features = encoder.transform(test_bundle)
    target = pd.to_numeric(train["label"], errors="raise").to_numpy(dtype=np.int8)
    weight, weight_audit = combined_training_weight(target, row_ratio)
    model = _fit_weighted_xgb(
        config["model"]["parameters"],
        seed=primary,
        threads=int(p1_config.raw["project"]["threads"]),
        features=train_features,
        target=target,
        weight=weight,
    )
    saved = SavedTabularModel(
        backend="xgboost",
        encoder=encoder,
        model=model,
        postprocess=dict(config["model"]["deployment_postprocess"]),
        feature_mode="offline",
        feature_hash=stable_hash(p1_config.features.__dict__),
        iteration_count=700,
        seed=primary,
    )
    model_path = artifact / "models" / "P1_TARGET_COVARIATE_DENSITY_RATIO_XGB_V1.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists():
        raise FileExistsError(model_path)
    joblib.dump(saved, model_path, compress=3)
    submission, probability = predict_submission(saved, test, test_bundle)
    candidate_path = artifact / "candidate" / "P1_TARGET_COVARIATE_DENSITY_RATIO_XGB_V1.csv"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    if candidate_path.exists():
        raise FileExistsError(candidate_path)
    write_submission(submission, candidate_path)
    validation = validate_submission(candidate_path, test)
    loaded = joblib.load(model_path)
    reproduced, reproduced_probability = predict_submission(loaded, test, test_bundle)
    reproduced_path = artifact / "candidate" / "reproduced.csv"
    write_submission(reproduced, reproduced_path)
    if candidate_path.read_bytes() != reproduced_path.read_bytes():
        raise RuntimeError("saved density-weighted model did not reproduce candidate bytes")
    if not np.array_equal(probability, reproduced_probability):
        raise RuntimeError("saved density-weighted model probabilities changed on reload")
    incumbent_path = _resolve_repo(str(config["paths"]["incumbent_submission"]))
    round_a_path = _resolve_repo(str(config["paths"]["round_a_candidate"]))
    return {
        "decision": "CANDIDATE_READY_NOT_UPLOADED",
        "model": {
            "path": str(model_path),
            "bytes": model_path.stat().st_size,
            "sha256": _sha256(model_path),
            "feature_count": len(saved.encoder.feature_columns),
            "backend": saved.backend,
            "seed": saved.seed,
            "iteration_count": saved.iteration_count,
            "postprocess": saved.postprocess,
        },
        "candidate": {
            "path": str(candidate_path),
            "bytes": candidate_path.stat().st_size,
            "sha256": _sha256(candidate_path),
            "rows": len(submission),
            "positive_rows": int(submission["label"].sum()),
            "positive_rate": float(submission["label"].mean()),
            "differs_from_incumbent_bytes": candidate_path.read_bytes()
            != incumbent_path.read_bytes(),
            "differs_from_round_a_bytes": candidate_path.read_bytes() != round_a_path.read_bytes(),
        },
        "reproduction": {
            "path": str(reproduced_path),
            "sha256": _sha256(reproduced_path),
            "byte_identical": True,
            "probabilities_array_identical": True,
        },
        "full_training_weight": weight_audit,
        "strict_validation": validation,
        "test_label_reads": 0,
        "submission_uploads": 0,
    }


def _fallback_candidate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_bundle: FeatureBundle,
    test_bundle: FeatureBundle,
    config: Mapping[str, Any],
    artifact: Path,
) -> dict[str, Any]:
    fallback_result = _json(_resolve_repo(str(config["paths"]["fallback_result"])))
    branch = fallback_result["selected_hypothesis"]
    summary = fallback_result["branch_summary"][branch]
    permitted = (
        branch == config["fallback"]["id"]
        and summary["gate_checks"]["all_leakage_checks"]
        and summary["gate_checks"]["all_reproducibility_checks"]
    )
    if not permitted:
        return {"decision": "NO_GO_FALLBACK_NOT_PERMITTED", "candidate": None}
    runner_path = _resolve_repo("scripts/run_p1_meaningful_learning_curve_generation_v1.py")
    spec = importlib.util.spec_from_file_location("p1_frozen_curve_runner", runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen fallback implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fallback_config = _json(_resolve_repo(str(config["paths"]["fallback_config"])))
    full_idx = np.arange(len(train), dtype=np.int64)
    encoder = TabularEncoder().fit(train_bundle, full_idx)
    train_features = encoder.transform(train_bundle)
    test_features = encoder.transform(test_bundle)
    target = pd.to_numeric(train["label"], errors="raise").to_numpy(dtype=np.int8)
    train_metadata = train.loc[:, ["station", "layer", "time"]].reset_index(drop=True)
    test_metadata = test.loc[:, ["station", "layer", "time"]].reset_index(drop=True)
    probability, packages, fit_count, _ = module._fit_candidate(
        branch,
        fallback_config,
        [int(value) for value in fallback_config["seeds"]],
        train_features,
        target,
        train_metadata,
        train["anomaly_type"].reset_index(drop=True),
        test_features,
        test_metadata,
    )
    plateau = detect_plateaus(test).to_numpy(dtype=bool)
    spike = detect_singleton_spikes(test).to_numpy(dtype=bool)
    prediction = apply_postprocess(
        test, probability, plateau, spike, fallback_config["deployment_postprocess"]
    )
    types = np.full(len(test), "", dtype=object)
    types[plateau & prediction.astype(bool)] = "flatline"
    types[spike & prediction.astype(bool)] = "spike"
    submission = build_submission(test, prediction, types)
    candidate_path = artifact / "candidate" / "P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1.csv"
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission(submission, candidate_path)
    model_path = artifact / "models" / "P1_EVENT_DAY_BALANCED_LGBM_FALLBACK_V1.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "branch": branch,
            "encoder": encoder,
            "packages": packages,
            "seeds": fallback_config["seeds"],
            "postprocess": fallback_config["deployment_postprocess"],
        },
        model_path,
        compress=3,
    )
    return {
        "decision": "STRUCTURAL_FALLBACK_READY_NOT_UPLOADED",
        "trigger": "density_structural_gate_failure",
        "model_fit_count": fit_count,
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "bytes": candidate_path.stat().st_size,
        },
        "model": {"path": str(model_path), "sha256": _sha256(model_path)},
        "strict_validation": validate_submission(candidate_path, test),
        "submission_uploads": 0,
        "test_label_reads": 0,
    }


def _manifest(artifact: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and not path.name.endswith(".partial"):
            files[str(path.relative_to(PROJECT_ROOT))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return {
        "schema_version": "p1_target_covariate_density_ratio.manifest.v1",
        "created_at_kst": _now_kst(),
        "artifacts": files,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    config = _json(config_path)
    _validate_contract(config)
    data_dir = _data_dir()
    immutable_before = _verify_immutable(config, data_dir)
    artifact = _artifact_dir(config)
    artifact.mkdir(parents=True, exist_ok=True)
    seal_path = artifact / "preexecution_seal.json"
    implementation = {
        "config": _sha256(config_path),
        "module": _sha256(_resolve_repo("src/p1_qc/target_covariate_density_ratio.py")),
        "runner": _sha256(Path(__file__)),
        "tests": _sha256(_resolve_repo("tests/test_p1_target_covariate_density_ratio_xgb_v1.py")),
    }
    if seal_path.exists():
        seal = _json(seal_path)
        if seal["implementation_sha256"] != implementation:
            raise RuntimeError("sealed implementation changed during resume")
    else:
        seal = {
            "schema_version": "p1_target_covariate_density_ratio.preexecution_seal.v1",
            "experiment_id": config["experiment_id"],
            "sealed_at_kst": _now_kst(),
            "implementation_sha256": implementation,
            "immutable_inputs": immutable_before,
            "test_label_reads": 0,
            "official_score_reads": 0,
            "submission_uploads": 0,
        }
        _write_json_new(seal_path, seal)
    _emit("p1_density_preflight_sealed", artifact=str(artifact))

    p1_config = _effective_p1_config(config, data_dir)
    train = load_dataset(data_dir / "train.csv", kind="train", audit=True)
    test = load_dataset(data_dir / "test.csv", kind="test", audit=True)
    train_bundle = _load_cached_bundle(
        train,
        _resolve_repo(str(config["paths"]["train_feature_cache"])),
        _resolve_repo(str(config["paths"]["train_feature_metadata"])),
    )
    test_bundle = _load_cached_bundle(
        test,
        _resolve_repo(str(config["paths"]["test_feature_cache"])),
        _resolve_repo(str(config["paths"]["test_feature_metadata"])),
    )
    if len(train_bundle.feature_columns) != int(config["model"]["feature_count"]):
        raise RuntimeError("frozen feature count changed")

    round_a = _round_a_reproduction(
        config, p1_config, data_dir, test, test_bundle, artifact
    )
    round_a_path = artifact / "round_a_reproduction.json"
    if not round_a_path.exists():
        _write_json_new(round_a_path, round_a)
    _emit("p1_round_a_reproduced", sha256=round_a["reproduced_sha256"])

    row_ratios, domain_audit, structural_passed = _domain_weights(train, test, config)
    domain_path = artifact / "domain_audit.json"
    if domain_path.exists():
        existing = _json(domain_path)
        if existing["seeds"] != domain_audit["seeds"]:
            raise RuntimeError("domain audit changed on deterministic resume")
    else:
        _write_json_new(domain_path, domain_audit)
    _emit(
        "p1_density_structural_gate",
        passed=structural_passed,
        primary_daily_ess=domain_audit["seeds"][str(config["validation"]["primary_seed"])][
            "daily_ratio_ess_fraction"
        ],
        primary_row_ess=domain_audit["seeds"][str(config["validation"]["primary_seed"])][
            "row_ratio_ess_fraction"
        ],
    )

    if not structural_passed:
        fallback = _fallback_candidate(train, test, train_bundle, test_bundle, config, artifact)
        result = {
            "schema_version": "p1_target_covariate_density_ratio.result.v1",
            "experiment_id": config["experiment_id"],
            "status": fallback["decision"],
            "round_a": round_a,
            "domain_structural_gate_passed": False,
            "fallback": fallback,
            "source_mutations": 0,
            "frozen_mutations": 0,
            "submission_uploads": 0,
            "test_label_reads": 0,
        }
    else:
        incumbent_oof = pd.read_parquet(_resolve_repo(str(config["paths"]["incumbent_oof"])))
        folds = _fold_runtime(train, p1_config, incumbent_oof)
        completion = _build_prediction_parts(
            train, train_bundle, p1_config, config, folds, row_ratios, artifact
        )
        metrics = _score_predictions(train, test, incumbent_oof, config, artifact)
        metrics_path = artifact / "metrics.json"
        if metrics_path.exists():
            if _json(metrics_path) != metrics:
                raise RuntimeError("metrics changed on deterministic resume")
        else:
            _write_json_new(metrics_path, metrics)
        _emit(
            "p1_density_local_gate",
            safety_passed=metrics["safety_gate_passed"],
            research_passed=metrics["research_promotion_local_gate_passed"],
            weighted_f1_delta=metrics["points"][-1]["weighted_f1_delta"],
            ci90=metrics["points"][-1]["paired_weighted_bootstrap"]["difference_ci90"],
        )
        full_fit: dict[str, Any] | None = None
        status = "NO_GO_LOCAL_SAFETY"
        if metrics["safety_gate_passed"]:
            full_fit = _full_fit_candidate(
                train,
                test,
                train_bundle,
                test_bundle,
                p1_config,
                config,
                row_ratios[int(config["validation"]["primary_seed"])],
                artifact,
            )
            status = (
                "CANDIDATE_READY_LOCAL_PROMOTION_PASS_NOT_UPLOADED"
                if metrics["research_promotion_local_gate_passed"]
                else "CANDIDATE_READY_RESEARCH_ONLY_NOT_UPLOADED"
            )
        result = {
            "schema_version": "p1_target_covariate_density_ratio.result.v1",
            "experiment_id": config["experiment_id"],
            "completed_at_kst": _now_kst(),
            "status": status,
            "round_a": round_a,
            "domain_structural_gate_passed": True,
            "prediction_parts": completion,
            "local_metrics": metrics,
            "full_fit": full_fit,
            "fallback": {"triggered": False, "reason": "primary structural gate passed"},
            "operation_counters": {
                "domain_label_reads": 0,
                "domain_official_score_reads": 0,
                "target_fold_validation_label_reads_before_prediction": 0,
                "curve_model_fits": completion["model_fit_count"],
                "full_fit_model_fits": int(full_fit is not None),
                "test_prediction_generations": 2 if full_fit else 0,
                "source_mutations": 0,
                "frozen_mutations": 0,
                "submission_uploads": 0,
                "test_label_reads": 0,
            },
        }

    immutable_after = _verify_immutable(config, data_dir)
    if immutable_before != immutable_after:
        raise RuntimeError("protected inputs changed during experiment")
    result["protected_inputs_unchanged"] = True
    result_path = artifact / "result.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    _write_json_new(result_path, result)
    manifest_path = artifact / "manifest.json"
    _write_json_new(manifest_path, _manifest(artifact))
    _emit("p1_density_complete", status=result["status"], result=str(result_path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        config = _json(args.config.resolve(strict=True))
        _validate_contract(config)
        print(
            json.dumps(
                {
                    "decision": "DRY_RUN_CONFIG_VALID",
                    "experiment_id": config["experiment_id"],
                    "artifact_dir": config["artifact_dir"],
                    "execute_required": True,
                },
                indent=2,
            )
        )
        return 0
    result = run(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
