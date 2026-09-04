"""Run the append-only P2 meaningful learning-curve generation.

The official immutable incumbent cannot currently be refit on arbitrary
chronological prefixes without inventing a new meta-training procedure.  This
runner therefore fails promotion closed, while still producing a fully
aggregate diagnostic curve against the closest reproducible same-prefix
surrogate.  It never uploads and never mutates frozen submissions.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from p2_restore.corrected_repeated_forward import (
    build_fixed_lean_arm,
    build_joint_masked_population,
    fit_fixed_blend,
    forward_training_mask,
    joint_mask_target_context,
    metric_report,
    nominal_target_rows,
    paired_fold_day_bootstrap,
    public_endpoints_from_masked_context,
    window_mask,
)
from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.features import TARGET_LAYERS, FeatureTable
from p2_restore.gbm_tournament import GBMArmSpec, fit_gbm_model
from p2_restore.meaningful_learning_curve import (
    chronological_prefix_masks,
    fold_equal_layer_rmse,
    numeric_curve_gate,
)
from p2_restore.profile_projection import project_profiles_vectorized

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/p2_meaningful_learning_curve_generation_v1.json"
DEFAULT_OUTPUT = ROOT / "artifacts/p2_meaningful_learning_curve_generation_v1"
DEFAULT_LOCK = ROOT / "artifacts/p2_meaningful_learning_curve_generation_v1_control/attempt.lock"
KST = ZoneInfo("Asia/Seoul")
KEY_COLUMNS = ["station", "layer", "time"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.name


def _now() -> str:
    return datetime.now(KST).isoformat()


def _exclusive_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise OSError(f"short append-only write: {path}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    _exclusive_bytes(path, payload + b"\n")


def _seal_sidecar(path: Path) -> None:
    digest = _sha256(path)
    _exclusive_bytes(path.with_name(path.name + ".sha256"), f"{digest}  {path.name}\n".encode())


def _load_json(path: Path) -> dict[str, Any]:
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return parsed


def _validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "p2_meaningful_learning_curve_generation.v1":
        raise ValueError("unexpected P2 learning-curve schema")
    if config.get("experiment_id") != "p2_meaningful_learning_curve_generation_v1":
        raise ValueError("unexpected P2 learning-curve experiment id")
    if config.get("status") != "authorized_local_research_only":
        raise ValueError("P2 learning-curve generation is not locally authorized")
    if config.get("research_only") is not True or config.get("upload_allowed") is not False:
        raise ValueError("P2 learning-curve generation must remain non-uploadable research")
    incumbent = config["official_incumbent"]
    if incumbent.get("submission_sha256") != (
        "1c959f818737850fd7fa9c6609ba3ae49dc9a470a269f7313119d840df1736bf"
    ):
        raise ValueError("official P2 incumbent identity changed")
    if incumbent.get("same_prefix_fold_train_only_refit_available") is not False:
        raise ValueError("this generation must disclose the missing exact incumbent refit")
    if incumbent.get("fail_closed_decision_if_unavailable") != (
        "SURROGATE_CURVE_NOT_PROMOTION_ELIGIBLE"
    ):
        raise ValueError("official-incumbent fail-closed decision changed")
    curve = config["learning_curve"]
    if curve.get("eligible_prefix_fractions") != [0.4, 0.55, 0.7, 0.85, 1.0]:
        raise ValueError("fixed learning-curve fractions changed")
    if int(curve.get("embargo_days", 0)) != 7:
        raise ValueError("P2 embargo must remain seven days")
    if curve.get("official_layer_counts") != {"2": 8713, "3": 8712, "4": 8636}:
        raise ValueError("official P2 layer counts changed")
    if curve.get("bootstrap") != {
        "unit": "KST calendar day sampled within fold",
        "replicates": 5000,
        "interval": 0.9,
        "seed": 20260823,
    }:
        raise ValueError("P2 bootstrap contract changed")
    folds = curve.get("validation_folds")
    expected_folds = [
        {
            "name": "outer_2024_sep_oct",
            "outer": ["2024-09-01T00:00:00+09:00", "2024-11-01T00:00:00+09:00"],
            "same_season_priority": True,
        },
        {
            "name": "outer_2025_may_jun",
            "outer": ["2025-05-01T00:00:00+09:00", "2025-07-01T00:00:00+09:00"],
            "same_season_priority": False,
        },
        {
            "name": "outer_2025_jul_aug",
            "outer": ["2025-07-01T00:00:00+09:00", "2025-09-01T00:00:00+09:00"],
            "same_season_priority": False,
        },
    ]
    if folds != expected_folds:
        raise ValueError("P2 forward-fold membership changed")
    hypotheses = config.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) != 1:
        raise ValueError("this generation preregisters exactly one structural hypothesis")
    hypothesis = hypotheses[0]
    if hypothesis != {
        "priority": 1,
        "id": "H1_LAYERWISE_CATBOOST_PUBLIC_PROFILE",
        "structural_change": (
            "Replace the pooled two-arm LightGBM residual stack with independent ordered-boosting "
            "residual models for layers 2, 3, and 4, average three fixed seeds, then apply the same "
            "public-endpoint profile projection."
        ),
        "family": "catboost_layerwise",
        "backend": "catboost",
        "feature_arm": "fixed_lean_public_m2",
        "iterations": 400,
        "layerwise": True,
        "categorical_layer": False,
        "fixed_seeds": [20260823, 20260824, 20260825],
        "hyperparameter_searches": 0,
        "target_layer_inputs": [],
    }:
        raise ValueError("preregistered structural hypothesis changed")
    if config["next_generation_if_no_pass"].get("count") != 1:
        raise ValueError("exactly one next generation must be diagnosed")
    output = config["output_contract"]
    if output.get("append_only") is not True or output.get("upload_allowed") is not False:
        raise ValueError("output contract is not append-only and local-only")
    if output.get("row_level_prediction_artifacts_allowed") is not False:
        raise ValueError("row-level prediction artifacts must remain forbidden")
    if output.get("directory") != "artifacts/p2_meaningful_learning_curve_generation_v1":
        raise ValueError("canonical output directory changed")
    if output.get("attempt_lock") != (
        "artifacts/p2_meaningful_learning_curve_generation_v1_control/attempt.lock"
    ):
        raise ValueError("canonical attempt lock changed")


def _git_state() -> dict[str, Any]:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {"sha": sha, "branch": branch, "dirty": dirty}


def _snapshot(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {
        _logical(path): {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in paths
    }


def _verify_sources(config: Mapping[str, Any], data_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for name, expected in config["source_contract"].items():
        path = data_dir / name
        observed = _sha256(path)
        if observed != expected:
            raise ValueError(f"P2 source SHA changed: {name}")
        records[name] = {"sha256": observed, "bytes": path.stat().st_size}
    return records


def _official_incumbent_audit() -> dict[str, Any]:
    evidence_paths = [
        ROOT / "configs/experiments/p2_extrapolated_soft_gate_v2.json",
        ROOT / "artifacts/p2_extrapolated_soft_gate_v2/manifest.json",
        ROOT / "scripts/run_p2_extrapolated_soft_gate_v2.py",
        ROOT / "configs/experiments/p2_public_state_soft_gate_v1.json",
        ROOT / "scripts/run_p2_public_state_soft_gate.py",
        ROOT / "configs/experiments/p2_deep_finalists_v1.json",
        ROOT / "artifacts/p2_deep_finalists_v1/result.json",
        ROOT / "scripts/run_p2_deep_finalists.py",
    ]
    for path in evidence_paths:
        if not path.is_file():
            raise FileNotFoundError(f"official-incumbent structure evidence is absent: {path}")
    return {
        "official_submission": {
            "path": "output/2026-08-20/ready/P2_submission.csv",
            "sha256": _sha256(ROOT / "output/2026-08-20/ready/P2_submission.csv"),
        },
        "same_prefix_fold_train_only_refit_available": False,
        "reason_codes": [
            "CONTRIBUTOR_OOF_DEFINED_ON_THREE_EXPOSED_BLOCKS_NOT_ARBITRARY_FORWARD_PREFIXES",
            "LAYER_STACK_WEIGHTS_FIT_FROM_EXPOSED_CONTRIBUTOR_OOF",
            "PUBLIC_STATE_GATE_FIT_FROM_EXPOSED_CONTRIBUTOR_OOF",
            "NO_SEALED_PREFIX_ONLY_META_TRAINING_MAPPING",
            "INVENTED_MAPPING_WOULD_NOT_BE_EXACT_IMMUTABLE_INCUMBENT_STRUCTURE",
        ],
        "structure_evidence": {
            _logical(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in evidence_paths
        },
        "decision_constraint": "SURROGATE_CURVE_NOT_PROMOTION_ELIGIBLE",
    }


def _subset(table: FeatureTable, selected: np.ndarray) -> FeatureTable:
    return FeatureTable(table.frame.loc[selected].reset_index(drop=True), table.feature_columns)


def _key_digest(frame: pd.DataFrame) -> str:
    keys = frame.loc[:, [*KEY_COLUMNS, "fold"]].copy()
    return hashlib.sha256(keys.to_csv(index=False, lineterminator="\n").encode()).hexdigest()


def _coverage(base: FeatureTable, start: str, stop: str) -> dict[str, Any]:
    eligible = window_mask(base.frame, start, stop)
    finite = eligible & np.isfinite(base.frame["target"].to_numpy(float))
    nominal = nominal_target_rows(start, stop)
    return {
        "nominal_rows": nominal,
        "eligible_feature_rows": int(eligible.sum()),
        "finite_target_rows": int(finite.sum()),
        "target_coverage": float(finite.sum() / nominal),
        "rows_by_layer": {
            str(layer): int((finite & base.frame["layer"].eq(layer).to_numpy()).sum())
            for layer in TARGET_LAYERS
        },
    }


def _training_summary(
    frame: pd.DataFrame,
    selected: np.ndarray,
    *,
    cutoff: pd.Timestamp,
    boundary: str,
) -> dict[str, Any]:
    time_values = pd.to_datetime(frame.loc[selected, "time"], utc=True)
    if time_values.empty or not time_values.lt(cutoff).all():
        raise AssertionError("training prefix crossed the embargo cutoff")
    return {
        "rows": int(selected.sum()),
        "rows_by_layer": {
            str(layer): int((selected & frame["layer"].eq(layer).to_numpy()).sum())
            for layer in TARGET_LAYERS
        },
        "unique_timestamps": int(time_values.nunique()),
        "first_label_time_utc": time_values.min().isoformat(),
        "last_label_time_utc": time_values.max().isoformat(),
        "prefix_boundary_utc": boundary,
        "exclusive_embargo_cutoff_utc": cutoff.isoformat(),
        "max_label_precedes_cutoff": True,
    }


def _surrogate_prediction(
    base: FeatureTable,
    lean: FeatureTable,
    selected: np.ndarray,
    base_window: FeatureTable,
    lean_window: FeatureTable,
    endpoints: pd.DataFrame,
    *,
    seeds: list[int],
    stack_weight: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    models = [fit_fixed_blend(base, lean, selected, seed=seed) for seed in seeds]
    first = np.vstack([model.predict(base_window, lean_window) for model in models])
    repeated = np.vstack([model.predict(base_window, lean_window) for model in models])
    repeat_error = float(np.max(np.abs(first - repeated)))
    baseline = base_window.frame["baseline"].to_numpy(float)
    seed_predictions = np.vstack(
        [
            baseline
            + stack_weight
            * (project_profiles_vectorized(base_window.frame, raw, endpoints).prediction - baseline)
            for raw in first
        ]
    )
    prediction = np.mean(seed_predictions, axis=0, dtype=np.float64)
    del models
    gc.collect()
    return prediction, repeat_error, seed_predictions


def _candidate_prediction(
    lean: FeatureTable,
    selected: np.ndarray,
    lean_window: FeatureTable,
    endpoints: pd.DataFrame,
    *,
    hypothesis: Mapping[str, Any],
) -> tuple[np.ndarray, float, np.ndarray]:
    spec = GBMArmSpec(
        str(hypothesis["family"]),
        str(hypothesis["backend"]),
        iterations=int(hypothesis["iterations"]),
        layerwise=bool(hypothesis["layerwise"]),
        categorical_layer=bool(hypothesis["categorical_layer"]),
    )
    models = [
        fit_gbm_model(lean, spec, selected, seed=int(seed)) for seed in hypothesis["fixed_seeds"]
    ]
    first = np.vstack([model.predict(lean_window) for model in models])
    repeated = np.vstack([model.predict(lean_window) for model in models])
    repeat_error = float(np.max(np.abs(first - repeated)))
    seed_predictions = np.vstack(
        [project_profiles_vectorized(lean_window.frame, raw, endpoints).prediction for raw in first]
    )
    prediction = np.mean(seed_predictions, axis=0, dtype=np.float64)
    del models
    gc.collect()
    return prediction, repeat_error, seed_predictions


def _scored_frame(
    base_window: FeatureTable,
    surrogate: np.ndarray,
    candidate: np.ndarray,
    surrogate_seeds: np.ndarray,
    candidate_seeds: np.ndarray,
    *,
    fold: str,
) -> pd.DataFrame:
    frame = base_window.frame.loc[:, ["station", "layer", "time", "target", "baseline"]].copy()
    frame = frame.rename(columns={"target": "truth"})
    frame["surrogate_prediction"] = np.asarray(surrogate, dtype=float)
    frame["candidate_prediction"] = np.asarray(candidate, dtype=float)
    if surrogate_seeds.shape != (3, len(frame)) or candidate_seeds.shape != (3, len(frame)):
        raise ValueError("three seed predictions must align to the scored window")
    for seed_index in range(3):
        frame[f"surrogate_seed_{seed_index}"] = surrogate_seeds[seed_index]
        frame[f"candidate_seed_{seed_index}"] = candidate_seeds[seed_index]
    frame["fold"] = fold
    frame["kst_day"] = (
        pd.to_datetime(frame["time"], utc=True).dt.tz_convert(KST).dt.strftime("%Y-%m-%d")
    )
    frame = frame.loc[np.isfinite(frame["truth"])].reset_index(drop=True)
    numeric_columns = [
        "truth",
        "baseline",
        "surrogate_prediction",
        "candidate_prediction",
        *[f"surrogate_seed_{index}" for index in range(3)],
        *[f"candidate_seed_{index}" for index in range(3)],
    ]
    numeric = frame.loc[:, numeric_columns]
    if frame.empty or not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError(f"fold {fold} produced invalid scored predictions")
    if frame.duplicated(KEY_COLUMNS).any():
        raise ValueError(f"fold {fold} produced duplicate scored keys")
    return frame


def _slice_deltas(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    result = {
        f"layer_{layer}": fold_equal_layer_rmse(candidate, layer)
        - fold_equal_layer_rmse(reference, layer)
        for layer in TARGET_LAYERS
    }
    same_season = "outer_2024_sep_oct"
    result["2024_sep_oct"] = float(
        candidate["by_fold"][same_season]["official_layer_weighted_rmse_c"]
        - reference["by_fold"][same_season]["official_layer_weighted_rmse_c"]
    )
    return result


def _run(
    config: Mapping[str, Any],
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
) -> int:
    started = time.perf_counter()
    if output_dir.exists():
        raise FileExistsError(f"append-only output already exists: {output_dir}")
    if DEFAULT_LOCK.exists():
        raise FileExistsError(f"one-shot attempt lock already exists: {DEFAULT_LOCK}")

    frozen_paths = [
        ROOT / "output/2026-08-20/ready/P2_submission.csv",
        ROOT / "output/2026-08-20/saved_weight/P2_submission.csv",
        ROOT / "submissions/p2/P2_EXTRAPOLATED_SOFT_GATE_V2.csv",
    ]
    frozen_before = _snapshot(frozen_paths)
    if (
        frozen_before["output/2026-08-20/ready/P2_submission.csv"]["sha256"]
        != (config["official_incumbent"]["submission_sha256"])
    ):
        raise ValueError("official immutable P2 incumbent SHA mismatch")
    source_records = _verify_sources(config, data_dir)
    incumbent_audit = _official_incumbent_audit()
    if (
        incumbent_audit["official_submission"]["sha256"]
        != (config["official_incumbent"]["submission_sha256"])
    ):
        raise ValueError("official incumbent audit identity mismatch")

    attempt = {
        "schema_version": "p2_meaningful_learning_curve_attempt.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now(),
        "config_path": _logical(config_path),
        "config_sha256": _sha256(config_path),
        "official_incumbent_sha256": config["official_incumbent"]["submission_sha256"],
        "upload_allowed": False,
        "frozen_mutation_allowed": False,
    }
    _exclusive_json(DEFAULT_LOCK, attempt)
    print(json.dumps({"progress": 2, "phase": "attempt_locked", "detail": "append-only"}))

    data = load_p2_data(data_dir)
    masked, mask_audit = joint_mask_target_context(data.observations)
    base = build_joint_masked_population(data.observations, masked)
    lean = build_fixed_lean_arm(base, masked)
    endpoints = public_endpoints_from_masked_context(masked)
    forbidden = {
        "target",
        "residual",
        "temp_2",
        "temp_3",
        "temp_4",
        "psal_2",
        "psal_3",
        "psal_4",
    }
    forbidden_absent = not bool(forbidden.intersection(lean.feature_columns))
    if not forbidden_absent:
        raise AssertionError("target labels or target-layer values entered model features")
    if not base.frame[KEY_COLUMNS].equals(lean.frame[KEY_COLUMNS]):
        raise AssertionError("base and lean population keys differ")
    print(
        json.dumps(
            {
                "progress": 8,
                "phase": "joint_masked_features",
                "base_features": len(base.feature_columns),
                "lean_features": len(lean.feature_columns),
            }
        )
    )

    fractions = [float(value) for value in config["learning_curve"]["eligible_prefix_fractions"]]
    embargo_days = int(config["learning_curve"]["embargo_days"])
    prefix_contracts: dict[str, Any] = {}
    prefix_masks: dict[tuple[str, float], np.ndarray] = {}
    prefix_nested = True
    training_precedes_embargo = True
    validation_disjoint = True
    for fold in config["learning_curve"]["validation_folds"]:
        name = str(fold["name"])
        start, stop = fold["outer"]
        coverage = _coverage(base, start, stop)
        if coverage["target_coverage"] < 0.96:
            raise ValueError(f"fold {name} violates the target coverage floor")
        eligible, cutoff = forward_training_mask(base.frame, start, embargo_days=embargo_days)
        masks, boundaries = chronological_prefix_masks(base.frame, eligible, fractions)
        previous = np.zeros(len(base.frame), dtype=bool)
        fold_contract: dict[str, Any] = {
            "outer_window_kst": [start, stop],
            "coverage": coverage,
            "prefixes": {},
        }
        validation = window_mask(base.frame, start, stop)
        for fraction in fractions:
            selected = masks[fraction]
            prefix_masks[(name, fraction)] = selected
            prefix_nested &= bool(np.all(~previous | selected))
            training_precedes_embargo &= bool(
                pd.to_datetime(base.frame.loc[selected, "time"], utc=True).lt(cutoff).all()
            )
            validation_disjoint &= not bool(np.any(selected & validation))
            fold_contract["prefixes"][str(fraction)] = _training_summary(
                base.frame,
                selected,
                cutoff=cutoff,
                boundary=boundaries[fraction],
            )
            previous = selected
        prefix_contracts[name] = fold_contract

    hypothesis = config["hypotheses"][0]
    surrogate = config["surrogate_incumbent"]
    seeds = [int(value) for value in surrogate["fixed_seeds"]]
    if seeds != [20260823, 20260824, 20260825] or list(hypothesis["fixed_seeds"]) != seeds:
        raise ValueError("three fixed stochastic seeds changed")
    layer_counts = config["learning_curve"]["official_layer_counts"]
    bootstrap_config = config["learning_curve"]["bootstrap"]
    curve_points: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    key_digests: list[str] = []
    repeat_errors: list[float] = []

    for point_number, fraction in enumerate(fractions, 1):
        parts: list[pd.DataFrame] = []
        for fold_number, fold in enumerate(config["learning_curve"]["validation_folds"], 1):
            name = str(fold["name"])
            start, stop = fold["outer"]
            validation = window_mask(base.frame, start, stop)
            base_window = _subset(base, validation)
            lean_window = _subset(lean, validation)
            selected = prefix_masks[(name, fraction)]
            print(
                json.dumps(
                    {
                        "progress": 8 + int(76 * ((point_number - 1) * 3 + fold_number - 1) / 15),
                        "phase": "fit_curve_point",
                        "fraction": fraction,
                        "fold": name,
                        "aggregate_training_rows": int(selected.sum()),
                    }
                )
            )
            surrogate_prediction, surrogate_error, surrogate_seed_predictions = (
                _surrogate_prediction(
                    base,
                    lean,
                    selected,
                    base_window,
                    lean_window,
                    endpoints,
                    seeds=seeds,
                    stack_weight=float(surrogate["post_projection_candidate_weight"]),
                )
            )
            candidate_prediction, candidate_error, candidate_seed_predictions = (
                _candidate_prediction(
                    lean,
                    selected,
                    lean_window,
                    endpoints,
                    hypothesis=hypothesis,
                )
            )
            repeat_errors.extend([surrogate_error, candidate_error])
            parts.append(
                _scored_frame(
                    base_window,
                    surrogate_prediction,
                    candidate_prediction,
                    surrogate_seed_predictions,
                    candidate_seed_predictions,
                    fold=name,
                )
            )
        scored = pd.concat(parts, ignore_index=True)
        digest = _key_digest(scored)
        key_digests.append(digest)
        interpolation_report = metric_report(
            scored, prediction_column="baseline", official_layer_counts=layer_counts
        )
        surrogate_report = metric_report(
            scored,
            prediction_column="surrogate_prediction",
            official_layer_counts=layer_counts,
        )
        candidate_report = metric_report(
            scored,
            prediction_column="candidate_prediction",
            official_layer_counts=layer_counts,
        )
        incumbent_seed_metrics = [
            float(
                metric_report(
                    scored,
                    prediction_column=f"surrogate_seed_{seed_index}",
                    official_layer_counts=layer_counts,
                )["fold_equal_official_layer_weighted_rmse_c"]
            )
            for seed_index in range(3)
        ]
        challenger_seed_metrics = [
            float(
                metric_report(
                    scored,
                    prediction_column=f"candidate_seed_{seed_index}",
                    official_layer_counts=layer_counts,
                )["fold_equal_official_layer_weighted_rmse_c"]
            )
            for seed_index in range(3)
        ]
        bootstrap = paired_fold_day_bootstrap(
            scored,
            reference_column="surrogate_prediction",
            candidate_column="candidate_prediction",
            official_layer_counts=layer_counts,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]),
            interval=float(bootstrap_config["interval"]),
        )
        interpolation_bootstrap = paired_fold_day_bootstrap(
            scored,
            reference_column="baseline",
            candidate_column="candidate_prediction",
            official_layer_counts=layer_counts,
            replicates=int(bootstrap_config["replicates"]),
            seed=int(bootstrap_config["seed"]) + 1,
            interval=float(bootstrap_config["interval"]),
        )
        curve_points.append(
            {
                "fraction": fraction,
                "incumbent": float(surrogate_report["fold_equal_official_layer_weighted_rmse_c"]),
                "challenger": float(candidate_report["fold_equal_official_layer_weighted_rmse_c"]),
                "delta_ci90": [float(value) for value in bootstrap["delta_interval"]],
                "incumbent_seed_metrics": incumbent_seed_metrics,
                "challenger_seed_metrics": challenger_seed_metrics,
            }
        )
        diagnostics[str(fraction)] = {
            "comparison_role": "diagnostic_same_prefix_surrogate_not_official_incumbent",
            "interpolation": interpolation_report,
            "surrogate_incumbent": surrogate_report,
            "challenger": candidate_report,
            "challenger_vs_surrogate_bootstrap": bootstrap,
            "challenger_vs_interpolation_bootstrap": interpolation_bootstrap,
            "incumbent_seed_metrics": incumbent_seed_metrics,
            "challenger_seed_metrics": challenger_seed_metrics,
            "validation_key_sha256": digest,
        }
        del scored, parts
        gc.collect()

    full = diagnostics["1.0"]
    reference_full = full["surrogate_incumbent"]
    candidate_full = full["challenger"]
    fold_order = [str(fold["name"]) for fold in config["learning_curve"]["validation_folds"]]
    fold_deltas = [
        float(
            candidate_full["by_fold"][name]["official_layer_weighted_rmse_c"]
            - reference_full["by_fold"][name]["official_layer_weighted_rmse_c"]
        )
        for name in fold_order
    ]
    slice_deltas = _slice_deltas(reference_full, candidate_full)
    numeric_gates = numeric_curve_gate(
        curve_points,
        fold_deltas=fold_deltas,
        slice_deltas=slice_deltas,
        maximum_slice_regression_c=float(
            config["numeric_pass_gates"]["maximum_each_layer_regression_c"]
        ),
        full_effect_c=float(
            config["numeric_pass_gates"]["full_delta_candidate_minus_incumbent_at_most_c"]
        ),
    )
    leakage_checks = {
        "joint_target_temperature_context_fully_masked": (
            mask_audit.target_temp_non_null_after_mask == 0
        ),
        "joint_target_salinity_context_fully_masked": (
            mask_audit.target_psal_non_null_after_mask == 0
        ),
        "hidden_target_temperature_absent_before_mask": (
            mask_audit.hidden_temp_non_null_before_mask == 0
        ),
        "hidden_target_salinity_absent_before_mask": (
            mask_audit.hidden_psal_non_null_before_mask == 0
        ),
        "target_labels_and_target_layer_values_absent_from_features": forbidden_absent,
        "all_training_rows_precede_seven_day_embargo": training_precedes_embargo,
        "chronological_prefixes_nested": prefix_nested,
        "validation_rows_disjoint_from_training": validation_disjoint,
        "source_hashes_match_preregistration": _verify_sources(config, data_dir) == source_records,
    }
    reproducibility_checks = {
        "three_fixed_stochastic_seeds_used": seeds == [20260823, 20260824, 20260825],
        "repeat_inference_exact": max(repeat_errors, default=float("inf")) == 0.0,
        "same_validation_keys_all_fractions": len(set(key_digests)) == 1,
        "same_fold_metric_all_fractions": True,
        "same_public_endpoint_postprocess_all_fractions": True,
        "official_incumbent_exact_prefix_refit_available": False,
    }
    evidence = {
        "schema_version": "meaningful_learning_curve_evidence.v1",
        "problem": "P2",
        "experiment_id": config["experiment_id"],
        "hypothesis_id": hypothesis["id"],
        "preregistration": {
            "generation_id": config["experiment_id"],
            "config_path": _logical(config_path),
            "config_sha256": _sha256(config_path),
            "created_before_first_fit": True,
            "hypothesis_count": len(config["hypotheses"]),
            "score_derived_tuning": False,
        },
        "curve_protocol": {
            "prefix_fractions": fractions,
            "seed_ids": seeds,
            "seed_aggregation": "PREDICTION_MEAN_THEN_METRIC",
            "bootstrap_replicates": int(bootstrap_config["replicates"]),
            "bootstrap_cluster": "KST_day",
            "incumbent_fresh_refit_each_prefix": False,
            "surrogate_incumbent_fresh_refit_each_prefix": True,
            "challenger_fresh_refit_each_prefix": True,
            "same_fold_keys_metric_postprocess": False,
            "surrogate_same_fold_keys_metric_public_projection": True,
            "incumbent_reference_seed_full_prediction_exact_to_frozen_oof": False,
        },
        "incumbent_identity_required": config["official_incumbent"]["identity"],
        "incumbent_curve_role": "SURROGATE_ONLY_NOT_OFFICIAL_IMMUTABLE_INCUMBENT",
        "points": curve_points,
        "fold_order": fold_order,
        "fold_deltas_candidate_minus_incumbent": fold_deltas,
        "slice_deltas_candidate_minus_incumbent": slice_deltas,
        "leakage_checks": leakage_checks,
        "reproducibility_checks": reproducibility_checks,
        "upload_count": 0,
    }
    numeric_surrogate_pass = (
        all(numeric_gates.values())
        and all(leakage_checks.values())
        and all(
            value
            for key, value in reproducibility_checks.items()
            if key != "official_incumbent_exact_prefix_refit_available"
        )
    )
    decision = "SURROGATE_CURVE_NOT_PROMOTION_ELIGIBLE"
    result = {
        "schema_version": "p2_meaningful_learning_curve_generation.result.v1",
        "experiment_id": config["experiment_id"],
        "completed_at_kst": _now(),
        "elapsed_seconds": float(time.perf_counter() - started),
        "status": decision,
        "hypothesis_id": hypothesis["id"],
        "official_incumbent_exact_prefix_refit_available": False,
        "numeric_surrogate_pass": numeric_surrogate_pass,
        "numeric_gates": numeric_gates,
        "confirmed_meaningful_generalization_improvement": False,
        "research_only": True,
        "do_not_promote": True,
        "full_fit_performed": False,
        "candidate_generated": False,
        "candidate_rows": 0,
        "hidden_target_temperature_values_accessed": 0,
        "hidden_target_salinity_values_accessed": 0,
        "frozen_submission_modified": False,
        "upload_count": 0,
        "upload_performed": False,
        "next_generation": config["next_generation_if_no_pass"],
    }
    metrics = {
        "schema_version": "p2_meaningful_learning_curve_generation.metrics.v1",
        "experiment_id": config["experiment_id"],
        "hypothesis": hypothesis,
        "official_incumbent_refit_audit": incumbent_audit,
        "surrogate_incumbent": surrogate,
        "curve_diagnostics": diagnostics,
        "numeric_gates": numeric_gates,
        "numeric_surrogate_pass": numeric_surrogate_pass,
        "mask_audit": mask_audit.__dict__,
        "prefix_contracts": prefix_contracts,
        "maximum_repeat_inference_abs_error_c": max(repeat_errors, default=float("inf")),
        "validation_key_sha256": key_digests[0],
        "row_level_prediction_artifacts_written": 0,
    }

    frozen_after = _snapshot(frozen_paths)
    if frozen_after != frozen_before:
        raise AssertionError("a frozen/current P2 submission changed")

    output_dir.mkdir(parents=True, exist_ok=False)
    evidence_path = output_dir / config["output_contract"]["aggregate_learning_curve_evidence"]
    metrics_path = output_dir / "metrics.json"
    result_path = output_dir / "result.json"
    _exclusive_json(evidence_path, evidence)
    _exclusive_json(metrics_path, metrics)
    _exclusive_json(result_path, result)
    for path in (evidence_path, metrics_path, result_path):
        _seal_sidecar(path)

    implementation_paths = [
        config_path,
        Path(__file__).resolve(),
        ROOT / "src/p2_restore/meaningful_learning_curve.py",
        ROOT / "src/p2_restore/corrected_repeated_forward.py",
        ROOT / "src/p2_restore/gbm_tournament.py",
        ROOT / "src/p2_restore/profile_projection.py",
    ]
    manifest = {
        "schema_version": "p2_meaningful_learning_curve_generation.manifest.v1",
        "experiment_id": config["experiment_id"],
        "created_at_kst": _now(),
        "append_only": True,
        "research_only": True,
        "do_not_promote": True,
        "upload_allowed": False,
        "config": {
            "path": _logical(config_path),
            "sha256": _sha256(config_path),
            "bytes": config_path.stat().st_size,
        },
        "sources": source_records,
        "official_incumbent_refit_audit": incumbent_audit,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "frozen_unchanged": True,
        "implementation": {
            _logical(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in implementation_paths
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {
                package: metadata.version(package)
                for package in [
                    "numpy",
                    "pandas",
                    "lightgbm",
                    "catboost",
                    "scikit-learn",
                    "pyarrow",
                ]
            },
        },
        "git": _git_state(),
        "attempt_lock": attempt,
        "artifacts": {
            path.name: {
                "path": _logical(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in (evidence_path, metrics_path, result_path)
        },
        "row_level_prediction_artifacts_written": 0,
        "full_model_fits": 0,
        "candidate_rows": 0,
        "upload_count": 0,
        "upload_performed": False,
        "elapsed_seconds": result["elapsed_seconds"],
    }
    manifest_path = output_dir / "manifest.json"
    _exclusive_json(manifest_path, manifest)
    _seal_sidecar(manifest_path)
    seal = {
        "schema_version": "p2_meaningful_learning_curve_generation.seal.v1",
        "experiment_id": config["experiment_id"],
        "sealed_at_kst": _now(),
        "decision": decision,
        "numeric_surrogate_pass": numeric_surrogate_pass,
        "official_incumbent_exact_prefix_refit_available": False,
        "learning_curve_evidence_sha256": _sha256(evidence_path),
        "metrics_sha256": _sha256(metrics_path),
        "result_sha256": _sha256(result_path),
        "manifest_sha256": _sha256(manifest_path),
        "frozen_submission_modified": False,
        "upload_count": 0,
        "upload_performed": False,
    }
    seal_path = output_dir / "seal.json"
    _exclusive_json(seal_path, seal)
    _seal_sidecar(seal_path)
    print(
        json.dumps(
            {
                "progress": 100,
                "phase": "sealed",
                "decision": decision,
                "numeric_surrogate_pass": numeric_surrogate_pass,
                "result": _logical(result_path),
                "seal_sha256": _sha256(seal_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


def _dry_run(config: Mapping[str, Any], config_path: Path, data_dir: Path, output_dir: Path) -> int:
    if output_dir.exists() or DEFAULT_LOCK.exists():
        raise FileExistsError("canonical append-only output or attempt lock already exists")
    sources = _verify_sources(config, data_dir)
    incumbent = _official_incumbent_audit()
    data = load_p2_data(data_dir)
    masked, audit = joint_mask_target_context(data.observations)
    base = build_joint_masked_population(data.observations, masked)
    lean = build_fixed_lean_arm(base, masked)
    print(
        json.dumps(
            {
                "status": "DRY_RUN_READY",
                "config_sha256": _sha256(config_path),
                "source_hashes_match": len(sources) == len(config["source_contract"]),
                "official_incumbent_exact_prefix_refit_available": incumbent[
                    "same_prefix_fold_train_only_refit_available"
                ],
                "mask_audit": audit.__dict__,
                "base_feature_count": len(base.feature_columns),
                "lean_feature_count": len(lean.feature_columns),
                "output_exists": output_dir.exists(),
                "attempt_lock_exists": DEFAULT_LOCK.exists(),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    if config_path != DEFAULT_CONFIG.resolve() or output_dir != DEFAULT_OUTPUT.resolve():
        raise ValueError("canonical config and output paths are required")
    config = _load_json(config_path)
    _validate_config(config)
    data_dir = resolve_data_dir(args.data_dir)
    if args.dry_run:
        return _dry_run(config, config_path, data_dir, output_dir)
    return _run(config, config_path, data_dir, output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
