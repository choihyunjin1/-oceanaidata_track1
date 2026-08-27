"""Run the preregistered NASA POWER incremental P2 residual one-shot.

The experiment is deliberately local-only.  It never constructs a hidden-period
prediction or a submission: three frozen seasonal OOF blocks are used to compare
the same small residual model with and without public hourly meteorology.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from p2_restore.data import load_p2_data, resolve_data_dir
from p2_restore.external_meteorology import (
    POWER_PARAMETERS,
    build_power_features,
    finite_coverage,
    load_power_hourly,
    summarize_power_quality,
)
from p2_restore.regime_gate import STATE_FEATURES, build_public_state_features

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "p2_nasa_power_residual_meta_v1"
OUTER_BLOCKS = ("2024_sep_oct", "2025_jul_aug", "2025_nov_dec")
CONTROL_FEATURES = (*STATE_FEATURES, "layer_2", "layer_3", "layer_4")
MODEL_KEYS = {
    "objective",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "random_state",
    "deterministic",
    "force_col_wise",
    "n_jobs",
    "verbosity",
}


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class Progress:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.started = time.perf_counter()

    def update(
        self,
        progress: float,
        phase: str,
        detail: str,
        *,
        status: str = "running",
        extra: dict[str, Any] | None = None,
    ) -> None:
        elapsed = time.perf_counter() - self.started
        bounded = min(max(float(progress), 0.1), 100.0)
        remaining = elapsed * (100.0 - bounded) / bounded if bounded < 100 else 0.0
        eta = datetime.now().astimezone() + timedelta(seconds=max(remaining, 0.0))
        payload: dict[str, Any] = {
            "title": "P2 NASA POWER residual meta-model one-shot",
            "experiment_id": EXPERIMENT_ID,
            "status": status,
            "progress": bounded,
            "phase": phase,
            "detail": detail,
            "elapsed_seconds": elapsed,
            "eta": eta.strftime("%Y-%m-%d %H:%M:%S KST"),
            "updated_at": datetime.now().astimezone().isoformat(),
        }
        if extra:
            payload.update(extra)
        _write_json(self.path, payload)


def _load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected NASA POWER experiment id")
    if contract.get("status") != "authorized_local_external_one_shot":
        raise ValueError("NASA POWER one-shot is not authorized")
    if contract.get("research_only") is not True or contract.get("upload_allowed") is not False:
        raise ValueError("NASA POWER experiment must remain research-only and non-uploadable")
    if contract["external_data_policy"].get("hidden_target_values_forbidden") is not True:
        raise ValueError("hidden-target prohibition is not active")
    if tuple(contract["expected_power_contract"]["parameters"]) != POWER_PARAMETERS:
        raise ValueError("NASA POWER parameter contract changed")
    if tuple(contract["validation"]["outer_blocks"]) != OUTER_BLOCKS:
        raise ValueError("seasonal outer blocks changed")
    alpha_grid = tuple(float(value) for value in contract["validation"]["alpha_grid"])
    if (
        len(alpha_grid) < 2
        or len(alpha_grid) > 7
        or tuple(sorted(set(alpha_grid))) != alpha_grid
        or alpha_grid[0] != 0.0
        or alpha_grid[-1] > 1.0
    ):
        raise ValueError("alpha grid is not the preregistered small ordered grid")
    if tuple(contract["features"]["control"]) != CONTROL_FEATURES:
        raise ValueError("control public-state features changed")
    model = dict(contract["model"])
    if model.pop("family", None) != "lightgbm.LGBMRegressor" or set(model) != MODEL_KEYS:
        raise ValueError("fixed LightGBM contract changed or contains a parameter grid")
    if model["n_jobs"] != 1 or model["deterministic"] is not True:
        raise ValueError("LightGBM must remain single-threaded and deterministic")
    if len(contract.get("raw_files", ())) != 2:
        raise ValueError("exactly two quarantined NASA POWER files are required")
    return contract


def _validate_raw_entry(entry: dict[str, Any]) -> dict[str, Any]:
    raw_path = _repo_path(entry["path"])
    manifest_path = _repo_path(entry["manifest_path"])
    if not raw_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("quarantined NASA POWER raw file or manifest is missing")
    raw_hash = _sha256(raw_path)
    manifest_hash = _sha256(manifest_path)
    if raw_hash != entry["file_sha256"]:
        raise ValueError(f"raw NASA POWER SHA changed: {raw_path.name}")
    if manifest_hash != entry["manifest_sha256"]:
        raise ValueError(f"NASA POWER manifest SHA changed: {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": "1.0",
        "source_id": "nasa_power_kors_meteorology",
        "local_file": entry["path"],
        "file_sha256": entry["file_sha256"],
        "observed_start": entry["observed_start"],
        "observed_end": entry["observed_end"],
        "row_count": int(entry["row_count"]),
        "variables": list(POWER_PARAMETERS),
        "transformation_log": entry["transformation_log"],
    }
    if manifest != expected:
        raise ValueError(f"NASA POWER manifest contract changed: {manifest_path.name}")
    return {
        "year": int(entry["year"]),
        "path": entry["path"],
        "bytes": raw_path.stat().st_size,
        "sha256": raw_hash,
        "manifest_path": entry["manifest_path"],
        "manifest_sha256": manifest_hash,
        "observed_start": entry["observed_start"],
        "observed_end": entry["observed_end"],
        "row_count": int(entry["row_count"]),
    }


def _validate_power_quality(hourly: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    expected = contract["expected_power_contract"]
    quality = summarize_power_quality(hourly)
    observed = {
        "rows": quality.rows,
        "start_utc": quality.start_utc,
        "end_utc": quality.end_utc,
        "duplicate_timestamps": quality.duplicate_timestamps,
        "missing_values": quality.missing_values,
        "non_hourly_gaps": quality.non_hourly_gaps,
        "maximum_gap_hours": quality.maximum_gap_hours,
    }
    exact = {
        "rows": int(expected["rows_after_cutoff"]),
        "start_utc": expected["start_utc"],
        "end_utc": expected["end_utc"],
    }
    for name, value in exact.items():
        if observed[name] != value:
            raise ValueError(f"NASA POWER post-cutoff {name} changed")
    if quality.missing_values > int(expected["maximum_missing_values"]):
        raise ValueError("NASA POWER contains unexpected missing values")
    if quality.duplicate_timestamps > int(expected["maximum_duplicate_timestamps"]):
        raise ValueError("NASA POWER contains duplicate timestamps")
    if quality.non_hourly_gaps > int(expected["maximum_non_hourly_gaps"]):
        raise ValueError("NASA POWER contains non-hourly gaps")
    if not np.isclose(quality.maximum_gap_hours, 1.0, rtol=0, atol=0):
        raise ValueError("NASA POWER maximum interval is not exactly one hour")
    return observed


def _rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction_array - truth_array) ** 2)))


def _load_frozen_oof(contract: dict[str, Any]) -> pd.DataFrame:
    frozen = contract["frozen_incumbent"]
    path = _repo_path(frozen["path"])
    if _sha256(path) != frozen["sha256"]:
        raise ValueError("frozen P2 incumbent OOF SHA changed")
    frame = pd.read_parquet(path)
    required = {"time", "layer", "truth", "block", frozen["prediction_column"]}
    if not required.issubset(frame.columns) or len(frame) != int(frozen["rows"]):
        raise ValueError("frozen P2 incumbent OOF schema or row count changed")
    frame = frame.loc[:, ["time", "layer", "truth", "block", frozen["prediction_column"]]].copy()
    frame = frame.rename(columns={frozen["prediction_column"]: "incumbent_prediction"})
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="raise")
    if frame[["time", "layer"]].duplicated().any():
        raise ValueError("frozen P2 incumbent OOF contains duplicate keys")
    if set(frame["block"]) != set(OUTER_BLOCKS) or set(frame["layer"].astype(int)) != {2, 3, 4}:
        raise ValueError("frozen P2 incumbent OOF block or layer set changed")
    counts = frame["block"].value_counts().to_dict()
    if counts != {name: int(value) for name, value in frozen["expected_block_rows"].items()}:
        raise ValueError("frozen P2 incumbent OOF block counts changed")
    numeric = frame[["truth", "incumbent_prediction"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("frozen P2 incumbent OOF contains non-finite values")
    incumbent_rmse = _rmse(frame["truth"], frame["incumbent_prediction"])
    if not np.isclose(incumbent_rmse, float(frozen["expected_rmse"]), rtol=0, atol=1e-12):
        raise ValueError("frozen P2 incumbent OOF RMSE changed")
    local_time = frame["time"].dt.tz_convert("Asia/Seoul")
    hidden = local_time.ge(pd.Timestamp("2025-09-01", tz="Asia/Seoul")) & local_time.lt(
        pd.Timestamp("2025-11-01", tz="Asia/Seoul")
    )
    if hidden.any():
        raise ValueError("frozen OOF unexpectedly includes the official hidden target interval")
    return frame.reset_index(drop=True)


def _reconcile_oof_truth(observations: pd.DataFrame, oof: pd.DataFrame) -> None:
    observed = observations.loc[:, ["time", "layer", "temp"]].copy()
    observed["time"] = pd.to_datetime(observed["time"], utc=True, errors="raise")
    observed = observed.rename(columns={"temp": "observed_truth"})
    merged = oof.loc[:, ["time", "layer", "truth"]].merge(
        observed,
        on=["time", "layer"],
        how="left",
        validate="one_to_one",
    )
    if len(merged) != len(oof) or not np.isfinite(merged["observed_truth"]).all():
        raise ValueError("frozen OOF keys do not reconcile to immutable P2 observations")
    if not np.allclose(merged["truth"], merged["observed_truth"], rtol=0, atol=1e-12):
        raise ValueError("frozen OOF truth differs from immutable P2 observations")


def _feature_frames(
    observations: pd.DataFrame,
    oof: pd.DataFrame,
    hourly: pd.DataFrame,
    contract: dict[str, Any],
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    state = build_public_state_features(observations, oof.loc[:, ["time", "layer"]])
    features = state.loc[:, STATE_FEATURES].astype(np.float32).reset_index(drop=True)
    for layer in (2, 3, 4):
        features[f"layer_{layer}"] = (oof["layer"].to_numpy(int) == layer).astype(np.float32)
    external = build_power_features(hourly, oof.loc[:, ["time"]])
    if set(features).intersection(external.columns):
        raise ValueError("NASA POWER feature names collide with control features")
    candidate = pd.concat([features, external.reset_index(drop=True)], axis=1)
    control_names = tuple(features.columns)
    external_names = tuple(external.columns)
    candidate_names = tuple(candidate.columns)
    forbidden = set(contract["features"]["forbidden_names"])
    if forbidden.intersection(candidate_names):
        raise ValueError("target-like name entered the residual feature matrix")
    if control_names != CONTROL_FEATURES or len(candidate_names) != len(set(candidate_names)):
        raise ValueError("residual feature contract changed")
    coverage = finite_coverage(external, external_names)
    fully_finite = float(np.isfinite(external.to_numpy(float)).all(axis=1).mean())
    minimum = float(contract["expected_power_contract"]["minimum_feature_coverage"])
    minimum_rows = float(contract["expected_power_contract"]["minimum_fully_finite_row_share"])
    standard_deviation = external.std(axis=0, skipna=True).to_numpy(float)
    constant = [
        name
        for name, spread in zip(external_names, standard_deviation, strict=True)
        if not np.isfinite(spread) or spread <= 0
    ]
    if coverage < minimum or fully_finite < minimum_rows or constant:
        raise ValueError("NASA POWER signal precheck failed coverage or non-constant gates")
    coverage_by_block = {
        block: finite_coverage(external.loc[oof["block"].eq(block)], external_names)
        for block in OUTER_BLOCKS
    }
    report = {
        "external_feature_count": len(external_names),
        "control_feature_count": len(control_names),
        "candidate_feature_count": len(candidate_names),
        "finite_value_coverage": coverage,
        "fully_finite_row_share": fully_finite,
        "zero_coverage_rows": int(external.notna().sum(axis=1).eq(0).sum()),
        "constant_external_features": constant,
        "coverage_by_block": coverage_by_block,
        "passed": True,
    }
    return candidate, control_names, candidate_names, report


def _model_parameters(contract: dict[str, Any]) -> dict[str, Any]:
    parameters = dict(contract["model"])
    parameters.pop("family")
    return parameters


def _fit_correction(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: Sequence[str],
    train_mask: np.ndarray,
    predict_mask: np.ndarray,
    model_parameters: dict[str, Any],
) -> np.ndarray:
    if not train_mask.any() or not predict_mask.any() or np.any(train_mask & predict_mask):
        raise ValueError("invalid residual model train/predict split")
    residual = frame["truth"].to_numpy(float) - frame["incumbent_prediction"].to_numpy(float)
    model = LGBMRegressor(**model_parameters)
    columns = list(feature_names)
    model.fit(features.loc[train_mask, columns], residual[train_mask])
    prediction = np.asarray(model.predict(features.loc[predict_mask, columns]), dtype=float)
    if len(prediction) != int(predict_mask.sum()) or not np.isfinite(prediction).all():
        raise ValueError("residual LightGBM emitted invalid predictions")
    return prediction


def _choose_alpha(
    truth: np.ndarray,
    incumbent: np.ndarray,
    correction: np.ndarray,
    alpha_grid: Sequence[float],
) -> tuple[float, list[dict[str, float]]]:
    scores = [
        {
            "alpha": float(alpha),
            "rmse": _rmse(truth, incumbent + float(alpha) * correction),
        }
        for alpha in alpha_grid
    ]
    selected = min(scores, key=lambda row: (row["rmse"], row["alpha"]))
    return float(selected["alpha"]), scores


def _outer_fold_prediction(
    frame: pd.DataFrame,
    features: pd.DataFrame,
    feature_names: Sequence[str],
    held_block: str,
    alpha_grid: Sequence[float],
    model_parameters: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    block = frame["block"].astype(str).to_numpy()
    outer_train = block != held_block
    outer_held = block == held_block
    train_blocks = tuple(sorted(set(block[outer_train])))
    if len(train_blocks) != 2 or not outer_held.any():
        raise ValueError("outer residual fold must contain two train blocks and one held block")
    inner_correction = np.full(len(frame), np.nan, dtype=float)
    inner_details: list[dict[str, Any]] = []
    for inner_held in train_blocks:
        inner_train = outer_train & (block != inner_held)
        inner_valid = outer_train & (block == inner_held)
        inner_correction[inner_valid] = _fit_correction(
            frame,
            features,
            feature_names,
            inner_train,
            inner_valid,
            model_parameters,
        )
        inner_details.append(
            {
                "held_block": inner_held,
                "train_rows": int(inner_train.sum()),
                "validation_rows": int(inner_valid.sum()),
            }
        )
    if not np.isfinite(inner_correction[outer_train]).all():
        raise ValueError("inner residual predictions are incomplete")
    truth = frame["truth"].to_numpy(float)
    incumbent = frame["incumbent_prediction"].to_numpy(float)
    alpha, alpha_scores = _choose_alpha(
        truth[outer_train],
        incumbent[outer_train],
        inner_correction[outer_train],
        alpha_grid,
    )
    correction = _fit_correction(
        frame,
        features,
        feature_names,
        outer_train,
        outer_held,
        model_parameters,
    )
    prediction = incumbent[outer_held] + alpha * correction
    return prediction, {
        "held_block": held_block,
        "train_blocks": list(train_blocks),
        "train_rows": int(outer_train.sum()),
        "held_rows": int(outer_held.sum()),
        "selected_alpha": alpha,
        "inner_alpha_scores": alpha_scores,
        "inner_folds": inner_details,
    }


def _comparison_metrics(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    truth = frame["truth"].to_numpy(float)
    result: dict[str, Any] = {
        "rows": len(frame),
        "baseline_rmse": _rmse(truth, baseline),
        "candidate_rmse": _rmse(truth, candidate),
    }
    result["delta_rmse"] = result["candidate_rmse"] - result["baseline_rmse"]
    result["by_block"] = {}
    block_values = frame["block"].astype(str).to_numpy()
    for block in OUTER_BLOCKS:
        keep = block_values == block
        baseline_rmse = _rmse(truth[keep], baseline[keep])
        candidate_rmse = _rmse(truth[keep], candidate[keep])
        result["by_block"][block] = {
            "rows": int(keep.sum()),
            "baseline_rmse": baseline_rmse,
            "candidate_rmse": candidate_rmse,
            "delta_rmse": candidate_rmse - baseline_rmse,
        }
    result["by_layer"] = {}
    layer_values = frame["layer"].to_numpy(int)
    for layer in (2, 3, 4):
        keep = layer_values == layer
        baseline_rmse = _rmse(truth[keep], baseline[keep])
        candidate_rmse = _rmse(truth[keep], candidate[keep])
        result["by_layer"][str(layer)] = {
            "rows": int(keep.sum()),
            "baseline_rmse": baseline_rmse,
            "candidate_rmse": candidate_rmse,
            "delta_rmse": candidate_rmse - baseline_rmse,
        }
    return result


def _paired_day_bootstrap(
    frame: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    truth = frame["truth"].to_numpy(float)
    day = (
        pd.to_datetime(frame["time"], utc=True).dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d")
    )
    aggregate = (
        pd.DataFrame(
            {
                "day": day,
                "baseline_squared_error": (np.asarray(baseline) - truth) ** 2,
                "candidate_squared_error": (np.asarray(candidate) - truth) ** 2,
            }
        )
        .groupby("day", sort=True)
        .agg(
            rows=("day", "size"),
            baseline_sse=("baseline_squared_error", "sum"),
            candidate_sse=("candidate_squared_error", "sum"),
        )
    )
    rows = aggregate["rows"].to_numpy(float)
    baseline_sse = aggregate["baseline_sse"].to_numpy(float)
    candidate_sse = aggregate["candidate_sse"].to_numpy(float)
    rng = np.random.default_rng(seed)
    deltas = np.empty(replicates, dtype=float)
    for number in range(replicates):
        sampled = rng.integers(0, len(aggregate), len(aggregate))
        count = rows[sampled].sum()
        deltas[number] = np.sqrt(candidate_sse[sampled].sum() / count) - np.sqrt(
            baseline_sse[sampled].sum() / count
        )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "unit": "Asia/Seoul calendar day",
        "kst_days": len(aggregate),
        "delta_rmse": _rmse(truth, candidate) - _rmse(truth, baseline),
        "ci90_low": float(np.quantile(deltas, 0.05)),
        "ci90_high": float(np.quantile(deltas, 0.95)),
        "probability_improved": float(np.mean(deltas < 0)),
    }


def _promotion_decision(
    external_incremental: dict[str, Any],
    external_bootstrap: dict[str, Any],
    final_vs_incumbent: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    checks = {
        "external_incremental_at_least_0_010c": external_incremental["delta_rmse"]
        <= -float(gate["minimum_external_incremental_improvement_c"]),
        "external_incremental_ci90_high_below_zero": external_bootstrap["ci90_high"]
        < float(gate["external_bootstrap_ci90_high_max"]),
        "external_improves_at_least_two_blocks": sum(
            row["delta_rmse"] < 0 for row in external_incremental["by_block"].values()
        )
        >= int(gate["minimum_external_improved_blocks"]),
        "external_layer_guard": max(
            row["delta_rmse"] for row in external_incremental["by_layer"].values()
        )
        <= float(gate["maximum_external_layer_regression_c"]),
        "final_beats_incumbent": final_vs_incumbent["delta_rmse"] < 0,
        "final_layer_guard_vs_incumbent": max(
            row["delta_rmse"] for row in final_vs_incumbent["by_layer"].values()
        )
        <= float(gate["maximum_final_layer_regression_vs_incumbent_c"]),
    }
    passed = all(checks.values())
    return {
        "checks": checks,
        "passed": passed,
        "decision": gate["pass_action"] if passed else gate["fail_action"],
        "interpretation": (
            "This is only an external-family research gate; it does not authorize hidden-period "
            "inference, model replacement, submission generation, or upload."
        ),
    }


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"sha": sha, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"sha": None, "dirty": None}


def run(args: argparse.Namespace, progress: Progress) -> dict[str, Any]:
    started = time.perf_counter()
    config_path = _repo_path(args.config)
    progress.update(2, "contract", "사전등록 계약과 경로를 검사 중")
    contract = _load_contract(config_path)
    output_dir = _repo_path(contract["outputs"]["directory"])
    result_path = _repo_path(contract["outputs"]["result"])
    if result_path.exists():
        raise FileExistsError("one-shot result already exists; post-score rerun is disabled")
    output_dir.mkdir(parents=True, exist_ok=True)

    progress.update(7, "raw_manifest", "격리 원본과 manifest SHA를 fail-closed 검사 중")
    raw_provenance = [_validate_raw_entry(entry) for entry in contract["raw_files"]]
    cutoff = contract["external_data_policy"]["cutoff_utc"]
    hourly = load_power_hourly(
        [_repo_path(entry["path"]) for entry in contract["raw_files"]],
        cutoff=cutoff,
    )
    power_quality = _validate_power_quality(hourly, contract)

    progress.update(14, "frozen_oof", "동결 incumbent OOF의 SHA·키·RMSE를 검사 중")
    oof = _load_frozen_oof(contract)
    data_dir = resolve_data_dir(args.data_dir)
    data = load_p2_data(data_dir)
    _reconcile_oof_truth(data.observations, oof)

    progress.update(
        25, "signal_precheck", "public state와 NASA 특징의 인과 정렬·coverage를 검사 중"
    )
    features, control_names, candidate_names, signal_precheck = _feature_frames(
        data.observations,
        oof,
        hourly,
        contract,
    )
    model_parameters = _model_parameters(contract)
    alpha_grid = tuple(float(value) for value in contract["validation"]["alpha_grid"])
    control_prediction = np.full(len(oof), np.nan, dtype=float)
    candidate_prediction = np.full(len(oof), np.nan, dtype=float)
    fold_results: dict[str, Any] = {}

    for number, held_block in enumerate(OUTER_BLOCKS, start=1):
        progress.update(
            30 + (number - 1) * 18,
            f"outer_{held_block}",
            f"{held_block}: train-only alpha를 선택하고 control/candidate를 적합 중",
        )
        held = oof["block"].eq(held_block).to_numpy()
        control_fold, control_detail = _outer_fold_prediction(
            oof,
            features,
            control_names,
            held_block,
            alpha_grid,
            model_parameters,
        )
        candidate_fold, candidate_detail = _outer_fold_prediction(
            oof,
            features,
            candidate_names,
            held_block,
            alpha_grid,
            model_parameters,
        )
        control_prediction[held] = control_fold
        candidate_prediction[held] = candidate_fold
        fold_results[held_block] = {
            "control": control_detail,
            "candidate": candidate_detail,
        }
    if not np.isfinite(control_prediction).all() or not np.isfinite(candidate_prediction).all():
        raise ValueError("outer OOF residual predictions are incomplete")

    progress.update(84, "evaluation", "RMSE·layer/block 분해와 KST-day bootstrap을 계산 중")
    incumbent = oof["incumbent_prediction"].to_numpy(float)
    external_incremental = _comparison_metrics(oof, control_prediction, candidate_prediction)
    control_vs_incumbent = _comparison_metrics(oof, incumbent, control_prediction)
    final_vs_incumbent = _comparison_metrics(oof, incumbent, candidate_prediction)
    replicates = int(contract["validation"]["bootstrap_replicates"])
    bootstrap_seed = int(contract["validation"]["bootstrap_seed"])
    external_bootstrap = _paired_day_bootstrap(
        oof,
        control_prediction,
        candidate_prediction,
        replicates=replicates,
        seed=bootstrap_seed,
    )
    final_bootstrap = _paired_day_bootstrap(
        oof,
        incumbent,
        candidate_prediction,
        replicates=replicates,
        seed=bootstrap_seed,
    )
    promotion = _promotion_decision(
        external_incremental,
        external_bootstrap,
        final_vs_incumbent,
        contract["promotion_gate"],
    )

    progress.update(94, "artifacts", "OOF·결과·provenance manifest를 원자적으로 기록 중")
    output = oof.copy()
    output["control_prediction"] = control_prediction
    output["candidate_prediction"] = candidate_prediction
    oof_path = _repo_path(contract["outputs"]["oof"])
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(oof_path, index=False, compression="zstd")
    elapsed = time.perf_counter() - started
    result: dict[str, Any] = {
        "experiment_id": EXPERIMENT_ID,
        "status": "complete",
        "research_only": True,
        "upload_allowed": False,
        "created_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed,
        "decision": promotion["decision"],
        "leakage_contract": {
            "hidden_2025_sep_oct_target_values_used": False,
            "held_outer_labels_used_for_model_or_alpha_selection": False,
            "external_alignment": "causal backward as-of, tolerance 70 minutes",
            "hidden_period_inference_run": False,
            "submission_created_or_modified": False,
        },
        "power_quality_after_cutoff": power_quality,
        "signal_precheck": signal_precheck,
        "feature_names": {
            "control": list(control_names),
            "external": [name for name in candidate_names if name not in control_names],
            "candidate_count": len(candidate_names),
        },
        "outer_folds": fold_results,
        "metrics": {
            "external_incremental_candidate_vs_control": external_incremental,
            "control_vs_incumbent": control_vs_incumbent,
            "final_candidate_vs_incumbent": final_vs_incumbent,
        },
        "bootstrap": {
            "external_incremental_candidate_vs_control": external_bootstrap,
            "final_candidate_vs_incumbent": final_bootstrap,
        },
        "promotion": promotion,
    }
    _write_json(result_path, result)

    observations_path = data_dir / "observations.csv"
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "git": _git_state(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "source": contract["external_data_policy"],
        "raw_files": raw_provenance,
        "frozen_inputs": {
            "incumbent_oof_path": contract["frozen_incumbent"]["path"],
            "incumbent_oof_sha256": contract["frozen_incumbent"]["sha256"],
            "observations_file": observations_path.name,
            "observations_sha256": _sha256(observations_path),
        },
        "code_and_contract": {
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "config_sha256": _sha256(config_path),
            "runner_path": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
            "runner_sha256": _sha256(Path(__file__).resolve()),
            "external_feature_module_path": "src/p2_restore/external_meteorology.py",
            "external_feature_module_sha256": _sha256(
                REPO_ROOT / "src/p2_restore/external_meteorology.py"
            ),
        },
        "outputs": {
            "oof_path": str(oof_path.relative_to(REPO_ROOT)),
            "oof_sha256": _sha256(oof_path),
            "result_path": str(result_path.relative_to(REPO_ROOT)),
            "result_sha256": _sha256(result_path),
        },
        "decision": promotion["decision"],
    }
    manifest_path = _repo_path(contract["outputs"]["manifest"])
    _write_json(manifest_path, manifest)
    progress.update(
        100,
        "complete",
        f"완료: {promotion['decision']}",
        status="complete",
        extra={
            "decision": promotion["decision"],
            "external_incremental_delta_rmse": external_incremental["delta_rmse"],
            "final_vs_incumbent_delta_rmse": final_vs_incumbent["delta_rmse"],
            "result_path": str(result_path),
        },
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiments/p2_nasa_power_residual_meta_v1.json",
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--status-file",
        default="artifacts/status/p2_nasa_power_residual_meta_oof.json",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    progress = Progress(_repo_path(args.status_file))
    try:
        run(args, progress)
    except Exception as exc:
        progress.update(
            100,
            "failed",
            f"{type(exc).__name__}: {exc}",
            status="failed",
        )
        raise


if __name__ == "__main__":
    main()
