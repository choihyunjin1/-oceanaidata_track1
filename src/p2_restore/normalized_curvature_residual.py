"""Preregistered normalized-curvature residual utilities for P2.

This module deliberately has no test-index, sample-submission, submission, or
upload code path.  The numerical target is a dimensionless residual relative
to the public-layer linear interpolation baseline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PUBLIC_LAYERS = (1, 5, 6, 7, 8)
TARGET_LAYERS = (2, 3, 4)
TEMPORAL_FEATURES = (
    "doy_sin",
    "doy_cos",
    "hour_sin",
    "hour_cos",
    "m2_sin",
    "m2_cos",
)

FORBIDDEN_FILE_NAMES = {
    "test_index.csv",
    "sample_submission.csv",
    "baseline_interp.csv",
}
FORBIDDEN_PATH_TOKENS = ("submission", "candidate")


@dataclass(frozen=True)
class NormalizedCurvatureDesign:
    """Leakage-safe model matrix and reversible normalized target."""

    keys: pd.DataFrame
    features: pd.DataFrame
    normalized_target: np.ndarray
    truth: np.ndarray
    baseline: np.ndarray
    profile_scale: np.ndarray


@dataclass(frozen=True)
class Stage1Split:
    """Strict past-only Stage 1 split with an embargo before validation."""

    train_mask: np.ndarray
    validation_mask: np.ndarray
    train_end_exclusive: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass(frozen=True)
class ExactAlignment:
    """Ordering needed to compare a candidate with the frozen exact incumbent."""

    candidate_positions: np.ndarray
    time: pd.DatetimeIndex
    layer: np.ndarray
    truth: np.ndarray
    incumbent_prediction: np.ndarray


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without interpreting file contents."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def assert_safe_input_path(path: Path) -> None:
    """Reject official-evaluation and submission-like paths before any read."""

    lowered_parts = tuple(part.lower() for part in path.parts)
    if path.name.lower() in FORBIDDEN_FILE_NAMES:
        raise RuntimeError(f"forbidden P2 input path: {path}")
    if any(token in part for part in lowered_parts for token in FORBIDDEN_PATH_TOKENS):
        raise RuntimeError(f"submission/candidate-like input path is forbidden: {path}")


def load_and_validate_preregistration(
    config_path: Path,
    repo_root: Path,
    *,
    expected_config_sha256: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load the sealed preregistration and verify every immutable local input.

    Reference files are checked only by size and streaming hash.  Their data
    values are not opened here.
    """

    config_path = config_path.resolve()
    repo_root = repo_root.resolve()
    if not config_path.is_relative_to(repo_root):
        raise RuntimeError("preregistration must stay inside the repository")
    observed_config_sha = sha256_file(config_path)
    if observed_config_sha != expected_config_sha256:
        raise RuntimeError("NCR_LGBM preregistration hash drift")
    parsed = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise TypeError("preregistration must be a JSON object")
    if parsed.get("schema_version") != (
        "p2_normalized_curvature_residual_lgbm_stage1.prereg.v1"
    ):
        raise ValueError("unexpected NCR_LGBM preregistration schema")
    if parsed.get("status") != "PREREGISTERED_NOT_EXECUTED":
        raise ValueError("NCR_LGBM Stage 1 is not in its preregistered state")
    if parsed.get("family") != "NCR_LGBM":
        raise ValueError("model family drift")

    policy = parsed["selection_policy"]
    if bool(policy["official_score_used_for_gate_or_tuning"]):
        raise ValueError("official-score calibration is forbidden")
    if bool(policy["result_based_retuning"]):
        raise ValueError("result-based tuning is forbidden")
    if int(policy["candidate_grid_size"]) != 1:
        raise ValueError("Stage 1 candidate grid changed")
    if int(policy["stage1_model_fit_count"]) != 3:
        raise ValueError("Stage 1 fit budget changed")

    output = parsed["output"]
    if not bool(output["aggregate_only"]):
        raise ValueError("aggregate-only output contract changed")
    if bool(output["row_predictions_written"]) or bool(output["csv_files_written"]):
        raise ValueError("row-level output is forbidden")
    if int(output["submission_files_generated"]) or int(output["uploads"]):
        raise ValueError("external action contract changed")
    if sorted(output["allowed_files"]) != ["manifest.json", "result.json"]:
        raise ValueError("output allow-list changed")

    feature_contract = parsed["feature_contract"]
    forbidden = set(feature_contract["forbidden_features"])
    if not {"year", "elapsed_days", "target-layer temperature", "target-layer salinity"}.issubset(
        forbidden
    ):
        raise ValueError("feature firewall weakened")

    observed: dict[str, dict[str, Any]] = {}
    for reference in parsed["immutable_references"].values():
        relative = Path(str(reference["path"]))
        assert_safe_input_path(relative)
        resolved = (repo_root / relative).resolve()
        if not resolved.is_relative_to(repo_root) or not resolved.is_file():
            raise FileNotFoundError(resolved)
        size = resolved.stat().st_size
        digest = sha256_file(resolved)
        if size != int(reference["bytes"]):
            raise RuntimeError(f"immutable reference size drift: {relative}")
        if digest != str(reference["sha256"]):
            raise RuntimeError(f"immutable reference hash drift: {relative}")
        observed[relative.as_posix()] = {"bytes": size, "sha256": digest}
    return parsed, observed


def resolve_observations_path(data_dir: Path, expected_sha256: str) -> Path:
    """Resolve and hash-pin the only allowed source-data file."""

    observations = data_dir.resolve() / "observations.csv"
    assert_safe_input_path(observations)
    if observations.name != "observations.csv" or not observations.is_file():
        raise FileNotFoundError(observations)
    if sha256_file(observations) != expected_sha256:
        raise RuntimeError("observations.csv hash drift")
    return observations


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise KeyError(f"required NCR_LGBM column missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _finite_row_bounds(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(values)
    count = finite.sum(axis=1)
    lower = np.min(np.where(finite, values, np.inf), axis=1)
    upper = np.max(np.where(finite, values, -np.inf), axis=1)
    lower[count == 0] = np.nan
    upper[count == 0] = np.nan
    return lower, upper, count


def compute_profile_scale(frame: pd.DataFrame, *, floor_c: float = 0.5) -> np.ndarray:
    """Compute the preregistered endpoint/public-range temperature scale."""

    if not np.isfinite(floor_c) or floor_c <= 0:
        raise ValueError("profile scale floor must be finite and positive")
    temp_1 = _numeric(frame, "temp_1")
    temp_5 = _numeric(frame, "temp_5")
    public_range = _numeric(frame, "public_temp_range")
    endpoints_available = np.isfinite(temp_1) & np.isfinite(temp_5)
    raw_scale = np.where(endpoints_available, np.abs(temp_1 - temp_5), public_range)
    scale = np.maximum(raw_scale, floor_c)
    if not np.isfinite(scale).all():
        raise ValueError("profile scale is unavailable for one or more rows")
    return scale


def encode_normalized_curvature(
    truth: np.ndarray,
    baseline: np.ndarray,
    profile_scale: np.ndarray,
) -> np.ndarray:
    """Encode temperature as dimensionless curvature around the baseline."""

    truth = np.asarray(truth, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    profile_scale = np.asarray(profile_scale, dtype=float)
    if truth.shape != baseline.shape or truth.shape != profile_scale.shape:
        raise ValueError("truth, baseline, and profile scale must have identical shapes")
    if not (np.isfinite(truth).all() and np.isfinite(baseline).all()):
        raise ValueError("truth and baseline must be finite")
    if not (np.isfinite(profile_scale).all() and (profile_scale > 0).all()):
        raise ValueError("profile scale must be finite and positive")
    return (truth - baseline) / profile_scale


def decode_normalized_curvature(
    predicted_curvature: np.ndarray,
    baseline: np.ndarray,
    profile_scale: np.ndarray,
) -> np.ndarray:
    """Decode dimensionless model output back to degrees Celsius."""

    predicted_curvature = np.asarray(predicted_curvature, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    profile_scale = np.asarray(profile_scale, dtype=float)
    if predicted_curvature.shape != baseline.shape or baseline.shape != profile_scale.shape:
        raise ValueError("prediction, baseline, and profile scale must have identical shapes")
    if not (
        np.isfinite(predicted_curvature).all()
        and np.isfinite(baseline).all()
        and np.isfinite(profile_scale).all()
        and (profile_scale > 0).all()
    ):
        raise ValueError("decode inputs must be finite and scales positive")
    return baseline + predicted_curvature * profile_scale


def build_normalized_curvature_design(
    frame: pd.DataFrame,
    *,
    scale_floor_c: float = 0.5,
    salinity_scale_floor: float = 0.05,
    depth_scale_floor_m: float = 1.0,
) -> NormalizedCurvatureDesign:
    """Build the fixed shift-invariant NCR_LGBM design matrix.

    Only public-layer covariates, target depth/layer, and cyclic time features
    are materialized.  Absolute baseline, raw year, elapsed days, and hidden
    target-layer temperature/salinity never enter ``features``.
    """

    if not np.isfinite(salinity_scale_floor) or salinity_scale_floor <= 0:
        raise ValueError("salinity scale floor must be finite and positive")
    if not np.isfinite(depth_scale_floor_m) or depth_scale_floor_m <= 0:
        raise ValueError("depth scale floor must be finite and positive")
    required = {
        "time",
        "layer",
        "target",
        "baseline",
        "target_depth",
        "public_temp_range",
        "public_temp_count",
        *TEMPORAL_FEATURES,
    }
    required.update(
        f"{prefix}_{layer}"
        for prefix in ("temp", "psal", "depth", "nominal")
        for layer in PUBLIC_LAYERS
    )
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"required NCR_LGBM columns missing: {missing}")

    truth = _numeric(frame, "target")
    baseline = _numeric(frame, "baseline")
    target_depth = _numeric(frame, "target_depth")
    layer_value = _numeric(frame, "layer")
    if not np.isin(layer_value, TARGET_LAYERS).all():
        raise ValueError("NCR_LGBM received a non-target layer")
    profile_scale = compute_profile_scale(frame, floor_c=scale_floor_c)
    normalized_target = encode_normalized_curvature(truth, baseline, profile_scale)

    public_temp = np.column_stack([_numeric(frame, f"temp_{layer}") for layer in PUBLIC_LAYERS])
    public_psal = np.column_stack([_numeric(frame, f"psal_{layer}") for layer in PUBLIC_LAYERS])
    public_depth = np.column_stack(
        [_numeric(frame, f"depth_{layer}") for layer in PUBLIC_LAYERS]
    )
    public_nominal = np.column_stack(
        [_numeric(frame, f"nominal_{layer}") for layer in PUBLIC_LAYERS]
    )

    psal_finite = np.isfinite(public_psal)
    psal_count = psal_finite.sum(axis=1)
    psal_mean = np.divide(
        np.nansum(public_psal, axis=1),
        psal_count,
        out=np.full(len(frame), np.nan),
        where=psal_count > 0,
    )
    psal_lower, psal_upper, _ = _finite_row_bounds(public_psal)
    psal_scale = np.maximum(psal_upper - psal_lower, salinity_scale_floor)

    nominal_lower, nominal_upper, nominal_count = _finite_row_bounds(public_nominal)
    depth_scale = np.maximum(nominal_upper - nominal_lower, depth_scale_floor_m)
    if not (np.isfinite(target_depth).all() and np.isfinite(depth_scale).all()):
        raise ValueError("target/public nominal depths are incomplete")

    values: dict[str, np.ndarray] = {
        "target_layer": layer_value,
        "target_depth_fraction": (target_depth - nominal_lower) / depth_scale,
        "public_temp_count_fraction": _numeric(frame, "public_temp_count")
        / float(len(PUBLIC_LAYERS)),
        "public_psal_count_fraction": psal_count.astype(float) / float(len(PUBLIC_LAYERS)),
        "public_nominal_count_fraction": nominal_count.astype(float)
        / float(len(PUBLIC_LAYERS)),
        "endpoint_pair_available": (
            np.isfinite(public_temp[:, 0]) & np.isfinite(public_temp[:, 1])
        ).astype(float),
        "log1p_profile_scale": np.log1p(profile_scale),
        "log1p_psal_scale": np.log1p(psal_scale),
        "log1p_depth_scale": np.log1p(depth_scale),
    }
    for name in TEMPORAL_FEATURES:
        values[name] = _numeric(frame, name)
    for index, public_layer in enumerate(PUBLIC_LAYERS):
        temp = public_temp[:, index]
        psal = public_psal[:, index]
        depth = public_depth[:, index]
        nominal = public_nominal[:, index]
        values[f"temp_offset_l{public_layer}"] = (temp - baseline) / profile_scale
        values[f"temp_present_l{public_layer}"] = np.isfinite(temp).astype(float)
        values[f"psal_anomaly_l{public_layer}"] = (psal - psal_mean) / psal_scale
        values[f"psal_present_l{public_layer}"] = np.isfinite(psal).astype(float)
        values[f"depth_offset_l{public_layer}"] = (depth - target_depth) / depth_scale
        values[f"depth_present_l{public_layer}"] = np.isfinite(depth).astype(float)
        values[f"nominal_offset_l{public_layer}"] = (
            nominal - target_depth
        ) / depth_scale
        values[f"nominal_present_l{public_layer}"] = np.isfinite(nominal).astype(float)

    features = pd.DataFrame(values, index=frame.index).reset_index(drop=True)
    forbidden_exact = {
        "year",
        "elapsed_days",
        "baseline",
        "target",
        "residual",
        *(f"temp_{layer}" for layer in TARGET_LAYERS),
        *(f"psal_{layer}" for layer in TARGET_LAYERS),
    }
    if forbidden_exact.intersection(features.columns):
        raise AssertionError("forbidden absolute/time/hidden feature leaked into NCR design")
    if not all(np.issubdtype(dtype, np.number) for dtype in features.dtypes):
        raise TypeError("NCR_LGBM features must be numeric")

    keys = frame.loc[:, ["time", "layer"]].copy().reset_index(drop=True)
    keys["time"] = pd.to_datetime(keys["time"], utc=True)
    keys["layer"] = pd.to_numeric(keys["layer"], errors="raise").astype(int)
    return NormalizedCurvatureDesign(
        keys=keys,
        features=features,
        normalized_target=normalized_target,
        truth=truth,
        baseline=baseline,
        profile_scale=profile_scale,
    )


def make_stage1_split(
    time: pd.Series | pd.Index,
    *,
    validation_start: str,
    validation_end: str,
    embargo_days: int,
) -> Stage1Split:
    """Create the preregistered strict-past/embargo split in KST."""

    if embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    parsed = pd.DatetimeIndex(pd.to_datetime(time, utc=True)).tz_convert("Asia/Seoul")
    start = pd.Timestamp(validation_start)
    end = pd.Timestamp(validation_end)
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("Stage 1 boundaries must be timezone-aware")
    start = start.tz_convert("Asia/Seoul")
    end = end.tz_convert("Asia/Seoul")
    if not start < end:
        raise ValueError("validation start must precede validation end")
    train_end = start - pd.Timedelta(days=embargo_days)
    train_mask = np.asarray(parsed < train_end, dtype=bool)
    validation_mask = np.asarray((parsed >= start) & (parsed < end), dtype=bool)
    if not train_mask.any():
        raise ValueError("Stage 1 split has no strict-past training rows")
    if not validation_mask.any():
        raise ValueError("Stage 1 split has no validation rows")
    if np.any(train_mask & validation_mask):
        raise AssertionError("Stage 1 train/validation overlap")
    return Stage1Split(
        train_mask=train_mask,
        validation_mask=validation_mask,
        train_end_exclusive=train_end,
        validation_start=start,
        validation_end=end,
    )


def align_exact_incumbent(
    validation_design: NormalizedCurvatureDesign,
    oof: pd.DataFrame,
    *,
    block: str,
    expected_rows: int,
    truth_column: str = "truth",
    prediction_column: str = "prediction",
    truth_tolerance_c: float = 1e-9,
) -> ExactAlignment:
    """Align candidate validation rows to the hash-pinned incumbent OOF panel."""

    required = {"time", "layer", "block", truth_column, prediction_column}
    missing = sorted(required.difference(oof.columns))
    if missing:
        raise KeyError(f"exact incumbent OOF columns missing: {missing}")
    exact = oof.loc[oof["block"].astype(str) == block, list(required)].copy()
    exact["_time_ns"] = pd.to_datetime(exact["time"], utc=True).astype("int64")
    exact["layer"] = pd.to_numeric(exact["layer"], errors="raise").astype(int)
    if exact.duplicated(["_time_ns", "layer"]).any():
        raise ValueError("exact incumbent OOF keys are not unique")

    candidate = validation_design.keys.copy()
    candidate["_time_ns"] = pd.to_datetime(candidate["time"], utc=True).astype("int64")
    candidate["_candidate_position"] = np.arange(len(candidate), dtype=int)
    candidate["_candidate_truth"] = validation_design.truth
    if candidate.duplicated(["_time_ns", "layer"]).any():
        raise ValueError("candidate validation keys are not unique")

    merged = candidate.merge(
        exact.loc[:, ["_time_ns", "layer", truth_column, prediction_column]],
        on=["_time_ns", "layer"],
        how="inner",
        validate="one_to_one",
    ).sort_values("_candidate_position")
    if len(merged) != expected_rows:
        raise ValueError(
            f"exact same-season alignment has {len(merged)} rows; expected {expected_rows}"
        )
    incumbent_truth = pd.to_numeric(merged[truth_column], errors="coerce").to_numpy(float)
    candidate_truth = merged["_candidate_truth"].to_numpy(float)
    if not (
        np.isfinite(incumbent_truth).all()
        and np.isfinite(candidate_truth).all()
        and np.max(np.abs(incumbent_truth - candidate_truth)) <= truth_tolerance_c
    ):
        raise RuntimeError("candidate/exact-incumbent truth alignment failed")
    incumbent_prediction = pd.to_numeric(
        merged[prediction_column], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(incumbent_prediction).all():
        raise ValueError("exact incumbent prediction contains non-finite values")
    return ExactAlignment(
        candidate_positions=merged["_candidate_position"].to_numpy(dtype=int),
        time=pd.DatetimeIndex(pd.to_datetime(merged["_time_ns"], utc=True)),
        layer=merged["layer"].to_numpy(dtype=int),
        truth=incumbent_truth,
        incumbent_prediction=incumbent_prediction,
    )


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.shape != prediction.shape or not truth.size:
        raise ValueError("RMSE inputs must have equal, non-empty shapes")
    if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        raise ValueError("RMSE inputs must be finite")
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def metric_report(
    truth: np.ndarray,
    prediction: np.ndarray,
    layer: np.ndarray,
) -> dict[str, Any]:
    """Return aggregate row-pooled and layer-aware RMSE only."""

    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    layer = np.asarray(layer, dtype=int)
    if truth.shape != prediction.shape or truth.shape != layer.shape:
        raise ValueError("metric arrays must have identical shapes")
    by_layer: dict[str, float] = {}
    layer_mse: list[float] = []
    for target_layer in TARGET_LAYERS:
        keep = layer == target_layer
        if not keep.any():
            raise ValueError(f"metric population has no layer {target_layer} rows")
        score = rmse(truth[keep], prediction[keep])
        by_layer[str(target_layer)] = score
        layer_mse.append(score**2)
    return {
        "rows": int(len(truth)),
        "row_pooled_rmse_c": rmse(truth, prediction),
        "layer_equal_rmse_c": float(np.sqrt(np.mean(layer_mse))),
        "by_layer_rmse_c": by_layer,
    }


def paired_day_bootstrap(
    truth: np.ndarray,
    incumbent_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    time: pd.DatetimeIndex,
    *,
    replicates: int,
    seed: int,
    confidence: float = 0.9,
) -> dict[str, Any]:
    """Paired KST-day bootstrap of candidate-minus-incumbent pooled RMSE."""

    truth = np.asarray(truth, dtype=float)
    incumbent_prediction = np.asarray(incumbent_prediction, dtype=float)
    candidate_prediction = np.asarray(candidate_prediction, dtype=float)
    if not (
        truth.shape == incumbent_prediction.shape == candidate_prediction.shape
        and len(time) == len(truth)
    ):
        raise ValueError("paired bootstrap inputs must have identical lengths")
    if replicates <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap contract")
    if not (
        np.isfinite(truth).all()
        and np.isfinite(incumbent_prediction).all()
        and np.isfinite(candidate_prediction).all()
    ):
        raise ValueError("paired bootstrap inputs must be finite")

    day = pd.DatetimeIndex(pd.to_datetime(time, utc=True)).tz_convert("Asia/Seoul").normalize()
    codes, unique_days = pd.factorize(day, sort=True)
    if len(unique_days) < 2:
        raise ValueError("paired day bootstrap requires at least two KST days")
    inc_sq = np.square(incumbent_prediction - truth)
    cand_sq = np.square(candidate_prediction - truth)
    day_count = np.bincount(codes, minlength=len(unique_days)).astype(float)
    day_inc = np.bincount(codes, weights=inc_sq, minlength=len(unique_days))
    day_cand = np.bincount(codes, weights=cand_sq, minlength=len(unique_days))

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique_days), size=(replicates, len(unique_days)))
    sampled_count = day_count[sampled].sum(axis=1)
    incumbent_rmse = np.sqrt(day_inc[sampled].sum(axis=1) / sampled_count)
    candidate_rmse = np.sqrt(day_cand[sampled].sum(axis=1) / sampled_count)
    delta = candidate_rmse - incumbent_rmse
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": int(replicates),
        "confidence": float(confidence),
        "seed": int(seed),
        "paired_kst_days": int(len(unique_days)),
        "point_delta_rmse_c": rmse(truth, candidate_prediction)
        - rmse(truth, incumbent_prediction),
        "ci90_c": [float(np.quantile(delta, alpha)), float(np.quantile(delta, 1 - alpha))],
        "probability_candidate_improves": float(np.mean(delta < 0)),
    }


def evaluate_stage1_gate(
    incumbent_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    bootstrap: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every fixed Stage 1 condition without tuning."""

    primary_delta = float(candidate_metrics["row_pooled_rmse_c"]) - float(
        incumbent_metrics["row_pooled_rmse_c"]
    )
    layer_delta = {
        str(layer): float(candidate_metrics["by_layer_rmse_c"][str(layer)])
        - float(incumbent_metrics["by_layer_rmse_c"][str(layer)])
        for layer in TARGET_LAYERS
    }
    checks = {
        "primary_delta": primary_delta
        <= float(gate["candidate_minus_incumbent_row_pooled_rmse_c_max"]),
        "bootstrap_ci_upper": float(bootstrap["ci90_c"][1])
        <= float(gate["paired_day_bootstrap_ci90_upper_c_max"]),
        "each_layer_delta": max(layer_delta.values())
        <= float(gate["candidate_minus_incumbent_each_layer_rmse_c_max"]),
    }
    passed = all(checks.values())
    return {
        "passed": bool(passed),
        "decision": gate["on_pass"] if passed else gate["on_fail"],
        "checks": checks,
        "candidate_minus_incumbent_row_pooled_rmse_c": primary_delta,
        "candidate_minus_incumbent_by_layer_rmse_c": layer_delta,
        "paired_day_bootstrap_ci90_c": [float(value) for value in bootstrap["ci90_c"]],
    }


def fit_lgbm_seed_ensemble(
    train_design: NormalizedCurvatureDesign,
    validation_design: NormalizedCurvatureDesign,
    *,
    seeds: list[int],
    parameters: dict[str, Any],
) -> np.ndarray:
    """Fit exactly the preregistered seeds and average decoded predictions."""

    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("NCR_LGBM Stage 1 requires exactly three distinct seeds")
    try:
        from lightgbm import LGBMRegressor
    except ImportError as error:  # pragma: no cover - environment-specific guard
        raise RuntimeError("lightgbm is required for NCR_LGBM Stage 1") from error

    if tuple(train_design.features.columns) != tuple(validation_design.features.columns):
        raise ValueError("train/validation NCR feature schemas differ")
    predictions: list[np.ndarray] = []
    for seed in seeds:
        model_parameters = dict(parameters)
        model_parameters["random_state"] = int(seed)
        model = LGBMRegressor(**model_parameters)
        model.fit(train_design.features, train_design.normalized_target)
        predicted_curvature = np.asarray(
            model.predict(validation_design.features), dtype=float
        )
        predictions.append(
            decode_normalized_curvature(
                predicted_curvature,
                validation_design.baseline,
                validation_design.profile_scale,
            )
        )
    return np.mean(np.vstack(predictions), axis=0)


def subset_design(
    design: NormalizedCurvatureDesign, mask: np.ndarray
) -> NormalizedCurvatureDesign:
    """Take a row subset while preserving feature and target alignment."""

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (len(design.features),):
        raise ValueError("design subset mask has the wrong shape")
    return NormalizedCurvatureDesign(
        keys=design.keys.loc[mask].reset_index(drop=True),
        features=design.features.loc[mask].reset_index(drop=True),
        normalized_target=design.normalized_target[mask],
        truth=design.truth[mask],
        baseline=design.baseline[mask],
        profile_scale=design.profile_scale[mask],
    )
