"""Point-only external I-ORS residual features for one nested P1 OOF experiment."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ocean_external.iors_ctd import (
    LooDataset,
    YearProfile,
    build_loo_dataset,
    depth_linear_baseline,
)

KEY_COLUMNS = ("station", "year", "layer", "time")
POINT_RESIDUAL_COLUMNS = (
    "ext_q50_signed_residual",
    "ext_q50_abs_residual",
    "ext_peer_count",
    "ext_peer_support",
    "ext_residual_median_24h",
    "ext_residual_slope_24h",
    "ext_residual_median_72h",
    "ext_residual_slope_72h",
)


@dataclass(frozen=True)
class P1IorsPanel:
    profile: YearProfile
    source_positions: np.ndarray
    frame_length: int


@dataclass(frozen=True)
class ExternalPointPrediction:
    q50: np.ndarray
    peer_count: np.ndarray
    eligible: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class CanonicalArtifactPaths:
    output_dir: Path
    status_file: Path
    outer_lock: Path


def canonical_artifact_paths(
    project_root: Path,
    artifact_contract: Mapping[str, Any],
    *,
    requested_output_dir: Path,
    requested_status_file: Path,
) -> CanonicalArtifactPaths:
    """Resolve and enforce the only paths allowed for a one-shot exposure lock."""

    root = project_root.resolve()

    def resolve(value: str | Path) -> Path:
        candidate = Path(value)
        return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()

    output_dir = resolve(str(artifact_contract["output_dir"]))
    status_file = resolve(str(artifact_contract["status"]))
    outer_lock = resolve(str(artifact_contract["outer_lock"]))
    requested_output = resolve(requested_output_dir)
    requested_status = resolve(requested_status_file)
    if requested_output != output_dir:
        raise ValueError("--output-dir must exactly equal the canonical preregistered path")
    if requested_status != status_file:
        raise ValueError("--status-file must exactly equal the canonical preregistered path")
    if outer_lock.parent != output_dir:
        raise ValueError("canonical outer lock must be a direct child of canonical output_dir")
    return CanonicalArtifactPaths(output_dir, status_file, outer_lock)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def build_p1_iors_panel(
    frame: pd.DataFrame,
    target_depth_by_layer: Mapping[int, float],
) -> P1IorsPanel:
    """Convert P1 I-ORS rows into the same depth-grid panel as external profiles."""

    required = {*KEY_COLUMNS, "temp", "psal", "depth"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"P1 frame is missing columns: {missing}")
    source_positions = np.flatnonzero(frame["station"].astype("string").eq("I-ORS").to_numpy())
    if source_positions.size == 0:
        raise ValueError("P1 frame contains no I-ORS rows")
    part = frame.iloc[source_positions].copy()
    if part.duplicated(["station", "year", "layer", "time"]).any():
        raise ValueError("I-ORS station/year/layer/time keys must be unique")
    years = sorted(int(value) for value in pd.to_numeric(part["year"], errors="raise").unique())
    if years != [2025]:
        raise ValueError(f"the frozen P1 I-ORS deployment must be 2025-only, got {years}")
    target_layers = np.asarray(
        sorted(int(value) for value in target_depth_by_layer), dtype=np.int16
    )
    target_depths = np.asarray(
        [float(target_depth_by_layer[int(layer)]) for layer in target_layers], dtype=np.float64
    )
    layer_to_position = {int(layer): position for position, layer in enumerate(target_layers)}
    layer_values = pd.to_numeric(part["layer"], errors="raise").to_numpy(dtype=np.int16)
    unexpected = sorted(set(int(value) for value in layer_values).difference(layer_to_position))
    if unexpected:
        raise ValueError(f"unexpected I-ORS layers: {unexpected}")
    parsed = pd.to_datetime(part["time"], errors="raise", utc=True, format="mixed")
    naive_utc = parsed.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[s]")
    unique_time, time_code = np.unique(naive_utc, return_inverse=True)
    shape = (unique_time.size, target_layers.size)
    temp = np.full(shape, np.nan, dtype=np.float64)
    psal = np.full(shape, np.nan, dtype=np.float64)
    depth = np.full(shape, np.nan, dtype=np.float64)
    depth_qc1 = np.zeros(shape, dtype=bool)
    original_position = np.full(shape, -1, dtype=np.int64)
    temp_value = _numeric(part, "temp")
    psal_value = _numeric(part, "psal")
    depth_value = _numeric(part, "depth")
    for local_position, (time_position, layer) in enumerate(
        zip(time_code, layer_values, strict=True)
    ):
        target_position = layer_to_position[int(layer)]
        if original_position[time_position, target_position] >= 0:
            raise ValueError("duplicate I-ORS time/layer position")
        temp[time_position, target_position] = temp_value[local_position]
        psal[time_position, target_position] = psal_value[local_position]
        if np.isfinite(depth_value[local_position]):
            depth[time_position, target_position] = depth_value[local_position]
            depth_qc1[time_position, target_position] = True
        original_position[time_position, target_position] = source_positions[local_position]
    mapping: list[dict[str, Any]] = []
    for target_position, layer in enumerate(target_layers):
        available = depth_qc1[:, target_position]
        median_depth = (
            float(np.median(depth[available, target_position]))
            if available.any()
            else float(target_depths[target_position])
        )
        depth[:, target_position] = np.where(available, depth[:, target_position], median_depth)
        mapping.append(
            {
                "target_layer": int(layer),
                "target_depth_m": float(target_depths[target_position]),
                "median_p1_depth_m": median_depth,
                "absolute_difference_m": abs(median_depth - float(target_depths[target_position])),
                "rows": int((original_position[:, target_position] >= 0).sum()),
            }
        )
    profile = YearProfile(
        year=2025,
        time_utc=unique_time,
        target_layers=target_layers,
        target_depths=target_depths,
        temp=temp,
        psal=psal,
        depth=depth,
        depth_qc1=depth_qc1,
        mapping=tuple(mapping),
        audit={
            "year": 2025,
            "time_rows": int(unique_time.size),
            "source_rows": int(source_positions.size),
            "mapping": mapping,
        },
    )
    return P1IorsPanel(
        profile=profile,
        source_positions=original_position,
        frame_length=len(frame),
    )


def _loo_source_positions(
    panel: P1IorsPanel,
    *,
    min_peer_temperatures: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions: list[np.ndarray] = []
    layers: list[np.ndarray] = []
    profile = panel.profile
    for target_position, layer in enumerate(profile.target_layers):
        peer_temp = profile.temp.copy()
        peer_temp[:, target_position] = np.nan
        peer_count = np.isfinite(peer_temp).sum(axis=1)
        baseline = depth_linear_baseline(profile, target_position)
        eligible = (
            np.isfinite(profile.temp[:, target_position])
            & (peer_count >= min_peer_temperatures)
            & np.isfinite(baseline)
            & (panel.source_positions[:, target_position] >= 0)
        )
        local = panel.source_positions[eligible, target_position]
        positions.append(local)
        layers.append(np.full(local.size, int(layer), dtype=np.int16))
    if not positions:
        raise ValueError("no P1 I-ORS LOO positions")
    return np.concatenate(positions), np.concatenate(layers)


def predict_external_q50(
    panel: P1IorsPanel,
    model: Any,
    *,
    min_peer_temperatures: int,
) -> ExternalPointPrediction:
    """Predict q50 with the target-layer temperature provably masked."""

    dataset: LooDataset = build_loo_dataset(
        [panel.profile],
        min_peer_temperatures=min_peer_temperatures,
        max_rows_per_year_layer=None,
    )
    source_position, expected_layer = _loo_source_positions(
        panel, min_peer_temperatures=min_peer_temperatures
    )
    if dataset.y.size != source_position.size:
        raise AssertionError("P1 LOO feature/key row counts differ")
    if not np.array_equal(dataset.layer, expected_layer):
        raise AssertionError("P1 LOO feature/key layer order differs")
    peer_count_column = dataset.feature_names.index("peer_count")
    for layer in np.unique(dataset.layer):
        rows = dataset.layer == layer
        target_column = dataset.feature_names.index(f"peer_temp_layer_{int(layer)}")
        if np.isfinite(dataset.x[rows, target_column]).any():
            raise AssertionError(f"target temperature leaked for I-ORS layer {layer}")
    prediction = np.asarray(model.predict(dataset.x), dtype=np.float64)
    if prediction.shape != dataset.y.shape or not np.isfinite(prediction).all():
        raise RuntimeError("external q50 model produced invalid P1 predictions")
    q50 = np.full(panel.frame_length, np.nan, dtype=np.float64)
    peer_count = np.full(panel.frame_length, np.nan, dtype=np.float64)
    eligible = np.zeros(panel.frame_length, dtype=bool)
    q50[source_position] = prediction
    peer_count[source_position] = dataset.x[:, peer_count_column]
    eligible[source_position] = True
    return ExternalPointPrediction(
        q50=q50,
        peer_count=peer_count,
        eligible=eligible,
        audit={
            "source_rows": int(panel.profile.audit["source_rows"]),
            "eligible_rows": int(eligible.sum()),
            "eligible_fraction_of_iors": float(eligible.sum() / panel.profile.audit["source_rows"]),
            "eligible_by_layer": {
                str(int(layer)): int((dataset.layer == layer).sum())
                for layer in np.unique(dataset.layer)
            },
            "feature_count": len(dataset.feature_names),
            "feature_names": list(dataset.feature_names),
            "target_temperature_masked": True,
            "quantiles_used": [0.5],
        },
    )


def _centered_median(
    values: pd.Series,
    segment: pd.Series,
    *,
    rows: int,
    minimum_fraction: float,
) -> pd.Series:
    minimum = max(3, int(np.ceil(rows * minimum_fraction)))
    result = (
        values.groupby(segment, sort=False, observed=True)
        .rolling(window=rows, min_periods=minimum, center=True)
        .median()
    )
    return result.reset_index(level=0, drop=True).sort_index()


def build_point_residual_features(
    frame: pd.DataFrame,
    prediction: ExternalPointPrediction,
    *,
    cadence_minutes: int = 10,
    minimum_fraction: float = 0.25,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build exactly eight point-residual features without crossing any gap."""

    if len(frame) != prediction.q50.size:
        raise ValueError("external predictions must align with the P1 frame")
    if cadence_minutes != 10:
        raise ValueError("the frozen point-residual contract requires 10-minute cadence")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must be in (0, 1]")
    station = frame["station"].astype("string")
    iors_position = np.flatnonzero(station.eq("I-ORS").to_numpy())
    temp = pd.to_numeric(frame["temp"], errors="coerce").to_numpy(dtype=np.float64)
    available = (
        station.eq("I-ORS").to_numpy()
        & prediction.eligible
        & np.isfinite(prediction.q50)
        & np.isfinite(temp)
        & np.isfinite(prediction.peer_count)
    )
    signed = np.where(available, temp - prediction.q50, np.nan)
    result = pd.DataFrame(np.nan, index=frame.index, columns=POINT_RESIDUAL_COLUMNS, dtype=float)
    result["ext_q50_signed_residual"] = signed
    result["ext_q50_abs_residual"] = np.abs(signed)
    result["ext_peer_count"] = np.where(available, prediction.peer_count, np.nan)
    result["ext_peer_support"] = np.where(available, prediction.peer_count / 6.0, np.nan)

    working = frame.iloc[iors_position].loc[:, ["station", "layer", "time"]].copy()
    working["__source_position"] = iors_position
    working["__time"] = pd.to_datetime(working["time"], errors="raise", utc=True, format="mixed")
    working["__residual"] = signed[iors_position]
    working = working.sort_values(
        ["station", "layer", "__time", "__source_position"], kind="mergesort"
    ).reset_index(drop=True)
    group_change = working["station"].ne(working["station"].shift()) | working["layer"].ne(
        working["layer"].shift()
    )
    cadence_break = working["__time"].diff().ne(pd.Timedelta(minutes=cadence_minutes))
    finite = working["__residual"].notna()
    missing_break = ~finite | ~finite.shift(fill_value=False)
    segment_start = group_change | cadence_break | missing_break
    segment = segment_start.cumsum().astype(np.int64)
    residual = working["__residual"]
    difference_per_hour = residual.groupby(segment, sort=False, observed=True).diff() * (
        60.0 / cadence_minutes
    )
    for hours in (24, 72):
        rows = hours * 60 // cadence_minutes
        median = _centered_median(
            residual,
            segment,
            rows=rows,
            minimum_fraction=minimum_fraction,
        )
        slope = _centered_median(
            difference_per_hour,
            segment,
            rows=rows,
            minimum_fraction=minimum_fraction,
        )
        source = working["__source_position"].to_numpy(dtype=np.int64)
        result.iloc[source, result.columns.get_loc(f"ext_residual_median_{hours}h")] = (
            median.to_numpy(dtype=np.float64)
        )
        result.iloc[source, result.columns.get_loc(f"ext_residual_slope_{hours}h")] = (
            slope.to_numpy(dtype=np.float64)
        )
    result = result.loc[:, POINT_RESIDUAL_COLUMNS]
    if any("q10" in column or "q90" in column for column in result.columns):
        raise AssertionError("q10/q90 features are forbidden")
    segment_sizes = finite.groupby(segment, sort=False, observed=True).sum()
    audit = {
        "columns": list(POINT_RESIDUAL_COLUMNS),
        "eligible_rows": int(available.sum()),
        "ineligible_iors_rows": int(len(iors_position) - available[iors_position].sum()),
        "segments": int(segment[finite].nunique()),
        "largest_contiguous_eligible_segment_rows": int(segment_sizes.max())
        if len(segment_sizes)
        else 0,
        "rolling_finite_fraction_of_eligible": {
            column: float(result.loc[available, column].notna().mean())
            for column in POINT_RESIDUAL_COLUMNS[4:]
        },
        "gap_safe": True,
        "centered_offline": True,
        "q10_q90_excluded": True,
    }
    return result, audit


def append_point_residual_matrix(
    base_matrix: np.ndarray,
    feature_frame: pd.DataFrame,
    positions: Sequence[int] | np.ndarray,
) -> np.ndarray:
    if tuple(feature_frame.columns) != POINT_RESIDUAL_COLUMNS:
        raise ValueError("point-residual columns differ from the frozen contract")
    index = np.asarray(positions, dtype=np.int64)
    values = feature_frame.iloc[index].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    if len(base_matrix) != len(values):
        raise ValueError("base and point-residual matrices have different rows")
    return np.column_stack([np.asarray(base_matrix, dtype=np.float32), values]).astype(
        np.float32, copy=False
    )


def select_inner_threshold(
    truth: Sequence[int] | np.ndarray,
    probability: Sequence[float] | np.ndarray,
    plateau: Sequence[bool] | np.ndarray,
    candidates: Sequence[float],
) -> tuple[float, np.ndarray, dict[str, Any]]:
    """Select one preregistered threshold; higher thresholds win exact ties."""

    from p1_qc.metrics import micro_f1

    target = np.asarray(truth, dtype=np.int8)
    score = np.asarray(probability, dtype=np.float64)
    plateau_mask = np.asarray(plateau, dtype=bool)
    if target.ndim != 1 or target.shape != score.shape or target.shape != plateau_mask.shape:
        raise ValueError("inner threshold inputs must be equal-length vectors")
    if not np.isin(target, [0, 1]).all() or not np.isfinite(score).all():
        raise ValueError("inner threshold inputs must be binary and finite")
    threshold_values = sorted({float(value) for value in candidates}, reverse=True)
    if not threshold_values or not all(0.0 <= value <= 1.0 for value in threshold_values):
        raise ValueError("threshold candidates must be a non-empty subset of [0, 1]")
    rows: list[dict[str, Any]] = []
    best_threshold: float | None = None
    best_prediction: np.ndarray | None = None
    best_f1 = -1.0
    for threshold in threshold_values:
        prediction = ((score >= threshold) | plateau_mask).astype(np.int8)
        f1 = micro_f1(target, prediction)
        rows.append(
            {
                "threshold": threshold,
                "f1": f1,
                "predicted_positive_rows": int(prediction.sum()),
            }
        )
        if f1 > best_f1:
            best_threshold = threshold
            best_prediction = prediction
            best_f1 = f1
    if best_threshold is None or best_prediction is None:
        raise RuntimeError("threshold selection produced no candidate")
    return (
        best_threshold,
        best_prediction,
        {
            "selected_threshold": best_threshold,
            "selected_f1": best_f1,
            "tie_break": "higher_threshold",
            "candidates": rows,
        },
    )


def apply_point_residual_gate(
    *,
    overall_weighted_f1_delta: float,
    iors_micro_f1_delta: float,
    anomaly_type_recall_delta: Mapping[str, float],
    normal_fp_day_relative_increase: float | None,
    worst_iors_layer_f1_delta: float,
    paired_bootstrap_ci90_lower: float,
    improved_folds: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen multi-criterion promotion gate without tuning."""

    required_types = ("spike", "noise", "flatline", "offset", "drift")
    missing = [value for value in required_types if value not in anomaly_type_recall_delta]
    if missing:
        raise KeyError(f"anomaly type deltas are missing {missing}")
    type_floor = float(contract["every_anomaly_type_recall_delta_min"])
    type_checks = {
        value: bool(float(anomaly_type_recall_delta[value]) >= type_floor)
        for value in required_types
    }
    offset_or_drift_delta = max(
        float(anomaly_type_recall_delta["offset"]),
        float(anomaly_type_recall_delta["drift"]),
    )
    checks = {
        "overall_weighted_f1": bool(
            overall_weighted_f1_delta >= float(contract["overall_weighted_f1_delta_min"])
        ),
        "iors_micro_f1": bool(iors_micro_f1_delta >= float(contract["iors_micro_f1_delta_min"])),
        "offset_or_drift_recall": bool(
            offset_or_drift_delta >= float(contract["offset_or_drift_recall_delta_min"])
        ),
        "every_anomaly_type_recall": all(type_checks.values()),
        "normal_fp_day": bool(
            normal_fp_day_relative_increase is not None
            and normal_fp_day_relative_increase
            < float(contract["normal_fp_day_relative_increase_lt"])
        ),
        "worst_iors_layer": bool(
            worst_iors_layer_f1_delta >= float(contract["worst_iors_layer_f1_delta_min"])
        ),
        "paired_bootstrap": bool(
            paired_bootstrap_ci90_lower > float(contract["paired_bootstrap_ci90_lower_gt"])
        ),
        "minimum_improved_folds": bool(improved_folds >= int(contract["minimum_improved_folds"])),
    }
    passed = all(checks.values())
    return {
        "passed": passed,
        "decision": "GO_POINT_RESIDUAL" if passed else "NO_GO_POINT_RESIDUAL",
        "checks": checks,
        "diagnostics": {
            "overall_weighted_f1_delta": float(overall_weighted_f1_delta),
            "iors_micro_f1_delta": float(iors_micro_f1_delta),
            "anomaly_type_recall_delta": {
                value: float(anomaly_type_recall_delta[value]) for value in required_types
            },
            "anomaly_type_checks": type_checks,
            "offset_or_drift_recall_delta": offset_or_drift_delta,
            "normal_fp_day_relative_increase": normal_fp_day_relative_increase,
            "worst_iors_layer_f1_delta": float(worst_iors_layer_f1_delta),
            "paired_bootstrap_ci90_lower": float(paired_bootstrap_ci90_lower),
            "improved_folds": int(improved_folds),
        },
    }


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def independent_expected_replacement_keys(
    reference: pd.DataFrame,
    reference_train_positions: Sequence[int] | np.ndarray,
    external_eligible: Sequence[bool] | np.ndarray,
    *,
    candidate_folds: Sequence[str],
) -> pd.DataFrame:
    """Derive candidate keys from reference-wide eligibility, never model output rows."""

    key = [*KEY_COLUMNS, "fold"]
    missing = sorted(set(key).difference(reference.columns))
    if missing:
        raise KeyError(f"reference is missing {missing}")
    positions = np.asarray(reference_train_positions, dtype=np.int64)
    eligibility = np.asarray(external_eligible, dtype=bool)
    if positions.shape != (len(reference),):
        raise ValueError("reference_train_positions must align with reference rows")
    if (positions < 0).any() or (positions >= len(eligibility)).any():
        raise ValueError("reference_train_positions are outside the eligibility vector")
    folds = tuple(str(value) for value in candidate_folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("candidate_folds must be unique and non-empty")
    candidate = reference["fold"].astype(str).isin(folds).to_numpy()
    iors = reference["station"].astype("string").eq("I-ORS").to_numpy()
    mask = candidate & iors & eligibility[positions]
    expected = reference.loc[mask, key].copy()
    if expected.empty or expected.duplicated(key).any():
        raise ValueError("independent expected replacement key set is empty or duplicated")
    return expected.reset_index(drop=True)


def compose_incumbent_predictions(
    reference: pd.DataFrame,
    replacements: pd.DataFrame,
    expected_eligible_keys: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace exactly eligible I-ORS rows and prove every other row unchanged."""

    key = [*KEY_COLUMNS, "fold"]
    reference_required = {*key, "prediction", "probability"}
    replacement_required = {*key, "candidate_prediction", "candidate_probability"}
    if missing := sorted(reference_required.difference(reference.columns)):
        raise KeyError(f"reference is missing {missing}")
    if missing := sorted(replacement_required.difference(replacements.columns)):
        raise KeyError(f"replacements are missing {missing}")
    if reference.duplicated(key).any() or replacements.duplicated(key).any():
        raise ValueError("OOF compose keys must be unique")
    if reference["probability"].dtype != np.dtype("float32"):
        raise TypeError("incumbent probability must retain its original float32 dtype")
    if reference["prediction"].dtype != np.dtype("int8"):
        raise TypeError("incumbent prediction must retain its original int8 dtype")
    reference_probability_raw = reference["probability"].to_numpy(copy=False)
    reference_prediction_raw = reference["prediction"].to_numpy(copy=False)
    if (
        not np.isfinite(reference_probability_raw).all()
        or ((reference_probability_raw < 0.0) | (reference_probability_raw > 1.0)).any()
    ):
        raise ValueError("incumbent probabilities must be finite and in [0, 1]")
    if not np.isin(reference_prediction_raw, [0, 1]).all():
        raise ValueError("incumbent predictions must be binary")
    replacement_probability = replacements["candidate_probability"].to_numpy()
    replacement_prediction = replacements["candidate_prediction"].to_numpy()
    if (
        not np.isfinite(replacement_probability).all()
        or ((replacement_probability < 0.0) | (replacement_probability > 1.0)).any()
    ):
        raise ValueError("replacement probabilities must be finite and in [0, 1]")
    if not np.isin(replacement_prediction, [0, 1]).all():
        raise ValueError("replacement predictions must be binary")
    if not replacements["station"].astype("string").eq("I-ORS").all():
        raise ValueError("only I-ORS rows may be replaced")
    expected_index = pd.MultiIndex.from_frame(expected_eligible_keys.loc[:, key])
    replacement_index = pd.MultiIndex.from_frame(replacements.loc[:, key])
    if not expected_index.is_unique or set(expected_index) != set(replacement_index):
        raise ValueError("replacement keys differ from all eligible I-ORS OOF keys")
    output = reference.loc[:, key].copy()
    incumbent_prediction = reference["prediction"].to_numpy(dtype=np.int8, copy=True)
    incumbent_probability = reference["probability"].to_numpy(dtype=np.float32, copy=True)
    candidate_prediction = incumbent_prediction.copy()
    candidate_probability = incumbent_probability.copy()
    reference_index = pd.MultiIndex.from_frame(reference.loc[:, key])
    replacement_position = reference_index.get_indexer(replacement_index)
    if (replacement_position < 0).any():
        raise ValueError("replacement keys are absent from incumbent OOF")
    candidate_prediction[replacement_position] = replacements["candidate_prediction"].to_numpy(
        dtype=np.int8
    )
    candidate_probability[replacement_position] = replacements["candidate_probability"].to_numpy(
        dtype=np.float32
    )
    external_eligible = np.zeros(len(output), dtype=bool)
    external_eligible[replacement_position] = True
    unchanged = ~external_eligible
    sg = ~reference["station"].astype("string").eq("I-ORS").to_numpy()
    ineligible_i = ~sg & unchanged
    if not np.array_equal(candidate_prediction[unchanged], incumbent_prediction[unchanged]):
        raise AssertionError("an ineligible prediction changed")
    if candidate_probability[unchanged].tobytes() != incumbent_probability[unchanged].tobytes():
        raise AssertionError("an ineligible probability changed bytes")
    output["incumbent_probability"] = incumbent_probability
    output["incumbent_prediction"] = incumbent_prediction
    output["candidate_probability"] = candidate_probability
    output["candidate_prediction"] = candidate_prediction
    output["external_eligible"] = external_eligible
    audit = {
        "rows": len(output),
        "reference_probability_dtype": str(reference["probability"].dtype),
        "reference_prediction_dtype": str(reference["prediction"].dtype),
        "reference_probability_raw_bytes_sha256": hashlib.sha256(
            reference_probability_raw.tobytes()
        ).hexdigest(),
        "reference_prediction_raw_bytes_sha256": hashlib.sha256(
            reference_prediction_raw.tobytes()
        ).hexdigest(),
        "replaced_iors_rows": int(external_eligible.sum()),
        "unchanged_sg_rows": int(sg.sum()),
        "unchanged_ineligible_iors_rows": int(ineligible_i.sum()),
        "sg_prediction_sha256_before": _array_sha256(incumbent_prediction[sg]),
        "sg_prediction_sha256_after": _array_sha256(candidate_prediction[sg]),
        "sg_probability_sha256_before": _array_sha256(incumbent_probability[sg]),
        "sg_probability_sha256_after": _array_sha256(candidate_probability[sg]),
        "ineligible_i_prediction_sha256_before": _array_sha256(incumbent_prediction[ineligible_i]),
        "ineligible_i_prediction_sha256_after": _array_sha256(candidate_prediction[ineligible_i]),
        "ineligible_i_probability_sha256_before": _array_sha256(
            incumbent_probability[ineligible_i]
        ),
        "ineligible_i_probability_sha256_after": _array_sha256(candidate_probability[ineligible_i]),
        "sg_byte_identical": True,
        "ineligible_iors_byte_identical": True,
    }
    return output, audit


__all__ = [
    "CanonicalArtifactPaths",
    "ExternalPointPrediction",
    "KEY_COLUMNS",
    "P1IorsPanel",
    "POINT_RESIDUAL_COLUMNS",
    "append_point_residual_matrix",
    "apply_point_residual_gate",
    "build_p1_iors_panel",
    "build_point_residual_features",
    "canonical_artifact_paths",
    "compose_incumbent_predictions",
    "independent_expected_replacement_keys",
    "predict_external_q50",
    "select_inner_threshold",
]
