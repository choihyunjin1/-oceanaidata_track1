"""Executable numerical closure for the preregistered P1 Cycle-1 screen.

The module has no filesystem entry point.  It is loaded only after the v2
runner has authenticated its external authorization, seal, complete
transitive code closure, runtime, and private immutable input snapshot.
Official evaluation inputs and submission construction are intentionally
absent.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from . import long_event_segment_proposal_rescore as frozen
from .change_points import ChangePointConfig, propose_change_intervals
from .metrics import binary_counts
from .validation import normal_station_layer_day_fp, paired_block_bootstrap

EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v2"
SCIENTIFIC_EXPERIMENT_ID = "p1_long_event_segment_proposal_rescore_20260826_v1"
DESIGN_SHA256 = "31b0bde27d8ef7e2b42135709563cca0bcca61c6ec6fdabefbb3530906869563"
OPERATIONAL_AMENDMENT_V2_SHA256 = "b33f7d386e05cd7ab79976f58e9f4ab752f37cfe6a8856849867ef5f541cb276"
EXECUTION_CLOSURE_V3_SHA256 = "b7afec5a11e908f6e5fd8a1ef28404cbb062da39764b2649ddabc2659a56ad99"
TRUST_FIREWALL_V5_SHA256 = "bd0370c7100ae7602eb2b045b4ef69bf7808d345fcb3255234c0726225e57563"

KEY_COLUMNS = ("station", "year", "layer", "time")
ROUND_B_PREFIX = "event_day_balanced_binary_lgbm"
ROUND_B_SEEDS = (20260813, 20260829, 20260847)
SEGMENT_SEEDS = (20260826, 20260843, 20260871)
FOLD_ORDER = ("2025_q2", "2025_q3", "2025_q4")
CONTEXT_BANKS = ((24, 72), (48, 168), (24, 72, 168))
DECODERS = ("CONNECTED_ONLY", "DUAL_BOUNDARY_DISCONNECTED_ALLOWED")
THRESHOLDS = (0.75, 0.85, 0.92)
MIN_INTERVAL_ROWS = 19

CHANGE_POINT_PARAMETERS = {
    "high_seed_threshold": 0.65,
    "low_seed_threshold": 0.35,
    "min_baseline_rows": 6,
    "min_return_rows": 3,
    "mean_gain_threshold": 0.5,
    "variance_gain_threshold": 0.25,
    "slope_gain_threshold": 0.25,
    "baseline_z_threshold": 3.0,
    "return_z_threshold": 3.0,
    "max_candidates_per_seed_run": 8,
    "robust_epsilon": 1.0e-6,
}

SEGMENT_PARAMETERS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "n_jobs": 1,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


@dataclass
class Surface:
    """One isolated OOS surface with frozen anchor outputs."""

    surface_id: str
    frame: pd.DataFrame
    truth: np.ndarray | None
    anomaly_type: pd.Series | None
    anchor_probability: np.ndarray
    anchor_prediction: np.ndarray
    plateau: np.ndarray
    spike: np.ndarray
    first_time: pd.Timestamp
    last_time: pd.Timestamp
    key_sha256: str
    proposal_central_start: pd.Timestamp | None = None
    proposal_central_end_exclusive: pd.Timestamp | None = None


@dataclass
class BankContext:
    """Target-free proposals/features plus optional inner-only targets."""

    surface: Surface
    bank: tuple[int, ...]
    proposals: tuple[frozen.SegmentRecord, ...]
    features: pd.DataFrame
    targets: np.ndarray | None


@dataclass
class InnerCellRun:
    """Three-seed validation probabilities for one window/cell."""

    window_id: str
    cell_id: str
    bank: tuple[int, ...]
    decoder: str
    seed_probabilities: tuple[np.ndarray, ...]
    ensemble_probability: np.ndarray


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_sha(values: Sequence[Any], *, dtype: str) -> str:
    array = np.asarray(values, dtype=dtype)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _key_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for ordinal, row in enumerate(
        frame.loc[:, list(KEY_COLUMNS)].itertuples(index=False, name=None)
    ):
        for value in (ordinal, str(row[0]), int(row[1]), int(row[2]), str(row[3])):
            encoded = str(value).encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def _deadline(deadline_epoch: float, phase: str) -> None:
    if time.time() >= deadline_epoch:
        raise TimeoutError(f"absolute deadline expired before {phase}")


def _measurement_columns(frame: pd.DataFrame) -> tuple[str, str]:
    temp = "temp" if "temp" in frame.columns else "temp_raw"
    psal = "psal" if "psal" in frame.columns else "psal_raw"
    if temp not in frame.columns or psal not in frame.columns:
        raise KeyError("temperature or salinity column is missing")
    return temp, psal


def _segment_rolling(
    values: np.ndarray,
    segment_ids: np.ndarray,
    *,
    window: int,
    statistic: str,
    reverse: bool = False,
) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    for segment_id in np.unique(segment_ids):
        positions = np.flatnonzero(segment_ids == segment_id)
        current = pd.Series(np.asarray(values[positions], dtype=np.float64))
        if reverse:
            current = current.iloc[::-1].reset_index(drop=True)
        rolling = current.rolling(window=window, min_periods=6, center=False)
        if statistic == "median":
            calculated = rolling.median()
        elif statistic == "sum":
            calculated = rolling.sum()
        elif statistic == "count":
            calculated = rolling.count()
        else:
            raise ValueError("unknown bounded rolling statistic")
        array = calculated.to_numpy(dtype=np.float64)
        if reverse:
            array = array[::-1]
        result[positions] = array
    return result


def _bounded_robust_channel(
    values: np.ndarray,
    segment_ids: np.ndarray,
    *,
    context_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Centered robust z plus bounded leading/trailing CUSUM."""

    center = np.full(len(values), np.nan, dtype=np.float64)
    mad = np.full(len(values), np.nan, dtype=np.float64)
    for segment_id in np.unique(segment_ids):
        positions = np.flatnonzero(segment_ids == segment_id)
        current = pd.Series(np.asarray(values[positions], dtype=np.float64))
        rolling = current.rolling(
            window=context_rows,
            min_periods=6,
            center=True,
        )
        local_center = rolling.median()
        local_mad = rolling.apply(
            lambda sample: float(np.median(np.abs(sample - np.median(sample)))),
            raw=True,
        )
        center[positions] = local_center.to_numpy(dtype=np.float64)
        mad[positions] = local_mad.to_numpy(dtype=np.float64)
    scale = np.maximum(1.4826 * mad, CHANGE_POINT_PARAMETERS["robust_epsilon"])
    robust_z = (np.asarray(values, dtype=np.float64) - center) / scale
    trailing_sum = _segment_rolling(
        robust_z,
        segment_ids,
        window=context_rows,
        statistic="sum",
    )
    trailing_count = _segment_rolling(
        robust_z,
        segment_ids,
        window=context_rows,
        statistic="count",
    )
    leading_sum = _segment_rolling(
        robust_z,
        segment_ids,
        window=context_rows,
        statistic="sum",
        reverse=True,
    )
    leading_count = _segment_rolling(
        robust_z,
        segment_ids,
        window=context_rows,
        statistic="count",
        reverse=True,
    )
    trailing = np.abs(trailing_sum) / np.sqrt(np.maximum(trailing_count, 1.0))
    leading = np.abs(leading_sum) / np.sqrt(np.maximum(leading_count, 1.0))
    bounded_cusum = np.maximum(trailing, leading)
    bounded_cusum[(trailing_count < 6) & (leading_count < 6)] = np.nan
    return robust_z, bounded_cusum


def generate_bounded_target_free_proposals(
    frame: pd.DataFrame,
    anchor_probability: Sequence[float],
    context_bank_hours: Sequence[int],
) -> tuple[frozen.SegmentRecord, ...]:
    """Generate proposals with no target and no context beyond the bank/surface."""

    forbidden = {"label", "anomaly_type", "derived_error_type"}.intersection(frame.columns)
    if forbidden:
        raise ValueError("proposal frame contains target/evaluation columns")
    bank = tuple(int(value) for value in context_bank_hours)
    if bank not in CONTEXT_BANKS:
        raise ValueError("context bank is not preregistered")
    probability = np.asarray(anchor_probability, dtype=np.float64)
    if probability.shape != (len(frame),) or not np.isfinite(probability).all():
        raise ValueError("anchor probability shape/finite contract failed")
    if ((probability < 0.0) | (probability > 1.0)).any():
        raise ValueError("anchor probability lies outside [0,1]")
    segment_ids = frozen.exact_gap_safe_segment_ids(frame)
    robust_channels: list[np.ndarray] = []
    cusum_channels: list[np.ndarray] = []
    for column in _measurement_columns(frame):
        values = frame[column].to_numpy(dtype=np.float64)
        for hours in bank:
            robust_z, bounded_cusum = _bounded_robust_channel(
                values,
                segment_ids,
                context_rows=int(hours) * 6,
            )
            robust_channels.append(np.abs(robust_z))
            cusum_channels.append(bounded_cusum)
    physical = np.nanmax(
        np.column_stack([*robust_channels, *cusum_channels]),
        axis=1,
    )
    finite = np.isfinite(physical)
    proposal_probability = np.full(len(frame), np.nan, dtype=np.float64)
    proposal_probability[finite] = np.minimum(
        np.clip(physical[finite] / 6.0, 0.0, 1.0),
        1.0 - probability[finite],
    )
    # The bounded robust mean is used only as a target-free measurement
    # channel for the pre-existing proposal primitive.
    residual = np.nanmean(np.column_stack(robust_channels), axis=1)
    config = ChangePointConfig(
        mode="offline",
        max_flank_rows=max(bank) * 6,
        min_interval_rows=MIN_INTERVAL_ROWS,
        max_interval_rows=max(bank) * 12,
        **CHANGE_POINT_PARAMETERS,
    )
    generated = propose_change_intervals(
        residual,
        proposal_probability,
        segment_ids,
        station=frame["station"].astype(str).to_numpy(),
        layer=frame["layer"].to_numpy(),
        row_ids=np.arange(len(frame), dtype=np.int64),
        times=frame["time"].to_numpy(),
        config=config,
    )
    records: dict[tuple[str, int, int, int], frozen.SegmentRecord] = {}
    for proposal in generated.proposals:
        if proposal.duration_rows < MIN_INTERVAL_ROWS:
            continue
        key = (
            str(proposal.station),
            int(proposal.layer),
            int(proposal.start),
            int(proposal.stop),
        )
        record = frozen.SegmentRecord(
            proposal_id=str(proposal.proposal_id),
            station=key[0],
            layer=key[1],
            segment_id=int(proposal.segment_id),
            start=key[2],
            stop=key[3],
            start_boundary_score=float(1.0 - np.exp(-abs(proposal.baseline_z))),
            end_boundary_score=float(1.0 - np.exp(-abs(proposal.return_z or 0.0))),
            source=str(proposal.source),
        )
        incumbent = records.get(key)
        if incumbent is None or (
            record.start_boundary_score + record.end_boundary_score
            > incumbent.start_boundary_score + incumbent.end_boundary_score
        ):
            records[key] = record
    return tuple(
        records[key]
        for key in sorted(records, key=lambda item: (item[0], item[1], item[2], item[3]))
    )


def _same_station_cross_layer_residual(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Same-station/same-time other-layer peer; G has no peer by contract."""

    values = frame[column].to_numpy(dtype=np.float64)
    station = frame["station"].astype(str).to_numpy()
    layer = frame["layer"].to_numpy()
    time_values = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    work = pd.DataFrame(
        {
            "station": station,
            "layer": layer,
            "time": time_values,
            "value": values,
            "position": np.arange(len(frame), dtype=np.int64),
        }
    )
    residual = np.full(len(frame), np.nan, dtype=np.float64)
    count = np.zeros(len(frame), dtype=np.int64)
    eligible = work["station"].isin(["I-ORS", "S-ORS"])
    for (_station, _time), group in work.loc[eligible].groupby(
        ["station", "time"],
        sort=False,
        observed=True,
    ):
        finite = group[np.isfinite(group["value"].to_numpy(dtype=np.float64))]
        for row in group.itertuples(index=False):
            if not np.isfinite(row.value):
                continue
            peers = finite.loc[finite["layer"].ne(row.layer), "value"].to_numpy(dtype=np.float64)
            if len(peers):
                residual[int(row.position)] = float(row.value - np.median(peers))
                count[int(row.position)] = len(peers)
    if np.isfinite(residual[station == "G-ORS"]).any() or count[station == "G-ORS"].any():
        raise AssertionError("G-ORS used a prohibited peer")
    return residual, count


def build_bounded_segment_features(
    frame: pd.DataFrame,
    anchor_probability: Sequence[float],
    anchor_prediction: Sequence[int],
    proposals: Sequence[frozen.SegmentRecord],
    context_bank_hours: Sequence[int],
) -> pd.DataFrame:
    """Build bounded features and the corrected same-station cross-layer peer."""

    forbidden = {"label", "anomaly_type", "derived_error_type"}.intersection(frame.columns)
    if forbidden:
        raise ValueError("feature frame contains target/evaluation columns")
    bank = tuple(int(value) for value in context_bank_hours)
    if bank not in CONTEXT_BANKS:
        raise ValueError("context bank is not preregistered")
    probability = np.asarray(anchor_probability, dtype=np.float64)
    prediction = np.asarray(anchor_prediction, dtype=np.int8)
    if probability.shape != (len(frame),) or prediction.shape != (len(frame),):
        raise ValueError("anchor arrays differ from feature frame")
    segment_ids = frozen.exact_gap_safe_segment_ids(frame)
    measurements = _measurement_columns(frame)
    peer = {column: _same_station_cross_layer_residual(frame, column) for column in measurements}
    parsed_time = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    rows: list[dict[str, float | int | str]] = []
    for proposal in proposals:
        if not 0 <= proposal.start < proposal.stop <= len(frame):
            raise ValueError("proposal exceeds isolated surface")
        proposal_segments = np.unique(segment_ids[proposal.start : proposal.stop])
        if len(proposal_segments) != 1 or int(proposal_segments[0]) != proposal.segment_id:
            raise ValueError("proposal crosses an exact-cadence segment")
        positions = np.flatnonzero(segment_ids == proposal.segment_id)
        segment_start = int(positions[0])
        segment_stop = int(positions[-1]) + 1
        if (
            str(frame.iloc[proposal.start]["station"]) != proposal.station
            or int(frame.iloc[proposal.start]["layer"]) != proposal.layer
        ):
            raise ValueError("proposal identity differs from isolated surface")
        interior = slice(proposal.start, proposal.stop)
        row: dict[str, float | int | str] = {
            "proposal_id": proposal.proposal_id,
            "station_code": {"G-ORS": 0, "I-ORS": 1, "S-ORS": 2}.get(
                proposal.station,
                -1,
            ),
            "layer": proposal.layer,
            "duration_rows": proposal.duration_rows,
            "start_boundary_score": proposal.start_boundary_score,
            "end_boundary_score": proposal.end_boundary_score,
            "anchor_probability_mean": float(np.mean(probability[interior])),
            "anchor_probability_min": float(np.min(probability[interior])),
            "anchor_probability_max": float(np.max(probability[interior])),
            "anchor_positive_fraction": float(np.mean(prediction[interior])),
            "anchor_negative_fraction": float(np.mean(prediction[interior] == 0)),
            "season_sin": float(
                np.sin(2.0 * np.pi * parsed_time.iloc[proposal.start].month / 12.0)
            ),
            "season_cos": float(
                np.cos(2.0 * np.pi * parsed_time.iloc[proposal.start].month / 12.0)
            ),
        }
        depth_column = "depth" if "depth" in frame.columns else "depth_raw"
        depth = frame[depth_column].to_numpy(dtype=np.float64)[interior]
        finite_depth = depth[np.isfinite(depth)]
        row["depth_regime_median"] = float(np.median(finite_depth)) if len(finite_depth) else 0.0
        for hours in bank:
            flank = int(hours) * 6
            pre = slice(max(segment_start, proposal.start - flank), proposal.start)
            post = slice(proposal.stop, min(segment_stop, proposal.stop + flank))
            for column in measurements:
                values = frame[column].to_numpy(dtype=np.float64)
                inside = values[interior]
                left = values[pre]
                right = values[post]
                finite_inside = inside[np.isfinite(inside)]
                finite_left = left[np.isfinite(left)]
                finite_right = right[np.isfinite(right)]
                prefix = f"{column}_{hours}h"
                row[f"{prefix}_missing_fraction"] = float(np.mean(~np.isfinite(inside)))
                inside_median = float(np.median(finite_inside)) if len(finite_inside) else 0.0
                left_median = float(np.median(finite_left)) if len(finite_left) else 0.0
                right_median = float(np.median(finite_right)) if len(finite_right) else 0.0
                row[f"{prefix}_interior_median"] = inside_median
                row[f"{prefix}_pre_contrast"] = inside_median - left_median
                row[f"{prefix}_post_contrast"] = inside_median - right_median
                row[f"{prefix}_return_to_baseline"] = abs(left_median - right_median)
                flank_values = np.concatenate((finite_left, finite_right))
                if len(flank_values):
                    flank_median = float(np.median(flank_values))
                    flank_mad = 1.4826 * float(np.median(np.abs(flank_values - flank_median)))
                else:
                    flank_median = flank_mad = 0.0
                row[f"{prefix}_leave_center_out_median"] = flank_median
                row[f"{prefix}_leave_center_out_mad"] = flank_mad
                if len(finite_inside) >= 2:
                    coordinate = np.arange(len(finite_inside), dtype=np.float64)
                    row[f"{prefix}_slope"] = float(np.polyfit(coordinate, finite_inside, 1)[0])
                    row[f"{prefix}_variance"] = float(np.var(finite_inside))
                else:
                    row[f"{prefix}_slope"] = 0.0
                    row[f"{prefix}_variance"] = 0.0
                if len(finite_inside) >= 3:
                    coordinate = np.arange(len(finite_inside), dtype=np.float64)
                    row[f"{prefix}_curvature"] = float(np.polyfit(coordinate, finite_inside, 2)[0])
                else:
                    row[f"{prefix}_curvature"] = 0.0
                flank_variance = float(np.var(flank_values)) if len(flank_values) >= 2 else 0.0
                row[f"{prefix}_variance_ratio"] = float(
                    row[f"{prefix}_variance"] / max(flank_variance, 1.0e-6)
                )
                peer_residual, peer_count = peer[column]
                peer_inside = peer_residual[interior]
                finite_peer = peer_inside[np.isfinite(peer_inside)]
                row[f"{prefix}_same_station_cross_layer_residual_median"] = (
                    float(np.median(finite_peer)) if len(finite_peer) else 0.0
                )
                row[f"{prefix}_same_station_cross_layer_available_fraction"] = float(
                    np.mean(peer_count[interior] > 0)
                )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    numeric = result.drop(columns=["proposal_id"])
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise RuntimeError("bounded segment features contain non-finite values")
    return result


def fit_segment_model(
    features: pd.DataFrame,
    target: Sequence[int],
    *,
    seed: int,
) -> Any:
    """Fit exactly one prospectively authorized segment LightGBM model."""

    if int(seed) not in SEGMENT_SEEDS:
        raise ValueError("segment seed is not registered")
    y = np.asarray(target, dtype=np.int8)
    if y.shape != (len(features),) or len(np.unique(y)) != 2:
        raise ValueError("segment target must be aligned and contain both classes")
    if "proposal_id" not in features.columns:
        raise KeyError("segment features lack proposal_id")
    import lightgbm as lgb

    seeded = {
        **SEGMENT_PARAMETERS,
        "random_state": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "extra_seed": int(seed),
    }
    model = lgb.LGBMClassifier(**seeded)
    model.fit(features.drop(columns=["proposal_id"]), y)
    return model


def _surface_frame(train: pd.DataFrame, positions: np.ndarray) -> pd.DataFrame:
    columns = ["station", "year", "layer", "time", "temp", "psal", "depth"]
    missing = sorted(set(columns).difference(train.columns))
    if missing:
        raise KeyError(f"source surface columns missing: {missing}")
    return train.iloc[positions].loc[:, columns].reset_index(drop=True).copy()


def _make_surface(
    surface_id: str,
    train: pd.DataFrame,
    positions: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    plateau: np.ndarray,
    spike: np.ndarray,
    *,
    proposal_central_start: pd.Timestamp | None = None,
    proposal_central_end_exclusive: pd.Timestamp | None = None,
) -> Surface:
    frame = _surface_frame(train, positions)
    truth = train.iloc[positions]["label"].to_numpy(dtype=np.int8)
    anomaly_type = train.iloc[positions]["anomaly_type"].reset_index(drop=True)
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    if not np.isin(truth, [0, 1]).all():
        raise ValueError("surface truth is not binary")
    return Surface(
        surface_id=surface_id,
        frame=frame,
        truth=truth,
        anomaly_type=anomaly_type,
        anchor_probability=np.asarray(probability, dtype=np.float64),
        anchor_prediction=np.asarray(prediction, dtype=np.int8),
        plateau=np.asarray(plateau, dtype=bool),
        spike=np.asarray(spike, dtype=bool),
        first_time=parsed.min(),
        last_time=parsed.max(),
        key_sha256=_key_sha(frame),
        proposal_central_start=proposal_central_start,
        proposal_central_end_exclusive=proposal_central_end_exclusive,
    )


def _positions_between(
    parsed_time: pd.Series,
    start: str | None,
    end: str,
) -> np.ndarray:
    end_time = pd.Timestamp(end).tz_convert("UTC")
    mask = parsed_time.le(end_time)
    if start is not None:
        mask &= parsed_time.ge(pd.Timestamp(start).tz_convert("UTC"))
    return np.flatnonzero(mask.to_numpy())


def _positions_half_open(
    parsed_time: pd.Series,
    start: str,
    end_exclusive: str,
) -> np.ndarray:
    start_time = pd.Timestamp(start).tz_convert("UTC")
    end_time = pd.Timestamp(end_exclusive).tz_convert("UTC")
    mask = parsed_time.ge(start_time) & parsed_time.lt(end_time)
    return np.flatnonzero(mask.to_numpy())


def _fit_anchor_surfaces(
    state: Mapping[str, Any],
    numerical: Any,
    closure: Mapping[str, Any],
    journal: Any,
    deadline_epoch: float,
) -> dict[str, dict[str, Surface]]:
    train = state["train"]
    bundle = state["bundle"]
    parsed = pd.to_datetime(train["time"], errors="raise", utc=True, format="mixed")
    postprocess = closure["out_of_sample_anchor_surfaces"]["uniform_postprocess"]
    surfaces: dict[str, dict[str, Surface]] = {}
    for window in closure["out_of_sample_anchor_surfaces"]["windows"]:
        window_id = str(window["id"])
        fit_positions = _positions_between(
            parsed,
            None,
            str(window["anchor_fit_end_inclusive"]),
        )
        support_positions = _positions_half_open(
            parsed,
            str(window["support_surface_start"]),
            str(window["support_surface_end_exclusive"]),
        )
        calibration_positions = _positions_half_open(
            parsed,
            str(window["segment_calibration_start"]),
            str(window["segment_calibration_end_exclusive"]),
        )
        validation_positions = _positions_between(
            parsed,
            str(window["inner_validation_start"]),
            str(window["inner_validation_end_inclusive"]),
        )
        if (
            not len(fit_positions)
            or not len(support_positions)
            or not len(calibration_positions)
            or not len(validation_positions)
        ):
            raise RuntimeError("OOS anchor split has an empty partition")
        fit_max = parsed.iloc[fit_positions].max()
        calibration_min = parsed.iloc[calibration_positions].min()
        calibration_max = parsed.iloc[calibration_positions].max()
        validation_min = parsed.iloc[validation_positions].min()
        if calibration_min - fit_max < pd.Timedelta(days=7):
            raise RuntimeError("anchor/calibration purge is shorter than seven days")
        if validation_min - calibration_max < pd.Timedelta(days=7):
            raise RuntimeError("calibration/validation purge is shorter than seven days")
        support_min = parsed.iloc[support_positions].min()
        support_max = parsed.iloc[support_positions].max()
        if support_min <= fit_max or support_max >= validation_min:
            raise RuntimeError("proposal support surface crosses fit or validation boundary")
        maximum_context = pd.Timedelta(hours=168)
        if calibration_min - maximum_context <= fit_max:
            raise RuntimeError("central shelf lacks lower 168-hour support")
        if calibration_max + pd.Timedelta(minutes=10) + maximum_context > validation_min:
            raise RuntimeError("central shelf lacks upper 168-hour support")
        # A crash after any feature/surface construction permanently consumes
        # the one-shot namespace.  Reserve the scientific materialization
        # before encoder transforms or predictions so the durable journal can
        # never under-report work that was already started.
        journal.reserve_materialization(f"inner_anchor_surface:{window_id}")
        encoder = numerical.TabularEncoder().fit(bundle, fit_positions)
        fit_features = encoder.transform(bundle, fit_positions)
        support_features = encoder.transform(bundle, support_positions)
        validation_features = encoder.transform(bundle, validation_positions)
        fit_target = train.iloc[fit_positions]["label"].to_numpy(dtype=np.int8)
        fit_metadata = train.iloc[fit_positions][["station", "layer", "time"]].reset_index(
            drop=True
        )
        support_seed_probability: list[np.ndarray] = []
        validation_seed_probability: list[np.ndarray] = []
        for seed in ROUND_B_SEEDS:
            _deadline(deadline_epoch, f"anchor fit reservation {window_id}/{seed}")
            ordinal = journal.reserve_fit(
                "INNER_ANCHOR",
                window_id,
                "ROUND_B_SHARED",
                int(seed),
            )
            model = frozen.fit_round_b_anchor_model(
                fit_features,
                fit_target,
                fit_metadata,
                seed=int(seed),
            )
            journal.complete_fit(ordinal)
            support_seed_probability.append(model.predict_proba(support_features)[:, 1])
            validation_seed_probability.append(model.predict_proba(validation_features)[:, 1])

        def finalize(
            surface_id: str,
            positions: np.ndarray,
            probabilities: Sequence[np.ndarray],
        ) -> Surface:
            frame = _surface_frame(train, positions)
            probability = np.mean(np.vstack(probabilities), axis=0)
            plateau = numerical.detect_plateaus(frame).to_numpy(dtype=bool)
            spike = numerical.detect_singleton_spikes(frame).to_numpy(dtype=bool)
            prediction = numerical.apply_postprocess(
                frame,
                probability,
                plateau,
                spike,
                postprocess,
            )
            return _make_surface(
                surface_id,
                train,
                positions,
                probability,
                prediction,
                plateau,
                spike,
            )

        support_frame = _surface_frame(train, support_positions)
        support_probability = np.mean(np.vstack(support_seed_probability), axis=0)
        if (
            len(support_probability) != len(support_positions)
            or not np.isfinite(support_probability).all()
            or np.any((support_probability < 0.0) | (support_probability > 1.0))
        ):
            raise RuntimeError("full OOS support probability coverage is invalid")
        support_plateau = numerical.detect_plateaus(support_frame).to_numpy(dtype=bool)
        support_spike = numerical.detect_singleton_spikes(support_frame).to_numpy(dtype=bool)
        support_prediction = numerical.apply_postprocess(
            support_frame,
            support_probability,
            support_plateau,
            support_spike,
            postprocess,
        )
        support_time = pd.to_datetime(
            support_frame["time"], errors="raise", utc=True, format="mixed"
        )
        if not support_time.gt(fit_max).all():
            raise RuntimeError("support anchor probability is not strictly future OOS")
        central_start = pd.Timestamp(window["segment_calibration_start"]).tz_convert("UTC")
        central_end = pd.Timestamp(window["segment_calibration_end_exclusive"]).tz_convert("UTC")
        central_mask = support_time.ge(central_start) & support_time.lt(central_end)
        central_support_positions = support_positions[central_mask.to_numpy()]
        if not np.array_equal(central_support_positions, calibration_positions):
            raise RuntimeError("central shelf/support key alignment changed")

        surfaces[window_id] = {
            "calibration": _make_surface(
                f"{window_id}:calibration_support",
                train,
                support_positions,
                support_probability,
                support_prediction,
                support_plateau,
                support_spike,
                proposal_central_start=central_start,
                proposal_central_end_exclusive=central_end,
            ),
            "validation": finalize(
                f"{window_id}:validation",
                validation_positions,
                validation_seed_probability,
            ),
        }
    return surfaces


def _build_context(
    surface: Surface,
    bank: tuple[int, ...],
    *,
    include_targets: bool,
) -> BankContext:
    proposals = generate_bounded_target_free_proposals(
        surface.frame,
        surface.anchor_probability,
        bank,
    )
    if surface.proposal_central_start is not None:
        parsed = pd.to_datetime(surface.frame["time"], errors="raise", utc=True, format="mixed")
        central_end = surface.proposal_central_end_exclusive
        if central_end is None:
            raise RuntimeError("central shelf has no exclusive end")
        proposals = tuple(
            proposal
            for proposal in proposals
            if parsed.iloc[proposal.start] >= surface.proposal_central_start
            and (parsed.iloc[proposal.stop - 1] + pd.Timedelta(minutes=10) <= central_end)
            and (parsed.iloc[proposal.start] - pd.Timedelta(hours=max(bank)) >= surface.first_time)
            and (
                parsed.iloc[proposal.stop - 1]
                + pd.Timedelta(minutes=10)
                + pd.Timedelta(hours=max(bank))
                <= surface.last_time + pd.Timedelta(minutes=10)
            )
        )
    features = build_bounded_segment_features(
        surface.frame,
        surface.anchor_probability,
        surface.anchor_prediction,
        proposals,
        bank,
    )
    if include_targets and (surface.truth is None or surface.anomaly_type is None):
        raise RuntimeError("inner target surface is unavailable")
    targets = (
        frozen.segment_training_targets(
            surface.truth,
            surface.anomaly_type,
            surface.frame,
            proposals,
        )
        if include_targets
        else None
    )
    if len(features) != len(proposals):
        raise RuntimeError("proposal/feature count differs")
    return BankContext(surface, bank, proposals, features, targets)


def _binary_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    counts = binary_counts(truth, prediction)
    denominator = 2 * counts.tp + counts.fp + counts.fn
    precision = counts.tp / (counts.tp + counts.fp) if counts.tp + counts.fp else 0.0
    recall = counts.tp / (counts.tp + counts.fn) if counts.tp + counts.fn else 0.0
    return {
        "rows": len(truth),
        "tp": int(counts.tp),
        "fp": int(counts.fp),
        "fn": int(counts.fn),
        "tn": int(counts.tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * counts.tp / denominator) if denominator else 0.0,
    }


def _f1_delta(truth: np.ndarray, candidate: np.ndarray, anchor: np.ndarray) -> float:
    return float(_binary_metrics(truth, candidate)["f1"] - _binary_metrics(truth, anchor)["f1"])


def _cell_id(bank: tuple[int, ...], decoder: str) -> str:
    return f"bank_{'_'.join(str(value) for value in bank)}__{decoder.lower()}"


def _fit_inner_cells(
    surfaces: Mapping[str, Mapping[str, Surface]],
    journal: Any,
    deadline_epoch: float,
) -> tuple[
    dict[tuple[str, tuple[int, ...], str], InnerCellRun],
    dict[tuple[str, tuple[int, ...], str], Any],
]:
    contexts: dict[tuple[str, tuple[int, ...], str], Any] = {}
    for window_id, pair in surfaces.items():
        for bank in CONTEXT_BANKS:
            journal.reserve_materialization(
                f"inner_context_surface:{window_id}:{'_'.join(map(str, bank))}"
            )
            contexts[(window_id, bank, "calibration")] = _build_context(
                pair["calibration"],
                bank,
                include_targets=True,
            )
            contexts[(window_id, bank, "validation")] = _build_context(
                pair["validation"],
                bank,
                include_targets=True,
            )
            calibration_targets = contexts[(window_id, bank, "calibration")].targets
            if (
                calibration_targets is None
                or len(calibration_targets) == 0
                or len(np.unique(calibration_targets)) != 2
            ):
                raise RuntimeError(
                    "OOS shelf proposal target is empty or single-class; fail closed"
                )
    runs: dict[tuple[str, tuple[int, ...], str], InnerCellRun] = {}
    for window_id in surfaces:
        for bank in CONTEXT_BANKS:
            train_context = contexts[(window_id, bank, "calibration")]
            validation_context = contexts[(window_id, bank, "validation")]
            if train_context.targets is None or validation_context.targets is None:
                raise AssertionError("inner targets are unavailable")
            for decoder in DECODERS:
                cell_id = _cell_id(bank, decoder)
                seed_probabilities: list[np.ndarray] = []
                for seed in SEGMENT_SEEDS:
                    _deadline(
                        deadline_epoch,
                        f"inner segment fit {window_id}/{cell_id}/{seed}",
                    )
                    ordinal = journal.reserve_fit(
                        "INNER_SEGMENT",
                        window_id,
                        cell_id,
                        int(seed),
                    )
                    model = fit_segment_model(
                        train_context.features,
                        train_context.targets,
                        seed=int(seed),
                    )
                    journal.complete_fit(ordinal)
                    seed_probabilities.append(
                        model.predict_proba(
                            validation_context.features.drop(columns=["proposal_id"])
                        )[:, 1]
                    )
                runs[(window_id, bank, decoder)] = InnerCellRun(
                    window_id=window_id,
                    cell_id=cell_id,
                    bank=bank,
                    decoder=decoder,
                    seed_probabilities=tuple(seed_probabilities),
                    ensemble_probability=np.mean(
                        np.vstack(seed_probabilities),
                        axis=0,
                    ),
                )
    return runs, contexts


def _select_inner_cell(
    surfaces: Mapping[str, Mapping[str, Surface]],
    runs: Mapping[tuple[str, tuple[int, ...], str], InnerCellRun],
    contexts: Mapping[tuple[str, tuple[int, ...], str], BankContext],
) -> tuple[frozen.InnerCellSummary, dict[str, Any]]:
    summaries: list[frozen.InnerCellSummary] = []
    evidence: dict[str, Any] = {}
    for bank in CONTEXT_BANKS:
        for decoder in DECODERS:
            cell_id = _cell_id(bank, decoder)
            probabilities = np.concatenate(
                [runs[(window_id, bank, decoder)].ensemble_probability for window_id in surfaces]
            )
            targets = np.concatenate(
                [contexts[(window_id, bank, "validation")].targets for window_id in surfaces]
            )
            threshold, precision = frozen.select_inner_threshold(
                probabilities,
                targets,
            )
            window_deltas: list[float] = []
            added_rows = 0
            if threshold is not None:
                for window_id, pair in surfaces.items():
                    context = contexts[(window_id, bank, "validation")]
                    candidate, audit = frozen.decode_segments(
                        pair["validation"].anchor_prediction,
                        context.proposals,
                        runs[(window_id, bank, decoder)].ensemble_probability,
                        threshold=threshold,
                        decoder_mode=decoder,
                        spike_protected=pair["validation"].spike,
                        flatline_protected=pair["validation"].plateau,
                    )
                    window_deltas.append(
                        _f1_delta(
                            pair["validation"].truth,
                            candidate,
                            pair["validation"].anchor_prediction,
                        )
                    )
                    added_rows += int(audit["added_rows"])
            else:
                window_deltas = [-math.inf, -math.inf, -math.inf]
            summary = frozen.InnerCellSummary(
                cell_id=cell_id,
                context_bank_hours=bank,
                decoder_mode=decoder,
                threshold=threshold,
                interval_precision=float(precision),
                window_f1_deltas=tuple(window_deltas),
                added_rows=int(added_rows),
                eligible=threshold is not None,
            )
            summaries.append(summary)
            evidence[cell_id] = {
                "context_bank_hours": list(bank),
                "decoder_mode": decoder,
                "threshold": threshold,
                "pooled_interval_precision": float(precision),
                "window_f1_deltas": window_deltas,
                "equal_weight_mean_inner_f1_delta": (
                    float(np.mean(window_deltas)) if threshold is not None else None
                ),
                "worst_inner_f1_delta": (
                    float(min(window_deltas)) if threshold is not None else None
                ),
                "added_rows": int(added_rows),
                "eligible": threshold is not None,
            }
    selected = frozen.select_structure_cell(summaries)
    evidence["selected_cell_id"] = selected.cell_id
    return selected, evidence


def _outer_surface(
    state: Mapping[str, Any],
    numerical: Any,
    fold_name: str,
) -> Surface:
    surface = state["surface"]
    mask = surface["fold"].astype(str).eq(fold_name).to_numpy()
    part = surface.loc[mask].reset_index(drop=True)
    keys = list(KEY_COLUMNS)
    frame = part[keys].copy()
    source = (
        state["train"]
        .loc[:, ["temp", "psal", "depth"]]
        .iloc[part["row_position"].to_numpy(dtype=np.int64)]
    )
    for column in ("temp", "psal", "depth"):
        frame[column] = source[column].to_numpy()
    plateau = numerical.detect_plateaus(frame).to_numpy(dtype=bool)
    spike = numerical.detect_singleton_spikes(frame).to_numpy(dtype=bool)
    if not np.array_equal(spike, part["spike_candidate"].to_numpy(dtype=bool)):
        raise RuntimeError("outer spike protection differs from frozen p100 surface")
    parsed = pd.to_datetime(frame["time"], errors="raise", utc=True, format="mixed")
    return Surface(
        surface_id=fold_name,
        frame=frame,
        truth=None,
        anomaly_type=None,
        anchor_probability=part[f"{ROUND_B_PREFIX}__probability"].to_numpy(dtype=np.float64),
        anchor_prediction=part[f"{ROUND_B_PREFIX}__prediction"].to_numpy(dtype=np.int8),
        plateau=plateau,
        spike=spike,
        first_time=parsed.min(),
        last_time=parsed.max(),
        key_sha256=_key_sha(frame),
    )


def _concatenate_training_contexts(
    contexts: Sequence[BankContext],
) -> tuple[pd.DataFrame, np.ndarray]:
    feature_columns: tuple[str, ...] | None = None
    frames: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    for context in contexts:
        if context.targets is None:
            raise RuntimeError("outer training context lacks inner-only target")
        current = tuple(context.features.columns)
        if feature_columns is None:
            feature_columns = current
        elif current != feature_columns:
            raise RuntimeError("segment feature schema differs across OOS surfaces")
        frames.append(context.features)
        targets.append(context.targets)
    result_features = pd.concat(frames, ignore_index=True)
    result_targets = np.concatenate(targets)
    if len(np.unique(result_targets)) != 2:
        raise RuntimeError("outer segment training corpus lacks both classes")
    return result_features, result_targets


def _accepted_interval_audit(
    context: BankContext,
    probability: np.ndarray,
    truth: np.ndarray,
    anomaly_type: pd.Series,
    *,
    threshold: float,
    decoder: str,
) -> dict[str, Any]:
    anchor = context.surface.anchor_prediction
    targets = frozen.segment_training_targets(
        truth,
        anomaly_type,
        context.surface.frame,
        context.proposals,
    )
    disconnected_targets: list[int] = []
    added_lengths: list[int] = []
    accepted = 0
    for ordinal, (proposal, score) in enumerate(zip(context.proposals, probability, strict=True)):
        if score < threshold:
            continue
        interval = np.arange(proposal.start, proposal.stop, dtype=np.int64)
        connected = bool(
            anchor[interval].any()
            or (proposal.start > 0 and anchor[proposal.start - 1] == 1)
            or (proposal.stop < len(anchor) and anchor[proposal.stop] == 1)
        )
        if not connected:
            if decoder == "CONNECTED_ONLY":
                continue
            if (
                proposal.duration_rows < MIN_INTERVAL_ROWS
                or proposal.start_boundary_score < threshold
                or proposal.end_boundary_score < threshold
            ):
                continue
            disconnected_targets.append(int(targets[ordinal]))
        allowed = interval[~(context.surface.spike[interval] | context.surface.plateau[interval])]
        newly_added = int(np.sum(anchor[allowed] == 0))
        if newly_added:
            added_lengths.append(newly_added)
        accepted += 1
    return {
        "accepted_intervals": accepted,
        "disconnected_targets": disconnected_targets,
        "added_lengths": added_lengths,
    }


def _fit_outer(
    state: Mapping[str, Any],
    numerical: Any,
    surfaces: Mapping[str, Mapping[str, Surface]],
    inner_contexts: Mapping[tuple[str, tuple[int, ...], str], BankContext],
    selected: frozen.InnerCellSummary,
    journal: Any,
    deadline_epoch: float,
) -> dict[str, Any]:
    bank = tuple(selected.context_bank_hours)
    decoder = str(selected.decoder_mode)
    threshold = float(selected.threshold)
    outer_surfaces = {fold: _outer_surface(state, numerical, fold) for fold in FOLD_ORDER}
    outer_contexts: dict[tuple[str, tuple[int, ...]], BankContext] = {}
    for fold in FOLD_ORDER:
        for current_bank in CONTEXT_BANKS:
            journal.reserve_materialization(
                f"outer_context_surface:{fold}:{'_'.join(map(str, current_bank))}"
            )
            outer_contexts[(fold, current_bank)] = _build_context(
                outer_surfaces[fold],
                current_bank,
                include_targets=False,
            )
    ensemble_candidates: list[np.ndarray] = []
    anchor_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    fold_labels: list[np.ndarray] = []
    seed_candidate_parts: dict[int, list[np.ndarray]] = {seed: [] for seed in SEGMENT_SEEDS}
    evaluation_items: list[dict[str, Any]] = []
    fold_candidate_digests: dict[str, Any] = {}
    for fold in FOLD_ORDER:
        target_surface = outer_surfaces[fold]
        latest_training_time = target_surface.first_time - pd.Timedelta(days=7)
        training_contexts = [
            inner_contexts[(window_id, bank, "calibration")]
            for window_id, pair in surfaces.items()
            if pair["calibration"].proposal_central_end_exclusive <= latest_training_time
        ]
        if len(training_contexts) != 3:
            raise RuntimeError("outer fold lacks all three preregistered OOS shelves")
        train_features, train_targets = _concatenate_training_contexts(training_contexts)
        validation_context = outer_contexts[(fold, bank)]
        seed_probabilities: list[np.ndarray] = []
        seed_candidates: dict[int, np.ndarray] = {}
        for seed in SEGMENT_SEEDS:
            _deadline(deadline_epoch, f"outer segment fit {fold}/{seed}")
            ordinal = journal.reserve_fit(
                "OUTER_SEGMENT",
                fold,
                selected.cell_id,
                int(seed),
            )
            model = fit_segment_model(train_features, train_targets, seed=int(seed))
            journal.complete_fit(ordinal)
            probability = model.predict_proba(
                validation_context.features.drop(columns=["proposal_id"])
            )[:, 1]
            seed_probabilities.append(probability)
            seed_candidate, _audit = frozen.decode_segments(
                target_surface.anchor_prediction,
                validation_context.proposals,
                probability,
                threshold=threshold,
                decoder_mode=decoder,
                spike_protected=target_surface.spike,
                flatline_protected=target_surface.plateau,
            )
            seed_candidates[int(seed)] = seed_candidate
            seed_candidate_parts[int(seed)].append(seed_candidate)
        ensemble_probability = np.mean(np.vstack(seed_probabilities), axis=0)
        candidate, decoder_audit = frozen.decode_segments(
            target_surface.anchor_prediction,
            validation_context.proposals,
            ensemble_probability,
            threshold=threshold,
            decoder_mode=decoder,
            spike_protected=target_surface.spike,
            flatline_protected=target_surface.plateau,
        )
        evaluation_items.append(
            {
                "fold": fold,
                "context": validation_context,
                "ensemble_probability": ensemble_probability,
                "threshold": threshold,
                "decoder": decoder,
            }
        )
        fold_candidate_digests[fold] = {
            "key_sha256": target_surface.key_sha256,
            "anchor_prediction_sha256": _array_sha(
                target_surface.anchor_prediction,
                dtype="<i1",
            ),
            "candidate_prediction_sha256": _array_sha(candidate, dtype="<i1"),
            "seed_candidate_sha256": {
                str(seed): _array_sha(seed_candidates[seed], dtype="<i1") for seed in SEGMENT_SEEDS
            },
            "decoder_audit": decoder_audit,
        }
        ensemble_candidates.append(candidate)
        anchor_parts.append(target_surface.anchor_prediction)
        metadata_parts.append(target_surface.frame[["station", "layer", "time"]])
        fold_labels.append(np.asarray([fold] * len(candidate), dtype=object))
    candidate = np.concatenate(ensemble_candidates)
    anchor = np.concatenate(anchor_parts)
    metadata = pd.concat(metadata_parts, ignore_index=True)
    folds = np.concatenate(fold_labels)
    seed_candidates = {seed: np.concatenate(parts) for seed, parts in seed_candidate_parts.items()}
    freeze = {
        "rows": len(candidate),
        "fold_order": list(FOLD_ORDER),
        "candidate_sha256": _array_sha(candidate, dtype="<i1"),
        "anchor_sha256": _array_sha(anchor, dtype="<i1"),
        "seed_candidate_sha256": {
            str(seed): _array_sha(values, dtype="<i1") for seed, values in seed_candidates.items()
        },
        "folds": fold_candidate_digests,
    }
    journal.record_outer_freeze(freeze)
    return {
        "anchor": anchor,
        "candidate": candidate,
        "metadata": metadata,
        "folds": folds,
        "seed_candidates": seed_candidates,
        "evaluation_items": evaluation_items,
        "freeze": freeze,
    }


def _event_masks(
    truth: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[np.ndarray, list[np.ndarray]]:
    work = metadata.reset_index(drop=True).copy()
    work["position"] = np.arange(len(work), dtype=np.int64)
    work["truth"] = truth
    work["parsed"] = pd.to_datetime(work["time"], errors="raise", utc=True, format="mixed")
    work.sort_values(["station", "layer", "parsed", "position"], inplace=True)
    grouped = work.groupby(["station", "layer"], sort=False, observed=True)
    contiguous = grouped["parsed"].diff().dt.total_seconds().eq(600)
    prior = grouped["truth"].shift(1).fillna(0).eq(1)
    starts = work["truth"].eq(1) & (~contiguous | ~prior)
    work["event"] = starts.cumsum().where(work["truth"].eq(1), -1).astype(np.int64)
    restored = work.sort_values("position", kind="mergesort")
    event_id = restored["event"].to_numpy(dtype=np.int64)
    events = [np.flatnonzero(event_id == value) for value in np.unique(event_id[event_id >= 0])]
    return event_id, events


def _recall_delta(
    truth: np.ndarray,
    candidate: np.ndarray,
    anchor: np.ndarray,
    mask: np.ndarray,
) -> float:
    positive = mask & (truth == 1)
    if not positive.any():
        return 0.0
    return float(np.mean(candidate[positive] == 1) - np.mean(anchor[positive] == 1))


def _score_outer(
    outer: Mapping[str, Any],
    held_truth: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate = outer["candidate"]
    anchor = outer["anchor"]
    metadata = outer["metadata"]
    folds = outer["folds"]
    truth_parts: list[np.ndarray] = []
    anomaly_parts: list[pd.Series] = []
    interval_audits: list[dict[str, Any]] = []
    for item in outer["evaluation_items"]:
        fold = str(item["fold"])
        context = item["context"]
        truth_part = held_truth.loc[held_truth["fold"].astype(str).eq(fold)].reset_index(drop=True)
        if not context.surface.frame[list(KEY_COLUMNS)].equals(truth_part[list(KEY_COLUMNS)]):
            raise RuntimeError("outer held truth keys differ after prediction freeze")
        truth_array = truth_part["label"].to_numpy(dtype=np.int8)
        anomaly_part = truth_part["anomaly_type"].reset_index(drop=True)
        truth_parts.append(truth_array)
        anomaly_parts.append(anomaly_part)
        interval_audits.append(
            _accepted_interval_audit(
                context,
                item["ensemble_probability"],
                truth_array,
                anomaly_part,
                threshold=float(item["threshold"]),
                decoder=str(item["decoder"]),
            )
        )
    truth = np.concatenate(truth_parts)
    anomaly_type = pd.concat(anomaly_parts, ignore_index=True)
    if len(truth) != len(candidate) or len(anomaly_type) != len(candidate):
        raise RuntimeError("post-freeze outer truth length differs from predictions")
    pooled_candidate = _binary_metrics(truth, candidate)
    pooled_anchor = _binary_metrics(truth, anchor)
    pooled_delta = float(pooled_candidate["f1"] - pooled_anchor["f1"])
    bootstrap = paired_block_bootstrap(
        truth,
        candidate,
        anchor,
        metadata,
        replicates=5000,
        seed=20260826,
        cadence_minutes=10,
        normal_day_timezone="Asia/Seoul",
    )
    fold_metrics: dict[str, Any] = {}
    for fold in FOLD_ORDER:
        mask = folds == fold
        fold_metrics[fold] = {
            "candidate": _binary_metrics(truth[mask], candidate[mask]),
            "anchor": _binary_metrics(truth[mask], anchor[mask]),
            "f1_delta": _f1_delta(truth[mask], candidate[mask], anchor[mask]),
        }
    station_values = metadata["station"].astype(str).to_numpy()
    station_metrics: dict[str, Any] = {}
    for station in sorted(set(station_values)):
        mask = station_values == station
        station_metrics[station] = {
            "candidate": _binary_metrics(truth[mask], candidate[mask]),
            "anchor": _binary_metrics(truth[mask], anchor[mask]),
            "f1_delta": _f1_delta(truth[mask], candidate[mask], anchor[mask]),
        }
    _event_id, events = _event_masks(truth, metadata)
    parsed = pd.to_datetime(metadata["time"], errors="raise", utc=True, format="mixed")
    supported_deltas: list[float] = []
    support_cells: dict[str, Any] = {}
    for station in sorted(set(station_values)):
        for fold in FOLD_ORDER:
            mask = (station_values == station) & (folds == fold)
            positive_rows = int(np.sum(truth[mask] == 1))
            positions = np.flatnonzero(mask)
            position_set = set(positions.tolist())
            positive_events = sum(
                1 for event in events if any(int(value) in position_set for value in event)
            )
            positive_days = int(
                pd.DataFrame(
                    {
                        "station": station_values[mask],
                        "layer": metadata.loc[mask, "layer"].to_numpy(),
                        "day": parsed.loc[mask]
                        .dt.tz_convert("Asia/Seoul")
                        .dt.strftime("%Y-%m-%d")
                        .to_numpy(),
                        "truth": truth[mask],
                    }
                )
                .loc[lambda value: value["truth"].eq(1)]
                .drop_duplicates(["station", "layer", "day"])
                .shape[0]
            )
            delta = _f1_delta(truth[mask], candidate[mask], anchor[mask])
            supported = positive_rows >= 100 and positive_events >= 5 and positive_days >= 10
            support_cells[f"{station}|{fold}"] = {
                "positive_rows": positive_rows,
                "positive_events": positive_events,
                "positive_kst_days": positive_days,
                "f1_delta": delta,
                "supported": supported,
            }
            if supported:
                supported_deltas.append(delta)
    if not supported_deltas:
        raise RuntimeError("no adequately supported station-fold cell")
    fp_day = normal_station_layer_day_fp(truth, candidate, anchor, metadata)
    candidate_rate = fp_day["candidate"]["false_positive_rows_per_normal_station_layer_day"]
    anchor_rate = fp_day["baseline"]["false_positive_rows_per_normal_station_layer_day"]
    fp_ratio = (
        float(candidate_rate / anchor_rate)
        if anchor_rate not in (None, 0)
        else (1.0 if candidate_rate == anchor_rate else float("inf"))
    )
    type_tokens = anomaly_type.astype("string").fillna("")
    noise = type_tokens.str.contains(r"(?:^|\+)noise(?:\+|$)").to_numpy()
    offset = type_tokens.str.contains(r"(?:^|\+)offset(?:\+|$)").to_numpy()
    drift = type_tokens.str.contains(r"(?:^|\+)drift(?:\+|$)").to_numpy()
    spike_type = type_tokens.str.contains(r"(?:^|\+)spike(?:\+|$)").to_numpy()
    flatline_type = type_tokens.str.contains(r"(?:^|\+)flatline(?:\+|$)").to_numpy()
    long_mask = np.zeros(len(truth), dtype=bool)
    for event in events:
        if len(event) >= 48 * 6:
            long_mask[event] = True
    disconnected_targets = [
        value for audit in interval_audits for value in audit["disconnected_targets"]
    ]
    added_lengths = [value for audit in interval_audits for value in audit["added_lengths"]]
    disconnected_precision = float(np.mean(disconnected_targets)) if disconnected_targets else 1.0
    seed_deltas = [
        _f1_delta(truth, outer["seed_candidates"][seed], anchor) for seed in SEGMENT_SEEDS
    ]
    gate_metrics = {
        "pooled_f1_delta": pooled_delta,
        "paired_ci90_lower": float(bootstrap["difference_ci90"][0]),
        "improving_outer_folds": sum(fold_metrics[fold]["f1_delta"] > 0.0 for fold in FOLD_ORDER),
        "improving_stations": sum(value["f1_delta"] > 0.0 for value in station_metrics.values()),
        "equal_weight_supported_station_by_fold_f1_delta": float(np.mean(supported_deltas)),
        "q3_f1_delta": float(fold_metrics["2025_q3"]["f1_delta"]),
        "station_g_f1_delta": float(station_metrics["G-ORS"]["f1_delta"]),
        "noise_recall_delta": _recall_delta(truth, candidate, anchor, noise),
        "offset_plus_drift_recall_delta": _recall_delta(
            truth,
            candidate,
            anchor,
            offset | drift,
        ),
        "at_least_48h_event_recall_delta": _recall_delta(
            truth,
            candidate,
            anchor,
            long_mask,
        ),
        "spike_predictions_exactly_preserved": bool(
            np.array_equal(candidate[spike_type], anchor[spike_type])
        ),
        "flatline_predictions_exactly_preserved": bool(
            np.array_equal(candidate[flatline_type], anchor[flatline_type])
        ),
        "false_positives_per_day_ratio": fp_ratio,
        "disconnected_interval_precision": disconnected_precision,
        "minimum_added_interval_rows": min(added_lengths) if added_lengths else 0,
        "all_hash_chronology_leakage_and_key_checks": True,
        "seed_f1_deltas": seed_deltas,
        "independent_aggregate_QA": "PENDING",
        "exact_reproduction_from_pinned_inputs": True,
    }
    gates = frozen.evaluate_decision_gates(gate_metrics)
    metrics = {
        "primary_metric": "pooled row-level binary micro F1",
        "primary_delta_direction": "candidate_minus_anchor",
        "pooled": {
            "candidate": pooled_candidate,
            "anchor": pooled_anchor,
            "f1_delta": pooled_delta,
        },
        "paired_bootstrap": bootstrap,
        "folds": fold_metrics,
        "stations": station_metrics,
        "supported_station_fold_cells": support_cells,
        "normal_station_layer_day_fp": fp_day,
        "post_freeze_truth_sha256": _array_sha(truth, dtype="<i1"),
        "gate_metrics": gate_metrics,
        "gates": gates,
    }
    return metrics, gates


def _parse_held_outer_truth_after_freeze(state: Mapping[str, Any]) -> pd.DataFrame:
    """Parse target columns only after the all-fold prediction freeze."""

    raw = bytes(state["frozen_truth_oof_bytes"])
    expected_sha256 = str(state["frozen_truth_oof_sha256"])
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise RuntimeError("held frozen truth bytes changed before post-freeze parse")
    truth = pd.read_parquet(
        io.BytesIO(raw),
        columns=[*KEY_COLUMNS, "label", "anomaly_type", "fold"],
    )
    if len(truth) != int(state["frozen_truth_oof_rows"]):
        raise RuntimeError("post-freeze frozen truth row count changed")
    if truth.duplicated([*KEY_COLUMNS, "fold"]).any():
        raise RuntimeError("post-freeze frozen truth keys are duplicated")
    return truth


def _verify_round_b_equivalence_after_freeze(
    metrics: Mapping[str, Any],
    truth: pd.DataFrame,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    observed = metrics["pooled"]["anchor"]
    for name in ("tp", "fp", "fn"):
        if int(observed[name]) != int(expected[name]):
            raise RuntimeError(f"post-freeze exact Round-B count changed: {name}")
    for name in ("f1", "precision", "recall"):
        if not np.isclose(
            float(observed[name]),
            float(expected[name]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"post-freeze exact Round-B metric changed: {name}")
    positive_rows = int(truth["label"].to_numpy(dtype=np.int8).sum())
    if positive_rows != int(expected["positive_rows"]):
        raise RuntimeError("post-freeze exact Round-B positive-row count changed")
    return {
        "status": "PASS_EXACT_ROUND_B_EQUIVALENCE_AFTER_OUTER_FREEZE",
        "truth_key_alignment": "exact_ordered_per_fold",
        "baseline_metrics": dict(observed),
        "positive_rows": positive_rows,
        "outer_prediction_was_frozen_before_truth_parse": True,
    }


def run_authorized_screen(
    state: Mapping[str, Any],
    numerical: Any,
    closure: Mapping[str, Any],
    journal: Any,
    deadline_epoch: float,
) -> dict[str, Any]:
    """Execute the single fixed 72-fit screen and return aggregate-only output."""

    _deadline(deadline_epoch, "numerical screen")
    surfaces = _fit_anchor_surfaces(
        state,
        numerical,
        closure,
        journal,
        deadline_epoch,
    )
    runs, contexts = _fit_inner_cells(surfaces, journal, deadline_epoch)
    selected, inner_evidence = _select_inner_cell(surfaces, runs, contexts)
    outer = _fit_outer(
        state,
        numerical,
        surfaces,
        contexts,
        selected,
        journal,
        deadline_epoch,
    )
    # These are the first outer-target accesses in the numerical module.  The
    # all-fold prediction digest was durably journaled by _fit_outer above.
    held_truth = _parse_held_outer_truth_after_freeze(state)
    metrics, gates = _score_outer(outer, held_truth)
    metrics["round_b_equivalence_post_freeze"] = _verify_round_b_equivalence_after_freeze(
        metrics,
        held_truth,
        state["expected_base_metrics"],
    )
    if journal.fit_reservations != 72 or journal.fits_completed != 72:
        raise RuntimeError("physical fit ledger differs from exactly 72")
    if journal.materializations != 21:
        raise RuntimeError("scientific materialization ledger differs from exactly 21")
    return {
        "schema_version": "p1_long_event_segment_proposal_rescore.aggregate_result.v2",
        "experiment_id": EXPERIMENT_ID,
        "scientific_experiment_id": SCIENTIFIC_EXPERIMENT_ID,
        "status": "COMPLETE_LOCAL_RESEARCH_SCREEN_PARENT_QA_PENDING",
        "decision": gates["decision"],
        "RESEARCH_GO": bool(gates["RESEARCH_GO"]),
        "SUBMISSION_GO_RESEARCH_ONLY": bool(gates["SUBMISSION_GO_RESEARCH_ONLY"]),
        "selected_inner_cell": {
            "cell_id": selected.cell_id,
            "context_bank_hours": list(selected.context_bank_hours),
            "decoder_mode": selected.decoder_mode,
            "threshold": selected.threshold,
        },
        "inner_evidence": inner_evidence,
        "outer_freeze": outer["freeze"],
        "metrics": metrics,
        "operation_counters": {
            "claims": 1,
            "inner_anchor_physical_fits": 9,
            "inner_segment_physical_fits": 54,
            "outer_segment_physical_fits": 9,
            "physical_fits": 72,
            "scientific_materializations": 21,
            "outer_scores": 1,
            "candidate_files": 0,
            "official_test_reads": 0,
            "sample_format_reads": 0,
            "submission_candidate_reads": 0,
            "uploads": 0,
        },
    }


__all__ = [
    "BankContext",
    "CONTEXT_BANKS",
    "DECODERS",
    "EXECUTION_CLOSURE_V3_SHA256",
    "EXPERIMENT_ID",
    "ROUND_B_SEEDS",
    "SEGMENT_SEEDS",
    "SEGMENT_PARAMETERS",
    "Surface",
    "build_bounded_segment_features",
    "fit_segment_model",
    "generate_bounded_target_free_proposals",
    "run_authorized_screen",
]
