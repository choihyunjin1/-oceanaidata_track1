"""Label-free clean-state and CAPA-style collective decoder for P1.

The module intentionally has no target-column argument.  It fits robust
station-layer seasonal states on an explicit historical prefix, builds an
adjacent-layer residual, and detects non-overlapping collective level/drift
segments with fixed likelihood penalties.  Isolated points are winsorized for
the collective scan; no row is deleted.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

KEY_COLUMNS = ("station", "year", "layer", "time")
INPUT_ONLY_COLUMNS = KEY_COLUMNS + ("temp", "psal", "depth")
ANNUAL_HARMONICS = (1, 2, 3, 4)
DIURNAL_HARMONICS = (1, 2)
ANNUAL_PERIOD_DAYS = 365.2425
HUBER_ITERATIONS = 8
HUBER_DELTA = 1.5
RIDGE = 1.0e-6
MINIMUM_GROUP_ROWS = 32
GAP_BREAK_MINUTES = 30
WINDOW_ROWS = (48, 96, 192, 384, 519)


class CapaContractError(ValueError):
    """The frozen decoder contract was violated."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def deep_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def seasonal_design(time_values: Any) -> np.ndarray:
    parsed = pd.to_datetime(time_values, errors="raise", utc=True, format="mixed")
    seconds = parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64).astype(np.float64)
    day = seconds / (86_400.0 * 1.0e9)
    columns = [np.ones(len(day), dtype=np.float64)]
    for harmonic in ANNUAL_HARMONICS:
        angle = 2.0 * np.pi * harmonic * day / ANNUAL_PERIOD_DAYS
        columns.extend((np.sin(angle), np.cos(angle)))
    for harmonic in DIURNAL_HARMONICS:
        angle = 2.0 * np.pi * harmonic * day
        columns.extend((np.sin(angle), np.cos(angle)))
    return np.column_stack(columns)


def robust_scale(values: Any) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 1.0e-6
    center = float(np.median(finite))
    return max(1.4826 * float(np.median(np.abs(finite - center))), 1.0e-6)


def fixed_huber_irls(design: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    finite = np.isfinite(target) & np.isfinite(design).all(axis=1)
    x = np.asarray(design[finite], dtype=np.float64)
    y = np.asarray(target[finite], dtype=np.float64)
    if len(y) < MINIMUM_GROUP_ROWS:
        raise CapaContractError("seasonal group has too few finite prefix rows")
    ridge = np.eye(x.shape[1], dtype=np.float64) * RIDGE
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    scale = robust_scale(y - x @ beta)
    for _ in range(HUBER_ITERATIONS):
        residual = y - x @ beta
        scale = robust_scale(residual)
        ratio = np.abs(residual) / (HUBER_DELTA * scale)
        weights = np.ones_like(ratio)
        large = ratio > 1.0
        weights[large] = 1.0 / ratio[large]
        root = np.sqrt(weights)
        weighted_x = x * root[:, None]
        beta = np.linalg.solve(weighted_x.T @ weighted_x + ridge, weighted_x.T @ (y * root))
    return beta, robust_scale(y - x @ beta)


def _group_key(station: object, layer: object) -> str:
    return f"{station}|{int(layer)}"


def _edge_key(station: object, low: int, high: int) -> str:
    return f"{station}|{low}|{high}"


@dataclass(frozen=True)
class CleanState:
    coefficients: dict[str, tuple[float, ...]]
    seasonal_scales: dict[str, float]
    edge_deltas: dict[str, float]
    edge_scales: dict[str, float]
    prefix_rows: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "p1.clean_state_capa.clean_state.v1",
            "coefficients": {key: list(value) for key, value in sorted(self.coefficients.items())},
            "seasonal_scales": dict(sorted(self.seasonal_scales.items())),
            "edge_deltas": dict(sorted(self.edge_deltas.items())),
            "edge_scales": dict(sorted(self.edge_scales.items())),
            "prefix_rows": self.prefix_rows,
        }

    @property
    def sha256(self) -> str:
        return deep_sha256(self.as_dict())


def _require_input_only(frame: pd.DataFrame) -> None:
    if tuple(frame.columns) != INPUT_ONLY_COLUMNS:
        raise CapaContractError("frame is not the exact input-only projection")
    if frame.loc[:, list(KEY_COLUMNS)].isna().any().any():
        raise CapaContractError("key columns contain nulls")
    if frame.duplicated(list(KEY_COLUMNS)).any():
        raise CapaContractError("key columns are not unique")


def fit_clean_state(prefix: pd.DataFrame) -> CleanState:
    """Fit the frozen robust seasonal and adjacent-layer state."""

    _require_input_only(prefix)
    design = seasonal_design(prefix["time"])
    coefficients: dict[str, tuple[float, ...]] = {}
    seasonal_scales: dict[str, float] = {}
    residual = np.full(len(prefix), np.nan, dtype=np.float64)
    grouped = prefix.groupby(["station", "layer"], sort=True, observed=True).indices
    for (station, layer), positions_raw in grouped.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        beta, scale = fixed_huber_irls(
            design[positions], prefix.iloc[positions]["temp"].to_numpy(dtype=np.float64)
        )
        key = _group_key(station, layer)
        coefficients[key] = tuple(float(item) for item in beta)
        seasonal_scales[key] = float(scale)
        residual[positions] = (
            prefix.iloc[positions]["temp"].to_numpy(dtype=np.float64) - design[positions] @ beta
        )

    working = prefix.loc[:, ["station", "layer", "time"]].copy()
    working["_residual"] = residual
    edge_deltas: dict[str, float] = {}
    edge_scales: dict[str, float] = {}
    for station, station_rows in working.groupby("station", sort=True, observed=True):
        pivot = station_rows.pivot(index="time", columns="layer", values="_residual")
        layers = sorted(int(item) for item in pivot.columns)
        for low, high in zip(layers[:-1], layers[1:], strict=True):
            delta = pivot[low].to_numpy(dtype=np.float64) - pivot[high].to_numpy(dtype=np.float64)
            finite = delta[np.isfinite(delta)]
            if len(finite) < MINIMUM_GROUP_ROWS:
                continue
            key = _edge_key(station, low, high)
            edge_deltas[key] = float(np.median(finite))
            edge_scales[key] = robust_scale(finite)
    return CleanState(
        coefficients=coefficients,
        seasonal_scales=seasonal_scales,
        edge_deltas=edge_deltas,
        edge_scales=edge_scales,
        prefix_rows=len(prefix),
    )


def apply_clean_state(frame: pd.DataFrame, state: CleanState) -> pd.DataFrame:
    """Project rows to seasonal and adjacent-layer residual z-scores."""

    _require_input_only(frame)
    design = seasonal_design(frame["time"])
    seasonal = np.full(len(frame), np.nan, dtype=np.float64)
    seasonal_scale = np.full(len(frame), np.nan, dtype=np.float64)
    for (station, layer), positions_raw in frame.groupby(
        ["station", "layer"], sort=True, observed=True
    ).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        key = _group_key(station, layer)
        if key not in state.coefficients:
            continue
        beta = np.asarray(state.coefficients[key], dtype=np.float64)
        seasonal[positions] = (
            frame.iloc[positions]["temp"].to_numpy(dtype=np.float64) - design[positions] @ beta
        )
        seasonal_scale[positions] = state.seasonal_scales[key]

    working = frame.loc[:, ["station", "layer", "time"]].copy()
    working["_residual"] = seasonal
    consensus = np.full(len(frame), np.nan, dtype=np.float64)
    consensus_scale = np.full(len(frame), np.nan, dtype=np.float64)
    peer_count = np.zeros(len(frame), dtype=np.int16)

    for station, positions_raw in working.groupby(
        "station", sort=True, observed=True
    ).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        station_rows = working.iloc[positions]
        pivot = station_rows.pivot(index="time", columns="layer", values="_residual").sort_index()
        layers = sorted(int(item) for item in pivot.columns)
        layer_consensus: dict[int, pd.Series] = {}
        layer_scale: dict[int, float] = {}
        layer_peers: dict[int, int] = {}
        for layer in layers:
            candidates: list[np.ndarray] = []
            scales: list[float] = []
            if layer - 1 in layers:
                edge = _edge_key(station, layer - 1, layer)
                if edge in state.edge_deltas:
                    candidates.append(
                        pivot[layer - 1].to_numpy(dtype=np.float64) - state.edge_deltas[edge]
                    )
                    scales.append(state.edge_scales[edge])
            if layer + 1 in layers:
                edge = _edge_key(station, layer, layer + 1)
                if edge in state.edge_deltas:
                    candidates.append(
                        pivot[layer + 1].to_numpy(dtype=np.float64) + state.edge_deltas[edge]
                    )
                    scales.append(state.edge_scales[edge])
            if candidates:
                stacked = np.vstack(candidates)
                with np.errstate(all="ignore"):
                    values = np.nanmedian(stacked, axis=0)
                layer_consensus[layer] = pd.Series(values, index=pivot.index)
                layer_scale[layer] = float(np.median(scales))
                layer_peers[layer] = len(candidates)
        for layer, local_raw in station_rows.groupby(
            "layer", sort=True, observed=True
        ).indices.items():
            layer_int = int(layer)
            if layer_int not in layer_consensus:
                continue
            local = np.asarray(local_raw, dtype=np.int64)
            global_positions = positions[local]
            times = station_rows.iloc[local]["time"]
            consensus[global_positions] = (
                layer_consensus[layer_int].reindex(times).to_numpy(dtype=np.float64)
            )
            consensus_scale[global_positions] = layer_scale[layer_int]
            peer_count[global_positions] = layer_peers[layer_int]

    clean_state_available = np.isfinite(seasonal) & np.isfinite(seasonal_scale)
    safe_seasonal_scale = np.maximum(seasonal_scale, 1.0e-6)
    seasonal_z = np.zeros(len(frame), dtype=np.float64)
    seasonal_z[clean_state_available] = (
        seasonal[clean_state_available] / safe_seasonal_scale[clean_state_available]
    )
    graph_available = clean_state_available & np.isfinite(consensus) & np.isfinite(consensus_scale)
    graph_z = seasonal_z.copy()
    graph_z[graph_available] = (
        seasonal[graph_available] - consensus[graph_available]
    ) / np.maximum(consensus_scale[graph_available], 1.0e-6)
    signal = seasonal_z.copy()
    signal[graph_available] = 0.5 * seasonal_z[graph_available] + 0.5 * graph_z[graph_available]
    if not np.isfinite(signal).all():
        raise CapaContractError("clean-state projection contains nonfinite decoder signal")
    return pd.DataFrame(
        {
            "seasonal_residual_z": seasonal_z,
            "graph_residual_z": graph_z,
            "clean_state_available": clean_state_available,
            "graph_available": graph_available,
            "peer_count": peer_count,
            "decoder_signal": signal,
        },
        index=frame.index,
    )


@dataclass(frozen=True)
class SegmentCandidate:
    start: int
    end: int
    score: float
    gain: float
    penalty: float
    window_rows: int
    model: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "score": self.score,
            "gain": self.gain,
            "penalty": self.penalty,
            "window_rows": self.window_rows,
            "model": self.model,
        }


def _window_gains(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    if window > n:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty
    indices = np.arange(n, dtype=np.float64)
    prefix_y = np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))
    prefix_iy = np.concatenate(([0.0], np.cumsum(indices * values, dtype=np.float64)))
    starts = np.arange(n - window + 1, dtype=np.int64)
    ends = starts + window
    sum_y = prefix_y[ends] - prefix_y[starts]
    sum_iy = prefix_iy[ends] - prefix_iy[starts]
    level_gain = np.square(sum_y) / float(window)
    mean_x = 0.5 * float(window - 1)
    centered_xy = sum_iy - starts.astype(np.float64) * sum_y - mean_x * sum_y
    sxx = float(window * (window * window - 1) / 12.0)
    trend_gain = level_gain + np.square(centered_xy) / max(sxx, 1.0)
    return level_gain, trend_gain


def _peak_indices(score: np.ndarray, distance: int) -> np.ndarray:
    if not len(score):
        return np.empty(0, dtype=np.int64)
    peaks, _properties = find_peaks(score, height=0.0, distance=distance)
    extras: list[int] = []
    if score[0] > 0.0 and (len(score) == 1 or score[0] > score[1]):
        extras.append(0)
    if len(score) > 1 and score[-1] > 0.0 and score[-1] > score[-2]:
        extras.append(len(score) - 1)
    if extras:
        peaks = np.unique(np.concatenate((peaks, np.asarray(extras, dtype=np.int64))))
    return peaks.astype(np.int64, copy=False)


def generate_candidates(values: Any) -> tuple[list[SegmentCandidate], dict[str, Any]]:
    signal = np.asarray(values, dtype=np.float64)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise CapaContractError("decoder signal must be finite one-dimensional")
    n = len(signal)
    if n < min(WINDOW_ROWS):
        return [], {"rows": n, "point_candidates": 0, "raw_collective_candidates": 0}
    point_penalty = 2.0 * math.log(float(n)) + 4.0
    point_threshold = math.sqrt(point_penalty)
    point_mask = np.abs(signal) > point_threshold
    collective_signal = np.clip(signal, -point_threshold, point_threshold)
    candidates: list[SegmentCandidate] = []
    for window in WINDOW_ROWS:
        if window > n:
            continue
        level_gain, trend_gain = _window_gains(collective_signal, window)
        best_gain = np.maximum(level_gain, trend_gain)
        penalty = 8.0 * math.log(float(n)) + 2.0 * math.log(float(window))
        score = best_gain - penalty
        for start in _peak_indices(score, max(1, window // 4)):
            model = "linear_drift" if trend_gain[start] > level_gain[start] else "mean_shift"
            candidates.append(
                SegmentCandidate(
                    start=int(start),
                    end=int(start + window),
                    score=float(score[start]),
                    gain=float(best_gain[start]),
                    penalty=float(penalty),
                    window_rows=window,
                    model=model,
                )
            )
    candidates.sort(key=lambda item: (item.end, item.start, item.window_rows, item.model))
    return candidates, {
        "rows": n,
        "point_penalty": point_penalty,
        "point_threshold": point_threshold,
        "point_candidates": int(point_mask.sum()),
        "raw_collective_candidates": len(candidates),
    }


def weighted_interval_schedule(candidates: list[SegmentCandidate]) -> list[SegmentCandidate]:
    """Choose a deterministic maximum-total-score non-overlapping subset."""

    ordered = sorted(
        candidates, key=lambda item: (item.end, item.start, item.window_rows, item.model)
    )
    if not ordered:
        return []
    ends = [item.end for item in ordered]
    predecessor = [
        bisect.bisect_right(ends, item.start, hi=index) - 1 for index, item in enumerate(ordered)
    ]
    best = np.zeros(len(ordered) + 1, dtype=np.float64)
    take = np.zeros(len(ordered), dtype=bool)
    for index, item in enumerate(ordered):
        include = item.score + best[predecessor[index] + 1]
        exclude = best[index]
        if include > exclude + 1.0e-12:
            best[index + 1] = include
            take[index] = True
        else:
            best[index + 1] = exclude
    selected: list[SegmentCandidate] = []
    index = len(ordered) - 1
    while index >= 0:
        include = ordered[index].score + best[predecessor[index] + 1]
        if take[index] and include > best[index] + 1.0e-12:
            selected.append(ordered[index])
            index = predecessor[index]
        else:
            index -= 1
    selected.reverse()
    if any(left.end > right.start for left, right in zip(selected[:-1], selected[1:], strict=True)):
        raise AssertionError("weighted interval schedule emitted overlaps")
    return selected


def decode_signal(values: Any) -> tuple[np.ndarray, list[SegmentCandidate], dict[str, Any]]:
    candidates, audit = generate_candidates(values)
    selected = weighted_interval_schedule(candidates)
    mask = np.zeros(len(np.asarray(values)), dtype=bool)
    for item in selected:
        mask[item.start : item.end] = True
    audit = {
        **audit,
        "selected_collective_segments": len(selected),
        "selected_rows": int(mask.sum()),
        "proposal_fingerprint": deep_sha256([item.as_dict() for item in selected]),
    }
    return mask, selected, audit


def _contiguous_bounds(time_ns: np.ndarray) -> list[tuple[int, int]]:
    if not len(time_ns):
        return []
    maximum_gap = GAP_BREAK_MINUTES * 60 * 1_000_000_000
    changes = np.flatnonzero(np.diff(time_ns) > maximum_gap) + 1
    edges = np.concatenate(([0], changes, [len(time_ns)]))
    return [(int(start), int(end)) for start, end in zip(edges[:-1], edges[1:], strict=True)]


def decode_frame(
    input_only_frame: pd.DataFrame, projection: pd.DataFrame
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Decode all station-layer contiguous series and return row-aligned proposals."""

    _require_input_only(input_only_frame)
    if len(projection) != len(input_only_frame) or "decoder_signal" not in projection:
        raise CapaContractError("projection does not align with input frame")
    output = np.zeros(len(input_only_frame), dtype=bool)
    selected_records: list[dict[str, Any]] = []
    point_candidates = 0
    raw_candidates = 0
    contiguous_segments = 0
    abstained_station_layer_groups = 0
    abstained_rows = 0
    for (station, layer), positions_raw in input_only_frame.groupby(
        ["station", "layer"], sort=True, observed=True
    ).indices.items():
        positions = np.asarray(positions_raw, dtype=np.int64)
        available = projection.iloc[positions]["clean_state_available"].to_numpy(dtype=bool)
        if not available.all():
            if available.any():
                raise CapaContractError(
                    "clean-state availability must be constant within a station-layer"
                )
            abstained_station_layer_groups += 1
            abstained_rows += len(positions)
            continue
        parsed = (
            pd.to_datetime(
                input_only_frame.iloc[positions]["time"], errors="raise", utc=True, format="mixed"
            )
            .to_numpy(dtype="datetime64[ns]")
            .astype(np.int64)
        )
        order = np.argsort(parsed, kind="stable")
        positions = positions[order]
        parsed = parsed[order]
        signal = projection.iloc[positions]["decoder_signal"].to_numpy(dtype=np.float64)
        for lower, upper in _contiguous_bounds(parsed):
            contiguous_segments += 1
            mask, selected, audit = decode_signal(signal[lower:upper])
            local_positions = positions[lower:upper]
            output[local_positions[mask]] = True
            point_candidates += int(audit["point_candidates"])
            raw_candidates += int(audit["raw_collective_candidates"])
            for item in selected:
                global_rows = local_positions[item.start : item.end]
                selected_records.append(
                    {
                        "station": str(station),
                        "layer": int(layer),
                        "start_row_in_frame": int(global_rows[0]),
                        "last_row_in_frame": int(global_rows[-1]),
                        "start_time": str(input_only_frame.iloc[global_rows[0]]["time"]),
                        "end_time": str(input_only_frame.iloc[global_rows[-1]]["time"]),
                        **item.as_dict(),
                    }
                )
    return (
        output,
        selected_records,
        {
            "contiguous_segments": contiguous_segments,
            "point_candidates": point_candidates,
            "raw_collective_candidates": raw_candidates,
            "selected_collective_segments": len(selected_records),
            "selected_rows": int(output.sum()),
            "abstained_station_layer_groups": abstained_station_layer_groups,
            "abstained_rows": abstained_rows,
            "proposal_fingerprint": deep_sha256(selected_records),
        },
    )


def protected_union(incumbent: Any, additions: Any) -> np.ndarray:
    baseline = np.asarray(incumbent)
    proposed = np.asarray(additions)
    if baseline.dtype != np.dtype("int8") or baseline.ndim != 1:
        raise CapaContractError("incumbent must be one-dimensional int8")
    if proposed.dtype != np.dtype("bool") or proposed.shape != baseline.shape:
        raise CapaContractError("addition mask must be aligned boolean")
    candidate = np.bitwise_or(baseline, proposed.astype(np.int8)).astype(np.int8, copy=False)
    if not np.array_equal(candidate[baseline == 1], baseline[baseline == 1]):
        raise AssertionError("protected union removed an incumbent positive")
    return candidate


def synthetic_contract_audit() -> dict[str, Any]:
    """Exercise the fixed decoder without reading any competition row."""

    signal = np.zeros(720, dtype=np.float64)
    signal[120:240] = 2.5
    signal[360:520] = np.linspace(0.0, 4.0, 160, endpoint=False)
    signal[650] = 12.0
    mask_a, selected_a, audit_a = decode_signal(signal)
    mask_b, selected_b, audit_b = decode_signal(signal)
    incumbent = np.zeros(len(signal), dtype=np.int8)
    incumbent[10:14] = 1
    candidate = protected_union(incumbent, mask_a)
    offset_recall = float(mask_a[120:240].mean())
    drift_recall = float(mask_a[400:520].mean())
    checks = {
        "offset_collective_recovered": offset_recall >= 0.8,
        "drift_collective_recovered": drift_recall >= 0.8,
        "isolated_point_not_promoted_to_collective": not bool(mask_a[650]),
        "deterministic_mask": np.array_equal(mask_a, mask_b),
        "deterministic_fingerprint": audit_a["proposal_fingerprint"]
        == audit_b["proposal_fingerprint"],
        "nonoverlap": all(
            left.end <= right.start
            for left, right in zip(selected_a[:-1], selected_a[1:], strict=True)
        ),
        "incumbent_preserved": np.array_equal(candidate[10:14], incumbent[10:14]),
        "no_target_parameter": True,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "offset_recall": offset_recall,
        "drift_recall": drift_recall,
        "selected_segments": [item.as_dict() for item in selected_a],
        "audit": audit_a,
    }


__all__ = [
    "CapaContractError",
    "CleanState",
    "INPUT_ONLY_COLUMNS",
    "KEY_COLUMNS",
    "SegmentCandidate",
    "apply_clean_state",
    "decode_frame",
    "decode_signal",
    "deep_sha256",
    "fit_clean_state",
    "protected_union",
    "synthetic_contract_audit",
    "weighted_interval_schedule",
]
