"""Deterministic, gap-safe interval proposals around anomaly-probability seeds.

The module is deliberately label-free.  It converts a residual signal and
model probabilities into bounded *proposals*, not final labels.  Offline mode
uses paired normal -> anomaly -> normal context.  Causal mode emits a proposal
at the first high-threshold crossing and never reads beyond that decision row.

``IntervalProposal.start`` and ``stop`` are half-open positions in the original
input arrays, so a proposal can be applied directly as ``mask[start:stop]``.
No proposal crosses a station, layer, segment, non-finite residual, or
non-finite probability boundary.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

PROPOSAL_SOURCES = ("mean", "variance", "slope")
ALGORITHM_VERSION = "epidemic-cpop-lite-v1"


@dataclass(frozen=True)
class ChangePointConfig:
    """Pre-registered proposal-generator settings.

    ``min_interval_rows`` and ``max_interval_rows`` affect only
    ``duration_soft_score``.  They never clip or reject an interval.
    """

    mode: Literal["offline", "causal"] = "offline"
    high_seed_threshold: float = 0.65
    low_seed_threshold: float = 0.35
    max_flank_rows: int = 72
    min_interval_rows: int = 6
    max_interval_rows: int = 600
    min_baseline_rows: int = 6
    min_return_rows: int = 3
    mean_gain_threshold: float = 0.5
    variance_gain_threshold: float = 0.25
    slope_gain_threshold: float = 0.25
    baseline_z_threshold: float = 3.0
    return_z_threshold: float = 3.0
    max_candidates_per_seed_run: int = 8
    robust_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in {"offline", "causal"}:
            raise ValueError("mode must be 'offline' or 'causal'")
        if not 0 <= self.low_seed_threshold <= self.high_seed_threshold <= 1:
            raise ValueError("seed thresholds must satisfy 0 <= low <= high <= 1")
        for name in (
            "max_flank_rows",
            "min_interval_rows",
            "max_interval_rows",
            "min_baseline_rows",
            "min_return_rows",
            "max_candidates_per_seed_run",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.min_interval_rows > self.max_interval_rows:
            raise ValueError("min_interval_rows must not exceed max_interval_rows")
        for name in (
            "mean_gain_threshold",
            "variance_gain_threshold",
            "slope_gain_threshold",
            "baseline_z_threshold",
            "return_z_threshold",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.robust_epsilon <= 0:
            raise ValueError("robust_epsilon must be positive")


@dataclass(frozen=True)
class IntervalProposal:
    """One auditable half-open interval in original input coordinates."""

    proposal_id: str
    seed_run_id: int
    start: int
    stop: int
    local_start: int
    local_stop: int
    seed_start: int
    seed_stop: int
    decision_stop: int
    context_start: int
    context_stop: int
    station: Hashable | None
    layer: Hashable | None
    segment_id: Hashable
    start_row_id: Hashable
    stop_row_id: Hashable
    start_time: Any | None
    stop_time: Any | None
    sources: tuple[str, ...]
    mean_gain: float
    variance_gain: float
    slope_gain: float
    baseline_z: float
    return_z: float | None
    duration_rows: int
    duration_soft_score: float
    total_score: float
    mode: Literal["offline", "causal"]
    robust_center: float
    robust_scale: float
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_return(self) -> bool:
        return bool(self.return_z is not None and np.isfinite(self.return_z))

    @property
    def source(self) -> str:
        """Canonical ``+``-joined expert-source string for diagnostics."""

        return "+".join(self.sources)

    @property
    def score(self) -> float:
        """Compatibility name for the proposal ranking score."""

        return self.total_score

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChangePointResult:
    """Proposal collection plus generator-level provenance."""

    proposals: tuple[IntervalProposal, ...]
    seed_runs: int
    skipped_seed_runs: Mapping[str, int]
    mode: Literal["offline", "causal"]
    config: ChangePointConfig
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposals": [proposal.to_dict() for proposal in self.proposals],
            "seed_runs": self.seed_runs,
            "skipped_seed_runs": dict(self.skipped_seed_runs),
            "mode": self.mode,
            "config": asdict(self.config),
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class _Prefix:
    total: np.ndarray
    square: np.ndarray

    @classmethod
    def from_values(cls, values: np.ndarray) -> _Prefix:
        return cls(
            total=np.concatenate(([0.0], np.cumsum(values, dtype=np.float64))),
            square=np.concatenate(([0.0], np.cumsum(values * values, dtype=np.float64))),
        )

    def stats(self, start: int, stop: int) -> tuple[int, float, float, float]:
        count = stop - start
        if count <= 0:
            return 0, 0.0, 0.0, 0.0
        total = float(self.total[stop] - self.total[start])
        square = float(self.square[stop] - self.square[start])
        mean = total / count
        variance = max(0.0, square / count - mean * mean)
        sse = max(0.0, square - total * total / count)
        return count, mean, variance, sse


def _as_vector(values: Sequence[Any] | np.ndarray, *, length: int | None, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} must have length {length}")
    return array


def _metadata_vector(
    values: Sequence[Any] | np.ndarray | None,
    *,
    length: int,
    default: Any,
    name: str,
) -> np.ndarray:
    if values is None:
        result = np.empty(length, dtype=object)
        result[:] = default
        return result
    return _as_vector(values, length=length, name=name).astype(object, copy=False)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = value != value
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _equal(left: Any, right: Any) -> bool:
    if _missing(left) or _missing(right):
        return False
    try:
        value = left == right
    except (TypeError, ValueError):
        return False
    return bool(value) if isinstance(value, (bool, np.bool_)) else False


def _optional_equal(left: Any, right: Any) -> bool:
    """Equality for optional group metadata; two omitted values are equal."""

    if _missing(left) and _missing(right):
        return True
    return _equal(left, right)


def _physical_runs(
    residual: np.ndarray,
    probability: np.ndarray,
    station: np.ndarray,
    layer: np.ndarray,
    segment: np.ndarray,
) -> list[tuple[int, int]]:
    """Return contiguous finite runs without ever coalescing repeated IDs."""

    runs: list[tuple[int, int]] = []
    start: int | None = None
    for position in range(len(residual) + 1):
        valid = (
            position < len(residual)
            and np.isfinite(residual[position])
            and np.isfinite(probability[position])
            and not _missing(segment[position])
        )
        continues = False
        if valid and start is not None:
            continues = (
                _optional_equal(station[position], station[position - 1])
                and _optional_equal(layer[position], layer[position - 1])
                and _equal(segment[position], segment[position - 1])
            )
        if start is not None and not continues:
            runs.append((start, position))
            start = None
        if valid and start is None:
            start = position
    return runs


def _robust_location_scale(values: np.ndarray, epsilon: float) -> tuple[float, float]:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale <= epsilon:
        q25, q75 = np.quantile(values, [0.25, 0.75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= epsilon:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale <= epsilon:
        scale = 1.0
    return center, scale


def _duration_soft_score(duration: int, config: ChangePointConfig) -> float:
    if duration < config.min_interval_rows:
        return float(duration / config.min_interval_rows)
    if duration <= config.max_interval_rows:
        return 1.0
    return float(config.max_interval_rows / duration)


def _boundary_pool(
    prefix: _Prefix,
    *,
    candidates: range,
    anchor_start: int,
    anchor_stop: int,
    side: Literal["start", "stop"],
    run_start: int,
    run_stop: int,
    flank: int,
) -> tuple[int, ...]:
    """Select a small deterministic boundary pool using prefix mean contrast."""

    scored: list[tuple[float, int]] = []
    for boundary in candidates:
        if side == "start":
            left = (max(run_start, boundary - flank), boundary)
            right = (boundary, anchor_stop)
        else:
            left = (anchor_start, boundary)
            right = (boundary, min(run_stop, boundary + flank))
        if left[1] <= left[0] or right[1] <= right[0]:
            continue
        _, left_mean, _, _ = prefix.stats(*left)
        _, right_mean, _, _ = prefix.stats(*right)
        scored.append((abs(right_mean - left_mean), boundary))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = [boundary for _, boundary in scored[:4]]
    if side == "start":
        selected.extend((candidates.start, candidates.stop - 1, anchor_start))
    else:
        selected.extend((candidates.start, candidates.stop - 1, anchor_stop))
    return tuple(sorted(set(selected)))


def _combined_stats(
    prefix: _Prefix,
    ranges: Sequence[tuple[int, int]],
) -> tuple[int, float, float, float]:
    count = 0
    total = 0.0
    square = 0.0
    for start, stop in ranges:
        local_count = stop - start
        if local_count <= 0:
            continue
        count += local_count
        total += float(prefix.total[stop] - prefix.total[start])
        square += float(prefix.square[stop] - prefix.square[start])
    if count == 0:
        return 0, 0.0, 0.0, 0.0
    mean = total / count
    variance = max(0.0, square / count - mean * mean)
    sse = max(0.0, square - total * total / count)
    return count, mean, variance, sse


def _continuous_slope_gain(values: np.ndarray, start: int, stop: int) -> float:
    """Per-row SSE gain of a continuous two-hinge line over one global line."""

    count = len(values)
    if count < 5:
        return 0.0
    x = np.arange(count, dtype=np.float64)
    local_start = float(start)
    local_stop = float(stop)
    null_design = np.column_stack((np.ones(count), x))
    hinge_design = np.column_stack(
        (
            np.ones(count),
            x,
            np.maximum(0.0, x - local_start),
            np.maximum(0.0, x - local_stop),
        )
    )
    try:
        null_fit = null_design @ np.linalg.lstsq(null_design, values, rcond=None)[0]
        hinge_fit = hinge_design @ np.linalg.lstsq(hinge_design, values, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 0.0
    null_sse = float(np.square(values - null_fit).sum())
    hinge_sse = float(np.square(values - hinge_fit).sum())
    return max(0.0, (null_sse - hinge_sse) / count)


def _score_candidate(
    z: np.ndarray,
    prefix: _Prefix,
    *,
    run_start: int,
    run_stop: int,
    start: int,
    stop: int,
    causal: bool,
    config: ChangePointConfig,
) -> tuple[float, float, float, float, float | None, float] | None:
    left_start = max(run_start, start - config.max_flank_rows)
    left = (left_start, start)
    if left[1] - left[0] < config.min_baseline_rows:
        return None
    right = (stop, min(run_stop, stop + config.max_flank_rows))
    if not causal and right[1] - right[0] < config.min_return_rows:
        return None

    interval = (start, stop)
    interval_count, interval_mean, interval_var, interval_sse = prefix.stats(*interval)
    if interval_count <= 0:
        return None
    normal_ranges = [left] if causal else [left, right]
    normal_count, normal_mean, normal_var, normal_sse = _combined_stats(prefix, normal_ranges)
    pooled_count, _, pooled_var, pooled_sse = _combined_stats(
        prefix,
        [*normal_ranges, interval],
    )
    mean_gain = max(0.0, (pooled_sse - normal_sse - interval_sse) / pooled_count)
    eps = config.robust_epsilon
    pooled_cost = pooled_count * np.log(max(pooled_var, eps))
    split_cost = normal_count * np.log(max(normal_var, eps)) + interval_count * np.log(
        max(interval_var, eps)
    )
    variance_gain = max(0.0, float((pooled_cost - split_cost) / pooled_count))

    context_start = left[0]
    context_stop = stop if causal else right[1]
    local_values = z[context_start:context_stop]
    slope_gain = _continuous_slope_gain(
        local_values,
        start - context_start,
        stop - context_start,
    )
    baseline_z = abs(float(normal_mean if causal else prefix.stats(*left)[1]))
    return_z: float | None = None
    if not causal:
        _, left_mean, _, _ = prefix.stats(*left)
        _, right_mean, _, _ = prefix.stats(*right)
        return_z = abs(float(right_mean - left_mean))
    if baseline_z > config.baseline_z_threshold:
        return None
    if return_z is not None and return_z > config.return_z_threshold:
        return None
    duration_score = _duration_soft_score(stop - start, config)
    total_score = mean_gain + variance_gain + slope_gain + 0.1 * duration_score
    # Retain the component values even when only one expert is activated.
    _ = interval_mean
    return mean_gain, variance_gain, slope_gain, baseline_z, return_z, total_score


def _source_tuple(
    mean_gain: float,
    variance_gain: float,
    slope_gain: float,
    config: ChangePointConfig,
) -> tuple[str, ...]:
    source: list[str] = []
    if mean_gain >= config.mean_gain_threshold:
        source.append("mean")
    if variance_gain >= config.variance_gain_threshold:
        source.append("variance")
    if slope_gain >= config.slope_gain_threshold:
        source.append("slope")
    return tuple(source)


def _offline_seed_runs(probability: np.ndarray, start: int, stop: int, config: ChangePointConfig):
    position = start
    while position < stop:
        if probability[position] < config.low_seed_threshold:
            position += 1
            continue
        seed_start = position
        while position < stop and probability[position] >= config.low_seed_threshold:
            position += 1
        seed_stop = position
        if np.any(probability[seed_start:seed_stop] >= config.high_seed_threshold):
            yield seed_start, seed_stop, seed_stop


def _causal_seed_runs(probability: np.ndarray, start: int, stop: int, config: ChangePointConfig):
    candidate_start: int | None = None
    seen_high = False
    for position in range(start, stop):
        if probability[position] < config.low_seed_threshold:
            candidate_start = None
            seen_high = False
            continue
        if candidate_start is None:
            candidate_start = position
        high = probability[position] >= config.high_seed_threshold
        if not high or seen_high:
            continue
        seen_high = True
        # Decision is made when the first high row arrives; no future low-run
        # tail or future normalisation value is inspected.
        yield candidate_start, position + 1, position + 1


def propose_change_intervals(
    residuals: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    segment_ids: Sequence[Hashable] | np.ndarray,
    *,
    station: Sequence[Hashable] | np.ndarray | None = None,
    layer: Sequence[Hashable] | np.ndarray | None = None,
    row_ids: Sequence[Hashable] | np.ndarray | None = None,
    times: Sequence[Any] | np.ndarray | None = None,
    config: ChangePointConfig | None = None,
) -> ChangePointResult:
    """Propose deterministic intervals around model-probability seed runs.

    The function has intentionally no label or anomaly-type argument.  Robust
    location/scale is fit separately inside each physical segment.  In causal
    mode it is re-fit on the prefix ending at ``decision_stop``.
    """

    cfg = config or ChangePointConfig()
    residual = _as_vector(residuals, length=None, name="residuals").astype(float, copy=False)
    probability = _as_vector(
        probabilities,
        length=len(residual),
        name="probabilities",
    ).astype(float, copy=False)
    if np.isfinite(probability).any():
        finite_probability = probability[np.isfinite(probability)]
        if ((finite_probability < 0) | (finite_probability > 1)).any():
            raise ValueError("finite probabilities must lie in [0, 1]")
    segments = _metadata_vector(
        segment_ids,
        length=len(residual),
        default=None,
        name="segment_ids",
    )
    stations = _metadata_vector(station, length=len(residual), default=None, name="station")
    layers = _metadata_vector(layer, length=len(residual), default=None, name="layer")
    rows = _metadata_vector(row_ids, length=len(residual), default=None, name="row_ids")
    if row_ids is None:
        rows = np.arange(len(residual), dtype=np.int64).astype(object)
    timestamp = _metadata_vector(times, length=len(residual), default=None, name="times")

    proposals: list[IntervalProposal] = []
    skipped = {"short_context": 0, "baseline_or_return": 0, "below_gain": 0}
    seed_run_id = 0
    physical_runs = _physical_runs(residual, probability, stations, layers, segments)
    for run_start, run_stop in physical_runs:
        seed_iterator = (
            _offline_seed_runs(probability, run_start, run_stop, cfg)
            if cfg.mode == "offline"
            else _causal_seed_runs(probability, run_start, run_stop, cfg)
        )
        for seed_start, seed_stop, decision_stop in seed_iterator:
            current_seed_id = seed_run_id
            seed_run_id += 1
            scale_stop = run_stop if cfg.mode == "offline" else decision_stop
            prefix_values = residual[run_start:scale_stop]
            if len(prefix_values) < cfg.min_baseline_rows + 1:
                skipped["short_context"] += 1
                continue
            center, scale = _robust_location_scale(prefix_values, cfg.robust_epsilon)
            # Prefix arrays use physical-run-local coordinates.
            z = (prefix_values - center) / scale
            prefix = _Prefix.from_values(z)
            local_seed_start = seed_start - run_start
            local_seed_stop = seed_stop - run_start
            local_run_stop = scale_stop - run_start
            start_min = max(
                cfg.min_baseline_rows,
                local_seed_start - cfg.max_flank_rows,
            )
            start_range = range(start_min, local_seed_start + 1)
            if cfg.mode == "offline":
                stop_max = min(
                    local_run_stop - cfg.min_return_rows,
                    local_seed_stop + cfg.max_flank_rows,
                )
            else:
                stop_max = local_seed_stop
            stop_range = range(local_seed_stop, stop_max + 1)
            if not start_range or not stop_range:
                skipped["short_context"] += 1
                continue
            starts = _boundary_pool(
                prefix,
                candidates=start_range,
                anchor_start=local_seed_start,
                anchor_stop=local_seed_stop,
                side="start",
                run_start=0,
                run_stop=local_run_stop,
                flank=cfg.max_flank_rows,
            )
            if cfg.mode == "causal":
                stops = (local_seed_stop,)
            else:
                stops = _boundary_pool(
                    prefix,
                    candidates=stop_range,
                    anchor_start=local_seed_start,
                    anchor_stop=local_seed_stop,
                    side="stop",
                    run_start=0,
                    run_stop=local_run_stop,
                    flank=cfg.max_flank_rows,
                )

            candidates: list[IntervalProposal] = []
            rejected_context = 0
            rejected_gain = 0
            for local_start in starts:
                for local_stop in stops:
                    if local_start >= local_stop:
                        continue
                    scored = _score_candidate(
                        z,
                        prefix,
                        run_start=0,
                        run_stop=local_run_stop,
                        start=local_start,
                        stop=local_stop,
                        causal=cfg.mode == "causal",
                        config=cfg,
                    )
                    if scored is None:
                        rejected_context += 1
                        continue
                    mean_gain, variance_gain, slope_gain, baseline_z, return_z, total = scored
                    sources = _source_tuple(mean_gain, variance_gain, slope_gain, cfg)
                    if not sources:
                        rejected_gain += 1
                        continue
                    global_start = run_start + local_start
                    global_stop = run_start + local_stop
                    context_start = max(run_start, global_start - cfg.max_flank_rows)
                    context_stop = (
                        decision_stop
                        if cfg.mode == "causal"
                        else min(run_stop, global_stop + cfg.max_flank_rows)
                    )
                    duration = global_stop - global_start
                    candidates.append(
                        IntervalProposal(
                            proposal_id="",  # assigned after deterministic ranking
                            seed_run_id=current_seed_id,
                            start=global_start,
                            stop=global_stop,
                            local_start=local_start,
                            local_stop=local_stop,
                            seed_start=seed_start,
                            seed_stop=seed_stop,
                            decision_stop=decision_stop,
                            context_start=context_start,
                            context_stop=context_stop,
                            station=stations[run_start],
                            layer=layers[run_start],
                            segment_id=segments[run_start],
                            start_row_id=rows[global_start],
                            stop_row_id=rows[global_stop - 1],
                            start_time=timestamp[global_start],
                            stop_time=timestamp[global_stop - 1],
                            sources=sources,
                            mean_gain=float(mean_gain),
                            variance_gain=float(variance_gain),
                            slope_gain=float(slope_gain),
                            baseline_z=float(baseline_z),
                            return_z=None if return_z is None else float(return_z),
                            duration_rows=duration,
                            duration_soft_score=_duration_soft_score(duration, cfg),
                            total_score=float(total),
                            mode=cfg.mode,
                            robust_center=center,
                            robust_scale=scale,
                            provenance={
                                "algorithm": ALGORITHM_VERSION,
                                "coordinate": "input_positional_half_open",
                                "labels_used": False,
                                "normalizer_scope": (
                                    "segment" if cfg.mode == "offline" else "decision_prefix"
                                ),
                                "duration_is_soft": True,
                            },
                        )
                    )
            candidates.sort(
                key=lambda proposal: (
                    -proposal.total_score,
                    proposal.start,
                    proposal.stop,
                    proposal.sources,
                )
            )
            if not candidates:
                if rejected_context:
                    skipped["baseline_or_return"] += 1
                elif rejected_gain:
                    skipped["below_gain"] += 1
                else:
                    skipped["short_context"] += 1
                continue
            for rank, proposal in enumerate(candidates[: cfg.max_candidates_per_seed_run]):
                proposals.append(
                    IntervalProposal(
                        **{
                            **proposal.to_dict(),
                            "proposal_id": f"seed-{current_seed_id:06d}-rank-{rank:02d}",
                        }
                    )
                )

    return ChangePointResult(
        proposals=tuple(proposals),
        seed_runs=seed_run_id,
        skipped_seed_runs=skipped,
        mode=cfg.mode,
        config=cfg,
        provenance={
            "algorithm": ALGORITHM_VERSION,
            "coordinate": "input_positional_half_open",
            "labels_used": False,
            "normalization": "segment_median_mad_with_iqr_std_fallback",
            "source_order": PROPOSAL_SOURCES,
            "station_layer_required_for_global_segment_ids": station is not None
            or layer is not None,
        },
    )


def _proposal_tuple(
    proposals: ChangePointResult | Sequence[IntervalProposal],
) -> tuple[IntervalProposal, ...]:
    return proposals.proposals if isinstance(proposals, ChangePointResult) else tuple(proposals)


def filter_proposals(
    proposals: ChangePointResult | Sequence[IntervalProposal],
    *,
    sources: Iterable[str] | None = None,
    min_total_score: float | None = None,
    require_return: bool | None = None,
) -> tuple[IntervalProposal, ...]:
    """Filter proposal diagnostics before any interval union is performed."""

    required_sources = set(sources or ())
    unknown = required_sources.difference(PROPOSAL_SOURCES)
    if unknown:
        raise ValueError(f"unknown proposal sources: {sorted(unknown)}")
    threshold = -np.inf if min_total_score is None else float(min_total_score)
    selected = []
    for proposal in _proposal_tuple(proposals):
        if required_sources and required_sources.isdisjoint(proposal.sources):
            continue
        if proposal.total_score < threshold:
            continue
        if require_return is not None and proposal.has_return is not require_return:
            continue
        selected.append(proposal)
    return tuple(selected)


def best_per_seed(
    proposals: ChangePointResult | Sequence[IntervalProposal],
    *,
    top_k: int = 1,
    sources: Iterable[str] | None = None,
    min_total_score: float | None = None,
    require_return: bool | None = None,
) -> tuple[IntervalProposal, ...]:
    """Select at most ``top_k`` ranked proposals for each seed run."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    filtered = filter_proposals(
        proposals,
        sources=sources,
        min_total_score=min_total_score,
        require_return=require_return,
    )
    grouped: dict[int, list[IntervalProposal]] = {}
    for proposal in filtered:
        grouped.setdefault(proposal.seed_run_id, []).append(proposal)
    result: list[IntervalProposal] = []
    for seed_run_id in sorted(grouped):
        ranked = sorted(
            grouped[seed_run_id],
            key=lambda proposal: (-proposal.total_score, proposal.start, proposal.stop),
        )
        result.extend(ranked[:top_k])
    return tuple(result)


def best_per_seed_run(*args: Any, **kwargs: Any) -> tuple[IntervalProposal, ...]:
    """Compatibility alias for :func:`best_per_seed`."""

    return best_per_seed(*args, **kwargs)


def proposals_to_mask(
    proposals: ChangePointResult | Sequence[IntervalProposal],
    length: int,
    *,
    top_k_per_seed: int | None = None,
    sources: Iterable[str] | None = None,
    min_total_score: float | None = None,
    require_return: bool | None = None,
) -> np.ndarray:
    """Union selected half-open proposals into a boolean positional mask."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if top_k_per_seed is None:
        selected = filter_proposals(
            proposals,
            sources=sources,
            min_total_score=min_total_score,
            require_return=require_return,
        )
    else:
        selected = best_per_seed(
            proposals,
            top_k=top_k_per_seed,
            sources=sources,
            min_total_score=min_total_score,
            require_return=require_return,
        )
    mask = np.zeros(length, dtype=bool)
    for proposal in selected:
        if not 0 <= proposal.start < proposal.stop <= length:
            raise ValueError("proposal coordinates lie outside mask length")
        mask[proposal.start : proposal.stop] = True
    return mask


__all__ = [
    "ALGORITHM_VERSION",
    "PROPOSAL_SOURCES",
    "ChangePointConfig",
    "ChangePointResult",
    "IntervalProposal",
    "best_per_seed",
    "best_per_seed_run",
    "filter_proposals",
    "proposals_to_mask",
    "propose_change_intervals",
]
