"""Event-balanced dense-trajectory probe for P3.

This module deliberately contains only deterministic, local-data models.  The
official six leads are a subset of a 72-step, 20-minute future trajectory.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = (3, 6, 9, 12, 18, 24)
OFFICIAL_STEPS = (9, 18, 27, 36, 54, 72)
STATIONS = ("G-ORS", "I-ORS", "S-ORS")
HISTORY_INTERVALS = 144
HISTORY_POINTS = HISTORY_INTERVALS + 1
PATH_STEPS = 72


@dataclass(frozen=True)
class TrajectoryDataset:
    """All official anchors plus dense targets for trainable anchors."""

    anchors: pd.DataFrame
    history: np.ndarray
    path_target: np.ndarray
    official_target: np.ndarray
    complete_path: np.ndarray


def _station_order(values: Iterable[str]) -> list[str]:
    observed = list(dict.fromkeys(str(value) for value in values))
    return [station for station in STATIONS if station in observed] + sorted(
        set(observed).difference(STATIONS)
    )


def _high_wave_episode_ids(high_wave: np.ndarray, *, offset: int) -> tuple[np.ndarray, int]:
    high_wave = np.asarray(high_wave, dtype=bool)
    previous = np.r_[False, high_wave[:-1]]
    starts = high_wave & ~previous
    local = np.cumsum(starts, dtype=np.int64) - 1
    result = np.full(len(high_wave), -1, dtype=np.int64)
    result[high_wave] = local[high_wave] + offset
    return result, offset + int(starts.sum())


def build_trajectory_dataset(wave: pd.DataFrame) -> TrajectoryDataset:
    """Build 20-minute anchors without changing the original anchor-id contract.

    Every returned anchor has valid values at the six official leads.  The
    ``complete_path`` flag is true only when all +20m ... +24h values are valid;
    only those rows may be used for dense-trajectory training.
    """

    required = {"station", "time", "hs"}
    missing = required.difference(wave.columns)
    if missing:
        raise ValueError(f"train_wave is missing columns: {sorted(missing)}")

    records: list[pd.DataFrame] = []
    histories: list[np.ndarray] = []
    paths: list[np.ndarray] = []
    official_targets: list[np.ndarray] = []
    complete_flags: list[np.ndarray] = []
    anchor_offset = 0
    episode_offset = 0

    for station in _station_order(wave["station"].astype(str).tolist()):
        group = wave.loc[wave["station"].astype(str).eq(station)].copy()
        group["time"] = pd.to_datetime(group["time"], utc=True, errors="raise")
        group = group.sort_values("time").reset_index(drop=True)
        cadence = group["time"].diff().dt.total_seconds().div(60).dropna()
        if not cadence.eq(20).all():
            raise ValueError(f"20-minute wave cadence violated at {station}")

        hs = pd.to_numeric(group["hs"], errors="coerce").to_numpy(dtype=np.float64)
        time = group["time"].to_numpy()
        high_wave = np.isfinite(hs) & (hs >= 1.5)
        episode_id, episode_offset = _high_wave_episode_ids(
            high_wave, offset=episode_offset
        )

        candidate = np.arange(HISTORY_INTERVALS, len(group) - PATH_STEPS, dtype=np.int64)
        if not len(candidate):
            continue
        official_index = candidate[:, None] + np.asarray(OFFICIAL_STEPS, dtype=np.int64)
        official = hs[official_index]
        eligible = high_wave[candidate] & np.isfinite(official).all(axis=1)
        position = candidate[eligible]
        official = official[eligible]
        if not len(position):
            continue

        history_index = position[:, None] + np.arange(
            -HISTORY_INTERVALS, 1, dtype=np.int64
        )
        path_index = position[:, None] + np.arange(1, PATH_STEPS + 1, dtype=np.int64)
        history = hs[history_index]
        path = hs[path_index]
        complete = np.isfinite(path).all(axis=1)
        count = len(position)
        frame = pd.DataFrame(
            {
                "anchor_id": np.arange(anchor_offset, anchor_offset + count, dtype=np.int64),
                "station": station,
                "anchor_time": time[position],
                "current_hs": hs[position],
                "episode_id": episode_id[position],
                "grid_position_20m": position,
            }
        )
        records.append(frame)
        histories.append(history.astype(np.float32))
        paths.append(path.astype(np.float32))
        official_targets.append(official.astype(np.float32))
        complete_flags.append(complete)
        anchor_offset += count

    if not records:
        raise ValueError("no eligible high-wave trajectory anchors")
    anchors = pd.concat(records, ignore_index=True)
    anchors["anchor_time"] = pd.to_datetime(anchors["anchor_time"], utc=True)
    if not np.array_equal(anchors["anchor_id"].to_numpy(), np.arange(len(anchors))):
        raise ValueError("anchor ids are not contiguous")
    return TrajectoryDataset(
        anchors=anchors,
        history=np.concatenate(histories),
        path_target=np.concatenate(paths),
        official_target=np.concatenate(official_targets),
        complete_path=np.concatenate(complete_flags),
    )


def select_lattice_phase(
    anchors: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    phase_hours: int,
    gap_hours: int = 78,
) -> np.ndarray:
    """Select a frozen 78-hour lattice after a predeclared phase shift."""

    if not 0 <= phase_hours < gap_hours:
        raise ValueError("phase_hours must be in [0, gap_hours)")
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    phase_start = start_ts + pd.Timedelta(hours=phase_hours)

    chosen: list[int] = []
    for _, group in anchors.groupby("station", sort=True, observed=True):
        eligible = group.loc[
            group["anchor_time"].ge(phase_start) & group["anchor_time"].lt(end_ts)
        ].sort_values("anchor_time")
        next_time: pd.Timestamp | None = None
        for row in eligible.itertuples(index=False):
            timestamp = pd.Timestamp(row.anchor_time)
            if next_time is None or timestamp >= next_time:
                chosen.append(int(row.anchor_id))
                next_time = timestamp + pd.Timedelta(hours=gap_hours)
    return np.asarray(sorted(chosen), dtype=np.int64)


def event_balanced_weights(anchors: pd.DataFrame, anchor_ids: np.ndarray) -> np.ndarray:
    """Return the predeclared per-anchor weight 1/sqrt(n_event), mean-normalized."""

    ids = np.asarray(anchor_ids, dtype=np.int64)
    if not len(ids):
        raise ValueError("cannot weight an empty anchor set")
    lookup = anchors.set_index("anchor_id")
    event = lookup.loc[ids, "episode_id"]
    if event.lt(0).any():
        raise ValueError("training anchor without a high-wave episode")
    size = event.map(event.value_counts()).to_numpy(dtype=np.float64)
    weight = 1.0 / np.sqrt(size)
    return weight / np.mean(weight)


def _interpolate_history(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=np.float64).copy()
    if values.ndim != 2 or values.shape[1] != HISTORY_POINTS:
        raise ValueError(f"history must have shape (n, {HISTORY_POINTS})")
    x = np.arange(values.shape[1], dtype=np.float64)
    for row in range(len(values)):
        valid = np.isfinite(values[row])
        if not valid.any():
            raise ValueError("history row is entirely missing")
        if not valid.all():
            values[row] = np.interp(x, x[valid], values[row, valid])
    return values


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window < 3 or window % 2 == 0:
        raise ValueError("trend window must be an odd integer >=3")
    pad = window // 2
    padded = np.pad(values, ((0, 0), (pad, pad)), mode="edge")
    cumulative = np.pad(np.cumsum(padded, axis=1), ((0, 0), (1, 0)))
    return (cumulative[:, window:] - cumulative[:, :-window]) / float(window)


@dataclass
class ClosedFormTrajectoryRegressor:
    """Fixed weighted ridge implementation of NLinear or a DLinear trend variant."""

    variant: str
    alpha: float = 100.0
    trend_window: int = 13
    auxiliary_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.variant not in {"nlinear", "dlinear_trend"}:
            raise ValueError(f"unknown trajectory variant: {self.variant}")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 < self.auxiliary_weight <= 1:
            raise ValueError("auxiliary_weight must be in (0, 1]")
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.target_mean_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None

    @property
    def loss_weights(self) -> np.ndarray:
        weights = np.full(PATH_STEPS, self.auxiliary_weight, dtype=np.float64)
        weights[np.asarray(OFFICIAL_STEPS) - 1] = 1.0
        return weights

    def _features(self, history: np.ndarray, station: np.ndarray) -> np.ndarray:
        values = _interpolate_history(history)
        current = values[:, -1:]
        if self.variant == "nlinear":
            numeric = values[:, :-1] - current
        else:
            trend = _moving_average(values, self.trend_window)
            seasonal = values - trend
            numeric = np.concatenate([trend - current, seasonal], axis=1)
        station_text = np.asarray(station, dtype=str)
        one_hot = np.column_stack([station_text == value for value in STATIONS]).astype(float)
        unknown = ~np.isin(station_text, STATIONS)
        if unknown.any():
            raise ValueError(f"unknown station values: {sorted(set(station_text[unknown]))}")
        return np.concatenate([numeric, one_hot], axis=1)

    def fit(
        self,
        history: np.ndarray,
        station: np.ndarray,
        target_delta: np.ndarray,
        sample_weight: np.ndarray,
    ) -> ClosedFormTrajectoryRegressor:
        matrix = self._features(history, station)
        target = np.asarray(target_delta, dtype=np.float64)
        weight = np.asarray(sample_weight, dtype=np.float64)
        if target.shape != (len(matrix), PATH_STEPS):
            raise ValueError(f"target_delta must have shape (n, {PATH_STEPS})")
        if weight.shape != (len(matrix),) or not np.isfinite(weight).all() or (weight <= 0).any():
            raise ValueError("sample_weight must be finite, positive, and one-dimensional")
        if not np.isfinite(matrix).all() or not np.isfinite(target).all():
            raise ValueError("model inputs and targets must be finite")

        total = float(weight.sum())
        feature_mean = np.sum(matrix * weight[:, None], axis=0) / total
        centered = matrix - feature_mean
        variance = np.sum(np.square(centered) * weight[:, None], axis=0) / total
        feature_scale = np.sqrt(np.maximum(variance, 1e-12))
        feature_scale[feature_scale < 1e-6] = 1.0
        design = centered / feature_scale
        target_mean = np.sum(target * weight[:, None], axis=0) / total
        target_centered = target - target_mean

        gram = design.T @ (design * weight[:, None])
        cross = design.T @ (target_centered * weight[:, None])
        identity = np.eye(gram.shape[0], dtype=np.float64)
        coef = np.empty((gram.shape[0], PATH_STEPS), dtype=np.float64)
        for output_weight in np.unique(self.loss_weights):
            columns = np.flatnonzero(np.isclose(self.loss_weights, output_weight))
            system = gram + (self.alpha / float(output_weight)) * identity
            coef[:, columns] = np.linalg.solve(system, cross[:, columns])

        self.feature_mean_ = feature_mean
        self.feature_scale_ = feature_scale
        self.target_mean_ = target_mean
        self.coef_ = coef
        return self

    def predict_delta(self, history: np.ndarray, station: np.ndarray) -> np.ndarray:
        if any(
            value is None
            for value in (self.feature_mean_, self.feature_scale_, self.target_mean_, self.coef_)
        ):
            raise RuntimeError("trajectory regressor is not fitted")
        matrix = self._features(history, station)
        design = (matrix - self.feature_mean_) / self.feature_scale_
        return design @ self.coef_ + self.target_mean_

    def save(self, path: str | Path) -> None:
        if any(
            value is None
            for value in (self.feature_mean_, self.feature_scale_, self.target_mean_, self.coef_)
        ):
            raise RuntimeError("trajectory regressor is not fitted")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            variant=np.asarray(self.variant),
            alpha=np.asarray(self.alpha),
            trend_window=np.asarray(self.trend_window),
            auxiliary_weight=np.asarray(self.auxiliary_weight),
            feature_mean=self.feature_mean_,
            feature_scale=self.feature_scale_,
            target_mean=self.target_mean_,
            coef=self.coef_,
        )


def build_blind_prediction_frame(
    dataset: TrajectoryDataset,
    anchor_ids: np.ndarray,
    path_delta: np.ndarray,
    *,
    fold: str,
    phase: str,
    variant: str,
) -> pd.DataFrame:
    """Expand predictions to official leads without exposing target columns."""

    ids = np.asarray(anchor_ids, dtype=np.int64)
    delta = np.asarray(path_delta, dtype=np.float64)
    if delta.shape != (len(ids), PATH_STEPS):
        raise ValueError(f"path_delta must have shape (n, {PATH_STEPS})")
    lookup = dataset.anchors.set_index("anchor_id").loc[ids]
    rows: list[pd.DataFrame] = []
    for lead, step in zip(LEADS, OFFICIAL_STEPS, strict=True):
        rows.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "phase": phase,
                    "variant": variant,
                    "anchor_id": ids,
                    "station": lookup["station"].astype(str).to_numpy(),
                    "episode_id": lookup["episode_id"].to_numpy(dtype=np.int64),
                    "lead_h": lead,
                    "current_hs": lookup["current_hs"].to_numpy(dtype=float),
                    "prediction": np.clip(
                        lookup["current_hs"].to_numpy(dtype=float) + delta[:, step - 1],
                        0.0,
                        30.0,
                    ),
                }
            )
        )
    result = pd.concat(rows, ignore_index=True)
    forbidden = {"target_hs", "path_target", "official_target"}.intersection(result.columns)
    if forbidden:
        raise AssertionError(f"blind frame contains target columns: {sorted(forbidden)}")
    return result


def attach_official_targets(
    dataset: TrajectoryDataset, blind: pd.DataFrame
) -> pd.DataFrame:
    """Open official validation labels only after blind predictions are materialized."""

    result = blind.copy()
    ids = result["anchor_id"].to_numpy(dtype=np.int64)
    lead_to_column = {lead: column for column, lead in enumerate(LEADS)}
    columns = result["lead_h"].map(lead_to_column).to_numpy(dtype=np.int64)
    result["target_hs"] = dataset.official_target[ids, columns].astype(float)
    result["persistence"] = result["current_hs"].astype(float)
    if not np.isfinite(result[["target_hs", "prediction", "persistence"]]).all().all():
        raise ValueError("non-finite evaluated trajectory row")
    return result


def rmse(truth: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series) -> float:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(prediction_array - truth_array))))


def metric_slices(frame: pd.DataFrame, prediction_column: str = "prediction") -> dict[str, object]:
    result: dict[str, object] = {
        "rmse": rmse(frame["target_hs"], frame[prediction_column]),
        "rows": int(len(frame)),
        "cases": int(frame[["fold", "anchor_id"]].drop_duplicates().shape[0]),
        "by_lead": {},
        "by_station": {},
        "by_fold": {},
    }
    for column, key in (("lead_h", "by_lead"), ("station", "by_station"), ("fold", "by_fold")):
        result[key] = {
            str(value): rmse(group["target_hs"], group[prediction_column])
            for value, group in frame.groupby(column, sort=True, observed=True)
        }
    return result


def paired_event_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    """Bootstrap complete storm-episode blocks and return paired RMSE deltas."""

    if replicates < 1:
        raise ValueError("replicates must be positive")
    blocks = [
        group.index.to_numpy(dtype=np.int64)
        for _, group in frame.reset_index(drop=True).groupby(
            ["fold", "station", "episode_id"], sort=False, observed=True
        )
    ]
    if not blocks:
        raise ValueError("no event blocks available")
    work = frame.reset_index(drop=True)
    truth = work["target_hs"].to_numpy(dtype=float)
    candidate = work[candidate_column].to_numpy(dtype=float)
    baseline = work[baseline_column].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        indices = np.concatenate(
            [blocks[index] for index in rng.integers(0, len(blocks), size=len(blocks))]
        )
        delta[replicate] = rmse(truth[indices], candidate[indices]) - rmse(
            truth[indices], baseline[indices]
        )
    return {
        "replicates": int(replicates),
        "seed": int(seed),
        "blocks": int(len(blocks)),
        "ci90": np.quantile(delta, [0.05, 0.95]).tolist(),
        "median": float(np.median(delta)),
        "probability_improved": float(np.mean(delta < 0.0)),
    }
