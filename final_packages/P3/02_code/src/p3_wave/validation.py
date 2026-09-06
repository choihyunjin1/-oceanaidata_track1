"""Chronological, 78-hour-separated validation for P3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import LEADS, select_independent_validation


@dataclass(frozen=True)
class ForecastFold:
    name: str
    train_ids: np.ndarray
    validation_ids: np.ndarray
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


DEFAULT_WINDOWS = (
    ("2024_h2_storm", "2024-07-01", "2024-11-01"),
    ("winter_transition", "2024-11-01", "2025-03-01"),
    ("2025_h1", "2025-03-01", "2025-06-25"),
)


def build_forecast_folds(
    anchors: pd.DataFrame,
    *,
    windows: tuple[tuple[str, str, str], ...] = DEFAULT_WINDOWS,
    embargo_hours: int = 78,
) -> tuple[ForecastFold, ...]:
    folds: list[ForecastFold] = []
    for name, start, end in windows:
        validation_start = pd.Timestamp(start, tz="UTC")
        validation_end = pd.Timestamp(end, tz="UTC")
        train_end = validation_start - pd.Timedelta(hours=embargo_hours)
        train_ids = anchors.loc[anchors["anchor_time"].lt(train_end), "anchor_id"].to_numpy(
            dtype=np.int64
        )
        validation_ids = select_independent_validation(
            anchors, start=validation_start, end=validation_end, gap_hours=78
        )
        if len(train_ids) == 0 or len(validation_ids) == 0:
            raise ValueError(f"empty train or validation rows in {name}")
        if np.intersect1d(train_ids, validation_ids).size:
            raise ValueError(f"train/validation overlap in {name}")
        folds.append(
            ForecastFold(
                name,
                train_ids,
                validation_ids,
                validation_start,
                validation_end,
            )
        )
    return tuple(folds)


def expand_leads(
    features: pd.DataFrame,
    anchors: pd.DataFrame,
    anchor_ids: np.ndarray,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Expand anchors to the official pooled station-lead regression grain."""

    feature_lookup = features.set_index("anchor_id")
    anchor_lookup = anchors.set_index("anchor_id")
    blocks: list[pd.DataFrame] = []
    targets: list[np.ndarray] = []
    metadata: list[pd.DataFrame] = []
    for lead in LEADS:
        block = feature_lookup.loc[anchor_ids, feature_columns].reset_index(drop=True)
        station = feature_lookup.loc[anchor_ids, "station"].astype(str).reset_index(drop=True)
        current = anchor_lookup.loc[anchor_ids, "current_hs"].to_numpy(dtype=float)
        target = anchor_lookup.loc[anchor_ids, f"target_{lead}"].to_numpy(dtype=float)
        block.insert(0, "station", station)
        block.insert(1, "lead_h", lead)
        block.insert(2, "current_hs_for_residual", current)
        blocks.append(block)
        targets.append(target - current)
        metadata.append(
            pd.DataFrame(
                {
                    "anchor_id": anchor_ids,
                    "station": station,
                    "lead_h": lead,
                    "current_hs": current,
                    "target_hs": target,
                }
            )
        )
    return (
        pd.concat(blocks, ignore_index=True),
        np.concatenate(targets),
        pd.concat(metadata, ignore_index=True),
    )


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def metric_slices(metadata: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    frame = metadata.copy()
    frame["prediction"] = prediction
    result: dict[str, object] = {
        "rmse": rmse(frame["target_hs"].to_numpy(), frame["prediction"].to_numpy()),
        "n": int(len(frame)),
        "by_lead": {},
        "by_station": {},
    }
    for lead, group in frame.groupby("lead_h", observed=True):
        result["by_lead"][str(int(lead))] = rmse(
            group["target_hs"].to_numpy(), group["prediction"].to_numpy()
        )
    for station, group in frame.groupby("station", observed=True):
        result["by_station"][str(station)] = rmse(
            group["target_hs"].to_numpy(), group["prediction"].to_numpy()
        )
    return result
