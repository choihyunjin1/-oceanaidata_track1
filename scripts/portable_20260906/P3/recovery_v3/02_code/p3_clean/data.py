"""Minimal clean recipe functions extracted without algorithm changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LEADS = (3, 6, 9, 12, 18, 24)


STATIONS = ("G-ORS", "I-ORS", "S-ORS")


@dataclass(frozen=True)
class P3Data:
    wave: pd.DataFrame
    atmos: pd.DataFrame
    test_context: pd.DataFrame
    test_index: pd.DataFrame
    sample_submission: pd.DataFrame
    baseline: pd.DataFrame


def build_training_grid(data: P3Data) -> pd.DataFrame:
    """Align each station to the public 10-minute test-context grid."""

    pieces: list[pd.DataFrame] = []
    for station in STATIONS:
        wave = data.wave.loc[data.wave["station"].eq(station)].drop(columns="station")
        atmos = data.atmos.loc[data.atmos["station"].eq(station)].drop(columns="station")
        start = wave["time"].min()
        end = wave["time"].max() + pd.Timedelta(minutes=10)
        index = pd.date_range(start=start, end=end, freq="10min", tz="UTC")
        part = pd.DataFrame({"time": index})
        part = part.merge(wave, on="time", how="left", validate="one_to_one")
        part = part.merge(atmos, on="time", how="left", validate="one_to_one")
        part.insert(0, "station", station)
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def build_anchor_table(grid: pd.DataFrame, *, dense_spacing_minutes: int = 60) -> pd.DataFrame:
    """Build eligible train anchors and six future targets without using test labels."""

    if dense_spacing_minutes % 20:
        raise ValueError("dense spacing must be a multiple of the 20-minute wave cadence")
    records: list[pd.DataFrame] = []
    for station, group in grid.groupby("station", sort=False, observed=True):
        group = group.sort_values("time").reset_index(drop=True)
        eligible = group["hs"].ge(1.5)
        targets: dict[int, pd.Series] = {}
        for lead in LEADS:
            target = group["hs"].shift(-(lead * 6))
            targets[lead] = target
            eligible &= target.notna()
        eligible &= group["time"].ge(group["time"].min() + pd.Timedelta(hours=48))
        wave_number = np.arange(len(group)) // 2
        stride = dense_spacing_minutes // 20
        dense = eligible & ((wave_number % stride) == 0)
        idx = np.flatnonzero(dense.to_numpy())
        frame = pd.DataFrame(
            {
                "station": station,
                "anchor_time": group.loc[idx, "time"].to_numpy(),
                "grid_position": idx,
                "current_hs": group.loc[idx, "hs"].to_numpy(),
            }
        )
        for lead in LEADS:
            frame[f"target_{lead}"] = targets[lead].iloc[idx].to_numpy()
        records.append(frame)
    anchors = pd.concat(records, ignore_index=True)
    anchors.insert(0, "anchor_id", np.arange(len(anchors), dtype=np.int64))
    return anchors
