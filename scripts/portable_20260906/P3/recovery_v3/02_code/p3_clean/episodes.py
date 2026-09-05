"""Minimal clean recipe functions extracted without algorithm changes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def assign_storm_episodes_from_wave(
    anchors: pd.DataFrame,
    wave: pd.DataFrame,
) -> pd.DataFrame:
    """Attach episodes defined on the complete observed wave stream.

    Episode boundaries must not depend on whether a future six-lead target happens to be
    available. Defining them on eligible anchors alone could split one physical storm when
    target validity has a hole, so this path uses only current/past raw ``hs`` observations.
    """

    required_anchor = {"anchor_id", "station", "anchor_time", "current_hs"}
    required_wave = {"station", "time", "hs"}
    if not required_anchor.issubset(anchors.columns):
        raise ValueError(
            f"anchor metadata is missing: {sorted(required_anchor - set(anchors.columns))}"
        )
    if not required_wave.issubset(wave.columns):
        raise ValueError(f"wave data is missing: {sorted(required_wave - set(wave.columns))}")

    result = anchors.copy()
    result["anchor_time"] = pd.to_datetime(result["anchor_time"], utc=True)
    if result["current_hs"].lt(1.5).any():
        raise ValueError("episode table contains an ineligible anchor below 1.5m")

    source = wave.loc[:, ["station", "time", "hs"]].copy()
    source["time"] = pd.to_datetime(source["time"], utc=True)
    if source.duplicated(["station", "time"]).any():
        raise ValueError("wave data contains duplicate station/time keys")

    high_rows: list[pd.DataFrame] = []
    next_episode = 0
    for _, group in source.groupby("station", sort=True, observed=True):
        ordered = group.sort_values("time").copy()
        high = ordered["hs"].ge(1.5) & ordered["hs"].notna()
        contiguous = ordered["time"].diff().eq(pd.Timedelta(minutes=20))
        previous_high = high.shift(fill_value=False)
        start = high & (~previous_high | ~contiguous)
        local_episode = start.cumsum().astype(np.int64) - 1 + next_episode
        selected = ordered.loc[high, ["station", "time", "hs"]].copy()
        selected["episode_id"] = local_episode.loc[high].to_numpy(dtype=np.int64)
        high_rows.append(selected)
        if high.any():
            next_episode = int(local_episode.loc[high].max()) + 1

    mapping = pd.concat(high_rows, ignore_index=True).rename(
        columns={"time": "anchor_time", "hs": "raw_current_hs"}
    )
    result = result.merge(
        mapping,
        on=["station", "anchor_time"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    if result["episode_id"].isna().any():
        raise ValueError("eligible anchor could not be mapped to a raw-wave storm episode")
    if not np.allclose(
        result["current_hs"].to_numpy(dtype=np.float64),
        result["raw_current_hs"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("anchor cache current_hs differs from immutable raw-wave hs")
    result["episode_id"] = result["episode_id"].astype(np.int64)
    return result.drop(columns="raw_current_hs")
