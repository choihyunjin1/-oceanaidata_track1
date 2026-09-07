"""Fixed time-grid smoothing, confined to one station/layer/evaluation fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
from p2_c60_l120_equal_projection_20260907_v1 import main
from threadpoolctl import threadpool_limits


def smooth7(frame, prediction):
    values = np.asarray(prediction, dtype=float)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("finite aligned predictions required")
    keyed = frame[["station", "layer", "time"]].copy()
    keyed["time"] = pd.to_datetime(keyed.time, utc=True)
    if keyed.duplicated().any():
        raise ValueError("duplicate keys")
    keyed["position"] = np.arange(len(keyed))
    out = values.copy()
    for _, group in keyed.groupby(["station", "layer"], sort=False):
        group = group.sort_values("time")
        times = pd.DatetimeIndex(group.time)
        grid = pd.date_range(times.min(), times.max(), freq="10min")
        if not times.isin(grid).all():
            raise ValueError("timestamps not on shared ten-minute grid")
        series = pd.Series(values[group.position.to_numpy()], index=times)
        smoothed = series.reindex(grid).rolling(7, center=True, min_periods=1).mean()
        out[group.position.to_numpy()] = smoothed.reindex(times).to_numpy()
    assert np.isfinite(out).all()
    return out


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main("p2_l120_s3_smooth7_projection_20260907_v1", smooth7)
