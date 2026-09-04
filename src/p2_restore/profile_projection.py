"""Label-blind physical projection for the three reconstructed P2 layers.

The operation uses only the public layer-1/layer-5 temperatures at the same
timestamp.  When both endpoints and all three target predictions are present,
it projects the targets onto the endpoint envelope and the endpoint-implied
vertical order.  Missing public endpoints are an exact no-op.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TARGET_LAYERS = (2, 3, 4)


@dataclass(frozen=True)
class ProjectionResult:
    prediction: np.ndarray
    eligible_mask: np.ndarray
    active_mask: np.ndarray

    def diagnostics(self) -> dict[str, float | int]:
        correction = self.prediction
        return {
            "rows": int(len(correction)),
            "eligible_rows": int(self.eligible_mask.sum()),
            "eligible_share": float(self.eligible_mask.mean()),
            "active_rows": int(self.active_mask.sum()),
            "active_share": float(self.active_mask.mean()),
        }


def public_endpoint_frame(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one UTC timestamp row containing public layer 1/5 temperature."""

    required = {"time", "layer", "temp"}
    if missing := required.difference(observations.columns):
        raise ValueError(f"observations are missing endpoint columns: {sorted(missing)}")
    selected = observations.loc[observations["layer"].isin((1, 5)), ["time", "layer", "temp"]]
    keyed = selected.assign(time=pd.to_datetime(selected["time"], utc=True))
    if keyed.duplicated(["time", "layer"]).any():
        raise ValueError("public endpoint keys are not unique")
    wide = keyed.pivot(index="time", columns="layer", values="temp").rename(
        columns={1: "temp_1", 5: "temp_5"}
    )
    for column in ("temp_1", "temp_5"):
        if column not in wide:
            wide[column] = np.nan
    return wide.loc[:, ["temp_1", "temp_5"]].reset_index()


def _isotonic_three(values: np.ndarray, *, increasing: bool) -> np.ndarray:
    """Exact unit-weight PAVA for a three-value vector."""

    source = np.asarray(values, dtype=np.float64)
    if source.shape != (3,) or not np.isfinite(source).all():
        raise ValueError("three finite target predictions are required")
    sign = 1.0 if increasing else -1.0
    transformed = source * sign
    blocks: list[tuple[float, int, list[int]]] = [
        (float(transformed[index]), 1, [index]) for index in range(3)
    ]
    position = 0
    while position < len(blocks) - 1:
        left, right = blocks[position], blocks[position + 1]
        if left[0] <= right[0]:
            position += 1
            continue
        weight = left[1] + right[1]
        merged = (
            (left[0] * left[1] + right[0] * right[1]) / weight,
            weight,
            left[2] + right[2],
        )
        blocks[position : position + 2] = [merged]
        position = max(position - 1, 0)
    result = np.empty(3, dtype=np.float64)
    for value, _, indices in blocks:
        result[indices] = value
    return result * sign


def project_profiles(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    endpoints: pd.DataFrame,
) -> ProjectionResult:
    """Project aligned long-form target predictions without using target labels."""

    required = {"time", "layer"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"prediction frame is missing keys: {sorted(missing)}")
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("prediction must be a finite vector aligned to frame rows")
    if frame.duplicated(
        [column for column in ("station", "time", "layer") if column in frame]
    ).any():
        raise ValueError("prediction keys are not unique")
    endpoint_required = {"time", "temp_1", "temp_5"}
    if missing := endpoint_required.difference(endpoints.columns):
        raise ValueError(f"endpoint frame is missing columns: {sorted(missing)}")

    keyed = frame.loc[
        :, [column for column in ("station", "time", "layer") if column in frame]
    ].copy()
    keyed["time"] = pd.to_datetime(keyed["time"], utc=True)
    keyed["_row"] = np.arange(len(keyed))
    public = endpoints.loc[:, ["time", "temp_1", "temp_5"]].copy()
    public["time"] = pd.to_datetime(public["time"], utc=True)
    if public.duplicated("time").any():
        raise ValueError("endpoint timestamps are not unique")
    merged = keyed.merge(public, on="time", how="left", validate="many_to_one")

    output = values.copy()
    eligible = np.zeros(len(values), dtype=bool)
    group_columns = [column for column in ("station", "time") if column in merged]
    for _, group in merged.groupby(group_columns, sort=False, dropna=False):
        if len(group) != 3 or set(group["layer"].astype(int)) != set(TARGET_LAYERS):
            continue
        ordered = group.sort_values("layer")
        temp_1 = float(ordered["temp_1"].iloc[0])
        temp_5 = float(ordered["temp_5"].iloc[0])
        if not np.isfinite(temp_1) or not np.isfinite(temp_5):
            continue
        rows = ordered["_row"].to_numpy(dtype=int)
        projected = _isotonic_three(values[rows], increasing=temp_1 <= temp_5)
        projected = np.clip(projected, min(temp_1, temp_5), max(temp_1, temp_5))
        output[rows] = projected
        eligible[rows] = True

    active = eligible & ~np.isclose(output, values, rtol=0.0, atol=1e-12)
    return ProjectionResult(output, eligible, active)


def project_profiles_vectorized(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    endpoints: pd.DataFrame,
) -> ProjectionResult:
    """Vectorized equivalent of :func:`project_profiles` for score searches."""

    required = {"time", "layer"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"prediction frame is missing keys: {sorted(missing)}")
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError("prediction must be a finite vector aligned to frame rows")
    key_columns = [column for column in ("station", "time") if column in frame]
    unique_columns = [*key_columns, "layer"]
    if frame.duplicated(unique_columns).any():
        raise ValueError("prediction keys are not unique")

    keyed = frame.loc[:, unique_columns].copy()
    keyed["time"] = pd.to_datetime(keyed["time"], utc=True)
    keyed["_row"] = np.arange(len(keyed))
    public = endpoints.loc[:, ["time", "temp_1", "temp_5"]].copy()
    public["time"] = pd.to_datetime(public["time"], utc=True)
    if public.duplicated("time").any():
        raise ValueError("endpoint timestamps are not unique")
    merged = keyed.merge(public, on="time", how="left", validate="many_to_one")
    group_size = merged.groupby(key_columns, sort=False, dropna=False)["layer"].transform("size")
    complete = merged.loc[
        group_size.eq(3)
        & merged["layer"].isin(TARGET_LAYERS)
        & np.isfinite(merged["temp_1"])
        & np.isfinite(merged["temp_5"])
    ].sort_values([*key_columns, "layer"])
    if complete.empty:
        empty = np.zeros(len(values), dtype=bool)
        return ProjectionResult(values.copy(), empty, empty.copy())
    valid_group = complete.groupby(key_columns, sort=False, dropna=False)["layer"].transform(
        lambda column: set(column.astype(int)) == set(TARGET_LAYERS)
    )
    complete = complete.loc[valid_group]
    rows = complete["_row"].to_numpy(int).reshape(-1, 3)
    profiles = values[rows]
    temp_1 = complete["temp_1"].to_numpy(float).reshape(-1, 3)[:, 0]
    temp_5 = complete["temp_5"].to_numpy(float).reshape(-1, 3)[:, 0]
    sign = np.where(temp_1 <= temp_5, 1.0, -1.0)
    transformed = profiles * sign[:, None]
    projected = transformed.copy()
    mean_all = transformed.mean(axis=1)

    first_violation = transformed[:, 0] > transformed[:, 1]
    mean01 = (transformed[:, 0] + transformed[:, 1]) / 2.0
    pair01 = first_violation & (mean01 <= transformed[:, 2])
    projected[pair01, 0] = mean01[pair01]
    projected[pair01, 1] = mean01[pair01]
    all01 = first_violation & ~pair01
    projected[all01] = mean_all[all01, None]

    second_violation = ~first_violation & (transformed[:, 1] > transformed[:, 2])
    mean12 = (transformed[:, 1] + transformed[:, 2]) / 2.0
    pair12 = second_violation & (transformed[:, 0] <= mean12)
    projected[pair12, 1] = mean12[pair12]
    projected[pair12, 2] = mean12[pair12]
    all12 = second_violation & ~pair12
    projected[all12] = mean_all[all12, None]

    projected *= sign[:, None]
    projected = np.clip(
        projected,
        np.minimum(temp_1, temp_5)[:, None],
        np.maximum(temp_1, temp_5)[:, None],
    )
    output = values.copy()
    output[rows.ravel()] = projected.ravel()
    eligible = np.zeros(len(values), dtype=bool)
    eligible[rows.ravel()] = True
    active = eligible & ~np.isclose(output, values, rtol=0.0, atol=1e-12)
    return ProjectionResult(output, eligible, active)
