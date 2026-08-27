"""Strict aggregate-only learning-curve helpers for P2.

The helpers in this module contain no file I/O and never inspect hidden target
values.  They make the chronological prefix and meaningful-effect gates easy
to unit test independently from the expensive model fits.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from p2_restore.features import TARGET_LAYERS


def chronological_prefix_masks(
    frame: pd.DataFrame,
    eligible: np.ndarray,
    fractions: Sequence[float],
) -> tuple[dict[float, np.ndarray], dict[float, str]]:
    """Return nested earliest-timestamp prefixes of one eligible training set."""

    selected = np.asarray(eligible, dtype=bool)
    if selected.shape != (len(frame),) or not selected.any():
        raise ValueError("eligible training mask is empty or misaligned")
    parsed = tuple(float(value) for value in fractions)
    if parsed != tuple(sorted(set(parsed))) or not parsed:
        raise ValueError("prefix fractions must be unique and increasing")
    if parsed[-1] != 1.0 or any(value <= 0.0 or value > 1.0 for value in parsed):
        raise ValueError("prefix fractions must be in (0, 1] and end at one")

    times = pd.to_datetime(frame["time"], utc=True)
    eligible_times = np.sort(times.loc[selected].unique())
    masks: dict[float, np.ndarray] = {}
    boundaries: dict[float, str] = {}
    previous = np.zeros(len(frame), dtype=bool)
    for fraction in parsed:
        count = min(
            len(eligible_times),
            max(1, int(np.ceil(fraction * len(eligible_times)))),
        )
        boundary = pd.Timestamp(eligible_times[count - 1])
        current = selected & times.le(boundary).to_numpy(bool)
        if not np.all(~previous | current):
            raise AssertionError("chronological training prefixes are not nested")
        if fraction == 1.0 and not np.array_equal(current, selected):
            raise AssertionError("full chronological prefix differs from eligible mask")
        masks[fraction] = current
        boundaries[fraction] = boundary.isoformat()
        previous = current
    return masks, boundaries


def fold_equal_layer_rmse(report: Mapping[str, object], layer: int) -> float:
    """Recover one layer's equal-fold RMSE from an aggregate metric report."""

    if layer not in TARGET_LAYERS:
        raise ValueError("unexpected P2 target layer")
    by_fold = report.get("by_fold")
    if not isinstance(by_fold, Mapping) or len(by_fold) != 3:
        raise ValueError("exactly three aggregate fold reports are required")
    mse: list[float] = []
    for fold in by_fold.values():
        if not isinstance(fold, Mapping):
            raise ValueError("invalid aggregate fold report")
        by_layer = fold.get("by_layer")
        if not isinstance(by_layer, Mapping):
            raise ValueError("aggregate fold report lacks layer metrics")
        layer_report = by_layer.get(str(layer))
        if not isinstance(layer_report, Mapping):
            raise ValueError(f"aggregate fold report lacks layer {layer}")
        value = float(layer_report["rmse_c"])
        if not np.isfinite(value):
            raise ValueError("layer RMSE is not finite")
        mse.append(value**2)
    return float(np.sqrt(np.mean(mse)))


def numeric_curve_gate(
    points: Sequence[Mapping[str, object]],
    *,
    fold_deltas: Sequence[float],
    slice_deltas: Mapping[str, float],
    maximum_slice_regression_c: float = 0.0075,
    full_effect_c: float = -0.03,
) -> dict[str, bool]:
    """Evaluate the fixed lower-is-better P2 learning-curve gates."""

    keyed = {float(point["fraction"]): point for point in points}
    if set(keyed) != {0.4, 0.55, 0.7, 0.85, 1.0}:
        raise ValueError("P2 learning curve must contain the five fixed fractions")
    if len(fold_deltas) != 3:
        raise ValueError("P2 learning curve needs exactly three fold deltas")
    required_slices = {"layer_2", "layer_3", "layer_4", "2024_sep_oct"}
    if set(slice_deltas) != required_slices:
        raise ValueError("P2 critical slice keys differ from the fixed contract")

    def delta(point: Mapping[str, object]) -> float:
        return float(point["challenger"]) - float(point["incumbent"])

    def upper(point: Mapping[str, object]) -> float:
        interval = point["delta_ci90"]
        if not isinstance(interval, Sequence) or len(interval) != 2:
            raise ValueError("delta_ci90 must contain two numbers")
        return float(interval[1])

    return {
        "late_fractions_all_improve": all(delta(keyed[value]) < 0.0 for value in (0.7, 0.85, 1.0)),
        "full_fraction_ci90_excludes_zero": upper(keyed[1.0]) < 0.0,
        "another_late_fraction_ci90_excludes_zero": any(
            upper(keyed[value]) < 0.0 for value in (0.7, 0.85)
        ),
        "full_effect_meets_absolute_threshold": delta(keyed[1.0]) <= full_effect_c,
        "minimum_two_of_three_folds_improve": sum(float(value) < 0.0 for value in fold_deltas) >= 2,
        "critical_slice_regression_within_limit": all(
            float(value) <= maximum_slice_regression_c for value in slice_deltas.values()
        ),
    }
