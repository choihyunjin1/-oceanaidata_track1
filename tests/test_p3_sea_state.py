from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p3_wave.sea_state import SEA_STATE_FEATURES, append_sea_state_features


def _frame() -> pd.DataFrame:
    values: dict[str, list[float]] = {}
    for suffix in ("current", "mean_3h", "mean_12h", "mean_24h"):
        values[f"tp_{suffix}"] = [6.0, 8.0]
        values[f"wspd_{suffix}"] = [12.0, 4.0]
        values[f"wind_wave_alignment_{suffix}"] = [1.0, -0.5]
    return pd.DataFrame(values)


def test_sea_state_features_are_fixed_finite_and_label_blind() -> None:
    source = _frame()
    result = append_sea_state_features(source)
    assert tuple(result.columns[-8:]) == SEA_STATE_FEATURES
    assert np.isfinite(result[list(SEA_STATE_FEATURES)].to_numpy()).all()
    assert (result.filter(like="growth_potential").to_numpy() >= 0.0).all()
    pd.testing.assert_frame_equal(source, _frame())


def test_sea_state_features_reject_targets_and_missing_sources() -> None:
    with pytest.raises(ValueError, match="target columns"):
        append_sea_state_features(_frame().assign(target_hs=2.0))
    with pytest.raises(ValueError, match="missing sea-state"):
        append_sea_state_features(_frame().drop(columns="tp_current"))
