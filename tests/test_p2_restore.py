from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from p2_restore.data import KEYS
from p2_restore.features import TARGET_LAYERS, _nearest_public_baseline
from p2_restore.submission import build_submission, validate_submission


def test_nearest_public_baseline_interpolates_and_falls_back() -> None:
    temp = np.array([[10.0, 20.0, 30.0], [np.nan, 20.0, 30.0]])
    depth = np.array([[0.0, 10.0, 20.0], [0.0, 10.0, 20.0]])
    target = np.array([5.0, 5.0])
    result = _nearest_public_baseline(temp, depth, target)
    assert result[0] == pytest.approx(15.0)
    assert result[1] == pytest.approx(15.0)  # extrapolate from the two available depths


def test_target_layers_are_the_three_hidden_layers() -> None:
    assert TARGET_LAYERS == (2, 3, 4)


def test_submission_contract_and_order() -> None:
    index = pd.DataFrame(
        {
            "station": ["S-ORS", "S-ORS"],
            "layer": [2, 3],
            "time": ["2025-09-01T00:00:00+09:00", "2025-09-01T00:00:00+09:00"],
            "nominal_depth": [7.04, 9.44],
        }
    )
    frame = build_submission(index, np.array([23.4, 23.1]))
    assert list(frame.columns) == KEYS + ["temp"]
    report = validate_submission(frame, index)
    assert report["rows"] == 2


def test_submission_rejects_bad_order_and_nonfinite() -> None:
    index = pd.DataFrame(
        {
            "station": ["S-ORS"],
            "layer": [2],
            "time": ["2025-09-01T00:00:00+09:00"],
            "nominal_depth": [7.04],
        }
    )
    with pytest.raises(ValueError):
        build_submission(index, np.array([np.nan]))
    frame = build_submission(index, np.array([23.4]))
    frame = frame[["layer", "station", "time", "temp"]]
    with pytest.raises(ValueError):
        validate_submission(frame, index)
