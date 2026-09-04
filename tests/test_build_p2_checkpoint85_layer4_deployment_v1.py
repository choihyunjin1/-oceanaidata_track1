from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_p2_checkpoint85_layer4_deployment_v1 import assemble_candidate


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.DataFrame(
        {
            "station": pd.Series(["S-ORS"] * 3, dtype="string"),
            "layer": [2, 3, 4],
            "time": pd.Series(["2025-09-01T00:00:00+09:00"] * 3, dtype="string"),
            "nominal_depth": [7.04, 9.44, 14.74],
        }
    )
    anchor = index[["station", "layer", "time"]].copy()
    anchor["temp"] = [20.0, 19.0, 18.0]
    return index, anchor


def test_assemble_candidate_changes_only_layer4() -> None:
    index, anchor = _frames()
    result = assemble_candidate(anchor, index, np.array([16.0]), alpha=0.5)
    assert result["temp"].tolist() == [20.0, 19.0, 17.0]


def test_assemble_candidate_rejects_invalid_alpha() -> None:
    index, anchor = _frames()
    with pytest.raises(ValueError, match="alpha"):
        assemble_candidate(anchor, index, np.array([16.0]), alpha=1.5)
