from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "validate_p2_gbm_tournament.py"
    spec = importlib.util.spec_from_file_location("validate_p2_gbm_tournament", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_independent_validator_uses_strict_numeric_reconciliation() -> None:
    module = _module()
    module._assert_close(0.75, 0.75 + 5e-13, "metric")
    with pytest.raises(ValueError, match="mismatch"):
        module._assert_close(0.75, 0.75001, "metric")


def test_independent_validator_normalizes_time_and_sort_order() -> None:
    module = _module()
    import pandas as pd

    frame = pd.DataFrame(
        {
            "time": ["2024-01-02T00:00:00+09:00", "2024-01-01T00:00:00+09:00"],
            "layer": [2, 2],
            "block": ["b", "a"],
        }
    )
    normalized = module._aligned(frame)
    assert str(normalized.loc[0, "time"].tz) == "UTC"
    assert normalized.loc[0, "block"] == "a"
