from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v7_causal_path_crossmoment_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v7_causal_path_crossmoment_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v7_path_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _frame(rows: int = 40) -> pd.DataFrame:
    time = pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "station": ["I-ORS"] * rows,
            "layer": [2] * rows,
            "_time": time,
            "temp": np.linspace(10.0, 12.0, rows),
            "psal": np.where(np.arange(rows) % 7 == 0, np.nan, np.linspace(30.0, 31.0, rows)),
            "depth": np.where(np.arange(rows) % 11 == 0, np.nan, 7.8),
        }
    )


def test_time_contract_is_nanoseconds_and_fold_boundaries_are_distinct() -> None:
    module = _module()
    times = pd.Series(pd.date_range("2024-01-01", periods=10, freq="10min", tz="UTC"))
    values = module._time_ns(times)
    assert values.dtype == np.int64
    assert values[0] > 10**18
    assert values[1] - values[0] == 600_000_000_000
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cutoffs = []
    for item in config["parts"].values():
        audit = json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))
        cutoffs.append(pd.Timestamp(audit["adjusted_cutoff_utc"]).value)
    assert len(set(cutoffs)) == 3


def test_groupwise_ffill_and_rolling_are_future_invariant() -> None:
    module = _module()
    original = _frame()
    changed = original.copy()
    changed.loc[25:, "temp"] = 1000.0
    changed.loc[25:, "psal"] = -1000.0
    first, names = module.path_features(original, (3, 12))
    second, changed_names = module.path_features(changed, (3, 12))
    assert names == changed_names
    np.testing.assert_allclose(first[:25], second[:25], atol=0, rtol=0)


def test_signed_area_features_have_finite_expected_shape() -> None:
    module = _module()
    features, names = module.path_features(_frame(), (3, 5))
    assert features.shape == (40, 24)
    assert np.isfinite(features).all()
    assert [name for name in names if "signed_area" in name] == [
        "temp_psal_signed_area_3",
        "temp_psal_signed_area_5",
    ]


def test_add_only_cap_preserves_every_anchor_positive() -> None:
    module = _module()
    scores = np.asarray([0.99, 0.98, 0.97, 0.96, 0.95])
    incumbent = np.asarray([1, 0, 0, 0, 0], dtype=np.int8)
    chosen = {"threshold": 0.95}
    additions = module._additions(scores, incumbent, chosen, 0.4)
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    assert additions.sum() == 2
    assert not additions[0]
    assert np.all(candidate[incumbent == 1] == 1)


def test_nine_fit_budget_and_fixed_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["fits"] == 9
    assert len(config["model"]["seeds"]) * len(config["parts"]) == 9
    assert config["model"]["sweep"] == 0
    assert config["selection"]["wilson90_lcb_minimum"] == 0.55
    assert config["selection"]["outer_tuning"] == 0
    assert config["anchor"] == {"operation": "bitwise_or", "removals": 0}
