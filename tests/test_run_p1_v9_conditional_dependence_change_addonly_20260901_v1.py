from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v9_conditional_dependence_change_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v9_conditional_dependence_change_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v9_dependence_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _panel(station: str = "I-ORS", rows: int = 40) -> pd.DataFrame:
    times = pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC")
    values = np.sin(np.arange(rows) / 5.0)
    records = []
    for index, timestamp in enumerate(times):
        records.append(
            {
                "station": station,
                "layer": 1,
                "_time": timestamp,
                "temp": values[index],
            }
        )
        records.append(
            {
                "station": station,
                "layer": 2,
                "_time": timestamp,
                "temp": values[index] + 0.01 * np.cos(index),
            }
        )
    return pd.DataFrame(records)


def test_time_contract_is_ns_and_fold_cutoffs_are_distinct() -> None:
    module = _module()
    times = pd.Series(pd.date_range("2024-01-01", periods=5, freq="10min", tz="UTC"))
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
    assert all(10**18 < cutoff < 2 * 10**18 for cutoff in cutoffs)


def test_dependence_features_are_finite_and_future_invariant() -> None:
    module = _module()
    original = _panel()
    changed = original.copy()
    future = changed["_time"] >= pd.Timestamp("2025-01-01 04:30:00+00:00")
    changed.loc[future & (changed["layer"] == 2), "temp"] *= -1000.0
    boundary = pd.Timestamp("2025-01-01 03:30:00+00:00").value
    first = module.conditional_dependence_features(original, boundary, 6, 0.25)
    second = module.conditional_dependence_features(changed, boundary, 6, 0.25)
    prefix = original["_time"] < pd.Timestamp("2025-01-01 04:30:00+00:00")
    assert first.shape == (80, 5)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first[prefix], second[prefix], atol=0, rtol=0)


def test_station_panel_state_resets_between_groups() -> None:
    module = _module()
    first = _panel("A", 30)
    second = _panel("B", 30)
    second.index = np.arange(len(first), len(first) + len(second))
    combined = pd.concat([first, second])
    boundary = pd.Timestamp("2025-01-01 03:30:00+00:00").value
    together = module.conditional_dependence_features(combined, boundary, 5, 0.25)
    second.index = np.arange(len(second))
    alone = module.conditional_dependence_features(second, boundary, 5, 0.25)
    np.testing.assert_allclose(together[len(first) :], alone, atol=0, rtol=0)


def test_precision_change_responds_when_marginal_scale_is_preserved() -> None:
    module = _module()
    frame = _panel(rows=60)
    late = frame["_time"] >= pd.Timestamp("2025-01-01 05:00:00+00:00")
    frame.loc[late & (frame["layer"] == 2), "temp"] *= -1.0
    boundary = pd.Timestamp("2025-01-01 03:30:00+00:00").value
    features = module.conditional_dependence_features(frame, boundary, 5, 0.25)
    early = frame["_time"].between(
        pd.Timestamp("2025-01-01 02:00:00+00:00"),
        pd.Timestamp("2025-01-01 03:30:00+00:00"),
    )
    late_mask = frame["_time"] >= pd.Timestamp("2025-01-01 08:00:00+00:00")
    assert features[late_mask, 2].mean() > features[early, 2].mean()


def test_add_only_cap_preserves_anchor_positives() -> None:
    module = _module()
    scores = np.asarray([0.99, 0.98, 0.97, 0.96, 0.95])
    incumbent = np.asarray([1, 0, 0, 0, 0], dtype=np.int8)
    additions = module._additions(scores, incumbent, {"threshold": 0.95}, 0.4)
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
