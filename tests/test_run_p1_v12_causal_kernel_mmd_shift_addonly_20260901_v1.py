from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v12_causal_kernel_mmd_shift_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v12_causal_kernel_mmd_shift_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v12_mmd_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _frame(station: str = "I-ORS", rows: int = 300) -> pd.DataFrame:
    return pd.DataFrame({"station": [station] * rows, "layer": [2] * rows, "_time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"), "temp": np.sin(np.arange(rows) / 8.0)})


def _representation() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))["representation"]


def test_time_contract_is_ns_and_cutoffs_are_distinct() -> None:
    module = _module()
    values = module.core._time_ns(pd.Series(pd.date_range("2024-01-01", periods=3, freq="10min", tz="UTC")))
    assert values[0] > 10**18 and values[1] - values[0] == 600_000_000_000
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cutoffs = [pd.Timestamp(json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))["adjusted_cutoff_utc"]).value for item in config["parts"].values()]
    assert len(set(cutoffs)) == 3


def test_features_are_finite_and_future_invariant() -> None:
    module = _module()
    original = _frame()
    changed = original.copy()
    changed.loc[250:, "temp"] = 1000.0
    boundary = original.loc[210, "_time"].value
    first = module.kernel_mmd_features(original, boundary, _representation())
    second = module.kernel_mmd_features(changed, boundary, _representation())
    assert first.shape == (300, 9) and np.isfinite(first).all()
    np.testing.assert_allclose(first[:250], second[:250], atol=0, rtol=0)


def test_group_and_gap_reset_prevent_state_carryover() -> None:
    module = _module()
    first = _frame("A")
    second = _frame("B")
    second.index = np.arange(len(first), len(first) + len(second))
    boundary = first.loc[210, "_time"].value
    together = module.kernel_mmd_features(pd.concat([first, second]), boundary, _representation())
    second.index = np.arange(len(second))
    alone = module.kernel_mmd_features(second, boundary, _representation())
    np.testing.assert_allclose(together[len(first) :], alone, atol=0, rtol=0)
    gapped = first.copy()
    gapped.loc[250:, "_time"] += pd.Timedelta(minutes=10)
    reset = module.kernel_mmd_features(gapped, boundary, _representation())
    assert np.isfinite(reset[250]).all()


def test_kernel_embedding_detects_level_and_shape_shift() -> None:
    module = _module()
    boundary_frame = _frame()
    boundary = boundary_frame.loc[210, "_time"].value
    level = boundary_frame.copy()
    level.loc[240:, "temp"] += 4.0
    shape = boundary_frame.copy()
    shape.loc[240:, "temp"] = 2.5 * np.sign(np.sin(np.arange(240, 300)))
    base = module.kernel_mmd_features(boundary_frame, boundary, _representation())
    shifted = module.kernel_mmd_features(level, boundary, _representation())
    reshaped = module.kernel_mmd_features(shape, boundary, _representation())
    assert shifted[270:].mean() > base[270:].mean()
    assert reshaped[270:].mean() > base[270:].mean()


def test_add_only_and_fixed_nine_fit_gate() -> None:
    module = _module()
    incumbent = np.asarray([1, 0, 0, 0, 0], dtype=np.int8)
    additions = module.base._additions(np.asarray([0.99, 0.98, 0.97, 0.96, 0.95]), incumbent, {"threshold": 0.95}, 0.4)
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    assert np.all(candidate[incumbent == 1] == 1)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert len(config["model"]["seeds"]) * len(config["parts"]) == config["model"]["fits"] == 9
    assert config["selection"]["wilson90_lcb_minimum"] == 0.55
    assert config["selection"]["outer_tuning"] == config["operations"]["retry_retune"] == 0
