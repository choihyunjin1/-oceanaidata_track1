from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v11_causal_wavelet_scattering_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v11_causal_wavelet_scattering_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v11_wavelet_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def _frame(station: str = "I-ORS", rows: int = 220) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station": [station] * rows,
            "layer": [2] * rows,
            "_time": pd.date_range("2025-01-01", periods=rows, freq="10min", tz="UTC"),
            "temp": np.sin(np.arange(rows) / 8.0),
        }
    )


def _representation() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))["representation"]


def test_time_contract_is_ns_and_fold_cutoffs_are_distinct() -> None:
    module = _module()
    values = module.core._time_ns(
        pd.Series(pd.date_range("2024-01-01", periods=4, freq="10min", tz="UTC"))
    )
    assert values[0] > 10**18
    assert values[1] - values[0] == 600_000_000_000
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    cutoffs = []
    for item in config["parts"].values():
        audit = json.loads((ROOT / item["audit"]).read_text(encoding="utf-8"))
        cutoffs.append(pd.Timestamp(audit["adjusted_cutoff_utc"]).value)
    assert len(set(cutoffs)) == 3


def test_features_are_finite_and_future_invariant() -> None:
    module = _module()
    original = _frame()
    changed = original.copy()
    changed.loc[170:, "temp"] = 1000.0
    boundary = original.loc[120, "_time"].value
    first = module.wavelet_scattering_features(original, boundary, _representation())
    second = module.wavelet_scattering_features(changed, boundary, _representation())
    assert first.shape == (220, 14)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first[:170], second[:170], atol=0, rtol=0)


def test_group_and_gap_reset_prevent_state_carryover() -> None:
    module = _module()
    first = _frame("A")
    second = _frame("B")
    second.index = np.arange(len(first), len(first) + len(second))
    combined = pd.concat([first, second])
    boundary = first.loc[120, "_time"].value
    together = module.wavelet_scattering_features(combined, boundary, _representation())
    second.index = np.arange(len(second))
    alone = module.wavelet_scattering_features(second, boundary, _representation())
    np.testing.assert_allclose(together[len(first) :], alone, atol=0, rtol=0)
    gapped = first.copy()
    gapped.loc[160:, "_time"] += pd.Timedelta(minutes=10)
    values = module.wavelet_scattering_features(gapped, boundary, _representation())
    assert np.all(values[160] == 0)


def test_constant_path_has_zero_scattering() -> None:
    module = _module()
    frame = _frame()
    frame["temp"] = 4.25
    boundary = frame.loc[120, "_time"].value
    features = module.wavelet_scattering_features(frame, boundary, _representation())
    np.testing.assert_allclose(features, 0.0, atol=0, rtol=0)


def test_multiscale_energy_and_second_order_modulation_are_present() -> None:
    module = _module()
    frame = _frame()
    amplitude = np.ones(len(frame))
    amplitude[150:] = 4.0
    frame["temp"] = amplitude * np.sin(np.arange(len(frame)) / 2.0)
    boundary = frame.loc[120, "_time"].value
    features = module.wavelet_scattering_features(frame, boundary, _representation())
    assert features[170:, :7].mean() > features[100:120, :7].mean()
    assert features[170:, 7:12].mean() > 0


def test_add_only_and_nine_fit_fixed_gate() -> None:
    module = _module()
    scores = np.asarray([0.99, 0.98, 0.97, 0.96, 0.95])
    incumbent = np.asarray([1, 0, 0, 0, 0], dtype=np.int8)
    additions = module.base._additions(scores, incumbent, {"threshold": 0.95}, 0.4)
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    assert np.all(candidate[incumbent == 1] == 1)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["fits"] == 9
    assert len(config["model"]["seeds"]) * len(config["parts"]) == 9
    assert config["selection"]["wilson90_lcb_minimum"] == 0.55
    assert config["selection"]["outer_tuning"] == 0
    assert config["operations"]["retry_retune"] == 0
