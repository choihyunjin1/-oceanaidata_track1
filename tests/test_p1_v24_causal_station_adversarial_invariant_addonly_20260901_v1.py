from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v24_causal_station_adversarial_invariant_addonly_20260901_v1.py"
SPEC = importlib.util.spec_from_file_location("p1_v24_tested", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    pieces = []
    for station, offset in (("A", 0.0), ("B", 2.0)):
        pieces.append(pd.DataFrame({"station": station, "layer": 1, "_time": pd.date_range("2025-01-01", periods=16, freq="10min", tz="UTC"), "temp": np.arange(16, dtype=float) + offset}))
    return pd.concat(pieces, ignore_index=True)


def test_causal_features_reset_groups_and_ignore_future_values(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.shared, "_set_transport_context", lambda *_: None)
    frame = _frame()
    boundary = int(pd.Timestamp(frame.loc[11, "_time"]).value)
    first = MODULE.causal_station_features(frame, boundary, {})
    changed = frame.copy()
    future = MODULE.core._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] = 1_000_000.0
    second = MODULE.causal_station_features(changed, boundary, {})
    prefix = ~future
    np.testing.assert_array_equal(first[prefix], second[prefix])
    assert first[0, 1] == 0.0 and first[16, 1] == 0.0
    assert first[0, 3] == 1.0 and first[16, 3] == 1.0


def test_gradient_reversal_negates_and_scales_gradient() -> None:
    value = torch.tensor([[2.0]], requires_grad=True)
    output = MODULE._Reverse.apply(value, 0.25)
    output.sum().backward()
    assert value.grad is not None
    assert float(value.grad.item()) == -0.25


def test_small_network_outputs_have_expected_shapes() -> None:
    network = MODULE._Network(4, 8, 3)
    anomaly, domain = network(torch.zeros((5, 4)), 0.1)
    assert anomaly.shape == (5,)
    assert domain.shape == (5, 3)
    assert torch.isfinite(anomaly).all() and torch.isfinite(domain).all()


def test_config_freezes_domain_objective_and_nine_fit_addonly_contract() -> None:
    config = json.loads(MODULE.CONFIG.read_text(encoding="utf-8"))
    assert config["objective"]["kind"] == "anomaly_BCE_plus_gradient_reversed_station_cross_entropy"
    assert config["objective"]["outer_rows_in_training"] == 0
    assert config["objective"]["outer_domain_rows_in_training"] == 0
    assert config["model"]["fits"] == 9
    assert len(config["model"]["seeds"]) * len(config["parts"]) == 9
    assert config["selection"]["outer_tuning"] == 0
    assert config["anchor"]["operation"] == "bitwise_or"
    assert config["anchor"]["removals"] == 0
