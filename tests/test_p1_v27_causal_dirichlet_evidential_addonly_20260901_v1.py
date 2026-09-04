from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v27_causal_dirichlet_evidential_addonly_20260901_v1.py"
SPEC = importlib.util.spec_from_file_location("p1_v27_tested", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    pieces = []
    for station, offset in (("A", 0.0), ("B", 2.0)):
        pieces.append(pd.DataFrame({"station": station, "layer": 1, "_time": pd.date_range("2025-01-01", periods=60, freq="10min", tz="UTC"), "temp": np.arange(60, dtype=float) + offset}))
    return pd.concat(pieces, ignore_index=True)


def test_evidential_features_are_group_reset_and_future_invariant(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.shared, "_set_transport_context", lambda *_: None)
    frame = _frame()
    boundary = int(pd.Timestamp(frame.loc[49, "_time"]).value)
    representation = {"lag_rows": [1, 6, 36]}
    first = MODULE.causal_evidential_features(frame, boundary, representation)
    changed = frame.copy()
    future = MODULE.core._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] = 1_000_000.0
    second = MODULE.causal_evidential_features(changed, boundary, representation)
    np.testing.assert_array_equal(first[~future], second[~future])
    assert first[0, -1] == 1.0 and first[60, -1] == 1.0


def test_evidential_loss_is_finite_nonnegative_and_differentiable() -> None:
    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    labels = torch.tensor([1, 0])
    loss = MODULE.evidential_loss(logits, labels, 0.01)
    assert loss.shape == (2,)
    assert torch.isfinite(loss).all() and torch.all(loss >= 0)
    loss.mean().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_amended_transport_rejects_same_station_cell_two_halves(monkeypatch) -> None:
    candidate = {"count": 30, "precision_lcb": 0.9, "quantile": 0.999, "transport_stability": {"passed": True, "supported_environments": [{"station": "S", "layer": "5", "half": 0}, {"station": "S", "layer": "5", "half": 1}]}}
    monkeypatch.setattr(MODULE.shared.shared, "_select_transport", lambda *_: {"candidates": [candidate], "chosen": candidate})
    selection = {"minimum_additions": 25, "wilson90_lcb_minimum": 0.55, "transport_stability": {"minimum_distinct_station_layer_identities": 2, "minimum_distinct_stations": 2}}
    assert MODULE._select_amended(np.zeros(3), np.zeros(3), selection)["chosen"] is None


def test_config_freezes_evidential_nine_fit_addonly_contract() -> None:
    config = json.loads(MODULE.CONFIG.read_text(encoding="utf-8"))
    assert config["objective"]["kind"] == "dirichlet_expected_mse_variance_plus_wrong_class_uniform_kl"
    assert config["objective"]["uncertainty_use"] == "diagnostic_only"
    assert config["model"]["fits"] == 9
    assert len(config["model"]["seeds"]) * len(config["parts"]) == 9
    assert config["selection"]["transport_stability"]["minimum_distinct_stations"] == 2
    assert config["anchor"]["operation"] == "bitwise_or"
    assert config["anchor"]["removals"] == 0
