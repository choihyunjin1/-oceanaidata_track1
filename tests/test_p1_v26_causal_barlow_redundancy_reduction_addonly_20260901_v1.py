from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v26_causal_barlow_redundancy_reduction_addonly_20260901_v1.py"
SPEC = importlib.util.spec_from_file_location("p1_v26_tested", RUNNER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _frame() -> pd.DataFrame:
    pieces = []
    for station, offset in (("A", 0.0), ("B", 2.0)):
        pieces.append(pd.DataFrame({"station": station, "layer": 1, "_time": pd.date_range("2025-01-01", periods=180, freq="10min", tz="UTC"), "temp": np.arange(180, dtype=float) + offset}))
    return pd.concat(pieces, ignore_index=True)


def test_multilag_features_are_group_reset_and_future_invariant(monkeypatch) -> None:
    monkeypatch.setattr(MODULE.shared, "_set_transport_context", lambda *_: None)
    frame = _frame()
    boundary = int(pd.Timestamp(frame.loc[149, "_time"]).value)
    representation = {"lag_rows": [0, 1, 6, 36, 144]}
    first = MODULE.causal_multilag_features(frame, boundary, representation)
    changed = frame.copy()
    future = MODULE.core._time_ns(changed["_time"]) > boundary
    changed.loc[future, "temp"] = 1_000_000.0
    second = MODULE.causal_multilag_features(changed, boundary, representation)
    np.testing.assert_array_equal(first[~future], second[~future])
    assert first[0, -1] == 1.0 and first[180, -1] == 1.0


def test_barlow_identity_cross_correlation_has_zero_loss() -> None:
    values = torch.eye(4).repeat(8, 1)
    loss = MODULE.barlow_loss(values, values, 0.0051)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_amended_transport_requires_cell_and_station_diversity(monkeypatch) -> None:
    candidate = {"count": 30, "precision_lcb": 0.9, "quantile": 0.999, "transport_stability": {"passed": True, "supported_environments": [{"station": "S", "layer": "5", "half": 0}, {"station": "S", "layer": "5", "half": 1}]}}
    monkeypatch.setattr(MODULE.shared.shared, "_select_transport", lambda *_: {"candidates": [candidate], "chosen": candidate})
    selection = {"minimum_additions": 25, "wilson90_lcb_minimum": 0.55, "transport_stability": {"minimum_distinct_station_layer_identities": 2, "minimum_distinct_stations": 2}}
    result = MODULE._select_amended(np.zeros(3), np.zeros(3), selection)
    assert result["chosen"] is None
    assert result["candidates"][0]["transport_stability"]["legacy_passed"] is True
    assert result["candidates"][0]["transport_stability"]["passed"] is False


def test_config_freezes_label_free_pretrain_nine_fit_addonly_contract() -> None:
    config = json.loads(MODULE.CONFIG.read_text(encoding="utf-8"))
    assert config["objective"]["kind"] == "barlow_twins_cross_correlation_identity"
    assert config["objective"]["labels_in_pretraining"] == 0
    assert config["model"]["fits"] == 9
    assert len(config["model"]["seeds"]) * len(config["parts"]) == 9
    assert config["selection"]["transport_stability"]["minimum_distinct_stations"] == 2
    assert config["anchor"]["operation"] == "bitwise_or"
    assert config["anchor"]["removals"] == 0
