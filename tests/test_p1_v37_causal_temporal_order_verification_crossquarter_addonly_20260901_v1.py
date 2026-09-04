from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v37_causal_temporal_order_verification_crossquarter_addonly_20260901_v1.py"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v37_tested", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


mod = _module()


def _frame(groups: int = 2, rows: int = 80) -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "station": np.repeat([f"S-{index}" for index in range(groups)], rows),
            "layer": np.repeat([f"L{index}" for index in range(groups)], rows),
            "_time": np.tile(times, groups),
            "temp": np.tile(np.arange(rows, dtype=np.float64), groups),
        }
    )


def test_exact_cadence_lag_sequence_and_group_reset() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    frame = _frame()
    boundary = int(frame.loc[39, "_time"].value)
    features = mod.temporal_order_features(frame, boundary, config["representation"])
    assert features.shape == (160, 9)
    assert np.all(features[:36, -1] == 0)
    assert np.all(features[80:116, -1] == 0)
    assert features[36, -1] == features[116, -1] == 1
    assert np.allclose(features[36, :8], features[116, :8])


def test_ns_cutoff_and_future_invariance() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    frame = _frame(groups=1)
    boundary = int(frame.loc[39, "_time"].value)
    first = mod.temporal_order_features(frame, boundary, config["representation"])
    later = frame.copy()
    mask = mod.base._time_ns(later["_time"]) > boundary
    later.loc[mask, "temp"] += 1000.0
    second = mod.temporal_order_features(later, boundary, config["representation"])
    assert mod.base._time_ns(frame["_time"]).dtype == np.dtype("int64")
    assert int(frame.loc[38, "_time"].value) < boundary < int(frame.loc[40, "_time"].value)
    assert np.array_equal(first[~mask], second[~mask])


def test_reverse_pretext_is_fixed_distinct_and_involutive() -> None:
    row = torch.tensor([[0.0, 1.0, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 1.0]])
    reverse = row.clone()
    reverse[:, :-1] = torch.flip(row[:, :-1], dims=[1])
    assert not torch.equal(row, reverse)
    assert torch.equal(torch.flip(reverse[:, :-1], dims=[1]), row[:, :-1])
    assert reverse[0, -1] == row[0, -1]


def test_add_only_crossquarter_and_auditability_contract() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    assert config["anchor"] == {"operation": "bitwise_or", "removals": 0}
    assert config["model"]["maximum_fits"] == 9
    assert config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q2_q3_threshold_selection"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"]
    assert config["auditability_amendment"]["preserve_all_pre_q2_threshold_q2_label_blind_actions"]


def test_semantic_pretext_has_no_target_or_outer_input() -> None:
    config = json.loads(mod.CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["pretext_semantic_labels"] == 0
    assert config["model"]["pretext_outer_rows"] == 0
    assert config["semantic_audit"]["decision"] == "NOVEL_P1_OBJECTIVE_PROCEED_ONCE"
    assert config["operations"]["official"] == config["operations"]["csv"] == config["operations"]["uploads"] == 0
