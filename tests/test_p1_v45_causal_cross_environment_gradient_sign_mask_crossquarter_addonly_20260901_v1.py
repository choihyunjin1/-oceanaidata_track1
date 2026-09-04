from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v45_causal_cross_environment_gradient_sign_mask_crossquarter_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v45_causal_cross_environment_gradient_sign_mask_crossquarter_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("test_p1_v45_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(rows: int = 96) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    return pd.DataFrame({"station": np.repeat(["G-ORS", "I-ORS"], rows), "layer": np.repeat([1, 2], rows), "_time": np.tile(time, 2), "temp": np.tile(np.sin(np.arange(rows) / 8.0), 2)})


def test_preregistered_support_budget_and_transport_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    gate = config["representation_support_gate"]
    assert gate["minimum_rows_per_supported_environment"] == 4096
    assert gate["minimum_distinct_layers_per_supported_environment"] == 2
    assert gate["minimum_supported_environments"] == 5
    assert gate["v44_minimum_100_not_reused_or_relaxed"] is True
    assert config["model"]["fits"] == 3
    assert config["model"]["maximum_fits"] == 9
    assert config["model"]["sweep"] == 0
    assert config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"] is True
    assert config["anchor"]["removals"] == 0


def test_and_mask_unanimity_zeroes_conflicting_coordinate() -> None:
    module = _module()
    gradients = [[torch.tensor([2.0, 2.0, -1.0]), torch.tensor([1.0])], [torch.tensor([1.0, -3.0, -2.0]), torch.tensor([2.0])], [torch.tensor([3.0, 4.0, -4.0]), torch.tensor([3.0])]]
    masked, share = module.and_mask_gradients(gradients, 1.0)
    assert torch.allclose(masked[0], torch.tensor([2.0, 0.0, -7.0 / 3.0]))
    assert torch.allclose(masked[1], torch.tensor([2.0]))
    assert share == 0.75


def test_and_mask_is_environment_order_invariant() -> None:
    module = _module()
    gradients = [[torch.tensor([1.0, -2.0])], [torch.tensor([3.0, 4.0])], [torch.tensor([2.0, -1.0])]]
    first, first_share = module.and_mask_gradients(gradients, 1.0)
    second, second_share = module.and_mask_gradients([gradients[2], gradients[0], gradients[1]], 1.0)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    assert first_share == second_share


def test_metadata_decoding_survives_standardization() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = _frame()
    boundary = int(pd.DatetimeIndex([frame["_time"].iloc[-1]]).as_unit("ns").asi8[0])
    features = module.CAUSAL_ENVIRONMENT_FEATURES(frame, boundary, config["representation"])
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    environments, layers = module.decode_source_metadata(scaled)
    assert features.shape == (192, 23)
    assert len(np.unique(environments)) == 2
    assert np.all(layers[:96] == 0)
    assert np.all(layers[96:] == 1)


def test_group_reset_future_invariance_and_environment_blind_inference() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    frame = _frame()
    time_ns = pd.DatetimeIndex(frame["_time"]).as_unit("ns").asi8
    boundary = int(time_ns[47])
    first = module.CAUSAL_ENVIRONMENT_FEATURES(frame, boundary, config["representation"])
    changed = frame.copy()
    future = time_ns > boundary
    changed.loc[future, "temp"] += 1000.0
    second = module.CAUSAL_ENVIRONMENT_FEATURES(changed, boundary, config["representation"])
    assert int(time_ns[46]) < boundary < int(time_ns[48])
    assert np.array_equal(first[~future], second[~future])
    assert np.array_equal(first[:96, :8], first[96:, :8])
    classifier = module.ANDMaskClassifier(23, config["model"], 17)
    altered = first.copy()
    altered[:, 8:] = altered[::-1, 8:]
    assert np.array_equal(classifier.predict_score(first), classifier.predict_score(altered))


def test_all_synthetic_guards_pass() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(module._synthetic_guards(config["representation"]).values())
