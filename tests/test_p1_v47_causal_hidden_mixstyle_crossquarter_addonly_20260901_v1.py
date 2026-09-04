from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v47_causal_hidden_mixstyle_crossquarter_addonly_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("test_p1_v47_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frame(rows: int = 96) -> pd.DataFrame:
    time = pd.date_range("2024-01-01", periods=rows, freq="10min", tz="UTC")
    return pd.DataFrame(
        {
            "station": np.repeat(["G-ORS", "I-ORS"], rows),
            "layer": np.repeat([1, 2], rows),
            "_time": np.tile(time, 2),
            "temp": np.tile(np.sin(np.arange(rows) / 8.0), 2),
        }
    )


def test_preregistered_mixstyle_budget_and_transport_contract() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["model"]["mix_probability"] == 0.5
    assert config["model"]["beta_alpha"] == 0.1
    assert config["model"]["labels_mixed"] is False
    assert config["model"]["fits"] == 3
    assert config["model"]["maximum_fits"] == 9
    assert config["model"]["sweep"] == 0
    assert config["selection"]["threshold_quantiles"] == [0.995, 0.9975, 0.999]
    assert config["selection"]["q2_q3_refits"] == 0
    assert config["selection"]["q4_open_only_after_q2_q3_pass"] is True
    assert config["anchor"]["removals"] == 0


def test_mixstyle_endpoint_statistics() -> None:
    module = _module()
    content = torch.tensor([[1.0, 2.0, 4.0, 8.0], [-3.0, -1.0, 2.0, 5.0]])
    donor = torch.tensor([[10.0, 12.0, 14.0, 16.0], [20.0, 21.0, 24.0, 29.0]])
    own = module.mixstyle_hidden(content, donor, torch.ones(2, 1), 1e-6)
    borrowed = module.mixstyle_hidden(content, donor, torch.zeros(2, 1), 1e-6)
    assert torch.allclose(own, content, atol=2e-6, rtol=2e-6)
    assert torch.allclose(borrowed.mean(dim=1), donor.mean(dim=1), atol=2e-6)
    assert torch.allclose(
        torch.sqrt(borrowed.var(dim=1, unbiased=False) + 1e-6),
        torch.sqrt(donor.var(dim=1, unbiased=False) + 1e-6),
        atol=5e-6,
    )


def test_partner_selection_is_always_cross_environment() -> None:
    module = _module()
    environments = np.repeat(np.arange(5, dtype=np.int64), 20)
    pools = {value: np.flatnonzero(environments == value) for value in range(5)}
    partners = module.choose_partner_rows(environments, pools, np.random.default_rng(17))
    assert partners.shape == environments.shape
    assert np.all(environments[partners] != environments)


def test_standardized_metadata_decode_and_environment_blind_inference() -> None:
    module = _module()
    rows = 24
    metadata = np.zeros((rows, module.ENVIRONMENT_BITS), dtype=np.float32)
    station = np.arange(rows) % len(module.STATIONS)
    layer = np.arange(rows) % len(module.LAYERS)
    quarter = np.arange(rows) % len(module.QUARTERS)
    metadata[np.arange(rows), station] = 1.0
    metadata[np.arange(rows), len(module.STATIONS) + layer] = 1.0
    metadata[np.arange(rows), len(module.STATIONS) + len(module.LAYERS) + quarter] = 1.0
    values = np.concatenate([np.zeros((rows, module.SCIENTIFIC_FEATURES)), metadata], axis=1)
    scaled = module.base.StandardScaler().fit_transform(values).astype(np.float32)
    decoded_environment, decoded_layer = module.decode_source_metadata(scaled)
    assert np.array_equal(decoded_environment, station * 10 + quarter)
    assert np.array_equal(decoded_layer, layer)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    classifier = module.MixStyleClassifier(values.shape[1], config["model"], 17)
    first = classifier.predict_score(scaled)
    changed = scaled.copy()
    changed[:, module.SCIENTIFIC_FEATURES:] += np.arange(module.ENVIRONMENT_BITS)
    assert np.array_equal(first, classifier.predict_score(changed))


def test_ns_group_reset_and_future_invariance() -> None:
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
    assert np.array_equal(first[:96, : module.SCIENTIFIC_FEATURES], first[96:, : module.SCIENTIFIC_FEATURES])


def test_all_synthetic_guards_pass() -> None:
    module = _module()
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert all(module._synthetic_guards(config["representation"]).values())
