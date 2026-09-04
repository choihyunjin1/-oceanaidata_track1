from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v8_missingness_decay_gru_latent_state_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v8_missingness_decay_gru_latent_state_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v8_missing_gru_runner", RUNNER)
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
            "psal": np.where(
                np.arange(rows) % 7 == 0,
                np.nan,
                np.linspace(30.0, 31.0, rows),
            ),
            "depth": np.where(np.arange(rows) % 11 == 0, np.nan, 7.8),
        }
    )


def test_time_contract_is_nanoseconds_and_cutoffs_are_distinct() -> None:
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
    assert all(10**18 < cutoff < 2 * 10**18 for cutoff in cutoffs)


def test_missingness_decay_is_groupwise_finite_and_future_invariant() -> None:
    module = _module()
    original = _frame()
    changed = original.copy()
    changed.loc[25:, "temp"] = 1000.0
    changed.loc[25:, "psal"] = -1000.0
    means = np.asarray([11.0, 30.5, 7.8], dtype=np.float32)
    scales = np.asarray([1.0, 0.5, 1.0], dtype=np.float32)
    first = module.missingness_decay_features(original, means, scales, 24.0)
    second = module.missingness_decay_features(changed, means, scales, 24.0)
    assert first.shape == (40, 10)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first[:25], second[:25], atol=0, rtol=0)


def test_elapsed_channel_resets_only_after_observation() -> None:
    module = _module()
    frame = _frame(8)
    frame["psal"] = [30.0, np.nan, np.nan, 31.0, np.nan, np.nan, np.nan, 32.0]
    features = module.missingness_decay_features(
        frame,
        np.asarray([11.0, 30.0, 7.8], dtype=np.float32),
        np.ones(3, dtype=np.float32),
        24.0,
    )
    assert features[0, 7] == 0.0
    assert 0.0 < features[1, 7] < features[2, 7]
    assert features[3, 7] == 0.0
    assert features[4, 7] > 0.0


def test_missingness_features_reset_at_station_layer_boundary() -> None:
    module = _module()
    first = _frame(8)
    first.loc[:, "station"] = "A"
    first.loc[:, "psal"] = np.nan
    second = _frame(8)
    second.loc[:, "station"] = "B"
    second.loc[:, "psal"] = np.nan
    second.index = np.arange(8, 16)
    combined = pd.concat([first, second])
    means = np.asarray([11.0, 30.0, 7.8], dtype=np.float32)
    scales = np.ones(3, dtype=np.float32)
    together = module.missingness_decay_features(combined, means, scales, 24.0)
    second.index = np.arange(8)
    alone = module.missingness_decay_features(second, means, scales, 24.0)
    np.testing.assert_allclose(together[8:], alone, atol=0, rtol=0)


def test_recurrent_head_is_strictly_causal() -> None:
    module = _module()
    torch.manual_seed(7)
    model = module.CausalMissingnessGRU(10, 4).eval()
    values = torch.randn(30, 10)
    changed = values.clone()
    changed[20:] = 1000.0
    with torch.no_grad():
        first, _ = model(values)
        second, _ = model(changed)
    torch.testing.assert_close(first[:20], second[:20], atol=0, rtol=0)


def test_station_layer_recurrent_state_is_reset() -> None:
    module = _module()
    torch.manual_seed(11)
    model = module.CausalMissingnessGRU(10, 4).eval()
    first = torch.randn(12, 10)
    second = torch.randn(9, 10)
    with torch.no_grad():
        reset_a, _ = model(second, None)
        _, carried_state = model(first, None)
        carried, _ = model(second, carried_state)
        reset_b, _ = model(second, None)
    torch.testing.assert_close(reset_a, reset_b, atol=0, rtol=0)
    assert not torch.equal(reset_a, carried)


def test_post_cutoff_perturbation_preserves_prefix_threshold() -> None:
    module = _module()
    torch.manual_seed(13)
    original = _frame()
    changed = original.copy()
    changed.loc[25:, ["temp", "psal", "depth"]] = 1000.0
    means = np.asarray([11.0, 30.5, 7.8], dtype=np.float32)
    scales = np.asarray([1.0, 0.5, 1.0], dtype=np.float32)
    first = module.missingness_decay_features(original, means, scales, 24.0)
    second = module.missingness_decay_features(changed, means, scales, 24.0)
    model = module.CausalMissingnessGRU(10, 4).eval()
    with torch.no_grad():
        first_logits, _ = model(torch.from_numpy(first))
        second_logits, _ = model(torch.from_numpy(second))
    first_scores = torch.sigmoid(first_logits[:25]).numpy()
    second_scores = torch.sigmoid(second_logits[:25]).numpy()
    np.testing.assert_allclose(first_scores, second_scores, atol=0, rtol=0)
    labels = (np.arange(25) % 3 == 0).astype(np.int8)
    selection = {
        "threshold_quantiles": [0.8],
        "minimum_additions_for_precision_gate": 1,
        "maximum_addition_share": 1.0,
        "precision_lcb_minimum": 0.0,
        "wilson_z": 1.6448536269514722,
    }
    assert module._select_threshold(first_scores, labels, selection) == (
        module._select_threshold(second_scores, labels, selection)
    )


def test_add_only_cap_preserves_every_anchor_positive() -> None:
    module = _module()
    scores = np.asarray([0.99, 0.98, 0.97, 0.96, 0.95])
    incumbent = np.asarray([1, 0, 0, 0, 0], dtype=np.int8)
    additions = module._capped_additions(
        scores,
        incumbent,
        {"threshold": 0.95},
        0.4,
    )
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    assert additions.sum() == 2
    assert not additions[0]
    assert np.all(candidate[incumbent == 1] == 1)


def test_nine_fit_budget_and_fixed_gate() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["architecture"]["maximum_fits"] == 9
    assert len(config["architecture"]["seeds"]) * len(config["parts"]) == 9
    assert config["architecture"]["sweeps"] == 0
    assert config["selection"]["precision_lcb_minimum"] == 0.55
    assert config["selection"]["outer_tuning"] == 0
    assert config["anchor"] == {"operation": "bitwise_or", "removals": 0}
