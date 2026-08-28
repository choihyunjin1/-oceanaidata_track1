from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from p1_qc.direct_interval_set_torch import (
    DirectIntervalConfig,
    DirectIntervalSetPredictor,
    interval_set_loss,
)

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / "scripts" / "run_p1_direct_interval_set_torch_nested_20260828_v1.py"
    spec = importlib.util.spec_from_file_location("test_p1_direct_interval_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_interval_model_has_finite_trainable_path() -> None:
    config = DirectIntervalConfig(input_features=4, window_rows=64, patch_rows=8, d_model=16, heads=4, encoder_layers=1, queries=2)
    model = DirectIntervalSetPredictor(config)
    features = torch.randn(3, 64, 4)
    targets = [np.asarray([[0.2, 0.6]], dtype=np.float32), np.empty((0, 2), dtype=np.float32), np.asarray([[0.5, 0.8]], dtype=np.float32)]
    logits, intervals, patch_logits = model(features)
    loss = interval_set_loss(logits, intervals, patch_logits, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_anchor_union_and_exact_noop_are_preserved() -> None:
    runner = _runner()
    blind = runner.BlindPrediction(
        "q3",
        None,
        np.asarray([1, 0, 0, 1], dtype=np.int8),
        np.asarray([0, 1, 0, 0], dtype=np.int8),
        np.zeros(4, dtype=np.float32),
        {},
    )
    truth = np.asarray([1, 1, 0, 0], dtype=np.int8)
    no_op = runner._score(blind, truth, use_model=False)
    model = runner._score(blind, truth, use_model=True)
    assert no_op["added_rows"] == 0
    assert no_op["anchor_positive_removed_rows"] == 0
    assert model["anchor_positive_removed_rows"] == 0


def test_config_forbids_result_driven_retry_and_official_actions() -> None:
    config = _runner()._load_config()
    assert config["training"]["result_based_retry"] is False
    assert config["anchor"]["exact_zero_add_no_op_arm"] is True
    assert all(config["prohibitions"].values())
