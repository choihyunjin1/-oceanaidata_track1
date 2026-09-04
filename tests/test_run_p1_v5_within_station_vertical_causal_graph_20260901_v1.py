from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_p1_v5_within_station_vertical_causal_graph_20260901_v1.py"
CONFIG = ROOT / "configs/experiments/p1_v5_within_station_vertical_causal_graph_20260901_v1.json"


def _module():
    spec = importlib.util.spec_from_file_location("p1_v5_vertical_graph_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    value = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = value
    spec.loader.exec_module(value)
    return value


def test_semantic_audit_is_novel_and_vertical_only() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["semantic_audit"]["decision"] == "NOVEL_PROCEED_ONCE"
    assert config["semantic_audit"]["exact_duplicate"] is False
    assert config["semantic_audit"]["semantic_duplicate"] is False
    assert config["architecture"]["horizontal_edges"] == 0
    assert "adjacent" in config["architecture"]["vertical_edges"]


def test_architecture_and_execution_budget_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    architecture = config["architecture"]
    assert architecture["seeds"] == [20260901, 20260917, 20260943]
    assert architecture["outer_folds"] == ["2025_q2", "2025_q3", "2025_q4"]
    assert architecture["maximum_fits"] == 9
    assert architecture["epochs"] == 4
    assert architecture["hidden_width"] == 16
    assert architecture["temporal_kernel"] == 9
    assert architecture["weight_decay"] == 0.0001
    assert architecture["sweeps"] == 0
    assert config["selection"]["outer_tuning"] == 0
    assert config["anchor"] == {"operation": "bitwise_or", "removals": 0}


def test_temporal_encoder_is_strictly_future_invariant() -> None:
    module = _module()
    torch.manual_seed(7)
    model = module.CausalVerticalGraph(input_width=6, hidden_width=5, kernel=3).eval()
    adjacency = torch.tensor(
        [[0.0, 1.0, 0.0], [0.5, 0.0, 0.5], [0.0, 1.0, 0.0]], dtype=torch.float32
    )
    values = torch.randn(12, 3, 6)
    changed = values.clone()
    changed[7:] = torch.randn_like(changed[7:]) * 1000
    with torch.no_grad():
        original = model(values, adjacency)
        counterfactual = model(changed, adjacency)
    np.testing.assert_allclose(original[:7].numpy(), counterfactual[:7].numpy(), atol=0, rtol=0)


def test_addition_cap_never_removes_anchor() -> None:
    module = _module()
    scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5])
    incumbent = np.asarray([1, 0, 0, 0, 0], dtype=np.int8)
    additions = module._capped_additions(scores, incumbent, threshold=0.5, share=0.4)
    candidate = np.bitwise_or(incumbent, additions.astype(np.int8))
    assert additions.sum() == 2
    assert additions[0] == 0
    assert np.all(candidate[incumbent == 1] == 1)
