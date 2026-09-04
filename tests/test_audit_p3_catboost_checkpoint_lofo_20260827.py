from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts/audit_p3_catboost_checkpoint_lofo_20260827.py"
    )
    spec = importlib.util.spec_from_file_location("p3_lofo_audit_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weighted_lofo_selection_excludes_held_fold_and_breaks_ties_early() -> None:
    module = _module()
    grid = [
        {
            "single_tree_fraction": 0.5,
            "multi_tree_fraction": 0.5,
            "fold_rmse_m": {"a": 1.0, "b": 1.0, "held": 99.0},
        },
        {
            "single_tree_fraction": 1.0,
            "multi_tree_fraction": 1.0,
            "fold_rmse_m": {"a": 1.0, "b": 1.0, "held": 0.0},
        },
    ]
    selected = module._select_pair(
        grid,
        training_folds=("a", "b"),
        fold_counts={"a": 1, "b": 100, "held": 2},
    )
    assert selected["selection_rmse_m"] == 1.0
    assert selected["grid_row"]["single_tree_fraction"] == 0.5
    assert selected["grid_row"]["multi_tree_fraction"] == 0.5


def test_pooled_rmse_uses_sse_weights() -> None:
    module = _module()
    observed = module._pooled_rmse([(1, 0.0), (3, 2.0)])
    assert observed == (3.0**0.5)
